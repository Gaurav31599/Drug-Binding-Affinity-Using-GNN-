"""Featurization for the affinity models.

Two parallel representations, one per model family:

* Baseline (classical regressors): Morgan (ECFP-like) fingerprint for the drug,
  concatenated with the protein's amino-acid composition (AAC). A fixed-length
  numeric vector — exactly what scikit-learn estimators expect.

* GNN (GraphDTA-style): the drug as a PyTorch Geometric molecular graph (atom
  feature matrix + bond edge index) and the protein as an integer-encoded
  sequence consumed by a 1-D CNN inside the model. The drug graph carries the
  protein encoding on its ``target`` attribute so a single ``Data`` object holds
  the full (drug, target) pair.
"""

from __future__ import annotations

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
except ImportError as exc:  # pragma: no cover
    raise ImportError("rdkit is required for featurization.") from exc


# --------------------------------------------------------------------------- #
# Protein sequence encoding (shared)
# --------------------------------------------------------------------------- #
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved = pad/unk
MAX_SEQ_LEN = 1000  # GraphDTA convention: truncate/pad to a fixed window


def amino_acid_composition(sequence: str) -> np.ndarray:
    """20-dim normalised amino-acid frequency vector (baseline protein feature)."""
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    for ch in sequence:
        idx = _AA_TO_IDX.get(ch.upper())
        if idx is not None:
            counts[idx - 1] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


def encode_sequence(sequence: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Integer-encode a protein sequence to fixed length (GNN protein feature)."""
    enc = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(sequence[:max_len]):
        enc[i] = _AA_TO_IDX.get(ch.upper(), 0)
    return enc


# --------------------------------------------------------------------------- #
# Baseline: Morgan fingerprint + AAC
# --------------------------------------------------------------------------- #
def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Binary Morgan fingerprint as a float32 vector. Zeros if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    ConvertToNumpyArray(fp, arr)
    return arr


def featurize_baseline(
    smiles_list: list[str],
    sequences: list[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """Stack [Morgan(drug) | AAC(target)] into an (N, n_bits + 20) matrix."""
    rows = [
        np.concatenate(
            [morgan_fingerprint(smi, radius, n_bits), amino_acid_composition(seq)]
        )
        for smi, seq in zip(smiles_list, sequences)
    ]
    return np.vstack(rows).astype(np.float32)


# --------------------------------------------------------------------------- #
# GNN: molecular graph
# --------------------------------------------------------------------------- #
def _atom_features(atom) -> list[float]:
    """Per-atom feature vector. Length must match ATOM_FEATURE_DIM below."""

    def one_hot(value, choices):
        vec = [0.0] * (len(choices) + 1)
        vec[choices.index(value) if value in choices else -1] = 1.0
        return vec

    symbols = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
    feats: list[float] = []
    feats += one_hot(atom.GetSymbol(), symbols)              # 10
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])   # 7
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 6
    feats += one_hot(int(atom.GetHybridization()), [1, 2, 3, 4, 5])  # 6
    feats += [float(atom.GetIsAromatic())]                   # 1
    feats += [float(atom.GetFormalCharge())]                 # 1
    return feats


# Computed once so consumers (and tests) can assert shapes without a molecule.
ATOM_FEATURE_DIM = len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def smiles_to_graph(smiles: str, affinity: float, sequence: str):
    """Convert one (drug, target, affinity) row to a PyG ``Data`` object.

    Returns ``None`` if the SMILES cannot be parsed, so callers can skip it.
    """
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor(
        [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )

    edges: list[list[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]  # undirected -> both directions
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:  # single-atom molecule: no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([affinity], dtype=torch.float),
        target=torch.tensor(encode_sequence(sequence), dtype=torch.long).unsqueeze(0),
    )


def build_graph_dataset(smiles_list, affinities, sequences) -> list:
    """Vectorised ``smiles_to_graph`` over a dataset, dropping unparseable rows."""
    graphs = []
    for smi, aff, seq in zip(smiles_list, affinities, sequences):
        g = smiles_to_graph(smi, float(aff), seq)
        if g is not None:
            graphs.append(g)
    return graphs
