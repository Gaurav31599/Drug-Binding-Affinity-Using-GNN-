"""KIBA drug-target binding-affinity dataset loader.

KIBA is a standard DTA benchmark (used by DeepDTA, GraphDTA, etc.). We source it
from the DeepDTA GitHub mirror, which distributes it without registration:

    https://github.com/hkmztrk/DeepDTA  ->  data/kiba/

The relevant files are:
    ligands_can.txt   JSON dict {ligand_id: canonical_SMILES}
    proteins.txt      JSON dict {protein_id: amino_acid_sequence}
    Y                 pickled numpy array, shape (n_drugs, n_targets), with
                      NaN where a (drug, target) pair was not measured.

We flatten ``Y`` into a tidy long-format table of measured pairs only.

KIBA affinity note
------------------
KIBA scores are an integrated affinity metric (already on a unified scale); they
are *not* pKd. We keep the published KIBA score as the regression target and say
so explicitly, rather than mislabelling it pKd. (DAVIS, by contrast, ships Kd in
nM which is converted to pKd = -log10(Kd / 1e9).)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KIBA_FILES = {
    "ligands": "ligands_can.txt",
    "proteins": "proteins.txt",
    "affinity": "Y",
}

# Raw mirror (DeepDTA). Pinned to a commit-agnostic raw path; if the layout
# changes, point DATA_DIR at a local copy instead of downloading.
KIBA_RAW_BASE = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/"
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "kiba"


@dataclass
class DTADataset:
    """Tidy long-format drug-target affinity table.

    ``frame`` columns: drug_id, target_id, smiles, sequence, affinity.
    Convenience accessors return column arrays in row order.
    """

    frame: pd.DataFrame

    @property
    def smiles(self) -> list[str]:
        return self.frame["smiles"].tolist()

    @property
    def sequences(self) -> list[str]:
        return self.frame["sequence"].tolist()

    @property
    def affinity(self) -> np.ndarray:
        return self.frame["affinity"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)


def download_kiba(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the three KIBA files into ``data_dir`` if missing.

    Kept dependency-light (urllib only). Returns the directory. Network access
    is required the first time; subsequent calls are no-ops.
    """
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in KIBA_FILES.values():
        dest = data_dir / fname
        if dest.exists():
            continue
        url = KIBA_RAW_BASE + fname
        print(f"[kiba] downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return data_dir


def load_kiba(
    data_dir: Path = DEFAULT_DATA_DIR,
    download: bool = True,
    max_pairs: int | None = None,
    seed: int = 42,
) -> DTADataset:
    """Load KIBA as a tidy long table of measured (drug, target) pairs.

    Parameters
    ----------
    data_dir : where the raw KIBA files live (or will be downloaded to).
    download : fetch files if absent.
    max_pairs : optionally subsample to N pairs (used for fast smoke tests).
    seed : subsample RNG seed.
    """
    data_dir = Path(data_dir)
    if download:
        download_kiba(data_dir)

    with open(data_dir / KIBA_FILES["ligands"]) as fh:
        ligands: dict[str, str] = json.load(fh)
    with open(data_dir / KIBA_FILES["proteins"]) as fh:
        proteins: dict[str, str] = json.load(fh)
    with open(data_dir / KIBA_FILES["affinity"], "rb") as fh:
        Y = pickle.load(fh, encoding="latin1")
    Y = np.asarray(Y, dtype=np.float64)

    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())

    # Flatten the (n_drugs, n_targets) matrix into measured pairs (drop NaN).
    rows, cols = np.where(~np.isnan(Y))
    records = {
        "drug_id": [drug_ids[r] for r in rows],
        "target_id": [target_ids[c] for c in cols],
        "smiles": [ligands[drug_ids[r]] for r in rows],
        "sequence": [proteins[target_ids[c]] for c in cols],
        "affinity": Y[rows, cols].astype(np.float32),
    }
    frame = pd.DataFrame(records)

    if max_pairs is not None and len(frame) > max_pairs:
        frame = frame.sample(n=max_pairs, random_state=seed).reset_index(drop=True)

    return DTADataset(frame=frame.reset_index(drop=True))


def load_toy(n: int = 64, seed: int = 0) -> DTADataset:
    """A tiny self-contained dataset for smoke tests (no network).

    Uses a handful of real drug SMILES paired against short dummy protein
    sequences with synthetic affinities. Enough scaffold diversity to exercise
    the scaffold splitter and featurizers without downloading KIBA.
    """
    smiles_pool = [
        "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",    # caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",      # ibuprofen
        "CN1CCC[C@H]1c1cccnc1",            # nicotine
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",      # theobromine
        "OC(=O)c1ccccc1O",                # salicylic acid
        "Clc1ccccc1",                      # chlorobenzene
        "c1ccccc1",                        # benzene
    ]
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=120))
            for _ in range(4)]
    records = []
    for i in range(n):
        smi = smiles_pool[i % len(smiles_pool)]
        seq = seqs[i % len(seqs)]
        # synthetic but deterministic affinity
        records.append(
            {
                "drug_id": f"D{i % len(smiles_pool)}",
                "target_id": f"T{i % len(seqs)}",
                "smiles": smi,
                "sequence": seq,
                "affinity": np.float32(10.0 + rng.normal(0, 1)),
            }
        )
    return DTADataset(frame=pd.DataFrame(records))
