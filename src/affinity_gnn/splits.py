"""Scaffold splitting for drug-target affinity datasets.

Why scaffold splitting (and not a random split)?
------------------------------------------------
A random train/test split leaks structurally similar molecules across the
boundary: a near-identical analogue of a test compound can sit in the training
set, so the model effectively memorises a scaffold rather than generalising to
new chemistry. This inflates apparent performance. Scaffold splitting groups
molecules by their Bemis-Murcko scaffold and assigns *whole scaffold groups* to
a single split, so the test set contains chemotypes the model never saw. This
is the accepted standard in molecular ML and we treat it as a methodological
strength, not an implementation detail.

The split here operates on the *drug* axis of a drug-target dataset: every
(drug, target) pair inherits the split of its drug's scaffold. That guarantees
no drug scaffold appears in more than one split.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - exercised only without rdkit
    raise ImportError(
        "rdkit is required for scaffold splitting. Install with "
        "`pip install rdkit`."
    ) from exc


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return the canonical Bemis-Murcko scaffold SMILES for a molecule.

    Molecules that fail to parse return the empty string, which groups them
    together into a single "unparseable" scaffold bucket rather than silently
    leaking them across splits.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""


def scaffold_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.0,
    frac_test: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices by Bemis-Murcko scaffold.

    Largest scaffold groups are assigned first (deterministic, the DeepChem
    convention) so that the most common chemotypes populate the training set.
    Group *order among equal sizes* is shuffled with ``seed`` for reproducibility.

    Returns three arrays of integer indices into ``smiles_list``:
    ``(train_idx, valid_idx, test_idx)``. When ``frac_valid == 0`` the valid
    array is empty.
    """
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6, (
        "fractions must sum to 1"
    )

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold_to_indices[murcko_scaffold(smi)].append(idx)

    # Sort groups by descending size; break ties deterministically via a seeded
    # permutation so the split is reproducible but not biased by dict order.
    rng = np.random.default_rng(seed)
    groups = list(scaffold_to_indices.values())
    sizes = np.array([len(g) for g in groups])
    tie_breaker = rng.random(len(groups))
    order = np.lexsort((tie_breaker, -sizes))  # primary: -size, secondary: random

    n_total = len(smiles_list)
    n_train_target = frac_train * n_total
    n_valid_target = frac_valid * n_total

    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []
    for gi in order:
        group = groups[gi]
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid_target:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(valid_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def assert_no_scaffold_overlap(
    smiles_list: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Raise ``AssertionError`` if any scaffold appears in both splits.

    This is the invariant the unit tests assert against and is cheap enough to
    run as a guard inside training pipelines.
    """
    train_scaffolds = {murcko_scaffold(smiles_list[i]) for i in train_idx}
    test_scaffolds = {murcko_scaffold(smiles_list[i]) for i in test_idx}
    overlap = train_scaffolds & test_scaffolds
    assert not overlap, f"scaffold overlap between train and test: {overlap}"
