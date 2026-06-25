"""Scaffold-split correctness: the core methodological guarantee."""

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")

from affinity_gnn.splits import (  # noqa: E402
    assert_no_scaffold_overlap,
    murcko_scaffold,
    scaffold_split,
)

# Mix of distinct scaffolds plus deliberate near-duplicates that share a
# scaffold (e.g. substituted benzenes), so a *random* split could leak but a
# scaffold split must not.
SMILES = [
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1",   # benzene scaffold
    "c1ccncc1", "Cc1ccncc1",                                  # pyridine scaffold
    "C1CCCCC1", "CC1CCCCC1",                                  # cyclohexane scaffold
    "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1",                     # naphthalene scaffold
    "CC(=O)Oc1ccccc1C(=O)O",                                 # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                          # caffeine
]


def test_no_scaffold_overlap():
    train_idx, _, test_idx = scaffold_split(SMILES, 0.7, 0.0, 0.3, seed=1)
    # Should not raise.
    assert_no_scaffold_overlap(SMILES, train_idx, test_idx)


def test_partition_is_complete_and_disjoint():
    train_idx, valid_idx, test_idx = scaffold_split(SMILES, 0.6, 0.2, 0.2, seed=3)
    all_idx = np.concatenate([train_idx, valid_idx, test_idx])
    assert sorted(all_idx.tolist()) == list(range(len(SMILES)))
    assert len(set(all_idx.tolist())) == len(SMILES)  # disjoint


def test_scaffolds_actually_group():
    # Same scaffold for substituted benzenes.
    assert murcko_scaffold("Cc1ccccc1") == murcko_scaffold("Clc1ccccc1")
    # Different scaffold from pyridine.
    assert murcko_scaffold("c1ccccc1") != murcko_scaffold("c1ccncc1")


def test_unparseable_smiles_grouped_not_leaked():
    smiles = SMILES + ["not_a_molecule", "also::invalid"]
    train_idx, _, test_idx = scaffold_split(smiles, 0.7, 0.0, 0.3, seed=2)
    assert_no_scaffold_overlap(smiles, train_idx, test_idx)


def test_deterministic_given_seed():
    a = scaffold_split(SMILES, 0.7, 0.0, 0.3, seed=7)
    b = scaffold_split(SMILES, 0.7, 0.0, 0.3, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
