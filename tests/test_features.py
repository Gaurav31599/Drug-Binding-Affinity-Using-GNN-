"""Featurization output shape / dtype checks."""

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")

from affinity_gnn import features  # noqa: E402

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
SEQS = ["MKTAYIAKQR", "ACDEFGHIKLMNPQRSTVWY"]


def test_morgan_fingerprint_shape_dtype():
    fp = features.morgan_fingerprint(SMILES[0], radius=2, n_bits=1024)
    assert fp.shape == (1024,)
    assert fp.dtype == np.float32
    assert set(np.unique(fp)).issubset({0.0, 1.0})  # binary


def test_morgan_invalid_smiles_returns_zeros():
    fp = features.morgan_fingerprint("not_a_molecule", n_bits=512)
    assert fp.shape == (512,)
    assert fp.sum() == 0.0


def test_aac_is_normalised_20dim():
    aac = features.amino_acid_composition(SEQS[1])
    assert aac.shape == (20,)
    assert aac.sum() == pytest.approx(1.0)


def test_encode_sequence_fixed_length():
    enc = features.encode_sequence("ACDACD", max_len=10)
    assert enc.shape == (10,)
    assert enc.dtype == np.int64
    assert enc[6:].sum() == 0  # padded with zeros


def test_featurize_baseline_concatenation_shape():
    X = features.featurize_baseline(SMILES, SEQS, radius=2, n_bits=1024)
    assert X.shape == (2, 1024 + 20)
    assert X.dtype == np.float32


def test_smiles_to_graph_structure():
    pytest.importorskip("torch_geometric")
    g = features.smiles_to_graph("CCO", affinity=5.0, sequence=SEQS[0])
    assert g is not None
    assert g.x.shape[0] == 3                     # 3 heavy atoms (C, C, O)
    assert g.x.shape[1] == features.ATOM_FEATURE_DIM
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] == 4            # 2 bonds x 2 directions
    assert g.y.item() == pytest.approx(5.0)
    assert g.target.shape == (1, features.MAX_SEQ_LEN)


def test_smiles_to_graph_invalid_returns_none():
    pytest.importorskip("torch_geometric")
    assert features.smiles_to_graph("xxx", 0.0, SEQS[0]) is None
