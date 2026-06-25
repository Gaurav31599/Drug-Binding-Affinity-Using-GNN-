"""End-to-end smoke test: the full GNN training loop runs for one epoch on a
tiny data slice without error, and produces finite predictions.

Skipped automatically if torch_geometric / rdkit are not installed, so the rest
of the suite still runs in a lighter environment.
"""

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from affinity_gnn import features  # noqa: E402
from affinity_gnn.data.loader import load_toy  # noqa: E402
from affinity_gnn.gnn_model import GraphDTA, predict, train_one_epoch  # noqa: E402
from affinity_gnn.splits import assert_no_scaffold_overlap, scaffold_split  # noqa: E402


def test_full_gnn_loop_one_epoch():
    ds = load_toy(n=32, seed=0)
    train_idx, _, test_idx = scaffold_split(ds.smiles, 0.75, 0.0, 0.25, seed=0)
    assert_no_scaffold_overlap(ds.smiles, train_idx, test_idx)

    frame = ds.frame
    train_rows = frame.iloc[train_idx]
    test_rows = frame.iloc[test_idx]

    train_graphs = features.build_graph_dataset(
        train_rows["smiles"].tolist(),
        train_rows["affinity"].tolist(),
        train_rows["sequence"].tolist(),
    )
    test_graphs = features.build_graph_dataset(
        test_rows["smiles"].tolist(),
        test_rows["affinity"].tolist(),
        test_rows["sequence"].tolist(),
    )
    assert len(train_graphs) > 0 and len(test_graphs) > 0

    device = torch.device("cpu")
    # Small protein window keeps the smoke test fast.
    model = GraphDTA(hidden_dim=16, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8)

    loss = train_one_epoch(model, train_loader, optimizer, device)
    assert np.isfinite(loss)

    y_true, y_pred = predict(model, test_loader, device)
    assert y_pred.shape == y_true.shape
    assert np.all(np.isfinite(y_pred))
