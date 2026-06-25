"""GraphDTA-style GNN for drug-target affinity regression.

Architecture
------------
* Drug branch: a stack of GCN message-passing layers over the molecular graph,
  followed by global mean pooling to a fixed-length molecule embedding.
* Target branch: an embedding + 1-D CNN over the integer-encoded protein
  sequence, global-max-pooled to a fixed-length protein embedding.
* The two embeddings are concatenated and passed through an MLP head that
  regresses a single continuous affinity value.

This predicts a continuous score (regression), not a class — consistent with the
metrics module. Hyperparameters (``hidden_dim``, ``num_layers``, etc.) are
constructor arguments so ``train.py`` can log them to MLflow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .features import AMINO_ACIDS, ATOM_FEATURE_DIM, MAX_SEQ_LEN


class GraphDTA(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
        protein_embed_dim: int = 128,
        dropout: float = 0.2,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.num_layers = num_layers

        # --- drug graph branch ---
        self.convs = nn.ModuleList()
        in_dim = atom_feature_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.drug_fc = nn.Linear(hidden_dim, hidden_dim)

        # --- protein sequence branch (1-D CNN) ---
        # +1 vocab slot for the 0 pad/unknown index used by encode_sequence.
        self.prot_embed = nn.Embedding(len(AMINO_ACIDS) + 1, protein_embed_dim,
                                       padding_idx=0)
        self.prot_conv = nn.Sequential(
            nn.Conv1d(protein_embed_dim, hidden_dim, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.max_seq_len = max_seq_len

        # --- joint regression head ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        # drug branch
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        drug = F.relu(self.drug_fc(global_mean_pool(x, batch)))

        # protein branch: data.target is (batch_size, max_seq_len) long tensor
        prot = self.prot_embed(data.target)             # (B, L, E)
        prot = prot.permute(0, 2, 1)                    # (B, E, L)
        prot = self.prot_conv(prot).squeeze(-1)         # (B, hidden)

        joint = torch.cat([drug, prot], dim=1)
        return self.head(joint).squeeze(-1)             # (B,)


def train_one_epoch(model, loader, optimizer, device) -> float:
    """One training pass. Returns mean MSE loss over the epoch."""
    model.train()
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, y_pred) numpy arrays over a loader."""
    model.eval()
    trues, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        trues.append(batch.y.view(-1).cpu())
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()
