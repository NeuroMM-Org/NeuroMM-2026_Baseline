"""Binary classification heads."""

from __future__ import annotations

import torch.nn as nn


class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dim is None:
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, 1),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, features):
        return self.network(features)
