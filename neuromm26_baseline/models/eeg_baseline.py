"""Single-modal EEG baseline model."""

from __future__ import annotations

import torch.nn as nn

from .backbone import EEGResNetEncoder
from .heads.classification_head import BinaryClassificationHead


class EEGBaselineModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, temporal_pool: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = EEGResNetEncoder(
            base=backbone,
            pretrained=pretrained,
            temporal_pool=temporal_pool,
            dropout=dropout,
        )
        self.head = BinaryClassificationHead(self.encoder.feature_dim, dropout=dropout)

    def forward(self, batch):
        features = self.encoder(batch["eeg"])
        return self.head(features)
