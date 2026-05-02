"""Project video features into a fixed embedding size."""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoFeatureProjector(nn.Module):
    def __init__(self, output_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, video_feature: torch.Tensor) -> torch.Tensor:
        return self.network(video_feature.float())
