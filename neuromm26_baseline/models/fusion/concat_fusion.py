"""Concatenation fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def forward(self, eeg_feature: torch.Tensor, video_feature: torch.Tensor) -> torch.Tensor:
        return torch.cat([eeg_feature, video_feature], dim=-1)
