"""ResNet encoder for EEG arrays."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class EEGResNetEncoder(nn.Module):
    def __init__(
        self,
        base: str = "resnet18",
        pretrained: bool = False,
        temporal_pool: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if base == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            feature_dim = 512
        elif base == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported EEG backbone: {base}")

        conv1 = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
            if conv1.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv1.bias)
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()

        self.feature_dim = feature_dim
        self.temporal_pool = (
            nn.AvgPool2d(kernel_size=(1, temporal_pool), stride=(1, temporal_pool))
            if temporal_pool and temporal_pool > 1
            else None
        )
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.dim() == 3:
            eeg = eeg.unsqueeze(1)
        if eeg.dim() != 4:
            raise ValueError(f"Expected EEG tensor with 3 or 4 dims, got {tuple(eeg.shape)}")
        eeg = eeg.float()
        if self.temporal_pool is not None:
            eeg = self.temporal_pool(eeg)
        features = self.backbone(eeg)
        return self.dropout(features)
