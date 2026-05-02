"""Migrated legacy EEG model module: cbramod.

CBraMod uses a criss-cross Transformer (spatial + temporal attention) on
patch-level EEG embeddings.  Requires pretrained weights for best results.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod_src.cbramod import CBraMod

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """CBraMod adapter for the baseline pipeline.

    Input:  [B, 26, 2000]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 19,
        input_samples: int = 2000,
        patch_size: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        dropout: float = 0.3,
        pretrained_path: str = "",
        # unused — kept for registry compatibility
        pretrained: bool = False,
        temporal_pool: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.seq_len = input_samples // patch_size  # 2000 // 200 = 10

        self.backbone = CBraMod(
            in_dim=d_model, out_dim=d_model, d_model=d_model,
            dim_feedforward=dim_feedforward, seq_len=self.seq_len,
            n_layer=n_layer, nhead=nhead,
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

        # Disable the backbone's projection head
        self.backbone.proj_out = nn.Identity()

        # Classifier: [B, C, S, D] -> [B, C*S*D] -> [B, num_classes]
        flatten_dim = input_channels * self.seq_len * d_model
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Dropout(dropout),
            nn.Linear(flatten_dim, num_classes),
        )

    def _load_pretrained_weights(self, path: str) -> None:
        logger.info("loading CBraMod pretrained weights from %s", path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        filtered = OrderedDict(
            (k, v) for k, v in state_dict.items() if 'proj_out' not in k
        )
        msg = self.backbone.load_state_dict(filtered, strict=False)
        if msg.missing_keys:
            logger.info("CBraMod missing keys (expected): %s", msg.missing_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 26, 2000] -> truncate to input_channels -> [B, C, 10, 200]
        if x.dim() == 3:
            x = x[:, :self.input_channels, :]
            x = x.view(x.size(0), self.input_channels, self.seq_len, self.patch_size)
        feats = self.backbone(x)
        return self.classifier(feats)
