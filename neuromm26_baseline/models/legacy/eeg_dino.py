"""Migrated legacy EEG model module: eeg_dino.

EEG-DINO uses a DINO-pretrained ViT backbone with EEG-specific patch
embedding (time-domain CNN + spectral FFT + channel positional encoding).
Only uses the first 19 channels (standard 10-20 EEG montage).

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch Embedding (time-domain + spectral + channel)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, model_type: str = "small"):
        super().__init__()
        self.d_model = d_model
        self.num_channels = 19

        self.time_encoding = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 5), stride=(1, 1),
                      padding=(0, 2), groups=d_model),
        )

        # proj_in output: out_ch * 8 (CNN output width) == d_model
        if model_type == "large":
            # 128 * 8 = 1024
            self.proj_in = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
                nn.GroupNorm(16, 128), nn.GELU(),
                nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
                nn.GroupNorm(16, 256), nn.GELU(),
                nn.Conv2d(256, 128, kernel_size=(1, 3), padding=(0, 1)),
                nn.GroupNorm(16, 128), nn.GELU(),
            )
        elif model_type == "medium":
            # 64 * 8 = 512
            self.proj_in = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
                nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
                nn.GroupNorm(8, 128), nn.GELU(),
                nn.Conv2d(128, 64, kernel_size=(1, 3), padding=(0, 1)),
                nn.GroupNorm(8, 64), nn.GELU(),
            )
        else:  # small: 25 * 8 = 200
            self.proj_in = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
                nn.GroupNorm(5, 25), nn.GELU(),
                nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
                nn.GroupNorm(5, 25), nn.GELU(),
                nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
                nn.GroupNorm(5, 25), nn.GELU(),
            )

        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model), nn.Dropout(0.1))
        self.channel_embedding = nn.Linear(self.num_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        channel_in = torch.arange(self.num_channels + 1, device=x.device)

        x_flat = x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(x_flat)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(
            bz, ch_num, patch_num, self.d_model)

        spectral_input = x_flat.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(spectral_input, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        group_channels = channel_in[:ch_num]
        group_one_hot = F.one_hot(group_channels, num_classes=self.num_channels).float()
        group_emb = self.channel_embedding(group_one_hot).unsqueeze(0).unsqueeze(2)
        patch_emb = patch_emb + group_emb.expand(bz, -1, patch_num, -1)

        time_emb = self.time_encoding(patch_emb.permute(0, 3, 1, 2))
        patch_emb = patch_emb + time_emb.permute(0, 2, 3, 1)
        return patch_emb


# ---------------------------------------------------------------------------
# ViT Blocks (with separate q_bias / v_bias to match DINO checkpoint)
# ---------------------------------------------------------------------------

class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads,
                               attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# EEG Encoder (backbone)
# Uses `patch_embedding` and `encoder_layers` to match checkpoint key names
# ---------------------------------------------------------------------------

class EEGEncoder(nn.Module):
    def __init__(self, embed_dim: int = 200, depth: int = 12, num_heads: int = 8,
                 mlp_ratio: float = 4.0, model_type: str = "small"):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(d_model=embed_dim, model_type=model_type)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.encoder_layers = nn.Sequential(*[
            _Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, T = x.shape
        x = self.patch_embedding(x)
        x = x.view(B, C * N, self.embed_dim)
        x = self.encoder_layers(x)
        x = self.norm(x)
        return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Presets matched to actual checkpoint architectures
# ---------------------------------------------------------------------------

PRESETS: Dict[str, dict] = {
    "eeg_dino_s": dict(embed_dim=200, depth=12, num_heads=8, mlp_ratio=2.56, model_type="small"),
    "eeg_dino_m": dict(embed_dim=512, depth=16, num_heads=8, mlp_ratio=2.0, model_type="medium"),
    "eeg_dino_l": dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=2.0, model_type="large"),
}


# ---------------------------------------------------------------------------
# Net wrapper
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """EEG-DINO adapter for the baseline pipeline.

    Input:  [B, 26, 2000] (truncated to first 19 channels internally)
    Output: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 1,
        model_name: str = "eeg_dino_s",
        dropout: float = 0.3,
        pretrained_path: str = "",
        freeze_ratio: float = 0.0,
        pretrained: bool = False,
        temporal_pool: int = 1,
    ):
        super().__init__()
        cfg = PRESETS[model_name]

        self.backbone = EEGEncoder(
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            model_type=cfg["model_type"],
        )
        embed_dim = cfg["embed_dim"]

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

        # Freeze early transformer layers to prevent overfitting for large models
        if freeze_ratio > 0:
            depth = cfg["depth"]
            n_freeze = int(depth * freeze_ratio)
            for i, block in enumerate(self.backbone.encoder_layers):
                if i < n_freeze:
                    for p in block.parameters():
                        p.requires_grad = False
            # Also freeze patch embedding
            for p in self.backbone.patch_embedding.parameters():
                p.requires_grad = False
            logger.info("EEG-DINO frozen: patch_embedding + %d/%d encoder layers", n_freeze, depth)

    def _load_pretrained_weights(self, path: str) -> None:
        logger.info("loading EEG-DINO weights from %s", path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        elif 'student' in checkpoint:
            state_dict = checkpoint['student']
        else:
            state_dict = checkpoint

        # Strip prefixes: module.student.xxx -> xxx
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k
            for prefix in ("module.student.", "module.", "backbone."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            # Skip non-backbone keys
            if any(name.startswith(p) for p in ("global_tokens", "projector", "crops", "dino_loss", "ibot")):
                continue
            cleaned[name] = v

        msg = self.backbone.load_state_dict(cleaned, strict=False)
        loaded = len(cleaned) - len(msg.unexpected_keys)
        logger.info("EEG-DINO loaded %d params (missing=%d, unexpected=%d)",
                     loaded, len(msg.missing_keys), len(msg.unexpected_keys))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :19, :]  # [B, 19, 2000]
        B, C, T = x.shape
        x = x.view(B, C, T // 200, 200)  # [B, 19, 10, 200]
        features = self.backbone(x)
        return self.head(features)
