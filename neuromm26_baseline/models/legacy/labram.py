"""Migrated legacy EEG model module: labram.

Large Brain Model (LaBraM) for EEG classification.  Based on BEiT-v2 with
temporal convolution patch embedding, channel/time positional encoding, and
a standard Vision Transformer backbone.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) >= self.drop_prob
        return x / (1.0 - self.drop_prob) * keep


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm=None, qk_scale: float | None = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_norm=None, qk_scale: float | None = None,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 init_values: float | None = None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_norm=qk_norm, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    """Temporal convolution patch embedding for raw EEG input."""
    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, A, T] -> rearrange to [B, N*A, T] -> [B, 1, N*A, T]
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


# ---------------------------------------------------------------------------
# NeuralTransformer (LaBraM backbone)
# ---------------------------------------------------------------------------

class NeuralTransformer(nn.Module):
    def __init__(
        self,
        EEG_size: int = 1600,
        patch_size: int = 200,
        in_chans: int = 1,
        out_chans: int = 8,
        num_classes: int = 1000,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm=None,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_values: float | None = None,
        use_abs_pos_emb: bool = True,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        input_channels: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, input_channels + 1, embed_dim)) if use_abs_pos_emb else None
        self.time_embed = nn.Parameter(torch.zeros(1, self.time_window, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.time_embed is not None:
            nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
        self._fix_init_weight()
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, input_chans=None) -> torch.Tensor:
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(
                batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed

        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
                batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.fc_norm is not None:
            return self.fc_norm(x[:, 1:, :].mean(1))
        return x[:, 0]

    def forward(self, x: torch.Tensor, input_chans=None, **kwargs) -> torch.Tensor:
        x = self.forward_features(x, input_chans=input_chans)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Net wrapper for the baseline pipeline
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """LaBraM adapter for the baseline pipeline.

    Input:  [B, 26, 2000]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 26,
        input_samples: int = 2000,
        patch_size: int = 200,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        out_chans: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        dropout: float = 0.3,
        pretrained_path: str = "",
        # unused — kept for registry compatibility
        pretrained: bool = False,
        temporal_pool: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.seq_len = input_samples // patch_size

        self.backbone = NeuralTransformer(
            EEG_size=input_samples,
            patch_size=patch_size,
            in_chans=1,
            out_chans=out_chans,
            num_classes=0,  # no head inside backbone
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=1e-4,
            input_channels=input_channels,
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

        self.pre_logits_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def _load_pretrained_weights(self, path: str) -> None:
        logger.info("loading LaBraM pretrained weights from %s", path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)

        # LaBraM checkpoints wrap weights under "student." prefix — strip it
        stripped = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('student.'):
                stripped[k[len('student.'):]] = v
        if stripped:
            state_dict = stripped

        # Filter out head / projection layers
        skip_prefixes = ('head.', 'lm_head.', 'projection_head.')
        filtered = OrderedDict(
            (k, v) for k, v in state_dict.items()
            if not any(k.startswith(p) for p in skip_prefixes)
        )

        # pos_embed / time_embed may differ in size — interpolate to match
        for embed_key in ('pos_embed', 'time_embed'):
            if embed_key in filtered and embed_key in dict(self.backbone.named_parameters()):
                src = filtered[embed_key]  # [1, src_len, dim]
                tgt = dict(self.backbone.named_parameters())[embed_key]
                if src.shape != tgt.shape:
                    logger.info("LaBraM %s: interpolating %s -> %s",
                                embed_key, tuple(src.shape), tuple(tgt.shape))
                    # Interpolate along the token dimension: [1, src_len, dim] -> [1, tgt_len, dim]
                    src_t = src.permute(0, 2, 1)  # [1, dim, src_len]
                    resized = torch.nn.functional.interpolate(
                        src_t, size=tgt.shape[1], mode='linear', align_corners=False,
                    )  # [1, dim, tgt_len]
                    filtered[embed_key] = resized.permute(0, 2, 1)  # [1, tgt_len, dim]

        msg = self.backbone.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(msg.unexpected_keys)
        logger.info("LaBraM loaded %d/%d params (missing=%d, unexpected=%d)",
                     loaded, len(filtered), len(msg.missing_keys), len(msg.unexpected_keys))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 26, 2000] -> [B, 26, 10, 200]
        if x.dim() == 3:
            x = x.view(x.size(0), x.size(1), self.seq_len, self.patch_size)
        feats = self.backbone.forward_features(x)
        feats = self.pre_logits_drop(feats)
        return self.classifier(feats)
