"""Migrated legacy EEG model module: rnn_eeg_big.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 前端：可选的时域卷积（不降采样或小步幅）
# ---------------------------
class TemporalConvFront(nn.Module):
    def __init__(self, channels: int = 26, layers: int = 2, kernel_size: int = 9, stride: int = 1, expansion: int = 2):
        super().__init__()
        assert layers >= 1
        blocks = []
        c_in = channels
        for i in range(layers):
            dw = nn.Conv1d(c_in, c_in, kernel_size, stride=stride if i == 0 else 1,
                           padding=kernel_size // 2, groups=c_in, bias=False)
            pw = nn.Conv1d(c_in, c_in * expansion, kernel_size=1, bias=False)
            bn1 = nn.BatchNorm1d(c_in * expansion)
            act = nn.ReLU(inplace=True)
            pw2 = nn.Conv1d(c_in * expansion, c_in, kernel_size=1, bias=False)
            bn2 = nn.BatchNorm1d(c_in)
            blocks += [dw, pw, bn1, act, pw2, bn2, nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# EEGNet-style spatial frontend: temporal conv → spatial depthwise → pooling
# Reduces raw [B, 26, 2000] to [B, out_channels, T'] with spatial abstraction
# ---------------------------
class EEGSpatialFront(nn.Module):
    def __init__(
        self,
        in_chans: int = 26,
        F1: int = 16,
        D: int = 2,
        eeg_kernel: int = 64,
        pool1: int = 4,
        pool2: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        F2 = F1 * D
        self.out_channels = F2

        # 1) Temporal convolution: [B,1,C,T] -> [B,F1,C,T]
        self.conv_t = nn.Conv2d(1, F1, kernel_size=(1, eeg_kernel),
                                padding=(0, eeg_kernel // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # 2) Spatial depthwise: [B,F1,C,T] -> [B,F2,1,T]
        self.conv_s = nn.Conv2d(F1, F2, kernel_size=(in_chans, 1),
                                groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # 3) Separable temporal: [B,F2,1,T'] -> [B,F2,1,T'']
        self.conv_sep = nn.Conv2d(F2, F2, kernel_size=(1, 16),
                                  padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2))
        self.drop2 = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.conv_t.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_s.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_sep.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 26, T] -> [B, 1, 26, T]
        x = x.unsqueeze(1)
        x = self.conv_t(x)
        x = self.bn1(x)
        x = self.conv_s(x)         # [B, F2, 1, T]
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv_sep(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.squeeze(2)        # [B, F2, T']


# ---------------------------
# 多头注意力池化
# ---------------------------
class MultiHeadAttnPool(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, head_dim: Optional[int] = None, concat: bool = False):
        super().__init__()
        assert num_heads >= 1
        self.dim = dim
        self.h = num_heads
        self.concat = concat

        if head_dim is None:
            head_dim = max(32, dim // num_heads)
        self.head_dim = head_dim

        self.q = nn.Parameter(torch.randn(num_heads, head_dim))
        self.proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out = nn.Linear(num_heads * head_dim if concat else head_dim, dim, bias=False)

        nn.init.normal_(self.q, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.h, self.head_dim
        kv = self.proj(x).view(B, T, h, d)        # [B,T,H,D]
        att = torch.einsum('bthd,hd->bht', kv, self.q) / math.sqrt(d)
        att = torch.softmax(att, dim=-1)          # [B,H,T]
        pooled = torch.einsum('bht,bthd->bhd', att, kv)  # [B,H,D]
        pooled = pooled.reshape(B, h * d) if self.concat else pooled.mean(dim=1)  # [B,H*D] or [B,D]
        return self.out(pooled)                   # [B,C]


# ---------------------------
# 残差 BiRNN Block（投影+LayerNorm）
# ---------------------------
class ResidualBiRNNBlock(nn.Module):
    def __init__(self, rnn_type: str, model_dim: int, hidden: int, dropout: float, ln_eps: float = 1e-5):
        super().__init__()
        if rnn_type == "lstm":
            rnn = nn.LSTM(model_dim, hidden, num_layers=1, dropout=0.0,
                          bidirectional=True, batch_first=True)
        elif rnn_type == "gru":
            rnn = nn.GRU(model_dim, hidden, num_layers=1, dropout=0.0,
                         bidirectional=True, batch_first=True)
        elif rnn_type == "rnn_tanh":
            rnn = nn.RNN(model_dim, hidden, num_layers=1, nonlinearity="tanh",
                         dropout=0.0, bidirectional=True, batch_first=True)
        elif rnn_type == "rnn_relu":
            rnn = nn.RNN(model_dim, hidden, num_layers=1, nonlinearity="relu",
                         dropout=0.0, bidirectional=True, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.rnn = rnn
        self.proj = nn.Linear(2 * hidden, model_dim, bias=True)
        self.ln = nn.LayerNorm(model_dim, eps=ln_eps)
        self.do = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                if isinstance(self.rnn, nn.LSTM):
                    hs = self.rnn.hidden_size
                    p.data[hs:2*hs].fill_(1.0)  # forget gate

    def forward(self, x):
        y, _ = self.rnn(x)     # [B,T,2*hidden]
        y = self.proj(y)       # [B,T,C]
        y = self.do(y)
        return self.ln(x + y)


# ---------------------------
# 大型（缩半版）RNN 网络
# ---------------------------
class Net(nn.Module):
    """
    缩半版大型 RNN EEG 分类模型（从头训练）：
      - 输入 [B, 26, T]
      - 轻量深度前端卷积（默认 stride=1，不降采样）
      - 1x1 Conv: 26 -> model_dim（缩半）
      - 残差 BiRNN blocks（层数与 hidden 缩减）
      - 多头注意力池化（头数缩减）
      - MLP 头缩半
    """
    PRESETS: Dict[str, dict] = {
        "lstm_big": dict(
            rnn_type="lstm", model_dim=128, hidden=256, blocks=3,
            mh_heads=4, mh_concat=False, block_dropout=0.15,
            # EEG spatial frontend params
            eeg_F1=16, eeg_D=4, eeg_kernel=64, eeg_pool1=4, eeg_pool2=2,
            head_mlp=[256], head_dropout=0.3
        ),
        "lstm_huge": dict(
            rnn_type="lstm", model_dim=192, hidden=384, blocks=4,
            mh_heads=4, mh_concat=False, block_dropout=0.15,
            eeg_F1=24, eeg_D=4, eeg_kernel=64, eeg_pool1=4, eeg_pool2=2,
            head_mlp=[384, 192], head_dropout=0.3
        ),
        "gru_big": dict(
            rnn_type="gru", model_dim=128, hidden=256, blocks=3,
            mh_heads=4, mh_concat=False, block_dropout=0.15,
            eeg_F1=16, eeg_D=4, eeg_kernel=64, eeg_pool1=4, eeg_pool2=2,
            head_mlp=[256], head_dropout=0.3
        ),
    }

    def __init__(
        self,
        num_classes: int = 1,
        base: str = "lstm_big",
        pretrained: bool = False,
        temporal_pool: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if base not in self.PRESETS:
            raise ValueError(f"Unsupported base: {base}. Choose from {list(self.PRESETS)}")
        cfg = dict(self.PRESETS[base])

        self.model_dim = cfg["model_dim"]

        # EEGNet-style spatial frontend: [B,26,2000] -> [B,F2,T']
        self.front = EEGSpatialFront(
            in_chans=26,
            F1=cfg["eeg_F1"],
            D=cfg["eeg_D"],
            eeg_kernel=cfg["eeg_kernel"],
            pool1=cfg["eeg_pool1"],
            pool2=cfg["eeg_pool2"],
            dropout=0.15,
        )
        front_out_ch = self.front.out_channels  # F1 * D

        self.input_proj = nn.Conv1d(front_out_ch, self.model_dim, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        blocks = []
        for _ in range(cfg["blocks"]):
            blocks.append(
                ResidualBiRNNBlock(
                    rnn_type=cfg["rnn_type"],
                    model_dim=self.model_dim,
                    hidden=cfg["hidden"],
                    dropout=cfg["block_dropout"],
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.pool = MultiHeadAttnPool(
            dim=self.model_dim,
            num_heads=cfg["mh_heads"],
            head_dim=None,
            concat=cfg["mh_concat"],
        )

        head_layers = []
        in_dim = self.model_dim
        for hd in cfg["head_mlp"]:
            fc = nn.Linear(in_dim, hd)
            nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
            nn.init.zeros_(fc.bias)
            head_layers += [fc, nn.ReLU(inplace=True), nn.Dropout(cfg["head_dropout"])]
            in_dim = hd
        self.head = nn.Sequential(*head_layers)

        self.global_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if x.dim() != 3 or x.size(1) != 26:
            raise ValueError(f"Expected x as [B, 26, T], got {tuple(x.shape)}")
        x = x.float()

        x = self.front(x)                      # [B,F2,T'] (spatially abstracted + temporally reduced)
        x = self.input_proj(x)                 # [B,Cm,T']
        x = x.transpose(1, 2).contiguous()     # [B,T',Cm]

        x = self.blocks(x)                     # [B,T',Cm]
        x = self.pool(x)                       # [B,Cm]
        x = self.head(x)                       # [B,H]
        x = self.global_dropout(x)
        logits = self.classifier(x)            # [B,num_classes]
        return logits
