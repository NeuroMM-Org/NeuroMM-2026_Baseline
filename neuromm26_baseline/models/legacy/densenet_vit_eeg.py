"""Migrated legacy EEG model module: densenet_vit_eeg.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

# models/densenet_vit_eeg.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models import densenet121

import timm


# -------------------------------------------------
# 1) DenseNet 前端：适配 EEG 的高(H=26) + 1ch + 下采样
# -------------------------------------------------
class DenseNetFrontend(nn.Module):
    """
    用 ImageNet 预训练的 densenet121 做 CNN 前端：
    - 把 stem 的 stride 从 2 -> 1
    - 把 pool0 关掉（H=26 本来就很矮）
    - 把第一层 3ch 改成 1ch，并把权重在通道维做平均迁移
    - 最后投影到 3 通道，给 ViT 用
    """
    def __init__(self, pretrained: bool = True, proj_out: int = 3):
        super().__init__()
        # 1) 加载预训练 densenet121
        m = densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        # 2) 把第一层 conv 改成 1 通道
        conv0 = m.features.conv0  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=(1, 1),              # ★ stride 调成 1
            padding=conv0.padding,
            bias=False,
        )
        with torch.no_grad():
            # 把 3 通道的预训练权重平均到 1 通道
            w = conv0.weight  # [64, 3, 7, 7]
            w_mean = w.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            new_conv0.weight.copy_(w_mean)
        m.features.conv0 = new_conv0

        # 3) 把 pool0 干掉（不然 26 再除 2 高度就太矮了）
        m.features.pool0 = nn.Identity()

        self.backbone = m.features  # DenseNet 的特征部分（到 norm5）
        # 这个输出一般是 [B, 1024, H', W']，H' ~ 26, W' ~ T/...
        # 我们再接一个 1x1 把 1024 压到 3 通道，给 ViT 当成 “强化后图像”
        self.proj = nn.Conv2d(1024, proj_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]   (H≈26, W≈2000/temporal_pool)
        feats = self.backbone(x)     # [B, 1024, H', W']
        feats = self.proj(feats)     # [B, 3, H', W']
        return feats


# -------------------------------------------------
# 2) ViT 主干（timm），去掉分类头，输出特征
# -------------------------------------------------
def _build_vit_backbone(
    name: str = "vit_base_patch16_224",
    pretrained: bool = True,
):
    """
    timm ViT:
    - num_classes=0 删分类头
    - global_pool='' 不做 pool
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )
    feat_dim = model.num_features
    return model, feat_dim


# -------------------------------------------------
# 3) 整体模型：DenseNet -> 融合 -> ViT -> BCE
# -------------------------------------------------
class Net(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        # densenet 相关
        pretrained_dense: bool = True,
        # vit 相关
        vit_name: str = "vit_base_patch16_224",
        pretrained_vit: bool = True,
        # 输入尺寸 & 下采样
        img_size: int = 224,
        temporal_pool: int = 10,
        # 融合
        fusion: str = "residual",  # 'residual' | 'concat'
        dropout: float = 0.3,
    ):
        super().__init__()
        assert fusion in ["residual", "concat"]
        self.fusion = fusion
        self.img_size = img_size
        self.temporal_pool = temporal_pool

        # 1) DenseNet 特征前端
        self.cnn = DenseNetFrontend(pretrained=pretrained_dense, proj_out=3)

        # 2) ViT 主干
        self.vit, self.feat_dim = _build_vit_backbone(vit_name, pretrained=pretrained_vit)

        # 3) 融合层
        if self.fusion == "residual":
            # 可学习的融合权重 α
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            # concat 时把 [3 + 3] -> 3
            self.fuse_conv = nn.Conv2d(6, 3, kernel_size=1, bias=True)

        # 4) 尺寸对齐
        self.upsample_to_vit = nn.Upsample(size=(img_size, img_size),
                                           mode="bilinear", align_corners=False)

        # 5) 分类头
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    # ----------------- 预处理 -----------------
    def _to_eeg_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 26, T] or [B, 1, 26, T]
        return: [B, 1, H, W]  (H>=32 -> pad -> resize)
        """
        if x.dim() == 3:
            # [B, 26, T] -> [B, 1, 26, T]
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = x.float()

        # 时间维做降采样
        if self.temporal_pool is not None and self.temporal_pool > 1:
            x = F.avg_pool2d(x, kernel_size=(1, self.temporal_pool),
                             stride=(1, self.temporal_pool))

        # 高度不足 32 就反射填充到 32（和你其他模型一致）
        H = x.size(2)
        if H < 32:
            need = 32 - H
            pad_top = need // 2
            pad_bot = need - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bot), mode="reflect")

        # resize -> vit 输入尺寸
        x = self.upsample_to_vit(x)  # [B, 1, 224, 224]
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # 1) EEG -> 1ch image
    #     img_1ch = self._to_eeg_image(x)         # [B,1,224,224]

    #     # 2) DenseNet 前端
    #     dense_feats = self.cnn(img_1ch)         # [B,3,h',w']
    #     dense_feats_up = F.interpolate(
    #         dense_feats,
    #         size=img_1ch.shape[-2:],
    #         mode="bilinear",
    #         align_corners=False,
    #     )                                       # [B,3,224,224]

    #     # 3) 原始 1ch -> 假 3ch，保持与 ImageNet 输入一致
    #     img_3ch = img_1ch.repeat(1, 3, 1, 1)    # [B,3,224,224]

    #     # 4) 融合
    #     if self.fusion == "residual":
    #         alpha = torch.sigmoid(self.alpha)
    #         fused = alpha * dense_feats_up + (1.0 - alpha) * img_3ch   # [B,3,224,224]
    #     else:
    #         fused = torch.cat([img_3ch, dense_feats_up], dim=1)        # [B,6,224,224]
    #         fused = self.fuse_conv(fused)                              # [B,3,224,224]

    #     # 5) ViT
    #     feats = self.vit.forward_features(fused)                       # [B,768] for vit-base
    #     feats = self.vit.forward_head(feats, pre_logits=True)          # [B,768]
    #     feats = self.dropout(feats)
    #     logits = self.classifier(feats)                                # [B,1] or [B]

    #     # 6) 强制 [B,1]
    #     if logits.dim() == 1:
    #         logits = logits.unsqueeze(1)
    #     elif logits.size(1) != 1:
    #         logits = logits[:, :1]

    #     return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) EEG -> 1ch image
        img_1ch = self._to_eeg_image(x)         # [B,1,224,224]

        # 2) DenseNet 前端
        dense_feats = self.cnn(img_1ch)         # [B,3,h',w']
        dense_feats_up = F.interpolate(
            dense_feats,
            size=img_1ch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )                                       # [B,3,224,224]

        # 3) 原始 1ch -> 假 3ch
        img_3ch = img_1ch.repeat(1, 3, 1, 1)    # [B,3,224,224]

        # 4) 融合
        if self.fusion == "residual":
            alpha = torch.sigmoid(self.alpha)
            fused = alpha * dense_feats_up + (1.0 - alpha) * img_3ch
        else:
            fused = torch.cat([img_3ch, dense_feats_up], dim=1)  # [B,6,224,224]
            fused = self.fuse_conv(fused)                        # [B,3,224,224]

        # 5) ViT
        feats = self.vit.forward_features(fused)                  # [B,768]
        feats = self.vit.forward_head(feats, pre_logits=True)     # [B,768]
        feats = self.dropout(feats)
        logits = self.classifier(feats)                           # 可能是 [B,1] 或 [B,1,1]

        # ✅ 强制成 [B, 1]
        logits = logits.view(logits.size(0), -1)                  # [B, 1] or [B, d]
        if logits.size(1) > 1:
            logits = logits[:, :1]
        return logits
