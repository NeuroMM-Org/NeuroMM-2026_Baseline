"""Migrated legacy EEG model module: vit_hybrid_eeg.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

# models/vit_hybrid_eeg.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torchvision
from torchvision.models import resnet18, resnet50


# ------------------------------
# 1) ResNet 前端（截到指定 stage 输出特征图） -> 投影到 3 通道
# ------------------------------
class ResNetFrontend(nn.Module):
    """
    使用 torchvision ResNet 作为 CNN 前端：
    - 保留 stem + layer1 + layer2 (+ layer3 可选)，输出一个特征图
    - 再用 1x1 Conv 投影到 3 通道，便于与 ViT 对接（当作“增强后”的伪 RGB 图）
    """
    def __init__(self, base: str = "resnet18", pretrained: bool = True, out_stage: int = 2):
        """
        base: 'resnet18' | 'resnet50'
        out_stage: 2 or 3（取到 layer2 或 layer3；3 更“语义化”，显存更大）
        """
        super().__init__()
        assert base in ["resnet18", "resnet50"], f"Unsupported ResNet base: {base}"
        assert out_stage in [2, 3], "out_stage must be 2 or 3"

        if base == "resnet18":
            m = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_planes = 128 if out_stage == 2 else 256
        else:
            m = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            in_planes = 512 if out_stage == 2 else 1024

        # stem
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool

        # stages
        self.layer1 = m.layer1  # stride /4
        self.layer2 = m.layer2  # stride /8
        self.layer3 = m.layer3  # stride /16

        self.out_stage = out_stage
        # 输出到 3 通道，供 ViT 使用
        self.proj3 = nn.Conv2d(in_planes, 3, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] 已经是伪 RGB EEG 图
        return: [B, 3, H', W'] (后续会再上采样到 ViT 输入尺寸)
        """
        x = self.conv1(x)   # [B, 64, /2, /2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, /4, /4]

        x = self.layer1(x)  # [B, C1, /4, /4]
        x = self.layer2(x)  # [B, C2, /8, /8]
        if self.out_stage == 3:
            x = self.layer3(x)  # [B, C3, /16, /16]

        x = self.proj3(x)   # [B, 3, /s, /s]
        return x


# ------------------------------
# 2) ViT 主干（来自 timm）
# ------------------------------
def _build_vit_backbone(name: str = "vit_base_patch16_224", pretrained: bool = True):
    """
    返回一个去掉分类头的 ViT：
      - num_classes=0：去掉原分类头
      - global_pool='token'：让 forward_head 返回 CLS/token 池化后的 768 维向量
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,       # 删除分类头
        global_pool='token'  # 使 forward_head 返回 CLS / token 池化特征
    )
    feat_dim = model.num_features  # vit-base = 768
    return model, feat_dim


# ------------------------------
# 3) 混合模型：CNN 前端增强 + ViT 判别
# ------------------------------
class Net(nn.Module):
    """
    EEG => 伪 RGB => (ResNet 前端增强) => 融合 => ViT (timm 预训练) => Dropout => Linear

    关键点：
    - ResNet & ViT 都可加载 ImageNet 预训练
    - 融合策略默认 residual：enh = conv(resnet_feats upsample)；y = σ(α)*enh + (1-σ(α))*rgb
    - 输出 logits 形状固定为 [B,1]，匹配 BCE / 焦点损失
    """
    def __init__(
        self,
        num_classes: int = 1,
        base_vit: str = "vit_base_patch16_224",
        base_cnn: str = "resnet18",
        pretrained_vit: bool = True,
        pretrained_cnn: bool = True,
        img_size: int = 224,
        temporal_pool: int = 1,     # >1: 先在时间维做 AvgPool1d(temporal_pool)
        dropout: float = 0.3,
        cnn_out_stage: int = 2,     # 2(layer2) 或 3(layer3)
        fusion: str = "residual",   # 'residual' | 'concat'
        freeze_cnn_stem: bool = False,  # 可选：冻结 ResNet 的浅层
    ):
        super().__init__()
        assert fusion in ["residual", "concat"]

        # 前端：ResNet
        self.cnn = ResNetFrontend(base=base_cnn, pretrained=pretrained_cnn, out_stage=cnn_out_stage)

        # 主干：ViT（token 池化，返回固定 768 维）
        self.vit, self.feat_dim = _build_vit_backbone(base_vit, pretrained=pretrained_vit)

        # 分类头（统一 Dropout）
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # 融合参数
        self.fusion = fusion
        if self.fusion == "residual":
            # alpha 可学习，最终通过 sigmoid 映射到 (0,1)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            # concat: 拼接后用 1x1 降到 3 通道再给 ViT
            self.fuse_conv = nn.Conv2d(6, 3, kernel_size=1, bias=True)

        # 预处理相关
        self.img_size = img_size
        self.temporal_pool = temporal_pool
        self.upsample_to_vit = nn.Upsample(size=(img_size, img_size), mode="bilinear", align_corners=False)

        if freeze_cnn_stem:
            for p in self.cnn.conv1.parameters():
                p.requires_grad = False
            for p in self.cnn.bn1.parameters():
                p.requires_grad = False
            # 可按需继续冻结 layer1 等

    # ---- 预处理：按你的数据形状做通道&尺寸对齐 ----
    def _preprocess_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 26, T] 或 [B, 1, 26, T]
        return: [B, 3, img_size, img_size]
        """
        if x.dim() == 3:  # [B, C(=26), T]
            x = x.unsqueeze(1)  # -> [B, 1, 26, T]
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected x [B, 26, T] or [B, 1, 26, T], got {tuple(x.shape)}")

        x = x.float()

        # 时间下采样（可选）
        if self.temporal_pool is not None and self.temporal_pool > 1:
            # pool over width (time) 维度
            x = F.avg_pool2d(x, kernel_size=(1, self.temporal_pool), stride=(1, self.temporal_pool))

        # 保证 H>=32，避免早期步幅问题
        H = x.size(2)  # 高度是通道维(26)
        if H < 32:
            need = 32 - H
            pad_top = need // 2
            pad_bot = need - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bot), mode='reflect')  # 只 pad 高度

        # resize -> img_size
        x = self.upsample_to_vit(x)  # [B, 1, img, img]
        # 复制到 3 通道，保持与 ImageNet 预训练一致
        if x.size(1) == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
        else:
            x_rgb = x[:, :3, :, :]
        return x_rgb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 预处理为伪 RGB 图像
        rgb = self._preprocess_to_rgb(x)          # [B,3,H,H]

        # 2) CNN 前端增强（带预训练）
        feat3 = self.cnn(rgb)                     # [B,3,h,w]
        feat3_up = F.interpolate(feat3, size=rgb.shape[-2:], mode='bilinear', align_corners=False)

        # 3) 融合到 ViT 输入
        if self.fusion == "residual":
            alpha = torch.sigmoid(self.alpha)     # (0,1)
            fused = alpha * feat3_up + (1.0 - alpha) * rgb
        else:  # concat
            fused = torch.cat([rgb, feat3_up], dim=1)  # [B,6,H,W]
            fused = self.fuse_conv(fused)              # [B,3,H,W]

        # 4) ViT 特征（固定 768 维） + Dropout + Linear
        feats = self.vit.forward_features(fused)                 # 可能是 [B, 197, 768] 或 [B, 768]（视 timm 版本而定）
        if feats.dim() == 3:
            # 如果返回 token 序列，取 CLS
            feats = feats[:, 0, :]                               # [B, 768]
        elif feats.dim() == 2:
            # 已是 [B, 768]
            pass
        else:
            raise RuntimeError(f"Unexpected ViT features shape: {tuple(feats.shape)}")

        # （可选）也可以直接用 forward_head（global_pool='token' 已设置）
        # feats = self.vit.forward_head(feats, pre_logits=True)   # 通常也是 [B, 768]

        feats = self.dropout(feats)
        logits = self.classifier(feats)                          # 期望 [B, 1]

        # 兜底保证 [B,1]
        logits = logits.reshape(logits.size(0), -1)
        if logits.size(1) != 1:
            logits = logits[:, :1]
        return logits
