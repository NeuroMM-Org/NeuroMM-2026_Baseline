"""Migrated legacy EEG model module: resnet_eeg.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    # torchvision >= 0.13 的权重写法
    import torchvision
    from torchvision.models import resnet18, resnet50
    from torchvision.models import ResNet18_Weights, ResNet50_Weights
    _HAS_TV_WEIGHTS_ENUM = True
except Exception:
    import torchvision
    from torchvision.models import resnet18, resnet50
    _HAS_TV_WEIGHTS_ENUM = False


def _load_resnet(base: str, pretrained: bool):
    """
    返回 torchvision 的 resnet backbone，并把 fc 替换为 Identity，
    方便后续接自定义分类头。
    """
    if base == 'resnet18':
        if pretrained:
            if _HAS_TV_WEIGHTS_ENUM:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = resnet18(pretrained=True)
        else:
            model = resnet18(weights=None) if _HAS_TV_WEIGHTS_ENUM else resnet18(pretrained=False)
        feat_dim = 512
    elif base == 'resnet50':
        if pretrained:
            if _HAS_TV_WEIGHTS_ENUM:
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = resnet50(pretrained=True)
        else:
            model = resnet50(weights=None) if _HAS_TV_WEIGHTS_ENUM else resnet50(pretrained=False)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported base: {base}")

    # 去掉最终分类层，保留 backbone + avgpool
    model.fc = nn.Identity()
    return model, feat_dim


def _convert_first_conv_to_single_channel(backbone: nn.Module):
    """
    将 ResNet 的首层 7x7 卷积从 3 通道改为 1 通道，并迁移预训练权重：
    使用 RGB 权重在通道维平均。
    """
    conv1: nn.Conv2d = backbone.conv1
    if conv1.in_channels == 1:
        return backbone  # 已经是单通道

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=(conv1.bias is not None)
    )

    with torch.no_grad():
        # 原权重 [out_c, 3, k, k] -> 取通道均值到 [out_c, 1, k, k]
        new_weight = conv1.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(new_weight)
        if conv1.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv1.bias)

    backbone.conv1 = new_conv
    return backbone


class Net(nn.Module):
    """
    将 EEG [B, 26, 2000] 视作单通道图像 [B, 1, 26, 2000]，
    送入 ResNet Backbone（预训练）后，全局池化 + Dropout + 线性层 -> logits。
    支持 temporal_pool：对时间轴（宽度维）做平均池化以降采样。
    """
    def __init__(
        self,
        num_classes: int = 1,
        base: str = 'resnet50',
        pretrained: bool = True,
        temporal_pool: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_classes >= 1
        assert base in ['resnet18', 'resnet50']

        backbone, feat_dim = _load_resnet(base, pretrained=pretrained)
        backbone = _convert_first_conv_to_single_channel(backbone)
        self.backbone = backbone

        # 时间轴降采样（对宽度维做池化），不改变通道数/高度
        self.temporal_pool: Optional[nn.Module]
        if temporal_pool is not None and temporal_pool > 1:
            self.temporal_pool = nn.AvgPool2d(kernel_size=(1, temporal_pool), stride=(1, temporal_pool))
        else:
            self.temporal_pool = None

        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        """
        x: [B, 26, 2000]  -> unsqueeze -> [B, 1, 26, 2000]
        返回: logits [B, num_classes]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, 26, 2000]
        elif x.dim() == 4:
            # 允许传入 [B, 1, 26, 2000]
            pass
        else:
            raise ValueError(f"Expected x with shape [B, 26, T] or [B, 1, 26, T], got {x.shape}")

        x = x.float()

        # 可选的时间轴降采样
        if self.temporal_pool is not None:
            x = self.temporal_pool(x)  # [B, 1, 26, 2000//temporal_pool]

        # Backbone forward：resnet 已包含 avgpool（AdaptiveAvgPool2d -> [B, C, 1, 1] -> flatten）
        feats = self.backbone(x)   # [B, feat_dim]
        feats = self.dropout(feats)
        logits = self.classifier(feats)  # [B, num_classes]
        return logits
