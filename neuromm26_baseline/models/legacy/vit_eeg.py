"""Migrated legacy EEG model module: vit_eeg.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

# models/vit_eeg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 工具：timm/torchvision 预训练兜底 -----------------
def _avg_to_1ch_conv3(w3):  # [out,3,k,k] -> [out,1,k,k]
    with torch.no_grad():
        return w3.mean(dim=1, keepdim=True)

def _build_vit_any(base: str, pretrained: bool, in_chans: int = 3, img_size: int = 224):
    """
    优先 timm -> 若失败回退 torchvision vit_b_16 -> 若还失败则 timm(pretrained=False)
    返回 (model, feat_dim, set_head_fn)
    """
    # 1) timm
    try:
        import timm
        model = timm.create_model(
            base, pretrained=pretrained, in_chans=in_chans, img_size=img_size, num_classes=0
        )
        feat_dim = getattr(model, "num_features", None)
        if feat_dim is None and hasattr(model, "head") and hasattr(model.head, "in_features"):
            feat_dim = model.head.in_features
        assert feat_dim is not None

        def set_head(m, new_head):
            # timm ViT 通常有 reset_classifier，这里直接替换 head
            if hasattr(m, "head"):
                m.head = new_head
            elif hasattr(m, "fc"):
                m.fc = new_head
            else:
                raise RuntimeError("Cannot set head for timm vit model")

        return model, feat_dim, set_head
    except Exception as e:
        print(f"[vit_eeg] WARN: timm load failed -> {e}")

    # 2) torchvision vit_b_16
    try:
        import torchvision
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        tv = vit_b_16(weights=(ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None))
        # 改第一层 conv_proj 的通道数到 in_chans
        if tv.conv_proj.in_channels != in_chans:
            new_proj = nn.Conv2d(
                in_chans, tv.conv_proj.out_channels,
                kernel_size=tv.conv_proj.kernel_size,
                stride=tv.conv_proj.stride,
                padding=tv.conv_proj.padding,
                bias=False
            )
            with torch.no_grad():
                w = tv.conv_proj.weight
                if w.shape[1] == 3 and in_chans == 1:
                    new_proj.weight.copy_(_avg_to_1ch_conv3(w))
                elif w.shape[1] == 3 and in_chans == 3:
                    new_proj.weight.copy_(w)
                else:
                    nn.init.kaiming_normal_(new_proj.weight, nonlinearity="linear")
            tv.conv_proj = new_proj

        feat_dim = tv.heads.head.in_features
        tv.heads.head = nn.Identity()

        def set_head(m, new_head):
            m.heads.head = new_head

        return tv, feat_dim, set_head
    except Exception as e:
        print(f"[vit_eeg] WARN: torchvision vit fallback failed -> {e}")

    # 3) timm(无预训练)
    import timm
    print("[vit_eeg] Fallback -> timm pretrained=False.")
    model = timm.create_model(base, pretrained=False, in_chans=in_chans, img_size=img_size, num_classes=0)
    feat_dim = getattr(model, "num_features", None)
    if feat_dim is None and hasattr(model, "head") and hasattr(model.head, "in_features"):
        feat_dim = model.head.in_features
    assert feat_dim is not None

    def set_head(m, new_head):
        if hasattr(m, "head"):
            m.head = new_head
        elif hasattr(m, "fc"):
            m.fc = new_head
        else:
            raise RuntimeError("Cannot set head for timm vit model")

    return model, feat_dim, set_head


# ----------------- EEG -> ViT 的可学习适配器（关键） -----------------
class EEG2Image(nn.Module):
    """
    把 [B,26,T] 变为 [B,3,H,W] 以适配 ImageNet 预训练的 ViT。
    设计点：
      - per-sample 标准化（z-norm），减少幅值尺度偏移
      - 可学习 2D Conv-stem 生成 3 通道“EEG 图像”，而非简单 repeat
      - H 方向 padding 到 >=32；最后 resize 到 224x224；按 ImageNet mean/std 归一化
    """
    def __init__(self, out_h: int = 32, out_img: int = 224):
        super().__init__()
        self.out_h = out_h
        self.out_img = out_img

        # 轻量 2D conv stem：1->32->16->3
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1, bias=True),
        )

        # ImageNet 标准化（和 ViT 预训练对齐）
        self.register_buffer("imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("imnet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)

    @staticmethod
    def _z_norm(x, dim=-1, eps=1e-6):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, x):
        # x: [B, 26, T]
        if x.dim() != 3:
            raise ValueError(f"EEG2Image expects [B,26,T], got {tuple(x.shape)}")
        B, C, T = x.shape

        # 每样本 z-norm（在时间维）
        x = self._z_norm(x, dim=-1)
        # 转成 “单通道图像” [B,1,26,T]
        x = x.unsqueeze(1)

        # 可学习 stem 生成 3 通道特征
        x = self.stem(x)   # [B,3,26,T]

        # pad 高度到 >= 32（ConvNeXt 的经验；ViT 也可稳定些）
        H = x.shape[2]
        if H < self.out_h:
            need = self.out_h - H
            pad_top = need // 2
            pad_bot = need - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bot), mode='reflect')

        # 双线性缩放到 224x224
        x = F.interpolate(x, size=(self.out_img, self.out_img), mode="bilinear", align_corners=False)

        # ImageNet 标准化
        x = (x - self.imnet_mean) / self.imnet_std
        return x


# ----------------- 主模型 -----------------
class Net(nn.Module):
    """
    ViT-EEG（改良版）：
      - EEG2Image 可学习适配器：把 [B,26,T] -> [B,3,224,224]
      - ViT backbone（timm/torchvision 预训练，离线兜底）
      - 分类头前统一 Dropout
    """
    def __init__(
        self,
        num_classes: int = 1,
        base: str = "vit_base_patch16_224",
        pretrained: bool = True,
        temporal_pool: int = 1,      # 仍保留占位，若你要在 dataloader 前做下采样，可以传 >1
        dropout: float = 0.3,
        img_size: int = 224,
    ):
        super().__init__()
        self.adapter = EEG2Image(out_h=32, out_img=img_size)

        # 构建 ViT 骨干
        self.backbone, feat_dim, set_head = _build_vit_any(base, pretrained, in_chans=3, img_size=img_size)
        # 去掉原分类头
        set_head(self.backbone, nn.Identity())

        self.pre_logits_drop = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

        # 轻微初始化分类头
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # x: [B, 26, T] 或 [B,1,26,T]
        if x.dim() == 4:
            # 如果你的 pipeline 已经是 [B,1,26,T]，还原到 [B,26,T]
            if x.size(1) == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(f"Expected input [B,26,T] or [B,1,26,T], got {tuple(x.shape)}")

        x_img = self.adapter(x)                # [B,3,224,224]
        feats = self.backbone(x_img)           # [B, feat_dim]
        feats = self.pre_logits_drop(feats)
        logits = self.classifier(feats)        # [B, num_classes]
        return logits
