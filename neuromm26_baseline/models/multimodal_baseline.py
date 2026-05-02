"""EEG + video feature fusion baseline."""

from __future__ import annotations

import torch.nn as nn

from .backbone import EEGResNetEncoder, VideoFeatureProjector
from .fusion.concat_fusion import ConcatFusion
from .heads.classification_head import BinaryClassificationHead


class EEGVideoFusionModel(nn.Module):
    def __init__(
        self,
        eeg_backbone: str = "resnet18",
        eeg_pretrained: bool = False,
        eeg_temporal_pool: int = 10,
        eeg_dropout: float = 0.2,
        video_embedding_dim: int = 256,
        fusion_hidden_dim: int = 256,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.eeg_encoder = EEGResNetEncoder(
            base=eeg_backbone,
            pretrained=eeg_pretrained,
            temporal_pool=eeg_temporal_pool,
            dropout=eeg_dropout,
        )
        self.video_projector = VideoFeatureProjector(output_dim=video_embedding_dim, dropout=classifier_dropout)
        self.fusion = ConcatFusion()
        self.head = BinaryClassificationHead(
            input_dim=self.eeg_encoder.feature_dim + video_embedding_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=classifier_dropout,
        )

    def forward(self, batch):
        eeg_feature = self.eeg_encoder(batch["eeg"])
        video_feature = self.video_projector(batch["video_feature"])
        fused = self.fusion(eeg_feature, video_feature)
        return self.head(fused)
