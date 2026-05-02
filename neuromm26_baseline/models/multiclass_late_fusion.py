"""5-class late-fusion baseline for Task3.

Architecture:
    EEG batch ---[legacy EEG model with num_classes=5]---> eeg_logits (B, 5)
                                                                     \\
                                                                      --> Linear(10, 5) --> final logits (B, 5)
                                                                     /
    video feature (T,D) -[mean-pool]-[MLP D->H->5]-> video_logits (B, 5)

Combiner is initialized so the initial behaviour is the per-class average of
the two branch logits; gradient lets it adapt the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .legacy.registry_task3 import build_legacy_eeg_model_with_num_classes


class EEGVideoLateFusion5Class(nn.Module):
    def __init__(
        self,
        eeg_model_name: str,
        video_feature_dim: int,
        video_hidden_dim: int = 256,
        video_dropout: float = 0.2,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eeg_model = build_legacy_eeg_model_with_num_classes(eeg_model_name, num_classes)
        self.video_branch = nn.Sequential(
            nn.Linear(video_feature_dim, video_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(video_dropout),
            nn.Linear(video_hidden_dim, num_classes),
        )
        # Combine [eeg_logits (B, C), video_logits (B, C)] -> (B, C)
        self.combiner = nn.Linear(2 * num_classes, num_classes)
        with torch.no_grad():
            w = torch.zeros(num_classes, 2 * num_classes)
            for c in range(num_classes):
                w[c, c] = 0.5
                w[c, num_classes + c] = 0.5
            self.combiner.weight.copy_(w)
            self.combiner.bias.zero_()

    @staticmethod
    def _pool_video(v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        if v.ndim == 3:
            v = v.mean(dim=1)
        elif v.ndim == 2:
            pass
        else:
            v = v.view(v.shape[0], -1)
        return v

    def forward(self, batch) -> torch.Tensor:
        eeg_logits = self.eeg_model(batch).view(-1, self.num_classes)
        v = self._pool_video(batch["video_feature"])
        video_logits = self.video_branch(v).view(-1, self.num_classes)
        combined = torch.cat([eeg_logits, video_logits], dim=1)
        return self.combiner(combined).view(-1, self.num_classes)
