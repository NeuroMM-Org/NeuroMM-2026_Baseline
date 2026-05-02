"""Late-fusion multimodal baseline: legacy EEG model logit + video MLP logit.

This is a separate model class so the existing EEGVideoFusionModel
(`models/multimodal_baseline.py`) remains untouched and any code that depends
on it keeps working.

Architecture:

    EEG batch ----[legacy EEG model (full, returns 1 logit)]---> eeg_logit
                                                                  \\
                                                                   --> [Linear(2,1)] --> final logit
                                                                  /
    video feature (T,D) -[mean-pool over T]-[MLP D->H->1]-> video_logit

Training is end-to-end. The combiner is initialized to mean(eeg, video) so
the model starts as average-of-experts and can adapt the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .legacy.registry import build_legacy_eeg_model


class EEGVideoLateFusion(nn.Module):
    def __init__(
        self,
        eeg_model_name: str,
        video_feature_dim: int,
        video_hidden_dim: int = 256,
        video_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Full legacy EEG model that returns (B, 1) or (B,) logits.
        self.eeg_model = build_legacy_eeg_model(eeg_model_name)
        self.video_branch = nn.Sequential(
            nn.Linear(video_feature_dim, video_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(video_dropout),
            nn.Linear(video_hidden_dim, 1),
        )
        # 2-input combiner. Init to (0.5, 0.5) bias 0 so initial behaviour is mean.
        self.combiner = nn.Linear(2, 1)
        with torch.no_grad():
            self.combiner.weight.fill_(0.5)
            self.combiner.bias.fill_(0.0)

    @staticmethod
    def _pool_video(video_feature: torch.Tensor) -> torch.Tensor:
        v = video_feature.float()
        if v.ndim == 3:        # (B, T, D)
            v = v.mean(dim=1)
        elif v.ndim == 2:      # (B, D)
            pass
        else:                  # fallback flatten
            v = v.view(v.shape[0], -1)
        return v

    def forward(self, batch) -> torch.Tensor:
        eeg_logit = self.eeg_model(batch).view(-1, 1)
        v = self._pool_video(batch["video_feature"])
        video_logit = self.video_branch(v).view(-1, 1)
        combined = torch.cat([eeg_logit, video_logit], dim=1)  # (B, 2)
        return self.combiner(combined).view(-1)

    def get_branch_logits(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagnostic: return (eeg_logit, video_logit) per sample."""
        eeg_logit = self.eeg_model(batch).view(-1)
        v = self._pool_video(batch["video_feature"])
        video_logit = self.video_branch(v).view(-1)
        return eeg_logit, video_logit
