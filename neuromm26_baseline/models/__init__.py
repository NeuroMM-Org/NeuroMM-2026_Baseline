"""Top-level model package for NeuroMM 2026 baseline training and evaluation."""

from __future__ import annotations

from .eeg_baseline import EEGBaselineModel
from .legacy import LEGACY_EEG_MODEL_NAMES, build_legacy_eeg_model, is_legacy_eeg_model_name
from .multimodal_baseline import EEGVideoFusionModel

MODEL_NAME_ALIASES = {
    "baseline_eeg": "baseline/eeg",
    "baseline_resnet18": "baseline/resnet18",
    "baseline_resnet50": "baseline/resnet50",
    "eeg_video_fusion": "multimodal/eeg_video_fusion",
}
MODEL_NAME_ALIASES.update({name: f"legacy/{name}" for name in LEGACY_EEG_MODEL_NAMES})

SUPPORTED_MODEL_NAMES = (
    "baseline/eeg",
    "baseline/resnet18",
    "baseline/resnet50",
    "multimodal/eeg_video_fusion",
    *(f"legacy/{name}" for name in LEGACY_EEG_MODEL_NAMES),
)


def normalize_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return MODEL_NAME_ALIASES.get(model_name, model_name)


__all__ = [
    "EEGBaselineModel",
    "EEGVideoFusionModel",
    "LEGACY_EEG_MODEL_NAMES",
    "MODEL_NAME_ALIASES",
    "SUPPORTED_MODEL_NAMES",
    "build_legacy_eeg_model",
    "is_legacy_eeg_model_name",
    "normalize_model_name",
]
