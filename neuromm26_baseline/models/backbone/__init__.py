"""Backbone encoders for each modality."""

from .eeg_resnet import EEGResNetEncoder
from .video_projector import VideoFeatureProjector

__all__ = ["EEGResNetEncoder", "VideoFeatureProjector"]
