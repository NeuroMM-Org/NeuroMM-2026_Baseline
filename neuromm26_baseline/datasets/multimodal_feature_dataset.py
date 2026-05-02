"""EEG + video feature dataset for multimodal baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .eeg_feature_dataset import EEGFeatureDataset


class NeuroMMMultimodalFeatureDataset(Dataset):
    def __init__(
        self,
        eeg_feature_root: str,
        video_feature_root: str,
        video_feature_name: str,
        split_csv: str | None = None,
        manifest_csv: str | None = None,
        split: str | None = None,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
        require_video_feature: bool = True,
    ) -> None:
        self.split_csv = Path(split_csv) if split_csv is not None else None
        self.manifest_csv = Path(manifest_csv) if manifest_csv is not None else None
        self.requested_split = split
        self.video_feature_root = Path(video_feature_root)
        self.video_feature_name = video_feature_name
        self.require_video_feature = require_video_feature
        self.eeg_dataset = EEGFeatureDataset(
            split_csv=split_csv,
            manifest_csv=manifest_csv,
            split=split,
            eeg_feature_root=eeg_feature_root,
            preload_in_memory=preload_in_memory,
            target_shape=target_shape,
        )
        self.preload_in_memory = preload_in_memory
        self.sample_ids = self.eeg_dataset.sample_ids
        self.labels = self.eeg_dataset.labels

        self._video_cache: list[torch.Tensor | None] | None = None
        if self.preload_in_memory:
            self._video_cache = []
            desc = self.requested_split or (self.split_csv.stem if self.split_csv is not None else "dataset")
            for sample_id in tqdm(self.sample_ids, desc=f"Preloading {video_feature_name} {desc}"):
                self._video_cache.append(self._load_video_tensor(sample_id))

    def __len__(self) -> int:
        return len(self.eeg_dataset)

    def _load_video_tensor(self, sample_id: str) -> torch.Tensor | None:
        path = self.video_feature_root / self.video_feature_name / f"{sample_id}.npy"
        if not path.exists():
            if self.require_video_feature:
                raise FileNotFoundError(f"Missing video feature: {path}")
            return None
        feature = np.load(path)
        return torch.from_numpy(feature.copy()).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.eeg_dataset[index]
        video_feature = self._video_cache[index] if self._video_cache is not None else self._load_video_tensor(self.sample_ids[index])
        sample["video_feature"] = video_feature
        return sample
