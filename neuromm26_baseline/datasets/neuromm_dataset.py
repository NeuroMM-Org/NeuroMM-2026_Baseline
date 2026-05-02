"""Manifest-driven NeuroMM dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .base_dataset import BaseDataset


class NeuroMMDataset(BaseDataset):
    def __init__(
        self,
        dataset_root: str = "neuromm26_datasets",
        split: str = "train",
        eeg_feature_root: str | None = None,
        eeg_feature_name: str = "eeg",
        eeg_feature_ext: str = ".npy",
        load_eeg: bool = True,
        video_feature_root: str | None = None,
        load_video_feature: bool = False,
        video_feature_name: str | None = None,
        video_feature_ext: str = ".npy",
        require_video_feature: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.eeg_feature_root = (
            Path(eeg_feature_root)
            if eeg_feature_root
            else self.dataset_root / "processed" / "features" / eeg_feature_name
        )
        self.eeg_feature_ext = eeg_feature_ext
        self.load_eeg = load_eeg
        self.video_feature_root = (
            Path(video_feature_root)
            if video_feature_root
            else self.dataset_root / "processed" / "features"
        )
        self.load_video_feature = load_video_feature
        self.video_feature_name = video_feature_name
        self.video_feature_ext = video_feature_ext
        self.require_video_feature = require_video_feature
        manifest_path = self.dataset_root / "splits" / f"{split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.records = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_eeg_path(self, record: dict[str, str]) -> Path:
        return self.eeg_feature_root / f"{record['sample_id']}{self.eeg_feature_ext}"

    def _resolve_video_feature_path(self, record: dict[str, str]) -> Path | None:
        if not self.video_feature_name:
            return None
        return self.video_feature_root / self.video_feature_name / f"{record['sample_id']}{self.video_feature_ext}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "label": None if record.get("label", "") == "" else int(record["label"]),
            "raw_video_path": record.get("raw_video_relpath", ""),
            "subject_id": record.get("subject_id", ""),
            "eeg_source_path": record.get("eeg_source_relpath", ""),
        }

        eeg_path = self._resolve_eeg_path(record)
        sample["eeg_path"] = str(eeg_path)
        if self.load_eeg:
            sample["eeg"] = np.load(eeg_path)

        video_feature_path = self._resolve_video_feature_path(record)
        sample["video_feature_path"] = None if video_feature_path is None else str(video_feature_path)
        if video_feature_path is not None and self.load_video_feature:
            if video_feature_path.exists():
                sample["video_feature"] = np.load(video_feature_path)
            elif self.require_video_feature:
                raise FileNotFoundError(f"Missing video feature file: {video_feature_path}")

        return sample
