"""Task3 datasets: 5-class seizure-type localization.

Filters rows from the underlying manifests where label_type > 0 (i.e. the
positive seizure samples) and returns the class index 0..4 as a torch.long
label suitable for CrossEntropyLoss.

Three classes:
- Task3EEGFeatureDataset      EEG-only
- Task3MultimodalFeatureDataset  EEG + video feature
- Task3VideoFeatureDataset    Video feature only (mean-pooled, for video MLP)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _is_positive_row(row: dict) -> bool:
    v = (row.get("label_type", "") or "").strip()
    if v == "" or v == "0":
        return False
    try:
        return int(v) > 0
    except ValueError:
        return False


def _class_index(row: dict) -> int:
    """0-indexed class (0..4) for label_type 1..5."""
    return int(row["label_type"]) - 1


def _load_filtered(manifest_csv: str, split: str) -> list[dict]:
    with Path(manifest_csv).open("r", encoding="utf-8-sig") as h:
        rows = [r for r in csv.DictReader(h) if r.get("split") == split]
    rows = [r for r in rows if _is_positive_row(r)]
    if not rows:
        raise ValueError(f"No positive samples for split={split} in {manifest_csv}")
    return rows


class Task3EEGFeatureDataset(Dataset):
    """EEG-only multi-class dataset. Reuses the EEG normalization scheme
    from EEGFeatureDataset (29-channel raw -> 26-channel after ECG/EMG
    derivation, padded/cropped to (26, 2000))."""

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        eeg_feature_root: str,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
    ) -> None:
        self.records = _load_filtered(manifest_csv, split)
        self.eeg_feature_root = Path(eeg_feature_root)
        self.preload_in_memory = preload_in_memory
        self.target_shape = target_shape
        self.labels = [_class_index(r) for r in self.records]
        self.sample_ids = [r["sample_id"] for r in self.records]

        self._cache: list[torch.Tensor] | None = None
        if preload_in_memory:
            self._cache = []
            desc = f"Preloading EEG {split} (task3)"
            for sid in tqdm(self.sample_ids, desc=desc):
                self._cache.append(self._load_eeg_tensor(sid))

    @staticmethod
    def _normalize(wave: np.ndarray) -> np.ndarray:
        wave = wave.astype(np.float32, copy=True)
        if wave.shape[0] >= 29:
            wave[:23, ...] = wave[:23, ...] / 1e-3
            wave[23:, ...] = wave[23:, ...] * 1e-2
            heart_wave = wave[23, :] - wave[24, :]
            muscle_wave1 = wave[25, :] - wave[26, :]
            muscle_wave2 = wave[27, :] - wave[28, :]
            heart_muscle = np.stack([heart_wave, muscle_wave1, muscle_wave2], axis=0)
            wave = np.concatenate([wave[:23, ...], heart_muscle], axis=0)
        return wave

    def _pad_or_crop(self, wave: np.ndarray) -> np.ndarray:
        c, t = self.target_shape
        padded = np.zeros((c, t), dtype=np.float32)
        ch, ts = wave.shape
        padded[: min(ch, c), : min(ts, t)] = wave[: min(ch, c), : min(ts, t)]
        return padded

    def _load_eeg_tensor(self, sid: str) -> torch.Tensor:
        path = self.eeg_feature_root / f"{sid}.npy"
        wave = np.load(path)
        wave = self._normalize(wave)
        wave = self._pad_or_crop(wave)
        return torch.from_numpy(wave.copy()).float()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        eeg = self._cache[idx] if self._cache is not None else self._load_eeg_tensor(self.sample_ids[idx])
        return {
            "sample_id": self.sample_ids[idx],
            "eeg": eeg,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class Task3MultimodalFeatureDataset(Dataset):
    """EEG + Video for 5-class fusion training."""

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        eeg_feature_root: str,
        video_feature_root: str,
        video_feature_name: str,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
        require_video_feature: bool = True,
    ) -> None:
        self.eeg_dataset = Task3EEGFeatureDataset(
            manifest_csv=manifest_csv,
            split=split,
            eeg_feature_root=eeg_feature_root,
            preload_in_memory=preload_in_memory,
            target_shape=target_shape,
        )
        self.sample_ids = self.eeg_dataset.sample_ids
        self.labels = self.eeg_dataset.labels
        self.video_feature_root = Path(video_feature_root)
        self.video_feature_name = video_feature_name
        self.require_video_feature = require_video_feature

        self._video_cache: list[torch.Tensor | None] | None = None
        if preload_in_memory:
            self._video_cache = []
            desc = f"Preloading {video_feature_name} {split} (task3)"
            for sid in tqdm(self.sample_ids, desc=desc):
                self._video_cache.append(self._load_video_tensor(sid))

    def _load_video_tensor(self, sid: str) -> torch.Tensor | None:
        path = self.video_feature_root / self.video_feature_name / f"{sid}.npy"
        if not path.exists():
            if self.require_video_feature:
                raise FileNotFoundError(f"Missing video feature: {path}")
            return None
        return torch.from_numpy(np.load(path).copy()).float()

    def __len__(self) -> int:
        return len(self.eeg_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.eeg_dataset[idx]
        v = self._video_cache[idx] if self._video_cache is not None else self._load_video_tensor(self.sample_ids[idx])
        sample["video_feature"] = v
        return sample


class Task3VideoFeatureDataset(Dataset):
    """Video-only multi-class dataset. Mean-pools (T,D) features to (D,)
    in __init__ so the runtime forward is a plain MLP."""

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        feature_root: str,
        pool: str = "mean",
    ) -> None:
        rows = _load_filtered(manifest_csv, split)
        feature_root = Path(feature_root)
        feats: list[torch.Tensor] = []
        labels: list[int] = []
        sids: list[str] = []
        skipped = 0
        for r in rows:
            sid = r["sample_id"]
            feat_path = feature_root / f"{sid}.npy"
            if not feat_path.exists():
                skipped += 1
                continue
            arr = np.load(feat_path).astype(np.float32)
            if arr.ndim == 1:
                pooled = arr
            elif pool == "mean":
                pooled = arr.mean(axis=0)
            elif pool == "max":
                pooled = arr.max(axis=0)
            else:
                raise ValueError(f"Unknown pool: {pool}")
            feats.append(torch.from_numpy(pooled))
            labels.append(_class_index(r))
            sids.append(sid)
        if not feats:
            raise ValueError(f"No samples for split={split} under {feature_root}")
        self.features = torch.stack(feats)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_ids = sids
        self.feature_dim = self.features.shape[1]
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "feature": self.features[idx],
            "label": self.labels[idx],
            "sample_id": self.sample_ids[idx],
        }
