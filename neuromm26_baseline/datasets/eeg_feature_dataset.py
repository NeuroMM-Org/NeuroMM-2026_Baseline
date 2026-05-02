"""Single-modal EEG feature dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class EEGFeatureDataset(Dataset):
    def __init__(
        self,
        eeg_feature_root: str,
        split_csv: str | None = None,
        manifest_csv: str | None = None,
        split: str | None = None,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
    ) -> None:
        self.split_csv = Path(split_csv) if split_csv is not None else None
        self.manifest_csv = Path(manifest_csv) if manifest_csv is not None else None
        self.requested_split = split
        self.eeg_feature_root = Path(eeg_feature_root)
        self.preload_in_memory = preload_in_memory
        self.target_shape = target_shape
        self.records = self._load_records()

        self.labels = [self._parse_label(record) for record in self.records]
        self.sample_ids = [record["sample_id"] for record in self.records]
        self._cache: list[torch.Tensor] | None = None

        if self.preload_in_memory:
            self._cache = []
            desc = self.requested_split or (self.split_csv.stem if self.split_csv is not None else "dataset")
            for sample_id in tqdm(self.sample_ids, desc=f"Preloading EEG {desc}"):
                self._cache.append(self._load_eeg_tensor(sample_id))

    def __len__(self) -> int:
        return len(self.records)

    def _load_records(self) -> list[dict[str, str]]:
        if self.manifest_csv is not None:
            with self.manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if self.requested_split is not None:
                rows = [row for row in rows if row.get("split") == self.requested_split]
            return rows

        if self.split_csv is None:
            raise ValueError("Either split_csv or manifest_csv must be provided.")

        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _parse_label(self, record: dict[str, str]) -> int | None:
        value = record.get("label", "")
        return None if value == "" else int(value)

    def _normalize(self, wave: np.ndarray) -> np.ndarray:
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
        channels, timesteps = self.target_shape
        padded = np.zeros((channels, timesteps), dtype=np.float32)
        c, t = wave.shape
        padded[: min(c, channels), : min(t, timesteps)] = wave[: min(c, channels), : min(t, timesteps)]
        return padded

    def _load_eeg_tensor(self, sample_id: str) -> torch.Tensor:
        path = self.eeg_feature_root / f"{sample_id}.npy"
        wave = np.load(path)
        wave = self._normalize(wave)
        wave = self._pad_or_crop(wave)
        return torch.from_numpy(wave.copy()).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        eeg = self._cache[index] if self._cache is not None else self._load_eeg_tensor(self.sample_ids[index])
        label = self.labels[index]
        sample = {
            "sample_id": self.sample_ids[index],
            "eeg": eeg,
            "label": torch.tensor(label, dtype=torch.float32) if label is not None else None,
        }
        return sample
