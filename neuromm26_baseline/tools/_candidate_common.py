"""Shared helpers for candidate-set inference (predict_candidate_test{1,2,3}).

The competition candidate set is delivered separately with this layout:

    <candidate-dir>/
    ├── candidate_ids.txt          one opaque id per line (20,000 ids)
    ├── eeg/<id>.npy               (29, 2000) float32
    └── video/<backbone>/<id>.npy  per-backbone visual features

Participants run a trained checkpoint over EVERY id and submit one CSV; the
organizers' Codabench scorer keeps only the hidden official ids for that test
and ignores all the rest.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import torch


def resolve_device(name: str = "cuda") -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def read_candidate_ids(candidate_dir: str) -> list[str]:
    p = Path(candidate_dir) / "candidate_ids.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Point --candidate-dir at the extracted candidate "
            f"set (must contain candidate_ids.txt, eeg/, video/).")
    ids = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    if not ids:
        raise ValueError(f"{p} is empty")
    return ids


def build_candidate_manifest(candidate_dir: str, out_csv: str) -> int:
    """Write a manifest the project datasets understand.

    Every row gets split='candidate', label=0 and label_type=1. The dummy
    label_type=1 keeps the Task-3 datasets (which filter label_type>0) from
    dropping rows; labels are unused at inference (model.eval()/no_grad).
    """
    ids = read_candidate_ids(candidate_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "split", "label", "label_type",
                    "raw_video_relpath", "subject_id", "eeg_source_relpath"])
        for i in ids:
            w.writerow([i, "candidate", 0, 1, "", "", ""])
    return len(ids)


def load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return torch.load(path, map_location=device)


def write_submission_csv(out_csv: str, sample_ids, predictions,
                         float_fmt: str | None = "%.8f") -> None:
    """Write the Codabench submission CSV: columns sample_id,prediction.

    float_fmt None  -> write predictions as-is (e.g. integer class labels).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "prediction"])
        for sid, p in zip(sample_ids, predictions):
            w.writerow([sid, (float_fmt % float(p)) if float_fmt else p])
    n = len(sample_ids)
    print(f"[ok] wrote {n} rows -> {out_csv}")
    if n != 20000:
        print(f"[warn] expected 20000 candidate rows, got {n}; submit ALL "
              f"candidate ids (non-scored ids are ignored by the scorer).")
