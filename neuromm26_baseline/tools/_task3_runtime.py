"""Shared helpers for the three Task3 trainers (eeg / video_mlp / fusion).

Kept private (underscore prefix) — not a public CLI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuromm26_baseline.utils.multiclass_metrics import (
    compute_multiclass_metrics,
    order_multiclass_metrics,
)


def move_to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
    criterion: nn.Module | None = None,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_sids: list[str] = []
    for batch in loader:
        batch = move_to(batch, device)
        logits = model(batch).view(-1, num_classes)
        labels = batch.get("label")
        if criterion is not None and isinstance(labels, torch.Tensor):
            total_loss += float(criterion(logits, labels.long()).item())
            n_batches += 1
        all_logits.append(logits.cpu())
        if isinstance(labels, torch.Tensor):
            all_labels.append(labels.cpu())
        all_sids.extend(batch["sample_id"])
    logits = torch.cat(all_logits) if all_logits else torch.empty(0, num_classes)
    labels = torch.cat(all_labels) if all_labels else None
    if labels is None or labels.numel() == 0:
        metrics = {"num_samples": 0, "num_classes": num_classes}
    else:
        metrics = compute_multiclass_metrics(logits, labels, num_classes=num_classes)
    metrics = order_multiclass_metrics(metrics)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    return {
        "loss": (total_loss / n_batches) if n_batches else None,
        "metrics": metrics,
        "sample_ids": all_sids,
        "logits": logits.numpy(),
        "probabilities": probs,
        "predictions": preds,
        "labels": labels.numpy() if labels is not None else None,
    }


def save_predictions_csv(
    path: Path,
    sample_ids: list[str],
    probabilities,
    predictions,
    labels,
    num_classes: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "prediction"] + [f"prob_{c}" for c in range(num_classes)] + ["label"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, sid in enumerate(sample_ids):
            row = [sid, int(predictions[i])]
            row.extend(float(p) for p in probabilities[i].tolist())
            row.append(int(labels[i]) if labels is not None else "")
            w.writerow(row)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def class_balanced_weights(labels: list[int], num_classes: int = 5, eps: float = 1.0) -> torch.Tensor:
    """Inverse frequency weights for CrossEntropyLoss."""
    counts = [eps] * num_classes
    for lab in labels:
        counts[int(lab)] += 1
    total = sum(counts)
    weights = torch.tensor([total / (num_classes * c) for c in counts], dtype=torch.float32)
    return weights
