"""Evaluation loop for binary classification baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics


@dataclass
class EvaluationResult:
    loss: float | None
    metrics: dict[str, Any]
    sample_ids: list[str]
    probabilities: list[float]
    labels: list[float] | None


class Evaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def evaluate(self, model, dataloader: DataLoader, criterion=None) -> EvaluationResult:
        model.eval()
        total_loss = 0.0
        total_batches = 0
        all_logits = []
        all_labels = []
        all_sample_ids: list[str] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch)
                logits = model(batch).view(-1)
                labels = batch.get("label")
                if isinstance(labels, torch.Tensor):
                    labels = labels.view(-1).float()
                    all_labels.append(labels.cpu())
                    if criterion is not None:
                        total_loss += float(criterion(logits, labels).item())
                        total_batches += 1
                all_logits.append(logits.cpu())
                all_sample_ids.extend(batch["sample_id"])

        logits_tensor = torch.cat(all_logits) if all_logits else torch.empty(0)
        labels_tensor = torch.cat(all_labels) if all_labels else None
        probabilities = torch.sigmoid(logits_tensor).tolist()
        metrics = {}
        if labels_tensor is not None and labels_tensor.numel() > 0:
            metrics = compute_binary_classification_metrics(logits_tensor, labels_tensor)

        return EvaluationResult(
            loss=(total_loss / total_batches) if total_batches else None,
            metrics=metrics,
            sample_ids=all_sample_ids,
            probabilities=probabilities,
            labels=None if labels_tensor is None else labels_tensor.tolist(),
        )
