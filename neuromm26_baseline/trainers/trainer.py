"""Training loop for binary classification baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.utils.metrics import PRIMARY_SELECTION_METRIC

from .evaluator import Evaluator


@dataclass
class TrainResult:
    best_metric_name: str
    best_metric_value: float
    best_metrics: dict[str, Any]
    best_checkpoint_path: str | None


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: torch.device,
        logger,
        checkpoint_dir: str | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.max_grad_norm = max_grad_norm
        self.evaluator = Evaluator(device=device)
        self.primary_metric_name = PRIMARY_SELECTION_METRIC

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def train_epoch(self, dataloader: DataLoader, epoch_index: int, log_interval: int = 20) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for step, batch in enumerate(dataloader, start=1):
            batch = self._move_batch(batch)
            labels = batch["label"].view(-1).float()

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch).view(-1)
            loss = self.criterion(logits, labels)
            loss.backward()

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

            if log_interval and step % log_interval == 0:
                self.logger.info(
                    "epoch=%s step=%s/%s train_loss=%.6f",
                    epoch_index,
                    step,
                    len(dataloader),
                    total_loss / total_batches,
                )

        return total_loss / max(total_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_name: str = "best.pt",
        log_interval: int = 20,
    ) -> TrainResult:
        best_metric_value = float("-inf")
        best_metrics: dict[str, Any] = {}
        best_checkpoint_path = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch_index=epoch, log_interval=log_interval)
            val_result = self.evaluator.evaluate(self.model, val_loader, criterion=self.criterion)
            score = val_result.metrics.get(self.primary_metric_name, float("-inf")) if val_result.metrics else float("-inf")
            self.logger.info(
                "epoch=%s train_loss=%.6f val_loss=%s val_metrics=%s",
                epoch,
                train_loss,
                "None" if val_result.loss is None else f"{val_result.loss:.6f}",
                val_result.metrics,
            )

            if score > best_metric_value:
                best_metric_value = score
                best_metrics = dict(val_result.metrics)
                if self.checkpoint_dir is not None:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = self.checkpoint_dir / checkpoint_name
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_metric_name": self.primary_metric_name,
                            "best_metric_value": best_metric_value,
                            "best_metrics": best_metrics,
                            "epoch": epoch,
                        },
                        checkpoint_path,
                    )
                    best_checkpoint_path = str(checkpoint_path)
                    self.logger.info(
                        "saved_checkpoint=%s best_%s=%.6f metrics=%s",
                        checkpoint_path,
                        self.primary_metric_name,
                        best_metric_value,
                        best_metrics,
                    )

        return TrainResult(
            best_metric_name=self.primary_metric_name,
            best_metric_value=best_metric_value,
            best_metrics=best_metrics,
            best_checkpoint_path=best_checkpoint_path,
        )
