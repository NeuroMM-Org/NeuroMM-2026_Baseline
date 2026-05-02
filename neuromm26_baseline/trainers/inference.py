"""Inference helpers for checkpointed baselines."""

from __future__ import annotations

import csv
from pathlib import Path

from .evaluator import Evaluator


class InferenceRunner:
    def __init__(self, device) -> None:
        self.evaluator = Evaluator(device=device)

    def run(self, model, dataloader, output_csv: str) -> str:
        result = self.evaluator.evaluate(model, dataloader, criterion=None)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "probability"])
            writer.writerows(zip(result.sample_ids, result.probabilities))
        return str(output_path)
