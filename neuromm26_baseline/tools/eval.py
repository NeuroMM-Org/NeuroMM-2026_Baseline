"""CLI entrypoint for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neuromm26_baseline.trainers.evaluator import Evaluator
from neuromm26_baseline.utils.config import load_config
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import format_metric_summary, order_metric_dict

from .runtime import build_criterion, build_dataloader, build_dataset, build_model, resolve_device, save_json, save_prediction_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    experiment_name = config.get("experiment_name", "neuromm26_eval")
    logger = get_logger("neuromm26.eval", str(output_root / "logs" / f"{experiment_name}.log"))

    split = args.split or config["eval"].get("split", config["dataset"]["val_split"])
    device = resolve_device(config["runtime"].get("device", "cuda"))
    dataset = build_dataset(config, split)
    dataloader = build_dataloader(config, dataset, batch_size=config["eval"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    checkpoint_path = args.checkpoint or config["eval"].get("checkpoint_path")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        logger.info("loaded_checkpoint=%s", checkpoint_path)

    criterion = build_criterion(config, labels=getattr(dataset, "labels", None)).to(device)
    evaluator = Evaluator(device=device)
    result = evaluator.evaluate(model, dataloader, criterion=criterion)
    ordered_metrics = order_metric_dict(result.metrics)

    metrics_payload = {
        "experiment_name": experiment_name,
        "split": split,
        "checkpoint_path": checkpoint_path,
        "loss": result.loss,
        "metrics": ordered_metrics,
    }
    metrics_path = output_root / "metrics" / f"{experiment_name}_{split}.json"
    save_json(metrics_payload, str(metrics_path))

    if config["runtime"].get("save_predictions", True):
        prediction_path = output_root / "predictions" / f"{experiment_name}_{split}.csv"
        save_prediction_csv(result.sample_ids, result.probabilities, result.labels, str(prediction_path))

    logger.info("evaluation_finished split=%s %s", split, format_metric_summary(ordered_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
