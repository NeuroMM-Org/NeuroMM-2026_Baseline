"""CLI entrypoint for inference and submission-style prediction export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neuromm26_baseline.trainers.inference import InferenceRunner
from neuromm26_baseline.utils.config import load_config
from neuromm26_baseline.utils.logger import get_logger

from .runtime import build_dataloader, build_dataset, build_model, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    experiment_name = config.get("experiment_name", "neuromm26_infer")
    logger = get_logger("neuromm26.infer", str(output_root / "logs" / f"{experiment_name}.log"))

    split = args.split or config["dataset"].get("test_split", "test")
    device = resolve_device(config["runtime"].get("device", "cuda"))
    dataset = build_dataset(config, split)
    dataloader = build_dataloader(config, dataset, batch_size=config["eval"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    output_path = args.output or str(output_root / "submissions" / f"{experiment_name}_{split}.csv")
    runner = InferenceRunner(device=device)
    saved_path = runner.run(model, dataloader, output_path)
    logger.info("inference_finished split=%s output=%s", split, saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
