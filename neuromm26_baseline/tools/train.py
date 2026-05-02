"""CLI entrypoint for training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from neuromm26_baseline.models import SUPPORTED_MODEL_NAMES, normalize_model_name
from neuromm26_baseline.trainers.trainer import Trainer
from neuromm26_baseline.utils.config import apply_overrides, load_config, parse_override_strings
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import format_metric_summary
from neuromm26_baseline.utils.seed import set_seed

from .runtime import build_criterion, build_dataloader, build_dataset, build_model, resolve_device, save_json


def _infer_task_type(model_name: str | None) -> str | None:
    model_name = normalize_model_name(model_name)
    if not model_name:
        return None
    if model_name.startswith("multimodal/"):
        return "eeg_video_fusion"
    return "eeg_only"


def _sanitize_token(value: Any) -> str:
    token = str(value).strip().replace("/", "-")
    return token.replace(" ", "_")


def _build_experiment_name(base_name: str, args: argparse.Namespace) -> str:
    tokens = [base_name]
    if args.model_name:
        tokens.append(_sanitize_token(normalize_model_name(args.model_name)))
    if args.seed is not None:
        tokens.append(f"seed{args.seed}")
    if args.learning_rate is not None:
        tokens.append(f"lr{_sanitize_token(args.learning_rate)}")
    return "__".join(tokens)


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.model_name is not None:
        overrides["model.name"] = normalize_model_name(args.model_name)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.learning_rate is not None:
        overrides["train.learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        overrides["eval.batch_size"] = args.eval_batch_size
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.task_type is not None:
        overrides["model.task_type"] = args.task_type
    if args.device is not None:
        overrides["runtime.device"] = args.device
    if args.set:
        overrides.update(parse_override_strings(args.set))
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--task-type", choices=["eeg_only", "eeg_video_fusion"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with dotted keys, e.g. --set model.eeg_dropout=0.3",
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(SUPPORTED_MODEL_NAMES))
        return 0

    config = load_config(args.config)
    cli_overrides = _collect_overrides(args)
    apply_overrides(config, cli_overrides)

    if "model.name" in cli_overrides and "model.task_type" not in cli_overrides:
        inferred_task_type = _infer_task_type(str(cli_overrides["model.name"]))
        if inferred_task_type is not None:
            config.setdefault("model", {})["task_type"] = inferred_task_type

    if args.experiment_name is None and any(
        value is not None for value in (args.model_name, args.seed, args.learning_rate)
    ):
        config["experiment_name"] = _build_experiment_name(
            config.get("experiment_name", "neuromm26_train"),
            args,
        )

    if config.get("model", {}).get("name") is not None:
        config["model"]["name"] = normalize_model_name(config["model"]["name"])

    set_seed(int(config.get("seed", 42)))

    output_root = Path(config["paths"]["output_root"])
    experiment_name = config.get("experiment_name", "neuromm26_train")
    log_path = output_root / "logs" / f"{experiment_name}.log"
    logger = get_logger("neuromm26.train", str(log_path))
    logger.info("loaded_config=%s", args.config)
    logger.info("task_type=%s", config["model"]["task_type"])
    logger.info("model_name=%s", config["model"].get("name", "baseline/eeg"))
    logger.info("seed=%s", config.get("seed", 42))
    if cli_overrides:
        logger.info("cli_overrides=%s", cli_overrides)

    device = resolve_device(config["runtime"].get("device", "cuda"))
    logger.info("device=%s", device)

    train_dataset = build_dataset(config, config["dataset"]["train_split"])
    val_dataset = build_dataset(config, config["dataset"]["val_split"])
    train_loader = build_dataloader(config, train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = build_dataloader(config, val_dataset, batch_size=config["eval"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    criterion = build_criterion(config, labels=getattr(train_dataset, "labels", None)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )

    checkpoint_dir = ensure_dir(output_root / "checkpoints" / experiment_name)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        checkpoint_dir=str(checkpoint_dir) if config["runtime"].get("save_checkpoints", True) else None,
        max_grad_norm=config["train"].get("max_grad_norm"),
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["train"]["epochs"],
        checkpoint_name=config["train"].get("checkpoint_name", "best.pt"),
        log_interval=config["train"].get("log_interval", 20),
    )

    resolved_config_path = output_root / "metrics" / f"{experiment_name}_config.json"
    save_json(config, str(resolved_config_path))

    metrics_path = output_root / "metrics" / f"{experiment_name}_train_summary.json"
    save_json(
        {
            "experiment_name": experiment_name,
            "task_type": config["model"]["task_type"],
            "model_name": config["model"].get("name", "baseline/eeg"),
            "seed": config.get("seed", 42),
            "learning_rate": config["train"]["learning_rate"],
            "best_metric_name": result.best_metric_name,
            "best_metric_value": result.best_metric_value,
            "best_metrics": result.best_metrics,
            "best_checkpoint_path": result.best_checkpoint_path,
            "config_path": str(Path(args.config).resolve()),
            "resolved_config_path": str(resolved_config_path.resolve()),
            "cli_overrides": cli_overrides,
        },
        str(metrics_path),
    )
    logger.info(
        "training_finished best_%s=%.6f %s best_checkpoint=%s",
        result.best_metric_name,
        result.best_metric_value,
        format_metric_summary(result.best_metrics),
        result.best_checkpoint_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
