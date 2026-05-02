"""Task3 video-only MLP 5-class trainer (positive samples only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets.task3_datasets import Task3VideoFeatureDataset
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.multiclass_metrics import format_multiclass_summary
from neuromm26_baseline.utils.seed import set_seed

from ._task3_runtime import (
    class_balanced_weights,
    evaluate_multiclass,
    move_to,
    save_json,
    save_predictions_csv,
)


def _collate(batch):
    return {
        "feature": torch.stack([b["feature"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }


class MLP5(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        return self.net(batch["feature"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--feature-root", default=None,
                        help="Defaults to neuromm26_datasets/processed/features/video/<feature-name>")
    parser.add_argument("--train-val-manifest",
                        default="neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv")
    parser.add_argument("--test-manifest",
                        default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--output-root", default="neuromm26_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(int(args.seed))

    feature_root = (
        args.feature_root
        or f"neuromm26_datasets/processed/features/video/{args.feature_name}"
    )
    feature_safe = args.feature_name.replace("/", "-")
    exp_name = f"task3_video_mlp__{feature_safe}__seed{args.seed}__lr{args.learning_rate}"
    model_name_stamp = f"task3_video_mlp/{feature_safe}"

    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    metrics_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.task3_video_mlp", str(log_path))
    logger.info("experiment=%s feature=%s seed=%d", exp_name, args.feature_name, args.seed)

    device = torch.device(args.device)

    train_ds = Task3VideoFeatureDataset(args.train_val_manifest, "train", feature_root)
    val_ds = Task3VideoFeatureDataset(args.train_val_manifest, "val", feature_root)
    test_ds = None
    test_loader = None
    if args.test_manifest and Path(args.test_manifest).exists() and args.test_split:
        test_ds = Task3VideoFeatureDataset(args.test_manifest, args.test_split, feature_root)
        logger.info(
            "train=%d val=%d test=%d feat_dim=%d",
            len(train_ds), len(val_ds), len(test_ds), train_ds.feature_dim,
        )
    else:
        logger.info(
            "train=%d val=%d test=- feat_dim=%d (no test manifest)",
            len(train_ds), len(val_ds), train_ds.feature_dim,
        )

    g = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=_collate)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=_collate)

    model = MLP5(train_ds.feature_dim, args.hidden_dim, args.dropout, args.num_classes).to(device)

    if args.use_class_weights:
        cw = class_balanced_weights(train_ds.labels.tolist(), args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_metric = float("-inf")
    best_record = None
    epoch_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = move_to(batch, device)
            labels = batch["label"].long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch).view(-1, args.num_classes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_res = evaluate_multiclass(model, val_loader, device,
                                      num_classes=args.num_classes, criterion=criterion)
        val_metrics = val_res["metrics"]
        score = float(val_metrics.get("weighted_f1", float("-inf")))
        epoch_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_res["loss"],
            "val_weighted_f1": score,
            "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        })
        logger.info("epoch=%d train_loss=%.4f %s",
                    epoch, train_loss, format_multiclass_summary(val_metrics))
        if score > best_metric:
            best_metric = score
            best_record = {"epoch": epoch, "metrics": dict(val_metrics)}
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric_name": "weighted_f1",
                "best_metric_value": score,
                "best_metrics": dict(val_metrics),
                "feature_name": args.feature_name,
                "feature_dim": train_ds.feature_dim,
                "hidden_dim": args.hidden_dim,
                "num_classes": args.num_classes,
            }, ckpt_dir / "best.pt")

    if best_record is None:
        raise RuntimeError("Training did not produce any valid metric.")

    save_json({
        "experiment_name": exp_name,
        "task_type": "task3_video_mlp",
        "model_name": model_name_stamp,
        "feature_name": args.feature_name,
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "num_classes": args.num_classes,
        "best_metric_name": "weighted_f1",
        "best_metric_value": best_metric,
        "best_metrics": best_record["metrics"],
        "best_epoch": best_record["epoch"],
        "best_checkpoint_path": str(ckpt_dir / "best.pt"),
        "epoch_log": epoch_log,
    }, metrics_dir / f"{exp_name}_train_summary.json")

    if test_ds is not None and test_loader is not None:
        state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        test_res = evaluate_multiclass(model, test_loader, device, num_classes=args.num_classes)
        save_json({
            "experiment_name": exp_name,
            "task_type": "task3_video_mlp",
            "model_name": model_name_stamp,
            "feature_name": args.feature_name,
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "task": "task3",
            "split": args.test_split,
            "manifest": args.test_manifest,
            "checkpoint_path": str(ckpt_dir / "best.pt"),
            "num_samples": len(test_ds),
            "metrics": dict(test_res["metrics"]),
        }, metrics_dir / f"{exp_name}_{args.test_split}.json")
        save_predictions_csv(
            pred_dir / f"{exp_name}_{args.test_split}.csv",
            test_res["sample_ids"], test_res["probabilities"], test_res["predictions"],
            test_res["labels"], num_classes=args.num_classes,
        )
        logger.info(
            "DONE val_weighted_f1=%.4f@epoch%d  test_weighted_f1=%.4f  test_macro_f1=%.4f",
            best_metric, best_record["epoch"],
            float(test_res["metrics"].get("weighted_f1", 0.0)),
            float(test_res["metrics"].get("macro_f1", 0.0)),
        )
    else:
        logger.info(
            "DONE val_weighted_f1=%.4f@epoch%d  (no test eval)",
            best_metric, best_record["epoch"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
