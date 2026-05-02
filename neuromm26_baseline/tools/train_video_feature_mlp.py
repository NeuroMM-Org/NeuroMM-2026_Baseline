"""Single-modality video-feature MLP baseline.

Loads precomputed video features from
    neuromm26_datasets/processed/features/video/{feature_name}/{sample_id}.npy
mean-pools across the temporal dimension, trains a 2-layer MLP to predict the
binary spike label, and after each epoch:
  * evaluates on val (split=val from neuromm2026_train_val_patient_split.csv)
  * keeps best.pt by val AUPRC

When training finishes the val-best checkpoint is reloaded and used to score
the test set (default: test_task2). Two JSON summaries land under
neuromm26_results/metrics/, named so that the existing aggregate_results.py
and aggregate_test_results.py pick them up automatically.

Run example:
    python -m neuromm26_baseline.tools.train_video_feature_mlp \\
        --feature-name clip-base --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import (
    compute_binary_classification_metrics,
    format_metric_summary,
    order_metric_dict,
)
from neuromm26_baseline.utils.seed import set_seed


class VideoFeatureDataset(Dataset):
    """Loads a (T, D) or (D,) feature per sample; mean-pools to (D,) at init."""

    def __init__(self, manifest_csv: str, split: str, feature_root: str, pool: str = "mean"):
        with open(manifest_csv, encoding="utf-8-sig") as h:
            rows = [r for r in csv.DictReader(h) if r["split"] == split]
        feature_root = Path(feature_root)

        feats: list[torch.Tensor] = []
        labels: list[float] = []
        sids: list[str] = []
        skipped = 0
        for r in rows:
            sid = r["sample_id"]
            label = r.get("label", "").strip()
            if not label:
                skipped += 1
                continue
            feat_path = feature_root / f"{sid}.npy"
            if not feat_path.exists():
                skipped += 1
                continue
            arr = np.load(feat_path).astype(np.float32)
            if arr.ndim == 1:
                pooled = arr
            elif pool == "mean":
                pooled = arr.mean(axis=0)
            elif pool == "max":
                pooled = arr.max(axis=0)
            else:
                raise ValueError(f"Unknown pool: {pool}")
            feats.append(torch.from_numpy(pooled))
            labels.append(float(int(label)))
            sids.append(sid)

        if not feats:
            raise ValueError(f"No samples for split={split} under {feature_root}")
        self.features = torch.stack(feats)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.sample_ids = sids
        self.feature_dim = self.features.shape[1]
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "feature": self.features[idx],
            "label": self.labels[idx],
            "sample_id": self.sample_ids[idx],
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "feature": torch.stack([b["feature"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels, all_sids = [], [], []
    for batch in loader:
        x = batch["feature"].to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].view(-1).cpu())
        all_sids.extend(batch["sample_id"])
    logits = torch.cat(all_logits) if all_logits else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    metrics = order_metric_dict(compute_binary_classification_metrics(logits, labels))
    probs = torch.sigmoid(logits).tolist()
    return metrics, all_sids, probs, labels.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-name", required=True,
                        help="Feature subdirectory under processed/features/video/, e.g. clip-base, videomae-base")
    parser.add_argument("--feature-root", default=None,
                        help="Override feature dir. Defaults to neuromm26_datasets/processed/features/video/<feature-name>")
    parser.add_argument("--train-val-manifest",
                        default="neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv")
    parser.add_argument("--test-manifest",
                        default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--output-root", default="neuromm26_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    feature_root = (
        args.feature_root
        or f"neuromm26_datasets/processed/features/video/{args.feature_name}"
    )
    feature_safe = args.feature_name.replace("/", "-")
    exp_name = f"video_mlp__{feature_safe}__seed{args.seed}__lr{args.learning_rate}"

    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    metrics_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.video_mlp", str(log_path))
    logger.info("experiment=%s feature=%s seed=%d lr=%s", exp_name, args.feature_name, args.seed, args.learning_rate)

    device = torch.device(args.device)
    logger.info("device=%s feature_root=%s", device, feature_root)

    train_ds = VideoFeatureDataset(args.train_val_manifest, "train", feature_root)
    val_ds = VideoFeatureDataset(args.train_val_manifest, "val", feature_root)
    test_ds = None
    test_loader = None
    if args.test_manifest and Path(args.test_manifest).exists() and args.test_split:
        test_ds = VideoFeatureDataset(args.test_manifest, args.test_split, feature_root)
        logger.info(
            "datasets train=%d val=%d test=%d (skipped train=%d val=%d test=%d) feat_dim=%d",
            len(train_ds), len(val_ds), len(test_ds),
            train_ds.skipped, val_ds.skipped, test_ds.skipped,
            train_ds.feature_dim,
        )
    else:
        logger.info(
            "datasets train=%d val=%d test=- (skipped train=%d val=%d) feat_dim=%d (no test manifest)",
            len(train_ds), len(val_ds),
            train_ds.skipped, val_ds.skipped,
            train_ds.feature_dim,
        )

    g = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=_collate)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=_collate)

    model = MLPHead(train_ds.feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    pos = float((train_ds.labels == 1).sum())
    neg = float((train_ds.labels == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_auprc = float("-inf")
    best_val_record: dict[str, Any] | None = None
    epoch_log: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["feature"].to(device)
            y = batch["label"].to(device).view(-1)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_metrics, *_ = _evaluate(model, val_loader, device)
        val_auprc = float(val_metrics["auprc"])
        epoch_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auprc": val_auprc,
            "val_binary_f1": float(val_metrics["binary_f1"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        })
        logger.info("epoch=%d train_loss=%.4f %s", epoch, train_loss, format_metric_summary(val_metrics))

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_record = {"epoch": epoch, "metrics": dict(val_metrics)}
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": dict(val_metrics),
                "feature_dim": train_ds.feature_dim,
                "hidden_dim": args.hidden_dim,
            }, ckpt_dir / "best.pt")

    if best_val_record is None:
        raise RuntimeError("Training produced no valid val metric")

    model_name = f"video_mlp/{feature_safe}"
    train_summary = {
        "experiment_name": exp_name,
        "feature_name": args.feature_name,
        "model_name": model_name,
        "task_type": "video_only",
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "best_metric_name": "auprc",
        "best_metric_value": best_val_auprc,
        "best_metrics": best_val_record["metrics"],
        "best_checkpoint_path": str(ckpt_dir / "best.pt"),
        "best_epoch": best_val_record["epoch"],
        "epoch_log": epoch_log,
    }
    train_summary_path = metrics_dir / f"{exp_name}_train_summary.json"
    train_summary_path.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False))

    if test_ds is not None and test_loader is not None:
        # Reload best-val checkpoint, evaluate on test
        state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        test_metrics, sids, probs, labels = _evaluate(model, test_loader, device)
        test_summary = {
            "experiment_name": exp_name,
            "feature_name": args.feature_name,
            "model_name": model_name,
            "task_type": "video_only",
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "task": "task2" if args.test_split == "test_task2" else args.test_split,
            "split": args.test_split,
            "manifest": args.test_manifest,
            "checkpoint_path": str(ckpt_dir / "best.pt"),
            "num_samples": len(test_ds),
            "metrics": dict(test_metrics),
        }
        test_summary_path = metrics_dir / f"{exp_name}_{args.test_split}.json"
        test_summary_path.write_text(json.dumps(test_summary, indent=2, ensure_ascii=False))
        pred_path = pred_dir / f"{exp_name}_{args.test_split}.csv"
        with pred_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "probability", "label"])
            for sid, p, lab in zip(sids, probs, labels):
                w.writerow([sid, p, lab])
        logger.info(
            "DONE val_auprc=%.4f@epoch%d  test_auprc=%.4f",
            best_val_auprc, best_val_record["epoch"], test_metrics["auprc"],
        )
    else:
        logger.info(
            "DONE val_auprc=%.4f@epoch%d  (no test eval)",
            best_val_auprc, best_val_record["epoch"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
