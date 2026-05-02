"""End-to-end EEG + Video late-fusion training entrypoint for Task2.

Loads:
  EEG features      from neuromm26_datasets/processed/features/eeg/{sample_id}.npy
  Video features    from neuromm26_datasets/processed/features/video/{video_feature_name}/{sample_id}.npy

Training set : neuromm2026_train_val_patient_split.csv (split=train)
Validation   : neuromm2026_train_val_patient_split.csv (split=val)
Test         : neuromm2026_test_task2.csv              (split=test_task2)

Saves best.pt by val AUPRC, then evaluates on test_task2 with that ckpt.
JSON outputs are named so existing aggregate_results.py and
aggregate_test_results.py pick them up.

Run example:
    python -m neuromm26_baseline.tools.train_eeg_video_fusion \\
        --eeg-model tcnet_eeg --video-feature-name clip-base \\
        --seed 0 --epochs 30 --learning-rate 5e-4
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
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import NeuroMMMultimodalFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.models.multimodal_late_fusion import EEGVideoLateFusion
from neuromm26_baseline.trainers.evaluator import Evaluator
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import format_metric_summary, order_metric_dict
from neuromm26_baseline.utils.seed import build_torch_generator, seed_worker, set_seed


def detect_video_feature_dim(video_feature_root: str, video_feature_name: str) -> int:
    """Pick any .npy under that dir and return its trailing dim."""
    root = Path(video_feature_root) / video_feature_name
    samples = list(root.glob("*.npy"))
    if not samples:
        raise FileNotFoundError(f"No .npy under {root}")
    arr = np.load(samples[0])
    if arr.ndim == 1:
        return int(arr.shape[0])
    return int(arr.shape[-1])


def build_loaders(
    train_val_manifest: str,
    test_manifest: str,
    test_split: str,
    eeg_feature_root: str,
    video_feature_root: str,
    video_feature_name: str,
    eeg_target_shape: tuple[int, int],
    batch_size: int,
    eval_batch_size: int,
    seed: int,
):
    train_ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=train_val_manifest,
        split="train",
        eeg_feature_root=eeg_feature_root,
        video_feature_root=video_feature_root,
        video_feature_name=video_feature_name,
        preload_in_memory=True,
        target_shape=eeg_target_shape,
        require_video_feature=True,
    )
    val_ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=train_val_manifest,
        split="val",
        eeg_feature_root=eeg_feature_root,
        video_feature_root=video_feature_root,
        video_feature_name=video_feature_name,
        preload_in_memory=True,
        target_shape=eeg_target_shape,
        require_video_feature=True,
    )
    test_ds = None
    test_loader = None
    if test_manifest and Path(test_manifest).exists() and test_split:
        test_ds = NeuroMMMultimodalFeatureDataset(
            manifest_csv=test_manifest,
            split=test_split,
            eeg_feature_root=eeg_feature_root,
            video_feature_root=video_feature_root,
            video_feature_name=video_feature_name,
            preload_in_memory=True,
            target_shape=eeg_target_shape,
            require_video_feature=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=neuromm_collate,
        generator=build_torch_generator(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=neuromm_collate,
    )
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=neuromm_collate,
        )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-model", required=True,
                        help="legacy EEG model name, e.g. tcnet_eeg, eegnet, efficientnet_v2_s_eeg")
    parser.add_argument(
        "--video-feature-name", required=True,
        help="Subdirectory under <video-feature-root>/, e.g. clip-base, videomae-base, "
             "dinov2-base, dinov2-large, siglip-base, videomae-large, timesformer-k400",
    )
    parser.add_argument("--train-val-manifest",
                        default="neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv")
    parser.add_argument("--test-manifest",
                        default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--eeg-feature-root", default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--video-feature-root", default="neuromm26_datasets/processed/features/video")
    parser.add_argument("--output-root", default="neuromm26_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--video-hidden-dim", type=int, default=256)
    parser.add_argument("--video-dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eeg-shape", type=int, nargs=2, default=[26, 2000])
    args = parser.parse_args()

    set_seed(int(args.seed))

    feature_dim = detect_video_feature_dim(args.video_feature_root, args.video_feature_name)
    eeg_safe = args.eeg_model.replace("/", "-")
    video_safe = args.video_feature_name.replace("/", "-")
    exp_name = (
        f"eeg_video_fusion__{eeg_safe}__{video_safe}"
        f"__seed{args.seed}__lr{args.learning_rate}"
    )
    model_name_stamp = f"fusion/{eeg_safe}+{video_safe}"

    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    metrics_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.eeg_video_fusion", str(log_path))
    logger.info(
        "experiment=%s eeg=%s video=%s feat_dim=%d seed=%d lr=%s",
        exp_name, args.eeg_model, args.video_feature_name, feature_dim, args.seed, args.learning_rate,
    )

    device = torch.device(args.device)
    logger.info("device=%s", device)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        args.train_val_manifest,
        args.test_manifest,
        args.test_split,
        args.eeg_feature_root,
        args.video_feature_root,
        args.video_feature_name,
        eeg_target_shape=tuple(args.eeg_shape),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=int(args.seed),
    )
    if test_ds is not None:
        logger.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
    else:
        logger.info("train=%d val=%d test=- (no test manifest)", len(train_ds), len(val_ds))

    model = EEGVideoLateFusion(
        eeg_model_name=args.eeg_model,
        video_feature_dim=feature_dim,
        video_hidden_dim=args.video_hidden_dim,
        video_dropout=args.video_dropout,
    ).to(device)

    train_labels = np.array(train_ds.eeg_dataset.labels, dtype=np.float32)
    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    evaluator = Evaluator(device=device)
    best_val_auprc = float("-inf")
    best_val_record: dict[str, Any] | None = None
    epoch_log: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch["label"].view(-1).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_result = evaluator.evaluate(model, val_loader, criterion=criterion)
        val_metrics = order_metric_dict(val_result.metrics)
        val_auprc = float(val_metrics.get("auprc", float("-inf")))
        epoch_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_auprc": val_auprc,
            "val_binary_f1": float(val_metrics.get("binary_f1", 0.0)),
            "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
        })
        logger.info(
            "epoch=%d train_loss=%.4f %s",
            epoch, train_loss, format_metric_summary(val_metrics),
        )
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_record = {"epoch": epoch, "metrics": dict(val_metrics)}
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": dict(val_metrics),
                "best_metric_name": "auprc",
                "best_metric_value": best_val_auprc,
                "best_metrics": dict(val_metrics),
                "eeg_model": args.eeg_model,
                "video_feature_name": args.video_feature_name,
                "video_feature_dim": feature_dim,
                "video_hidden_dim": args.video_hidden_dim,
            }, ckpt_dir / "best.pt")

    if best_val_record is None:
        raise RuntimeError("Training produced no valid val metric")

    train_summary = {
        "experiment_name": exp_name,
        "eeg_model": args.eeg_model,
        "video_feature_name": args.video_feature_name,
        "model_name": model_name_stamp,
        "task_type": "eeg_video_fusion",
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "best_metric_name": "auprc",
        "best_metric_value": best_val_auprc,
        "best_metrics": best_val_record["metrics"],
        "best_checkpoint_path": str(ckpt_dir / "best.pt"),
        "best_epoch": best_val_record["epoch"],
        "epoch_log": epoch_log,
    }
    (metrics_dir / f"{exp_name}_train_summary.json").write_text(
        json.dumps(train_summary, indent=2, ensure_ascii=False)
    )

    if test_ds is not None and test_loader is not None:
        # Reload best-val ckpt and evaluate on test
        state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        test_result = evaluator.evaluate(model, test_loader, criterion=None)
        test_metrics = order_metric_dict(test_result.metrics)
        test_summary = {
            "experiment_name": exp_name,
            "eeg_model": args.eeg_model,
            "video_feature_name": args.video_feature_name,
            "model_name": model_name_stamp,
            "task_type": "eeg_video_fusion",
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "task": "task2" if args.test_split == "test_task2" else args.test_split,
            "split": args.test_split,
            "manifest": args.test_manifest,
            "checkpoint_path": str(ckpt_dir / "best.pt"),
            "num_samples": len(test_ds),
            "metrics": dict(test_metrics),
        }
        (metrics_dir / f"{exp_name}_{args.test_split}.json").write_text(
            json.dumps(test_summary, indent=2, ensure_ascii=False)
        )
        pred_path = pred_dir / f"{exp_name}_{args.test_split}.csv"
        with pred_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "probability", "label"])
            for sid, p, lab in zip(test_result.sample_ids, test_result.probabilities, test_result.labels or []):
                w.writerow([sid, p, lab])
        logger.info(
            "DONE val_auprc=%.4f@epoch%d  test_auprc=%.4f",
            best_val_auprc, best_val_record["epoch"], test_metrics.get("auprc", float("nan")),
        )
    else:
        logger.info(
            "DONE val_auprc=%.4f@epoch%d  (no test eval)",
            best_val_auprc, best_val_record["epoch"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
