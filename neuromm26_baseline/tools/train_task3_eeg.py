"""Task3 EEG-only 5-class trainer (seizure subtype on positives).

Train+val from neuromm2026_train_val_patient_split.csv (filtered to label_type>0).
Test from neuromm2026_test_task3.csv (176 positives).

Selection metric: weighted_f1 (i.e. F1 weighted by class support).
Outputs (compatible with aggregate_task3_results.py):
    metrics/<exp>_train_summary.json
    metrics/<exp>_test_task3.json
    predictions/<exp>_test_task3.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets.task3_datasets import Task3EEGFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.models.legacy.registry_task3 import (
    build_legacy_eeg_model_with_num_classes,
)
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.multiclass_metrics import format_multiclass_summary
from neuromm26_baseline.utils.seed import build_torch_generator, set_seed

from ._task3_runtime import (
    class_balanced_weights,
    evaluate_multiclass,
    move_to,
    save_json,
    save_predictions_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-model", required=True,
                        help="legacy EEG model name, e.g. tcnet_eeg, eegnet")
    parser.add_argument("--train-val-manifest",
                        default="neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv")
    parser.add_argument("--test-manifest",
                        default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--eeg-feature-root",
                        default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--output-root", default="neuromm26_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use inverse-frequency class weights in CrossEntropy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eeg-shape", type=int, nargs=2, default=[26, 2000])
    args = parser.parse_args()

    set_seed(int(args.seed))

    eeg_safe = args.eeg_model.replace("/", "-")
    exp_name = f"task3_eeg__{eeg_safe}__seed{args.seed}__lr{args.learning_rate}"
    model_name_stamp = f"task3_eeg/{eeg_safe}"

    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    metrics_dir = ensure_dir(out_root / "metrics")
    pred_dir = ensure_dir(out_root / "predictions")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.task3_eeg", str(log_path))
    logger.info("experiment=%s eeg=%s seed=%d lr=%s", exp_name, args.eeg_model, args.seed, args.learning_rate)

    device = torch.device(args.device)

    train_ds = Task3EEGFeatureDataset(args.train_val_manifest, "train",
                                      args.eeg_feature_root, target_shape=tuple(args.eeg_shape))
    val_ds = Task3EEGFeatureDataset(args.train_val_manifest, "val",
                                    args.eeg_feature_root, target_shape=tuple(args.eeg_shape))
    test_ds = None
    test_loader = None
    if args.test_manifest and Path(args.test_manifest).exists() and args.test_split:
        test_ds = Task3EEGFeatureDataset(args.test_manifest, args.test_split,
                                         args.eeg_feature_root, target_shape=tuple(args.eeg_shape))
        logger.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
    else:
        logger.info("train=%d val=%d test=- (no test manifest)", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False,
                              collate_fn=neuromm_collate,
                              generator=build_torch_generator(int(args.seed)))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=neuromm_collate)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False, collate_fn=neuromm_collate)

    model = build_legacy_eeg_model_with_num_classes(args.eeg_model, args.num_classes).to(device)

    if args.use_class_weights:
        cw = class_balanced_weights(train_ds.labels, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        logger.info("class_weights=%s", cw.tolist())
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
                "eeg_model": args.eeg_model,
                "num_classes": args.num_classes,
            }, ckpt_dir / "best.pt")

    if best_record is None:
        raise RuntimeError("Training did not produce any valid metric.")

    save_json({
        "experiment_name": exp_name,
        "task_type": "task3_eeg",
        "model_name": model_name_stamp,
        "eeg_model": args.eeg_model,
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
            "task_type": "task3_eeg",
            "model_name": model_name_stamp,
            "eeg_model": args.eeg_model,
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
            "DONE val_weighted_f1=%.4f@epoch%d  test_weighted_f1=%.4f  test_macro_f1=%.4f  test_accuracy=%.4f",
            best_metric, best_record["epoch"],
            float(test_res["metrics"].get("weighted_f1", 0.0)),
            float(test_res["metrics"].get("macro_f1", 0.0)),
            float(test_res["metrics"].get("accuracy", 0.0)),
        )
    else:
        logger.info(
            "DONE val_weighted_f1=%.4f@epoch%d  (no test eval)",
            best_metric, best_record["epoch"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
