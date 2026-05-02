"""Aggregate Task3 results across seeds per (model, lr).

Reads neuromm26_results/metrics/<exp>_train_summary.json (val) or
                                 <exp>_test_task3.json     (test)
filtered by task_type starting with 'task3_'. Mirrors aggregate_results.py
output style.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HIGHLIGHT_METRICS = (
    "weighted_f1",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
)


def discover(metrics_dir: Path, kind: str, seeds: set[int] | None = None):
    """kind = 'val' (read *_train_summary.json best_metrics) or 'test_task3'."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    if kind == "val":
        pattern = "*_train_summary.json"
    elif kind == "test_task3":
        pattern = "*_test_task3.json"
    else:
        raise ValueError(kind)

    for path in sorted(metrics_dir.glob(pattern)):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        task_type = data.get("task_type", "")
        if not task_type.startswith("task3"):
            continue
        if kind == "val":
            metrics = data.get("best_metrics") or {}
        else:
            metrics = data.get("metrics") or {}
        if not metrics:
            continue
        model = data.get("model_name")
        seed = data.get("seed")
        lr = float(data.get("learning_rate", 0.0) or 0.0)
        if model is None:
            continue
        if seeds is not None and seed not in seeds:
            continue
        groups[(model, lr)].append({"seed": seed, "metrics": metrics})
    return dict(groups)


def aggregate(groups):
    rows = []
    for (model, lr) in sorted(groups):
        entries = groups[(model, lr)]
        seeds_found = sorted(e["seed"] for e in entries if e["seed"] is not None)
        row = {"model": model, "lr": lr, "n_seeds": len(entries), "seeds": seeds_found}
        per_metric: dict[str, list[float]] = defaultdict(list)
        for e in entries:
            for k in HIGHLIGHT_METRICS:
                if k in e["metrics"]:
                    per_metric[k].append(float(e["metrics"][k]))
        for k in HIGHLIGHT_METRICS:
            vals = per_metric.get(k, [])
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_std"] = float(np.std(vals)) if vals else None
        rows.append(row)
    return rows


def print_table(rows, title: str):
    if not rows:
        print(f"No results for {title}")
        return
    width = max(len("Model"), max(len(r["model"]) for r in rows))
    header = [f"{'Model':<{width}}", f"{'LR':>8}", f"{'N':>2}"]
    for k in HIGHLIGHT_METRICS:
        header.append(f"{k:>16}")
    line = "  ".join(header)
    print()
    print(f"== {title} ==")
    print(line)
    print("-" * len(line))
    sorted_rows = sorted(rows, key=lambda r: r.get("weighted_f1_mean") or 0, reverse=True)
    for r in sorted_rows:
        lr_str = f"{r['lr']:.0e}" if r["lr"] >= 1e-6 else "0"
        parts = [f"{r['model']:<{width}}", f"{lr_str:>8}", f"{r['n_seeds']:>2}"]
        for k in HIGHLIGHT_METRICS:
            m = r.get(f"{k}_mean")
            s = r.get(f"{k}_std")
            if m is None:
                parts.append(f"{'---':>16}")
            else:
                parts.append(f"{m:.4f}±{s:.4f}")
        print("  ".join(parts))
    print()


def save_csv(rows, path: Path):
    fieldnames = ["model", "lr", "n_seeds", "seeds"]
    for k in HIGHLIGHT_METRICS:
        fieldnames.extend([f"{k}_mean", f"{k}_std"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.get("weighted_f1_mean") or 0, reverse=True):
            row = {k: r.get(k) for k in fieldnames}
            row["seeds"] = str(r.get("seeds", []))
            w.writerow(row)
    print(f"CSV saved: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate NeuroMM 2026 Task3 results across seeds.")
    parser.add_argument("--kind", choices=["val", "test", "both"], default="both")
    parser.add_argument("--metrics-dir", default="neuromm26_results/metrics")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    seeds = set(args.seeds) if args.seeds is not None else None

    if args.kind in {"val", "both"}:
        groups = discover(metrics_dir, "val", seeds=seeds)
        rows = aggregate(groups)
        print_table(rows, "Task3 VAL (best by weighted_f1)")
        save_csv(rows, metrics_dir / "baseline_task3_val_summary.csv")
    if args.kind in {"test", "both"}:
        groups = discover(metrics_dir, "test_task3", seeds=seeds)
        rows = aggregate(groups)
        print_table(rows, "Task3 TEST (split=test_task3)")
        save_csv(rows, metrics_dir / "baseline_task3_test_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
