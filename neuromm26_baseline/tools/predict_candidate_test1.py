"""Test 1 (EEG-only, binary spike detection) — predict the candidate set.

Runs a trained EEG-only checkpoint over EVERY candidate id and writes a
Codabench submission CSV (columns: sample_id,prediction) where `prediction`
is P(spike) in [0,1]. AUPRC is the official Test-1 metric.

Example:
    python -m neuromm26_baseline.tools.predict_candidate_test1 \
        --checkpoint neuromm26_results/checkpoints/<exp>/best.pt \
        --config     neuromm26_results/metrics/<exp>_config.json \
        --candidate-dir /path/to/candidate_set \
        --out submission_test1.csv

`--config` is the resolved config JSON written next to the metrics during
training (`neuromm26_results/metrics/<exp>_config.json`). Submit the CSV
inside a zip as `prediction.csv`.
"""
from __future__ import annotations

import argparse
import json
import tempfile

from neuromm26_baseline.trainers.evaluator import Evaluator

from .runtime import build_dataloader, build_dataset, build_model
from ._candidate_common import (
    build_candidate_manifest, load_checkpoint, resolve_device,
    write_submission_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True,
                    help="resolved config JSON written during training "
                         "(neuromm26_results/metrics/<exp>_config.json)")
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--out", default="submission_test1.csv")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    with open(args.config) as f:
        cfg = json.load(f)

    manifest = tempfile.NamedTemporaryFile(
        "w", suffix="_cand_manifest.csv", delete=False).name
    n = build_candidate_manifest(args.candidate_dir, manifest)
    print(f"[info] candidate ids: {n}")

    cfg.setdefault("paths", {})
    cfg["paths"]["annotation_manifest"] = manifest
    cfg["paths"]["eeg_feature_root"] = f"{args.candidate_dir.rstrip('/')}/eeg"
    cfg.setdefault("runtime", {})["preload_in_memory"] = True
    if cfg.get("model", {}).get("task_type") not in (None, "eeg_only"):
        raise SystemExit(
            f"Test 1 expects an eeg_only checkpoint, got task_type="
            f"{cfg['model'].get('task_type')}")

    ds = build_dataset(cfg, "candidate")
    if len(ds) != n:
        print(f"[warn] dataset rows {len(ds)} != candidate ids {n}")
    loader = build_dataloader(cfg, ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(cfg).to(device)
    state = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state["model_state_dict"])

    res = Evaluator(device=device).evaluate(model, loader, criterion=None)
    write_submission_csv(args.out, res.sample_ids, res.probabilities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
