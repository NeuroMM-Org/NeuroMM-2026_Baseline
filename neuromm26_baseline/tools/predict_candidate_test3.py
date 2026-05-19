"""Test 3 (5-class seizure subtype) — predict the candidate set.

Auto-detects from the checkpoint whether it is a Task-3 EEG-only model
(train_task3_eeg.py) or an EEG+Video fusion model (train_task3_fusion.py),
rebuilds it, runs it over EVERY candidate id, and writes a Codabench
submission CSV (sample_id,prediction).

`prediction` is the predicted seizure subtype in the SAME label space as the
ground truth, i.e. label_type in {1,2,3,4,5} (model class argmax 0..4, +1).
Weighted-F1 is the official Test-3 metric.

Example:
    python -m neuromm26_baseline.tools.predict_candidate_test3 \
        --checkpoint neuromm26_results/checkpoints/<task3_exp>/best.pt \
        --candidate-dir /path/to/candidate_set \
        --out submission.csv
"""
from __future__ import annotations

import argparse
import tempfile

from torch.utils.data import DataLoader

from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.task3_datasets import (
    Task3EEGFeatureDataset, Task3MultimodalFeatureDataset,
)
from neuromm26_baseline.models.legacy.registry_task3 import (
    build_legacy_eeg_model_with_num_classes,
)
from neuromm26_baseline.models.multiclass_late_fusion import (
    EEGVideoLateFusion5Class,
)

from ._candidate_common import (
    build_candidate_manifest, load_checkpoint, resolve_device,
    write_submission_csv,
)
from ._task3_runtime import evaluate_multiclass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--video-feature-name", default=None,
                    help="override the backbone stored in a fusion checkpoint")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    if "eeg_model" not in ckpt:
        raise SystemExit(
            "checkpoint has no 'eeg_model'; only Task-3 EEG-only and "
            "EEG+Video fusion checkpoints are supported by this script "
            "(video-only MLP is not).")
    n_cls = int(ckpt.get("num_classes", 5))
    eeg_model = ckpt["eeg_model"]
    vfn = args.video_feature_name or ckpt.get("video_feature_name")
    is_fusion = bool(vfn) and "video_feature_dim" in ckpt

    manifest = tempfile.NamedTemporaryFile(
        "w", suffix="_cand_manifest.csv", delete=False).name
    n = build_candidate_manifest(args.candidate_dir, manifest)
    cdir = args.candidate_dir.rstrip("/")
    print(f"[info] mode={'fusion' if is_fusion else 'eeg-only'} "
          f"eeg_model={eeg_model} num_classes={n_cls} candidate_ids={n}")

    if is_fusion:
        ds = Task3MultimodalFeatureDataset(
            manifest_csv=manifest, split="candidate",
            eeg_feature_root=f"{cdir}/eeg",
            video_feature_root=f"{cdir}/video",
            video_feature_name=vfn,
            preload_in_memory=True, target_shape=(26, 2000),
            require_video_feature=True,
        )
        model = EEGVideoLateFusion5Class(
            eeg_model_name=eeg_model,
            video_feature_dim=int(ckpt["video_feature_dim"]),
            video_hidden_dim=int(ckpt.get("video_hidden_dim", 256)),
            num_classes=n_cls,
        ).to(device)
    else:
        ds = Task3EEGFeatureDataset(
            manifest_csv=manifest, split="candidate",
            eeg_feature_root=f"{cdir}/eeg",
            preload_in_memory=True, target_shape=(26, 2000),
        )
        model = build_legacy_eeg_model_with_num_classes(
            eeg_model, n_cls).to(device)

    if len(ds) != n:
        print(f"[warn] dataset rows {len(ds)} != candidate ids {n} "
              f"(candidate manifest sets label_type=1 so none are filtered)")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=neuromm_collate)
    model.load_state_dict(ckpt["model_state_dict"])

    res = evaluate_multiclass(model, loader, device, num_classes=n_cls)
    # model class 0..(C-1)  ->  ground-truth label_type 1..C
    preds = [int(p) + 1 for p in res["predictions"]]
    write_submission_csv(args.out, res["sample_ids"], preds, float_fmt=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
