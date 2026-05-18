"""Test 2 (EEG + Video, binary spike detection) — predict the candidate set.

Rebuilds the EEG+Video late-fusion model directly from the checkpoint
metadata (eeg_model / video_feature_name / video_feature_dim /
video_hidden_dim saved by train_eeg_video_fusion.py), runs it over EVERY
candidate id, and writes a Codabench submission CSV (sample_id,prediction)
where `prediction` is P(spike) in [0,1]. AUPRC is the official metric.

Example:
    python -m neuromm26_baseline.tools.predict_candidate_test2 \
        --checkpoint neuromm26_results/checkpoints/<exp>/best.pt \
        --candidate-dir /path/to/candidate_set \
        --out submission_test2.csv

The video backbone is taken from the checkpoint; override with
--video-feature-name only if you know it differs.
"""
from __future__ import annotations

import argparse
import tempfile

import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import NeuroMMMultimodalFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.models.multimodal_late_fusion import EEGVideoLateFusion
from neuromm26_baseline.trainers.evaluator import Evaluator

from ._candidate_common import (
    build_candidate_manifest, load_checkpoint, resolve_device,
    write_submission_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--out", default="submission_test2.csv")
    ap.add_argument("--video-feature-name", default=None,
                    help="override the backbone stored in the checkpoint")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    for k in ("eeg_model", "video_feature_dim"):
        if k not in ckpt:
            raise SystemExit(
                f"checkpoint missing '{k}'; this does not look like a Test-2 "
                f"(eeg_video_fusion) checkpoint from train_eeg_video_fusion.py")
    eeg_model = ckpt["eeg_model"]
    vfn = args.video_feature_name or ckpt.get("video_feature_name")
    vdim = int(ckpt["video_feature_dim"])
    vhid = int(ckpt.get("video_hidden_dim", 256))
    if not vfn:
        raise SystemExit("video_feature_name unknown; pass --video-feature-name")
    print(f"[info] eeg_model={eeg_model} video={vfn} dim={vdim} hidden={vhid}")

    manifest = tempfile.NamedTemporaryFile(
        "w", suffix="_cand_manifest.csv", delete=False).name
    n = build_candidate_manifest(args.candidate_dir, manifest)
    print(f"[info] candidate ids: {n}")

    cdir = args.candidate_dir.rstrip("/")
    ds = NeuroMMMultimodalFeatureDataset(
        eeg_feature_root=f"{cdir}/eeg",
        video_feature_root=f"{cdir}/video",
        video_feature_name=vfn,
        manifest_csv=manifest,
        split="candidate",
        preload_in_memory=True,
        target_shape=(26, 2000),
        require_video_feature=True,
    )
    if len(ds) != n:
        print(f"[warn] dataset rows {len(ds)} != candidate ids {n}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=neuromm_collate)

    model = EEGVideoLateFusion(
        eeg_model_name=eeg_model,
        video_feature_dim=vdim,
        video_hidden_dim=vhid,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    res = Evaluator(device=device).evaluate(model, loader, criterion=None)
    write_submission_csv(args.out, res.sample_ids, res.probabilities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
