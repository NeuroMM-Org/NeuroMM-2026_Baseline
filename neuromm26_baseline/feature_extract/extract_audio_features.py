"""NeuroMM 2026 Audio Feature Extraction (OpenSMILE).

Reads a manifest with `sample_id` and `raw_video_relpath` columns, demuxes the
audio track of each video to a 16 kHz mono WAV via ffmpeg, then runs OpenSMILE
to compute either eGeMAPSv02 (88-D) or ComParE 2016 (6373-D) functionals.

Output shape per sample: (D,) float32 saved as numpy under
    {output_dir}/{feature_name}/{sample_id}.npy

Usage:
    python -m neuromm26_baseline.feature_extract.extract_audio_features \\
        --manifest neuromm26_datasets/annotations/neuromm2026_full.csv \\
        --raw-video-dir neuromm26_datasets \\
        --output-dir neuromm26_datasets/processed/features/audio \\
        --feature-name opensmile-egemaps
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


def lazy_import_opensmile():
    import opensmile  # noqa: WPS433  (intentional: keep import optional)

    return opensmile


def feature_set_for(name: str):
    """Return (opensmile.FeatureSet, expected_dim, label) for a registry name."""
    opensmile = lazy_import_opensmile()
    table = {
        "opensmile-egemaps": (opensmile.FeatureSet.eGeMAPSv02, 88, "eGeMAPSv02 / Functionals"),
        "opensmile-compare": (opensmile.FeatureSet.ComParE_2016, 6373, "ComParE_2016 / Functionals"),
    }
    if name not in table:
        raise ValueError(f"Unsupported feature_name: {name}. Supported: {list(table)}")
    return table[name]


def ensure_audio_wav(video_path: Path, dest_wav: Path, sample_rate: int = 16000) -> bool:
    """Demux audio from video to mono WAV. Returns True on success."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                   # no video
        "-ac", "1",              # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        str(dest_wav),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        # Audio track missing or other ffmpeg error -- caller will report.
        return False
    return dest_wav.exists() and dest_wav.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroMM 2026 OpenSMILE Audio Feature Extraction")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--raw-video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root directory for feature output (e.g. processed/features/audio)")
    parser.add_argument("--feature-name", type=str, required=True,
                        choices=["opensmile-egemaps", "opensmile-compare"])
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate for the intermediate WAV (default 16000)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")

    opensmile = lazy_import_opensmile()
    feature_set, expected_dim, label = feature_set_for(args.feature_name)

    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feature_output_dir = Path(args.output_dir) / args.feature_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Backend: OpenSMILE {opensmile.__version__}  set={label}  expected_dim={expected_dim}")
    print(f"Output  : {feature_output_dir}")

    df = pd.read_csv(args.manifest)
    if "raw_video_relpath" not in df.columns:
        raise ValueError("Manifest must have a 'raw_video_relpath' column")
    df = df.dropna(subset=["raw_video_relpath"])
    df = df[df["raw_video_relpath"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["sample_id"])
    print(f"Loaded {len(df)} unique samples with video from {args.manifest}")

    success = 0
    skipped = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix="opensmile_audio_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {args.feature_name}"):
            sample_id = row["sample_id"]
            video_relpath = row["raw_video_relpath"]
            video_path = Path(args.raw_video_dir) / video_relpath
            output_npy = feature_output_dir / f"{sample_id}.npy"

            if output_npy.exists():
                skipped += 1
                continue

            if not video_path.exists():
                tqdm.write(f"[Warn] missing video: {video_path}")
                failed += 1
                continue

            wav_path = tmpdir_path / f"{sample_id}.wav"
            ok = ensure_audio_wav(video_path, wav_path, sample_rate=args.sample_rate)
            if not ok:
                tqdm.write(f"[Warn] no audio extracted: {sample_id}")
                failed += 1
                continue

            try:
                features_df = smile.process_file(str(wav_path))
                # OpenSMILE Functionals returns 1xD per file.
                features = features_df.to_numpy(dtype=np.float32).reshape(-1)
                if features.shape[0] != expected_dim:
                    tqdm.write(f"[Warn] {sample_id}: dim {features.shape[0]} != expected {expected_dim}")
                np.save(output_npy, features)
                success += 1
            except Exception as exc:
                tqdm.write(f"[Error] {sample_id}: {exc}")
                failed += 1
            finally:
                wav_path.unlink(missing_ok=True)

    print()
    print("Extraction Complete!")
    print(f"  Success={success}  Skipped(already)={skipped}  Failed={failed}")
    print(f"  Per-file shape: ({expected_dim},)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
