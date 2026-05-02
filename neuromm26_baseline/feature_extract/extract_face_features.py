"""NeuroMM 2026 Facial Feature Extraction (OpenFace 2.x).

Wraps the OpenFace `FeatureExtraction` binary to produce per-frame facial
features (Action Units, gaze, head pose) for each video in a manifest. By
default outputs 31 dims/frame: 17 AU intensities + 6 head pose + 8 gaze.

Output shape per sample: (T_frames, D) float32 saved as numpy under
    {output_dir}/{feature_name}/{sample_id}.npy

Usage:
    python -m neuromm26_baseline.feature_extract.extract_face_features \\
        --manifest neuromm26_datasets/annotations/neuromm2026_full.csv \\
        --raw-video-dir neuromm26_datasets \\
        --output-dir neuromm26_datasets/processed/features/face \\
        --feature-name openface

OpenFace binary needs to be on PATH. If not installed, see the install hints
printed when this script aborts with "FeatureExtraction not found".
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

# Default columns we keep from OpenFace's CSV. Stable across OpenFace 2.x.
AU_INTENSITY_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
]
HEAD_POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE_COLS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]
FEATURE_COLUMNS = AU_INTENSITY_COLS + HEAD_POSE_COLS + GAZE_COLS  # 17 + 6 + 8 = 31

INSTALL_HINTS = """
OpenFace binary 'FeatureExtraction' was not found on PATH.

Install options:

(A) Apt (Ubuntu 20.04 / Debian-based, easiest):
    apt-get update && apt-get install -y openface
    # Provides FeatureExtraction in /usr/bin

(B) Build from source (if (A) fails or gives mismatched version):
    git clone https://github.com/TadasBaltrusaitis/OpenFace.git
    cd OpenFace
    bash download_models.sh
    mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make -j4
    # binary in: ./bin/FeatureExtraction
    # add to PATH or pass --binary /path/to/FeatureExtraction

(C) Use docker image:
    docker run -v /data:/data tadasbaltrusaitis/openface:latest \\
        FeatureExtraction -f /data/clip.mp4 -out_dir /data/out
"""


def find_binary(explicit: str | None = None) -> str:
    if explicit:
        if Path(explicit).is_file() and os.access(explicit, os.X_OK):
            return explicit
        raise FileNotFoundError(f"--binary points to non-executable: {explicit}")
    candidate = shutil.which("FeatureExtraction")
    if candidate:
        return candidate
    print(INSTALL_HINTS)
    raise FileNotFoundError("FeatureExtraction not found on PATH; set --binary or install OpenFace")


def run_openface(binary: str, video_path: Path, out_dir: Path) -> Path:
    """Run FeatureExtraction on one video; return path to its CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        "-f", str(video_path),
        "-out_dir", str(out_dir),
        "-aus", "-gaze", "-pose",
        "-q",  # quieter
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    csv = out_dir / f"{video_path.stem}.csv"
    if not csv.exists():
        raise RuntimeError(f"OpenFace produced no CSV for {video_path}")
    return csv


def parse_features(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OpenFace CSV missing columns: {missing[:5]}...")
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroMM 2026 OpenFace Face Feature Extraction")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--raw-video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root directory for feature output (e.g. processed/features/face)")
    parser.add_argument("--feature-name", type=str, default="openface", choices=["openface"])
    parser.add_argument("--binary", type=str, default=None, help="Path to FeatureExtraction binary")
    args = parser.parse_args()

    binary = find_binary(args.binary)
    print(f"Using OpenFace binary: {binary}")

    feature_output_dir = Path(args.output_dir) / args.feature_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output  : {feature_output_dir}")

    df = pd.read_csv(args.manifest)
    if "raw_video_relpath" not in df.columns:
        raise ValueError("Manifest must have a 'raw_video_relpath' column")
    df = df.dropna(subset=["raw_video_relpath"])
    df = df[df["raw_video_relpath"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["sample_id"])
    print(f"Loaded {len(df)} unique samples with video from {args.manifest}")

    success = skipped = failed = 0
    with tempfile.TemporaryDirectory(prefix="openface_") as tmp:
        tmp_path = Path(tmp)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting openface"):
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

            try:
                csv = run_openface(binary, video_path, tmp_path / sample_id)
                arr = parse_features(csv)
                np.save(output_npy, arr)
                success += 1
            except Exception as exc:
                tqdm.write(f"[Error] {sample_id}: {exc}")
                failed += 1
            finally:
                # Clean per-sample tmp subdir to avoid filling disk
                shutil.rmtree(tmp_path / sample_id, ignore_errors=True)

    print()
    print("Extraction Complete!")
    print(f"  Success={success}  Skipped(already)={skipped}  Failed={failed}")
    print(f"  Per-file shape: (T_frames, {len(FEATURE_COLUMNS)})  cols={FEATURE_COLUMNS[:5]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
