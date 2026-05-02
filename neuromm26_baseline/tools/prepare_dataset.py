"""Normalize raw annotations and build a release-safe dataset layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_ANNOTATION_FILENAME = "neuromm2026_train_val.csv"

SPLIT_MAP = {
    "0": "train",
    "1": "val",
    "2": "test",
    "train": "train",
    "val": "val",
    "test": "test",
}


def normalize_raw_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip()
    marker = "vepiset-dataset_add_video/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def infer_subject_id(video_relpath: str) -> str:
    parts = Path(video_relpath).parts
    return parts[-2] if len(parts) >= 2 else ""


def load_records(raw_csv: Path) -> list[dict[str, str]]:
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    records: list[dict[str, str]] = []
    for row in rows:
        eeg_source_rel = normalize_raw_path(row["file_path"])
        video_source_rel = normalize_raw_path(row["video_file_path"])
        sample_id = Path(eeg_source_rel).stem
        split = SPLIT_MAP.get(row["train_val"], row["train_val"])
        label = row["target"] if split != "test" else ""
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": label,
                "raw_video_relpath": f"raw/eeg_videos/{video_source_rel}" if video_source_rel else "",
                "subject_id": infer_subject_id(video_source_rel),
                "eeg_source_relpath": f"raw/eeg_videos/{eeg_source_rel}",
            }
        )
    return records


def copy_public_eeg(records: list[dict[str, str]], dataset_root: Path) -> None:
    target_dir = dataset_root / "processed" / "features" / "eeg"
    target_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = dataset_root / record["eeg_source_relpath"]
        target = target_dir / f"{record['sample_id']}.npy"
        if not target.exists():
            shutil.copy2(source, target)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_unmatched_videos(dataset_root: Path, records: list[dict[str, str]]) -> list[dict[str, str]]:
    video_root = dataset_root / "raw" / "eeg_videos" / "videos"
    manifest_video_names = {Path(record["raw_video_relpath"]).name for record in records if record["raw_video_relpath"]}
    unmatched = []
    for path in sorted(video_root.rglob("*.mp4")):
        if path.name not in manifest_video_names:
            unmatched.append(
                {
                    "sample_id": path.stem,
                    "raw_video_relpath": str(path.relative_to(dataset_root)).replace("\\", "/"),
                }
            )
    return unmatched


def write_summary(dataset_root: Path, records: list[dict[str, str]], unmatched_videos: list[dict[str, str]]) -> None:
    summary = {
        "num_samples": len(records),
        "split_counts": Counter(record["split"] for record in records),
        "label_counts": Counter(record["label"] for record in records if record["label"] != ""),
        "num_unmatched_raw_videos": len(unmatched_videos),
        "recommended_feature_layout": {
            "eeg": "processed/features/eeg/{sample_id}.npy",
            "video": "processed/features/{video_feature_name}/{sample_id}.npy",
        },
    }
    output = dataset_root / "processed" / "metadata" / "dataset_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="neuromm26_datasets")
    parser.add_argument("--raw-subdir", default="raw/eeg_videos")
    parser.add_argument("--copy-eeg", action="store_true")
    parser.add_argument("--annotation-filename", default=DEFAULT_ANNOTATION_FILENAME)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    raw_root = dataset_root / args.raw_subdir
    raw_csv = raw_root / "vepiset_spike_sleep_video.csv"
    records = load_records(raw_csv)

    (dataset_root / "processed" / "features" / "eeg").mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "splits").mkdir(parents=True, exist_ok=True)

    if args.copy_eeg:
        copy_public_eeg(records, dataset_root)

    fieldnames = [
        "sample_id",
        "split",
        "label",
        "raw_video_relpath",
        "subject_id",
        "eeg_source_relpath",
    ]
    write_csv(dataset_root / "annotations" / args.annotation_filename, fieldnames, records)

    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        write_csv(dataset_root / "splits" / f"{split}.csv", fieldnames, split_records)

    unmatched_videos = collect_unmatched_videos(dataset_root, records)
    write_csv(
        dataset_root / "annotations" / "unmatched_raw_videos.csv",
        ["sample_id", "raw_video_relpath"],
        unmatched_videos,
    )
    write_summary(dataset_root, records, unmatched_videos)

    print(f"Prepared {len(records)} samples from {raw_csv}")
    print(f"Annotation manifest: {dataset_root / 'annotations' / args.annotation_filename}")
    print(f"Split counts: {dict(Counter(record['split'] for record in records))}")
    print(f"Unmatched raw videos: {len(unmatched_videos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
