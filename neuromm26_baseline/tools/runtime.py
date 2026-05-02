"""Shared runtime helpers for train/eval/infer entrypoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import EEGFeatureDataset, NeuroMMMultimodalFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.models import (
    EEGBaselineModel,
    EEGVideoFusionModel,
    build_legacy_eeg_model,
    is_legacy_eeg_model_name,
    normalize_model_name,
)
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.seed import build_torch_generator, seed_worker


DEFAULT_ANNOTATION_MANIFEST = "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_split_csv(dataset_root: str, split: str) -> str:
    return str(Path(dataset_root) / "splits" / f"{split}.csv")


def resolve_annotation_manifest(config: dict[str, Any]) -> str | None:
    paths_cfg = config["paths"]
    manifest_path = paths_cfg.get("annotation_manifest") or DEFAULT_ANNOTATION_MANIFEST
    path = Path(manifest_path)
    return str(path) if path.exists() else None


def build_dataset(config: dict[str, Any], split: str):
    dataset_cfg = config["dataset"]
    paths_cfg = config["paths"]
    runtime_cfg = config["runtime"]
    task_type = config["model"]["task_type"]
    split_csv = resolve_split_csv(paths_cfg["dataset_root"], split)
    manifest_csv = resolve_annotation_manifest(config)
    target_shape = tuple(dataset_cfg.get("eeg_shape", [26, 2000]))

    if task_type == "eeg_only":
        return EEGFeatureDataset(
            split_csv=None if manifest_csv else split_csv,
            manifest_csv=manifest_csv,
            split=split,
            eeg_feature_root=paths_cfg["eeg_feature_root"],
            preload_in_memory=runtime_cfg.get("preload_in_memory", True),
            target_shape=target_shape,
        )
    if task_type == "eeg_video_fusion":
        return NeuroMMMultimodalFeatureDataset(
            split_csv=None if manifest_csv else split_csv,
            manifest_csv=manifest_csv,
            split=split,
            eeg_feature_root=paths_cfg["eeg_feature_root"],
            video_feature_root=paths_cfg["video_feature_root"],
            video_feature_name=dataset_cfg["video_feature_name"],
            preload_in_memory=runtime_cfg.get("preload_in_memory", True),
            target_shape=target_shape,
            require_video_feature=dataset_cfg.get("require_video_feature", True),
        )
    raise ValueError(f"Unsupported task_type: {task_type}")


def build_dataloader(config: dict[str, Any], dataset, batch_size: int, shuffle: bool) -> DataLoader:
    runtime_cfg = config["runtime"]
    seed = int(config.get("seed", 42)) + (1 if shuffle else 0)
    num_workers = runtime_cfg.get("num_workers", 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=runtime_cfg.get("pin_memory", False),
        collate_fn=neuromm_collate,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=build_torch_generator(seed),
    )


def build_model(config: dict[str, Any]):
    model_cfg = config["model"]
    model_name = normalize_model_name(model_cfg.get("name"))
    if model_name is not None:
        model_cfg["name"] = model_name

    if model_name is not None and model_name.startswith("legacy/"):
        legacy_name = model_name.split("/", 1)[1]
        if is_legacy_eeg_model_name(legacy_name):
            return build_legacy_eeg_model(legacy_name)

    if model_name == "baseline/resnet18":
        return EEGBaselineModel(
            backbone="resnet18",
            pretrained=model_cfg.get("eeg_pretrained", False),
            temporal_pool=model_cfg.get("eeg_temporal_pool", 10),
            dropout=model_cfg.get("eeg_dropout", 0.2),
        )

    if model_name == "baseline/resnet50":
        return EEGBaselineModel(
            backbone="resnet50",
            pretrained=model_cfg.get("eeg_pretrained", False),
            temporal_pool=model_cfg.get("eeg_temporal_pool", 10),
            dropout=model_cfg.get("eeg_dropout", 0.2),
        )

    if model_name in {None, "baseline/eeg"} and model_cfg["task_type"] == "eeg_only":
        return EEGBaselineModel(
            backbone=model_cfg.get("eeg_backbone", "resnet18"),
            pretrained=model_cfg.get("eeg_pretrained", False),
            temporal_pool=model_cfg.get("eeg_temporal_pool", 10),
            dropout=model_cfg.get("eeg_dropout", 0.2),
        )

    if model_name == "multimodal/eeg_video_fusion" or model_cfg["task_type"] == "eeg_video_fusion":
        return EEGVideoFusionModel(
            eeg_backbone=model_cfg.get("eeg_backbone", "resnet18"),
            eeg_pretrained=model_cfg.get("eeg_pretrained", False),
            eeg_temporal_pool=model_cfg.get("eeg_temporal_pool", 10),
            eeg_dropout=model_cfg.get("eeg_dropout", 0.2),
            video_embedding_dim=model_cfg.get("video_embedding_dim", 256),
            fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 256),
            classifier_dropout=model_cfg.get("classifier_dropout", 0.2),
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def build_criterion(config: dict[str, Any], labels: list[int | None] | None = None):
    train_cfg = config["train"]
    pos_weight_value = train_cfg.get("positive_class_weight")
    if pos_weight_value is None and labels is not None:
        valid_labels = [label for label in labels if label is not None]
        pos_count = sum(valid_labels)
        neg_count = len(valid_labels) - pos_count
        if pos_count > 0:
            pos_weight_value = neg_count / max(pos_count, 1)
    if pos_weight_value is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight_value)]))
    return nn.BCEWithLogitsLoss()


def save_json(data: dict[str, Any], path: str) -> str:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return str(output_path)


def save_prediction_csv(sample_ids: list[str], probabilities: list[float], labels: list[float] | None, path: str) -> str:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        headers = ["sample_id", "probability"]
        if labels is not None:
            headers.append("label")
        writer.writerow(headers)
        for idx, sample_id in enumerate(sample_ids):
            row = [sample_id, probabilities[idx]]
            if labels is not None:
                row.append(labels[idx])
            writer.writerow(row)
    return str(output_path)
