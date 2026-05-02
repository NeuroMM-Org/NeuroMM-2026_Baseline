# Project Structure

## Data Organization

- `neuromm26_datasets/raw/eeg_videos/`: private source package, not for open release
- `neuromm26_datasets/processed/features/eeg/`: public EEG `.npy` windows copied from raw
- `neuromm26_datasets/processed/features/video/{feature_name}/`: public video feature directories
- `neuromm26_datasets/annotations/`: normalized manifests and dataset diagnostics
- `neuromm26_datasets/splits/`: train/val/test manifests used by baseline loaders

## Loader Contract

Baseline code should load split manifests from `neuromm26_datasets/splits/` and select the visual feature backbone by `video_feature_name`. This keeps CLI and training code stable while allowing new feature extractors to write into `processed/features/video/<feature_name>/`.
