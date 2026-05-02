# Data Format

## Release-Safe Layout

The public baseline should be built against processed artifacts rather than the raw video directory.

- EEG windows: `neuromm26_datasets/processed/features/eeg/{sample_id}.npy`
- Video features: `neuromm26_datasets/processed/features/{video_feature_name}/{sample_id}.npy`
- Full normalized manifest: `neuromm26_datasets/annotations/vepiset_spike_sleep_video.relative.csv`
- Split manifests: `neuromm26_datasets/splits/{train,val,test}.csv`

## Manifest Columns

- `sample_id`: unique sample/window id
- `split`: `train`, `val`, or future `test`
- `label`: target label, blank for public test data
- `raw_video_relpath`: relative path to raw video, used internally for feature extraction only
- `subject_id`: subject/session identifier parsed from the raw video path
- `eeg_source_relpath`: original raw EEG path for traceability

## Runtime Path Resolution

The manifest no longer stores feature paths. Runtime code resolves them from configuration.

- EEG path: `{eeg_feature_root}/{sample_id}.npy`
- Video feature path: `{video_feature_root}/{video_feature_name}/{sample_id}.npy`

This keeps the manifest stable while allowing you to switch between `clip`, `videomae`, and future backbones by only changing config or environment variables.
