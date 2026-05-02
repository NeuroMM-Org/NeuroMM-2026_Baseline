#!/usr/bin/env bash
python -m neuromm26_baseline.tools.prepare_dataset \
  --dataset-root neuromm26_datasets \
  --raw-subdir raw/eeg_videos \
  --copy-eeg \
  "$@"
