# NeuroMM-2026 Baseline

Official baseline code for the **NeuroMM-2026 multimodal seizure detection challenge** ([2026.neuromm.org](https://2026.neuromm.org)).

The challenge has three tasks evaluated on a private held-out test set:

| Task | Modalities | Type | Selection metric |
|---|---|---|---|
| **Task 1** | EEG only | Binary spike vs non-spike | AUPRC |
| **Task 2** | EEG + Video | Binary spike vs non-spike | AUPRC |
| **Task 3** | EEG + Video | 5-class seizure subtype on positives only | weighted F1 |

This repository contains:
- Data loaders for Task 1 / Task 2 (binary) and Task 3 (5-class, positives only)
- 30+ EEG encoders (TCN, EEGNet, ResNet/ConvNeXt/EfficientNet/MobileNet/DenseNet/ViT, LSTM/GRU, ACTNet, LMDA, plus self-supervised CBraMod / LaBraM / EEG-DINO)
- Multi-modal late-fusion models (binary and 5-class)
- 7 video feature backbones via the same extractor (CLIP, VideoMAE-base/large, DINOv2-base/large, SigLIP, TimeSformer)
<!-- - Audio (OpenSMILE) and face (OpenFace skeleton) feature extractors -->
- Trainers + per-epoch logger + per-seed mean/std aggregator

---

## Setup

```bash
git clone git@github.com:NeuroMM-Org/NeuroMM-2026_Baseline.git
cd NeuroMM-2026_Baseline
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` if you want to override paths.

## Get the data

The train + val data and pre-extracted visual features are released as a **gated** HuggingFace dataset:

> [https://huggingface.co/datasets/NeuroMM/NeuroMM-2026](https://huggingface.co/datasets/NeuroMM/NeuroMM-2026)

After your team's gated request is approved:

```bash
huggingface-cli login   # use your own approved token
huggingface-cli download NeuroMM/NeuroMM-2026 --repo-type=dataset --local-dir ./neuromm26_data

# extract the archives into the directory layout the code expects
cd neuromm26_data
tar -xf archives/eeg.tar             # → processed/features/eeg/
tar -xf archives/video_clip-base.tar # → processed/features/video/clip-base/
# ...same for the other 6 video features you want to use
```

Place the extracted `processed/` directory at:
```
neuromm26_datasets/processed/
```
(or symlink, or override via `--eeg-feature-root` / `--video-feature-root` CLI flags).

The shipped manifest `neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv` already has the patient-disjoint train/val split:

| split | rows | unique patients | positive ratio |
|---|---:|---:|---:|
| train | 20,298 | 67 | 9.6 % |
| val | 5,128 | 17 | 9.6 % |

---

## Training tutorials

All trainers default to `lr=5e-4`, `batch_size=64`, `epochs=30`. Each writes to:
```
neuromm26_results/checkpoints/<exp_name>/best.pt        ← best by val metric
neuromm26_results/metrics/<exp_name>_train_summary.json ← per-epoch + best val metrics
neuromm26_results/logs/<exp_name>.log                   ← human-readable log
```

### Task 1 — EEG-only (binary)

Train one EEG model on one seed:
```bash
python -m neuromm26_baseline.tools.train \
    --config configs/train_eeg_only.yaml \
    --model-name legacy/tcnet_eeg \
    --seed 0 --learning-rate 5e-4 --batch-size 64 --epochs 30
```

Available `--model-name` values include:
```
legacy/tcnet_eeg               legacy/eegnet                  legacy/eegnet_xl
legacy/resnet18_eeg            legacy/resnet50_eeg            legacy/densenet121_eeg
legacy/convnext_tiny_eeg       legacy/convnext_base_eeg       legacy/convnext_large_eeg
legacy/efficientnet_v2_s_eeg   legacy/mobilenet_v3_large_eeg  legacy/lmda_eeg
legacy/actnet_s                legacy/actnet_m                legacy/lstm_big_eeg
legacy/lstm_huge_eeg           legacy/gru_big_eeg             legacy/cbramod
legacy/labram_base             legacy/eeg_dino_s              legacy/eeg_dino_m
legacy/eeg_dino_l              legacy/vit_small_eeg           legacy/vit_base_eeg
legacy/vit_base_384_eeg        legacy/tcnet_eeg_l             ...
```
List the full set with `python -m neuromm26_baseline.tools.train --list-models`.

Sweep 4 seeds (single GPU, sequential):
```bash
for seed in 0 1 2 3; do
  python -m neuromm26_baseline.tools.train \
      --config configs/train_eeg_only.yaml \
      --model-name legacy/tcnet_eeg \
      --seed $seed --learning-rate 5e-4
done
```

Sweep 4 seeds in parallel on 4 GPUs:
```bash
for seed in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$seed python -m neuromm26_baseline.tools.train \
      --config configs/train_eeg_only.yaml \
      --model-name legacy/tcnet_eeg \
      --seed $seed --learning-rate 5e-4 &
done
wait
```

### Task 2 — EEG + Video late fusion (binary)

```bash
python -m neuromm26_baseline.tools.train_eeg_video_fusion \
    --eeg-model tcnet_eeg \
    --video-feature-name dinov2-base \
    --seed 0 --learning-rate 5e-4 --epochs 30
```

`--video-feature-name` is the subdirectory under `neuromm26_datasets/processed/features/video/` — use any of:
`clip-base`, `videomae-base`, `videomae-large`, `dinov2-base`, `dinov2-large`, `siglip-base`, `timesformer-k400`.

Pure video-MLP baseline (no EEG branch):
```bash
python -m neuromm26_baseline.tools.train_video_feature_mlp \
    --feature-name dinov2-base \
    --seed 0 --learning-rate 1e-3 --epochs 30
```

### Task 3 — 5-class seizure subtype (positives only)

Train+val automatically filter to `label_type ∈ {1,2,3,4,5}` (≈ 2,022 train / 492 val). Selection metric is **weighted F1**.

EEG-only:
```bash
python -m neuromm26_baseline.tools.train_task3_eeg \
    --eeg-model tcnet_eeg \
    --seed 0 --learning-rate 5e-4 --epochs 30
```

Video-only MLP:
```bash
python -m neuromm26_baseline.tools.train_task3_video_mlp \
    --feature-name dinov2-base \
    --seed 0 --learning-rate 5e-4 --epochs 30
```

EEG + Video late fusion:
```bash
python -m neuromm26_baseline.tools.train_task3_fusion \
    --eeg-model tcnet_eeg \
    --video-feature-name dinov2-base \
    --seed 0 --learning-rate 5e-4 --epochs 30
```

Add `--use-class-weights` to enable inverse-frequency class weighting (helps minority classes 4, 5).

---

## Evaluation tutorial

Every trainer **automatically evaluates on val each epoch** and saves the best-val checkpoint. Re-running val evaluation on a saved checkpoint:

```bash
python -m neuromm26_baseline.tools.eval \
    --config configs/eval_eeg_only.yaml \
    --checkpoint neuromm26_results/checkpoints/<exp_name>/best.pt \
    --split val
```

Outputs `neuromm26_results/metrics/<exp_name>_val.json` and `predictions/<exp_name>_val.csv`.

> Test evaluation uses a private test manifest and the official organizer's leaderboard. The trainers also accept `--test-manifest <path>` and `--test-split <name>` to optionally evaluate on any held-out CSV you provide; if absent, only val metrics are produced.

---

## Generating a submission (candidate set)

The official test is run as a **result-only** competition on Codabench. The
organizers release a **candidate set** separately, laid out as:

```
<candidate-dir>/
├── candidate_ids.txt              # one id per line
├── eeg/<id>.npy                   # (29, 2000) float32        (Track 1/2)
└── video/<backbone>/<id>.npy      # pre-extracted features    (Track 2/3)
```

Run your trained checkpoint over **every** id in `candidate_ids.txt` to
produce `submission.csv` (`sample_id,prediction`), then zip it as
`sample_submission.zip` (the ZIP must contain exactly one file named
`submission.csv`) and upload to the matching Codabench competition. Only the
official samples are scored; predicting every id is required.

### Track 1 — EEG-only (binary, AUPRC)

```bash
python -m neuromm26_baseline.tools.predict_candidate_test1 \
    --checkpoint neuromm26_results/checkpoints/<exp>/best.pt \
    --config     neuromm26_results/metrics/<exp>_config.json \
    --candidate-dir /path/to/candidate_set \
    --out submission.csv
```

### Track 2 — EEG + Video late fusion (binary, AUPRC)

```bash
python -m neuromm26_baseline.tools.predict_candidate_test2 \
    --checkpoint neuromm26_results/checkpoints/<eeg_video_fusion_exp>/best.pt \
    --candidate-dir /path/to/candidate_set \
    --out submission.csv
```
The EEG model and video backbone are read from the checkpoint; override with
`--video-feature-name <name>` if needed.

### Track 3 — 5-class seizure subtype (weighted F1)

```bash
python -m neuromm26_baseline.tools.predict_candidate_test3 \
    --checkpoint neuromm26_results/checkpoints/<task3_exp>/best.pt \
    --candidate-dir /path/to/candidate_set \
    --out submission.csv
```
Auto-detects EEG-only (`train_task3_eeg.py`) vs EEG+Video fusion
(`train_task3_fusion.py`) from the checkpoint. `prediction` is the predicted
seizure subtype in `{1,2,3,4,5}`.

`--device cpu` works but is slow; predictions are batch-independent
(`model.eval()` + `no_grad()`), so a larger `--batch-size` only changes speed.

---

## Aggregation tutorial — view scores

After running multiple seeds for the same model, summarize mean ± std:

### Task 1 / Task 2 (binary)

```bash
python scripts/aggregate_results.py
```
Reads every `*_train_summary.json` under `neuromm26_results/metrics/` and prints a table sorted by val AUPRC, columns:
```
Model | LR | N | auprc | binary_f1 | macro_f1 | weighted_f1 | accuracy | balanced_accuracy
```
Also writes `neuromm26_results/metrics/baseline_summary.csv`.

Filter only certain runs:
```bash
python scripts/aggregate_results.py --seeds 0 1 2 3
```

### Task 3 (5-class)

```bash
python scripts/aggregate_task3_results.py --kind val
```
Prints val table sorted by weighted F1, columns:
```
Model | LR | N | weighted_f1 | macro_f1 | accuracy | macro_precision | macro_recall
```

---

## Code layout

```
neuromm26_baseline/
├── datasets/                      # 4 dataset classes (binary EEG/multimodal, task3 EEG/multimodal/video)
├── models/
│   ├── backbone/                  # EEG ResNet encoder + video projector
│   ├── fusion/                    # ConcatFusion / cross-attention / temporal
│   ├── heads/                     # binary + multiclass classification heads
│   ├── legacy/                    # 30+ EEG encoders + registry + task3 num_classes builder
│   ├── multimodal_baseline.py     # binary EEG+Video fusion (built-in EEGResNetEncoder)
│   ├── multimodal_late_fusion.py  # binary late fusion w/ any legacy EEG model
│   └── multiclass_late_fusion.py  # 5-class late fusion (Task 3)
├── trainers/                      # train/eval loops
├── utils/                         # config, logger, metrics (binary + multiclass), seed
├── feature_extract/               # EEG/ECG/EMG/video/audio/face feature extractors
└── tools/
    ├── train.py                   # binary trainer (Task 1)
    ├── train_eeg_video_fusion.py  # binary fusion trainer (Task 2)
    ├── train_video_feature_mlp.py # video-only MLP baseline
    ├── train_task3_eeg.py         # Task 3 EEG-only
    ├── train_task3_video_mlp.py   # Task 3 video MLP
    ├── train_task3_fusion.py      # Task 3 fusion
    ├── eval.py                    # standalone evaluator
    ├── extract_features.py        # CLI for feature extraction
    ├── predict_candidate_test1.py # candidate-set submission (Track 1)
    ├── predict_candidate_test2.py # candidate-set submission (Track 2)
    └── predict_candidate_test3.py # candidate-set submission (Track 3)

configs/                            # YAML configs (default + task-specific)
scripts/                            # aggregators + helper shell wrappers
neuromm26_datasets/annotations/     # train+val manifest (no test)
```

---

## Optional — explore the codebase with Understand Anything

This repository ships a pre-built knowledge graph under `.understand-anything/`. [Understand Anything](https://github.com/Lum1104/Understand-Anything) is an AI-agent skill that turns the graph into an interactive web dashboard, a guided tour through the architecture, and a chat interface for asking questions about the code. It works across multiple AI coding agents (Claude Code, Copilot CLI, Gemini CLI, …) — see the Understand Anything README for install and usage in your environment.

---

## Citation

```bibtex
@article{lin2025eeg,
  title={An EEG dataset for interictal epileptiform discharge with spatial distribution information},
  author={Lin, Nan and Zheng, Mengxuan and Li, Lian and Hu, Peng and Gao, Weifang and Sun, Heyang and Xu, Chang and Yuan, Gonglin and Liang, Zi and Dong, Yisu and others},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={229},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## License

Code: see `LICENSE`.
Dataset: CC BY-NC 4.0 — academic research and NeuroMM-2026 challenge participation only.
