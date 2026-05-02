# Dataset directory

Download train+val features from HuggingFace:

  https://huggingface.co/datasets/NeuroMM/NeuroMM-2026

Extract such that this directory looks like:

```
neuromm26_datasets/
├── annotations/
│   └── neuromm2026_train_val_patient_split.csv     # included in repo
└── processed/
    └── features/
        ├── eeg/{sample_id}.npy                     # 25,426 EEG features
        └── video/{feature_name}/{sample_id}.npy    # 25,426 per video feature
```

Test data is private and provided to participants only after challenge registration.
