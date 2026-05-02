#!/bin/bash
# ============================================================
#  NeuroMM-2026 Baseline Batch Training
#  For each model, run 4 seeds in parallel on 4 GPUs.
# ============================================================

CONFIG="configs/train_eeg_only.yaml"
LR="5e-4"
BS=64
EPOCHS=30
SEEDS=(0 1 2 3)
GPUS=(0 1 2 3)

# ---- Models to run (from each .py file) ----
MODELS=(
  # # resnet_eeg.py
  # "legacy/resnet18_eeg"
  # "legacy/resnet50_eeg"
  # # cnn_eeg.py
  # "legacy/convnext_tiny_eeg"
  # "legacy/convnext_base_eeg"
  # "legacy/convnext_large_eeg"
  # "legacy/efficientnet_v2_s_eeg"
  # "legacy/mobilenet_v3_large_eeg"
  # "legacy/densenet121_eeg"
  # # eegnet.py
  # "legacy/eegnet"
  # "legacy/eegnet_m"
  # "legacy/eegnet_xl"
  # # rnn_eeg_big.py
  # "legacy/lstm_big_eeg"
  # "legacy/lstm_huge_eeg"
  # "legacy/gru_big_eeg"
  # # tcnet.py
  # "legacy/tcnet_eeg"
  # "legacy/tcnet_eeg_l"
  # actnet.py
  # "legacy/actnet_s"
  # "legacy/actnet_m"
  # "legacy/actnet_l"
)

# ---- Log directory ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG_DIR="neuromm26_results/logs/batch_${TIMESTAMP}"
mkdir -p "$BATCH_LOG_DIR"

TOTAL=${#MODELS[@]}
CURRENT=0
FAIL_COUNT=0
FAILED_RUNS=()

echo "============================================"
echo "  NeuroMM-2026 Baseline Batch Training"
echo "  Models : ${TOTAL}"
echo "  Seeds  : ${SEEDS[*]}"
echo "  GPUs   : ${GPUS[*]}"
echo "  LR=$LR  BS=$BS  Epochs=$EPOCHS"
echo "  Log dir: $BATCH_LOG_DIR"
echo "============================================"

for model in "${MODELS[@]}"; do
  CURRENT=$((CURRENT + 1))
  model_safe="${model//\//-}"

  echo ""
  echo "[$CURRENT/$TOTAL] $model"

  pids=()
  for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    gpu=${GPUS[$i]}
    run_log="${BATCH_LOG_DIR}/${model_safe}_seed${seed}.log"

    CUDA_VISIBLE_DEVICES=$gpu python -m neuromm26_baseline.tools.train \
      --config "$CONFIG" \
      --model-name "$model" \
      --seed "$seed" \
      --learning-rate "$LR" \
      --batch-size "$BS" \
      --epochs "$EPOCHS" \
      > "$run_log" 2>&1 &
    pids+=($!)
  done

  # Wait for all 4 seeds and report status
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}"
    ec=$?
    if [ $ec -ne 0 ]; then
      FAIL_COUNT=$((FAIL_COUNT + 1))
      FAILED_RUNS+=("${model} seed=${SEEDS[$j]}")
      echo "  [FAIL] seed=${SEEDS[$j]}  (exit=$ec)  -> ${BATCH_LOG_DIR}/${model_safe}_seed${SEEDS[$j]}.log"
    else
      echo "  [ OK ] seed=${SEEDS[$j]}"
    fi
  done
done

echo ""
echo "============================================"
echo "  Batch finished"
echo "  Succeeded: $(( TOTAL * ${#SEEDS[@]} - FAIL_COUNT )) / $(( TOTAL * ${#SEEDS[@]} ))"
if [ $FAIL_COUNT -gt 0 ]; then
  echo "  Failed ($FAIL_COUNT):"
  for f in "${FAILED_RUNS[@]}"; do
    echo "    - $f"
  done
fi
echo ""
echo "  Next step: aggregate results with"
echo "    python scripts/aggregate_results.py"
echo "============================================"
