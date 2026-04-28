#!/usr/bin/env bash
set -euo pipefail

# Sweep training runs for cartpole controller.
# Default base command:
# python ./train_cartpole_controller_diff.py \
#   --scene_file ./data/scenes/carpole_on_water.json \
#   --lr 0.01 --method 1 --iters 25 --difffr --backward_interval 10
#
# Note: train_cartpole_controller_diff.py uses --iters (not --iter).

SCENE_FILE="./data/scenes/carpole_on_water.json"
METHOD="1"
ITERS="18"
ENABLE_DIFFFR="1"

# lr range: [LR_MIN, LR_MAX] with LR_STEP
LR_MIN="0.001"
LR_MAX="0.1"
LR_STEP="0.02"

# backward_interval range: [BI_MIN, BI_MAX] with BI_STEP
BI_MIN="1"
BI_MAX="10"
BI_STEP="2"

# Extra args appended to every run. Example:
# EXTRA_ARGS=(--device cuda:0 --seed 42)
EXTRA_ARGS=()

LOG_DIR="./runs/cartpole_sweep_logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SCENE_FILE" ]]; then
  echo "Scene file not found: $SCENE_FILE" >&2
  exit 1
fi

mapfile -t LR_VALUES < <(seq "$LR_MIN" "$LR_STEP" "$LR_MAX")
mapfile -t BI_VALUES < <(seq "$BI_MIN" "$BI_STEP" "$BI_MAX")

TOTAL=$(( ${#LR_VALUES[@]} * ${#BI_VALUES[@]} ))
if [[ "$TOTAL" -le 0 ]]; then
  echo "No runs generated. Check LR/BI range settings." >&2
  exit 1
fi

IDX=0
for LR in "${LR_VALUES[@]}"; do
  for BI in "${BI_VALUES[@]}"; do
    IDX=$((IDX + 1))
    TIME_TAG=$(date +%Y%m%d-%H%M%S)
    LOG_FILE="$LOG_DIR/run_${IDX}_lr_${LR}_bi_${BI}_${TIME_TAG}.log"

    CMD=(
      /d/SoftWare/python-3.9/python.exe ./train_cartpole_controller_diff.py
      --scene_file "$SCENE_FILE"
      --lr "$LR"
      --method "$METHOD"
      --t 2
      --iters "$ITERS"
      --backward_interval "$BI"
      --model_path  checkpoints/et1.0/difffr/g2_custom_const0.0/_normed/20260423-210839_lr_0.001/interval_10/no_t_max_loss_10/model_last_loss_7.433915e-04.pth
      --custom_grad
      --norm_grad
    )

    if [[ "$ENABLE_DIFFFR" == "1" ]]; then
      CMD+=(--difffr)
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "[$IDX/$TOTAL] lr=$LR backward_interval=$BI"
    echo "Logging to: $LOG_FILE"

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  done
done

echo "Sweep complete. Total runs: $TOTAL"
