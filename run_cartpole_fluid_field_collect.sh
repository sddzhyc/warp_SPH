#!/usr/bin/env bash
set -euo pipefail

# Batch collect fluid velocity fields for CartPole-on-water.
# Each run uses the same base command and only varies --t.

PYTHON_BIN="/d/SoftWare/python-3.9/python.exe"
SCRIPT_PATH="./train_cartpole_controller_diff.py"

SCENE_FILE="./data/scenes/carpole_on_water.json"
LR="0.001"
METHOD="1"
ITERS="100"
CONTROL_INTERVAL="25"
BACKWARD_INTERVAL="5"
MODEL_PATH="checkpoints/et1.0/difffr/g2/_normed/20260427-115330_lr_0.001/interval_5/no_t_max_loss_100/model_best_loss_1.420312e-04.pth"

# Repeat the whole T sweep this many times.
REPEAT_COUNT=3

# List the simulation times you want to sweep.
T_VALUES=(
  1.0
  2.0
  3.0
  4.0
  5.0
  7.0
  9.0
  10.0
)

LOG_DIR="./logs/fluid_field_collect"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SCENE_FILE" ]]; then
  echo "Scene file not found: $SCENE_FILE" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ${#T_VALUES[@]} -eq 0 ]]; then
  echo "T_VALUES is empty." >&2
  exit 1
fi

if [[ "$REPEAT_COUNT" -le 0 ]]; then
  echo "REPEAT_COUNT must be greater than 0." >&2
  exit 1
fi

TOTAL=${#T_VALUES[@]}
TOTAL_RUNS=$((TOTAL * REPEAT_COUNT * 2))
INDEX=0

for REPEAT_INDEX in $(seq 1 "$REPEAT_COUNT"); do
  for T_VALUE in "${T_VALUES[@]}"; do
    INDEX=$((INDEX + 1))
    RUN_TAG=$(date +%Y%m%d-%H%M%S)
    SAFE_T=${T_VALUE//./p}
    LOG_FILE_NORM="$LOG_DIR/run_${INDEX}_rep_${REPEAT_INDEX}_t_${SAFE_T}_${RUN_TAG}_norm_grad.log"
    LOG_FILE_PLAIN="$LOG_DIR/run_${INDEX}_rep_${REPEAT_INDEX}_t_${SAFE_T}_${RUN_TAG}_plain.log"

    CMD_NORM=(
      "$PYTHON_BIN" "$SCRIPT_PATH"
      --scene_file "$SCENE_FILE"
      --lr "$LR"
      --method "$METHOD"
      --iters "$ITERS"
      --difffr
      --norm_grad
      --t "$T_VALUE"
      --test_only
      --disable_ply_export
      --control_interval "$CONTROL_INTERVAL"
      --bi "$BACKWARD_INTERVAL"
      --model_path "$MODEL_PATH"
      --save_fluid_field
    )

    CMD_PLAIN=(
      "$PYTHON_BIN" "$SCRIPT_PATH"
      --scene_file "$SCENE_FILE"
      --lr "$LR"
      --method "$METHOD"
      --iters "$ITERS"
      --difffr
      --t "$T_VALUE"
      --test_only
      --disable_ply_export
      --control_interval "$CONTROL_INTERVAL"
      --bi "$BACKWARD_INTERVAL"
      --model_path "$MODEL_PATH"
      --save_fluid_field
    )

    echo "[$INDEX/$TOTAL_RUNS] Repeat $REPEAT_INDEX/$REPEAT_COUNT | Collecting fluid field for t=$T_VALUE with norm_grad"
    echo "Logging to: $LOG_FILE_NORM"
    "${CMD_NORM[@]}" 2>&1 | tee "$LOG_FILE_NORM"

    echo "[$INDEX/$TOTAL_RUNS] Repeat $REPEAT_INDEX/$REPEAT_COUNT | Collecting fluid field for t=$T_VALUE without norm_grad"
    echo "Logging to: $LOG_FILE_PLAIN"
    "${CMD_PLAIN[@]}" 2>&1 | tee "$LOG_FILE_PLAIN"
  done
done

echo "Fluid field collection complete. Total runs: $TOTAL_RUNS"