#!/usr/bin/env bash
set -euo pipefail

# Progressive training over a manual list of t values.
# It starts from t=1 (or any manual list you provide), then uses each stage's
# best checkpoint as --model_path for the next stage.

PYTHON_BIN="/d/SoftWare/python-3.9/python.exe"
SCENE_FILE="./data/scenes/carpole_on_water.json"
TRAIN_SCRIPT="./train_cartpole_controller_diff.py"
ENCODER_PATH="fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth"

# Manual t schedule (edit this list as needed).
T_VALUES=(2 3 5 7 8 9 10)
# Optional: provide an initial model for the first stage.
# Leave empty to start stage-1 without --model_path.
INITIAL_MODEL_PATH="checkpoints/et1.0/difffr/g2_custom_const0.0/_normed/20260506-133351_lr_0.0001/interval_5/time_in_state_max_loss_100/model_best_loss_6.127490e-05.pth"

# Fixed training args matching your command.
COMMON_ARGS=(
  --scene_file "$SCENE_FILE"
  --lr 0.0001
  --method 1
  --iters 100
  --difffr
  --norm_grad
  --custom_grad
  --control_interval 50
  --bi 5
  --use_fluid
  --encoder_path "$ENCODER_PATH"
)


LOG_DIR="./runs/progressive_t_logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$SCENE_FILE" ]]; then
  echo "Scene file not found: $SCENE_FILE" >&2
  exit 1
fi

if [[ ! -f "$ENCODER_PATH" ]]; then
  echo "Encoder checkpoint not found: $ENCODER_PATH" >&2
  exit 1
fi

if [[ ${#T_VALUES[@]} -eq 0 ]]; then
  echo "T_VALUES is empty. Please specify at least one t." >&2
  exit 1
fi

CURRENT_MODEL_PATH="$INITIAL_MODEL_PATH"

for T in "${T_VALUES[@]}"; do
  TIME_TAG=$(date +%Y%m%d-%H%M%S)
  LOG_FILE="$LOG_DIR/t_${T}_${TIME_TAG}.log"

  CMD=(
    "$PYTHON_BIN" "$TRAIN_SCRIPT"
    "${COMMON_ARGS[@]}"
    --t "$T"
  )

  if [[ -n "$CURRENT_MODEL_PATH" ]]; then
    if [[ ! -f "$CURRENT_MODEL_PATH" ]]; then
      echo "Current model_path does not exist: $CURRENT_MODEL_PATH" >&2
      exit 1
    fi
    CMD+=(--model_path "$CURRENT_MODEL_PATH")
  fi

  echo "============================================================"
  echo "[progressive-t] Running stage t=$T"
  echo "[progressive-t] Log: $LOG_FILE"
  echo "[progressive-t] model_path: ${CURRENT_MODEL_PATH:-<none>}"
  echo "============================================================"

  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

  CKPT_DIR_LINE=$(grep -F "Checkpoint directory:" "$LOG_FILE" | tail -n1 || true)
  if [[ -z "$CKPT_DIR_LINE" ]]; then
    echo "Failed to parse checkpoint directory from log: $LOG_FILE" >&2
    exit 1
  fi

  CKPT_DIR=${CKPT_DIR_LINE#*Checkpoint directory: }
  CKPT_DIR=${CKPT_DIR//$'\r'/}
  CKPT_DIR=${CKPT_DIR//\\//}

  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint directory not found: $CKPT_DIR" >&2
    exit 1
  fi

  NEXT_MODEL_PATH=$(ls -1t "$CKPT_DIR"/model_best_*.pth 2>/dev/null | head -n1 || true)
  if [[ -z "$NEXT_MODEL_PATH" ]]; then
    echo "No model_best_*.pth found under: $CKPT_DIR" >&2
    exit 1
  fi

  echo "[progressive-t] best model for t=$T: $NEXT_MODEL_PATH"
  CURRENT_MODEL_PATH="$NEXT_MODEL_PATH"
done

echo "============================================================"
echo "Progressive t training finished."
echo "Final propagated model_path: $CURRENT_MODEL_PATH"
echo "============================================================"
