#!/usr/bin/env bash
# Sequential training script: trains controller with T=3, T=5, T=7
# Each run uses the best model from the previous run as initialization.
# LR is halved after each run (starting from 0.0001).

set -euo pipefail

# ---- Fixed arguments ----
SCENE_FILE="./data/scenes/fish_on_water_new.json"
METHOD=1
ITERS=100
CONTROL_INTERVAL=50
BACKWARD_INTERVAL=15
ENCODER_PATH="fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth"

# ---- Adjustable parameters ----
INITIAL_LR=0.0001
TIME_VALUES=(3 5 7)

# ---- State ----
MODEL_PATH=""
LR="$INITIAL_LR"
TEMP_FILE="train_output_tmp.txt"
COMMANDS=()

for T in "${TIME_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "  Training with T=$T, LR=$LR"
    echo "  Model path: ${MODEL_PATH:-None}"
    echo "========================================"

    # Build argument list
    ARGS=(
        ./train_controller_diff.py
        --scene_file "$SCENE_FILE"
        --lr "$LR"
        --method "$METHOD"
        --iters "$ITERS"
        --difffr
        --norm_grad
        --custom_grad
        --t "$T"
        --control_interval "$CONTROL_INTERVAL"
        --bi "$BACKWARD_INTERVAL"
        --use_fluid
        --encoder_path "$ENCODER_PATH"
        --random_start
        --use_relative_state
        --time_in_state
        --add_gru
        --use_v_loss
        --sine_action
    )

    if [ -n "$MODEL_PATH" ]; then
        ARGS+=(--model_path "$MODEL_PATH")
    fi

    # Build and record the full command string
    FULL_CMD="/d/SoftWare/python-3.9/python.exe ${ARGS[*]}"
    COMMANDS+=("$FULL_CMD")

    # Run training, capturing output to both console and temp file
    /d/SoftWare/python-3.9/python.exe "${ARGS[@]}" 2>&1 | tee "$TEMP_FILE"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "ERROR: Training failed with exit code ${PIPESTATUS[0]}"
        exit 1
    fi

    # Parse the [BEST_MODEL] line from output
    MODEL_PATH=$(grep -oP '\[BEST_MODEL\]\s+\K.+' "$TEMP_FILE" | tail -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Could not find [BEST_MODEL] in training output!"
        exit 1
    fi

    echo ""
    echo ">>> Best model: $MODEL_PATH"

    # Halve LR for next run
    LR=$(/d/SoftWare/python-3.9/python.exe -c "print($LR * 0.5)")
done

# Clean up temp file
rm -f "$TEMP_FILE"

echo ""
echo "========================================"
echo "  All training done!"
echo "  Final best model: $MODEL_PATH"
echo "========================================"
echo ""
echo "  Executed commands:"
echo "----------------------------------------"
for i in "${!COMMANDS[@]}"; do
    echo "  [$((i+1))]  ${COMMANDS[$i]}"
done
echo "----------------------------------------"
