#!/usr/bin/env bash
set -euo pipefail
# Grid search over vx/vy by repeatedly invoking run_simulation_diff.py.
# This script does not use Python-side --grid_search flags.
# It directly uses the fixed base command requested by the user.

# Base command (fixed):
# python ./run_simulation_diff.py --scene_file ./data/scenes/skip_stone_difffr_DFSPH.json \
#   --lr 0.2 --method 1 --reduce_lr --init_id 5 --use_difffr --train --iters 100

SCENE_FILE="./data/scenes/skip_stone_difffr_DFSPH.json"
LR="0.2"
METHOD="1"
INIT_ID="5"
ITERS="100"

# Feature flags used in command line (1=enabled, 0=disabled)
USE_REDUCE_LR="1"
USE_CUSTOM_GRAD="0"
USE_NORM_GRAD="1"
USE_DIFFFR="0"

# Grid configuration initialized from run_simulation_diff.py defaults:
#   vx_min=0, vx_max=30, vx_samples=4
#   vy_min=-30, vy_max=0, vy_samples=4
VX_MIN="0"
VX_MAX="30"
VX_SAMPLES="4"
VY_MIN="-30"
VY_MAX="0"
VY_SAMPLES="4"

# Derive step from min/max/samples to match Python-side default spacing.
VX_STEP=$(awk -v min="$VX_MIN" -v max="$VX_MAX" -v n="$VX_SAMPLES" 'BEGIN { if (n <= 1) print 1; else printf "%.10g", (max-min)/(n-1) }')
VY_STEP=$(awk -v min="$VY_MIN" -v max="$VY_MAX" -v n="$VY_SAMPLES" 'BEGIN { if (n <= 1) print 1; else printf "%.10g", (max-min)/(n-1) }')

if [[ ! -f "$SCENE_FILE" ]]; then
  echo "Scene file not found: $SCENE_FILE" >&2
  exit 1
fi

# timestamp for CSV filename
TIME_STR=$(date +%Y%m%d-%H%M%S)
OUT_DIR="grid_search_results"
mkdir -p "$OUT_DIR"

# Build filename tags from enabled command flags
NAME_TAGS=()
if [[ "$USE_REDUCE_LR" == "1" ]]; then NAME_TAGS+=("reduce_lr"); fi
if [[ "$USE_CUSTOM_GRAD" == "1" ]]; then NAME_TAGS+=("custom_grad"); fi
if [[ "$USE_NORM_GRAD" == "1" ]]; then NAME_TAGS+=("norm_grad"); fi
if [[ "$USE_DIFFFR" == "1" ]]; then NAME_TAGS+=("difffr"); fi

FLAGS_SUFFIX=""
if [[ ${#NAME_TAGS[@]} -gt 0 ]]; then
  FLAGS_SUFFIX="_$(IFS=_; echo "${NAME_TAGS[*]}")"
fi

CSV_PATH="$OUT_DIR/grid_results_iters${ITERS}_lr${LR}${FLAGS_SUFFIX}_${TIME_STR}.csv"
if [[ ! -f "$CSV_PATH" ]]; then
  echo "time,vx,vy,iters,lr,method,init_id,loss" > "$CSV_PATH"
fi

mapfile -t VX_VALUES < <(seq "$VX_MIN" "$VX_STEP" "$VX_MAX")
mapfile -t VY_VALUES < <(seq "$VY_MIN" "$VY_STEP" "$VY_MAX")

TOTAL=$(( ${#VX_VALUES[@]} * ${#VY_VALUES[@]} ))
if [[ "$TOTAL" -le 0 ]]; then
  echo "Grid is empty. Check min/max/step settings." >&2
  exit 1
fi

IDX=0
for VX in "${VX_VALUES[@]}"; do
  for VY in "${VY_VALUES[@]}"; do
    IDX=$((IDX + 1))
    echo "[$IDX/$TOTAL] Running sample with vx=$VX vy=$VY"

    # need to use /d/SoftWare/python-3.9/python.exe and capture FINAL_LOSS
    TMP_OUT=$(mktemp)
    /d/SoftWare/python-3.9/python.exe ./run_simulation_diff.py \
      --scene_file "$SCENE_FILE" \
      --lr "$LR" \
      --method "$METHOD" \
      $( [[ "$USE_REDUCE_LR" == "1" ]] && echo "--reduce_lr" ) \
      --init_id "$INIT_ID" \
      $( [[ "$USE_CUSTOM_GRAD" == "1" ]] && echo "--custom_grad" ) \
      $( [[ "$USE_NORM_GRAD" == "1" ]] && echo "--norm_grad" ) \
      $( [[ "$USE_DIFFFR" == "1" ]] && echo "--difffr" ) \
      --train \
      --iters "$ITERS" \
      --vx "$VX" \
      --vy "$VY" 2>&1 | tee "$TMP_OUT"

    # extract FINAL_LOSS printed by run_simulation_diff.py
    LOSS_LINE=$(grep -m1 "FINAL_LOSS:" "$TMP_OUT" || true)
    LOSS=""
    if [[ -n "$LOSS_LINE" ]]; then
      LOSS=$(echo "$LOSS_LINE" | awk -F":" '{print $2}' | tr -d '[:space:]')
    else
      LOSS="NaN"
    fi
    rm -f "$TMP_OUT"

    # append record to CSV
    echo "$(date +%Y%m%d-%H%M%S),$VX,$VY,$ITERS,$LR,$METHOD,$INIT_ID,$LOSS" >> "$CSV_PATH"
  done
done

echo "Grid search finished. Total runs: $TOTAL"
