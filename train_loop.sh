#!/bin/bash

PYTHON_PATH="/d/SoftWare/python-3.9/python.exe"

# Training parameters
base_t=8
t_increment=3
base_lr=0.00001
lr_factor=0.5
max_t=50
scene_file="./data/scenes/fish_on_water.json"
method=1
iters=100
control_interval=50
bi=15
encoder_path="fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth"

prev_model="checkpoints/et5.0/difffr/g2_custom_const0.0/_normed/20260602-231155_lr_1e-05/interval_15/no_t_max_loss_100/model_best_loss_3.793505e-04.pth"

iteration=0
for ((t=base_t; t<=max_t; t+=t_increment)); do
    current_lr=$($PYTHON_PATH -c "print(${base_lr} * (${lr_factor} ** ${iteration}))")

    cmd="$PYTHON_PATH ./train_controller_diff.py \
        --scene_file ${scene_file} \
        --lr ${current_lr} \
        --method ${method} \
        --iters ${iters} \
        --difffr \
        --norm_grad \
        --custom_grad \
        --t ${t} \
        --control_interval ${control_interval} \
        --bi ${bi} \
        --use_fluid \
        --encoder_path ${encoder_path} \
        --random_start \
        --use_relative_state"

    if [ -n "$prev_model" ]; then
        cmd="$cmd --resume $prev_model"
    fi

    echo "========================================="
    echo "Training with t=$t, lr=$current_lr"
    echo "Command: $cmd"
    echo "========================================="

    $cmd

    # Find best model from this round
    best_model=$(find . -name "*model_best*.pth" -type f -newer "checkpoint_temp_marker" 2>/dev/null | sort -r | head -1)

    if [ -n "$best_model" ]; then
        prev_model="$best_model"
        echo "Best model from this round: $prev_model"
    fi

    # Create marker for finding new files next iteration
    touch checkpoint_temp_marker

    ((iteration++))
done

echo "Training complete!"
