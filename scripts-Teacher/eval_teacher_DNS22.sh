#!/bin/bash

# Load environment variables (including INPUT_DIR_DNS22)
source ./path_config.sh

# Check if exactly one argument (GPU_ID) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GPU_ID"
    exit 1
fi

# Set GPU ID
GPU_ID="$1"

# Use INPUT_DIR_DNS22 environment variable as input directory
DEFAULT_INPUT_DIR="$INPUT_DIR_DNS22"

# Verify INPUT_DIR_DNS22 is set and directory exists
if [ -z "$DEFAULT_INPUT_DIR" ] || [ ! -d "$DEFAULT_INPUT_DIR" ]; then
    echo "❌ Error: INPUT_DIR_DNS22 is not set or directory does not exist."
    exit 1
fi

# Default values for checkpoints and task names
CHECKPOINTs=(
    # "./logs/Teacher_model/vcrsnu58/last.ckpt"
    "../sgmse/ckpt/train_vb_29nqe0uh_epoch=115.ckpt"
)

task_save_names=(
    # "teacher_DNS"
    "e115_DNS22"
)

# Define list of N values
N_list=(1 3 5 10 20 30)
N_list=(30)

# Function to run enhancement and evaluation
enhance_and_evaluate() {
    local N=$1
    local CHECKPOINT=$2
    local task_save_name=$3
    local OUTPUT_DIR="./out/$task_save_name"
    local ENHANCED_DIR="$OUTPUT_DIR/N_$N"

    mkdir -p "$ENHANCED_DIR"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    # Run the enhancement command
    ENHANCEMENT_CMD="python3 enhancement.py --test_dir \"$DEFAULT_INPUT_DIR\" --enhanced_dir \"$ENHANCED_DIR\" --ckpt \"$CHECKPOINT\" --N \"$N\""
    if ! eval $ENHANCEMENT_CMD; then
        echo "❌ Error during enhancement: $task_save_name, N=$N"
        exit 1
    fi
    echo "✅ Enhancement done: $task_save_name, N=$N"
}

# Main loop through checkpoints and tasks
for idx in "${!CHECKPOINTs[@]}"; do
    CHECKPOINT=${CHECKPOINTs[$idx]}
    task_save_name=${task_save_names[$idx]}

    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "❌ Checkpoint not found: $CHECKPOINT"
        exit 1
    fi

    for N in "${N_list[@]}"; do
        enhance_and_evaluate "$N" "$CHECKPOINT" "$task_save_name"
    done
done

echo "🎉 All tasks completed."
