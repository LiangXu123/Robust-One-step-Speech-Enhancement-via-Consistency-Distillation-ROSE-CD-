#!/bin/bash

# Load environment variables (including NOISY_DIR_TIMIT)
source ./path_config.sh

# Check if exactly one argument (GPU_ID) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "Only one parameter is required: the single GPU ID."
    exit 1
fi

# Set GPU ID
GPU_ID="$1"  # Single GPU ID

# Use NOISY_DIR_TIMIT from path_config.sh as input directory
DEFAULT_INPUT_DIR="$NOISY_DIR_TIMIT"

# Verify that NOISY_DIR_TIMIT is set and directory exists
if [ -z "$DEFAULT_INPUT_DIR" ] || [ ! -d "$DEFAULT_INPUT_DIR" ]; then
    echo "Error: NOISY_DIR_TIMIT is not set or directory does not exist."
    exit 1
fi

# Default values for checkpoints and task names
CHECKPOINTs=(
    "./logs/Teacher_model/vcrsnu58/last.ckpt"
)

task_save_names=(
    "teacher_TIMIT"
)

# Assert all checkpoint files exist
for CHECKPOINT in "${CHECKPOINTs[@]}"; do
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file '$CHECKPOINT' does not exist!"
        exit 1
    fi
done

# Define list of N values
N_list=(1 3 5 10 20 30)

# Function to run enhancement and metrics calculation
enhance_and_evaluate() {
    local N=$1
    local CHECKPOINT=$2
    local task_save_name=$3
    local OUTPUT_DIR="./out/$task_save_name"

    # Create base output directory
    if ! mkdir -p "$OUTPUT_DIR"; then
        echo "Error: Failed to create output directory: $OUTPUT_DIR"
        exit 1
    fi

    local ENHANCED_DIR="$OUTPUT_DIR/N_$N"
    
    # Create enhanced directory for this N value
    if ! mkdir -p "$ENHANCED_DIR"; then
        echo "Error: Failed to create enhanced directory: $ENHANCED_DIR"
        exit 1
    fi
    
    echo "Processing N: $N, Checkpoint: $CHECKPOINT, Task: $task_save_name on GPU $GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    # Construct and execute the enhancement script command
    ENHANCEMENT_CMD="python3 enhancement.py --test_dir \"$DEFAULT_INPUT_DIR\" --enhanced_dir \"$ENHANCED_DIR\" --ckpt \"$CHECKPOINT\" --N \"$N\""
    echo "Executing command: $ENHANCEMENT_CMD"
    if ! eval $ENHANCEMENT_CMD; then
        echo "Error: Enhancement failed for Checkpoint=$CHECKPOINT, Task=$task_save_name, N=$N on GPU $GPU_ID"
        exit 1
    fi
    echo "Enhancement script execution completed for Checkpoint=$CHECKPOINT, Task=$task_save_name, N=$N on GPU $GPU_ID."
    
    # Construct and execute the metrics calculation script command
    METRICS_CMD="sh cal_metrics_correct_TIMIT.sh \"$task_save_name/N_$N\""
    echo "Executing command: $METRICS_CMD"
    if ! eval $METRICS_CMD; then
        echo "Error: Metrics calculation failed for Checkpoint=$CHECKPOINT, Task=$task_save_name, N=$N on GPU $GPU_ID"
        exit 1
    fi
    echo "Metrics calculation script execution completed for Checkpoint=$CHECKPOINT, Task=$task_save_name, N=$N on GPU $GPU_ID."
}

# Loop through each checkpoint-task pair and run enhancement
for index in "${!CHECKPOINTs[@]}"; do
    CHECKPOINT=${CHECKPOINTs[$index]}
    task_save_name=${task_save_names[$index]}
    
    for N in "${N_list[@]}"; do
        enhance_and_evaluate $N $CHECKPOINT $task_save_name
    done
done

echo "All tasks completed."
