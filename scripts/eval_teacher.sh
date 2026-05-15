#!/bin/bash

# --- Load path configuration ---
source ./path_config.sh

# Check if exactly one argument (GPU_ID) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "Only one parameter is required: the single GPU ID."
    exit 1
fi

# Set GPU ID
GPU_ID="$1"  # Single GPU ID

# Use NOISY_DIR_VOICEBANK as input directory (from path_config.sh)
DEFAULT_INPUT_DIR="$NOISY_DIR_VOICEBANK"

# Default values for checkpoints and task names
CHECKPOINTs=(
    # "./logs/teacher/vcrsnu58/last.ckpt"
    "./logs/teacher/vcrsnu58/last.ckpt"
)

task_save_names=(
    "teacher"
)

# Assert all checkpoint files exist
for CHECKPOINT in "${CHECKPOINTs[@]}"; do
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file '$CHECKPOINT' does not exist!"
        exit 1
    fi
done

# Define list of N values
N_list=(1 5 10 30)
N_list=(30)

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
    
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    # Construct and execute the enhancement script command
    ENHANCEMENT_CMD="python3 enhancement.py --test_dir \"$DEFAULT_INPUT_DIR\" --enhanced_dir \"$ENHANCED_DIR\" --ckpt \"$CHECKPOINT\" --N \"$N\""
    echo "Running enhancement: $ENHANCEMENT_CMD"
    eval $ENHANCEMENT_CMD || { echo "Enhancement failed for N=$N"; exit 1; }
    
    # Construct and execute the metrics calculation script command
    METRICS_CMD="sh cal_metrics_VB.sh \"$task_save_name/N_$N\""
    echo "Calculating metrics: $METRICS_CMD"
    eval $METRICS_CMD || { echo "Metrics calculation failed for N=$N"; exit 1; }
}

# Loop through each checkpoint-task pair and run enhancement and evaluation
for index in "${!CHECKPOINTs[@]}"; do
    CHECKPOINT=${CHECKPOINTs[$index]}
    task_save_name=${task_save_names[$index]}
    
    for N in "${N_list[@]}"; do
        enhance_and_evaluate $N $CHECKPOINT $task_save_name
    done
done

echo "All tasks completed."
