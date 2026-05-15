#!/bin/bash

# --- Load path configuration ---
source ./path_config.sh

# Usage info
usage() {
    echo "Usage: $0 GPU_ID"
    echo "  GPU_ID : required, GPU device id"
    echo "Example:"
    echo "  $0 0"
    exit 1
}

# Check at least 1 argument is provided
if [ "$#" -lt 1 ]; then
    usage
fi

GPU_ID="$1"

# Use NOISY_DIR_VOICEBANK as input directory (from path_config.sh)
DEFAULT_INPUT_DIR="$NOISY_DIR_VOICEBANK"

# Define checkpoint + chunk_length_ms + buffer_chunks
# Format: "checkpoint_path,chunk_length_ms,buffer_chunks"
CHECKPOINTS_WITH_PARAMS=(
    # "./logs/Teacher_model/vcrsnu58/last.ckpt,20,0"
    # "./logs/Teacher_model/vcrsnu58/last.ckpt,20,5"
    "./logs/Teacher_model/vcrsnu58/last.ckpt,40,0"
    # "./logs/Teacher_model/vcrsnu58/last.ckpt,40,5"    
)

# Task save names, must align with above
TASK_SAVE_NAMES=(
    # "teacher_VB_20_0"
    # "teacher_VB_20_5"
    "teacher_VB_40_0"
    # "teacher_VB_40_5"
)

# N values
N_list=(30)

# Function to run enhancement and evaluation
enhance_and_evaluate() {
    local N=$1
    local CHECKPOINT=$2
    local CHUNK_LENGTH_MS=$3
    local BUFFER_CHUNKS=$4
    local TASK_NAME=$5
    local OUTPUT_DIR="./out/$TASK_NAME"

    # Create output directories
    mkdir -p "$OUTPUT_DIR" || { echo "Error: Failed to create output directory: $OUTPUT_DIR"; exit 1; }
    local ENHANCED_DIR="$OUTPUT_DIR/N_$N"
    mkdir -p "$ENHANCED_DIR" || { echo "Error: Failed to create enhanced directory: $ENHANCED_DIR"; exit 1; }

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    # Run enhancement
    ENHANCEMENT_CMD="python3 real_time_enhancement.py \
        --test_dir \"$DEFAULT_INPUT_DIR\" \
        --enhanced_dir \"$ENHANCED_DIR\" \
        --ckpt \"$CHECKPOINT\" \
        --N \"$N\" \
        --chunk_length_ms $CHUNK_LENGTH_MS \
        --buffer_chunks $BUFFER_CHUNKS"

    echo "Running enhancement: $ENHANCEMENT_CMD"
    eval $ENHANCEMENT_CMD || { echo "Enhancement failed for N=$N"; exit 1; }

    # Run metrics
    METRICS_CMD="sh cal_metrics_correct_VBDMD.sh \"$TASK_NAME/N_$N\""
    echo "Calculating metrics: $METRICS_CMD"
    eval $METRICS_CMD || { echo "Metrics calculation failed for N=$N"; exit 1; }
}

# Main loop
for index in "${!CHECKPOINTS_WITH_PARAMS[@]}"; do
    IFS=',' read -r CHECKPOINT CHUNK_LENGTH_MS BUFFER_CHUNKS <<< "${CHECKPOINTS_WITH_PARAMS[$index]}"
    TASK_NAME="${TASK_SAVE_NAMES[$index]}"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file '$CHECKPOINT' does not exist!"
        exit 1
    fi

    for N in "${N_list[@]}"; do
        enhance_and_evaluate $N "$CHECKPOINT" "$CHUNK_LENGTH_MS" "$BUFFER_CHUNKS" "$TASK_NAME"
    done
done

echo "All tasks completed."
