#!/bin/bash

# --- Load path configuration ---
source ./path_config.sh

# Check GPU ID argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "GPU_ID is required."
    exit 1
fi

GPU_ID="$1"

DEFAULT_INPUT_DIR="$NOISY_DIR_VOICEBANK"

# Define checkpoint + chunk_length_ms + buffer_chunks
# Format: "checkpoint_path,chunk_length_ms,buffer_chunks"
CHECKPOINTS_WITH_PARAMS=(
    "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,1000,3"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,20,0"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,20,5"    
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,40,0"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,40,5"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,260,0"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,520,0"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,260,3"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,520,3"
    # "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=075.ckpt,-1,0"
)

# Task save names
TASK_SAVE_NAMES=(
    "Onestep_pesq1e-3realtime_1000_3"
    # "Onestep_pesq1e-3realtime_20_0"
    # "Onestep_pesq1e-3realtime_20_5"
    # "Onestep_pesq1e-3realtime_40_0"
    # "Onestep_pesq1e-3realtime_40_5"
    # "Onestep_pesq1e-3realtime_260_0"
    # "Onestep_pesq1e-3realtime_520_0"
    # "Onestep_pesq1e-3realtime_260_3"
    # "Onestep_pesq1e-3realtime_520_3"
    # "Onestep_pesq1e-3realtime_-1_0"
)

# N values to iterate over
N_list=(1)

enhance_and_evaluate() {
    local N=$1
    local CHECKPOINT=$2
    local CHUNK_LENGTH_MS=$3
    local BUFFER_CHUNKS=$4
    local TASK_NAME=$5
    local OUTPUT_DIR="./out/$TASK_NAME/N_$N"
    
    mkdir -p "$OUTPUT_DIR" || { echo "Error: Failed to create $OUTPUT_DIR"; exit 1; }

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    ENHANCEMENT_CMD="python3 real_time_enhancement.py \
        --sampler_type 'stochastic' \
        --test_dir \"$DEFAULT_INPUT_DIR\" \
        --enhanced_dir \"$OUTPUT_DIR\" \
        --ckpt \"$CHECKPOINT\" \
        --N \"$N\" \
        --chunk_length_ms $CHUNK_LENGTH_MS \
        --buffer_chunks $BUFFER_CHUNKS"

    echo "Running enhancement: $ENHANCEMENT_CMD"
    if ! eval $ENHANCEMENT_CMD; then
        echo "Enhancement failed for N=$N"
        exit 1
    fi

    METRICS_CMD="sh cal_metrics_correct_VBDMD.sh \"$TASK_NAME/N_$N\""
    echo "Calculating metrics: $METRICS_CMD"
    if ! eval $METRICS_CMD; then
        echo "Metrics calculation failed for N=$N"
        exit 1
    fi
}

# Run enhancement and evaluation
for index in "${!CHECKPOINTS_WITH_PARAMS[@]}"; do
    IFS=',' read -r CHECKPOINT CHUNK_LENGTH_MS BUFFER_CHUNKS <<< "${CHECKPOINTS_WITH_PARAMS[$index]}"
    TASK_NAME="${TASK_SAVE_NAMES[$index]}"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file '$CHECKPOINT' does not exist!"
        exit 1
    fi

    for N in "${N_list[@]}"; do
        enhance_and_evaluate "$N" "$CHECKPOINT" "$CHUNK_LENGTH_MS" "$BUFFER_CHUNKS" "$TASK_NAME"
    done
done

echo "All tasks completed."
