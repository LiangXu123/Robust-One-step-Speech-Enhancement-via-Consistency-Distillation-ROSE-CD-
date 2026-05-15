#!/bin/bash

# Load environment variables (including NOISY_DIR_VOICEBANK)
source ./path_config.sh

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 GPU_ID CHECKPOINT_DIR"
    echo "Parameters required: single GPU ID and checkpoint directory."
    exit 1
fi

GPU_ID="$1"          # Single GPU ID
CHECKPOINT_DIR="$2"  # Directory containing .ckpt files
DEFAULT_INPUT_DIR="$NOISY_DIR_VOICEBANK"
OUTPUT_BASE="./out"
CHECKPOINT_PARENT_DIR=$(dirname "$CHECKPOINT_DIR")
CHECKPOINT_DIR_NAME=$(basename "$CHECKPOINT_PARENT_DIR")
RESULTS_FILE="./evaluation_results_${CHECKPOINT_DIR_NAME}.txt"
ENHANCED_DIR="$OUTPUT_BASE/${CHECKPOINT_DIR_NAME}/N_1"
N_list=(1)

# Clear previous results
# > "$RESULTS_FILE"

mkdir -p "$ENHANCED_DIR"

enhance_and_evaluate() {
    local CHECKPOINT_PATH="$1"
    local CHECKPOINT_NAME
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)

    for N in "${N_list[@]}"; do
        echo "Processing checkpoint: $CHECKPOINT_NAME with N=$N on GPU $GPU_ID"
        export CUDA_VISIBLE_DEVICES="$GPU_ID"

        # Clear the enhanced directory
        echo "Clearing enhanced directory: $ENHANCED_DIR"
        rm -rf "$ENHANCED_DIR"/*
        
        # Run enhancement
        ENHANCEMENT_CMD="python3 enhancement.py --sampler_type 'stochastic' --test_dir \"$DEFAULT_INPUT_DIR\" --enhanced_dir \"$ENHANCED_DIR\" --ckpt \"$CHECKPOINT_PATH\" --N \"$N\""
        echo "Executing: $ENHANCEMENT_CMD"
        if ! eval $ENHANCEMENT_CMD; then
            echo "Error: Enhancement failed for checkpoint $CHECKPOINT_NAME"
            exit 1
        fi

        # Calculate metrics
        METRICS_CMD="sh cal_metrics_correct_VBDMD.sh \"${CHECKPOINT_DIR_NAME}/N_$N\""
        echo "Executing: $METRICS_CMD"
        if ! eval $METRICS_CMD; then
            echo "Error: Metrics calculation failed for checkpoint $CHECKPOINT_NAME"
            exit 1
        fi

        # Parse the output metrics
        METRICS_OUTPUT_FILE="./metrics_output.txt"
        if [ -f "$METRICS_OUTPUT_FILE" ]; then
            PESQ_LINE=$(grep "PESQ:" "$METRICS_OUTPUT_FILE")
            ESTOI_LINE=$(grep "ESTOI:" "$METRICS_OUTPUT_FILE")
            SI_SDR_LINE=$(grep "SI-SDR:" "$METRICS_OUTPUT_FILE")
            SI_SIR_LINE=$(grep "SI-SIR:" "$METRICS_OUTPUT_FILE")
            SI_SAR_LINE=$(grep "SI-SAR:" "$METRICS_OUTPUT_FILE")
        else
            echo "Warning: No metrics output file found for checkpoint $CHECKPOINT_NAME"
            continue
        fi

        # Save results
        echo "$CHECKPOINT_NAME" >> "$RESULTS_FILE"
        echo "$PESQ_LINE" >> "$RESULTS_FILE"
        echo "$ESTOI_LINE" >> "$RESULTS_FILE"
        echo "$SI_SDR_LINE" >> "$RESULTS_FILE"
        echo "$SI_SIR_LINE" >> "$RESULTS_FILE"
        echo "$SI_SAR_LINE" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    done
}

# Loop through all checkpoints
for ckpt in "$CHECKPOINT_DIR"/*.ckpt; do
    if [ -f "$ckpt" ]; then
        enhance_and_evaluate "$ckpt"
    fi
done

echo "Enhancement and evaluation complete for all checkpoints."
echo "Results saved in $RESULTS_FILE"