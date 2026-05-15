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

DEFAULT_INPUT_DIR="$INPUT_DIR_SIG24"

# Checkpoints and corresponding task names
CHECKPOINTs=(
    "./logs/Onestep_pesq1e-3_pc2_L2_uniform/yscz8ix3/epoch=99-last.ckpt"
)
task_save_names=(
    "ROSE_CD_SIG24"
)
# Assert all checkpoint files exist
for CHECKPOINT in "${CHECKPOINTs[@]}"; do
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file '$CHECKPOINT' does not exist!"
        exit 1
    fi
done

# N values to iterate over
N_list=(1)

enhance_and_evaluate() {
    local N=$1
    local CHECKPOINT=$2
    local TASK_NAME=$3
    local OUTPUT_DIR="./out/$TASK_NAME/N_$N"
    mkdir -p "$OUTPUT_DIR" || { echo "Error: Failed to create $OUTPUT_DIR"; exit 1; }
    
    # Get absolute path
    local OUTPUT_CSV="$TASK_NAME.csv"
    local ABS_OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
    
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    
    # Enhancement command - using proper array and quoting
    local ENHANCEMENT_CMD=(
        python3 enhancement.py
        --sampler_type 'stochastic'
        --test_dir "$DEFAULT_INPUT_DIR"
        --enhanced_dir "$OUTPUT_DIR"
        --ckpt "$CHECKPOINT"
        --N "$N"
    )
    
    echo "Running enhancement: ${ENHANCEMENT_CMD[*]}"
    if ! "${ENHANCEMENT_CMD[@]}"; then
        echo "Enhancement failed for N=$N"
        exit 1
    fi
    
    # Metrics command
    echo "Calculating metrics: sh cal_metrics_correct_VBDMD.sh \"$TASK_NAME/N_$N\""
    if ! sh cal_metrics_correct_VBDMD.sh "$TASK_NAME/N_$N"; then
        echo "Metrics calculation failed for N=$N"
        exit 1
    fi
    
    # DNSMOS command - using pushd/popd for safer directory changes
    echo "Calculating DNSMOS metrics in: $ABS_OUTPUT_DIR"
    if ! (cd /workspace/exp_code/Eval_waspaa2025/MOS && sh run.sh "$GPU_ID" "$ABS_OUTPUT_DIR" "$OUTPUT_CSV"); then
        echo "DNSMOS Metrics calculation failed for N=$N"
        exit 1
    fi
}

# Run enhancement and evaluation
for index in "${!CHECKPOINTs[@]}"; do
    CHECKPOINT=${CHECKPOINTs[$index]}
    TASK_NAME=${task_save_names[$index]}

    for N in "${N_list[@]}"; do
        enhance_and_evaluate "$N" "$CHECKPOINT" "$TASK_NAME"
    done
done

echo "All tasks completed."
