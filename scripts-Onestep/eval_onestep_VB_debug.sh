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

# Checkpoints and corresponding task names
CHECKPOINTs=(
    "./logs/onestep_euler_x_mean_euler_L2_uniform/qx15s35z/last.ckpt"
    "./logs/onestep_euler_x_noise_euler_L2_uniform/23ttivfl/last.ckpt"
    "./logs/onestep_heun_x_mean_heun_L2_uniform/i247wf0u/last.ckpt"
    "./logs/onestep_heun_x_noise_heun_L2_uniform/xpw5zu4h/last.ckpt"
    "./logs/onestep_heun_x_mean_rosecd_heun_L2_uniform/9844h06o/last.ckpt"
    "./logs/onestep_heun_x_noise_rosecd_heun_L2_uniform/yzcxuiso/last.ckpt"
    "./logs/onestep_mix_mix_L2_uniform/v4qw2ce8/last.ckpt"
)
task_save_names=(
    "onestep_euler_x_mean"
    "onestep_euler_x_noise"
    "onestep_heun_x_mean"
    "onestep_heun_x_noise"    
    "onestep_heun_x_mean_rosecd"
    "onestep_heun_x_noise_rosecd"      
    "onestep_heun_x_noise_mix_sde"      
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

# Arrays to store task information for later metrics calculation
declare -a COMPLETED_TASKS_N
declare -a COMPLETED_TASKS_NAME

run_enhancement() {
    local N=$1
    local CHECKPOINT=$2
    local TASK_NAME=$3
    local OUTPUT_DIR="./out/$TASK_NAME/N_$N"
    mkdir -p "$OUTPUT_DIR" || { echo "Error: Failed to create $OUTPUT_DIR"; exit 1; }
    
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
        echo "Enhancement failed for N=$N, TASK=$TASK_NAME"
        exit 1
    fi
    
    # Store completed task info for later metrics calculation
    COMPLETED_TASKS_N+=("$N")
    COMPLETED_TASKS_NAME+=("$TASK_NAME")
    
    echo "Enhancement completed for N=$N, TASK=$TASK_NAME"
}

run_metrics() {
    local N=$1
    local TASK_NAME=$2
    local OUTPUT_DIR="./out/$TASK_NAME/N_$N"
    local OUTPUT_CSV="$TASK_NAME.csv"
    local ABS_OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
    
    # Metrics command
    echo "Calculating metrics: sh cal_metrics_correct_VBDMD.sh \"$TASK_NAME/N_$N\""
    if ! sh cal_metrics_correct_VBDMD.sh "$TASK_NAME/N_$N"; then
        echo "Metrics calculation failed for N=$N, TASK=$TASK_NAME"
        exit 1
    fi
    
    # DNSMOS command - using pushd/popd for safer directory changes
    # echo "Calculating DNSMOS metrics in: $ABS_OUTPUT_DIR"
    # if ! (cd /workspace/exp_code/Eval_waspaa2025/MOS && sh run.sh "$GPU_ID" "$ABS_OUTPUT_DIR" "$OUTPUT_CSV"); then
    #     echo "DNSMOS Metrics calculation failed for N=$N"
    #     exit 1
    # fi
    
    echo "Metrics completed for N=$N, TASK=$TASK_NAME"
}

echo "=== Phase 1: Running all enhancements ==="

# Run all enhancement commands first
for index in "${!CHECKPOINTs[@]}"; do
    CHECKPOINT=${CHECKPOINTs[$index]}
    TASK_NAME=${task_save_names[$index]}

    for N in "${N_list[@]}"; do
        run_enhancement "$N" "$CHECKPOINT" "$TASK_NAME"
    done
done

echo "=== Phase 2: Running all metrics calculations ==="

# Run all metrics calculations
for i in "${!COMPLETED_TASKS_N[@]}"; do
    N=${COMPLETED_TASKS_N[$i]}
    TASK_NAME=${COMPLETED_TASKS_NAME[$i]}
    run_metrics "$N" "$TASK_NAME"
done

echo "All tasks completed."