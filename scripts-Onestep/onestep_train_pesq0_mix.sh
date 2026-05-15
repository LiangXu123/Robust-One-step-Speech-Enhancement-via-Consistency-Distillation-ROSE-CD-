#!/bin/bash

# ------------------------------------------------------------------------------
# Script to train a speech enhancement model with consistency distillation.
# Accepts a GPU ID as the primary argument and uses default values for
# other training parameters.
# ------------------------------------------------------------------------------

# --- Load path configuration ---
source ./path_config.sh

# --- Argument Handling ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "  GPU_ID: The index of the GPU to use for training (e.g., 0)."
    exit 1
fi

GPU_ID=$1
echo "Using GPU ID: $GPU_ID"

# --- Default Parameters ---
model_name="onestep_mix"
DEFAULT_DISTILL_SOLVER="mix"
DEFAULT_DISTILL_LOSS_TYPE="L2"
DEFAULT_WEIGHT_SCHEDULE="uniform"
DEFAULT_BASE_DIR="$BASE_DIR_VOICEBANK"
DEFAULT_BACKBONE="ncsnpp_v2"
DEFAULT_CKPT="/workspace/exp_code/sgmse/logs/M3_gpu2_bs8_precond/tsfzrd6w/last.ckpt"
DEFAULT_LOG_DIR="./logs/${model_name}_${DEFAULT_DISTILL_SOLVER}_${DEFAULT_DISTILL_LOSS_TYPE}_${DEFAULT_WEIGHT_SCHEDULE}"

echo "Model Name: $model_name"
echo "Default Distillation Solver: $DEFAULT_DISTILL_SOLVER"
echo "Default Distillation Loss Type: $DEFAULT_DISTILL_LOSS_TYPE"
echo "Default Weight Schedule: $DEFAULT_WEIGHT_SCHEDULE"
echo "Default Base Directory: $DEFAULT_BASE_DIR"
echo "Default Backbone: $DEFAULT_BACKBONE"
echo "Default Checkpoint: $DEFAULT_CKPT"
echo "Log Directory: $DEFAULT_LOG_DIR"

# --- Directory Management ---
[ -d "./out" ] || { echo "Creating output directory: ./out"; mkdir -p "./out"; }
[ -d "./logs" ] || { echo "Creating logs directory: ./logs"; mkdir -p "./logs"; }

if [ ! -d "$DEFAULT_LOG_DIR" ]; then
    echo "Creating log directory: $DEFAULT_LOG_DIR"
    mkdir -p "$DEFAULT_LOG_DIR"
fi

WANDB_DIR="$DEFAULT_LOG_DIR/wandb"
if [ ! -d "$WANDB_DIR" ]; then
    echo "Creating wandb directory: $WANDB_DIR"
    mkdir -p "$WANDB_DIR"
fi

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

export WANDB_MODE=dryrun
echo "W&B mode set to: dryrun"

# --- Training Command ---
# The consistency model does not need input scale, so we set c_in to '1'
CMD="python3 onestep_train.py \
    --base_dir \"$DEFAULT_BASE_DIR\" \
    --backbone \"$DEFAULT_BACKBONE\" \
    --ckpt \"$DEFAULT_CKPT\" \
    --log_dir \"$DEFAULT_LOG_DIR\" \
    --accumulate_grad_batches 2 \
    --batch_size 14 \
    --distill_N 30 \
    --c_in '1' \
    --c_out 'edm' \
    --c_skip 'edm' \
    --loss_type 'precond_denoiser' \
    --loss_weighting 'edm' \
    --distill_solver \"$DEFAULT_DISTILL_SOLVER\" \
    --distill_loss_type \"$DEFAULT_DISTILL_LOSS_TYPE\" \
    --weight_schedule \"$DEFAULT_WEIGHT_SCHEDULE\""

# --- Execution ---
echo "Executing command:"
echo "$CMD"
eval $CMD

# --- Completion Message ---
echo "Training script finished execution."
