#!/bin/bash

# --- Load path configuration ---
source ./path_config.sh

# --- Argument Handling ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 GPU_ID [base_dir] [backbone_type] [log_dir]"
    echo "Default values will be used if arguments are not provided."
    exit 1
fi

GPU_ID=$1
echo "Using GPU ID: $GPU_ID"

# --- Default Parameters ---
DEFAULT_BASE_DIR="$BASE_DIR_VOICEBANK"
DEFAULT_BACKBONE="ncsnpp_v2"
DEFAULT_LOG_DIR="./logs/teacher"

# Get parameters or fallback to defaults
BASE_DIR=${2:-$DEFAULT_BASE_DIR}
BACKBONE=${3:-$DEFAULT_BACKBONE}
LOG_DIR=${4:-$DEFAULT_LOG_DIR}

echo "Base Directory: $BASE_DIR"
echo "Backbone: $BACKBONE"
echo "Log Directory: $LOG_DIR"

# --- Directory Management ---
[ -d "./out" ] || { echo "Creating output directory: ./out"; mkdir -p "./out"; }
[ -d "./logs" ] || { echo "Creating logs directory: ./logs"; mkdir -p "./logs"; }

if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

WANDB_DIR="$LOG_DIR/wandb"
[ -d "$WANDB_DIR" ] || { echo "Creating wandb directory: $WANDB_DIR"; mkdir -p "$WANDB_DIR"; }

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

export WANDB_MODE=dryrun
echo "W&B mode set to: dryrun"

# --- Timestamped Logging ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/train_${TIMESTAMP}.log"

# --- Training Command ---
CMD="python3 train.py \
    --base_dir \"$BASE_DIR\" \
    --backbone \"$BACKBONE\" \
    --log_dir \"$LOG_DIR\" \
    --accumulate_grad_batches 2 \
    --batch_size 16 \
    --c_in 'edm' \
    --c_out 'edm' \
    --c_skip 'edm' \
    --loss_type 'precond_denoiser' \
    --loss_weighting 'edm'"

# --- Execution ---
echo "Executing command:"
echo "$CMD"
echo "Logging to: $LOGFILE"
eval $CMD 2>&1 | tee "$LOGFILE"

# --- Completion Message ---
echo "Training script finished execution."
