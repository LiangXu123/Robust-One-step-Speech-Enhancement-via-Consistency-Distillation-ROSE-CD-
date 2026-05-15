#!/bin/bash

# --- Load path configuration ---
source ./path_config.sh

# Check if at least one argument (GPU ID) is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 GPU_ID [base_dir] [backbone_type] [log_dir]"
    echo "Default values will be used if arguments are not provided."
    exit 1
fi

# Set GPU ID from first parameter
GPU_ID=$1

# Set default values using path_config
DEFAULT_BASE_DIR="$BASE_DIR_VOICEBANK"
DEFAULT_BACKBONE="ncsnpp_v2"
DEFAULT_LOG_DIR="./logs/Teacher_model"

# Get parameters or fallback to defaults
BASE_DIR=${2:-$DEFAULT_BASE_DIR}
BACKBONE=${3:-$DEFAULT_BACKBONE}
LOG_DIR=${4:-$DEFAULT_LOG_DIR}

# Set CUDA device and disable W&B logging
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE=dryrun

# Ensure necessary directories exist
[ -d "./out" ] || mkdir -p "./out"
[ -d "./logs" ] || mkdir -p "./logs"

# Construct the training command
CMD="python3 train.py --base_dir \"$BASE_DIR\" --backbone \"$BACKBONE\" --log_dir \"$LOG_DIR\" \
    --accumulate_grad_batches 2 --batch_size 16 \
    --c_in 'edm' --c_out 'edm' --c_skip 'edm' \
    --loss_type 'precond_denoiser' --loss_weighting 'edm'"

# Print and execute the command
echo "Executing command: $CMD"
eval $CMD

echo "Training script finished execution."
