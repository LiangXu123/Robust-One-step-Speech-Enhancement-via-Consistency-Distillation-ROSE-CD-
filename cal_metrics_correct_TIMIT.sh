#!/bin/bash

# --- Load path configuration ---
source "$(dirname "$0")/path_config.sh"

# Get Test_Dir from the first argument or use default
Test_Dir=${1:-"115_n60"}

# Set enhanced output directory
ENHANCED_DIR="./out/$Test_Dir"

# Print directory paths
echo "Clean directory   : $CLEAN_DIR_TIMIT"
echo "Noisy directory   : $NOISY_DIR_TIMIT"
echo "Enhanced directory: $ENHANCED_DIR"

# Construct the command
CMD="python3 calc_metrics.py --clean_dir \"$CLEAN_DIR_TIMIT\" --noisy_dir \"$NOISY_DIR_TIMIT\" --enhanced_dir \"$ENHANCED_DIR\""

# Execute
echo "Executing command: $CMD"
eval $CMD

echo "Command execution completed."
