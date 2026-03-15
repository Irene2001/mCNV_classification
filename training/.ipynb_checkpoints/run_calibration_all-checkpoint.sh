#!/bin/bash

"""
# run_calibration_all.sh
# Purpose: Automates Temperature Scaling (TS) for all modality RunTags.
# Usage: 
#   chmod +x run_calibration_all.sh
#   ./run_calibration_all.sh
"""

MODEL="swin_tiny"
PROJECT_DIR="/data/Irene/SwinTransformer/Swin_Meta"
OOF_BASE_DIR="$PROJECT_DIR/outputs/oof_predictions/$MODEL"
CALIB_SCRIPT="$PROJECT_DIR/training/calibrate_oof_predictions.py"

# Verification: Ensure calibration script exists
if [ ! -f "$CALIB_SCRIPT" ]; then
    echo "Error: Calibration script not found at $CALIB_SCRIPT"
    exit 1
fi

# Iterate through specified modalities
for MOD in OCT0 OCT1 OCTA3
do
    MOD_DIR="$OOF_BASE_DIR/$MOD"

    if [ ! -d "$MOD_DIR" ]; then
        echo "Warning: Directory for $MOD not found ($MOD_DIR). Skipping."
        continue
    fi

    echo "------------------------------------------------"
    echo "Modality: $MOD"
    echo "------------------------------------------------"

    # Identify all RunTag directories within the modality folder
    for RUN_PATH in "$MOD_DIR"/*/ 
    do
        # Ensure it is a directory
        [ -d "$RUN_PATH" ] || continue

        # Extract RunTag name
        RUN=$(basename "$RUN_PATH")

        echo "Executing Calibration for RunTag: $RUN"

        # Execute Python calibration script
        python "$CALIB_SCRIPT" \
            --model_name "$MODEL" \
            --modality "$MOD" \
            --run_tag "$RUN"

        if [ $? -eq 0 ]; then
            echo "Status: COMPLETED for $MOD / $RUN"
        else
            echo "Status: FAILED for $MOD / $RUN"
        fi
    done
done

echo "------------------------------------------------"
echo "All calibration tasks are finished."
echo "------------------------------------------------"