# CountPairPatient.py

import os
import pandas as pd
from datetime import datetime

# --- 1. Configuration (Must match build_meta_dataset.py) ---
PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"
META_ROOT    = os.path.join(PROJECT_ROOT, "outputs", "meta_dataset")
MODEL_NAME   = "swin_tiny"
FEATURE_TYPE = "logit"
USE_CALIB    = False

# Current LRs from your run_tags
RUN_TAGS = {
    "OCT0":  "2e-06",
    "OCT1":  "4e-06",
    "OCTA3": "3e-06",
}

# --- 2. Resolve Paths ---
meta_tag  = f"{MODEL_NAME}__{FEATURE_TYPE}__calib{USE_CALIB}"
lr_folder = f"OCT0_LR{RUN_TAGS['OCT0']}_OCT1_LR{RUN_TAGS['OCT1']}_OCTA3_LR{RUN_TAGS['OCTA3']}"
target_dir = os.path.join(META_ROOT, meta_tag, lr_folder)

csv_path = os.path.join(target_dir, "meta_train_oof.csv")
output_txt = os.path.join(target_dir, "patient_statistics_report.txt")

# --- 3. Main Processing ---
def main():
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load the merged meta dataset
    df_paired = pd.read_csv(csv_path)

    # Calculate statistics directly from the loaded file
    # This avoids the KeyError: 'split_set'
    total_exams = len(df_paired)
    unique_patients = df_paired['patient_id'].nunique()

    # --- 4. Generate Report Content ---
    report = [
        "===========================================",
        "      PAIRED DATASET STATISTICS REPORT     ",
        "===========================================",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model Tag:    {meta_tag}",
        f"LR Folder:    {lr_folder}",
        "-------------------------------------------",
        "[PAIRED DATASET SUMMARY]",
        f" - Total Paired Exams:    {total_exams}",
        f" - Unique Paired Patients: {unique_patients}",
        "-------------------------------------------",
        "Note: Statistics are based on all samples in meta_train_oof.csv.",
        "Results represent the inner-joined data (OCT0, OCT1, OCTA3).",
        "==========================================="
    ]

    report_text = "\n".join(report)

    # Output to Console
    print(report_text)

    # Save to File
    with open(output_txt, "w") as f:
        f.write(report_text)
    
    print(f"\nReport successfully saved to:\n{output_txt}")

if __name__ == "__main__":
    main()