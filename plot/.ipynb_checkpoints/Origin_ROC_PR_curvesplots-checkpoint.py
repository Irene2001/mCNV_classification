# Origin_ROC_PR_curvesplots.py

import os
import pandas as pd

# ==========================================
# 1. Basic configuration
# ==========================================
PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"

# Modalities to be aggregated
MODALITIES = ["OCT0", "OCT1", "OCTA3", "Meta"]

# Models to be compared
MODELS = ["VGG16", "EffNetB0", "SwinT"]


# ==========================================
# 2. Path mapping for each model and modality
# Note:
# - Ensure all paths are correct before running
# - Meta paths must be manually verified
# ==========================================
PATH_MAP = {
    "VGG16": {
        "base": "VGG16_outputs/test_evaluation/vgg16/Partial_B5",
        "OCT0": "OCT0/BS16_EP99_LR3e-05_WD0.01_DR0.5_FIXED_BACKBONE_FL0.11_0.89_2_WSon_1_2.9/Best_fold5",
        "OCT1": "OCT1/BS16_EP100_LR9e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.113_0.887_2_WSon_1_2.8/Best_fold5",
        "OCTA3": "OCTA3/BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.13_0.87_2_WSon_1_2.6/Best_fold1",
        "Meta": "Meta/Stacking_Results_Folder"
    },
    "EffNetB0": {
        "base": "EffNetB0_outputs/test_evaluation/efficientnet_b0/Partial_B5_6",
        "OCT0": "OCT0/BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.11_0.89_2_WSon_1_2.9/Best_fold2",
        "OCT1": "OCT1/BS16_EP100_LR2e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.113_0.887_2_WSon_1_2.8/Best_fold4",
        "OCTA3": "OCTA3/BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.13_0.87_2_WSon_1_2.6/Best_fold2",
        "Meta": "Meta/Stacking_Results_Folder"
    },
    "SwinT": {
        "base": "outputs/test_evaluation/swin_tiny",
        "OCT0": "OCT0/BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9/Best_fold2/NoTitle_testmetrics",
        "OCT1": "OCT1/BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8/Best_fold1/NoTitle_testmetrics",
        "OCTA3": "OCTA3/BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6/Best_fold2/NoTitle_testmetrics",
        "Meta": "Meta/Stacking_Results_Folder"
    }
}


# ==========================================
# 3. Main function to generate merged CSV
# ==========================================
def generate_curve_csv(curve_type="roc"):
    """
    Generate a merged CSV file for ROC or PR curves.

    Parameters
    ----------
    curve_type : str
        "roc" -> FPR vs TPR
        "pr"  -> Recall vs Precision

    Output
    ------
    A wide-format CSV file that can be directly imported into OriginPro.
    Each curve is stored as a pair of columns:
        <Modality>_<Model>_<Metric>(X)
        <Modality>_<Model>_<Metric>(Y)
    """

    if curve_type == "roc":
        filename = "roc_data.csv"
        x_col_raw = "fpr"
        y_col_raw = "tpr"
        out_name = "OriginPro_ROC_Master.csv"
        print("Processing ROC curve data...")
    else:
        filename = "pr_data.csv"
        x_col_raw = "recall"
        y_col_raw = "precision"
        out_name = "OriginPro_PR_Master.csv"
        print("Processing PR curve data...")

    # List of Series objects to preserve independent curve lengths
    all_series = []

    # Iterate over all modalities and models
    for mod in MODALITIES:
        for model in MODELS:

            sub_path = PATH_MAP[model].get(mod, "")
            full_path = os.path.join(
                PROJECT_ROOT,
                PATH_MAP[model]["base"],
                sub_path,
                filename
            )

            # Define column names compatible with OriginPro
            # Example: OCT0_VGG16_FPR(X)
            x_col_name = f"{mod}_{model}_{x_col_raw.upper()}(X)"
            y_col_name = f"{mod}_{model}_{y_col_raw.upper()}(Y)"

            if os.path.exists(full_path):
                df = pd.read_csv(full_path)

                # Sort by X-axis to ensure proper curve plotting
                df = df.sort_values(by=x_col_raw)

                # Append as independent Series (no forced alignment)
                all_series.append(pd.Series(df[x_col_raw].values, name=x_col_name))
                all_series.append(pd.Series(df[y_col_raw].values, name=y_col_name))

            else:
                print(f"Missing file: {full_path}")

    # Concatenate all curves into a wide-format table
    if all_series:
        master_df = pd.concat(all_series, axis=1)
        master_df.to_csv(out_name, index=False)

        print(f"Output file generated: {out_name}")
        print(f"Total columns: {len(master_df.columns)}")

    else:
        print("No valid data found. CSV file was not generated.")


# ==========================================
# 4. Entry point
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("OriginPro Curve Aggregation Tool (ROC & PR)")
    print("=" * 60)

    generate_curve_csv("roc")
    generate_curve_csv("pr")

    print("Processing completed. Import the generated CSV files into OriginPro.")