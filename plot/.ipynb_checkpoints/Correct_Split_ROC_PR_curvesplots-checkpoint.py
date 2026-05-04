# Correct_Split_ROC_PR_curvesplots.py

# Revise: Monotonic Interpolation

import os
import pandas as pd
import numpy as np

# ==========================================
# 1. Basic configuration
# ==========================================
PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"
OUTPUT_DIR = "/data/Irene/SwinTransformer/Swin_Meta/plot"

ROC_DIR = os.path.join(OUTPUT_DIR, "ROC")
PR_DIR = os.path.join(OUTPUT_DIR, "PR")

MODALITIES = ["OCT0", "OCT1", "OCTA3", "Meta"]
MODELS = ["VGG16", "EffNetB0", "SwinT"]

os.makedirs(ROC_DIR, exist_ok=True)
os.makedirs(PR_DIR, exist_ok=True)

# ==========================================
# 2. Path mapping (update Meta paths if needed)
# ==========================================
PATH_MAP = {
    "VGG16": {
        "base": "VGG16_outputs",
        "OCT0": "test_evaluation/vgg16/Partial_B5/OCT0/BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.11_0.89_2_WSon_1_2.9/Best_fold2",
        "OCT1": "test_evaluation/vgg16/Partial_B5/OCT1/BS16_EP100_LR9e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.113_0.887_2_WSon_1_2.8/Best_fold5",
        "OCTA3": "test_evaluation/vgg16/Partial_B5/OCTA3/BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.13_0.87_2_WSon_1_2.6/Best_fold1",
        "Meta": "meta_training/vgg16__logit__calibFalse/Partial_B5/OCT0_LR8e-06_OCT1_LR9e-06_OCTA3_LR8e-06/test_evaluation"
    },
    "EffNetB0": {
        "base": "EffNetB0_outputs",
        "OCT0": "test_evaluation/efficientnet_b0/Partial_B5_6/OCT0/BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.11_0.89_2_WSon_1_2.9/Best_fold2",
        "OCT1": "test_evaluation/efficientnet_b0/Partial_B5_6/OCT1/BS16_EP100_LR2e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.113_0.887_2_WSon_1_2.8/Best_fold4",
        "OCTA3": "test_evaluation/efficientnet_b0/Partial_B5_6/OCTA3/BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.13_0.87_2_WSon_1_2.6/Best_fold2",
        "Meta": "meta_training/efficientnet_b0__logit__calibFalse/Partial_B5_6/OCT0_LR3e-05_OCT1_LR2e-05_OCTA3_LR3e-05/test_evaluation"
    },
    "SwinT": {
        "base": "outputs",
        "OCT0": "test_evaluation/swin_tiny/OCT0/BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9/Best_fold2/NoTitle_testmetrics",
        "OCT1": "test_evaluation/swin_tiny/OCT1/BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8/Best_fold1/NoTitle_testmetrics",
        "OCTA3": "test_evaluation/swin_tiny/OCTA3/BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6/Best_fold2/NoTitle_testmetrics",
        "Meta": "meta_training/swin_tiny__logit__calibFalse/OCT0_LR2e-06_OCT1_LR4e-06_OCTA3_LR3e-06/test_evaluation"
    }
}

# ==========================================
# 3. Core function: generate CSV per modality
# ==========================================
def generate_per_modality_csv(curve_type="roc"):
    """
    Generate separate CSV files for each modality.
    Each CSV contains 1 shared X-axis and 3 model Y-curves for OriginPro.
    """

    if curve_type == "roc":
        x_col_raw = "fpr"
        y_col_raw = "tpr"
        prefix = "ROC"
        save_dir = ROC_DIR
        x_axis_name = "FPR(X)"  # OriginPro 會自動把 X 軸標籤命名為 FPR
    else:
        x_col_raw = "recall"
        y_col_raw = "precision"
        prefix = "PR"
        save_dir = PR_DIR
        x_axis_name = "Recall(X)" # OriginPro 會自動把 X 軸標籤命名為 Recall

    print(f"\nProcessing {prefix} curves...")

    for mod in MODALITIES:
        print(f"\nProcessing modality: {mod}")

        # 1. 建立所有模型「共用」的 500 個 X 軸數據點
        shared_x = np.linspace(0, 1, 500)
        
        # 2. 初始化一個字典，第一欄放共用的 X 軸
        modality_data = {x_axis_name: shared_x}

        for model in MODELS:
            if mod == "Meta":
                filename = "meta_lr_roc_curve.csv" if curve_type == "roc" else "meta_lr_pr_curve.csv"
            else:
                filename = "roc_data.csv" if curve_type == "roc" else "pr_data.csv"

            current_sub_path = PATH_MAP[model].get(mod, "")
            
            full_path = os.path.join(
                PROJECT_ROOT,
                PATH_MAP[model]["base"],
                current_sub_path,
                filename
            )

            # 定義乾淨的 Y 軸名稱，這樣圖例(Legend)就會非常乾淨
            y_name = f"{model}(Y)"

            if os.path.exists(full_path):
                df = pd.read_csv(full_path)
                
                # 基礎排序與去重 (解決折返問題)
                df = df.sort_values(by=x_col_raw).drop_duplicates(subset=[x_col_raw], keep='last')
                
                if curve_type != "roc":
                    # PR 曲線專用的單調遞減處理 (符合 IEEE-JBHI 規範)
                    df[y_col_raw] = df[y_col_raw].iloc[::-1].cummax().iloc[::-1]
                
                # 對齊 shared_x 進行線性插值
                resampled_y = np.interp(shared_x, df[x_col_raw], df[y_col_raw])
                
                # 將計算好的 Y 值存入字典中
                modality_data[y_name] = resampled_y

            else:
                print(f"Missing file: {full_path}")

        # 3. 如果字典裡除了 X 軸以外還有模型的數據，就轉成 DataFrame 並存檔
        if len(modality_data) > 1:
            modality_df = pd.DataFrame(modality_data)          
            out_name = f"{prefix}_{mod}.csv"
            
            final_save_path = os.path.join(save_dir, out_name)
            
            modality_df.to_csv(final_save_path, index=False)
            print(f"Saved properly to: {final_save_path}")

        else:
            print(f"No data found for {mod}")

# ==========================================
# 4. Main
# ==========================================
if __name__ == "__main__":

    print("="*60)
    print("Curve Aggregation Tool (Per-Modality Output)")
    print("="*60)

    generate_per_modality_csv("roc")
    generate_per_modality_csv("pr")

    print("\nDone. CSV files ready for OriginPro plotting.")