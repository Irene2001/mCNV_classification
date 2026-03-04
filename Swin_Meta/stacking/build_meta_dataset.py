# build_meta_dataset.py

"""
Build meta-dataset for stacking logistic regression.

Compatible with:
train_singlemode_oof.py outputs.

Supports:
- logit / probability
- calibrated / uncalibrated

Usage example:
python build_meta_dataset.py \
    --model_name swin_tiny \
    --oct0_tag RUN1 \
    --oct1_tag RUN2 \
    --octa3_tag RUN3 \
    --feature_mode logit \
    --use_calib True
"""

import os
import json
import argparse
import pandas as pd


PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"

OOF_ROOT = os.path.join(PROJECT_ROOT, "outputs", "oof_predictions")
META_ROOT = os.path.join(PROJECT_ROOT, "outputs", "meta_dataset")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def resolve_feature(feature_mode, use_calib):

    if feature_mode == "logit" and not use_calib:
        return "logit_uncal"

    if feature_mode == "logit" and use_calib:
        return "logit_calib"

    if feature_mode == "prob" and not use_calib:
        return "prob_uncal"

    if feature_mode == "prob" and use_calib:
        return "prob_calib"


def load_oof(model_name, modality, run_tag):

    path = os.path.join(
        OOF_ROOT,
        model_name,
        modality,
        run_tag,
        "all_folds_oof.csv"
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df = df[df["split_set"] == "train_valid"].copy()

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)

    parser.add_argument("--oct0_tag", required=True)
    parser.add_argument("--oct1_tag", required=True)
    parser.add_argument("--octa3_tag", required=True)

    parser.add_argument("--feature_mode", choices=["logit","prob"], default="logit")
    parser.add_argument("--use_calib", type=lambda x: x=="True", default=True)

    args = parser.parse_args()

    feature_col = resolve_feature(args.feature_mode, args.use_calib)

    print("Feature column:", feature_col)

    df_oct0 = load_oof(args.model_name,"OCT0",args.oct0_tag)
    df_oct1 = load_oof(args.model_name,"OCT1",args.oct1_tag)
    df_octa3 = load_oof(args.model_name,"OCTA3",args.octa3_tag)

    df_oct0 = df_oct0.rename(columns={feature_col:"oct0_feat"})
    df_oct1 = df_oct1.rename(columns={feature_col:"oct1_feat"})
    df_octa3 = df_octa3.rename(columns={feature_col:"octa3_feat"})

    df_oct0 = df_oct0[[
        "exam_key",
        "patient_id",
        "fold_id",
        "y_true",
        "oct0_feat"
    ]]

    df_oct1 = df_oct1[[
        "exam_key",
        "oct1_feat"
    ]]

    df_octa3 = df_octa3[[
        "exam_key",
        "octa3_feat"
    ]]

    print("Merging modalities...")

    df_meta = df_oct0.merge(df_oct1,on="exam_key",how="inner")
    df_meta = df_meta.merge(df_octa3,on="exam_key",how="inner")

    print("Final meta samples:",len(df_meta))

    meta_tag = (
        f"OCT0_{args.oct0_tag}"
        f"__OCT1_{args.oct1_tag}"
        f"__OCTA3_{args.octa3_tag}"
        f"__{args.feature_mode}"
        f"__calib{args.use_calib}"
    )

    meta_dir = os.path.join(
        META_ROOT,
        args.model_name,
        meta_tag
    )

    ensure_dir(meta_dir)

    csv_path = os.path.join(meta_dir,"meta_dataset.csv")

    df_meta.to_csv(csv_path,index=False)

    info = {

        "model_name":args.model_name,

        "feature_mode":args.feature_mode,
        "use_calib":args.use_calib,
        "feature_column":feature_col,

        "num_samples":int(len(df_meta)),

        "modalities":[
            "OCT0",
            "OCT1",
            "OCTA3"
        ],

        "features":[
            "oct0_feat",
            "oct1_feat",
            "octa3_feat"
        ],

        "oof_sources":{
            "OCT0":args.oct0_tag,
            "OCT1":args.oct1_tag,
            "OCTA3":args.octa3_tag
        }
    }

    with open(os.path.join(meta_dir,"meta_dataset_info.json"),"w") as f:
        json.dump(info,f,indent=2)

    print("Saved:",csv_path)


if __name__ == "__main__":
    main()