# build_meta_dataset.py

"""
Purpose
-------
Construct meta-training dataset for stacking ensemble.

Meta learner input
------------------
X_meta = [z_OCT0, z_OCT1, z_OCTA3]
y      = disease label

Input paths
-----------
/data/Irene/SwinTransformer/Swin_Meta/outputs/oof_predictions/

    swin_tiny/
        OCT0/<RUN_TAG>/all_folds_oof_calibrated.csv
        OCT1/<RUN_TAG>/all_folds_oof_calibrated.csv
        OCTA3/<RUN_TAG>/all_folds_oof_calibrated.csv

Output paths
------------
/data/Irene/SwinTransformer/Swin_Meta/outputs/meta_dataset/
    swin_tiny__logit__calibTrue/
        meta_train_oof.csv
        pairing_log.csv
        meta_dataset_info.json
"""

import os
import json
import pandas as pd


# =========================
# PROJECT PATH
# =========================

PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"

OOF_ROOT = os.path.join(PROJECT_ROOT, "outputs", "oof_predictions")
META_ROOT = os.path.join(PROJECT_ROOT, "outputs", "meta_dataset")


# =========================
# EXPERIMENT CONFIG
# =========================

MODEL_NAME = "swin_tiny"

RUN_TAGS = {
    "OCT0": "BS16_EP100_LR1e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9",
    "OCT1": "BS16_EP100_LR1e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8",
    "OCTA3": "BS16_EP100_LR1e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6",
}

FEATURE_TYPE = "logit"
USE_CALIB = True
OOF_SOURCE = "calibrated"


# =========================
# UTILS
# =========================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def resolve_feature():

    if FEATURE_TYPE == "logit" and USE_CALIB:
        return "logit_calib"

    if FEATURE_TYPE == "logit":
        return "logit_uncal"

    if FEATURE_TYPE == "prob" and USE_CALIB:
        return "prob_calib"

    if FEATURE_TYPE == "prob":
        return "prob_uncal"

    raise ValueError("Invalid feature config")


def load_oof(modality):

    run_tag = RUN_TAGS[modality]

    base = os.path.join(
        OOF_ROOT,
        MODEL_NAME,
        modality,
        run_tag
    )

    if OOF_SOURCE == "calibrated":
        path = os.path.join(base, "all_folds_oof_calibrated.csv")
    else:
        path = os.path.join(base, "all_folds_oof.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "split_set" in df.columns:
        df = df[df["split_set"] == "train_valid"].copy()

    return df, path


# =========================
# MAIN
# =========================

def main():

    feature_col = resolve_feature()

    df_oct0, path0 = load_oof("OCT0")
    df_oct1, path1 = load_oof("OCT1")
    df_octa3, path3 = load_oof("OCTA3")

    print("OCT0 samples:", len(df_oct0))
    print("OCT1 samples:", len(df_oct1))
    print("OCTA3 samples:", len(df_octa3))


    # rename features
    df_oct0 = df_oct0.rename(columns={feature_col:"oct0_feat"})
    df_oct1 = df_oct1.rename(columns={feature_col:"oct1_feat"})
    df_octa3 = df_octa3.rename(columns={feature_col:"octa3_feat"})


    # collect exam keys
    set_oct0 = set(df_oct0.exam_key)
    set_oct1 = set(df_oct1.exam_key)
    set_octa3 = set(df_octa3.exam_key)

    all_keys = sorted(set_oct0 | set_oct1 | set_octa3)


    pairing_records = []

    for k in all_keys:

        o0 = int(k in set_oct0)
        o1 = int(k in set_oct1)
        o3 = int(k in set_octa3)

        paired = int(o0 and o1 and o3)

        pairing_records.append({
            "exam_key":k,
            "oct0":o0,
            "oct1":o1,
            "octa3":o3,
            "paired":paired
        })

    pairing_df = pd.DataFrame(pairing_records)

    paired_keys = pairing_df[pairing_df.paired==1].exam_key


    # =========================
    # BUILD META DATASET
    # =========================

    df_oct0 = df_oct0[df_oct0.exam_key.isin(paired_keys)]
    df_oct1 = df_oct1[df_oct1.exam_key.isin(paired_keys)]
    df_octa3 = df_octa3[df_octa3.exam_key.isin(paired_keys)]

    df_oct0 = df_oct0[["exam_key","patient_id","fold_id","y_true","oct0_feat"]]
    df_oct1 = df_oct1[["exam_key","oct1_feat"]]
    df_octa3 = df_octa3[["exam_key","octa3_feat"]]

    df_meta = df_oct0.merge(df_oct1,on="exam_key")
    df_meta = df_meta.merge(df_octa3,on="exam_key")

    print("Final meta samples:",len(df_meta))


    # =========================
    # OUTPUT
    # =========================

    meta_tag = f"{MODEL_NAME}__{FEATURE_TYPE}__calib{USE_CALIB}"

    meta_dir = os.path.join(META_ROOT,meta_tag)

    ensure_dir(meta_dir)


    meta_csv = os.path.join(meta_dir,"meta_train_oof.csv")
    pairing_csv = os.path.join(meta_dir,"pairing_log.csv")
    info_json = os.path.join(meta_dir,"meta_dataset_info.json")


    df_meta.to_csv(meta_csv,index=False)
    pairing_df.to_csv(pairing_csv,index=False)


    summary = {

        "model_name":MODEL_NAME,
        "feature_type":FEATURE_TYPE,
        "use_calib":USE_CALIB,

        "oct0_samples":len(set_oct0),
        "oct1_samples":len(set_oct1),
        "octa3_samples":len(set_octa3),

        "paired_samples":len(df_meta),
        "unpaired_samples":len(pairing_df)-len(df_meta),

        "oof_paths":{
            "OCT0":path0,
            "OCT1":path1,
            "OCTA3":path3
        }
    }

    with open(info_json,"w") as f:
        json.dump(summary,f,indent=2)


    print("Saved meta dataset:",meta_csv)
    print("Saved pairing log:",pairing_csv)


if __name__ == "__main__":
    main()