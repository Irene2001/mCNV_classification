# build_meta_dataset.py
"""
Construct meta-training dataset for stacking ensemble
mCNV binary classification (active / inactive)

Input
-----
outputs/oof_predictions/<model>/<modality>/<RUN_TAG>/all_folds_oof_calibrated.csv
  (or all_folds_oof.csv when OOF_SOURCE = "raw")
  Columns required: exam_key, patient_id, fold_id, y_true, split_set,
                    logit_calib / logit_uncal / prob_calib / prob_uncal

Output
------
outputs/meta_dataset/<meta_tag>/<lr_folder>/
    meta_train_oof.csv      ← meta-learner input (inner-joined, train_valid only)
    pairing_log.csv         ← per-exam modality availability + pairing status
    meta_dataset_info.json  ← full provenance record

where:
  meta_tag  = "{MODEL_NAME}__{FEATURE_TYPE}__calib{USE_CALIB}"
              e.g. swin_tiny__logit__calibTrue
  lr_folder = "OCT0_LR<lr>_OCT1_LR<lr>_OCTA3_LR<lr>"
              LR is extracted from each modality's RUN_TAG
              e.g. OCT0_LR2e-06_OCT1_LR3e-06_OCTA3_LR4e-06     
image title: 
"""

import os
import re
import json
import pandas as pd


# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"
OOF_ROOT     = os.path.join(PROJECT_ROOT, "outputs", "oof_predictions")
META_ROOT    = os.path.join(PROJECT_ROOT, "outputs", "meta_dataset")


# ── Experiment config ─────────────────────────────────────────────────────────
MODEL_NAME = "swin_tiny"

# Each modality may have a different run_tag (and therefore a different LR).
# LR is automatically extracted from the run_tag string.
RUN_TAGS = {
    "OCT0":  "BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9",
    "OCT1":  "BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8",
    "OCTA3": "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6",
}

# Feature to use as meta-input:
#   FEATURE_TYPE = "logit"  +  USE_CALIB = True   →  logit_calib   (recommended)
#   FEATURE_TYPE = "logit"  +  USE_CALIB = False  →  logit_uncal
#   FEATURE_TYPE = "prob"   +  USE_CALIB = True   →  prob_calib
#   FEATURE_TYPE = "prob"   +  USE_CALIB = False  →  prob_uncal
FEATURE_TYPE = "logit"
USE_CALIB    = True

# OOF file to read:
#   "calibrated" → all_folds_oof_calibrated.csv  (produced by calibrate_oof_predictions.py)
#   "raw"        → all_folds_oof.csv             (produced by train_singlemode_oof.py)
OOF_SOURCE = "calibrated"

# Modality pairing strategy:
#   "inner"  → keep only exams present in ALL three modalities (strictest, no missing)
PAIRING_STRATEGY = "inner"


# ── Utilities ─────────────────────────────────────────────────────────────────
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def extract_lr(run_tag: str) -> str:
    """
    Extract the LR value string from a run_tag.

    Matches patterns like:
      LR2e-06, LR1e-6, LR0.001, LR2.5e-5
    Returns the matched value string, e.g. "2e-06", "0.001".
    Raises ValueError if no LR token is found.
    """
    m = re.search(
        r'LR([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)',
        run_tag,
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(
            f"Cannot extract LR from run_tag: '{run_tag}'\n"
            "Expected pattern: ...LR<value>... e.g. LR2e-06 or LR0.001"
        )
    return m.group(1)


def build_lr_folder(run_tags: dict) -> str:
    """
    Build the lr_folder name from RUN_TAGS.

    Format: OCT0_LR<lr>_OCT1_LR<lr>_OCTA3_LR<lr>
    Example: OCT0_LR2e-06_OCT1_LR3e-06_OCTA3_LR4e-06
    """
    parts = []
    for mod in ["OCT0", "OCT1", "OCTA3"]:
        lr = extract_lr(run_tags[mod])
        parts.append(f"{mod}_LR{lr}")
    return "_".join(parts)


def resolve_feature_col() -> str:
    """Map (FEATURE_TYPE, USE_CALIB) to the actual CSV column name."""
    mapping = {
        ("logit", True):  "logit_calib",
        ("logit", False): "logit_uncal",
        ("prob",  True):  "prob_calib",
        ("prob",  False): "prob_uncal",
    }
    key = (FEATURE_TYPE, USE_CALIB)
    if key not in mapping:
        raise ValueError(f"Invalid (FEATURE_TYPE, USE_CALIB) = {key}")
    return mapping[key]


def load_oof(modality: str, feature_col: str):
    """
    Load OOF CSV for one modality.

    Returns (df, path) where df is filtered to split_set == 'train_valid'
    and has columns: exam_key, patient_id, fold_id, y_true, <feature_col>.
    """
    run_tag = RUN_TAGS[modality]
    base    = os.path.join(OOF_ROOT, MODEL_NAME, modality, run_tag)

    filename = ("all_folds_oof_calibrated.csv"
                if OOF_SOURCE == "calibrated"
                else "all_folds_oof.csv")
    path = os.path.join(base, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"OOF file not found: {path}\n"
            f"Expected OOF_SOURCE='{OOF_SOURCE}'.  "
            f"Run {'calibrate_oof_predictions.py' if OOF_SOURCE == 'calibrated' else 'train_singlemode_oof.py'} first."
        )

    df = pd.read_csv(path)

    # Keep train_valid only (test set is never used here)
    if "split_set" in df.columns:
        df = df[df["split_set"] == "train_valid"].copy()

    # Validate required columns
    required = {"exam_key", "patient_id", "fold_id", "y_true", feature_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{modality} OOF missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df["exam_key"] = df["exam_key"].astype(str)
    df["y_true"]   = df["y_true"].astype(int)

    return df, path


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── Resolve feature column
    feature_col = resolve_feature_col()
    print(f"Feature column : {feature_col}")
    print(f"OOF source     : {OOF_SOURCE}")

    # ── Load OOF per modality
    df_oct0,  path0 = load_oof("OCT0",  feature_col)
    df_oct1,  path1 = load_oof("OCT1",  feature_col)
    df_octa3, path3 = load_oof("OCTA3", feature_col)

    print(f"\nOCT0  OOF samples : {len(df_oct0)}")
    print(f"OCT1  OOF samples : {len(df_oct1)}")
    print(f"OCTA3 OOF samples : {len(df_octa3)}")

    # ── Rename feature column to modality-prefixed name
    df_oct0  = df_oct0.rename( columns={feature_col: "oct0_feat"})
    df_oct1  = df_oct1.rename( columns={feature_col: "oct1_feat"})
    df_octa3 = df_octa3.rename(columns={feature_col: "octa3_feat"})

    # ── Build pairing log (union of all exam keys)
    set_oct0  = set(df_oct0["exam_key"])
    set_oct1  = set(df_oct1["exam_key"])
    set_octa3 = set(df_octa3["exam_key"])
    all_keys  = sorted(set_oct0 | set_oct1 | set_octa3)

    pairing_records = []
    for k in all_keys:
        o0 = int(k in set_oct0)
        o1 = int(k in set_oct1)
        o3 = int(k in set_octa3)
        pairing_records.append({
            "exam_key": k,
            "has_oct0": o0,
            "has_oct1": o1,
            "has_octa3": o3,
            "paired": int(o0 and o1 and o3),
        })
    pairing_df = pd.DataFrame(pairing_records)

    n_paired   = int(pairing_df["paired"].sum())
    n_unpaired = len(pairing_df) - n_paired
    print(f"\nPairing ({PAIRING_STRATEGY}): "
          f"paired={n_paired}  unpaired={n_unpaired}")

    # ── Filter to paired exams only
    paired_keys = pairing_df.loc[pairing_df["paired"] == 1, "exam_key"]

    df_oct0  = df_oct0[df_oct0["exam_key"].isin(paired_keys)]
    df_oct1  = df_oct1[df_oct1["exam_key"].isin(paired_keys)]
    df_octa3 = df_octa3[df_octa3["exam_key"].isin(paired_keys)]

    # ── Keep only columns needed for meta-train
    df_oct0  = df_oct0[["exam_key", "patient_id", "fold_id", "y_true", "oct0_feat"]]
    df_oct1  = df_oct1[["exam_key", "oct1_feat"]]
    df_octa3 = df_octa3[["exam_key", "octa3_feat"]]

    # ── Inner join on exam_key
    df_meta = df_oct0.merge(df_oct1,  on="exam_key", how="inner")
    df_meta = df_meta.merge(df_octa3, on="exam_key", how="inner")
    df_meta = df_meta.sort_values("exam_key").reset_index(drop=True)

    n_meta = len(df_meta)
    na     = int((df_meta["y_true"] == 1).sum())
    ni     = int((df_meta["y_true"] == 0).sum())
    print(f"\nFinal meta-train : {n_meta} samples  "
          f"active={na} ({na/n_meta*100:.1f}%)  "
          f"inactive={ni} ({ni/n_meta*100:.1f}%)  "
          f"ratio={ni/na:.2f}:1")

    # ── Build output paths
    meta_tag  = f"{MODEL_NAME}__{FEATURE_TYPE}__calib{USE_CALIB}"
    lr_folder = build_lr_folder(RUN_TAGS)
    out_dir   = os.path.join(META_ROOT, meta_tag, lr_folder)
    ensure_dir(out_dir)

    print(f"\nmeta_tag  : {meta_tag}")
    print(f"lr_folder : {lr_folder}")
    print(f"out_dir   : {out_dir}")

    # ── Save outputs
    meta_csv    = os.path.join(out_dir, "meta_train_oof.csv")
    pairing_csv = os.path.join(out_dir, "pairing_log.csv")
    info_json   = os.path.join(out_dir, "meta_dataset_info.json")

    df_meta.to_csv(meta_csv, index=False, encoding="utf-8-sig")
    pairing_df.to_csv(pairing_csv, index=False, encoding="utf-8-sig")

    info = {
        "model_name":       MODEL_NAME,
        "feature_type":     FEATURE_TYPE,
        "use_calib":        USE_CALIB,
        "oof_source":       OOF_SOURCE,
        "feature_col":      feature_col,
        "pairing_strategy": PAIRING_STRATEGY,
        "meta_tag":         meta_tag,
        "lr_folder":        lr_folder,
        "run_tags":         RUN_TAGS,
        "lr_per_modality": {
            mod: extract_lr(RUN_TAGS[mod])
            for mod in ["OCT0", "OCT1", "OCTA3"]
        },
        "samples": {
            "oct0_oof":   len(set_oct0),
            "oct1_oof":   len(set_oct1),
            "octa3_oof":  len(set_octa3),
            "paired":     n_paired,
            "unpaired":   n_unpaired,
            "meta_train": n_meta,
            "active":     na,
            "inactive":   ni,
            "imbalance_ratio": round(ni / na, 4),
        },
        "oof_paths": {
            "OCT0":  path0,
            "OCT1":  path1,
            "OCTA3": path3,
        },
        "output_files": {
            "meta_train_oof":  meta_csv,
            "pairing_log":     pairing_csv,
            "meta_dataset_info": info_json,
        },
    }

    with open(info_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {meta_csv}")
    print(f"Saved: {pairing_csv}")
    print(f"Saved: {info_json}")


if __name__ == "__main__":
    main()