# build_global_patient_fold_map.py

"""
Create synchronized patient-level 5-fold assignment from maximum data usage.

Aassumptions
------------------
master_manifest.csv keeps all usable exam units:
- OCT0-only exams are allowed
- OCT1-only exams are allowed
- OCTA3 is optional
- invalid rows have already been excluded by build_master_manifest.py

Goal
----
1. Strict patient-level separation
2. Same fold map for OCT0 / OCT1 / OCTA3
3. Later each base model uses:
   - has_oct0 == 1  (for OCT0 model)
   - has_oct1 == 1  (for OCT1 model)
   - has_octa3 == 1 (for OCTA3 model)
4. Later stacking LR uses intersection of aligned OOF predictions by exam_key
"""

import os
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ===================== CONFIG =====================
PROJECT_ROOT_DIR = "/data/Irene/SwinTransformer/Swin_Meta"

INPUT_DIR = os.path.join(
    PROJECT_ROOT_DIR,
    "outputs",
    "manifests",
    "master_split"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT_DIR,
    "outputs",
    "manifests",
    "global_patient_fold_split"
)

MASTER_MANIFEST_CSV = os.path.join(INPUT_DIR, "master_manifest.csv")

PATIENT_FOLD_MAP_CSV = os.path.join(OUTPUT_DIR, "patient_fold_map.csv")
FOLD_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "patient_fold_summary.json")

NUM_FOLDS = 5
RANDOM_SEED = 42


# ===================== UTILS =====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_master_dtypes(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Force key columns to consistent dtypes to avoid join / map / isin mismatch.
    """
    df = master_df.copy()

    for col in ["exam_key", "patient_id", "eye", "exam_date", "split_set"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

    flag_cols = [
        "has_oct0", "has_oct1", "has_octa3",
        "has_oct_pair", "is_complete_three_path",
        "label_conflict"
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "fold_id" in df.columns:
        df["fold_id"] = pd.to_numeric(df["fold_id"], errors="coerce")

    return df


def validate_master_manifest(master_df: pd.DataFrame):
    required_cols = [
        "exam_id",
        "split_set",
        "exam_key",
        "patient_id",
        "y_true",
        "has_oct0",
        "has_oct1",
        "has_octa3",
        "has_oct_pair",
        "is_complete_three_path",
        "label_conflict",
        "fold_id"
    ]

    missing = [c for c in required_cols if c not in master_df.columns]
    if missing:
        raise ValueError(f"Missing columns in master_manifest.csv: {missing}")

    if len(master_df) == 0:
        raise ValueError("master_manifest.csv is empty")

    train_valid_df = master_df[master_df["split_set"] == "train_valid"].copy()
    if len(train_valid_df) == 0:
        raise ValueError("No train_valid rows found")

    if train_valid_df["y_true"].isna().any():
        raise ValueError("Invalid manifest: missing labels detected")

    if train_valid_df["label_conflict"].sum() != 0:
        raise ValueError("Invalid manifest: label_conflict rows exist")

    if ((train_valid_df["has_oct0"] == 0) & (train_valid_df["has_oct1"] == 0)).any():
        raise ValueError("Invalid manifest: row without OCT0 and OCT1 exists")


# ===================== PATIENT LABEL TABLE =====================
def build_patient_level_label_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build patient-level stratification labels from all train_valid exams.
    """
    train_valid_df = master_df[
        master_df["split_set"] == "train_valid"
    ].copy()

    train_valid_df["patient_id"] = train_valid_df["patient_id"].astype(str)
    train_valid_df["y_true"] = train_valid_df["y_true"].astype(int)

    grouped = train_valid_df.groupby("patient_id", sort=True)

    rows: List[Dict] = []

    for patient_id, g in grouped:
        n_exam_total = int(len(g))
        n_exam_inactive = int((g["y_true"] == 0).sum())
        n_exam_active = int((g["y_true"] == 1).sum())

        has_mixed_exam_labels = int(
            n_exam_inactive > 0 and n_exam_active > 0
        )

        patient_label_for_stratify = (
            1 if n_exam_active >= n_exam_inactive else 0
        )

        rows.append({
            "patient_id": str(patient_id),
            "n_exam_total": n_exam_total,
            "n_exam_inactive": n_exam_inactive,
            "n_exam_active": n_exam_active,
            "has_mixed_exam_labels": has_mixed_exam_labels,
            "patient_label_for_stratify": patient_label_for_stratify,
            "fold_id": -1
        })

    patient_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)

    if len(patient_df) == 0:
        raise ValueError("No valid patients found")

    return patient_df


# ===================== SUMMARY =====================
def compute_fold_summary(patient_df: pd.DataFrame,
                         master_df_folded: pd.DataFrame) -> Dict:

    train_valid_df = master_df_folded[
        master_df_folded["split_set"] == "train_valid"
    ].copy()

    test_df = master_df_folded[
        master_df_folded["split_set"] == "test"
    ].copy()

    train_valid_df["patient_id"] = train_valid_df["patient_id"].astype(str)
    test_df["patient_id"] = test_df["patient_id"].astype(str)
    patient_df["patient_id"] = patient_df["patient_id"].astype(str)

    # ---------- Global counts ----------
    total_train_valid_patients = int(len(patient_df))
    mixed_label_patients = int((patient_df["has_mixed_exam_labels"] == 1).sum())

    global_train_valid_exam_units = int(len(train_valid_df))
    global_test_exam_units = int(len(test_df))

    global_train_valid_oct0 = int(train_valid_df["has_oct0"].sum())
    global_train_valid_oct1 = int(train_valid_df["has_oct1"].sum())
    global_train_valid_octa3 = int(train_valid_df["has_octa3"].sum())
    global_train_valid_oct_pair = int(train_valid_df["has_oct_pair"].sum())
    global_train_valid_complete_three = int(train_valid_df["is_complete_three_path"].sum())

    global_test_oct0 = int(test_df["has_oct0"].sum())
    global_test_oct1 = int(test_df["has_oct1"].sum())
    global_test_octa3 = int(test_df["has_octa3"].sum())
    global_test_oct_pair = int(test_df["has_oct_pair"].sum())
    global_test_complete_three = int(test_df["is_complete_three_path"].sum())

    # ---------- Arithmetic checks ----------
    if global_train_valid_oct0 < global_train_valid_oct_pair:
        raise RuntimeError("Train_valid OCT0 count is smaller than oct_pair count")
    if global_train_valid_oct1 < global_train_valid_oct_pair:
        raise RuntimeError("Train_valid OCT1 count is smaller than oct_pair count")
    if global_train_valid_complete_three > global_train_valid_oct_pair:
        raise RuntimeError("Train_valid complete_three_path exceeds oct_pair count")

    if global_test_oct0 < global_test_oct_pair:
        raise RuntimeError("Test OCT0 count is smaller than oct_pair count")
    if global_test_oct1 < global_test_oct_pair:
        raise RuntimeError("Test OCT1 count is smaller than oct_pair count")
    if global_test_complete_three > global_test_oct_pair:
        raise RuntimeError("Test complete_three_path exceeds oct_pair count")

    summary = {
        "fold_setup": {
            "num_folds": NUM_FOLDS,
            "random_seed": RANDOM_SEED,
        },

        "patient_statistics": {
            "train_valid_patients": total_train_valid_patients,
            "mixed_label_patients": mixed_label_patients,
            "patient_stratify_label_distribution": {
                str(k): int(v)
                for k, v in patient_df[
                    "patient_label_for_stratify"
                ].value_counts().sort_index().to_dict().items()
            },
        },

        "global_usable_exam_counts": {
            "train_valid": {
                "exam_units": global_train_valid_exam_units,
                "OCT0": global_train_valid_oct0,
                "OCT1": global_train_valid_oct1,
                "OCTA3": global_train_valid_octa3,
                "oct_pair": global_train_valid_oct_pair,
                "complete_three_path": global_train_valid_complete_three,
            },
            "test": {
                "exam_units": global_test_exam_units,
                "OCT0": global_test_oct0,
                "OCT1": global_test_oct1,
                "OCTA3": global_test_octa3,
                "oct_pair": global_test_oct_pair,
                "complete_three_path": global_test_complete_three,
            }
        },

        "fold_statistics": {}
    }

    total_fold_patients = 0
    total_fold_exam_units = 0
    total_fold_oct0 = 0
    total_fold_oct1 = 0
    total_fold_octa3 = 0
    total_fold_oct_pair = 0
    total_fold_complete_three = 0

    for fold_id in range(1, NUM_FOLDS + 1):
        fold_patients = set(
            patient_df[patient_df["fold_id"] == fold_id]["patient_id"].astype(str)
        )

        fold_df = train_valid_df[
            train_valid_df["patient_id"].astype(str).isin(fold_patients)
        ].copy()

        fold_num_patients = int(len(fold_patients))
        fold_exam_units = int(len(fold_df))
        fold_oct0 = int(fold_df["has_oct0"].sum())
        fold_oct1 = int(fold_df["has_oct1"].sum())
        fold_octa3 = int(fold_df["has_octa3"].sum())
        fold_oct_pair = int(fold_df["has_oct_pair"].sum())
        fold_complete_three = int(fold_df["is_complete_three_path"].sum())

        # fold-level arithmetic checks
        if fold_oct0 < fold_oct_pair:
            raise RuntimeError(f"Fold {fold_id}: OCT0 count is smaller than oct_pair count")
        if fold_oct1 < fold_oct_pair:
            raise RuntimeError(f"Fold {fold_id}: OCT1 count is smaller than oct_pair count")
        if fold_complete_three > fold_oct_pair:
            raise RuntimeError(f"Fold {fold_id}: complete_three_path exceeds oct_pair count")

        summary["fold_statistics"][str(fold_id)] = {
            "patients": fold_num_patients,
            "exam_units": fold_exam_units,
            "OCT0": fold_oct0,
            "OCT1": fold_oct1,
            "OCTA3": fold_octa3,
            "oct_pair": fold_oct_pair,
            "complete_three_path": fold_complete_three,
            "exam_label_distribution": {
                str(k): int(v)
                for k, v in fold_df["y_true"].astype(int).value_counts().sort_index().to_dict().items()
            }
        }

        total_fold_patients += fold_num_patients
        total_fold_exam_units += fold_exam_units
        total_fold_oct0 += fold_oct0
        total_fold_oct1 += fold_oct1
        total_fold_octa3 += fold_octa3
        total_fold_oct_pair += fold_oct_pair
        total_fold_complete_three += fold_complete_three

    # ---------- Global consistency checks ----------
    if total_fold_patients != total_train_valid_patients:
        raise RuntimeError(
            f"Fold patient sum mismatch: sum={total_fold_patients}, global={total_train_valid_patients}"
        )

    if total_fold_exam_units != global_train_valid_exam_units:
        raise RuntimeError(
            f"Fold exam unit sum mismatch: sum={total_fold_exam_units}, global={global_train_valid_exam_units}"
        )

    if total_fold_oct0 != global_train_valid_oct0:
        raise RuntimeError(
            f"Fold OCT0 sum mismatch: sum={total_fold_oct0}, global={global_train_valid_oct0}"
        )

    if total_fold_oct1 != global_train_valid_oct1:
        raise RuntimeError(
            f"Fold OCT1 sum mismatch: sum={total_fold_oct1}, global={global_train_valid_oct1}"
        )

    if total_fold_octa3 != global_train_valid_octa3:
        raise RuntimeError(
            f"Fold OCTA3 sum mismatch: sum={total_fold_octa3}, global={global_train_valid_octa3}"
        )

    if total_fold_oct_pair != global_train_valid_oct_pair:
        raise RuntimeError(
            f"Fold oct_pair sum mismatch: sum={total_fold_oct_pair}, global={global_train_valid_oct_pair}"
        )

    if total_fold_complete_three != global_train_valid_complete_three:
        raise RuntimeError(
            f"Fold complete_three_path sum mismatch: sum={total_fold_complete_three}, global={global_train_valid_complete_three}"
        )

    return summary


# ===================== MAIN =====================
def main():
    ensure_dir(OUTPUT_DIR)

    print("[1/6] Loading master manifest ...")
    if not os.path.isfile(MASTER_MANIFEST_CSV):
        raise FileNotFoundError(MASTER_MANIFEST_CSV)

    master_df = pd.read_csv(MASTER_MANIFEST_CSV)

    print("[2/6] Normalizing dtypes ...")
    master_df = normalize_master_dtypes(master_df)

    print("[3/6] Validating manifest ...")
    validate_master_manifest(master_df)

    print("[4/6] Building patient stratification table ...")
    patient_df = build_patient_level_label_table(master_df)

    if len(patient_df) < NUM_FOLDS:
        raise ValueError(f"Not enough patients ({len(patient_df)}) for {NUM_FOLDS}-fold split.")

    print("[5/6] Running StratifiedGroupKFold ...")
    X = np.zeros(len(patient_df))
    y = patient_df["patient_label_for_stratify"].astype(int).values
    groups = patient_df["patient_id"].astype(str).values

    sgkf = StratifiedGroupKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        patient_df.loc[val_idx, "fold_id"] = fold_idx

    if (patient_df["fold_id"] <= 0).any():
        raise RuntimeError("Some patients were not assigned fold_id")

    patient_df["patient_id"] = patient_df["patient_id"].astype(str)
    master_df["patient_id"] = master_df["patient_id"].astype(str)

    patient_fold_map = pd.Series(
        patient_df["fold_id"].values,
        index=patient_df["patient_id"].astype(str)
    )

    master_df_folded = master_df.copy()
    train_mask = master_df_folded["split_set"] == "train_valid"

    master_df_folded.loc[train_mask, "fold_id"] = (
        master_df_folded.loc[train_mask, "patient_id"]
        .astype(str)
        .map(patient_fold_map)
    )

    master_df_folded.loc[~train_mask, "fold_id"] = 0

    if master_df_folded.loc[train_mask, "fold_id"].isna().any():
        bad_patients = master_df_folded.loc[
            train_mask & master_df_folded["fold_id"].isna(),
            "patient_id"
        ].astype(str).unique().tolist()
        raise RuntimeError(
            f"Some train_valid rows failed to receive fold_id. Example patient_ids: {bad_patients[:10]}"
        )

    master_df_folded["fold_id"] = master_df_folded["fold_id"].astype(int)

    valid_fold_values = set(master_df_folded.loc[train_mask, "fold_id"].unique().tolist())
    if not valid_fold_values.issubset(set(range(1, NUM_FOLDS + 1))):
        raise RuntimeError(f"Unexpected train_valid fold values: {sorted(valid_fold_values)}")

    print("[6/6] Saving outputs ...")
    patient_df = patient_df.sort_values("patient_id").reset_index(drop=True)
    patient_df.to_csv(PATIENT_FOLD_MAP_CSV, index=False, encoding="utf-8-sig")

    # write fold back into master manifest
    master_df_folded.to_csv(MASTER_MANIFEST_CSV, index=False, encoding="utf-8-sig")

    summary = compute_fold_summary(patient_df.copy(), master_df_folded.copy())
    with open(FOLD_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SAVED FILES]")
    print(f"1. {PATIENT_FOLD_MAP_CSV}")
    print(f"2. {FOLD_SUMMARY_JSON}")

    print("\n[IMPORTANT]")
    print("fold_id = 1~5 for train_valid")
    print("fold_id = 0 for independent test set")
    print("fold_id also written back into master_manifest.csv")
    print("All modalities (OCT0 / OCT1 / OCTA3) share exactly the same patient-level fold map")
    print("Later base models should filter rows by has_oct0 / has_oct1 / has_octa3")
    print("Later stacking LR should align OOF predictions by exam_key using intersection only")


if __name__ == "__main__":
    main()