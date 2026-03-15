# build_master_manifest.py

"""
Build ONE unified exam-level master manifest for mCNV experiments.
(Each base model uses the maximum available data of its own modality.)

Design rules
------------
1. One row = one exam unit = (patient_id, eye, exam_date)
2. OCT0 and OCT1 are usually expected together, but after manual QC deletion,
   OCT0-only or OCT1-only exams are allowed to remain
3. OCTA3 is optional
4. Do NOT discard unmatched OCT exams
5. Keep all usable exam-level rows for:
   - single-mode training/testing
   - synchronized patient-level 5-fold split
   - later OOF stacking logistic regression (using aligned intersection only)
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ===================== CONFIG =====================
DATA_BASE_DIR = "/data/Irene/Correct_Preprocessed_delete"
PROJECT_ROOT_DIR = "/data/Irene/SwinTransformer/Swin_Meta"
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "outputs", "manifests", "master_split")

CLASS_TO_LABEL = {
    "inactive": 0,
    "active": 1,
}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DATA_SPLITS = [
    {
        "split_set": "train_valid",
        "roots": [
            {
                "source_modality": "OCT",
                "root_dir": os.path.join(DATA_BASE_DIR, "OCT", "OCT_train_val"),
            },
            {
                "source_modality": "OCTA3",
                "root_dir": os.path.join(DATA_BASE_DIR, "OCTA3", "OCTA3_train_val"),
            },
        ],
    },
    {
        "split_set": "test",
        "roots": [
            {
                "source_modality": "OCT",
                "root_dir": os.path.join(DATA_BASE_DIR, "OCT", "OCT_test"),
            },
            {
                "source_modality": "OCTA3",
                "root_dir": os.path.join(DATA_BASE_DIR, "OCTA3", "OCTA3_test"),
            },
        ],
    },
]

MASTER_MANIFEST_CSV = os.path.join(OUTPUT_DIR, "master_manifest.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "manifest_summary.json")


# ===================== UTILS =====================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_path(path: str) -> str:
    return os.path.abspath(path)


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(VALID_EXTS)


def parse_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Expected examples:
      1443067OS 20170703_OCT0.jpg
      1443067OS 20170703_OCT1.jpg
      1443067OS 20170703_OCTA3.jpg
    """
    base_name = os.path.splitext(os.path.basename(filename))[0].strip()

    patterns = [
        r"^(?P<patient_id>\d+)(?P<eye>OD|OS)[ _-]?(?P<exam_date>\d{8})[ _-](?P<slot>OCT0|OCT1|OCTA3)$",
        r"^(?P<patient_id>\d+)(?P<eye>OD|OS)\s+(?P<exam_date>\d{8})_(?P<slot>OCT0|OCT1|OCTA3)$",
        r"^(?P<patient_id>\d+)(?P<eye>OD|OS)_(?P<exam_date>\d{8})_(?P<slot>OCT0|OCT1|OCTA3)$",
        r"^(?P<patient_id>\d+)(?P<eye>OD|OS)-(?P<exam_date>\d{8})-(?P<slot>OCT0|OCT1|OCTA3)$",
    ]

    for pat in patterns:
        m = re.match(pat, base_name, flags=re.IGNORECASE)
        if m:
            gd = m.groupdict()
            return {
                "patient_id": gd.get("patient_id"),
                "eye": gd.get("eye").upper() if gd.get("eye") else None,
                "exam_date": gd.get("exam_date"),
                "image_slot": gd.get("slot").upper() if gd.get("slot") else None,
                "parse_ok": True,
                "base_name": base_name,
            }

    # fallback parse for audit only
    patient_id = None
    eye = None
    exam_date = None
    image_slot = None

    m_pid = re.match(r"^(\d+)", base_name)
    if m_pid:
        patient_id = m_pid.group(1)

    m_eye = re.search(r"(OD|OS)", base_name, flags=re.IGNORECASE)
    if m_eye:
        eye = m_eye.group(1).upper()

    m_date = re.search(r"(\d{8})", base_name)
    if m_date:
        exam_date = m_date.group(1)

    m_slot = re.search(r"(OCT0|OCT1|OCTA3)$", base_name, flags=re.IGNORECASE)
    if m_slot:
        image_slot = m_slot.group(1).upper()

    parse_ok = all([patient_id, eye, exam_date, image_slot])

    return {
        "patient_id": patient_id,
        "eye": eye,
        "exam_date": exam_date,
        "image_slot": image_slot,
        "parse_ok": parse_ok,
        "base_name": base_name,
    }


def exam_key_from_parts(patient_id: str, eye: str, exam_date: str) -> str:
    return f"{patient_id}_{eye}_{exam_date}"


def choose_single_row(group_df: pd.DataFrame, slot_name: str) -> Tuple[Optional[str], int]:
    """
    Choose exactly one representative image path for a slot.
    Return:
        chosen_path, num_candidates
    """
    target = group_df[group_df["image_slot"] == slot_name].sort_values("image_path")
    if len(target) == 0:
        return None, 0
    row = target.iloc[0]
    return str(row["image_path"]), int(len(target))


# ===================== BUILD IMAGE RECORDS =====================
def build_image_records() -> pd.DataFrame:
    rows: List[Dict] = []

    for split_cfg in DATA_SPLITS:
        split_set = split_cfg["split_set"]

        for root_cfg in split_cfg["roots"]:
            source_modality = root_cfg["source_modality"]
            root_dir = root_cfg["root_dir"]

            if not os.path.isdir(root_dir):
                print(f"[WARN] root dir not found, skip: {root_dir}")
                continue

            for class_name, y_true in CLASS_TO_LABEL.items():
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    print(f"[WARN] class dir not found, skip: {class_dir}")
                    continue

                for fn in sorted(os.listdir(class_dir)):
                    if not is_image_file(fn):
                        continue

                    full_path = normalize_path(os.path.join(class_dir, fn))
                    parsed = parse_filename(fn)

                    patient_id = parsed["patient_id"]
                    eye = parsed["eye"]
                    exam_date = parsed["exam_date"]
                    image_slot = parsed["image_slot"]

                    exam_key = None
                    if patient_id and eye and exam_date:
                        exam_key = exam_key_from_parts(patient_id, eye, exam_date)

                    rows.append({
                        "split_set": split_set,
                        "source_modality": source_modality,
                        "image_path": full_path,
                        "base_name": parsed["base_name"],
                        "image_slot": image_slot,
                        "class_name": class_name,
                        "y_true": y_true,
                        "patient_id": patient_id,
                        "eye": eye,
                        "exam_date": exam_date,
                        "exam_key": exam_key,
                        "parse_ok": int(bool(parsed["parse_ok"])),
                    })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No images found. Please check DATA_BASE_DIR.")

    df = df.sort_values(
        by=[
            "split_set", "source_modality", "class_name",
            "patient_id", "eye", "exam_date", "image_slot", "base_name"
        ]
    ).reset_index(drop=True)

    return df


# ===================== BUILD MASTER MANIFEST =====================
def build_master_manifest(image_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Build unified exam-level manifest.

    Rule:
    - Keep OCT0-only / OCT1-only rows
    - Keep OCT pair rows
    - Keep OCTA3 optional
    - Exclude only obviously invalid rows:
        * parse failure
        * missing exam_key
        * missing y_true
        * label_conflict
        * no OCT0 and no OCT1
        * duplicate OCT0 or duplicate OCT1 candidates (strictly exclude)
    """
    valid_df = image_df[
        (image_df["parse_ok"] == 1) &
        (image_df["exam_key"].notna()) &
        (image_df["image_slot"].isin(["OCT0", "OCT1", "OCTA3"]))
    ].copy()

    grouped = valid_df.groupby(
        ["split_set", "exam_key", "patient_id", "eye", "exam_date"],
        dropna=False
    )

    kept_rows: List[Dict] = []
    audit_rows: List[Dict] = []

    for (split_set, exam_key, patient_id, eye, exam_date), g in grouped:
        unique_labels = sorted(g["y_true"].dropna().unique().tolist())
        label_conflict = int(len(unique_labels) > 1)
        y_true = unique_labels[0] if len(unique_labels) >= 1 else None

        oct0_path, n_oct0 = choose_single_row(g, "OCT0")
        oct1_path, n_oct1 = choose_single_row(g, "OCT1")
        octa3_path, n_octa3 = choose_single_row(g, "OCTA3")

        has_oct0 = int(oct0_path is not None)
        has_oct1 = int(oct1_path is not None)
        has_octa3 = int(octa3_path is not None)

        has_oct_pair = int(has_oct0 and has_oct1)
        is_complete_three_path = int(has_oct0 and has_oct1 and has_octa3)

        oct0_duplicate = int(n_oct0 > 1)
        oct1_duplicate = int(n_oct1 > 1)

        invalid_reason = None

        if label_conflict == 1:
            invalid_reason = "label_conflict"
        elif y_true is None:
            invalid_reason = "missing_label"
        elif has_oct0 == 0 and has_oct1 == 0:
            invalid_reason = "missing_oct_pair"
        elif oct0_duplicate == 1 or oct1_duplicate == 1:
            invalid_reason = "duplicate_oct_pair_candidates"

        row = {
            "exam_id": None,
            "split_set": split_set,
            "exam_key": exam_key,
            "patient_id": str(patient_id),
            "eye": str(eye),
            "exam_date": str(exam_date),
            "y_true": y_true,
            "label_conflict": label_conflict,

            "oct0_image_path": oct0_path,
            "oct1_image_path": oct1_path,
            "octa3_image_path": octa3_path,

            "n_oct0_candidates": n_oct0,
            "n_oct1_candidates": n_oct1,
            "n_octa3_candidates": n_octa3,

            "has_oct0": has_oct0,
            "has_oct1": has_oct1,
            "has_octa3": has_octa3,
            "has_oct_pair": has_oct_pair,
            "is_complete_three_path": is_complete_three_path,

            "fold_id": None,
        }

        audit_row = dict(row)
        audit_row["invalid_reason"] = invalid_reason
        audit_rows.append(audit_row)

        if invalid_reason is None:
            kept_rows.append(row)

    master_df = pd.DataFrame(kept_rows).sort_values(
        by=["split_set", "patient_id", "eye", "exam_date"]
    ).reset_index(drop=True)
    master_df["exam_id"] = range(len(master_df))

    audit_df = pd.DataFrame(audit_rows).sort_values(
        by=["split_set", "patient_id", "eye", "exam_date"]
    ).reset_index(drop=True)

    summary = build_summary(image_df=image_df, audit_df=audit_df, master_df=master_df)
    return master_df, summary


# ===================== SUMMARY =====================
def build_summary(image_df: pd.DataFrame,
                  audit_df: pd.DataFrame,
                  master_df: pd.DataFrame) -> Dict:
    """
    Keep only important training numbers:
    1. Total image counts (train_valid / test / total)
    2. Usable counts for each base model (OCT0 / OCT1 / OCTA3)
    3. Pairing statistics
    """

    raw_train_valid = image_df[image_df["split_set"] == "train_valid"].copy()
    raw_test = image_df[image_df["split_set"] == "test"].copy()

    master_train_valid = master_df[master_df["split_set"] == "train_valid"].copy()
    master_test = master_df[master_df["split_set"] == "test"].copy()

    invalid_df = audit_df[audit_df["invalid_reason"].notna()].copy()

    # ---------- 1) Total image counts ----------
    total_images = int(len(image_df))
    train_valid_images = int(len(raw_train_valid))
    test_images = int(len(raw_test))
    parse_ok_images = int((image_df["parse_ok"] == 1).sum())
    parse_fail_images = int((image_df["parse_ok"] == 0).sum())

    # arithmetic checks
    if total_images != train_valid_images + test_images:
        raise RuntimeError(
            f"Image count mismatch: total={total_images}, "
            f"train_valid+test={train_valid_images + test_images}"
        )
    if total_images != parse_ok_images + parse_fail_images:
        raise RuntimeError(
            f"Parse count mismatch: total={total_images}, "
            f"parse_ok+parse_fail={parse_ok_images + parse_fail_images}"
        )

    # ---------- 2) Base model usable counts ----------
    usable_oct0_train_valid = int(master_train_valid["has_oct0"].sum())
    usable_oct1_train_valid = int(master_train_valid["has_oct1"].sum())
    usable_octa3_train_valid = int(master_train_valid["has_octa3"].sum())

    usable_oct0_test = int(master_test["has_oct0"].sum())
    usable_oct1_test = int(master_test["has_oct1"].sum())
    usable_octa3_test = int(master_test["has_octa3"].sum())

    usable_oct0_total = usable_oct0_train_valid + usable_oct0_test
    usable_oct1_total = usable_oct1_train_valid + usable_oct1_test
    usable_octa3_total = usable_octa3_train_valid + usable_octa3_test

    # ---------- 3) Pairing statistics ----------
    train_valid_oct0_only = int(((master_train_valid["has_oct0"] == 1) & (master_train_valid["has_oct1"] == 0)).sum())
    train_valid_oct1_only = int(((master_train_valid["has_oct0"] == 0) & (master_train_valid["has_oct1"] == 1)).sum())
    train_valid_oct_pair = int(master_train_valid["has_oct_pair"].sum())
    train_valid_complete_three_path = int(master_train_valid["is_complete_three_path"].sum())

    test_oct0_only = int(((master_test["has_oct0"] == 1) & (master_test["has_oct1"] == 0)).sum())
    test_oct1_only = int(((master_test["has_oct0"] == 0) & (master_test["has_oct1"] == 1)).sum())
    test_oct_pair = int(master_test["has_oct_pair"].sum())
    test_complete_three_path = int(master_test["is_complete_three_path"].sum())

    total_oct0_only = train_valid_oct0_only + test_oct0_only
    total_oct1_only = train_valid_oct1_only + test_oct1_only
    total_oct_pair = train_valid_oct_pair + test_oct_pair
    total_complete_three_path = train_valid_complete_three_path + test_complete_three_path

    # ---------- Excluded exam statistics ----------
    excluded_exam_total = int(len(invalid_df))
    excluded_by_reason = {
        str(k): int(v)
        for k, v in invalid_df["invalid_reason"].value_counts().sort_index().to_dict().items()
    }

    # ---------- Arithmetic checks on exam-level logic ----------
    train_valid_exam_units = int(len(master_train_valid))
    test_exam_units = int(len(master_test))
    total_exam_units = int(len(master_df))

    if total_exam_units != train_valid_exam_units + test_exam_units:
        raise RuntimeError(
            f"Exam unit mismatch: total={total_exam_units}, "
            f"train_valid+test={train_valid_exam_units + test_exam_units}"
        )

    # OCT availability must decompose into pair + single-only
    if usable_oct0_train_valid != train_valid_oct_pair + train_valid_oct0_only:
        raise RuntimeError(
            f"Train_valid OCT0 arithmetic mismatch: "
            f"usable_oct0={usable_oct0_train_valid}, "
            f"oct_pair+oct0_only={train_valid_oct_pair + train_valid_oct0_only}"
        )
    if usable_oct1_train_valid != train_valid_oct_pair + train_valid_oct1_only:
        raise RuntimeError(
            f"Train_valid OCT1 arithmetic mismatch: "
            f"usable_oct1={usable_oct1_train_valid}, "
            f"oct_pair+oct1_only={train_valid_oct_pair + train_valid_oct1_only}"
        )
    if usable_oct0_test != test_oct_pair + test_oct0_only:
        raise RuntimeError(
            f"Test OCT0 arithmetic mismatch: "
            f"usable_oct0={usable_oct0_test}, "
            f"oct_pair+oct0_only={test_oct_pair + test_oct0_only}"
        )
    if usable_oct1_test != test_oct_pair + test_oct1_only:
        raise RuntimeError(
            f"Test OCT1 arithmetic mismatch: "
            f"usable_oct1={usable_oct1_test}, "
            f"oct_pair+oct1_only={test_oct_pair + test_oct1_only}"
        )

    # complete_three_path cannot exceed pair count
    if train_valid_complete_three_path > train_valid_oct_pair:
        raise RuntimeError("Train_valid complete_three_path exceeds oct_pair")
    if test_complete_three_path > test_oct_pair:
        raise RuntimeError("Test complete_three_path exceeds oct_pair")

    summary = {
        "image_counts": {
            "total_images": total_images,
            "train_valid_images": train_valid_images,
            "test_images": test_images,
            "parse_ok_images": parse_ok_images,
            "parse_fail_images": parse_fail_images,
        },

        "base_model_usable_image_counts": {
            "total": {
                "OCT0": usable_oct0_total,
                "OCT1": usable_oct1_total,
                "OCTA3": usable_octa3_total,
            },
            "train_valid": {
                "OCT0": usable_oct0_train_valid,
                "OCT1": usable_oct1_train_valid,
                "OCTA3": usable_octa3_train_valid,
            },
            "test": {
                "OCT0": usable_oct0_test,
                "OCT1": usable_oct1_test,
                "OCTA3": usable_octa3_test,
            },
        },

        "pairing_statistics": {
            "total": {
                "oct0_only": total_oct0_only,
                "oct1_only": total_oct1_only,
                "oct_pair": total_oct_pair,
                "complete_three_path": total_complete_three_path,
            },
            "train_valid": {
                "oct0_only": train_valid_oct0_only,
                "oct1_only": train_valid_oct1_only,
                "oct_pair": train_valid_oct_pair,
                "complete_three_path": train_valid_complete_three_path,
            },
            "test": {
                "oct0_only": test_oct0_only,
                "oct1_only": test_oct1_only,
                "oct_pair": test_oct_pair,
                "complete_three_path": test_complete_three_path,
            },
        },

        "exam_manifest_statistics": {
            "total_exam_units": total_exam_units,
            "train_valid_exam_units": train_valid_exam_units,
            "test_exam_units": test_exam_units,
            "train_valid_patients": int(master_train_valid["patient_id"].nunique()) if len(master_train_valid) else 0,
            "test_patients": int(master_test["patient_id"].nunique()) if len(master_test) else 0,
        },

        "excluded_exam_statistics": {
            "excluded_exam_units_total": excluded_exam_total,
            "excluded_by_reason": excluded_by_reason,
        },
    }

    return summary


# ===================== MAIN =====================
def main():
    ensure_dir(OUTPUT_DIR)

    print("[1/3] Building image records ...")
    image_df = build_image_records()

    print("[2/3] Building unified exam-level master manifest ...")
    master_df, summary = build_master_manifest(image_df)

    print("[3/3] Saving outputs ...")
    master_df.to_csv(MASTER_MANIFEST_CSV, index=False, encoding="utf-8-sig")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SAVED FILES]")
    print(f"1. {MASTER_MANIFEST_CSV}")
    print(f"2. {SUMMARY_JSON}")

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()