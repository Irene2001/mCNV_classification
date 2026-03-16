# test_meta_logistic_regression.py

"""
Run independent hold-out test evaluation for the stacking meta-model.

Inputs
------
- outputs/manifests/master_split/paired_test_manifest.csv
- outputs/results/{model_name}/{model_name}_oct0_test_predictions.csv
- outputs/results/{model_name}/{model_name}_oct1_test_predictions.csv
- outputs/results/{model_name}/{model_name}_octa3_test_predictions.csv
- outputs/models/{model_name}_meta_logistic_regression.pkl

Outputs
-------
- outputs/meta/{model_name}_meta_test_predictions.csv
- outputs/results/{model_name}/{model_name}_stacking_test_metrics.json

Notes
-----
- Only complete three-path paired test exams are evaluated.
- This script assumes OCT0/OCT1/OCTA3 test predictions were already generated
  by testing/test_singlemode.py
"""

import os
import json
import argparse
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

from training.model_factory import normalize_model_name


# ===================== CONFIG =====================
PROJECT_ROOT_DIR = "/data/Irene/SwinTransformer/Swin_OOF"

PAIRED_TEST_MANIFEST_CSV = os.path.join(
    PROJECT_ROOT_DIR,
    "outputs",
    "manifests",
    "master_split",
    "paired_test_manifest.csv",
)

RESULTS_ROOT = os.path.join(PROJECT_ROOT_DIR, "outputs", "results")
META_ROOT = os.path.join(PROJECT_ROOT_DIR, "outputs", "meta")
MODEL_ROOT = os.path.join(PROJECT_ROOT_DIR, "outputs", "models")

FEATURE_COLUMNS = [
    "prob_oct0",
    "prob_oct1",
    "prob_octa3",
]


# ===================== UTILS =====================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_paired_test_manifest() -> pd.DataFrame:
    if not os.path.isfile(PAIRED_TEST_MANIFEST_CSV):
        raise FileNotFoundError(PAIRED_TEST_MANIFEST_CSV)

    df = pd.read_csv(PAIRED_TEST_MANIFEST_CSV)

    required_cols = [
        "pair_id",
        "pair_key",
        "patient_id",
        "eye",
        "exam_date",
        "y_true",
        "oct0_dataset_index",
        "oct1_dataset_index",
        "octa3_dataset_index",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"paired_test_manifest.csv missing columns: {missing}")

    df = df[df["y_true"].notna()].copy()
    for c in ["oct0_dataset_index", "oct1_dataset_index", "octa3_dataset_index"]:
        df = df[df[c].notna()].copy()

    df["pair_id"] = df["pair_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)
    df["oct0_dataset_index"] = df["oct0_dataset_index"].astype(int)
    df["oct1_dataset_index"] = df["oct1_dataset_index"].astype(int)
    df["octa3_dataset_index"] = df["octa3_dataset_index"].astype(int)

    return df.reset_index(drop=True)


def load_singlemode_test_predictions(model_name: str, modality: str) -> pd.DataFrame:
    pred_csv = os.path.join(
        RESULTS_ROOT,
        model_name,
        f"{model_name}_{modality.lower()}_test_predictions.csv"
    )
    if not os.path.isfile(pred_csv):
        raise FileNotFoundError(pred_csv)

    df = pd.read_csv(pred_csv)
    required_cols = [
        "dataset_index", "patient_id", "pair_id", "pair_key",
        "image_path", "image_slot", "y_true",
        "ensemble_logit", "ensemble_prob", "ensemble_pred"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{pred_csv} missing columns: {missing}")

    df["dataset_index"] = df["dataset_index"].astype(int)
    df["pair_id"] = df["pair_id"].astype(int)
    df["y_true"] = df["y_true"].astype(int)

    # rename for modality-specific merge
    rename_map = {
        "dataset_index": f"{modality.lower()}_dataset_index",
        "image_path": f"{modality.lower()}_image_path",
        "image_slot": f"{modality.lower()}_image_slot",
        "ensemble_logit": f"logit_{modality.lower()}",
        "ensemble_prob": f"prob_{modality.lower()}",
        "ensemble_pred": f"pred_{modality.lower()}",
    }
    return df.rename(columns=rename_map)


# ===================== MAIN =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["swin_tiny", "vgg16", "efficientnet_b0"])
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.model_name)

    ensure_dir(META_ROOT)
    ensure_dir(os.path.join(RESULTS_ROOT, model_name))

    paired_test_df = load_paired_test_manifest()
    oct0_df = load_singlemode_test_predictions(model_name, "OCT0")
    oct1_df = load_singlemode_test_predictions(model_name, "OCT1")
    octa3_df = load_singlemode_test_predictions(model_name, "OCTA3")

    # merge with paired test manifest
    meta_test_df = paired_test_df.merge(
        oct0_df,
        how="inner",
        left_on="oct0_dataset_index",
        right_on="oct0_dataset_index",
    )

    meta_test_df = meta_test_df.merge(
        oct1_df,
        how="inner",
        left_on="oct1_dataset_index",
        right_on="oct1_dataset_index",
        suffixes=("", "_dup_oct1"),
    )

    meta_test_df = meta_test_df.merge(
        octa3_df,
        how="inner",
        left_on="octa3_dataset_index",
        right_on="octa3_dataset_index",
        suffixes=("", "_dup_octa3"),
    )

    if len(meta_test_df) == 0:
        raise RuntimeError("No complete paired test samples remained after merging OCT0/OCT1/OCTA3 predictions.")

    # consistency checks
    if "patient_id_dup_oct1" in meta_test_df.columns:
        if not (meta_test_df["patient_id"].astype(str) == meta_test_df["patient_id_dup_oct1"].astype(str)).all():
            raise ValueError("patient_id mismatch between paired test manifest and OCT1 predictions.")
    if "patient_id_dup_octa3" in meta_test_df.columns:
        if not (meta_test_df["patient_id"].astype(str) == meta_test_df["patient_id_dup_octa3"].astype(str)).all():
            raise ValueError("patient_id mismatch between paired test manifest and OCTA3 predictions.")

    if "pair_key_dup_oct1" in meta_test_df.columns:
        if not (meta_test_df["pair_key"].astype(str) == meta_test_df["pair_key_dup_oct1"].astype(str)).all():
            raise ValueError("pair_key mismatch between paired test manifest and OCT1 predictions.")
    if "pair_key_dup_octa3" in meta_test_df.columns:
        if not (meta_test_df["pair_key"].astype(str) == meta_test_df["pair_key_dup_octa3"].astype(str)).all():
            raise ValueError("pair_key mismatch between paired test manifest and OCTA3 predictions.")

    if "y_true_dup_oct1" in meta_test_df.columns:
        if not (meta_test_df["y_true"].astype(int) == meta_test_df["y_true_dup_oct1"].astype(int)).all():
            raise ValueError("y_true mismatch between paired test manifest and OCT1 predictions.")
    if "y_true_dup_octa3" in meta_test_df.columns:
        if not (meta_test_df["y_true"].astype(int) == meta_test_df["y_true_dup_octa3"].astype(int)).all():
            raise ValueError("y_true mismatch between paired test manifest and OCTA3 predictions.")

    # load meta model
    meta_model_path = os.path.join(MODEL_ROOT, f"{model_name}_meta_logistic_regression.pkl")
    if not os.path.isfile(meta_model_path):
        raise FileNotFoundError(meta_model_path)

    with open(meta_model_path, "rb") as f:
        meta_model = pickle.load(f)

    X_meta_test = meta_test_df[FEATURE_COLUMNS].astype(float).values
    y_true = meta_test_df["y_true"].astype(int).values

    y_prob = meta_model.predict_proba(X_meta_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    # export predictions
    pred_out = meta_test_df[
        [
            "pair_id", "pair_key", "patient_id", "eye", "exam_date", "y_true",
            "prob_oct0", "prob_oct1", "prob_octa3"
        ]
    ].copy()
    pred_out["meta_prob"] = y_prob
    pred_out["meta_pred"] = y_pred

    pred_csv = os.path.join(META_ROOT, f"{model_name}_meta_test_predictions.csv")
    pred_out.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    metrics_json = os.path.join(RESULTS_ROOT, model_name, f"{model_name}_stacking_test_metrics.json")
    summary = {
        "model_name": model_name,
        "meta_model_path": meta_model_path,
        "num_test_pairs": int(len(pred_out)),
        "num_test_patients": int(pred_out["patient_id"].astype(str).nunique()),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "meta_test_prediction_csv": pred_csv,
    }
    save_json(metrics_json, summary)

    print("\n[SAVED]")
    print(f"Meta test predictions : {pred_csv}")
    print(f"Stacking test metrics : {metrics_json}")


if __name__ == "__main__":
    main()