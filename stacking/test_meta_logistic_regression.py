# test_meta_logistic_regression.py

"""
Stacking Meta-Learner (Logistic Regression) — Independent Test Set Evaluation

Input paths (from test_singlemode.py)
  TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/test_preds.csv
"""

import os
import csv
import json
import time
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# ★ CONFIG  —  Edit only this section
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"

# ★ Model backbone name (must match training scripts)
MODEL_NAME   = "swin_tiny"

# ★ Modalities used in the stacking ensemble (must match build_meta_dataset.py)
MODALITIES   = ["OCT0", "OCT1", "OCTA3"]

# ★ Feature type — must EXACTLY match build_meta_dataset.py settings
#   FEATURE_TYPE + USE_CALIB together determine which column is used:
#     ("logit", True)  → logit_calib   ← recommended (post-TS calibrated)
#     ("logit", False) → logit_uncal
#     ("prob",  True)  → prob_calib
#     ("prob",  False) → prob_uncal
FEATURE_TYPE = "logit"
USE_CALIB    = True

# ★ Path to trained meta-learner pipeline (.pkl)
#   Produced by train_meta_logistic_regression.py
META_MODEL_PATH = (
    "/data/Irene/SwinTransformer/Swin_Meta/outputs/meta_dataset/"
    "swin_tiny__logit__calibTrue/"
    "OCT0_LR3e-06_OCT1_LR2e-06_OCTA3_LR2e-06/"
    "meta_lr_model.pkl"
)

# ★ test_preds.csv locations (one per modality)
#   Produced by test_singlemode.py, one run per modality.
#   Format: <TEST_EVAL_ROOT>/<model>/<modality>/<run_tag>/Best_fold{N}/test_preds.csv
TEST_EVAL_ROOT = os.path.join(PROJECT_ROOT, "outputs", "test_evaluation")

# Run-tag and best fold for each modality (from training output)
MODALITY_RUN_TAG = {
    "OCT0":  "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9",
    "OCT1":  "BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8",
    "OCTA3": "BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6",
}

# update to match your actual best fold
MODALITY_BEST_FOLD = {
    "OCT0":  5,
    "OCT1":  1,   
    "OCTA3": 1,   
}

# ★ Decision threshold (0.5 is standard; tune via val set if needed)
THRESHOLD = 0.5

# Output directory for test evaluation results
META_TEST_OUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "meta_test_evaluation")

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(p: str, obj) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def log(logf, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if logf:
        logf.write(line + "\n")
        logf.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Feature column resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_feature_col(feature_type: str, use_calib: bool) -> str:
    """
    Resolve the CSV column name from (FEATURE_TYPE, USE_CALIB).
    Must exactly mirror build_meta_dataset.py → resolve_feature_col().

    Mapping:
      ("logit", True)  → "logit_calib"
      ("logit", False) → "logit_uncal"
      ("prob",  True)  → "prob_calib"
      ("prob",  False) → "prob_uncal"
    """
    mapping = {
        ("logit", True):  "logit_calib",
        ("logit", False): "logit_uncal",
        ("prob",  True):  "prob_calib",
        ("prob",  False): "prob_uncal",
    }
    key = (feature_type, use_calib)
    if key not in mapping:
        raise ValueError(
            f"Invalid (FEATURE_TYPE, USE_CALIB) = {key}.\n"
            f"Valid options: {list(mapping.keys())}"
        )
    return mapping[key]


def modality_feat_col(modality: str) -> str:
    """
    Return the renamed feature column name for a given modality.
    Must exactly mirror build_meta_dataset.py rename logic:
      df_oct0.rename(columns={feature_col: "oct0_feat"})
    """
    return f"{modality.lower()}_feat"   # OCT0→oct0_feat, OCT1→oct1_feat, OCTA3→octa3_feat


# ─────────────────────────────────────────────────────────────────────────────
# Load base model test predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_test_preds(
    modality: str,
    feature_col: str,
    test_eval_root: str,
    model_name: str,
    run_tag: str,
    best_fold: int,
    logf,
) -> pd.DataFrame:
    """
    Load test_preds.csv produced by test_singlemode.py for one modality.

    Path:
      TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/test_preds.csv

    Columns in test_preds.csv (from test_singlemode.py → save_test_preds_csv):
      exam_key | y_true | logit_uncal | prob_uncal |
      temperature | logit_calib | prob_calib

    Returns DataFrame with columns:
      exam_key | y_true | <modality_feat_col>
      e.g.: exam_key | y_true | oct0_feat
    """
    path = os.path.join(
        test_eval_root, model_name, modality, run_tag,
        f"Best_fold{best_fold}", "test_preds.csv"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"test_preds.csv not found for {modality}:\n  {path}\n"
            "Run test_singlemode.py for this modality first."
        )

    df = pd.read_csv(path)

    # Validate required columns
    required = {"exam_key", "y_true", feature_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{modality} test_preds.csv missing columns: {missing}\n"
            f"Available: {list(df.columns)}\n"
            f"File: {path}"
        )

    df["exam_key"] = df["exam_key"].astype(str)
    df["y_true"]   = df["y_true"].astype(int)

    # Check for duplicates
    dupes = df["exam_key"].duplicated().sum()
    if dupes > 0:
        raise RuntimeError(
            f"{modality} test_preds.csv has {dupes} duplicate exam_key rows.\n"
            "Each exam should appear exactly once."
        )

    # Rename feature col to match build_meta_dataset.py convention
    # e.g. "logit_calib" → "oct0_feat"
    feat_renamed = modality_feat_col(modality)
    df = df[["exam_key", "y_true", feature_col]].copy()
    df = df.rename(columns={feature_col: feat_renamed})

    n_active   = int((df["y_true"] == 1).sum())
    n_inactive = int((df["y_true"] == 0).sum())
    log(logf, f"  {modality}: {len(df)} samples  "
              f"active={n_active}  inactive={n_inactive}  "
              f"feat_col={feature_col}→{feat_renamed}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build meta-test feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_meta_test(
    dfs: Dict[str, pd.DataFrame],
    feat_cols_trained: List[str],
    logf,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Inner-join test DataFrames on exam_key.
    Verifies feature column names match those used during training.

    Inner join mirrors build_meta_dataset.py PAIRING_STRATEGY = "inner":
      Only exams present in ALL modalities are used.

    Returns (aligned_df, X_test, y_test).
    """
    mods   = list(dfs.keys())
    merged = dfs[mods[0]][["exam_key", "y_true"]].copy()
    for mod in mods:
        merged = merged.merge(
            dfs[mod].drop(columns=["y_true"]),
            on="exam_key", how="inner"
        )

    # Strict check: all y_true values must be 0 or 1
    assert set(merged["y_true"].unique()).issubset({0, 1}), \
        "y_true contains values outside {0, 1}"

    # Verify feature columns match training exactly
    actual_feat = [c for c in merged.columns if c not in ("exam_key", "y_true")]
    if actual_feat != feat_cols_trained:
        raise RuntimeError(
            "Test feature columns do NOT match training feature columns.\n"
            f"  Expected (from training): {feat_cols_trained}\n"
            f"  Actual   (from test CSV): {actual_feat}\n"
            "Check FEATURE_TYPE, USE_CALIB, and MODALITIES settings."
        )

    n  = len(merged)
    na = int((merged["y_true"] == 1).sum())
    ni = int((merged["y_true"] == 0).sum())
    log(logf, f"Meta-test (inner join): {n} exams  "
              f"active={na} ({na/n*100:.1f}%)  "
              f"inactive={ni} ({ni/n*100:.1f}%)  "
              f"imbalance={ni/na:.2f}:1")
    log(logf, f"Feature columns: {actual_feat}")

    X = merged[actual_feat].to_numpy(dtype=np.float32)
    y = merged["y_true"].to_numpy(dtype=np.int32)
    return merged, X, y


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Metrics (per IEEE TMI / JBHI / Nature Medicine reporting standards):
      Discriminative : AUROC, AUPRC
      Threshold-based: Balanced Accuracy, Sensitivity, Specificity,
                       PPV, NPV, F1-active, F1-macro, Accuracy
      Calibration    : Brier Score
      Confusion      : TP, FP, FN, TN

    References:
      AUROC/AUPRC: Saito & Rehmsmeier (2015). PLOS ONE.
      NPV/Clinical: STARD-AI (2020). Nature Medicine.
    """
    # Single-class guard: AUROC undefined for single-class test set
    if len(np.unique(y_true)) < 2:
        raise RuntimeError(
            "Test set contains only one class — AUROC, ROC, and several "
            "binary metrics are undefined. Verify the test split."
        )

    y_pred = (y_prob >= threshold).astype(int)

    # confusion_matrix with labels=[0,1] ensures 2×2 even if one class absent
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        # Dataset composition
        "n_total":           int(len(y_true)),
        "n_active":          int((y_true == 1).sum()),
        "n_inactive":        int((y_true == 0).sum()),
        "threshold":         float(threshold),
        # Discriminative (threshold-independent)
        "auroc":             float(roc_auc_score(y_true, y_prob)),
        "auprc":             float(average_precision_score(y_true, y_prob)),
        # Threshold-based
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity":       float(sens),
        "specificity":       float(spec),
        "ppv":               float(ppv),
        "npv":               float(npv),
        "f1_active":         float(f1_score(y_true, y_pred,
                                             pos_label=1, average="binary")),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro")),
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        # Calibration
        "brier_score":       float(brier_score_loss(y_true, y_prob)),
        # Confusion matrix
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    aligned_df: pd.DataFrame,
    y_prob: np.ndarray,
    metrics: dict,
    run_cfg: dict,
    out_dir: str,
    logf,
) -> None:
    ensure_dir(out_dir)

    threshold = metrics["threshold"]
    y_true    = aligned_df["y_true"].to_numpy()
    y_pred    = (y_prob >= threshold).astype(int)

    # ── 1. Predictions CSV ────────────────────────────────────────────────────
    pred_df = aligned_df[["exam_key", "y_true"]].copy()
    pred_df["prob_active"] = y_prob
    pred_df["y_pred"]      = y_pred
    pred_df["correct"]     = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df.to_csv(
        os.path.join(out_dir, "meta_lr_test_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    # ── 2. Results JSON ───────────────────────────────────────────────────────
    save_json(
        os.path.join(out_dir, "meta_lr_test_results.json"),
        {"run_config": run_cfg, "metrics": metrics}
    )

    # ── 3. ROC curve CSV ──────────────────────────────────────────────────────
    # roc_curve returns fpr(N), tpr(N), thresholds(N) — all same length
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    with open(os.path.join(out_dir, "meta_lr_roc_curve.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for fpr_v, tpr_v, thr_v in zip(fpr, tpr, roc_thr):
            w.writerow([f"{fpr_v:.6f}", f"{tpr_v:.6f}", f"{thr_v:.6f}"])

    # ── 4. PR curve CSV ───────────────────────────────────────────────────────
    # precision_recall_curve returns pre(N+1), rec(N+1), thresholds(N)
    # last point is sentinel (pre=1, rec=0); pad thresholds with nan
    pre, rec, pr_thr = precision_recall_curve(y_true, y_prob)
    pr_thr_padded    = list(pr_thr) + [float("nan")]
    with open(os.path.join(out_dir, "meta_lr_pr_curve.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["precision", "recall", "threshold"])
        for p_v, r_v, t_v in zip(pre, rec, pr_thr_padded):
            t_str = f"{t_v:.6f}" if not (isinstance(t_v, float) and np.isnan(t_v)) else ""
            w.writerow([f"{p_v:.6f}", f"{r_v:.6f}", t_str])

    # ── 5. Human-readable report ──────────────────────────────────────────────
    sep  = "=" * 68
    sep2 = "─" * 68
    with open(os.path.join(out_dir, "meta_lr_test_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("STACKING META-LEARNER — INDEPENDENT TEST SET EVALUATION\n")
        f.write(f"{sep}\n\n")

        f.write("[RUN CONFIG]\n")
        for k, v in run_cfg.items():
            f.write(f"  {k:<34}: {v}\n")

        n, na, ni = metrics["n_total"], metrics["n_active"], metrics["n_inactive"]
        f.write("\n[TEST SET COMPOSITION]\n")
        f.write(f"  Total    : {n}\n")
        f.write(f"  Active   : {na}  ({na/n*100:.1f}%)\n")
        f.write(f"  Inactive : {ni}  ({ni/n*100:.1f}%)\n")
        imb = ni/na if na > 0 else float("nan")
        f.write(f"  Imbalance: {imb:.2f}:1\n")

        f.write("\n[1. DISCRIMINATIVE PERFORMANCE  (threshold-independent)]\n")
        f.write(f"  AUROC    : {metrics['auroc']:.4f}\n")
        f.write(f"  AUPRC    : {metrics['auprc']:.4f}\n")

        f.write(f"\n[2. THRESHOLD-BASED METRICS  (threshold={threshold:.2f})]\n")
        f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"  Sensitivity      : {metrics['sensitivity']:.4f}  (Recall / active)\n")
        f.write(f"  Specificity      : {metrics['specificity']:.4f}  (Recall / inactive)\n")
        f.write(f"  PPV              : {metrics['ppv']:.4f}  (Precision / active)\n")
        f.write(f"  F1-active        : {metrics['f1_active']:.4f}\n")
        f.write(f"  F1-macro         : {metrics['f1_macro']:.4f}\n")
        f.write(f"  Accuracy         : {metrics['accuracy']:.4f}\n")

        f.write("\n[3. CLINICAL SAFETY METRICS]\n")
        f.write(f"  NPV (Precision / inactive): {metrics['npv']:.4f}"
                "  ← rule-out safety\n")

        f.write("\n[4. CALIBRATION]\n")
        f.write(f"  Brier Score: {metrics['brier_score']:.4f}  (lower = better)\n")

        f.write("\n[5. CONFUSION MATRIX]\n")
        f.write( "                    Pred active  Pred inactive\n")
        f.write(f"  True active     {metrics['TP']:>11}  {metrics['FN']:>14}\n")
        f.write(f"  True inactive   {metrics['FP']:>11}  {metrics['TN']:>14}\n")

        f.write(f"\n{sep2}\n")
        f.write("SUMMARY TABLE\n")
        f.write(f"{sep2}\n")
        f.write(f"  {'Metric':<25}  {'Value':>10}\n")
        f.write(f"  {'-'*25}  {'-'*10}\n")
        for lbl, key in [
            ("AUROC",             "auroc"),
            ("AUPRC",             "auprc"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Sensitivity",       "sensitivity"),
            ("Specificity",       "specificity"),
            ("PPV",               "ppv"),
            ("NPV",               "npv"),
            ("F1 (active)",       "f1_active"),
            ("F1 (macro)",        "f1_macro"),
            ("Accuracy",          "accuracy"),
            ("Brier Score",       "brier_score"),
        ]:
            f.write(f"  {lbl:<25}  {metrics[key]:>10.4f}\n")
        f.write(f"{sep2}\n")

        f.write("\n[SKLEARN CLASSIFICATION REPORT]\n")
        f.write(classification_report(
            y_true, y_pred, target_names=["inactive", "active"]
        ))

        f.write("\n[REFERENCES]\n")
        f.write("  AUROC/AUPRC : Saito & Rehmsmeier, PLOS ONE 2015\n")
        f.write("    https://doi.org/10.1371/journal.pone.0118432\n")
        f.write("  NPV Clinical: STARD-AI, Nature Medicine 2020\n")
        f.write("    https://doi.org/10.1038/s41591-020-0941-1\n")

        f.write("\n[OUTPUT FILES]\n")
        for fname in ["meta_lr_test_predictions.csv",
                      "meta_lr_test_results.json",
                      "meta_lr_test_report.txt",
                      "meta_lr_roc_curve.csv",
                      "meta_lr_pr_curve.csv"]:
            f.write(f"  {fname}\n")

    log(logf, f"All results saved → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Step 0: Resolve feature column & output path ──────────────────────────
    feature_col = resolve_feature_col(FEATURE_TYPE, USE_CALIB)

    # Derive output directory name from meta model path (mirrors training layout)
    meta_dir = str(Path(META_MODEL_PATH).parent)
    out_dir  = os.path.join(meta_dir, "test_evaluation")
    ensure_dir(out_dir)

    logf = open(
        os.path.join(out_dir, "meta_test_evaluation.log"),
        "a", buffering=1, encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"STACKING META-LEARNER TEST EVALUATION")
    print(f"{'='*60}")

    log(logf, "=" * 60)
    log(logf, "STACKING META-LEARNER TEST EVALUATION")
    log(logf, "=" * 60)
    log(logf, f"model_name   : {MODEL_NAME}")
    log(logf, f"modalities   : {MODALITIES}")
    log(logf, f"feature_type : {FEATURE_TYPE}  use_calib={USE_CALIB}")
    log(logf, f"feature_col  : {feature_col}")
    log(logf, f"threshold    : {THRESHOLD}")
    log(logf, f"meta_model   : {META_MODEL_PATH}")
    log(logf, f"out_dir      : {out_dir}")

    # ── Step 1: Load trained meta-learner pipeline ────────────────────────────
    log(logf, "─" * 60)
    log(logf, "Step 1: Load trained meta-learner pipeline")

    if not os.path.isfile(META_MODEL_PATH):
        raise FileNotFoundError(
            f"Meta model not found: {META_MODEL_PATH}\n"
            "Run train_meta_logistic_regression.py first."
        )
    pipe = joblib.load(META_MODEL_PATH)
    log(logf, f"Pipeline loaded: {type(pipe).__name__}")

    # Extract feature column names from trained model (ground truth reference)
    fitted_lr = pipe.named_steps.get("lr") or pipe.named_steps.get("logisticregression")
    if fitted_lr is None:
        # Fallback: try last step
        fitted_lr = list(pipe.named_steps.values())[-1]

    if hasattr(fitted_lr, "feature_names_in_"):
        feat_cols_trained: List[str] = fitted_lr.feature_names_in_.tolist()
        log(logf, f"Feature cols (from model): {feat_cols_trained}")
    else:
        # Fallback: construct from modalities using build_meta_dataset.py convention
        feat_cols_trained = [modality_feat_col(m) for m in MODALITIES]
        log(logf, f"Feature cols (inferred): {feat_cols_trained}")

    # Also read from meta_lr_config.json if available (most reliable)
    config_path = os.path.join(meta_dir, "meta_lr_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg_data = json.load(f)
        if "feature_cols" in cfg_data:
            feat_cols_trained = cfg_data["feature_cols"]
            log(logf, f"Feature cols (from config JSON): {feat_cols_trained}")

    if hasattr(fitted_lr, "C_"):
        log(logf, f"Best C (from training): {float(fitted_lr.C_[0]):.6f}")

    # ── Step 2: Load base model test predictions ──────────────────────────────
    log(logf, "─" * 60)
    log(logf, "Step 2: Load base model test predictions (test_preds.csv)")
    log(logf, f"  Source: TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{{N}}/test_preds.csv")
    log(logf, f"  feature_col = '{feature_col}'  →  renamed to <modality>_feat")

    dfs: Dict[str, pd.DataFrame] = {}
    for mod in MODALITIES:
        dfs[mod] = load_test_preds(
            modality=mod,
            feature_col=feature_col,
            test_eval_root=TEST_EVAL_ROOT,
            model_name=MODEL_NAME,
            run_tag=MODALITY_RUN_TAG[mod],
            best_fold=MODALITY_BEST_FOLD[mod],
            logf=logf,
        )

    # ── Step 3: Build meta-test feature matrix (inner join) ───────────────────
    log(logf, "─" * 60)
    log(logf, "Step 3: Build meta-test feature matrix (inner join on exam_key)")

    aligned_df, X_test, y_test = build_meta_test(dfs, feat_cols_trained, logf)

    # ── Step 4: Predict ───────────────────────────────────────────────────────
    log(logf, "─" * 60)
    log(logf, "Step 4: Meta-learner inference")

    y_prob = pipe.predict_proba(X_test)[:, 1]   # P(active)
    log(logf, f"Prediction shape : {y_prob.shape}")
    log(logf, f"Prob range       : [{y_prob.min():.4f}, {y_prob.max():.4f}]  "
              f"mean={y_prob.mean():.4f}")

    # ── Step 5: Compute metrics ───────────────────────────────────────────────
    log(logf, "─" * 60)
    log(logf, f"Step 5: Compute metrics (threshold={THRESHOLD})")

    metrics = compute_metrics(y_test, y_prob, THRESHOLD)

    log(logf, "RESULTS:")
    log(logf, f"  AUROC            : {metrics['auroc']:.4f}")
    log(logf, f"  AUPRC            : {metrics['auprc']:.4f}")
    log(logf, f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    log(logf, f"  Sensitivity      : {metrics['sensitivity']:.4f}")
    log(logf, f"  Specificity      : {metrics['specificity']:.4f}")
    log(logf, f"  PPV              : {metrics['ppv']:.4f}")
    log(logf, f"  NPV              : {metrics['npv']:.4f}")
    log(logf, f"  F1 (active)      : {metrics['f1_active']:.4f}")
    log(logf, f"  F1 (macro)       : {metrics['f1_macro']:.4f}")
    log(logf, f"  Accuracy         : {metrics['accuracy']:.4f}")
    log(logf, f"  Brier Score      : {metrics['brier_score']:.4f}")
    log(logf, f"  TP={metrics['TP']}  FP={metrics['FP']}  "
              f"FN={metrics['FN']}  TN={metrics['TN']}")

    # ── Step 6: Save all results ──────────────────────────────────────────────
    log(logf, "─" * 60)
    log(logf, "Step 6: Save results")

    run_cfg = {
        "model_name":        MODEL_NAME,
        "modalities":        MODALITIES,
        "feature_type":      FEATURE_TYPE,
        "use_calib":         USE_CALIB,
        "feature_col":       feature_col,
        "feature_cols_X":    feat_cols_trained,
        "meta_model_path":   META_MODEL_PATH,
        "threshold":         THRESHOLD,
        "run_tags":          MODALITY_RUN_TAG,
        "best_folds":        MODALITY_BEST_FOLD,
        "test_eval_root":    TEST_EVAL_ROOT,
        "out_dir":           out_dir,
        "note": (
            "Independent test set — never seen during base model training, "
            "validation, or meta-learner training."
        ),
    }
    save_results(aligned_df, y_prob, metrics, run_cfg, out_dir, logf)

    # ── Done ──────────────────────────────────────────────────────────────────
    log(logf, "=" * 60)
    log(logf, "EVALUATION COMPLETE")
    log(logf, f"  AUROC   : {metrics['auroc']:.4f}")
    log(logf, f"  AUPRC   : {metrics['auprc']:.4f}")
    log(logf, f"  NPV     : {metrics['npv']:.4f}")
    log(logf, f"  Results : {out_dir}")
    log(logf, "=" * 60)
    logf.close()


if __name__ == "__main__":
    main()