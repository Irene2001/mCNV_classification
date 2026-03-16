# c_test_meta_logistic_regression.py

"""
evaluate_meta_on_test.py — Stacking Meta-Learner Test Evaluation
mCNV binary classification (active / inactive)

Design
------
Strictly separated from training (train_meta_logistic_regression.py).
The independent test set has never been seen by any base model or meta-learner.

Input  : meta_lr_model.pkl  (trained Pipeline from training script)
         test set predictions from each best-fold base model
         (one CSV per modality, produced by run_test_inference.py or similar)
Process: align test features → Pipeline.predict_proba → threshold → metrics
Output : meta_lr_test_report.txt + meta_lr_test_results.json
         + meta_lr_test_predictions.csv + roc/pr curve data

Usage
-----
python evaluate_meta_on_test.py \\
    --model_name swin_tiny \\
    --modalities OCT0 OCT1 OCTA3 \\
    --meta_model_path outputs/meta/swin_tiny/<run_tag>/<meta_tag>/meta_lr_model.pkl \\
    --test_pred_dir  outputs/test_predictions/swin_tiny/<run_tag>
"""

import os
import csv
import json
import time
import joblib
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from training.model_factory import normalize_model_name
except ImportError:
    def normalize_model_name(n: str) -> str: return n.lower().strip()


# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = "/data/Irene/SwinTransformer/Swin_OOF"
META_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "meta")
TEST_PRED_ROOT   = os.path.join(PROJECT_ROOT, "outputs", "test_predictions")

DEFAULT_MODALITIES = ["OCT0", "OCT1", "OCTA3"]

# Feature mode must match the training script setting
FEATURE_MODE  = "prob_calib"

# Decision threshold — 0.5 is standard; can be tuned via val set if needed
THRESHOLD     = 0.5

# Expected columns in test prediction CSV per modality
#   exam_key, y_true, logit_uncal, prob_uncal, temperature, logit_calib, prob_calib
TEST_PRED_REQUIRED_COLS = {"exam_key", "y_true", "prob_calib", "logit_uncal",
                            "prob_uncal", "logit_calib"}


# ── Utilities ─────────────────────────────────────────────────────────────────
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


# ── Feature column mapping (must match training script) ──────────────────────
def _feat_cols(modality: str, mode: str) -> Tuple[List[str], List[str]]:
    m = {
        "prob_calib":  (["prob_calib"],
                        [f"{modality}_prob_calib"]),
        "logit_uncal": (["logit_uncal"],
                        [f"{modality}_logit_uncal"]),
        "prob_uncal":  (["prob_uncal"],
                        [f"{modality}_prob_uncal"]),
        "both":        (["prob_calib", "logit_uncal"],
                        [f"{modality}_prob_calib", f"{modality}_logit_uncal"]),
    }
    if mode not in m:
        raise ValueError(f"Unknown FEATURE_MODE: {mode}")
    return m[mode]


# ── Load test predictions ─────────────────────────────────────────────────────
def load_test_pred(test_pred_dir: str, modality: str,
                   mode: str) -> pd.DataFrame:
    """
    Load best-fold test predictions for one modality.

    Expected file: <test_pred_dir>/<modality>_test_predictions.csv
    Required cols: exam_key, y_true, prob_calib (and others per mode)
    """
    path = os.path.join(test_pred_dir, f"{modality}_test_predictions.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Test prediction file not found: {path}\n"
            f"Run best-fold inference for {modality} first."
        )
    df = pd.read_csv(path)
    raw, renamed = _feat_cols(modality, mode)
    required = {"exam_key", "y_true"} | set(raw)
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{modality} test CSV missing columns: {missing}")

    out = df[["exam_key", "y_true"] + raw].copy()
    out = out.rename(columns=dict(zip(raw, renamed)))
    out["exam_key"] = out["exam_key"].astype(str)
    out["y_true"]   = out["y_true"].astype(int)

    dupes = out["exam_key"].duplicated().sum()
    if dupes:
        raise RuntimeError(f"{modality} test CSV has {dupes} duplicate exam_key rows.")
    return out


# ── Build meta-test feature matrix ───────────────────────────────────────────
def build_meta_test(
    dfs: Dict[str, pd.DataFrame],
    mode: str,
    feat_cols_expected: List[str],
    logf,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Align test DataFrames on exam_key (inner join).
    Verifies feature columns match those used in training.

    Returns (aligned_df, X_test, y_test).
    """
    mods   = list(dfs.keys())
    merged = dfs[mods[0]][["exam_key", "y_true"]].copy()
    for mod in mods:
        merged = merged.merge(dfs[mod].drop(columns=["y_true"]),
                              on="exam_key", how="inner")

    assert set(merged["y_true"].unique()).issubset({0, 1})

    # Verify columns match training
    actual_feat = [c for c in merged.columns if c not in ("exam_key", "y_true")]
    if actual_feat != feat_cols_expected:
        raise RuntimeError(
            f"Test feature columns do not match training.\n"
            f"  expected : {feat_cols_expected}\n"
            f"  actual   : {actual_feat}"
        )

    n, na, ni = len(merged), (merged["y_true"]==1).sum(), (merged["y_true"]==0).sum()
    log(logf, f"Meta-test : {n} samples  active={na} ({na/n*100:.1f}%)  "
              f"inactive={ni} ({ni/n*100:.1f}%)  ratio={ni/na:.2f}:1")
    log(logf, f"Features  : {actual_feat}")

    X = merged[actual_feat].to_numpy(dtype=np.float32)
    y = merged["y_true"].to_numpy(dtype=np.int32)
    return merged, X, y


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute full classification metrics at given threshold.

    Metrics
    -------
    AUC-ROC   : primary metric, threshold-independent
    AUPRC     : average precision, handles imbalance well
    F1-macro  : equal weight to active and inactive
    F1-active : sensitivity-focused binary F1 for minority class
    Sensitivity (Recall) for active  : TP / (TP + FN)  — clinical priority
    Specificity for inactive         : TN / (TN + FP)
    Accuracy  : included but not primary due to class imbalance
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold":      threshold,
        "n_total":        int(len(y_true)),
        "n_active":       int((y_true == 1).sum()),
        "n_inactive":     int((y_true == 0).sum()),
        # Primary metrics
        "auc_roc":        float(roc_auc_score(y_true, y_prob)),
        "auprc":          float(average_precision_score(y_true, y_prob)),
        # Classification metrics
        "f1_macro":       float(f1_score(y_true, y_pred, average="macro")),
        "f1_active":      float(f1_score(y_true, y_pred, pos_label=1,
                                          average="binary")),
        "f1_inactive":    float(f1_score(y_true, y_pred, pos_label=0,
                                          average="binary")),
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        # Clinical metrics
        "sensitivity":    float(sensitivity),   # recall for active
        "specificity":    float(specificity),   # recall for inactive
        "ppv":            float(ppv),           # precision for active
        "npv":            float(npv),           # precision for inactive
        # Confusion matrix
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[dict, dict]:
    """Return ROC and PR curve data as dicts (for CSV saving)."""
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    pre, rec, pr_thr  = precision_recall_curve(y_true, y_prob)
    roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
           "threshold": roc_thr.tolist()}
    prc = {"precision": pre.tolist(), "recall": rec.tolist(),
           "threshold": pr_thr.tolist()}
    return roc, prc


# ── Save Results ──────────────────────────────────────────────────────────────
def save_results(
    aligned_df: pd.DataFrame,
    y_prob: np.ndarray,
    metrics: dict,
    roc_data: dict,
    prc_data: dict,
    run_cfg: dict,
    out_dir: str,
    logf,
) -> None:
    ensure_dir(out_dir)

    # 1. Predictions CSV  (exam_key, y_true, y_prob, y_pred)
    pred_df = aligned_df[["exam_key", "y_true"]].copy()
    pred_df["prob_active"] = y_prob
    pred_df["y_pred"]      = (y_prob >= metrics["threshold"]).astype(int)
    pred_df["correct"]     = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df.to_csv(os.path.join(out_dir, "meta_lr_test_predictions.csv"),
                   index=False, encoding="utf-8-sig")

    # 2. Results JSON
    save_json(os.path.join(out_dir, "meta_lr_test_results.json"),
              {"run_config": run_cfg, "metrics": metrics})

    # 3. ROC curve CSV
    roc_rows = list(zip(roc_data["fpr"], roc_data["tpr"],
                        roc_data["threshold"] + [float("nan")]))
    with open(os.path.join(out_dir, "meta_lr_roc_curve.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for row in roc_rows:
            w.writerow([f"{v:.6f}" if not np.isnan(v) else "" for v in row])

    # 4. PR curve CSV
    pr_rows = list(zip(prc_data["precision"], prc_data["recall"],
                       prc_data["threshold"] + [float("nan")]))
    with open(os.path.join(out_dir, "meta_lr_pr_curve.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["precision", "recall", "threshold"])
        for row in pr_rows:
            w.writerow([f"{v:.6f}" if not np.isnan(v) else "" for v in row])

    # 5. Human-readable report
    y_true = aligned_df["y_true"].to_numpy()
    y_pred = (y_prob >= metrics["threshold"]).astype(int)
    with open(os.path.join(out_dir, "meta_lr_test_report.txt"),
              "w", encoding="utf-8") as f:
        sep = "=" * 65
        f.write(f"{sep}\nSTACKING META-LEARNER — TEST SET EVALUATION REPORT\n{sep}\n\n")

        f.write("[RUN CONFIG]\n")
        for k, v in run_cfg.items():
            f.write(f"  {k:<30}: {v}\n")

        f.write("\n[TEST SET COMPOSITION]\n")
        f.write(f"  total    : {metrics['n_total']}\n")
        f.write(f"  active   : {metrics['n_active']}  "
                f"({metrics['n_active']/metrics['n_total']*100:.1f}%)\n")
        f.write(f"  inactive : {metrics['n_inactive']}  "
                f"({metrics['n_inactive']/metrics['n_total']*100:.1f}%)\n")

        f.write("\n[PRIMARY METRICS]\n")
        f.write(f"  AUC-ROC  : {metrics['auc_roc']:.4f}\n")
        f.write(f"  AUPRC    : {metrics['auprc']:.4f}\n")

        f.write("\n[CLASSIFICATION METRICS  (threshold={:.2f})]\n".format(
            metrics["threshold"]))
        f.write(f"  F1-macro    : {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1-active   : {metrics['f1_active']:.4f}\n")
        f.write(f"  F1-inactive : {metrics['f1_inactive']:.4f}\n")
        f.write(f"  Accuracy    : {metrics['accuracy']:.4f}\n")

        f.write("\n[CLINICAL METRICS]\n")
        f.write(f"  Sensitivity (Recall / active) : {metrics['sensitivity']:.4f}\n")
        f.write(f"  Specificity (Recall / inactive): {metrics['specificity']:.4f}\n")
        f.write(f"  PPV (Precision / active)       : {metrics['ppv']:.4f}\n")
        f.write(f"  NPV (Precision / inactive)     : {metrics['npv']:.4f}\n")

        f.write("\n[CONFUSION MATRIX]\n")
        f.write(f"                  Pred active  Pred inactive\n")
        f.write(f"  True active     {metrics['TP']:>10}  {metrics['FN']:>13}\n")
        f.write(f"  True inactive   {metrics['FP']:>10}  {metrics['TN']:>13}\n")

        f.write("\n[SKLEARN CLASSIFICATION REPORT]\n")
        f.write(classification_report(y_true, y_pred,
                                      target_names=["inactive", "active"]))

    log(logf, f"Results saved → {out_dir}")
    for fname in ["meta_lr_test_predictions.csv", "meta_lr_test_results.json",
                  "meta_lr_test_report.txt", "meta_lr_roc_curve.csv",
                  "meta_lr_pr_curve.csv"]:
        log(logf, f"  {fname}")


# ── Auto-detect meta model path ───────────────────────────────────────────────
def _detect_meta_model(meta_root: str, model_name: str,
                        run_tag: str) -> str:
    """Walk meta/<model>/<run_tag>/**/meta_lr_model.pkl."""
    base = os.path.join(meta_root, model_name, run_tag)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Meta output dir not found: {base}")
    for root, _, files in os.walk(base):
        if "meta_lr_model.pkl" in files:
            return os.path.join(root, "meta_lr_model.pkl")
    raise FileNotFoundError(f"meta_lr_model.pkl not found under {base}")


# ── Argument Parser ───────────────────────────────────────────────────────────
def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate Stacking LR meta-learner on independent test set."
    )
    p.add_argument("--model_name",      default="swin_tiny",
                   choices=["swin_tiny", "vgg16", "efficientnet_b0"])
    p.add_argument("--modalities",      nargs="+", default=DEFAULT_MODALITIES,
                   choices=["OCT0", "OCT1", "OCTA3"])
    p.add_argument("--run_tag",         default=None,
                   help="Run tag shared across modalities.")
    p.add_argument("--meta_model_path", default=None,
                   help="Path to meta_lr_model.pkl. Auto-detected if None.")
    p.add_argument("--test_pred_dir",   default=None,
                   help=(
                       "Dir containing <MODALITY>_test_predictions.csv files. "
                       "Default: outputs/test_predictions/<model>/<run_tag>"
                   ))
    p.add_argument("--feature_mode",    default=FEATURE_MODE,
                   choices=["prob_calib", "logit_uncal", "prob_uncal", "both"])
    p.add_argument("--threshold",       default=THRESHOLD, type=float,
                   help="Decision threshold for binary prediction (default 0.5).")
    p.add_argument("--meta_out_root",   default=META_OUTPUT_ROOT)
    return p


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args       = build_args().parse_args()
    model_name = normalize_model_name(args.model_name)
    modalities = args.modalities
    feat_mode  = args.feature_mode

    # ── Locate meta model
    if args.meta_model_path and os.path.isfile(args.meta_model_path):
        meta_model_path = args.meta_model_path
    else:
        if not args.run_tag:
            raise ValueError("Provide --run_tag or --meta_model_path.")
        meta_model_path = _detect_meta_model(
            args.meta_out_root, model_name, args.run_tag)

    meta_dir = os.path.dirname(meta_model_path)

    # ── Output dir for test results (alongside training artefacts)
    out_dir = os.path.join(meta_dir, "test_evaluation")
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "meta_test_evaluation.log"),
                "a", buffering=1, encoding="utf-8")

    log(logf, "=" * 55)
    log(logf, "STACKING META-LEARNER TEST EVALUATION")
    log(logf, "=" * 55)
    log(logf, f"model      : {model_name}  modalities: {modalities}")
    log(logf, f"feat_mode  : {feat_mode}   threshold: {args.threshold}")
    log(logf, f"model_path : {meta_model_path}")

    # ── Load trained pipeline
    log(logf, "─" * 55)
    log(logf, "Step 1: Load trained meta-learner pipeline")
    pipe = joblib.load(meta_model_path)
    fitted_lr = pipe.named_steps["lr"]
    feat_cols_trained: List[str] = fitted_lr.feature_names_in_.tolist() \
        if hasattr(fitted_lr, "feature_names_in_") \
        else [f"{m}_{feat_mode}" for m in modalities]

    # Read expected feature columns from config JSON if available
    config_path = os.path.join(meta_dir, "meta_lr_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg_data = json.load(f)
        feat_cols_trained = cfg_data.get("feature_cols", feat_cols_trained)
        log(logf, f"Feature cols from training config: {feat_cols_trained}")
    else:
        log(logf, f"No config JSON found; inferring feature cols: {feat_cols_trained}")

    log(logf, f"Best C (from training): {float(fitted_lr.C_[0]):.4f}")

    # ── Load test predictions per modality
    log(logf, "─" * 55)
    log(logf, "Step 2: Load test predictions per modality")

    test_pred_dir = args.test_pred_dir or os.path.join(
        TEST_PRED_ROOT, model_name,
        args.run_tag if args.run_tag else ""
    )
    dfs: Dict[str, pd.DataFrame] = {}
    for mod in modalities:
        dfs[mod] = load_test_pred(test_pred_dir, mod, feat_mode)
        log(logf, f"  {mod}: {len(dfs[mod])} rows  "
                  f"active={(dfs[mod]['y_true']==1).sum()}  "
                  f"inactive={(dfs[mod]['y_true']==0).sum()}")

    # ── Build meta-test feature matrix
    log(logf, "─" * 55)
    log(logf, "Step 3: Build meta-test feature matrix")
    aligned_df, X_test, y_test = build_meta_test(
        dfs, feat_mode, feat_cols_trained, logf)

    # ── Predict
    log(logf, "─" * 55)
    log(logf, "Step 4: Predict on test set")
    y_prob = pipe.predict_proba(X_test)[:, 1]   # P(active)
    log(logf, f"  Prediction shape: {y_prob.shape}  "
              f"range=[{y_prob.min():.4f}, {y_prob.max():.4f}]  "
              f"mean={y_prob.mean():.4f}")

    # ── Compute metrics
    log(logf, "─" * 55)
    log(logf, f"Step 5: Compute metrics (threshold={args.threshold})")
    metrics = compute_metrics(y_test, y_prob, args.threshold)
    roc_data, prc_data = compute_roc_pr_curves(y_test, y_prob)

    log(logf, "TEST SET RESULTS:")
    log(logf, f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    log(logf, f"  AUPRC       : {metrics['auprc']:.4f}")
    log(logf, f"  F1-macro    : {metrics['f1_macro']:.4f}")
    log(logf, f"  F1-active   : {metrics['f1_active']:.4f}")
    log(logf, f"  Sensitivity : {metrics['sensitivity']:.4f}  (recall/active)")
    log(logf, f"  Specificity : {metrics['specificity']:.4f}  (recall/inactive)")
    log(logf, f"  PPV         : {metrics['ppv']:.4f}")
    log(logf, f"  NPV         : {metrics['npv']:.4f}")
    log(logf, f"  Accuracy    : {metrics['accuracy']:.4f}")
    log(logf, f"  TP={metrics['TP']}  FP={metrics['FP']}  "
              f"FN={metrics['FN']}  TN={metrics['TN']}")

    # ── Save all results
    log(logf, "─" * 55)
    log(logf, "Step 6: Save results")
    run_cfg = {
        "model_name":       model_name,
        "modalities":       modalities,
        "meta_model_path":  meta_model_path,
        "test_pred_dir":    test_pred_dir,
        "feature_mode":     feat_mode,
        "feature_cols":     feat_cols_trained,
        "threshold":        args.threshold,
        "best_C":           float(fitted_lr.C_[0]),
        "out_dir":          out_dir,
        "note": (
            "Independent test set — never seen during base model training, "
            "validation, or meta-learner training."
        ),
    }
    save_results(aligned_df, y_prob, metrics, roc_data, prc_data,
                 run_cfg, out_dir, logf)

    log(logf, "=" * 55)
    log(logf, "EVALUATION COMPLETE")
    log(logf, f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    log(logf, f"  AUPRC    : {metrics['auprc']:.4f}")
    log(logf, f"  F1-macro : {metrics['f1_macro']:.4f}")
    log(logf, f"  Results  : {out_dir}")
    log(logf, "=" * 55)

    logf.close()


if __name__ == "__main__":
    main()