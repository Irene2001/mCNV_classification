# export_origin_plots.py 

"""
Base Model — Training + Testing
==========================================================
Post-processing script for OriginLab journal-grade CSV export.
Reads ALL existing outputs from train_singlemode_oof.py and
test_singlemode.py.  NO retraining required.

Source paths (read-only, never modified):
  PROJECT_ROOT/outputs/training/<model>/<modality>/<run_tag>/
    Kfold/fold{k}/metrics.csv          per-epoch train/val loss, AUC, acc
    Kfold/fold{k}/fold_summary.json    best_val_auc, temperature
  PROJECT_ROOT/outputs/oof_predictions/<model>/<modality>/<run_tag>/
    all_folds_oof.csv                  5-fold OOF logit_uncal/calib, y_true
  PROJECT_ROOT/outputs/test_evaluation/<model>/<modality>/<run_tag>/Best_fold{N}/
    test_preds.csv                     test logit_uncal/calib, y_true

Output root (strictly isolated from training outputs):
  PICS_ROOT = /data/Irene/SwinTransformer/Swin_Meta/pics_outputs/
  └── <model>/<modality>/<run_tag>/
      ├── training/
      │   ├── learning_curves.csv          Fig T-1
      │   ├── val_auc_curves.csv           Fig T-2
      │   └── oof_calibration_bins.csv     Fig T-3
      └── testing/
          ├── test_roc.csv                 Fig E-1
          ├── test_pr.csv                  Fig E-2
          ├── test_calibration_bins.csv    Fig E-3
          └── test_confusion_matrix.csv    Fig E-4

References:
  [1] Guo et al. ICML 2017. On Calibration of Modern Neural Networks.
      https://arxiv.org/abs/1706.04599
  [2] Fawcett (2006). ROC analysis. Pattern Recognition Letters.
      https://doi.org/10.1016/j.patrec.2005.10.010
  [3] Saito & Rehmsmeier (2015). PR plot for imbalanced datasets. PLOS ONE.
      https://doi.org/10.1371/journal.pone.0118432
  [4] Radiology AI (RSNA 2024). Guide to CV for AI in Medical Imaging.
      https://pubs.rsna.org/doi/10.1148/ryai.220232
"""

import json
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# ★ CONFIG  --  Edit only this section
# ==============================================================================

PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"
PICS_ROOT    = "/data/Irene/SwinTransformer/Swin_Meta/pics_outputs"

MODEL_NAME = "swin_tiny"    # swin_tiny | vgg16 | efficientnet_b0
MODALITY   = "OCTA3"         # OCT0 | OCT1 | OCTA3
RUN_TAG    = (
    "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6"
)
BEST_FOLD  = 2              # best fold from training summary

# Switch comments for other modalities:
# OCT0:  "BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9"    BEST_FOLD=2
# OCT1:  "BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8"  BEST_FOLD=1
# OCTA3: "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6"    BEST_FOLD=2

N_FOLDS    = 5
N_ROC_PTS  = 100   # fixed grid points for ROC/PR interpolation
N_CAL_BINS = 10    # reliability diagram bins (Guo et al. 2017, M=10)

# Column name candidates in metrics.csv (tries each in order)
EPOCH_COLS      = ["epoch", "Epoch"]
TRAIN_LOSS_COLS = ["train_focal_loss", "train_loss", "loss"]
VAL_LOSS_COLS   = ["val_bce", "val_loss"]
TRAIN_ACC_COLS  = ["train_acc", "train_accuracy"]
VAL_ACC_COLS    = ["val_acc", "val_accuracy"]
VAL_AUC_COLS    = ["val_auc", "val_auroc"]

# ==============================================================================


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))).astype(np.float64)


def pics_path(*parts: str) -> str:
    full = os.path.join(PICS_ROOT, MODEL_NAME, MODALITY, RUN_TAG, *parts)
    ensure_dir(os.path.dirname(full))
    return full


def training_fold_dir(k: int) -> str:
    return os.path.join(
        PROJECT_ROOT, "outputs", "training",
        MODEL_NAME, MODALITY, RUN_TAG, "Kfold", f"fold{k}"
    )


def oof_csv_path() -> str:
    return os.path.join(
        PROJECT_ROOT, "outputs", "oof_predictions",
        MODEL_NAME, MODALITY, RUN_TAG, "all_folds_oof.csv"
    )


def test_preds_path() -> str:
    return os.path.join(
        PROJECT_ROOT, "outputs", "test_evaluation",
        MODEL_NAME, MODALITY, RUN_TAG,
        f"Best_fold{BEST_FOLD}", "test_preds.csv"
    )


def banner(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[pd.DataFrame, float]:
    """
    Equal-width reliability diagram bins.
    ECE = Σ_b (|B_b|/N) |acc_b - conf_b|  (Guo et al. 2017, Eq.3)
    Returns (DataFrame, ECE_float).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N     = len(y_true)
    ece   = 0.0
    rows  = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 \
                 else (y_prob >= lo) & (y_prob < hi)
        n   = int(mask.sum())
        acc = float(y_true[mask].mean()) if n > 0 else np.nan
        cf  = float(y_prob[mask].mean()) if n > 0 else np.nan
        mid = (lo + hi) / 2.0
        if n > 0:
            ece += (n / N) * abs(acc - cf)
        rows.append({
            "bin_lower": round(lo, 2),
            "bin_upper": round(hi, 2),
            "bin_mid":   round(mid, 3),
            "n_samples": n,
            "acc":       round(acc, 6) if not np.isnan(acc) else "",
            "conf":      round(cf,  6) if not np.isnan(cf)  else "",
            "diagonal":  round(mid, 3),
            "gap":       round(acc - cf, 6)
                         if (not np.isnan(acc) and not np.isnan(cf)) else "",
        })
    return pd.DataFrame(rows), round(ece, 6)


# ==============================================================================
#  TRAINING FIGURES  (from train_singlemode_oof.py)
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# Fig T-1  Learning Curves  (Mean ± SD, 5 folds)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "5-Fold Cross-Validation Learning Curves"
# X-axis : Epoch
# Y-axis : (Subplot A) BCE Loss — Train & Val
#           (Subplot B) Accuracy — Train & Val
# Format : Mean ± SD shading across 5 folds per epoch
# Ref    : Radiology AI 2024 — https://pubs.rsna.org/doi/10.1148/ryai.220232
#
# CSV columns:
#   epoch | train_loss_mean | train_loss_sd |
#   val_loss_mean | val_loss_sd |
#   train_acc_mean | train_acc_sd |
#   val_acc_mean | val_acc_sd
#
# OriginLab:
#   A=epoch(X)  B,C=train_loss_mean/sd (Y+yError, Line+Shadow)
#   D,E=val_loss_mean/sd               (Y+yError, Line+Shadow)
#   F,G=train_acc_mean/sd              (Y+yError)
#   H,I=val_acc_mean/sd                (Y+yError)
#
# Source : outputs/training/.../Kfold/fold{1-5}/metrics.csv
# Output : pics_outputs/.../training/learning_curves.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_T1_learning_curves() -> None:
    banner("Fig T-1: Learning Curves (5-Fold Mean ± SD)")

    fold_dfs = []
    for k in range(1, N_FOLDS + 1):
        p = os.path.join(training_fold_dir(k), "metrics.csv")
        if not os.path.isfile(p):
            warn(f"metrics.csv not found for fold {k}: {p}")
            continue
        df = pd.read_csv(p)

        ec  = find_col(df, *EPOCH_COLS)
        tlc = find_col(df, *TRAIN_LOSS_COLS)
        vlc = find_col(df, *VAL_LOSS_COLS)
        tac = find_col(df, *TRAIN_ACC_COLS)
        vac = find_col(df, *VAL_ACC_COLS)

        sub = {"epoch":      df[ec].values if ec else np.arange(1, len(df)+1),
               "train_loss": df[tlc].values if tlc else np.full(len(df), np.nan),
               "val_loss":   df[vlc].values if vlc else np.full(len(df), np.nan),
               "train_acc":  df[tac].values if tac else np.full(len(df), np.nan),
               "val_acc":    df[vac].values if vac else np.full(len(df), np.nan),
               "fold": k}
        fold_dfs.append(pd.DataFrame(sub))

    if not fold_dfs:
        warn("No metrics.csv found. Skipping Fig T-1.")
        return

    all_df = pd.concat(fold_dfs, ignore_index=True)
    grp    = all_df.groupby("epoch")[["train_loss", "val_loss",
                                       "train_acc", "val_acc"]]
    mean_  = grp.mean().add_suffix("_mean")
    sd_    = grp.std(ddof=1).add_suffix("_sd")
    out    = pd.concat([mean_, sd_], axis=1).reset_index()
    # Interleave columns: mean then sd for each metric
    cols   = ["epoch"]
    for m in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        cols += [f"{m}_mean", f"{m}_sd"]
    out = out[[c for c in cols if c in out.columns]].round(6)

    out_path = pics_path("training", "learning_curves.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved {len(out)} epochs × {len(out.columns)} cols")
    ok(f"→ {out_path}")
    print("  OriginLab: X=epoch")
    print("    Subplot A: train_loss_mean/sd vs val_loss_mean/sd (Line+Shadow)")
    print("    Subplot B: train_acc_mean/sd  vs val_acc_mean/sd  (Line+Shadow)")


# ──────────────────────────────────────────────────────────────────────────────
# Fig T-2  Per-Fold Val AUROC Summary  (bar + mean±SD)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "Per-Fold Validation AUROC (5-Fold CV)"
# X-axis : Fold (1, 2, 3, 4, 5, Mean)
# Y-axis : AUROC
# Format : Bar chart; last rows = Mean ± SD
#
# CSV columns:
#   fold | val_auc | val_loss | best_epoch | temperature
#   (last two rows: fold=Mean / fold=SD)
#
# OriginLab:
#   A=fold(X-text)  B=val_auc(Y-bar)
#   Error bar: draw manually from SD row value
#
# Source : outputs/training/.../Kfold/fold{1-5}/fold_summary.json
#          OR metrics.csv (row with best val_auc)
# Output : pics_outputs/.../training/val_auc_curves.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_T2_val_auc_summary() -> None:
    banner("Fig T-2: Per-Fold Val AUROC Summary (Bar + Mean±SD)")

    rows = []
    for k in range(1, N_FOLDS + 1):
        json_p = os.path.join(training_fold_dir(k), "fold_summary.json")
        csv_p  = os.path.join(training_fold_dir(k), "metrics.csv")

        if os.path.isfile(json_p):
            with open(json_p, encoding="utf-8") as f:
                s = json.load(f)
            rows.append({
                "fold":        k,
                "val_auc":     round(float(s.get("best_val_auc", np.nan)), 6),
                "val_loss":    round(float(s.get("best_val_bce",
                               s.get("best_val_key", np.nan))), 6),
                "best_epoch":  int(s.get("best_epoch", -1)),
                "temperature": round(float(s.get("temperature", 1.0)), 6),
            })
        elif os.path.isfile(csv_p):
            df  = pd.read_csv(csv_p)
            auc = find_col(df, *VAL_AUC_COLS)
            lss = find_col(df, *VAL_LOSS_COLS)
            if auc:
                br = df.loc[df[auc].idxmax()]
                rows.append({
                    "fold":        k,
                    "val_auc":     round(float(br[auc]), 6),
                    "val_loss":    round(float(br[lss]), 6) if lss else np.nan,
                    "best_epoch":  int(br.get("epoch", -1)),
                    "temperature": np.nan,
                })
        else:
            warn(f"No fold_summary.json or metrics.csv for fold {k}.")

    if not rows:
        warn("No per-fold data found. Skipping Fig T-2.")
        return

    main_df  = pd.DataFrame(rows)
    num_cols = [c for c in ["val_auc", "val_loss", "temperature"]
                if c in main_df.columns]
    mean_row = {"fold": "Mean"}
    sd_row   = {"fold": "SD"}
    for c in num_cols:
        vals = pd.to_numeric(main_df[c], errors="coerce").dropna()
        mean_row[c] = round(float(vals.mean()), 6)
        sd_row[c]   = round(float(vals.std(ddof=1)), 6)

    out = pd.concat([main_df, pd.DataFrame([mean_row, sd_row])],
                    ignore_index=True)
    out_path = pics_path("training", "val_auc_curves.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved {len(rows)} folds + Mean/SD rows")
    ok(f"→ {out_path}")
    print(f"  Mean Val AUROC = {mean_row.get('val_auc','N/A')} "
          f"± {sd_row.get('val_auc','N/A')}")
    print("  OriginLab: X=fold(text) | Y=val_auc (bar chart)")
    print("             Error bar value from SD row")


# ──────────────────────────────────────────────────────────────────────────────
# Fig T-3  OOF Reliability Diagram  (Before/After TS, all 5 folds)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "OOF Reliability Diagram — Before and After Temperature Scaling"
# X-axis : Mean Predicted Confidence (Bin Midpoint)
# Y-axis : Fraction of Positives (Observed Accuracy)
# Curves : uncalibrated (dashed) | calibrated (solid) | perfect (diagonal)
# Bar    : n_samples per bin (secondary Y-axis)
# Metric : ECE_uncal, ECE_calib annotated in figure
# Ref    : Guo et al. ICML 2017 — https://arxiv.org/abs/1706.04599
#
# CSV columns (N_CAL_BINS rows):
#   bin_lower | bin_upper | bin_mid | n_samples |
#   conf_uncal | acc_uncal |
#   conf_calib | acc_calib |
#   diagonal   | gap_calib
#
# OriginLab:
#   X=bin_mid
#   Y=acc_uncal (dashed orange) | Y=acc_calib (solid blue)
#   Y=diagonal  (dashed grey, perfect calibration)
#   Bar=n_samples (secondary Y-axis, grey bars)
#
# Source : outputs/oof_predictions/.../all_folds_oof.csv
# Output : pics_outputs/.../training/oof_calibration_bins.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_T3_oof_calibration() -> None:
    banner("Fig T-3: OOF Reliability Diagram (Before/After TS)")

    p = oof_csv_path()
    if not os.path.isfile(p):
        warn(f"all_folds_oof.csv not found: {p}. Skipping Fig T-3.")
        return

    df = pd.read_csv(p)
    ok(f"OOF loaded: {len(df)} samples")

    y_col    = find_col(df, "y_true", "label")
    prob_cal = find_col(df, "prob_calib")
    logit_u  = find_col(df, "logit_uncal")
    prob_u   = find_col(df, "prob_uncal")

    if not y_col or not prob_cal:
        warn(f"Required columns missing. Available: {list(df.columns)}")
        return

    y_true  = df[y_col].astype(int).values
    y_calib = df[prob_cal].astype(float).values
    y_uncal = (sigmoid(df[logit_u].astype(float).values) if logit_u
               else df[prob_u].astype(float).values if prob_u else None)

    bins_cal, ece_cal = calibration_bins(y_true, y_calib, N_CAL_BINS)
    out = bins_cal.rename(columns={"acc": "acc_calib", "conf": "conf_calib",
                                   "gap": "gap_calib"})

    if y_uncal is not None:
        bins_u, ece_uncal = calibration_bins(y_true, y_uncal, N_CAL_BINS)
        out["acc_uncal"]  = bins_u["acc"].values
        out["conf_uncal"] = bins_u["conf"].values
    else:
        ece_uncal = np.nan
        out["acc_uncal"]  = np.nan
        out["conf_uncal"] = np.nan

    out_path = pics_path("training", "oof_calibration_bins.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved {N_CAL_BINS} bins")
    ok(f"→ {out_path}")
    print(f"  ECE (calibrated)   = {ece_cal:.4f}")
    if not np.isnan(ece_uncal):
        print(f"  ECE (uncalibrated) = {ece_uncal:.4f}")
    print("  OriginLab: X=bin_mid")
    print("    Y=acc_calib (solid blue) | Y=acc_uncal (dashed orange)")
    print("    Y=diagonal (dashed grey) | Bar=n_samples (2nd axis)")


# ==============================================================================
#  TESTING FIGURES  (from test_singlemode.py)
# ==============================================================================

def _load_test_preds() -> Optional[pd.DataFrame]:
    p = test_preds_path()
    if not os.path.isfile(p):
        warn(f"test_preds.csv not found: {p}")
        return None
    df = pd.read_csv(p)
    ok(f"Test preds loaded: {len(df)} samples")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fig E-1  Test ROC Curve  (uncal vs calib)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "ROC Curve — Independent Test Set"
# X-axis : False Positive Rate (1 - Specificity)
# Y-axis : True Positive Rate (Sensitivity)
# Curves : Calibrated (solid blue, AUC in legend)
#           Uncalibrated (dashed orange, AUC in legend)
#           Random diagonal (dashed grey)
# Ref    : Fawcett 2006 — https://doi.org/10.1016/j.patrec.2005.10.010
#
# CSV columns (N_ROC_PTS rows):
#   fpr_calib | tpr_calib | thr_calib |
#   fpr_uncal | tpr_uncal | thr_uncal |
#   diagonal_x | diagonal_y
#
# OriginLab:
#   X=fpr_calib  Y=tpr_calib (solid, label: AUC=X.XXXX)
#   X=fpr_uncal  Y=tpr_uncal (dashed, label: AUC=X.XXXX)
#   X=diagonal_x Y=diagonal_y (dashed grey, label: Random)
#
# Source : outputs/test_evaluation/.../Best_fold{N}/test_preds.csv
# Output : pics_outputs/.../testing/test_roc.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_E1_roc() -> None:
    banner("Fig E-1: Test ROC Curve (Calibrated vs Uncalibrated)")

    df = _load_test_preds()
    if df is None:
        return

    y_col    = find_col(df, "y_true", "label")
    prob_cal = find_col(df, "prob_calib")
    logit_u  = find_col(df, "logit_uncal")
    prob_u   = find_col(df, "prob_uncal")

    if not y_col or not prob_cal:
        warn(f"Required columns missing. Available: {list(df.columns)}")
        return

    y_true  = df[y_col].astype(int).values
    y_calib = df[prob_cal].astype(float).values
    y_uncal = (sigmoid(df[logit_u].astype(float).values) if logit_u
               else df[prob_u].astype(float).values if prob_u else None)

    grid = np.linspace(0.0, 1.0, N_ROC_PTS)

    fpr_c, tpr_c, thr_c = roc_curve(y_true, y_calib)
    auc_c = roc_auc_score(y_true, y_calib)
    tpr_c_i = np.interp(grid, fpr_c, tpr_c)
    # sklearn: len(thr) may equal len(fpr) or len(fpr)-1 depending on version.
    # Use fpr[:len(thr)] for safe alignment across all sklearn versions.
    thr_c_i = np.interp(grid, fpr_c[:len(thr_c)], thr_c)

    out = pd.DataFrame({"fpr_calib": grid, "tpr_calib": tpr_c_i,
                         "thr_calib": thr_c_i})

    if y_uncal is not None:
        fpr_u, tpr_u, thr_u = roc_curve(y_true, y_uncal)
        auc_u = roc_auc_score(y_true, y_uncal)
        out["fpr_uncal"] = grid
        out["tpr_uncal"] = np.interp(grid, fpr_u, tpr_u)
        out["thr_uncal"] = np.interp(grid, fpr_u[:len(thr_u)], thr_u)
    else:
        auc_u = np.nan

    out["diagonal_x"] = grid
    out["diagonal_y"] = grid
    out = out.round(6)

    out_path = pics_path("testing", "test_roc.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved {N_ROC_PTS} interpolated points")
    ok(f"→ {out_path}")
    print(f"  AUROC (calibrated)   = {auc_c:.4f}")
    if not np.isnan(auc_u):
        print(f"  AUROC (uncalibrated) = {auc_u:.4f}")
    print("  OriginLab: X=fpr_calib | Y=tpr_calib (solid, label AUC)")
    print("             X=fpr_uncal | Y=tpr_uncal (dashed)")
    print("             X=diagonal_x | Y=diagonal_y (dashed grey)")


# ──────────────────────────────────────────────────────────────────────────────
# Fig E-2  Test PR Curve  (uncal vs calib)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "Precision-Recall Curve — Independent Test Set"
# X-axis : Recall (Sensitivity)
# Y-axis : Precision (PPV)
# Curves : Calibrated (solid, AUPRC in legend)
#           Uncalibrated (dashed, AUPRC in legend)
#           Prevalence baseline (dotted grey)
# Note   : Preferred over ROC for imbalanced datasets (active ~10%)
# Ref    : Saito & Rehmsmeier 2015 — https://doi.org/10.1371/journal.pone.0118432
#
# CSV columns (N_ROC_PTS rows):
#   recall_ref | prec_calib | prec_uncal | baseline
#
# OriginLab:
#   X=recall_ref  Y=prec_calib (solid blue, label: AUPRC=X.XXXX)
#   X=recall_ref  Y=prec_uncal (dashed orange, label: AUPRC=X.XXXX)
#   X=recall_ref  Y=baseline   (dotted grey, label: Prevalence=X.XXX)
#
# Source : outputs/test_evaluation/.../Best_fold{N}/test_preds.csv
# Output : pics_outputs/.../testing/test_pr.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_E2_pr() -> None:
    banner("Fig E-2: Test PR Curve (Calibrated vs Uncalibrated)")

    df = _load_test_preds()
    if df is None:
        return

    y_col    = find_col(df, "y_true", "label")
    prob_cal = find_col(df, "prob_calib")
    logit_u  = find_col(df, "logit_uncal")
    prob_u   = find_col(df, "prob_uncal")

    if not y_col or not prob_cal:
        warn(f"Required columns missing. Available: {list(df.columns)}")
        return

    y_true  = df[y_col].astype(int).values
    y_calib = df[prob_cal].astype(float).values
    y_uncal = (sigmoid(df[logit_u].astype(float).values) if logit_u
               else df[prob_u].astype(float).values if prob_u else None)

    prevalence = float(y_true.mean())
    grid       = np.linspace(0.0, 1.0, N_ROC_PTS)

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_calib)
    ap_c = average_precision_score(y_true, y_calib)
    prec_c_i = np.interp(grid, rec_c[::-1], prec_c[::-1])

    out = pd.DataFrame({"recall_ref": grid, "prec_calib": prec_c_i,
                         "baseline": np.full(N_ROC_PTS, round(prevalence, 6))})

    if y_uncal is not None:
        prec_u, rec_u, _ = precision_recall_curve(y_true, y_uncal)
        ap_u = average_precision_score(y_true, y_uncal)
        out["prec_uncal"] = np.interp(grid, rec_u[::-1], prec_u[::-1])
    else:
        ap_u = np.nan

    out = out.round(6)
    out_path = pics_path("testing", "test_pr.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved {N_ROC_PTS} interpolated points  prevalence={prevalence:.3f}")
    ok(f"→ {out_path}")
    print(f"  AUPRC (calibrated)   = {ap_c:.4f}")
    if not np.isnan(ap_u):
        print(f"  AUPRC (uncalibrated) = {ap_u:.4f}")
    print("  OriginLab: X=recall_ref | Y=prec_calib (solid)")
    print(f"             Y=prec_uncal (dashed) | Y=baseline (dotted, {prevalence:.3f})")


# ──────────────────────────────────────────────────────────────────────────────
# Fig E-3  Test Reliability Diagram  (Before/After TS)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "Reliability Diagram — Independent Test Set (Temperature Scaling)"
# X-axis : Mean Predicted Confidence (Bin Midpoint)
# Y-axis : Fraction of Positives (Observed Accuracy)
# Same format as T-3 but on independent test set.
# Annotation: ECE_uncal, ECE_calib, Brier Score
# Ref    : Guo et al. ICML 2017 — https://arxiv.org/abs/1706.04599
#          IEEE TMI 2024 (doi:10.1109/TMI.2024.3353762)
#
# CSV columns (N_CAL_BINS rows):
#   bin_lower | bin_upper | bin_mid | n_samples |
#   conf_uncal | acc_uncal |
#   conf_calib | acc_calib |
#   diagonal   | gap_calib
#
# OriginLab: same as T-3
#
# Source : outputs/test_evaluation/.../Best_fold{N}/test_preds.csv
# Output : pics_outputs/.../testing/test_calibration_bins.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_E3_test_calibration() -> None:
    banner("Fig E-3: Test Reliability Diagram (Before/After TS)")

    df = _load_test_preds()
    if df is None:
        return

    y_col    = find_col(df, "y_true", "label")
    prob_cal = find_col(df, "prob_calib")
    logit_u  = find_col(df, "logit_uncal")
    prob_u   = find_col(df, "prob_uncal")

    if not y_col or not prob_cal:
        warn(f"Required columns missing. Available: {list(df.columns)}")
        return

    y_true  = df[y_col].astype(int).values
    y_calib = df[prob_cal].astype(float).values
    y_uncal = (sigmoid(df[logit_u].astype(float).values) if logit_u
               else df[prob_u].astype(float).values if prob_u else None)

    bins_cal, ece_cal = calibration_bins(y_true, y_calib, N_CAL_BINS)
    out = bins_cal.rename(columns={"acc": "acc_calib", "conf": "conf_calib",
                                   "gap": "gap_calib"})

    if y_uncal is not None:
        bins_u, ece_uncal = calibration_bins(y_true, y_uncal, N_CAL_BINS)
        out["acc_uncal"]  = bins_u["acc"].values
        out["conf_uncal"] = bins_u["conf"].values
    else:
        ece_uncal = np.nan
        out["acc_uncal"]  = np.nan
        out["conf_uncal"] = np.nan

    brier = round(float(brier_score_loss(y_true, y_calib)), 6)

    out_path = pics_path("testing", "test_calibration_bins.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved {N_CAL_BINS} bins")
    ok(f"→ {out_path}")
    print(f"  ECE   (calibrated)   = {ece_cal:.4f}")
    if not np.isnan(ece_uncal):
        print(f"  ECE   (uncalibrated) = {ece_uncal:.4f}")
    print(f"  Brier (calibrated)   = {brier:.4f}")
    print("  OriginLab: X=bin_mid")
    print("    Y=acc_calib (solid blue) | Y=acc_uncal (dashed orange)")
    print("    Y=diagonal (dashed grey) | Bar=n_samples (2nd axis)")


# ──────────────────────────────────────────────────────────────────────────────
# Fig E-4  Normalised Confusion Matrix  (test set, calibrated)
# ──────────────────────────────────────────────────────────────────────────────
# Title  : "Confusion Matrix — Independent Test Set"
# X-axis : Predicted Label  (inactive | active)
# Y-axis : True Label       (inactive | active)
# Values : Count (raw) + Row-normalised % per cell
# Note   : Row normalisation = per-class recall (Sens/Spec per class)
#
# CSV flat format (4 rows = 4 cells):
#   true_label | pred_label | count | rate | cell_label
#   rows: TN, FP, FN, TP
#
# OriginLab: Heatmap; X=pred_label; Y=true_label; Z=rate (0–1)
#            Annotate each cell with cell_label (e.g. "42\n(87.5%)")
#
# Source : outputs/test_evaluation/.../Best_fold{N}/test_preds.csv
# Output : pics_outputs/.../testing/test_confusion_matrix.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_E4_confusion_matrix() -> None:
    banner("Fig E-4: Normalised Confusion Matrix (Test Set)")

    df = _load_test_preds()
    if df is None:
        return

    y_col    = find_col(df, "y_true", "label")
    prob_cal = find_col(df, "prob_calib")

    if not y_col or not prob_cal:
        warn(f"Required columns missing. Available: {list(df.columns)}")
        return

    y_true = df[y_col].astype(int).values
    y_pred = (df[prob_cal].astype(float).values >= 0.5).astype(int)

    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    rows = [
        {"true_label": "inactive (0)", "pred_label": "inactive (0)",
         "count": int(tn), "rate": round(float(cm_norm[0, 0]), 4),
         "cell_label": f"{tn}\n({cm_norm[0,0]*100:.1f}%)"},
        {"true_label": "inactive (0)", "pred_label": "active (1)",
         "count": int(fp), "rate": round(float(cm_norm[0, 1]), 4),
         "cell_label": f"{fp}\n({cm_norm[0,1]*100:.1f}%)"},
        {"true_label": "active (1)",   "pred_label": "inactive (0)",
         "count": int(fn), "rate": round(float(cm_norm[1, 0]), 4),
         "cell_label": f"{fn}\n({cm_norm[1,0]*100:.1f}%)"},
        {"true_label": "active (1)",   "pred_label": "active (1)",
         "count": int(tp), "rate": round(float(cm_norm[1, 1]), 4),
         "cell_label": f"{tp}\n({cm_norm[1,1]*100:.1f}%)"},
    ]
    out = pd.DataFrame(rows)
    out_path = pics_path("testing", "test_confusion_matrix.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    ok(f"Saved 4-cell confusion matrix")
    ok(f"→ {out_path}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Sensitivity = {cm_norm[1,1]:.4f}  Specificity = {cm_norm[0,0]:.4f}")
    print("  OriginLab: Heatmap; X=pred_label; Y=true_label; Z=rate")
    print("             Annotate each cell with cell_label (count + %)")


# ==============================================================================
# Summary table
# ==============================================================================

def print_summary_table() -> None:
    banner("Output Summary — All Origin CSV Files")

    ROOT = f"{PICS_ROOT}/{MODEL_NAME}/{MODALITY}/{RUN_TAG}"
    items = [
        ("T-1", "Training",
         "5-Fold CV Learning Curves",
         "Epoch",
         "BCE Loss / Accuracy (Mean ± SD)",
         "training/learning_curves.csv"),
        ("T-2", "Training",
         "Per-Fold Val AUROC Summary",
         "Fold ID (1–5, Mean, SD)",
         "Val AUROC",
         "training/val_auc_curves.csv"),
        ("T-3", "Training",
         "OOF Reliability Diagram (Before/After TS)",
         "Mean Predicted Confidence",
         "Fraction of Positives",
         "training/oof_calibration_bins.csv"),
        ("E-1", "Testing",
         "ROC Curve — Independent Test Set",
         "FPR (1 - Specificity)",
         "TPR (Sensitivity)",
         "testing/test_roc.csv"),
        ("E-2", "Testing",
         "Precision-Recall Curve — Test Set",
         "Recall (Sensitivity)",
         "Precision (PPV)",
         "testing/test_pr.csv"),
        ("E-3", "Testing",
         "Reliability Diagram — Test Set",
         "Mean Predicted Confidence",
         "Fraction of Positives",
         "testing/test_calibration_bins.csv"),
        ("E-4", "Testing",
         "Normalised Confusion Matrix — Test Set",
         "Predicted Label",
         "True Label",
         "testing/test_confusion_matrix.csv"),
    ]

    print(f"\n  Output root: {ROOT}\n")
    print(f"  {'Fig':<5} {'Stage':<10} {'Title':<44} {'X-axis':<28} {'File'}")
    print(f"  {'─'*5} {'─'*10} {'─'*44} {'─'*28} {'─'*38}")
    for r in items:
        full   = os.path.join(ROOT, r[5])
        status = "✓" if os.path.isfile(full) else "✗"
        print(f"  {r[0]:<5} {r[1]:<10} {r[2][:44]:<44} {r[3][:28]:<28} "
              f"{status} {r[5]}")

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("  EXPORT ORIGIN PLOTS  —  mCNV Base Model")
    print("=" * 60)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Modality : {MODALITY}")
    print(f"  Run tag  : {RUN_TAG}")
    print(f"  Best fold: {BEST_FOLD}")
    print(f"  Source   : {PROJECT_ROOT}/outputs/  (read-only)")
    print(f"  Output   : {PICS_ROOT}/  (new, isolated)")

    # ── Training ──────────────────────────────────────────────────────────────
    export_T1_learning_curves()
    export_T2_val_auc_summary()
    export_T3_oof_calibration()

    # ── Testing ───────────────────────────────────────────────────────────────
    export_E1_roc()
    export_E2_pr()
    export_E3_test_calibration()
    export_E4_confusion_matrix()

    print_summary_table()
    print("\n  DONE.\n")


if __name__ == "__main__":
    main()