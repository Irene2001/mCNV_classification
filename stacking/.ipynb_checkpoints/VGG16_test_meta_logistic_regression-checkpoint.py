# VGG16_test_meta_logistic_regression.py

"""
Stacking Meta-Learner (Logistic Regression) 

Data flow (strictly no leakage)
--------------------------------
  Base model test_preds.csv (split_set="test", from test_singlemode.py):
    exam_key | y_true | logit_uncal | prob_uncal | temperature | logit_calib | prob_calib

  Feature extraction (must match build_meta_dataset.py FEATURE_TYPE + USE_CALIB):
    e.g. logit_calib -> renamed to oct0_feat / oct1_feat / octa3_feat

  Inner join on exam_key -> X_test  (N x 3, same columns as training)
  Pipeline.predict_proba(X_test)[:, 1]  ->  P(active)

Feature naming contract (mirrors build_meta_dataset.py exactly)
----------------------------------------------------------------
  build_meta_dataset.py renames: feature_col -> "oct0_feat" / "oct1_feat" / "octa3_feat"
  train_meta_logistic_regression.py trains on those columns.
  This script MUST rename the same feature_col identically.

Input paths
-----------
  TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/test_preds.csv
"""

import csv
import json
import os
import time
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit as _stable_sigmoid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# * CONFIG  --  Edit only this section
# ==============================================================================

PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"
VGG16_BASE_DIR = os.path.join(PROJECT_ROOT, "VGG16_outputs")

# * Model backbone name
MODEL_NAME = "vgg16"
STRATEGY_NAME = "Partial_B5"

# * Modalities in the stacking ensemble (must match build_meta_dataset.py)
MODALITIES = ["OCT0", "OCT1", "OCTA3"]

# * Feature type -- must EXACTLY match build_meta_dataset.py settings
#   ("logit", True)  -> logit_calib  <- recommended
#   ("logit", False) -> logit_uncal
#   ("prob",  True)  -> prob_calib
#   ("prob",  False) -> prob_uncal
FEATURE_TYPE = "logit"
USE_CALIB    = True

# * Path to trained meta-learner pipeline (.pkl)
META_MODEL_PATH = (
    "/data/Irene/SwinTransformer/Swin_Meta/VGG16_outputs/meta_training/"
    "vgg16__logit__calibTrue/"
    "Partial_B5/"
    "OCT0_LR8e-06_OCT1_LR9e-06_OCTA3_LR8e-06/"
    "meta_lr_model.pkl"
)

# * test_preds.csv root (produced by test_singlemode.py)
TEST_EVAL_ROOT = os.path.join(VGG16_BASE_DIR, "test_evaluation")

# * run_tag and best_fold per modality
#   Must match runs whose OOF was used by build_meta_dataset.py
#   (derived from meta model folder: OCT0_LR2e-06_OCT1_LR4e-06_OCTA3_LR3e-06)
MODALITY_RUN_TAG = {
    "OCT0":  "BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.11_0.89_2_WSon_1_2.9",
    "OCT1":  "BS16_EP100_LR9e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.113_0.887_2_WSon_1_2.8",
    "OCTA3": "BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.13_0.87_2_WSon_1_2.6",
}
MODALITY_BEST_FOLD = {
    "OCT0":  2,
    "OCT1":  5,
    "OCTA3": 1,
}

# * Decision threshold
THRESHOLD = 0.5

# * ECE bins (Guo et al. 2017)
ECE_N_BINS = 10

CLASS_NAMES = ["inactive", "active"]

# ==============================================================================


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid via scipy.special.expit."""
    return _stable_sigmoid(x).astype(np.float32)


# ------------------------------------------------------------------------------
# Feature column resolution
# ------------------------------------------------------------------------------

def resolve_feature_col(feature_type: str, use_calib: bool) -> str:
    """
    Mirror build_meta_dataset.py -> resolve_feature_col().
    ("logit", True) -> "logit_calib", etc.
    """
    mapping = {
        ("logit", True):  "logit_calib",
        ("logit", False): "logit_uncal",
        ("prob",  True):  "prob_calib",
        ("prob",  False): "prob_uncal",
    }
    key = (feature_type, use_calib)
    if key not in mapping:
        raise ValueError(f"Invalid (FEATURE_TYPE, USE_CALIB)={key}. "
                         f"Valid: {list(mapping.keys())}")
    return mapping[key]


def modality_feat_col(modality: str) -> str:
    """
    Mirror build_meta_dataset.py rename convention:
      OCT0 -> oct0_feat, OCT1 -> oct1_feat, OCTA3 -> octa3_feat
    """
    return f"{modality.lower()}_feat"


# ------------------------------------------------------------------------------
# Load base model test predictions
# ------------------------------------------------------------------------------

def load_test_preds(modality: str, feature_col: str, logf) -> pd.DataFrame:
    """
    Load test_preds.csv from test_singlemode.py for one modality.

    Path: TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/test_preds.csv

    Columns in test_preds.csv:
      exam_key | y_true | logit_uncal | prob_uncal | temperature | logit_calib | prob_calib

    Returns DataFrame with columns:
      exam_key | y_true | <modality_feat_col>
      (feature col renamed to mirror build_meta_dataset.py convention)
    """
    path = os.path.join(
        TEST_EVAL_ROOT, MODEL_NAME, STRATEGY_NAME, modality,
        MODALITY_RUN_TAG[modality],
        f"Best_fold{MODALITY_BEST_FOLD[modality]}",
        "test_preds.csv",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"test_preds.csv not found for {modality}:\n  {path}\n"
            "Run test_singlemode.py for this modality first."
        )

    df = pd.read_csv(path)

    required = {"exam_key", "y_true", feature_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{modality} test_preds.csv missing columns: {missing}\n"
                         f"Available: {list(df.columns)}\nFile: {path}")

    dupes = df["exam_key"].astype(str).duplicated().sum()
    if dupes > 0:
        raise RuntimeError(f"{modality} test_preds.csv has {dupes} duplicate exam_key rows.")

    df["exam_key"] = df["exam_key"].astype(str)
    df["y_true"]   = df["y_true"].astype(int)

    feat_renamed = modality_feat_col(modality)
    df = df[["exam_key", "y_true", feature_col]].copy()
    df = df.rename(columns={feature_col: feat_renamed})

    n_active   = int((df["y_true"] == 1).sum())
    n_inactive = int((df["y_true"] == 0).sum())
    log(logf, f"  {modality}: {len(df)} samples  "
              f"active={n_active}  inactive={n_inactive}  "
              f"{feature_col} -> {feat_renamed}")
    return df


# ------------------------------------------------------------------------------
# Build meta-test feature matrix
# ------------------------------------------------------------------------------

def build_meta_test(
    dfs: Dict[str, pd.DataFrame],
    feat_cols_trained: List[str],
    logf,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Inner-join test DataFrames on exam_key (mirrors PAIRING_STRATEGY="inner").
    Verifies feature column names match training exactly.
    Returns (aligned_df, X_test, y_test).
    """
    mods   = list(dfs.keys())
    merged = dfs[mods[0]][["exam_key", "y_true"]].copy()
    for mod in mods:
        merged = merged.merge(
            dfs[mod].drop(columns=["y_true"]), on="exam_key", how="inner"
        )

    assert set(merged["y_true"].unique()).issubset({0, 1}), \
        "y_true contains values outside {0, 1}"

    actual_feat = [c for c in merged.columns if c not in ("exam_key", "y_true")]
    if actual_feat != feat_cols_trained:
        raise RuntimeError(
            "Test feature columns do NOT match training.\n"
            f"  Expected: {feat_cols_trained}\n"
            f"  Actual:   {actual_feat}"
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


# ------------------------------------------------------------------------------
# ECE  (Guo et al. 2017, equal-width bins)
# ------------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """
    ECE = sum_b (|B_b|/N) |acc(B_b) - conf(B_b)|
    acc(B_b)  = fraction of positives in bin = y_true[mask].mean()
    conf(B_b) = mean predicted P(Y=1) in bin = y_prob[mask].mean()
    Ref: Guo et al. (2017). ICML. Eq.3
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_n   = len(y_true)
    ece, bins_data = 0.0, []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask |= (y_prob == hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bins_data.append({"lower": lo, "upper": hi, "n": 0,
                              "accuracy": None, "confidence": None})
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (n_bin / total_n) * abs(acc - conf)
        bins_data.append({"lower": lo, "upper": hi, "n": n_bin,
                          "accuracy": acc, "confidence": conf})
    return float(ece), bins_data


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[dict, List[dict]]:
    """
    Comprehensive evaluation metrics.
    Returns (metrics_dict, ece_bins).

    Metrics (IEEE TMI / Nature Medicine):
      Discriminative : AUROC, AUPRC
      Threshold-based: Balanced Accuracy, Sensitivity, Specificity,
                       PPV, NPV, F1-active, F1-macro, Accuracy
      Calibration    : ECE (Guo 2017), Brier Score
      Confusion      : TP, FP, FN, TN
    """
    if len(np.unique(y_true)) < 2:
        raise RuntimeError(
            "Test set contains only one class -- AUROC and binary metrics undefined."
        )

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    n_inactive = int((y_true == 0).sum())
    n_active   = int((y_true == 1).sum())
    ece, bins  = compute_ece(y_true, y_prob, ECE_N_BINS)

    metrics = {
        # Dataset
        "n_total":           int(len(y_true)),
        "n_active":          n_active,
        "n_inactive":        n_inactive,
        "imbalance_ratio":   round(n_inactive / n_active, 2) if n_active > 0 else None,
        "threshold":         float(threshold),
        "ece_n_bins":        ECE_N_BINS,
        # Discriminative
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
        "ece":               ece,
        "brier_score":       float(brier_score_loss(y_true, y_prob)),
        # Confusion matrix
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    return metrics, bins


# ------------------------------------------------------------------------------
# Plots  (Matplotlib + Seaborn -- style identical to test_singlemode.py)
# Standard in medical imaging AI: IEEE TMI, Nature Medicine, MICCAI
# ------------------------------------------------------------------------------

# Style constants (adjust here to change all figures at once)
_TITLE_SIZE      = 16
_AXIS_LABEL_SIZE = 14
_TICK_SIZE       = 13
_LEGEND_SIZE     = 11
_ANNOT_SIZE      = 12
_FIG_DPI         = 150

# Curve colours (mirrors test_singlemode.py)
_COL_META        = "#2563EB"   # Meta-LR ROC line
_COL_META_PR     = "#7C3AED"   # Meta-LR PR line
_COL_META_REL    = "#2563EB"   # Meta-LR reliability diagram
_COL_RANDOM      = "#374151"   # Random baseline diagonal
_COL_BASELINE    = "#DC2626"   # Prevalence baseline (PR)
_COL_PERFECT_CAL = "#374151"   # Perfect calibration diagonal

# Confusion matrix
_CM_COLOR_LOW    = "#dbeafe"
_CM_COLOR_HIGH   = "#1e40af"
_CM_ANNOT_SIZE   = 20


def _set_style() -> None:
    """Apply global rcParams once per figure call."""
    plt.rcParams.update({
        "font.size":          _ANNOT_SIZE,
        "axes.titlesize":     _TITLE_SIZE,
        "axes.labelsize":     _AXIS_LABEL_SIZE,
        "xtick.labelsize":    _TICK_SIZE,
        "ytick.labelsize":    _TICK_SIZE,
        "figure.dpi":         _FIG_DPI,
        "savefig.dpi":        _FIG_DPI,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    out_path: str, title: str,
) -> None:
    _set_style()
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmap = LinearSegmentedColormap.from_list(
        "cm_cmap", [_CM_COLOR_LOW, _CM_COLOR_HIGH]
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=1.0, linecolor="white", ax=ax, cbar=True,
        annot_kws={"size": _CM_ANNOT_SIZE, "weight": "bold"},
    )
    ax.set_xlabel("Predicted label", fontweight="bold")
    ax.set_ylabel("True label",      fontweight="bold")
    ax.set_title(title,              fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    out_path: str, title: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve for meta-learner output probability."""
    _set_style()
    auc_val       = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=_COL_META, lw=2,
            label=f"Meta-LR  AUC={auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color=_COL_RANDOM, lw=0.8,
            linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=_LEGEND_SIZE)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return fpr, tpr, thr


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    out_path: str, title: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-Recall curve. AUPRC is primary for imbalanced data."""
    _set_style()
    ap       = average_precision_score(y_true, y_prob)
    baseline = float((y_true == 1).mean())
    pre, rec, thr = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, pre, color=_COL_META_PR, lw=2,
            label=f"Meta-LR  AUPRC={ap:.4f}")
    ax.axhline(y=baseline, color=_COL_BASELINE, lw=0.8, linestyle=":",
               label=f"Baseline prevalence={baseline:.3f}")
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax.set_ylabel("Precision (PPV)",      fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=_LEGEND_SIZE)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return pre, rec, thr


def plot_reliability_diagram(
    bins: List[dict], ece: float,
    out_path: str, title: str,
) -> None:
    """
    Reliability diagram for meta-learner output.
    acc(B) = fraction of positives (binary posterior calibration check).
    Ref: Guo et al. (2017). ICML.
    """
    _set_style()
    xs, ys, ns = [], [], []
    for b in bins:
        if b["accuracy"] is not None:
            xs.append(b["confidence"])
            ys.append(b["accuracy"])
            ns.append(b["n"])
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=_COL_PERFECT_CAL, lw=0.8,
            linestyle="--", alpha=0.5, label="Perfect calibration")
    if len(xs) > 0:
        ax.scatter(xs, ys, s=ns / max(ns.max(), 1) * 300 + 30,
                   color=_COL_META_REL, alpha=0.85, zorder=4,
                   label=f"Meta-LR  ECE={ece:.4f}")
        ax.plot(xs, ys, color=_COL_META_REL, lw=1.5)
    ax.set_xlabel("Mean predicted confidence", fontweight="bold")
    ax.set_ylabel("Fraction of positives (accuracy)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=_LEGEND_SIZE)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.25)
    ax.text(0.98, 0.04, f"ECE = {ece:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=_ANNOT_SIZE - 1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4",
                      edgecolor="#86efac"))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------------------
# Save helpers (CSV)
# ------------------------------------------------------------------------------

def save_metrics_csv(metrics: dict, out_path: str) -> None:
    """Tall-format CSV: one row per metric (metric | value)."""
    rows = [
        ("AUROC",             metrics["auroc"]),
        ("AUPRC",             metrics["auprc"]),
        ("Balanced_Accuracy", metrics["balanced_accuracy"]),
        ("Sensitivity",       metrics["sensitivity"]),
        ("Specificity",       metrics["specificity"]),
        ("PPV",               metrics["ppv"]),
        ("NPV",               metrics["npv"]),
        ("F1_active",         metrics["f1_active"]),
        ("F1_macro",          metrics["f1_macro"]),
        ("Accuracy",          metrics["accuracy"]),
        ("ECE",               metrics["ece"]),
        ("Brier_Score",       metrics["brier_score"]),
        ("threshold",         metrics["threshold"]),
        ("TP",                metrics["TP"]),
        ("FP",                metrics["FP"]),
        ("FN",                metrics["FN"]),
        ("TN",                metrics["TN"]),
        ("n_total",           metrics["n_total"]),
        ("n_active",          metrics["n_active"]),
        ("n_inactive",        metrics["n_inactive"]),
        ("imbalance_ratio",   metrics.get("imbalance_ratio")),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        out_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )


def save_calibration_csv(bins: List[dict], out_path: str) -> None:
    """Calibration bin data for Origin / Excel re-plotting."""
    rows = [
        {"bin_lower": b["lower"], "bin_upper": b["upper"],
         "bin_mid": (b["lower"] + b["upper"]) / 2,
         "n": b["n"], "accuracy": b["accuracy"], "confidence": b["confidence"]}
        for b in bins
    ]
    pd.DataFrame(rows).to_csv(
        out_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )


# ------------------------------------------------------------------------------
# Save results (all outputs)
# ------------------------------------------------------------------------------

def save_results(
    aligned_df: pd.DataFrame,
    y_prob: np.ndarray,
    metrics: dict,
    bins: List[dict],
    run_cfg: dict,
    out_dir: str,
    logf,
) -> None:
    ensure_dir(out_dir)

    threshold = metrics["threshold"]
    y_true    = aligned_df["y_true"].to_numpy()
    y_pred    = (y_prob >= threshold).astype(int)
    mod_str   = "_".join(run_cfg.get("modalities", MODALITIES))
    title_pfx = f"{MODEL_NAME.upper()} Meta-LR ({mod_str}) Test Set"

    # 1. Confusion matrix PNG
    plot_confusion_matrix(
        y_true, y_pred,
        out_path=os.path.join(out_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix",
    )

    # 2. ROC curve PNG + CSV
    fpr, tpr, roc_thr = plot_roc_curve(
        y_true, y_prob,
        out_path=os.path.join(out_dir, "roc_curve.png"),
        title=f"ROC Curve",
    )
    with open(os.path.join(out_dir, "meta_lr_roc_curve.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for fpr_v, tpr_v, thr_v in zip(fpr, tpr, roc_thr):
            w.writerow([f"{fpr_v:.6f}", f"{tpr_v:.6f}", f"{thr_v:.6f}"])

    # 3. PR curve PNG + CSV
    pre, rec, pr_thr = plot_pr_curve(
        y_true, y_prob,
        out_path=os.path.join(out_dir, "pr_curve.png"),
        title=f"Precision-Recall Curve",
    )
    pr_thr_padded = list(pr_thr) + [float("nan")]
    with open(os.path.join(out_dir, "meta_lr_pr_curve.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["precision", "recall", "threshold"])
        for p_v, r_v, t_v in zip(pre, rec, pr_thr_padded):
            t_str = f"{t_v:.6f}" if not (isinstance(t_v, float) and np.isnan(t_v)) else ""
            w.writerow([f"{p_v:.6f}", f"{r_v:.6f}", t_str])

    # 4. Reliability diagram PNG + calibration CSV
    plot_reliability_diagram(
        bins, metrics["ece"],
        out_path=os.path.join(out_dir, "reliability_diagram.png"),
        title=f"Reliability Diagram",
    )
    save_calibration_csv(bins, os.path.join(out_dir, "meta_lr_calibration_data.csv"))

    # 5. Metrics CSV (tall-format)
    save_metrics_csv(metrics, os.path.join(out_dir, "meta_lr_test_metrics.csv"))

    # 6. Predictions CSV
    pred_df = aligned_df[["exam_key", "y_true"]].copy()
    pred_df["prob_active"] = y_prob
    pred_df["y_pred"]      = y_pred
    pred_df["correct"]     = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df.to_csv(
        os.path.join(out_dir, "meta_lr_test_predictions.csv"),
        index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    # 7. Results JSON
    save_json(
        os.path.join(out_dir, "meta_lr_test_results.json"),
        {"run_config": run_cfg, "metrics": metrics}
    )

    # 8. Human-readable report
    sep  = "=" * 70
    sep2 = "-" * 70
    with open(os.path.join(out_dir, "meta_lr_test_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("STACKING META-LEARNER -- INDEPENDENT TEST SET EVALUATION\n")
        f.write(f"{sep}\n\n")

        f.write("[RUN CONFIG]\n")
        for k, v in run_cfg.items():
            f.write(f"  {k:<36}: {v}\n")

        n, na, ni = metrics["n_total"], metrics["n_active"], metrics["n_inactive"]
        f.write("\n[TEST SET COMPOSITION]\n")
        f.write(f"  Total    : {n}\n")
        f.write(f"  Active   : {na}  ({na/n*100:.1f}%)\n")
        f.write(f"  Inactive : {ni}  ({ni/n*100:.1f}%)\n")
        imb = metrics.get("imbalance_ratio")
        f.write(f"  Imbalance: {f'{imb:.2f}:1' if imb is not None else 'undefined'}\n")

        f.write("\n[1. DISCRIMINATIVE PERFORMANCE  (threshold-independent)]\n")
        f.write(f"  AUROC : {metrics['auroc']:.4f}\n")
        f.write(f"  AUPRC : {metrics['auprc']:.4f}\n")

        f.write(f"\n[2. THRESHOLD-BASED METRICS  (threshold={threshold:.2f})]\n")
        for lbl, key in [
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Sensitivity",       "sensitivity"),
            ("Specificity",       "specificity"),
            ("PPV",               "ppv"),
            ("F1 (active)",       "f1_active"),
            ("F1 (macro)",        "f1_macro"),
            ("Accuracy",          "accuracy"),
        ]:
            f.write(f"  {lbl:<20}: {metrics[key]:.4f}\n")

        f.write("\n[3. CLINICAL SAFETY METRICS]\n")
        f.write(f"  NPV  : {metrics['npv']:.4f}  <- rule-out safety\n")

        f.write("\n[4. CALIBRATION]\n")
        f.write(f"  ECE         : {metrics['ece']:.4f}  "
                f"({ECE_N_BINS}-bin equal-width\n")
        f.write(f"  Brier Score : {metrics['brier_score']:.4f}  (lower = better)\n")

        f.write("\n[5. CONFUSION MATRIX]\n")
        f.write( "                    Pred active  Pred inactive\n")
        f.write(f"  True active     {metrics['TP']:>11}  {metrics['FN']:>14}\n")
        f.write(f"  True inactive   {metrics['FP']:>11}  {metrics['TN']:>14}\n")

        f.write(f"\n{sep2}\n")
        f.write("SUMMARY TABLE\n")
        f.write(f"{sep2}\n")
        f.write(f"  {'Metric':<22}  {'Value':>10}\n")
        f.write(f"  {'-'*22}  {'-'*10}\n")
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
            ("ECE",               "ece"),
            ("Brier Score",       "brier_score"),
        ]:
            f.write(f"  {lbl:<22}  {metrics[key]:>10.4f}\n")
        f.write(f"{sep2}\n")

        f.write("\n[SKLEARN CLASSIFICATION REPORT]\n")
        f.write(classification_report(y_true, y_pred,
                                      target_names=["inactive", "active"]))

        f.write("\n[OUTPUT FILES]\n")
        for fname in [
            "confusion_matrix.png",
            "roc_curve.png",
            "pr_curve.png",
            "reliability_diagram.png",
            "meta_lr_test_metrics.csv      (tall-format: one row per metric)",
            "meta_lr_test_predictions.csv  (per-sample: prob_active, y_pred)",
            "meta_lr_test_results.json     (run_config + all metrics)",
            "meta_lr_roc_curve.csv",
            "meta_lr_pr_curve.csv",
            "meta_lr_calibration_data.csv",
            "meta_lr_test_report.txt",
            "meta_test_evaluation.log",
        ]:
            f.write(f"  {fname}\n")

        f.write("\n[REFERENCES]\n")
        f.write("  AUROC/AUPRC : Saito & Rehmsmeier, PLOS ONE 2015\n")
        f.write("    https://doi.org/10.1371/journal.pone.0118432\n")
        f.write("  ECE         : Guo et al., ICML 2017\n")
        f.write("    https://proceedings.mlr.press/v70/guo17a.html\n")
        f.write("  NPV Clinical: STARD-AI, Nature Medicine 2020\n")
        f.write("    https://doi.org/10.1038/s41591-020-0941-1\n")

    log(logf, f"All outputs saved -> {out_dir}")
    for fname in ["confusion_matrix.png", "roc_curve.png",
                  "pr_curve.png", "reliability_diagram.png",
                  "meta_lr_test_metrics.csv", "meta_lr_test_predictions.csv",
                  "meta_lr_test_results.json", "meta_lr_roc_curve.csv",
                  "meta_lr_pr_curve.csv", "meta_lr_calibration_data.csv",
                  "meta_lr_test_report.txt"]:
        log(logf, f"  {fname}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # Step 0: Resolve feature column + output path
    feature_col = resolve_feature_col(FEATURE_TYPE, USE_CALIB)
    meta_dir    = str(Path(META_MODEL_PATH).parent)
    out_dir     = os.path.join(meta_dir, "test_evaluation")
    ensure_dir(out_dir)

    logf = open(
        os.path.join(out_dir, "meta_test_evaluation.log"),
        "a", buffering=1, encoding="utf-8"
    )

    print(f"\n{'='*62}")
    print("VGG16 STACKING META-LEARNER TEST EVALUATION")
    print(f"{'='*62}")

    log(logf, "=" * 62)
    log(logf, "VGG16 STACKING META-LEARNER TEST EVALUATION")
    log(logf, "=" * 62)
    log(logf, f"model_name    : {MODEL_NAME}")
    log(logf, f"strategy_name : {STRATEGY_NAME}")
    log(logf, f"modalities    : {MODALITIES}")
    log(logf, f"feature_type  : {FEATURE_TYPE}  use_calib={USE_CALIB}")
    log(logf, f"feature_col   : {feature_col}")
    log(logf, f"threshold     : {THRESHOLD}")
    log(logf, f"ece_n_bins    : {ECE_N_BINS}")
    log(logf, f"meta_model    : {META_MODEL_PATH}")
    log(logf, f"out_dir       : {out_dir}")

    # Step 1: Load trained meta-learner pipeline
    log(logf, "-" * 62)
    log(logf, "Step 1: Load trained meta-learner pipeline")

    if not os.path.isfile(META_MODEL_PATH):
        raise FileNotFoundError(
            f"Meta model not found: {META_MODEL_PATH}\n"
            "Run train_meta_logistic_regression.py first."
        )
    pipe = joblib.load(META_MODEL_PATH)
    log(logf, f"Pipeline loaded: {type(pipe).__name__}")

    # Extract feature column names from trained model
    fitted_lr = (pipe.named_steps.get("lr")
                 or pipe.named_steps.get("logisticregression")
                 or list(pipe.named_steps.values())[-1])

    if hasattr(fitted_lr, "feature_names_in_"):
        feat_cols_trained: List[str] = fitted_lr.feature_names_in_.tolist()
        log(logf, f"Feature cols (from model): {feat_cols_trained}")
    else:
        feat_cols_trained = [modality_feat_col(m) for m in MODALITIES]
        log(logf, f"Feature cols (inferred): {feat_cols_trained}")

    # Override from meta_lr_config.json if available (most reliable)
    config_path = os.path.join(meta_dir, "meta_lr_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg_data = json.load(f)
        if "feature_cols" in cfg_data:
            feat_cols_trained = cfg_data["feature_cols"]
            log(logf, f"Feature cols (from config): {feat_cols_trained}")

    if hasattr(fitted_lr, "C_"):
        log(logf, f"Best C (from training): {float(fitted_lr.C_[0]):.6f}")

    # Step 2: Load base model test predictions
    log(logf, "-" * 62)
    log(logf, "Step 2: Load base model test predictions")
    log(logf, f"  feature_col='{feature_col}' -> renamed to <modality>_feat")

    dfs: Dict[str, pd.DataFrame] = {}
    for mod in MODALITIES:
        dfs[mod] = load_test_preds(mod, feature_col, logf)

    # Step 3: Inner join -> meta-test feature matrix
    log(logf, "-" * 62)
    log(logf, "Step 3: Build meta-test feature matrix (inner join)")

    aligned_df, X_test, y_test = build_meta_test(dfs, feat_cols_trained, logf)

    # Step 4: Predict
    log(logf, "-" * 62)
    log(logf, "Step 4: Meta-learner inference")

    y_prob = pipe.predict_proba(X_test)[:, 1]   # P(active)
    log(logf, f"Prediction : shape={y_prob.shape}  "
              f"range=[{y_prob.min():.4f}, {y_prob.max():.4f}]  "
              f"mean={y_prob.mean():.4f}")

    # Step 5: Compute metrics
    log(logf, "-" * 62)
    log(logf, f"Step 5: Compute metrics (threshold={THRESHOLD})")

    metrics, bins = compute_metrics(y_test, y_prob, THRESHOLD)

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
    log(logf, f"  ECE              : {metrics['ece']:.4f}")
    log(logf, f"  Brier Score      : {metrics['brier_score']:.4f}")
    log(logf, f"  TP={metrics['TP']}  FP={metrics['FP']}  "
              f"FN={metrics['FN']}  TN={metrics['TN']}")

    # Step 6: Generate plots + save all outputs
    log(logf, "-" * 62)
    log(logf, "Step 6: Generate plots and save all outputs")

    run_cfg = {
        "model_name":        MODEL_NAME,
        "modalities":        MODALITIES,
        "feature_type":      FEATURE_TYPE,
        "use_calib":         USE_CALIB,
        "feature_col":       feature_col,
        "feature_cols_X":    feat_cols_trained,
        "meta_model_path":   META_MODEL_PATH,
        "threshold":         THRESHOLD,
        "ece_n_bins":        ECE_N_BINS,
        "run_tags":          MODALITY_RUN_TAG,
        "best_folds":        MODALITY_BEST_FOLD,
        "test_eval_root":    TEST_EVAL_ROOT,
        "out_dir":           out_dir,
        "note": (
            "Independent test set -- never seen during base model training, "
            "validation, or meta-learner training."
        ),
    }
    save_results(aligned_df, y_prob, metrics, bins, run_cfg, out_dir, logf)

    # Done
    log(logf, "=" * 62)
    log(logf, "EVALUATION COMPLETE")
    log(logf, f"  AUROC   : {metrics['auroc']:.4f}")
    log(logf, f"  AUPRC   : {metrics['auprc']:.4f}")
    log(logf, f"  NPV     : {metrics['npv']:.4f}")
    log(logf, f"  ECE     : {metrics['ece']:.4f}")
    log(logf, f"  Results : {out_dir}")
    log(logf, "=" * 62)
    logf.close()


if __name__ == "__main__":
    main()