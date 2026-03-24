# VGG16_train_meta_logistic_regression.py

"""
Stacking Meta-Learner Training
mCNV binary classification (active / inactive)

Upstream
--------
build_meta_dataset.py → outputs/meta_dataset/<meta_tag>/<lr_tag>/meta_train_oof.csv
    Columns: exam_key, patient_id, fold_id, y_true,
             oct0_feat, oct1_feat, octa3_feat
    e.g. outputs/meta_dataset/swin_tiny__logit__calibTrue/
             OCT0_LR3e-06_OCT1_LR2e-06_OCTA3_LR2e-06/meta_train_oof.csv

Pipeline
--------
StandardScaler → LogisticRegressionCV(solver=lbfgs)

  alpha : compute_class_weight("balanced")
          α_j = N_total / (n_classes × N_j)
          ref: sklearn.utils.class_weight.compute_class_weight

  C     : LogisticRegressionCV grid {0.001,0.01,0.1,1,10}
          5-fold StratifiedKFold, scoring=roc_auc  ← C selection only
          ref: sklearn.linear_model.LogisticRegressionCV

  w     : lbfgs minimises L = -(1/N)Σαᵢ[y log p̂+(1-y)log(1-p̂)] + (λ/2)||w||²
          ref: sklearn.linear_model.LogisticRegression

Usage
-----
python train_meta_logistic_regression.py
"""

import os
import gc
import csv
import json
import time
import joblib

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, brier_score_loss, balanced_accuracy_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from training.model_factory import normalize_model_name, get_backbone_name
except ImportError:
    def normalize_model_name(n: str) -> str: return n.lower().strip()
    def get_backbone_name(n: str)    -> str: return n.lower().strip()


# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = "/data/Irene/SwinTransformer/Swin_Meta"

# Add VGG16_outputs & Partial_B5 folder!
VGG16_BASE_DIR = "/data/Irene/SwinTransformer/Swin_Meta/VGG16_outputs"
MODEL_NAME = "vgg16" 
STRATEGY_NAME = "Partial_B5"

META_DATASET_ROOT = os.path.join(VGG16_BASE_DIR, "meta_dataset")
META_OUTPUT_ROOT  = os.path.join(VGG16_BASE_DIR, "meta_training")

# ── Input path — ONLY THIS LINE needs to be changed per experiment run ────────
# Format: META_DATASET_ROOT/<meta_tag>/<lr_tag>/
#   meta_tag : "{model_name}__{feature_type}__calib{True|False}"
#   lr_tag   : per-modality learning rate identifiers from build_meta_dataset.py
META_DATASET_DIR  = os.path.join(
    META_DATASET_ROOT,
    "vgg16__logit__calibTrue",             
    "Partial_B5",
    "OCT0_LR8e-06_OCT1_LR9e-06_OCTA3_LR8e-06",
)

# Feature columns fixed by build_meta_dataset.py
FEAT_COLS = ["oct0_feat", "oct1_feat", "octa3_feat"]

# LogisticRegressionCV — sklearn >= 1.8: L2 is default, penalty= deprecated
C_GRID      = [0.001, 0.01, 0.1, 1.0, 10.0]   # log-uniform, 4 decades
CV_FOLDS    = 5           # internal — C selection only, NOT performance evaluation
CV_SCORING  = "roc_auc"   # binary AUC; robust to 1:8 imbalance
MAX_ITER    = 2000
RANDOM_SEED = 42


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


# ── Load meta-train dataset ───────────────────────────────────────────────────
def load_meta_train(meta_dataset_dir: str, logf) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load meta_train_oof.csv from build_meta_dataset.py.
    Validates columns, logs class distribution and alpha values.
    """
    path = os.path.join(meta_dataset_dir, "meta_train_oof.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"meta_train_oof.csv not found: {path}\n"
            "Run build_meta_dataset.py first."
        )

    df = pd.read_csv(path)
    missing = ({"exam_key", "y_true"} | set(FEAT_COLS)) - set(df.columns)
    if missing:
        raise ValueError(f"meta_train_oof.csv missing columns: {missing}")

    df["exam_key"] = df["exam_key"].astype(str)
    df["y_true"]   = df["y_true"].astype(int)
    assert set(df["y_true"].unique()).issubset({0, 1}), \
        f"Unexpected y_true values: {df['y_true'].unique()}"

    n  = len(df)
    na = int((df["y_true"] == 1).sum())
    ni = int((df["y_true"] == 0).sum())

    # α_j = N_total / (n_classes × N_j)  ← sklearn official formula
    cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                               y=df["y_true"].to_numpy())
    alpha_i, alpha_a = float(cw[0]), float(cw[1])

    log(logf, f"Loaded   : {path}")
    log(logf, f"Samples  : total={n}  active={na} ({na/n*100:.1f}%)  "
              f"inactive={ni} ({ni/n*100:.1f}%)  ratio={ni/na:.2f}:1")
    log(logf, f"Features : {FEAT_COLS}")
    log(logf, f"Alpha    : α_j = N/(2×N_j)  "
              f"α_active={alpha_a:.4f}  α_inactive={alpha_i:.4f}  "
              f"ratio={alpha_a/alpha_i:.2f}x  (active upweighted)")

    return df, FEAT_COLS


# ── Training ──────────────────────────────────────────────────────────────────
def train_meta_lr(
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: List[str],
    logf,
) -> Tuple[Pipeline, dict]:
    """
    Pipeline(StandardScaler → LogisticRegressionCV).

    StandardScaler : zero-mean unit-var; ensures L2 penalty equitable
                     across modalities (essential when mixing feature scales).
    LogisticRegressionCV:
      class_weight='balanced' — delegates alpha to compute_class_weight;
                                internally applies α per sample in loss.
      Cs / cv / scoring       — warm-start path over all Cs; StratifiedKFold
                                selects best C by roc_auc (C selection only).
      solver='lbfgs'          — L-BFGS-B; standard for binary LR with L2.
      refit=True              — final model trained on full X with best C.
    """
    log(logf, "Pipeline: StandardScaler → LogisticRegressionCV(solver=lbfgs)")
    log(logf, f"  Cs={C_GRID}  cv={CV_FOLDS}(StratifiedKFold)  "
              f"scoring={CV_SCORING}  class_weight=balanced  max_iter={MAX_ITER}")

    inner_cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    # sklearn >= 1.8: L2 is default; do NOT pass penalty= (deprecated)
    lrcv = LogisticRegressionCV(
        Cs=C_GRID,
        cv=inner_cv,
        scoring=CV_SCORING,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=MAX_ITER,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", lrcv)])

    t0 = time.time()
    pipe.fit(X, y)
    elapsed = time.time() - t0

    fitted_lr  = pipe.named_steps["lr"]
    best_C     = float(fitted_lr.C_[0])
    coef       = fitted_lr.coef_[0]        # post-scaling weights w
    intercept  = float(fitted_lr.intercept_[0])
    # scores_[1]: shape (n_folds, n_Cs) — roc_auc for class=1 per fold per C
    cv_scores  = fitted_lr.scores_[1]
    mean_cv    = cv_scores.mean(axis=0)
    std_cv     = cv_scores.std(axis=0)

    # ── Scaler statistics (for traceability)
    scaler     = pipe.named_steps["scaler"]
    scaler_mean  = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()

    # ── In-sample metrics — pre-computed once, used in log + results dict
    # Note: these are TRAINING metrics (informational / overfitting detection)
    # True performance → evaluate_meta_on_test.py
    y_prob   = pipe.predict_proba(X)[:, 1]
    y_pred   = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sens     = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec     = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc_val  = float(roc_auc_score(y, y_prob))
    f1_mac   = float(f1_score(y, y_pred, average="macro"))
    f1_act   = float(f1_score(y, y_pred, pos_label=1, average="binary"))
    acc_val  = float(accuracy_score(y, y_pred))
    bal_acc  = float(balanced_accuracy_score(y, y_pred))
    # Brier Score = (1/N) Σ (p̂ᵢ − yᵢ)²  — in-sample baseline for TS comparison
    brier    = float(brier_score_loss(y, y_prob))

    # ── Log
    log(logf, f"Done in {elapsed:.2f}s  |  best C={best_C}  λ=1/C={1/best_C:.4f}")
    log(logf, "C Grid CV AUC (mean ± std):  ← C selection only, NOT performance eval")
    for c, m, s in zip(C_GRID, mean_cv, std_cv):
        mark = " ← BEST" if c == best_C else ""
        log(logf, f"  C={c:<8}  {m:.4f} ± {s:.4f}{mark}")
    log(logf, "Modality weights w (post-scaling, sorted by |w|):")
    for fname, w in sorted(zip(feat_cols, coef), key=lambda x: -abs(x[1])):
        log(logf, f"  {fname:<20} w={w:+.6f}  |w|={abs(w):.6f}")
    log(logf, f"Intercept b = {intercept:+.6f}")
    log(logf, f"Decision: P(active) = σ(w·z + b)  "
              f"where z = StandardScaler(features)")
    log(logf, "In-sample metrics (training data — informational only):")
    log(logf, f"  AUC-ROC={auc_val:.4f}  AUPRC=N/A(train)  "
              f"Brier={brier:.4f}  BalAcc={bal_acc:.4f}")
    log(logf, f"  F1-macro={f1_mac:.4f}  F1-active={f1_act:.4f}  Acc={acc_val:.4f}")
    log(logf, f"  Sens={sens:.4f}  Spec={spec:.4f}")
    log(logf, f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    log(logf, "  [!] True performance → evaluate_meta_on_test.py (independent test set)")

    return pipe, {
        # ── Model parameters
        "best_C":              best_C,
        "lambda":              1.0 / best_C,
        "C_grid":              C_GRID,
        "cv_folds":            CV_FOLDS,
        "cv_scoring":          CV_SCORING,
        "solver":              "lbfgs",
        "class_weight":        "balanced",
        "feature_cols":        feat_cols,
        "weights":             {f: float(w) for f, w in zip(feat_cols, coef)},
        "intercept":           intercept,
        # ── Scaler (for traceability)
        "scaler_mean":         {f: float(m) for f, m in zip(feat_cols, scaler_mean)},
        "scaler_scale":        {f: float(s) for f, s in zip(feat_cols, scaler_scale)},
        # ── CV AUC per C (C selection record)
        "cv_mean_auc_per_C":   {str(c): float(m) for c, m in zip(C_GRID, mean_cv)},
        "cv_std_auc_per_C":    {str(c): float(s) for c, s in zip(C_GRID, std_cv)},
        "cv_scores_all_folds": {str(c): cv_scores[:, i].tolist()
                                for i, c in enumerate(C_GRID)},
        # ── In-sample metrics (training data, informational only)
        "insample_note":       "Training data only — overfitting reference, NOT for reporting",
        "insample_auc":        auc_val,
        "insample_f1_macro":   f1_mac,
        "insample_f1_active":  f1_act,
        "insample_accuracy":   acc_val,
        "insample_bal_acc":    bal_acc,
        "insample_brier":      brier,
        "insample_sensitivity": sens,
        "insample_specificity": spec,
        "insample_confusion":  {"TP": int(tp), "FP": int(fp),
                                "FN": int(fn), "TN": int(tn)},
        "training_time_sec":   elapsed,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_validation_curve(results: dict, out_dir: str) -> str:
    """
    Validation Curve: log(C) vs mean_AUC ± std.
    Visualises the C selection process from LogisticRegressionCV.
    """
    cs       = C_GRID
    means    = [results["cv_mean_auc_per_C"][str(c)] for c in cs]
    stds     = [results["cv_std_auc_per_C"][str(c)]  for c in cs]
    best_C   = results["best_C"]
    log_cs   = [np.log10(c) for c in cs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log_cs, means, "o-", color="#2563EB", linewidth=2,
            markersize=8, label="Mean CV AUC")
    ax.fill_between(log_cs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="#2563EB", label="±1 std")

    # Mark best C
    best_idx = cs.index(best_C)
    ax.axvline(x=log_cs[best_idx], color="#DC2626", linestyle="--",
               linewidth=1.5, label=f"Best C={best_C}  (λ={1/best_C:.4f})")
    ax.scatter([log_cs[best_idx]], [means[best_idx]],
               color="#DC2626", s=120, zorder=5)

    ax.set_xlabel("log₁₀(C)", fontsize=12)
    ax.set_ylabel("CV AUC (roc_auc)", fontsize=12)
    ax.set_title("Validation Curve — C Selection\n"
                 "(internal 5-fold CV, C selection only — not performance evaluation)",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(log_cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([max(0, min(means) - max(stds) - 0.05),
                 min(1, max(means) + max(stds) + 0.05)])

    plt.tight_layout()
    path = os.path.join(out_dir, "validation_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_modality_importance(results: dict, out_dir: str) -> str:
    """
    Modality Importance Bar Chart: |w| for each modality.
    Post-scaling weights are directly comparable across modalities.
    """
    weights = results["weights"]
    feat_cols = results["feature_cols"]
    w_vals   = [weights[f] for f in feat_cols]
    abs_vals = [abs(w) for w in w_vals]

    labels = [f.replace("_feat", "").upper() for f in feat_cols]
    colors = ["#2563EB", "#7C3AED", "#D97706"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, abs_vals, color=colors, width=0.5, edgecolor="white")

    # Annotate with raw w values
    for bar, w in zip(bars, w_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"w={w:+.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Modality", fontsize=12)
    ax.set_ylabel("|w|  (post-scaling)", fontsize=12)
    ax.set_title("Modality Importance (Stacking LR Weights)\n"
                 "Post-scaling |w| directly comparable across modalities",
                 fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(abs_vals) * 1.25])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "modality_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Save Artefacts ────────────────────────────────────────────────────────────
def save_artefacts(
    pipe: Pipeline,
    meta_df: pd.DataFrame,
    feat_cols: List[str],
    results: dict,
    run_cfg: dict,
    out_dir: str,
    logf,
) -> None:
    ensure_dir(out_dir)

    n  = len(meta_df)
    na = int((meta_df["y_true"] == 1).sum())
    ni = int((meta_df["y_true"] == 0).sum())
    cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                               y=meta_df["y_true"].to_numpy())
    alpha_i, alpha_a = float(cw[0]), float(cw[1])

    # ── 1. Model pipeline (StandardScaler + fitted LR)
    joblib.dump(pipe, os.path.join(out_dir, "meta_lr_model.pkl"))

    # ── 2. Config JSON (hyperparameters, paths, alpha — separated from results)
    config_obj = {
        **run_cfg,
        "alpha_active":   alpha_a,
        "alpha_inactive": alpha_i,
        "alpha_formula":  "α_j = N_total / (n_classes × N_j)",
        "alpha_ref":      "sklearn.utils.class_weight.compute_class_weight",
        "pipeline_ref":   "sklearn.pipeline.Pipeline(StandardScaler, LogisticRegressionCV)",
    }
    save_json(os.path.join(out_dir, "meta_lr_config.json"), config_obj)

    # ── 3. Results JSON (training outputs — separated from config)
    results_obj = {
        k: v for k, v in results.items()
        if k not in ("C_grid", "cv_folds", "cv_scoring", "solver",
                     "class_weight", "feature_cols")
    }
    results_obj["meta_dataset_source"] = os.path.join(
        run_cfg["meta_dataset_dir"], "meta_train_oof.csv")
    save_json(os.path.join(out_dir, "meta_lr_results.json"), results_obj)

    # ── 4. CV results CSV — per-C per-fold AUC + mean/std summary rows
    with open(os.path.join(out_dir, "meta_lr_cv_results.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "C", "fold", "auc"])
        for c_str, scores in results["cv_scores_all_folds"].items():
            for k, s in enumerate(scores, 1):
                w.writerow(["per_fold", c_str, k, f"{s:.6f}"])
        # Summary rows (mean ± std per C)
        for c_str in results["cv_scores_all_folds"]:
            m = results["cv_mean_auc_per_C"][c_str]
            s = results["cv_std_auc_per_C"][c_str]
            best_mark = "BEST" if float(c_str) == results["best_C"] else ""
            w.writerow(["summary", c_str, best_mark,
                        f"{m:.6f}±{s:.6f}"])

    # ── 5. Coef CSV — feature index + name + weight + |w| + intercept
    with open(os.path.join(out_dir, "meta_lr_coef.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_index", "feature", "weight_w", "abs_weight",
                    "scaler_mean", "scaler_scale"])
        for i, fname in enumerate(feat_cols):
            wv = results["weights"][fname]
            sm = results["scaler_mean"][fname]
            ss = results["scaler_scale"][fname]
            w.writerow([i, fname, f"{wv:.6f}", f"{abs(wv):.6f}",
                        f"{sm:.6f}", f"{ss:.6f}"])
        w.writerow(["intercept", "intercept_b",
                    f"{results['intercept']:.6f}", "", "", ""])

    # ── 6. Plots
    vc_path = plot_validation_curve(results, out_dir)
    mi_path = plot_modality_importance(results, out_dir)

    # ── 7. Human-readable training report
    with open(os.path.join(out_dir, "meta_lr_training_report.txt"),
              "w", encoding="utf-8") as f:
        sep = "=" * 70
        f.write(f"{sep}\nSTACKING META-LEARNER TRAINING REPORT\n{sep}\n\n")

        f.write("[RUN CONFIG]\n")
        for k, v in run_cfg.items():
            f.write(f"  {k:<32}: {v}\n")

        f.write("\n[META-TRAIN DATASET]\n")
        f.write(f"  source   : {run_cfg['meta_dataset_dir']}/meta_train_oof.csv\n")
        f.write(f"  total    : {n}\n")
        f.write(f"  active   : {na} ({na/n*100:.1f}%)\n")
        f.write(f"  inactive : {ni} ({ni/n*100:.1f}%)\n")
        f.write(f"  ratio    : {ni/na:.2f}:1\n")
        f.write(f"  features : {feat_cols}\n")

        f.write("\n[ALPHA  (sklearn compute_class_weight, balanced)]\n")
        f.write(f"  formula  : α_j = N_total / (n_classes × N_j)\n")
        f.write(f"  α_active   = {n}/(2×{na}) = {alpha_a:.4f}\n")
        f.write(f"  α_inactive = {n}/(2×{ni}) = {alpha_i:.4f}\n")
        f.write(f"  ratio      = {alpha_a/alpha_i:.2f}x  "
                f"(active upweighted to compensate {ni/na:.1f}:1 imbalance)\n")

        f.write("\n[C SELECTION  (LogisticRegressionCV 5-fold — C only, NOT evaluation)]\n")
        f.write(f"  {'C':<10}  {'mean_AUC':>10}  {'std_AUC':>8}  per-fold AUC\n")
        f.write(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*38}\n")
        for c_str, fold_scores in results["cv_scores_all_folds"].items():
            m   = results["cv_mean_auc_per_C"][c_str]
            s   = results["cv_std_auc_per_C"][c_str]
            mark  = "  ← BEST" if float(c_str) == results["best_C"] else ""
            folds = "  ".join(f"{x:.4f}" for x in fold_scores)
            f.write(f"  {c_str:<10}  {m:>10.4f}  {s:>8.4f}  [{folds}]{mark}\n")
        f.write(f"\n  Best C = {results['best_C']}  "
                f"(λ = 1/C = {results['lambda']:.4f})\n")

        f.write("\n[MODALITY WEIGHTS w  (lbfgs, post-StandardScaling)]\n")
        f.write("  Loss: L = -(1/N)Σαᵢ[y log p̂+(1-y)log(1-p̂)] + (λ/2)||w||²\n")
        f.write("  P(active) = σ(w₁·z_oct0 + w₂·z_oct1 + w₃·z_octa3 + b)\n")
        f.write("  where z = StandardScaler(feature)\n\n")
        f.write(f"  {'Feature':<20}  {'w':>10}  {'|w|':>10}  "
                f"{'scaler_mean':>12}  {'scaler_scale':>12}\n")
        f.write(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}\n")
        for fname in feat_cols:
            wv = results["weights"][fname]
            sm = results["scaler_mean"][fname]
            ss = results["scaler_scale"][fname]
            f.write(f"  {fname:<20}  {wv:>+10.6f}  {abs(wv):>10.6f}  "
                    f"{sm:>12.6f}  {ss:>12.6f}\n")
        f.write(f"  {'intercept b':<20}  {results['intercept']:>+10.6f}\n")

        f.write("\n[IN-SAMPLE METRICS  (training data — informational only)]\n")
        f.write("  Purpose: overfitting detection (compare with test set metrics)\n")
        f.write("  *** TRUE performance → evaluate_meta_on_test.py ***\n\n")
        f.write(f"  AUC-ROC        : {results['insample_auc']:.4f}\n")
        f.write(f"  Brier Score    : {results['insample_brier']:.4f}  "
                f"(baseline for TS calibration comparison)\n")
        f.write(f"  Balanced Acc   : {results['insample_bal_acc']:.4f}  "
                f"= (Sens+Spec)/2\n")
        f.write(f"  F1-macro       : {results['insample_f1_macro']:.4f}\n")
        f.write(f"  F1-active      : {results['insample_f1_active']:.4f}\n")
        f.write(f"  Accuracy       : {results['insample_accuracy']:.4f}\n")
        f.write(f"  Sensitivity    : {results['insample_sensitivity']:.4f}\n")
        f.write(f"  Specificity    : {results['insample_specificity']:.4f}\n")
        cm = results["insample_confusion"]
        f.write(f"  Confusion      : TP={cm['TP']}  FP={cm['FP']}  "
                f"FN={cm['FN']}  TN={cm['TN']}\n")

        f.write("\n[OUTPUT FILES]\n")
        for fname in ["meta_lr_model.pkl", "meta_lr_config.json",
                      "meta_lr_results.json", "meta_lr_cv_results.csv",
                      "meta_lr_coef.csv", "validation_curve.png",
                      "modality_importance.png"]:
            f.write(f"  {fname}\n")

    # ── Log
    log(logf, f"Artefacts → {out_dir}")
    for fname in ["meta_lr_model.pkl", "meta_lr_config.json",
                  "meta_lr_results.json", "meta_lr_cv_results.csv",
                  "meta_lr_coef.csv", "meta_lr_training_report.txt",
                  "validation_curve.png", "modality_importance.png"]:
        log(logf, f"  {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    model_name       = normalize_model_name(MODEL_NAME)
    meta_dataset_dir = os.path.abspath(META_DATASET_DIR)

    if not os.path.isfile(os.path.join(meta_dataset_dir, "meta_train_oof.csv")):
        raise FileNotFoundError(
            f"meta_train_oof.csv not found in: {meta_dataset_dir}\n"
            "Check META_DATASET_DIR in Config or run build_meta_dataset.py first."
        )

    # Derive run identifier from the input folder structure for traceability
    # Expected: .../<meta_tag>/<lr_tag>/  → extract last two path components
    parts    = meta_dataset_dir.rstrip(os.sep).split(os.sep)
    lr_tag   = parts[-1]   # e.g. OCT0_LR3e-06_OCT1_LR2e-06_OCTA3_LR2e-06
    strategy = parts[-2]   # e.g. Partial_B5
    meta_tag = parts[-3]   # e.g. swin_tiny__logit__calibTrue

    # Output path mirrors input structure
    out_dir = os.path.join(META_OUTPUT_ROOT, meta_tag, strategy, lr_tag)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "meta_training.log"),
                "a", buffering=1, encoding="utf-8")

    log(logf, "=" * 60)
    log(logf, "STACKING META-LEARNER TRAINING  (training only)")
    log(logf, "=" * 60)
    log(logf, f"model          : {model_name} ({get_backbone_name(model_name)})")
    log(logf, f"strategy        : {strategy}")
    log(logf, f"meta_dataset_dir: {meta_dataset_dir}")
    log(logf, f"meta_tag       : {meta_tag}")
    log(logf, f"lr_tag         : {lr_tag}")
    log(logf, f"feat_cols      : {FEAT_COLS}")
    log(logf, f"C_GRID         : {C_GRID}   cv_folds={CV_FOLDS}")
    log(logf, f"solver         : lbfgs   class_weight=balanced   max_iter={MAX_ITER}")
    log(logf, f"out_dir        : {out_dir}")

    # ── Step 1: Load meta-train dataset
    log(logf, "─" * 60)
    log(logf, "Step 1: Load meta-train dataset (from build_meta_dataset.py)")
    meta_df, feat_cols = load_meta_train(meta_dataset_dir, logf)
    X = meta_df[feat_cols].to_numpy(dtype=np.float32)
    y = meta_df["y_true"].to_numpy(dtype=np.int32)

    # ── Step 2: Train Pipeline
    log(logf, "─" * 60)
    log(logf, "Step 2: Train Pipeline(StandardScaler, LogisticRegressionCV)")
    pipe, results = train_meta_lr(X, y, feat_cols, logf)

    # ── Step 3: Save artefacts
    log(logf, "─" * 60)
    log(logf, "Step 3: Save artefacts")
    run_cfg = {
        "model_name":        model_name,
        "backbone":          get_backbone_name(model_name),
        "meta_tag":          meta_tag,
        "lr_tag":            lr_tag,
        "feature_cols":      FEAT_COLS,
        "meta_dataset_dir":  meta_dataset_dir,
        "C_grid":            C_GRID,
        "cv_folds":          CV_FOLDS,
        "cv_scoring":        CV_SCORING,
        "solver":            "lbfgs",
        "class_weight":      "balanced",
        "max_iter":          MAX_ITER,
        "random_seed":       RANDOM_SEED,
        "out_dir":           out_dir,
        "note": (
            "alpha: compute_class_weight(balanced), α_j=N/(n_classes×N_j); "
            "C: LogisticRegressionCV 5-fold roc_auc grid search (C selection only); "
            "w: lbfgs solver; "
            "true performance evaluation → evaluate_meta_on_test.py"
        ),
    }
    save_artefacts(pipe, meta_df, feat_cols, results, run_cfg, out_dir, logf)

    log(logf, "=" * 60)
    log(logf, "TRAINING COMPLETE")
    log(logf, f"  Best C         : {results['best_C']}  (λ={results['lambda']:.4f})")
    log(logf, f"  Weights        : " + "  ".join(
        f"{k}={v:+.4f}" for k, v in results["weights"].items()))
    log(logf, f"  Intercept b    : {results['intercept']:+.4f}")
    log(logf, f"  In-sample AUC  : {results['insample_auc']:.4f}  (training data only)")
    log(logf, f"  In-sample Brier: {results['insample_brier']:.4f}  "
              f"(baseline for TS comparison)")
    log(logf, f"  model saved    : {os.path.join(out_dir, 'meta_lr_model.pkl')}")
    log(logf, "  → evaluate_meta_on_test.py for test-set evaluation")
    log(logf, "=" * 60)

    logf.close()
    gc.collect()


if __name__ == "__main__":
    main()