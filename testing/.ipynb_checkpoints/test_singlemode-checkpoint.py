# test_singlemode.py
"""
Base Model Independent Test Set Evaluation  ─  Single-Modality
"""

# ── Standard library ──────────────────────────────────────────────────────────
import gc
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
import seaborn as sns

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# ★ CONFIG  —  Edit only these entries
# ══════════════════════════════════════════════════════════════════════════════

# ★ 1. Point to the Best_fold{N} directory produced by train_singlemode_oof.py.
#      Structure must be:
#        <PROJECT_ROOT>/outputs/training/<model_name>/<modality>/<run_tag>/Best_fold{N}
INPUT_DIR = (
    "/data/Irene/SwinTransformer/Swin_Meta/outputs/training/"
    "swin_tiny/OCT0/"
    "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9/"
    "Best_fold5"
)

# ★ 2. master_manifest.csv — same file used by train_singlemode_oof.py.
#      Leave "" to auto-detect from PROJECT_ROOT (recommended).
#      Explicit example:
#        "/data/Irene/SwinTransformer/Swin_Meta/outputs/manifests/
#         master_split/master_manifest.csv"
MASTER_MANIFEST_CSV = ""

# ★ 3. CHECKPOINT_ROOT — where train_singlemode_oof.py saved model weights.
#      Must match CHECKPOINT_ROOT in the training script.
#      Leave "" to auto-detect as PROJECT_ROOT/checkpoints.
CHECKPOINT_ROOT = ""

# ── Output root (auto-created) ────────────────────────────────────────────────
# Leave "" to auto-detect as PROJECT_ROOT/outputs/test_evaluation
TEST_EVAL_ROOT = ""

# ── Evaluation settings (no need to change) ───────────────────────────────────
THRESHOLD   = 0.5   # binary decision threshold
ECE_N_BINS  = 10    # equal-width bins for ECE (Guo et al. 2017)
BATCH_SIZE  = 16
NUM_WORKERS = 4
IMG_SIZE    = 224
RANDOM_SEED = 42

CLASS_NAMES = ["inactive", "active"]

# Backbone registry: model_name (from path) → timm model string
# Add entries here when supporting VGG16 / EfficientNet
TIMM_MODEL_MAP: Dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

# Image path column in master_manifest.csv per modality
MODALITY_IMG_COL: Dict[str, str] = {
    "OCT0":  "oct0_image_path",
    "OCT1":  "oct1_image_path",
    "OCTA3": "octa3_image_path",
}

# has_* column name per modality
MODALITY_HAS_COL: Dict[str, str] = {
    "OCT0":  "has_oct0",
    "OCT1":  "has_oct1",
    "OCTA3": "has_octa3",
}

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def log(logf, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if logf:
        logf.write(line + "\n")
        logf.flush()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# Parse INPUT_DIR → derive all path components
# ─────────────────────────────────────────────────────────────────────────────

def parse_input_dir(input_dir: str) -> dict:
    """
    Parse model_name, modality, run_tag, best_fold, project_root from INPUT_DIR.

    Required structure:
      <PROJECT_ROOT>/outputs/training/<model_name>/<modality>/<run_tag>/Best_fold{N}
         parents[5]    parents[4]  parents[3]   parents[2]  parents[0]  leaf

    Returns dict:
      model_name, modality, run_tag, best_fold (int),
      project_root (Path), input_dir (str)
    """
    p = Path(input_dir).resolve()

    if not p.is_dir():
        raise FileNotFoundError(
            f"INPUT_DIR not found: {input_dir}\n"
            "Check the path exists on this machine."
        )

    leaf = p.name  # e.g. "Best_fold5"
    if not leaf.startswith("Best_fold"):
        raise ValueError(
            f"INPUT_DIR must end with 'Best_fold{{N}}', got: '{leaf}'\n"
            f"Full path: {input_dir}"
        )
    try:
        best_fold = int(leaf.replace("Best_fold", ""))
    except ValueError:
        raise ValueError(
            f"Cannot parse fold number from '{leaf}'. "
            "Expected e.g. 'Best_fold5'."
        )

    # Use p.parents for unambiguous indexing:
    #   p            = .../Swin_Meta/outputs/training/swin_tiny/OCT0/<run_tag>/Best_fold{N}
    #   p.parents[0] = .../Swin_Meta/outputs/training/swin_tiny/OCT0/<run_tag>
    #   p.parents[1] = .../Swin_Meta/outputs/training/swin_tiny/OCT0
    #   p.parents[2] = .../Swin_Meta/outputs/training/swin_tiny
    #   p.parents[3] = .../Swin_Meta/outputs/training
    #   p.parents[4] = .../Swin_Meta/outputs
    #   p.parents[5] = .../Swin_Meta                  ← PROJECT_ROOT
    run_tag      = p.parents[0].name   # e.g. "BS16_EP100_..."
    modality     = p.parents[1].name   # e.g. "OCT0"
    model_name   = p.parents[2].name   # e.g. "swin_tiny"
    project_root = p.parents[5]        # e.g. /data/Irene/SwinTransformer/Swin_Meta

    if modality not in MODALITY_IMG_COL:
        raise ValueError(
            f"Parsed modality '{modality}' not in {list(MODALITY_IMG_COL.keys())}.\n"
            "Verify INPUT_DIR follows the required structure:\n"
            "  <PROJECT_ROOT>/outputs/training/<model_name>/<modality>/<run_tag>/Best_fold{{N}}"
        )

    return {
        "input_dir":    str(p),
        "model_name":   model_name,
        "modality":     modality,
        "run_tag":      run_tag,
        "best_fold":    best_fold,
        "project_root": project_root,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resolve optional paths (auto-detect when left "")
# ─────────────────────────────────────────────────────────────────────────────

def resolve_manifest_csv(cfg_value: str, project_root: Path) -> str:
    """
    Resolve master_manifest.csv path.

    Auto-detect order (mirrors MASTER_MANIFEST_CSV in training script):
      1. PROJECT_ROOT/outputs/manifests/master_split/master_manifest.csv  ← primary
      2. PROJECT_ROOT/outputs/manifests/master_manifest.csv
      3. PROJECT_ROOT/data_splits/master_manifest.csv
    """
    if cfg_value.strip():
        if os.path.isfile(cfg_value):
            return cfg_value
        raise FileNotFoundError(
            f"MASTER_MANIFEST_CSV not found:\n  {cfg_value}\n"
            "Set MASTER_MANIFEST_CSV = \"\" to auto-detect."
        )

    candidates = [
        project_root / "outputs" / "manifests" / "master_split" / "master_manifest.csv",
        project_root / "outputs" / "manifests" / "master_manifest.csv",
        project_root / "data_splits" / "master_manifest.csv",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    tried = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(candidates))
    raise FileNotFoundError(
        "master_manifest.csv could not be auto-detected. Tried:\n"
        f"{tried}\n\n"
        "Fix: set MASTER_MANIFEST_CSV to the absolute path, e.g.:\n"
        f"  MASTER_MANIFEST_CSV = \"{candidates[0]}\"\n\n"
        "Or run on your server to locate the file:\n"
        f"  find {project_root} -name 'master_manifest.csv'"
    )


def resolve_checkpoint_root(cfg_value: str, project_root: Path) -> str:
    """Auto-detect CHECKPOINT_ROOT as PROJECT_ROOT/checkpoints if not set."""
    if cfg_value.strip():
        return cfg_value
    return str(project_root / "checkpoints")


def resolve_test_eval_root(cfg_value: str, project_root: Path) -> str:
    """Auto-detect TEST_EVAL_ROOT as PROJECT_ROOT/outputs/test_evaluation if not set."""
    if cfg_value.strip():
        return cfg_value
    return str(project_root / "outputs" / "test_evaluation")


def resolve_checkpoint_path(
    checkpoint_root: str,
    model_name: str,
    modality: str,
    run_tag: str,
    best_fold: int,
) -> str:
    """
    Resolve model_best.pth from CHECKPOINT_ROOT.

    Train script saves to:
      Primary : CHECKPOINT_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/model_best.pth
      Fallback: CHECKPOINT_ROOT/<model>/<modality>/<run_tag>/Kfold/fold{N}/model_best.pth
    """
    base     = os.path.join(checkpoint_root, model_name, modality, run_tag)
    primary  = os.path.join(base, f"Best_fold{best_fold}", "model_best.pth")
    fallback = os.path.join(base, "Kfold", f"fold{best_fold}", "model_best.pth")

    if os.path.isfile(primary):
        return primary
    if os.path.isfile(fallback):
        return fallback

    raise FileNotFoundError(
        f"model_best.pth not found for fold {best_fold}.\n"
        f"  Primary  : {primary}\n"
        f"  Fallback : {fallback}\n"
        f"  CHECKPOINT_ROOT = {checkpoint_root}\n\n"
        "If CHECKPOINT_ROOT is wrong, set it explicitly in CONFIG."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load fold_summary.json
# ─────────────────────────────────────────────────────────────────────────────

def load_fold_summary(input_dir: str, parsed: dict) -> Tuple[float, dict]:
    """
    Read fold_summary.json from INPUT_DIR (Best_fold{N}/).

    Keys from train_singlemode_oof.py → save_json(fold_summary):
      "fold"          int   ← fold number (cross-check with path)
      "model_name"    str   ← cross-check with path
      "modality"      str   ← cross-check with path
      "temperature"   float ← T* fitted on validation set
      "nll_before_ts" float|None
      "nll_after_ts"  float|None
      "best_val_bce"  float
      "best_val_auc"  float
      "best_epoch"    int

    NOTE: "best_fold" key does NOT exist in fold_summary.json.
    The fold number is stored under "fold".
    """
    summary_path = os.path.join(input_dir, "fold_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"fold_summary.json not found: {summary_path}\n"
            "Ensure train_singlemode_oof.py completed successfully for this fold."
        )
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    # Cross-check path vs JSON metadata (catches copy-paste mistakes)
    errs = []
    if int(summary["fold"]) != parsed["best_fold"]:
        errs.append(
            f"  fold: path=Best_fold{parsed['best_fold']}, "
            f"JSON fold_summary['fold']={summary['fold']}"
        )
    if str(summary["modality"]) != parsed["modality"]:
        errs.append(
            f"  modality: path='{parsed['modality']}', "
            f"JSON='{summary['modality']}'"
        )
    if str(summary["model_name"]) != parsed["model_name"]:
        errs.append(
            f"  model_name: path='{parsed['model_name']}', "
            f"JSON='{summary['model_name']}'"
        )
    if errs:
        raise ValueError(
            "Mismatch between INPUT_DIR path and fold_summary.json:\n"
            + "\n".join(errs)
        )

    temperature = float(summary["temperature"])
    return temperature, summary


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (mirrors load_master_manifest in train_singlemode_oof.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_manifest_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply same dtype normalization as train_singlemode_oof.py
    → normalize_manifest_dtypes().
    Ensures numeric comparisons (== 0, == 1, notna) work correctly.
    """
    out = df.copy()
    for col in ["exam_key", "patient_id", "eye", "exam_date", "split_set"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "y_true" in out.columns:
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    for col in [
        "has_oct0", "has_oct1", "has_octa3",
        "has_oct_pair", "is_complete_three_path",
        "label_conflict", "fold_id",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_test_dataframe(
    manifest_csv: str,
    modality: str,
    logf,
) -> pd.DataFrame:
    """
    Load master_manifest.csv and return the test rows for the given modality.

    Applies IDENTICAL filters to train_singlemode_oof.py → load_master_manifest():
      1. label_conflict == 0
      2. y_true not null
      3. has_{modality} == 1
      4. {modality}_image_path not null / not empty-string
      5. split_set == "test"

    Also validates required columns are present.
    """
    required_cols = [
        "exam_key", "patient_id", "split_set", "y_true",
        "label_conflict",
        "has_oct0", "has_oct1", "has_octa3",
        "oct0_image_path", "oct1_image_path", "octa3_image_path",
    ]

    df = pd.read_csv(manifest_csv)
    df = normalize_manifest_dtypes(df)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"master_manifest.csv is missing required columns: {missing}\n"
            f"File: {manifest_csv}"
        )

    n_raw = len(df)

    # ── Filter 1: remove label conflicts (same as training) ──────────────
    df = df[df["label_conflict"] == 0].copy()
    log(logf, f"  After label_conflict==0 filter : {len(df)} / {n_raw} rows")

    # ── Filter 2: require known label ────────────────────────────────────
    df = df[df["y_true"].notna()].copy()
    log(logf, f"  After y_true notna filter      : {len(df)} rows")

    # ── Filter 3: require modality image exists ───────────────────────────
    has_col = MODALITY_HAS_COL[modality]
    img_col = MODALITY_IMG_COL[modality]

    df = df[df[has_col] == 1].copy()
    log(logf, f"  After {has_col}==1 filter       : {len(df)} rows")

    # ── Filter 4: require non-null, non-empty image path ─────────────────
    df = df[df[img_col].notna()].copy()
    df = df[df[img_col].astype(str).str.strip() != ""].copy()
    log(logf, f"  After {img_col} not-null filter : {len(df)} rows")

    # ── Filter 5: test split only ─────────────────────────────────────────
    df_all_splits = df.copy()
    df = df[df["split_set"] == "test"].copy()

    # Report split composition for verification
    for split in ["train_valid", "test"]:
        n = int((df_all_splits["split_set"] == split).sum())
        log(logf, f"  split_set='{split}' available  : {n} rows  "
                  f"(using 'test' only)")

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No test rows found after filtering for modality={modality}.\n"
            "Check that master_manifest.csv contains split_set=='test' rows "
            f"with {has_col}==1."
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str) -> nn.Module:
    """
    Build model with num_classes=1 (single sigmoid head, output [B,1]).
    Matches NUM_CLASSES=1 in train_singlemode_oof.py + FocalBCELoss([B,1]).
    """
    if model_name not in TIMM_MODEL_MAP:
        raise NotImplementedError(
            f"model_name='{model_name}' not in TIMM_MODEL_MAP.\n"
            f"Add entry to TIMM_MODEL_MAP. Registered: {list(TIMM_MODEL_MAP.keys())}"
        )
    return timm.create_model(
        TIMM_MODEL_MAP[model_name],
        pretrained=False,   # weights loaded from checkpoint
        num_classes=1,
    )


def load_checkpoint(
    model: nn.Module, ckpt_path: str, device: torch.device,
) -> nn.Module:
    """
    Load model_best.pth saved by train_singlemode_oof.py.

    Train checkpoint wrapper (after TS overwrite):
      {"model_state_dict": state_dict, "epoch", "best_epoch",
       "temperature", "val_nll_uncal", "val_acc", "val_auc",
       "nll_beforeTS", "nll_afterTS", "fold", "model_name", "modality"}
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw   # bare state dict
    else:
        state_dict = raw

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SingleModalityDataset(Dataset):
    """Test-set dataset. Returns (img_tensor, label, exam_key)."""

    def __init__(
        self, df: pd.DataFrame, img_col: str, transform,
    ) -> None:
        self.records   = df[["exam_key", "y_true", img_col]].copy()
        self.img_col   = img_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row      = self.records.iloc[idx]
        label    = int(row["y_true"])
        exam_key = str(row["exam_key"])
        img_path = str(row[self.img_col])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return black image rather than crashing; logged via logit range check
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label, exam_key


def get_test_transform() -> transforms.Compose:
    """
    Standard ImageNet eval transform — no augmentation.
    Mirrors tf_val in train_singlemode_oof.py.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    logf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (exam_keys, y_true, logits_uncal).

    Model output: [B, 1]  →  squeeze(1)  →  scalar logit per sample [B].
    DO NOT use out[:, 1] — that assumes a two-class [B,2] head and raises
    IndexError on the [B,1] single-sigmoid output used here.
    """
    all_keys, all_labels, all_logits = [], [], []
    model.eval()
    n_batches = len(loader)

    for i, (imgs, labels, keys) in enumerate(loader):
        imgs        = imgs.to(device)
        out         = model(imgs)                       # [B, 1]
        logits_flat = out.squeeze(1).cpu().numpy()      # [B]

        all_keys.extend(keys)
        all_labels.extend(labels.numpy())
        all_logits.append(logits_flat)

        if (i + 1) % max(1, n_batches // 5) == 0:
            log(logf, f"  Inference: {i+1}/{n_batches} batches done")

    return (
        np.array(all_keys),
        np.array(all_labels, dtype=np.int32),
        np.concatenate(all_logits).astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

def apply_temperature_scaling(
    logits: np.ndarray, temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    logit_calib = logit / T*  →  prob_calib = sigmoid(logit_calib)
    T* loaded from fold_summary.json (fitted on val set only; never test).
    Ref: Guo et al. (2017). ICML.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    lc = (logits / temperature).astype(np.float32)
    return lc, sigmoid(lc).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """ECE = Σ_b (|B_b|/N)|acc(B_b)−conf(B_b)|  (equal-width bins, Guo 2017)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _cls_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_prob: np.ndarray, prefix: str,
) -> dict:
    """
    Complete metric block: AUROC, AUPRC, Balanced Accuracy, Sensitivity,
    Specificity, PPV, NPV, F1-active, F1-macro, Accuracy, Brier, TP/FP/FN/TN.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        f"{prefix}_auroc":        float(roc_auc_score(y_true, y_prob)),
        f"{prefix}_auprc":        float(average_precision_score(y_true, y_prob)),
        f"{prefix}_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_sensitivity":  float(sens),
        f"{prefix}_specificity":  float(spec),
        f"{prefix}_ppv":          float(ppv),
        f"{prefix}_npv":          float(npv),
        f"{prefix}_f1_active":    float(f1_score(y_true, y_pred,
                                                  pos_label=1, average="binary")),
        f"{prefix}_f1_macro":     float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_accuracy":     float(accuracy_score(y_true, y_pred)),
        f"{prefix}_brier":        float(brier_score_loss(y_true, y_prob)),
        f"{prefix}_TP": int(tp), f"{prefix}_FP": int(fp),
        f"{prefix}_FN": int(fn), f"{prefix}_TN": int(tn),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    logits_uncal: np.ndarray,
    temperature: float,
    threshold: float = 0.5,
    ece_n_bins: int  = 10,
) -> Tuple[dict, List[dict], List[dict]]:
    prob_uncal           = sigmoid(logits_uncal)
    logits_cal, prob_cal = apply_temperature_scaling(logits_uncal, temperature)

    y_pred_uncal = (prob_uncal >= threshold).astype(int)
    y_pred_cal   = (prob_cal   >= threshold).astype(int)

    m_uncal = _cls_metrics(y_true, y_pred_uncal, prob_uncal, "uncal")
    m_calib = _cls_metrics(y_true, y_pred_cal,   prob_cal,   "calib")

    ece_uncal, bins_uncal = compute_ece(y_true, prob_uncal, ece_n_bins)
    ece_calib, bins_calib = compute_ece(y_true, prob_cal,   ece_n_bins)

    n_total    = len(y_true)
    n_active   = int((y_true == 1).sum())
    n_inactive = int((y_true == 0).sum())

    return {
        "n_total":         n_total,
        "n_active":        n_active,
        "n_inactive":      n_inactive,
        "imbalance_ratio": round(n_inactive / n_active, 2) if n_active > 0 else None,
        "threshold":       threshold,
        "temperature":     temperature,
        "ece_n_bins":      ece_n_bins,
        "ece_uncal":       ece_uncal,
        "ece_calib":       ece_calib,
        "ece_reduction":   ece_uncal - ece_calib,
        **m_uncal,
        **m_calib,
    }, bins_uncal, bins_calib


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _set_style() -> None:
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    out_path: str, title: str,
) -> None:
    _set_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = LinearSegmentedColormap.from_list("cm_cmap", ["#f0f4ff", "#2563EB"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="white", ax=ax, cbar=True)
    ax.set_xlabel("Predicted label", fontweight="bold")
    ax.set_ylabel("True label",      fontweight="bold")
    ax.set_title(title,              fontweight="bold")
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ax.text(1.05, 0.5,
            f"Sens = {sens:.3f}\nSpec = {spec:.3f}\nNPV  = {npv:.3f}",
            transform=ax.transAxes, va="center", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc",
                      edgecolor="#cbd5e1"))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    prob_uncal: np.ndarray, prob_calib: np.ndarray,
    out_path: str, title: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _set_style()
    auc_u = roc_auc_score(y_true, prob_uncal)
    auc_c = roc_auc_score(y_true, prob_calib)
    fpr_u, tpr_u, _     = roc_curve(y_true, prob_uncal)
    fpr_c, tpr_c, thr_c = roc_curve(y_true, prob_calib)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr_u, tpr_u, color="#94a3b8", lw=1.5, linestyle="--",
            label=f"Uncalibrated  AUC={auc_u:.4f}")
    ax.plot(fpr_c, tpr_c, color="#2563EB", lw=2,
            label=f"Calibrated (Post-TS)  AUC={auc_c:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return fpr_c, tpr_c, thr_c


def plot_pr_curve(
    y_true: np.ndarray,
    prob_uncal: np.ndarray, prob_calib: np.ndarray,
    out_path: str, title: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _set_style()
    ap_u     = average_precision_score(y_true, prob_uncal)
    ap_c     = average_precision_score(y_true, prob_calib)
    baseline = float((y_true == 1).mean())
    pre_u, rec_u, _     = precision_recall_curve(y_true, prob_uncal)
    pre_c, rec_c, thr_c = precision_recall_curve(y_true, prob_calib)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec_u, pre_u, color="#94a3b8", lw=1.5, linestyle="--",
            label=f"Uncalibrated  AUPRC={ap_u:.4f}")
    ax.plot(rec_c, pre_c, color="#7C3AED", lw=2,
            label=f"Calibrated (Post-TS)  AUPRC={ap_c:.4f}")
    ax.axhline(y=baseline, color="#DC2626", lw=0.8, linestyle=":",
               label=f"Baseline prevalence={baseline:.3f}")
    ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax.set_ylabel("Precision (PPV)",      fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return pre_c, rec_c, thr_c


def plot_reliability_diagram(
    bins_uncal: List[dict], bins_calib: List[dict],
    ece_uncal: float, ece_calib: float,
    out_path: str, title: str,
) -> None:
    _set_style()

    def _extract(bins):
        xs, ys, ns = [], [], []
        for b in bins:
            if b["accuracy"] is not None:
                xs.append(b["confidence"])
                ys.append(b["accuracy"])
                ns.append(b["n"])
        return np.array(xs), np.array(ys), np.array(ns)

    xu, yu, nu = _extract(bins_uncal)
    xc, yc, nc = _extract(bins_calib)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    if len(xu) > 0:
        ax.scatter(xu, yu, s=nu / max(nu.max(), 1) * 300 + 30,
                   color="#94a3b8", alpha=0.7, zorder=3,
                   label=f"Uncalibrated  ECE={ece_uncal:.4f}")
        ax.plot(xu, yu, color="#94a3b8", lw=1.2, linestyle="--", alpha=0.6)
    if len(xc) > 0:
        ax.scatter(xc, yc, s=nc / max(nc.max(), 1) * 300 + 30,
                   color="#2563EB", alpha=0.85, zorder=4,
                   label=f"Calibrated (Post-TS)  ECE={ece_calib:.4f}")
        ax.plot(xc, yc, color="#2563EB", lw=1.5)
    ax.set_xlabel("Mean predicted confidence", fontweight="bold")
    ax.set_ylabel("Fraction of positives (accuracy)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.25)
    ax.text(0.98, 0.04,
            f"ECE: {ece_uncal:.4f} → {ece_calib:.4f}\n"
            f"(↓{ece_uncal - ece_calib:.4f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4",
                      edgecolor="#86efac"))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save output files
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of (display_label, calib_key, uncal_key) for test_metrics.csv
_METRICS_TABLE = [
    # ── Discriminative ────────────────────────────────────────────────────────
    ("AUROC",             "calib_auroc",        "uncal_auroc"),
    ("AUPRC",             "calib_auprc",        "uncal_auprc"),
    # ── Threshold-based ──────────────────────────────────────────────────────
    ("Balanced_Accuracy", "calib_balanced_acc", "uncal_balanced_acc"),
    ("Sensitivity",       "calib_sensitivity",  "uncal_sensitivity"),
    ("Specificity",       "calib_specificity",  "uncal_specificity"),
    ("PPV",               "calib_ppv",          "uncal_ppv"),
    ("NPV",               "calib_npv",          "uncal_npv"),
    ("F1_active",         "calib_f1_active",    "uncal_f1_active"),
    ("F1_macro",          "calib_f1_macro",     "uncal_f1_macro"),
    ("Accuracy",          "calib_accuracy",     "uncal_accuracy"),
    # ── Calibration ───────────────────────────────────────────────────────────
    ("ECE",               "ece_calib",          "ece_uncal"),
    ("Brier_Score",       "calib_brier",        "uncal_brier"),
    ("ECE_reduction",     "ece_reduction",      "ece_reduction"),  # TS improvement: positive = better
]


def save_metrics_csv(metrics: dict, out_path: str) -> None:
    """
    Tall-format CSV: one row per metric, columns = metric / calibrated / uncalibrated.

    Layout (23 rows × 3 cols):

      metric               | calibrated (post-TS) | uncalibrated
      ---------------------+----------------------+-------------
      AUROC                | 0.9100               | 0.9000
      AUPRC                | 0.7200               | 0.7100
      Balanced_Accuracy    | ...                  | ...
      Sensitivity          | ...                  | ...
      Specificity          | ...                  | ...
      PPV                  | ...                  | ...
      NPV                  | ...                  | ...
      F1_active            | ...                  | ...
      F1_macro             | ...                  | ...
      Accuracy             | ...                  | ...
      ECE                  | ...                  | ...
      Brier_Score          | ...                  | ...
      ── context (shared) ──
      temperature          | T*                   | T*
      threshold            | 0.5                  | 0.5
      TP                   | calib_TP             | uncal_TP
      FP                   | calib_FP             | uncal_FP
      FN                   | calib_FN             | uncal_FN
      TN                   | calib_TN             | uncal_TN
      n_total              | N                    | N
      n_active             | N_pos                | N_pos
      n_inactive           | N_neg                | N_neg
      imbalance_ratio      | ratio                | ratio

    Easy to read in Excel / Origin: metric names in column A,
    calibrated values in column B, uncalibrated in column C.
    """
    rows = []

    # ── 12 performance metrics ─────────────────────────────────────────────
    for label, ck, uk in _METRICS_TABLE:
        rows.append({
            "metric":                 label,
            "calibrated (post-TS)":   round(float(metrics[ck]), 6) if metrics[ck] is not None else None,
            "uncalibrated":           round(float(metrics[uk]), 6) if metrics[uk] is not None else None,
        })

    # ── shared context columns (same value in both columns) ────────────────
    for label, val in [
        ("temperature",     metrics["temperature"]),
        ("threshold",       metrics["threshold"]),
    ]:
        rows.append({"metric": label,
                     "calibrated (post-TS)": val,
                     "uncalibrated":         val})

    # ── confusion matrix counts (differ between calibrated / uncalibrated) ─
    for col in ("TP", "FP", "FN", "TN"):
        rows.append({
            "metric":               col,
            "calibrated (post-TS)": metrics[f"calib_{col}"],
            "uncalibrated":         metrics[f"uncal_{col}"],
        })

    # ── dataset statistics (shared) ────────────────────────────────────────
    for col in ("n_total", "n_active", "n_inactive", "imbalance_ratio"):
        rows.append({"metric": col,
                     "calibrated (post-TS)": metrics[col],
                     "uncalibrated":         metrics[col]})

    pd.DataFrame(rows).to_csv(
        out_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )


def save_test_summary_json(
    metrics: dict,
    run_cfg: dict,
    fold_summary: dict,
    out_path: str,
) -> None:
    """
    Compact JSON summary — one file captures everything needed for
    downstream reporting, paper tables, and pipeline bookkeeping.

    Structure:
      run_info         : model / modality / fold / checkpoint / paths
      training_val     : best val metrics from fold_summary (for reference)
      temperature_scaling : T*, NLL before/after, ECE reduction
      test_dataset     : n_total, n_active, n_inactive, imbalance_ratio
      calibrated       : all post-TS test metrics  ← primary results
      uncalibrated     : all pre-TS  test metrics  ← reference
      confusion_matrix : TP, FP, FN, TN (calibrated)
    """
    import datetime

    def _r(v, ndigits=4):
        return round(float(v), ndigits) if v is not None else None

    summary = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "run_info": {
            "INPUT_DIR":      run_cfg.get("INPUT_DIR"),
            "model_name":     run_cfg.get("model_name"),
            "timm_backbone":  run_cfg.get("timm_backbone"),
            "modality":       run_cfg.get("modality"),
            "run_tag":        run_cfg.get("run_tag"),
            "best_fold":      run_cfg.get("best_fold"),
            "best_epoch":     run_cfg.get("best_epoch"),
            "checkpoint":     run_cfg.get("checkpoint"),
            "manifest_csv":   run_cfg.get("manifest_csv"),
            "out_dir":        run_cfg.get("out_dir"),
            "device":         run_cfg.get("device"),
            "threshold":      run_cfg.get("threshold"),
            "ece_n_bins":     run_cfg.get("ece_n_bins"),
        },

        "training_val_reference": {
            "best_val_bce": _r(fold_summary.get("best_val_bce"), 6),
            "best_val_auc": _r(fold_summary.get("best_val_auc"), 4),
            "best_epoch":   fold_summary.get("best_epoch"),
            "val_exam_units": fold_summary.get("val_exam_units"),
        },

        "temperature_scaling": {
            "temperature_T*":  _r(metrics["temperature"], 6),
            "nll_before_ts":   _r(fold_summary.get("nll_before_ts"), 6),
            "nll_after_ts":    _r(fold_summary.get("nll_after_ts"), 6),
            "ece_uncal":       _r(metrics["ece_uncal"], 4),
            "ece_calib":       _r(metrics["ece_calib"], 4),
            "ece_reduction":   _r(metrics["ece_reduction"], 4),
            "ece_improved":    metrics["ece_reduction"] > 0,
        },

        "test_dataset": {
            "n_total":         metrics["n_total"],
            "n_active":        metrics["n_active"],
            "n_inactive":      metrics["n_inactive"],
            "imbalance_ratio": metrics["imbalance_ratio"],
            "prevalence":      _r(metrics["n_active"] / metrics["n_total"], 4),
        },

        "calibrated": {
            "AUROC":             _r(metrics["calib_auroc"]),
            "AUPRC":             _r(metrics["calib_auprc"]),
            "Balanced_Accuracy": _r(metrics["calib_balanced_acc"]),
            "Sensitivity":       _r(metrics["calib_sensitivity"]),
            "Specificity":       _r(metrics["calib_specificity"]),
            "PPV":               _r(metrics["calib_ppv"]),
            "NPV":               _r(metrics["calib_npv"]),
            "F1_active":         _r(metrics["calib_f1_active"]),
            "F1_macro":          _r(metrics["calib_f1_macro"]),
            "Accuracy":          _r(metrics["calib_accuracy"]),
            "ECE":               _r(metrics["ece_calib"]),
            "Brier_Score":       _r(metrics["calib_brier"]),
        },

        "uncalibrated": {
            "AUROC":             _r(metrics["uncal_auroc"]),
            "AUPRC":             _r(metrics["uncal_auprc"]),
            "Balanced_Accuracy": _r(metrics["uncal_balanced_acc"]),
            "Sensitivity":       _r(metrics["uncal_sensitivity"]),
            "Specificity":       _r(metrics["uncal_specificity"]),
            "PPV":               _r(metrics["uncal_ppv"]),
            "NPV":               _r(metrics["uncal_npv"]),
            "F1_active":         _r(metrics["uncal_f1_active"]),
            "F1_macro":          _r(metrics["uncal_f1_macro"]),
            "Accuracy":          _r(metrics["uncal_accuracy"]),
            "ECE":               _r(metrics["ece_uncal"]),
            "Brier_Score":       _r(metrics["uncal_brier"]),
        },

        "confusion_matrix_calibrated": {
            "TP": metrics["calib_TP"],
            "FP": metrics["calib_FP"],
            "FN": metrics["calib_FN"],
            "TN": metrics["calib_TN"],
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)


def save_curve_csv(
    col1: np.ndarray, col2: np.ndarray, col3: np.ndarray,
    col_names: List[str], out_path: str,
) -> None:
    n = len(col1)
    if len(col3) == n - 1:
        col3 = np.append(col3, np.nan)
    pd.DataFrame({col_names[0]: col1, col_names[1]: col2,
                  col_names[2]: col3}).to_csv(
        out_path, index=False, encoding="utf-8-sig", float_format="%.6f")


def save_calibration_csv(
    bins_uncal: List[dict], bins_calib: List[dict], out_path: str,
) -> None:
    rows = []
    for bu, bc in zip(bins_uncal, bins_calib):
        rows.append({
            "bin_lower":  bu["lower"], "bin_upper": bu["upper"],
            "bin_mid":    (bu["lower"] + bu["upper"]) / 2,
            "n_uncal": bu["n"],    "acc_uncal":  bu["accuracy"],
            "conf_uncal": bu["confidence"],
            "n_calib": bc["n"],    "acc_calib":  bc["accuracy"],
            "conf_calib": bc["confidence"],
        })
    pd.DataFrame(rows).to_csv(
        out_path, index=False, encoding="utf-8-sig", float_format="%.6f")


def save_test_preds_csv(
    exam_keys: np.ndarray, y_true: np.ndarray,
    logits_uncal: np.ndarray, temperature: float, out_path: str,
) -> None:
    """
    KEY output — direct input for evaluate_meta_on_test.py (Stacking LR).
    Columns: exam_key, y_true, logit_uncal, prob_uncal,
             temperature, logit_calib, prob_calib
    """
    prob_uncal           = sigmoid(logits_uncal)
    logits_cal, prob_cal = apply_temperature_scaling(logits_uncal, temperature)
    pd.DataFrame({
        "exam_key":    exam_keys,
        "y_true":      y_true,
        "logit_uncal": logits_uncal,
        "prob_uncal":  prob_uncal,
        "temperature": temperature,
        "logit_calib": logits_cal,
        "prob_calib":  prob_cal,
    }).to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")


# ─────────────────────────────────────────────────────────────────────────────
# Report & summary logger
# ─────────────────────────────────────────────────────────────────────────────

def log_summary_table(logf, metrics: dict, modality: str, model_name: str) -> None:
    SEP = "─" * 68
    log(logf, SEP)
    log(logf, f"SUMMARY TABLE  —  {model_name.upper()} / {modality}  "
              f"(threshold={metrics['threshold']:.2f})")
    log(logf, SEP)
    log(logf, f"  {'Metric':<22}  {'Uncalib':>10}  {'Calibrated':>12}")
    log(logf, f"  {'─'*22}  {'─'*10}  {'─'*12}")
    for label, key in [
        ("AUROC",             "auroc"),
        ("AUPRC",             "auprc"),
        ("Balanced Accuracy", "balanced_acc"),
        ("Sensitivity",       "sensitivity"),
        ("Specificity",       "specificity"),
        ("PPV",               "ppv"),
        ("NPV",               "npv"),
        ("F1 (active)",       "f1_active"),
        ("F1 (macro)",        "f1_macro"),
        ("Accuracy",          "accuracy"),
        ("Brier Score",       "brier"),
    ]:
        log(logf, f"  {label:<22}  "
                  f"{metrics[f'uncal_{key}']:>10.4f}  "
                  f"{metrics[f'calib_{key}']:>12.4f}")
    log(logf, f"  {'─'*22}  {'─'*10}  {'─'*12}")
    log(logf, f"  {'ECE':<22}  "
              f"{metrics['ece_uncal']:>10.4f}  "
              f"{metrics['ece_calib']:>12.4f}")
    log(logf, SEP)
    log(logf, f"  Confusion Matrix (calib):  "
              f"TP={metrics['calib_TP']}  FP={metrics['calib_FP']}  "
              f"FN={metrics['calib_FN']}  TN={metrics['calib_TN']}")
    log(logf, SEP)


def write_report(metrics: dict, run_cfg: dict, out_path: str) -> None:
    sep  = "=" * 72
    sep2 = "─" * 72
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("BASE MODEL — INDEPENDENT TEST SET EVALUATION REPORT\n")
        f.write(f"{sep}\n\n")

        f.write("[RUN CONFIG]\n")
        for k, v in run_cfg.items():
            f.write(f"  {k:<36}: {v}\n")

        n, na, ni = metrics["n_total"], metrics["n_active"], metrics["n_inactive"]
        f.write("\n[TEST SET COMPOSITION]\n")
        f.write(f"  Total    : {n}\n")
        f.write(f"  Active   : {na}  ({na/n*100:.1f}%)\n")
        f.write(f"  Inactive : {ni}  ({ni/n*100:.1f}%)\n")
        f.write(f"  Imbalance: {metrics['imbalance_ratio']:.2f}:1\n")

        f.write("\n[TEMPERATURE SCALING]\n")
        f.write(f"  T* (val set)  : {metrics['temperature']:.6f}\n")
        f.write(f"  ECE uncal     : {metrics['ece_uncal']:.4f}\n")
        f.write(f"  ECE calib     : {metrics['ece_calib']:.4f}\n")
        d = "↓ improved" if metrics["ece_reduction"] > 0 else "↑ worsened"
        f.write(f"  ECE Δ         : {metrics['ece_reduction']:.4f}  ({d})\n")
        f.write(f"  Brier uncal   : {metrics['uncal_brier']:.4f}\n")
        f.write(f"  Brier calib   : {metrics['calib_brier']:.4f}\n")

        f.write("\n[1. DISCRIMINATIVE PERFORMANCE]\n")
        f.write(f"  {'Metric':<25}  {'Uncalib':>10}  {'Calibrated':>12}\n")
        f.write(f"  {'-'*25}  {'-'*10}  {'-'*12}\n")
        for m in ["auroc", "auprc"]:
            f.write(f"  {m.upper():<25}  "
                    f"{metrics[f'uncal_{m}']:>10.4f}  "
                    f"{metrics[f'calib_{m}']:>12.4f}\n")

        f.write(f"\n[2. THRESHOLD-BASED METRICS  (thr={metrics['threshold']:.2f})]\n")
        f.write(f"  {'Metric':<25}  {'Uncalib':>10}  {'Calibrated':>12}\n")
        f.write(f"  {'-'*25}  {'-'*10}  {'-'*12}\n")
        for m, lbl in [
            ("balanced_acc", "Balanced Accuracy"),
            ("sensitivity",  "Sensitivity"),
            ("specificity",  "Specificity"),
            ("ppv",          "PPV"),
            ("f1_active",    "F1 (active)"),
            ("f1_macro",     "F1 (macro)"),
            ("accuracy",     "Accuracy"),
        ]:
            f.write(f"  {lbl:<25}  "
                    f"{metrics[f'uncal_{m}']:>10.4f}  "
                    f"{metrics[f'calib_{m}']:>12.4f}\n")

        f.write("\n[3. CLINICAL SAFETY METRICS  (calibrated)]\n")
        f.write(f"  PPV (active)    : {metrics['calib_ppv']:.4f}\n")
        f.write(f"  NPV (inactive)  : {metrics['calib_npv']:.4f}  ← rule-out safety\n")
        f.write(f"  Balanced Acc    : {metrics['calib_balanced_acc']:.4f}\n")
        f.write("\n  Confusion Matrix (calibrated):\n")
        f.write( "                        Pred active   Pred inactive\n")
        f.write(f"    True active      {metrics['calib_TP']:>12}  "
                f"{metrics['calib_FN']:>14}\n")
        f.write(f"    True inactive    {metrics['calib_FP']:>12}  "
                f"{metrics['calib_TN']:>14}\n")

        f.write("\n[4. CALIBRATION QUALITY]\n")
        f.write(f"  ECE  : {metrics['ece_uncal']:.4f} → {metrics['ece_calib']:.4f}\n")
        f.write(f"  Brier: {metrics['uncal_brier']:.4f} → {metrics['calib_brier']:.4f}\n")

        f.write(f"\n{sep2}\n")
        f.write("SUMMARY TABLE  (calibrated, post-TS)\n")
        f.write(f"{sep2}\n")
        f.write(f"  {'Metric':<22}  {'Value':>10}\n")
        f.write(f"  {'-'*22}  {'-'*10}\n")
        for lbl, key in [
            ("AUROC",             "calib_auroc"),
            ("AUPRC",             "calib_auprc"),
            ("Balanced Accuracy", "calib_balanced_acc"),
            ("Sensitivity",       "calib_sensitivity"),
            ("Specificity",       "calib_specificity"),
            ("PPV",               "calib_ppv"),
            ("NPV",               "calib_npv"),
            ("F1 (active)",       "calib_f1_active"),
            ("F1 (macro)",        "calib_f1_macro"),
            ("Accuracy",          "calib_accuracy"),
            ("ECE",               "ece_calib"),
            ("Brier",             "calib_brier"),
        ]:
            f.write(f"  {lbl:<22}  {metrics[key]:>10.4f}\n")
        f.write(f"{sep2}\n")

        f.write("\n[REFERENCES]\n")
        f.write("  AUROC/AUPRC : Saito & Rehmsmeier, PLOS ONE 2015\n")
        f.write("    https://doi.org/10.1371/journal.pone.0118432\n")
        f.write("  ECE / TS    : Guo et al., ICML 2017\n")
        f.write("    https://proceedings.mlr.press/v70/guo17a.html\n")
        f.write("  NPV Clinical: STARD-AI, Nature Medicine 2020\n")
        f.write("    https://doi.org/10.1038/s41591-020-0941-1\n")

        f.write("\n[OUTPUT FILES]\n")
        for fname in [
            "confusion_matrix.png", "roc_curve.png", "pr_curve.png",
            "reliability_diagram.png",
            "test_metrics.csv      (wide: row1=calibrated, row2=uncalibrated)",
            "test_summary.json     (structured summary — run info + all metrics)",
            "roc_data.csv", "pr_data.csv", "calibration_data.csv",
            "test_preds.csv        (KEY: input for evaluate_meta_on_test.py)",
            "test_evaluation_report.txt", "test_evaluation.log",
        ]:
            f.write(f"  {fname}\n")

        f.write("\n[PIPELINE ROLE]\n")
        f.write("  test_preds.csv → direct input for evaluate_meta_on_test.py\n")
        f.write("  (Stacking LR reads prob_calib from OCT0 / OCT1 / OCTA3)\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 0: Parse INPUT_DIR and resolve all paths ─────────────────────────
    parsed       = parse_input_dir(INPUT_DIR)
    model_name   = parsed["model_name"]
    modality     = parsed["modality"]
    run_tag      = parsed["run_tag"]
    best_fold    = parsed["best_fold"]
    project_root = parsed["project_root"]

    manifest_csv   = resolve_manifest_csv(MASTER_MANIFEST_CSV, project_root)
    ckpt_root      = resolve_checkpoint_root(CHECKPOINT_ROOT, project_root)
    test_eval_root = resolve_test_eval_root(TEST_EVAL_ROOT, project_root)

    temperature, fold_summary = load_fold_summary(INPUT_DIR, parsed)

    ckpt_path = resolve_checkpoint_path(
        ckpt_root, model_name, modality, run_tag, best_fold
    )

    out_dir = os.path.join(test_eval_root, model_name, modality, run_tag)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "test_evaluation.log"),
                "a", buffering=1, encoding="utf-8")

    print(f"\n{'='*66}")
    print(f"  BASE MODEL TEST EVALUATION")
    print(f"  {model_name.upper()} / {modality} / fold {best_fold}")
    print(f"  INPUT_DIR: {INPUT_DIR}")
    print(f"{'='*66}")

    log(logf, "=" * 66)
    log(logf, "BASE MODEL TEST EVALUATION")
    log(logf, "=" * 66)
    log(logf, f"INPUT_DIR      : {INPUT_DIR}")
    log(logf, f"model_name     : {model_name}")
    log(logf, f"modality       : {modality}")
    log(logf, f"run_tag        : {run_tag}")
    log(logf, f"best_fold      : {best_fold}")
    log(logf, f"project_root   : {project_root}")
    log(logf, f"manifest_csv   : {manifest_csv}")
    log(logf, f"checkpoint     : {ckpt_path}")
    log(logf, f"out_dir        : {out_dir}")
    log(logf, f"device         : {device}")
    log(logf, f"timm_model     : {TIMM_MODEL_MAP[model_name]}")
    log(logf, f"num_classes    : 1  (single sigmoid head, output [B,1])")
    log(logf, f"temperature T* : {temperature:.6f}  (from val set)")

    nll_before = fold_summary.get("nll_before_ts")
    nll_after  = fold_summary.get("nll_after_ts")
    if nll_before is not None:
        log(logf, f"NLL before TS  : {nll_before:.6f}")
    if nll_after is not None:
        log(logf, f"NLL after  TS  : {nll_after:.6f}")
    log(logf, f"Best val BCE   : {fold_summary.get('best_val_bce', 'n/a')}")
    log(logf, f"Best val AUC   : {fold_summary.get('best_val_auc', 'n/a')}")
    log(logf, f"Best epoch     : {fold_summary.get('best_epoch', 'n/a')}")

    # ── Step 1: Load test data from master_manifest.csv ───────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 1: Load test split from master_manifest.csv")
    log(logf, f"  Applying same filters as train_singlemode_oof.py:")
    log(logf, f"  label_conflict==0  →  y_true notna  →  {MODALITY_HAS_COL[modality]}==1"
              f"  →  image_path notna  →  split_set=='test'")

    df_test = load_test_dataframe(manifest_csv, modality, logf)

    n_test     = len(df_test)
    n_active   = int((df_test["y_true"] == 1).sum())
    n_inactive = int((df_test["y_true"] == 0).sum())

    log(logf, f"Test set final : {n_test} samples  "
              f"active={n_active} ({n_active/n_test*100:.1f}%)  "
              f"inactive={n_inactive} ({n_inactive/n_test*100:.1f}%)  "
              f"imbalance={n_inactive/n_active:.2f}:1")

    # ── Step 2: Build model and load checkpoint ───────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 2: Build model and load checkpoint")

    model = build_model(model_name)
    model = load_checkpoint(model, ckpt_path, device)
    log(logf, f"Model ready: {TIMM_MODEL_MAP[model_name]}  "
              f"num_classes=1  output=[B,1]")

    # ── Step 3: DataLoader ────────────────────────────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 3: Create DataLoader")

    img_col     = MODALITY_IMG_COL[modality]
    test_ds     = SingleModalityDataset(df_test, img_col, get_test_transform())
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    log(logf, f"DataLoader: {len(test_ds)} samples  "
              f"{len(test_loader)} batches  bs={BATCH_SIZE}")

    # ── Step 4: Inference ─────────────────────────────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 4: Inference")

    exam_keys, y_true, logits_uncal = run_inference(
        model, test_loader, device, logf)

    log(logf, f"Done: {len(y_true)} samples  "
              f"logit [{logits_uncal.min():.3f}, {logits_uncal.max():.3f}]  "
              f"mean={logits_uncal.mean():.3f}")

    # ── Step 5: Temperature Scaling ───────────────────────────────────────────
    log(logf, "─" * 66)
    log(logf, f"Step 5: Temperature Scaling (T*={temperature:.6f})")

    logits_calib, prob_calib = apply_temperature_scaling(logits_uncal, temperature)
    prob_uncal = sigmoid(logits_uncal)

    log(logf, f"Prob uncal [{prob_uncal.min():.3f}, {prob_uncal.max():.3f}]  "
              f"mean={prob_uncal.mean():.3f}")
    log(logf, f"Prob calib [{prob_calib.min():.3f}, {prob_calib.max():.3f}]  "
              f"mean={prob_calib.mean():.3f}")

    # ── Step 6: Metrics ───────────────────────────────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 6: Compute metrics")

    metrics, bins_uncal, bins_calib = compute_all_metrics(
        y_true, logits_uncal, temperature,
        threshold=THRESHOLD, ece_n_bins=ECE_N_BINS,
    )
    log_summary_table(logf, metrics, modality, model_name)

    # ── Step 7: Plots ─────────────────────────────────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 7: Generate plots")

    tp_str       = f"{model_name.upper()} / {modality} (Test Set)"
    y_pred_calib = (prob_calib >= THRESHOLD).astype(int)

    plot_confusion_matrix(
        y_true, y_pred_calib,
        os.path.join(out_dir, "confusion_matrix.png"),
        f"Confusion Matrix — {tp_str}")

    fpr_c, tpr_c, roc_thr_c = plot_roc_curve(
        y_true, prob_uncal, prob_calib,
        os.path.join(out_dir, "roc_curve.png"),
        f"ROC Curve — {tp_str}")

    pre_c, rec_c, pr_thr_c = plot_pr_curve(
        y_true, prob_uncal, prob_calib,
        os.path.join(out_dir, "pr_curve.png"),
        f"Precision-Recall Curve — {tp_str}")

    plot_reliability_diagram(
        bins_uncal, bins_calib,
        metrics["ece_uncal"], metrics["ece_calib"],
        os.path.join(out_dir, "reliability_diagram.png"),
        f"Reliability Diagram — {tp_str}")

    log(logf, "Plots saved.")

    # ── Step 8: Build run_cfg + Save data files ──────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 8: Save data files")

    # Build run_cfg here so it is available for both test_summary.json (step 8)
    # and test_evaluation_report.txt (step 9).
    run_cfg = {
        "INPUT_DIR":        INPUT_DIR,
        "model_name":       model_name,
        "timm_backbone":    TIMM_MODEL_MAP[model_name],
        "num_classes":      1,
        "modality":         modality,
        "run_tag":          run_tag,
        "best_fold":        best_fold,
        "temperature_T*":   f"{temperature:.6f}",
        "nll_before_ts":    nll_before,
        "nll_after_ts":     nll_after,
        "best_val_bce":     fold_summary.get("best_val_bce"),
        "best_val_auc":     fold_summary.get("best_val_auc"),
        "best_epoch":       fold_summary.get("best_epoch"),
        "checkpoint":       ckpt_path,
        "manifest_csv":     manifest_csv,
        "split_set_filter": "test",
        "threshold":        THRESHOLD,
        "ece_n_bins":       ECE_N_BINS,
        "batch_size":       BATCH_SIZE,
        "img_size":         IMG_SIZE,
        "device":           str(device),
        "out_dir":          out_dir,
    }

    save_metrics_csv(metrics, os.path.join(out_dir, "test_metrics.csv"))
    save_curve_csv(fpr_c, tpr_c, roc_thr_c, ["fpr", "tpr", "threshold"],
                   os.path.join(out_dir, "roc_data.csv"))
    save_curve_csv(rec_c, pre_c, pr_thr_c, ["recall", "precision", "threshold"],
                   os.path.join(out_dir, "pr_data.csv"))
    save_calibration_csv(bins_uncal, bins_calib,
                         os.path.join(out_dir, "calibration_data.csv"))
    save_test_preds_csv(exam_keys, y_true, logits_uncal, temperature,
                        os.path.join(out_dir, "test_preds.csv"))
    save_test_summary_json(
        metrics, run_cfg, fold_summary,
        os.path.join(out_dir, "test_summary.json"),
    )

    log(logf, "All data files saved.")
    log(logf, "  test_metrics.csv  : wide table (row1=calibrated, row2=uncalibrated)")
    log(logf, "  test_summary.json : full structured summary")
    log(logf, "  [KEY] test_preds.csv → evaluate_meta_on_test.py")

    # ── Step 9: Write text report ────────────────────────────────────────────
    log(logf, "─" * 66)
    log(logf, "Step 9: Write report")

    write_report(metrics, run_cfg,
                 os.path.join(out_dir, "test_evaluation_report.txt"))

    # ── Done ──────────────────────────────────────────────────────────────────
    log(logf, "=" * 66)
    log(logf, "EVALUATION COMPLETE")
    log(logf, f"  Results : {out_dir}")
    log(logf, f"  [NEXT]  test_preds.csv → evaluate_meta_on_test.py")
    log(logf, "=" * 66)

    logf.close()
    gc.collect()


if __name__ == "__main__":
    main()