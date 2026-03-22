# train_singlemode_oof.py

"""
Base model training (OOF val output) for mCNV binary classification.

Outputs
-------
checkpoints/
    <model_name>/<modality>/<RunTag>/Kfold/foldx/
        model_best.pth
        checkpoint_last.pth
    <model_name>/<modality>/<RunTag>/Best_foldx/

outputs/training/
    <model_name>/<modality>/<RunTag>/Kfold/foldx/
        training.log
        metrics.csv
        learning_curves.png
        fold_summary.json
    <model_name>/<modality>/<RunTag>/Best_foldx/
    <model_name>/<modality>/<RunTag>/RUN_CONFIG.txt
    <model_name>/<modality>/<RunTag>/run_config.json
    <model_name>/<modality>/<RunTag>/summary.json

outputs/oof_predictions/
    <model_name>/<modality>/<RunTag>/all_folds_oof.csv

Terminal
-------
python train_singlemode_oof.py --model_name swin_tiny --modality OCT0
python train_singlemode_oof.py --model_name VGG16 --modality OCT0
python train_singlemode_oof.py --model_name efficientnet_b0 --modality OCT0
"""

import os
import gc
import csv
import json
import time
import random
import shutil
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from training.model_factory import (
        create_model,
        normalize_model_name,
        get_backbone_name,
    )
except Exception:
    from model_factory import (
        create_model,
        normalize_model_name,
        get_backbone_name,
    )


# deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ===================== DEFAULT CONFIG =====================
PROJECT_ROOT_DIR = "/data/Irene/SwinTransformer/Swin_Meta"

MASTER_MANIFEST_CSV = os.path.join(
    PROJECT_ROOT_DIR,
    "outputs", "manifests", "master_split", "master_manifest.csv"
)

CHECKPOINT_ROOT      = os.path.join(PROJECT_ROOT_DIR, "checkpoints")
TRAINING_OUTPUT_ROOT = os.path.join(PROJECT_ROOT_DIR, "outputs", "training")
OOF_ROOT             = os.path.join(PROJECT_ROOT_DIR, "outputs", "oof_predictions")

CLASS_NAMES = ["inactive", "active"]
NUM_CLASSES = 1
IMG_SIZE = 224
NUM_WORKERS = 4
RANDOM_SEED = 42
NUM_FOLDS = 5

# execution control
EXECUTE_SINGLE_FOLD = False
SINGLE_FOLD_INDEX = 1  # 1-based fold index if EXECUTE_SINGLE_FOLD=True

# train
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 2e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROP_RATE = 0.0

# optimizer / unfreeze
UNFREEZE_MODE = "FULL_FINETUNE"   # FIXED_BACKBONE / FULL_FINETUNE / LLRD_FULL
BACKBONE_LR_MULT = 0.1
LLRD_DECAY = 0.85

# focal-bce
FOCAL_LOSS_ALPHA = {
    "OCT0": [0.110, 0.890],
    "OCT1": [0.113, 0.887],
    "OCTA3": [0.130, 0.870],
}
"""
org:
FOCAL_LOSS_ALPHA = {
    "OCT0": [0.4, 3.2],
    "OCT1": [0.4, 3.2],
    "OCTA3": [0.3, 3.2],
}
"""
FOCAL_LOSS_GAMMA = 2.0

# weighted sampler
USE_WEIGHTED_SAMPLER = True
MANUAL_SAMPLE_WEIGHTS = {
    "OCT0": [1.0, 2.9],
    "OCT1": [1.0, 2.8],
    "OCTA3": [1.0, 2.6],
}
"""
org:
    "OCT0": [1.0, 3.0],
    "OCT1": [1.0, 3.0],
    "OCTA3": [1.0, 2.5],
"""

# TS / early stop
USE_TEMPERATURE_SCALING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4


# ===================== UTILS =====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fmt(x: float) -> str:
    return f"{x:.6g}"


def open_log(folder: str):
    ensure_dir(folder)
    return open(os.path.join(folder, "training.log"), "a", buffering=1, encoding="utf-8")


def log(logf, msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if logf is not None:
        logf.write(line + "\n")
        logf.flush()


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_run_config_txt(txt_path: str, params: dict):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RUN CONFIG\n")
        f.write("=" * 80 + "\n")
        for k, v in params.items():
            if isinstance(v, (list, tuple, dict)):
                f.write(f"{k}={json.dumps(v, ensure_ascii=False)}\n")
            else:
                f.write(f"{k}={v}\n")


def init_metrics_csv(csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "lr_max",
            "lr_min",
            "train_focal_loss",
            "train_bce",
            "train_acc",
            "val_bce",
            "val_acc",
            "val_auc",
        ])


def append_metrics_csv(
    csv_path: str,
    epoch: int,
    lr_max: float,
    lr_min: float,
    train_focal_loss: float,
    train_bce: float,
    train_acc: float,
    val_bce: float,
    val_acc: float,
    val_auc: float,
):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            int(epoch),
            float(lr_max),
            float(lr_min),
            float(train_focal_loss),
            float(train_bce),
            float(train_acc),
            float(val_bce),
            float(val_acc),
            float(val_auc),
        ])


def safe_auc(labels, probs) -> float:
    try:
        v = roc_auc_score(labels, probs)
        if not np.isfinite(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def normalize_manifest_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["exam_key", "patient_id", "eye", "exam_date", "split_set"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    if "y_true" in out.columns:
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

    for col in [
        "has_oct0", "has_oct1", "has_octa3",
        "has_oct_pair", "is_complete_three_path",
        "label_conflict", "fold_id"
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ===================== DATASET =====================
class ManifestImageDataset(Dataset):
    """
    One row = one exam unit from master_manifest.csv.
    Image path is selected by modality.
    """

    def __init__(self, df: pd.DataFrame, modality: str, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.modality = modality
        self.transform = transform

        path_col_map = {
            "OCT0": "oct0_image_path",
            "OCT1": "oct1_image_path",
            "OCTA3": "octa3_image_path",
        }
        self.path_col = path_col_map[modality]

        self.image_paths = self.df[self.path_col].astype(str).tolist()
        self.targets = self.df["y_true"].astype(int).to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        y = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# ===================== LOSSES =====================
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, eps=1e-7):
        super().__init__()
        if alpha is None:
            self.alpha_neg, self.alpha_pos = 1.0, 1.0
        else:
            self.alpha_neg = float(alpha[0])
            self.alpha_pos = float(alpha[1])
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, logits, targets):
        if logits.ndim != 2 or logits.size(1) != 1:
            raise ValueError("FocalBCELoss expects logits shape [B,1].")

        t = targets.view(-1, 1).float()
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)

        bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
        alpha = t * self.alpha_pos + (1.0 - t) * self.alpha_neg
        pt = p * t + (1.0 - p) * (1.0 - t)
        focal = (1.0 - pt).pow(self.gamma)

        return (alpha * focal * bce).mean()


# ===================== MODEL / OPTIMIZER =====================
def set_requires_grad(model: nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad


def apply_unfreeze_mode(model: nn.Module, mode: str):
    mode = str(mode).upper()

    if mode == "FIXED_BACKBONE":
        set_requires_grad(model, False)

        found = False
        for head_name in ["head", "classifier", "fc"]:
            if hasattr(model, head_name):
                head_module = getattr(model, head_name)
                if isinstance(head_module, nn.Module):
                    for p in head_module.parameters():
                        p.requires_grad = True
                    found = True

        if not found:
            for name, p in model.named_parameters():
                if any(k in name.lower() for k in ["head", "classifier", "fc"]):
                    p.requires_grad = True

    elif mode in ("FULL_FINETUNE", "LLRD_FULL"):
        set_requires_grad(model, True)

    else:
        raise ValueError(f"Unknown UNFREEZE_MODE={mode}")


def is_no_weight_decay(name: str, param: torch.Tensor) -> bool:
    if name.endswith(".bias"):
        return True
    if param.ndim == 1:
        return True
    lname = name.lower()
    if "norm" in lname or "bn" in lname:
        return True
    return False


def swin_layer_id_from_name(name: str) -> int:
    if name.startswith("patch_embed"):
        return 0
    if name.startswith("layers.0"):
        return 1
    if name.startswith("layers.1"):
        return 2
    if name.startswith("layers.2"):
        return 3
    if name.startswith("layers.3"):
        return 4
    if name.startswith("norm"):
        return 5
    return 0


def build_optimizer(
    model: nn.Module,
    model_name: str,
    lr: float,
    weight_decay: float,
    unfreeze_mode: str,
    backbone_lr_mult: float,
    llrd_decay: float,
):
    mode = str(unfreeze_mode).upper()

    if mode == "FULL_FINETUNE":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if mode == "FIXED_BACKBONE":
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if mode == "LLRD_FULL":
        # Swin gets LLRD; other backbones fallback to normal AdamW
        if model_name != "swin_tiny":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        groups = {}
        max_backbone_id = 5

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if name.startswith("head"):
                pg_lr = lr
            else:
                lid = swin_layer_id_from_name(name)
                pg_lr = lr * backbone_lr_mult * (llrd_decay ** (max_backbone_id - lid))

            pg_wd = 0.0 if is_no_weight_decay(name, p) else weight_decay
            key = (pg_lr, pg_wd)

            if key not in groups:
                groups[key] = {"params": [], "lr": pg_lr, "weight_decay": pg_wd}
            groups[key]["params"].append(p)

        return optim.AdamW(list(groups.values()))

    raise ValueError(f"Unknown UNFREEZE_MODE={mode}")


# ===================== TEMPERATURE SCALING =====================
def calibrate_temperature(val_loader, model, device):
    model.eval()
    nll = nn.BCEWithLogitsLoss()
    logits_all, targets_all = [], []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Collect val logits for TS"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = model(x)
            logits_all.append(z)
            targets_all.append(y.view(-1, 1).float())

    if len(logits_all) == 0:
        return 1.0, None, None

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)

    with torch.no_grad():
        before = float(nll(logits, targets).item())

    best_t = 1.0
    best_nll = before

    for init_t in [0.7, 1.0, 1.3, 1.8, 2.5]:
        log_t = nn.Parameter(torch.tensor([np.log(init_t)], device=device))
        opt = optim.LBFGS([log_t], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            t = torch.exp(log_t).clamp(1e-3, 10.0)
            loss = nll(logits / t, targets)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            with torch.no_grad():
                t = float(torch.exp(log_t).clamp(1e-3, 10.0).item())
                n = float(nll(logits / t, targets).item())
            if n < best_nll:
                best_nll = n
                best_t = t
        except RuntimeError:
            continue

    if best_nll >= before:
        return 1.0, before, before

    return best_t, before, best_nll


# ===================== PLOTS =====================
def plot_learning_curves(history: dict, save_path: str):
    """
    Left  : train/valid accuracy
    Right : train BCE / valid BCE
    """
    if len(history["train_focal_loss"]) == 0:
        return

    epochs = range(1, len(history["train_focal_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_acc"], "b-", label="Training", linewidth=2)
    ax1.plot(epochs, history["val_acc"], "r-", label="Validation", linewidth=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_bce"], "b-", label="Training", linewidth=2)
    ax2.plot(epochs, history["val_bce"], "r-", label="Validation", linewidth=2)
    ax2.set_title("Training and Validation Loss", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("BCE Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===================== DATA PREP =====================
def load_master_manifest(master_csv: str, modality: str) -> pd.DataFrame:
    if not os.path.isfile(master_csv):
        raise FileNotFoundError(master_csv)

    df = pd.read_csv(master_csv)
    df = normalize_manifest_dtypes(df)

    required_cols = [
        "exam_key", "patient_id", "split_set", "y_true", "fold_id",
        "has_oct0", "has_oct1", "has_octa3",
        "oct0_image_path", "oct1_image_path", "octa3_image_path"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in master_manifest.csv: {missing}")

    df = df[df["label_conflict"] == 0].copy()
    df = df[df["y_true"].notna()].copy()

    if modality == "OCT0":
        df = df[df["has_oct0"] == 1].copy()
        df = df[df["oct0_image_path"].notna()].copy()
    elif modality == "OCT1":
        df = df[df["has_oct1"] == 1].copy()
        df = df[df["oct1_image_path"].notna()].copy()
    elif modality == "OCTA3":
        df = df[df["has_octa3"] == 1].copy()
        df = df[df["octa3_image_path"].notna()].copy()
    else:
        raise ValueError(f"Unsupported modality={modality}")

    train_valid_df = df[df["split_set"] == "train_valid"].copy()
    valid_folds = sorted(train_valid_df["fold_id"].dropna().astype(int).unique().tolist())
    if set(valid_folds) != set(range(1, NUM_FOLDS + 1)):
        raise ValueError(
            f"Unexpected fold_id set for train_valid {modality}: {valid_folds}. "
            f"Expected {list(range(1, NUM_FOLDS + 1))}"
        )

    return df.reset_index(drop=True)


def build_fold_dfs(df: pd.DataFrame, fold_id: int):
    train_df = df[(df["split_set"] == "train_valid") & (df["fold_id"] != fold_id)].copy().reset_index(drop=True)
    val_df = df[(df["split_set"] == "train_valid") & (df["fold_id"] == fold_id)].copy().reset_index(drop=True)
    test_df = df[df["split_set"] == "test"].copy().reset_index(drop=True)

    train_patients = set(train_df["patient_id"].astype(str).tolist())
    val_patients = set(val_df["patient_id"].astype(str).tolist())
    if len(train_patients.intersection(val_patients)) != 0:
        raise RuntimeError(f"Patient leakage detected in fold {fold_id}")

    return train_df, val_df, test_df


# ===================== TRAIN ONE FOLD =====================
def train_one_fold(
    fold_num: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device,
    model_name: str,
    modality: str,
    ckpt_run_dir: str,
    train_run_dir: str,
    per_fold_oof_dir: str,
    args,
):
    ckpt_fold_dir = os.path.join(ckpt_run_dir, "Kfold", f"fold{fold_num}")
    train_fold_dir = os.path.join(train_run_dir, "Kfold", f"fold{fold_num}")
    ensure_dir(ckpt_fold_dir)
    ensure_dir(train_fold_dir)
    ensure_dir(per_fold_oof_dir)

    logf = open_log(train_fold_dir)
    metrics_csv_path = os.path.join(train_fold_dir, "metrics.csv")
    learning_curves_png_path = os.path.join(train_fold_dir, "learning_curves.png")

    model_best_path = os.path.join(ckpt_fold_dir, "model_best.pth")
    checkpoint_last_path = os.path.join(ckpt_fold_dir, "checkpoint_last.pth")

    init_metrics_csv(metrics_csv_path)

    tf_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.98, 1.02)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = ManifestImageDataset(train_df, modality=modality, transform=tf_train)
    val_set = ManifestImageDataset(val_df, modality=modality, transform=tf_val)

    with open(os.path.join(train_fold_dir, "fold_config.txt"), "w", encoding="utf-8") as fcfg:
        fcfg.write("=" * 80 + "\n")
        fcfg.write(f"FOLD {fold_num} CONFIG\n")
        fcfg.write("=" * 80 + "\n")
        fcfg.write(f"Train exam units: {len(train_df)}\n")
        fcfg.write(f"  Train[inactive] = {(train_df['y_true'] == 0).sum()}\n")
        fcfg.write(f"  Train[active]   = {(train_df['y_true'] == 1).sum()}\n")
        fcfg.write(f"Train patients: {train_df['patient_id'].nunique()}\n")
        fcfg.write(f"Val exam units: {len(val_df)}\n")
        fcfg.write(f"  Val[inactive] = {(val_df['y_true'] == 0).sum()}\n")
        fcfg.write(f"  Val[active]   = {(val_df['y_true'] == 1).sum()}\n")
        fcfg.write(f"Val patients: {val_df['patient_id'].nunique()}\n")

    with open(os.path.join(train_fold_dir, "train_exam_keys.txt"), "w", encoding="utf-8") as f:
        for k in train_df["exam_key"].astype(str).tolist():
            f.write(k + "\n")

    with open(os.path.join(train_fold_dir, "val_exam_keys.txt"), "w", encoding="utf-8") as f:
        for k in val_df["exam_key"].astype(str).tolist():
            f.write(k + "\n")

    if USE_WEIGHTED_SAMPLER:
        train_targets = train_df["y_true"].astype(int).to_numpy()
        cls_w = torch.as_tensor(MANUAL_SAMPLE_WEIGHTS[modality], dtype=torch.double)
        tr_targets_t = torch.as_tensor(train_targets, dtype=torch.long)
        sample_w = cls_w[tr_targets_t].double().cpu()

        sampler = WeightedRandomSampler(
            sample_w,
            num_samples=len(sample_w),
            replacement=True
        )

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0
        )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    model = create_model(
        model_name=model_name,
        num_classes=1,
        pretrained=not args.no_pretrained,
        drop_rate=DROP_RATE,
    ).to(device)

    apply_unfreeze_mode(model, UNFREEZE_MODE)

    focal = FocalBCELoss(alpha=FOCAL_LOSS_ALPHA[modality], gamma=FOCAL_LOSS_GAMMA)
    bce = nn.BCEWithLogitsLoss()

    optimizer = build_optimizer(
        model=model,
        model_name=model_name,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        unfreeze_mode=UNFREEZE_MODE,
        backbone_lr_mult=BACKBONE_LR_MULT,
        llrd_decay=LLRD_DECAY,
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

    history = {
        "train_focal_loss": [],
        "train_bce": [],
        "train_acc": [],
        "val_bce": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_val_oof_df = None
    best_key = float("inf")
    best_val_auc = 0.0
    best_epoch = -1
    patience = 0

    log(
        logf,
        f"Start Fold {fold_num} | MODEL={model_name}({get_backbone_name(model_name)}) "
        f"MODALITY={modality} BS={BATCH_SIZE} EP={NUM_EPOCHS} "
        f"LR={LR} WD={WEIGHT_DECAY} "
        f"UNFREEZE_MODE={UNFREEZE_MODE} "
        f"FocalAlpha={FOCAL_LOSS_ALPHA[modality]} FocalGamma={FOCAL_LOSS_GAMMA}"
    )

    try:
        for ep in range(NUM_EPOCHS):
            # ---------- train ----------
            model.train()
            train_focal_sum = 0.0
            train_bce_sum = 0.0
            train_corr = 0
            train_tot = 0

            num_batches = max(1, len(train_loader))
            pbar = tqdm(
                enumerate(train_loader),
                total=num_batches,
                desc=f"Fold{fold_num} Epoch {ep+1}/{NUM_EPOCHS}"
            )

            for batch_idx, (x, y) in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)

                loss = focal(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                scheduler.step(ep + batch_idx / num_batches)

                with torch.no_grad():
                    bs = y.size(0)
                    train_focal_sum += float(loss.item()) * bs

                    train_bce_batch = bce(logits.detach(), y.float().view(-1, 1))
                    train_bce_sum += float(train_bce_batch.item()) * bs

                    pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
                    train_corr += (pred == y).sum().item()
                    train_tot += bs

                pbar.set_postfix({
                    "focal": f"{train_focal_sum / max(1, train_tot):.4f}",
                    "bce": f"{train_bce_sum / max(1, train_tot):.4f}",
                    "acc": f"{train_corr / max(1, train_tot):.4f}",
                })

            train_focal_loss = train_focal_sum / max(1, train_tot)
            train_bce_loss = train_bce_sum / max(1, train_tot)
            train_acc = train_corr / max(1, train_tot)

            # ---------- validation ----------
            model.eval()
            val_bce_sum = 0.0
            val_corr = 0
            val_tot = 0

            probs, labels, logits_list = [], [], []

            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"[Val] Fold{fold_num} Epoch{ep+1}/{NUM_EPOCHS}"):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    z = model(x)
                    l = bce(z, y.float().view(-1, 1))

                    bs = y.size(0)
                    val_bce_sum += float(l.item()) * bs
                    val_tot += bs

                    pred = (torch.sigmoid(z) >= 0.5).long().view(-1)
                    val_corr += (pred == y).sum().item()

                    z_flat = z.view(-1)
                    p1 = torch.sigmoid(z_flat)

                    logits_list.extend(z_flat.detach().cpu().tolist())
                    probs.extend(p1.detach().cpu().tolist())
                    labels.extend(y.detach().cpu().tolist())

            val_bce = val_bce_sum / max(1, val_tot)
            val_acc = val_corr / max(1, val_tot)
            val_auc = safe_auc(labels, probs)

            history["train_focal_loss"].append(train_focal_loss)
            history["train_bce"].append(train_bce_loss)
            history["train_acc"].append(train_acc)
            history["val_bce"].append(val_bce)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)

            lrs = [pg["lr"] for pg in optimizer.param_groups]
            lr_max = max(lrs)
            lr_min = min(lrs)

            append_metrics_csv(
                metrics_csv_path,
                epoch=ep + 1,
                lr_max=lr_max,
                lr_min=lr_min,
                train_focal_loss=train_focal_loss,
                train_bce=train_bce_loss,
                train_acc=train_acc,
                val_bce=val_bce,
                val_acc=val_acc,
                val_auc=val_auc,
            )

            log(
                logf,
                f"[Epoch {ep+1}] "
                f"lr_max={lr_max:.6g} lr_min={lr_min:.6g} "
                f"train_focal={train_focal_loss:.6f} "
                f"train_bce={train_bce_loss:.6f} "
                f"train_acc={train_acc:.4f} "
                f"val_bce={val_bce:.6f} "
                f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            )

            # always save latest
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fold": fold_num,
                    "model_name": model_name,
                    "modality": modality,
                },
                checkpoint_last_path
            )

            improved = (best_key - val_bce) > EARLY_STOP_MIN_DELTA
            tie = abs(best_key - val_bce) <= EARLY_STOP_MIN_DELTA
            better_auc_on_tie = tie and (val_auc > best_val_auc)

            if improved or better_auc_on_tie:
                best_key = val_bce
                best_val_auc = val_auc
                best_epoch = ep + 1
                patience = 0

                torch.save(
                    {
                        "epoch": ep,
                        "best_epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_key_metric": float(best_key),
                        "val_nll": float(val_bce),
                        "val_acc": float(val_acc),
                        "val_auc": float(val_auc),
                        "fold": fold_num,
                        "model_name": model_name,
                        "modality": modality,
                    },
                    model_best_path
                )

                best_val_oof_df = val_df.copy().reset_index(drop=True)
                best_val_oof_df["logit_uncal"] = logits_list
                best_val_oof_df["prob_uncal"] = probs
                best_val_oof_df["fold"] = fold_num
                best_val_oof_df["best_epoch"] = best_epoch

                log(
                    logf,
                    f"[NEW_BEST] epoch={best_epoch} val_bce={val_bce:.6f} "
                    f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
                )
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    log(logf, f"[early stop] Fold {fold_num} at epoch {ep+1}")
                    break

        # save learning curves
        plot_learning_curves(history, learning_curves_png_path)
        log(logf, f"[saved learning curves] {learning_curves_png_path}")

        if not os.path.exists(model_best_path):
            raise RuntimeError(f"No best checkpoint saved for fold {fold_num}: {model_best_path}")

        cp = torch.load(model_best_path, map_location=device, weights_only=False)
        model.load_state_dict(cp["model_state_dict"])

        t_star, nll_before, nll_after = 1.0, None, None
        if USE_TEMPERATURE_SCALING:
            t_star, nll_before, nll_after = calibrate_temperature(val_loader, model, device)

        # overwrite best checkpoint with calibrated metadata
        torch.save(
            {
                "epoch": cp["epoch"],
                "best_epoch": cp.get("best_epoch", cp["epoch"]),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": cp.get("optimizer_state_dict"),
                "temperature": float(t_star),
                "val_nll_uncal": float(cp["val_nll"]),
                "val_acc": float(cp["val_acc"]),
                "val_auc": float(cp["val_auc"]),
                "nll_beforeTS": None if nll_before is None else float(nll_before),
                "nll_afterTS": None if nll_after is None else float(nll_after),
                "fold": fold_num,
                "model_name": model_name,
                "modality": modality,
            },
            model_best_path
        )

        log(logf, f"[TS] T*={t_star:.6f} | NLL {nll_before}->{nll_after}")
        log(logf, f"[saved checkpoint] {model_best_path}")

        # save per-fold OOF csv under training/Kfold/foldx and temp merge folder
        if best_val_oof_df is not None:
            best_val_oof_df["temperature"] = float(t_star)
            best_val_oof_df["logit_calib"] = best_val_oof_df["logit_uncal"] / float(t_star)
            best_val_oof_df["prob_calib"] = 1.0 / (1.0 + np.exp(-best_val_oof_df["logit_calib"]))
            best_val_oof_df["model_name"] = model_name
            best_val_oof_df["modality"] = modality

            keep_cols = [
                "exam_key", "patient_id", "split_set", "fold_id", "y_true",
                "has_oct0", "has_oct1", "has_octa3",
                "oct0_image_path", "oct1_image_path", "octa3_image_path",
                "model_name", "modality",
                "logit_uncal", "prob_uncal",
                "temperature", "logit_calib", "prob_calib",
                "fold", "best_epoch"
            ]

            fold_oof_csv_training = os.path.join(train_fold_dir, f"val_oof_predictions_fold{fold_num}.csv")
            best_val_oof_df[keep_cols].to_csv(fold_oof_csv_training, index=False, encoding="utf-8-sig")

            fold_oof_csv_merge = os.path.join(per_fold_oof_dir, f"fold{fold_num}_oof.csv")
            best_val_oof_df[keep_cols].to_csv(fold_oof_csv_merge, index=False, encoding="utf-8-sig")

            log(logf, f"[saved OOF] {fold_oof_csv_training}")
        else:
            fold_oof_csv_training = ""
            fold_oof_csv_merge = ""

        fold_summary = {
            "fold": fold_num,
            "model_name": model_name,
            "backbone_name": get_backbone_name(model_name),
            "modality": modality,
            "train_exam_units": int(len(train_df)),
            "val_exam_units": int(len(val_df)),
            "train_patients": int(train_df["patient_id"].nunique()),
            "val_patients": int(val_df["patient_id"].nunique()),
            "best_val_bce": float(best_key),
            "best_val_auc": float(best_val_auc),
            "best_epoch": int(best_epoch),
            "temperature": float(t_star),
            "nll_before_ts": None if nll_before is None else float(nll_before),
            "nll_after_ts": None if nll_after is None else float(nll_after),
            "artifacts": {
                "model_best": model_best_path,
                "checkpoint_last": checkpoint_last_path,
                "metrics_csv": metrics_csv_path,
                "learning_curves_png": learning_curves_png_path,
                "fold_oof_csv": fold_oof_csv_training,
            }
        }
        save_json(os.path.join(train_fold_dir, "fold_summary.json"), fold_summary)

        return {
            "fold": fold_num,
            "best_val_key": float(best_key),
            "best_val_auc": float(best_val_auc),
            "best_epoch": int(best_epoch),
            "temperature": float(t_star),
            "nll_before": None if nll_before is None else float(nll_before),
            "nll_after": None if nll_after is None else float(nll_after),
        }

    finally:
        try:
            logf.close()
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ===================== RUN SUMMARY =====================
def merge_all_fold_oof(per_fold_oof_dir: str, final_oof_csv: str) -> str:
    csvs = []
    if not os.path.isdir(per_fold_oof_dir):
        return ""

    for fn in sorted(os.listdir(per_fold_oof_dir)):
        if fn.endswith("_oof.csv"):
            csvs.append(os.path.join(per_fold_oof_dir, fn))

    if len(csvs) == 0:
        return ""

    dfs = [pd.read_csv(p) for p in csvs]
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_df = all_df.sort_values(["fold", "exam_key"]).reset_index(drop=True)

    ensure_dir(os.path.dirname(final_oof_csv))
    all_df.to_csv(final_oof_csv, index=False, encoding="utf-8-sig")
    return final_oof_csv


def save_training_summary(summary: dict, train_run_dir: str):
    txt_path = os.path.join(train_run_dir, "training_summary.txt")
    json_path = os.path.join(train_run_dir, "summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("K-FOLD SINGLE-MODE OOF TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        cfg = summary["configuration"]
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {cfg['model_name']} ({cfg['backbone_name']})\n")
        f.write(f"Modality: {cfg['modality']}\n")
        f.write(f"Num folds: {cfg['num_folds']}\n")
        f.write(f"Batch size: {cfg['batch_size']}\n")
        f.write(f"Epochs: {cfg['num_epochs']}\n")
        f.write(f"LR: {cfg['lr']}\n")
        f.write(f"Weight decay: {cfg['weight_decay']}\n")
        f.write(f"Weighted sampler: {cfg['weighted_sampler']}\n")
        f.write(f"Temperature scaling: {cfg['temperature_scaling']}\n")
        f.write(f"Unfreeze mode: {cfg['unfreeze_mode']}\n")
        f.write(f"Backbone LR mult: {cfg['backbone_lr_mult']}\n")
        f.write(f"LLRD decay: {cfg['llrd_decay']}\n")
        f.write(f"Focal alpha: {cfg['focal_alpha']}\n")
        f.write(f"Focal gamma: {cfg['focal_gamma']}\n")
        f.write(f"Run tag: {cfg['run_tag']}\n\n")

        f.write("FOLD RESULTS\n")
        f.write("-" * 40 + "\n")
        for r in summary["fold_results"]:
            f.write(
                f"Fold {r['fold']}: "
                f"best_val_bce={r['best_val_key']:.6f}, "
                f"best_val_auc={r['best_val_auc']:.4f}, "
                f"best_epoch={r['best_epoch']}, "
                f"T={r['temperature']:.4f}\n"
            )

        if len(summary["fold_results"]) > 1:
            vals_bce = [x["best_val_key"] for x in summary["fold_results"]]
            vals_auc = [x["best_val_auc"] for x in summary["fold_results"]]
            f.write("\nAVERAGE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"val_bce: {np.mean(vals_bce):.6f} (±{np.std(vals_bce):.6f})\n")
            f.write(f"val_auc: {np.mean(vals_auc):.4f} (±{np.std(vals_auc):.4f})\n")

        f.write("\n")
        f.write(f"Best fold: {summary['best_fold_num']}\n")
        f.write(f"All folds OOF CSV: {summary['all_folds_oof_csv']}\n")
        f.write(f"Total training time (min): {summary['training_time_minutes']:.2f}\n")

    save_json(json_path, summary)


# ===================== MAIN =====================
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="swin_tiny",
        choices=["swin_tiny", "vgg16", "efficientnet_b0"]
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["OCT0", "OCT1", "OCTA3"]
    )
    parser.add_argument("--master_manifest_csv", type=str, default=MASTER_MANIFEST_CSV)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()

    model_name = normalize_model_name(args.model_name)
    modality = args.modality

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_master_manifest(args.master_manifest_csv, modality=modality)

    train_valid_df = df[df["split_set"] == "train_valid"].copy()
    test_df = df[df["split_set"] == "test"].copy()

    ws_tag = (
        f"_WSon_{fmt(MANUAL_SAMPLE_WEIGHTS[modality][0])}_{fmt(MANUAL_SAMPLE_WEIGHTS[modality][1])}"
        if USE_WEIGHTED_SAMPLER else "_WSoff"
    )

    llrd_tag = ""
    if str(UNFREEZE_MODE).upper().startswith("LLRD"):
        llrd_tag = f"_LRMul{fmt(BACKBONE_LR_MULT)}_LRDec{fmt(LLRD_DECAY)}"

    run_tag = (
        f"BS{BATCH_SIZE}"
        f"_EP{NUM_EPOCHS}"
        f"_LR{fmt(LR)}"
        f"_WD{fmt(WEIGHT_DECAY)}"
        f"_{UNFREEZE_MODE}"
        f"{llrd_tag}"
        f"_FL{fmt(FOCAL_LOSS_ALPHA[modality][0])}_{fmt(FOCAL_LOSS_ALPHA[modality][1])}_{fmt(FOCAL_LOSS_GAMMA)}"
        f"{ws_tag}"
    )

    ckpt_run_dir     = os.path.join(CHECKPOINT_ROOT,      model_name, modality, run_tag)
    train_run_dir    = os.path.join(TRAINING_OUTPUT_ROOT, model_name, modality, run_tag)
    oof_run_dir      = os.path.join(OOF_ROOT,             model_name, modality, run_tag)
    per_fold_oof_dir = os.path.join(oof_run_dir, "_per_fold")

    ensure_dir(os.path.join(ckpt_run_dir, "Kfold"))
    ensure_dir(os.path.join(train_run_dir, "Kfold"))
    ensure_dir(oof_run_dir)
    ensure_dir(per_fold_oof_dir)

    run_log = open_log(train_run_dir)

    run_config = {
        "model_name": model_name,
        "backbone_name": get_backbone_name(model_name),
        "modality": modality,
        "master_manifest_csv": args.master_manifest_csv,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "weighted_sampler": USE_WEIGHTED_SAMPLER,
        "temperature_scaling": USE_TEMPERATURE_SCALING,
        "unfreeze_mode": UNFREEZE_MODE,
        "backbone_lr_mult": BACKBONE_LR_MULT,
        "llrd_decay": LLRD_DECAY,
        "focal_alpha": FOCAL_LOSS_ALPHA[modality],
        "focal_gamma": FOCAL_LOSS_GAMMA,
        "train_valid_exam_units": int(len(train_valid_df)),
        "test_exam_units": int(len(test_df)),
        "train_valid_patients": int(train_valid_df["patient_id"].nunique()),
        "test_patients": int(test_df["patient_id"].nunique()),
        "usable_folds": sorted(train_valid_df["fold_id"].astype(int).unique().tolist()),
        "run_tag": run_tag,
        "execute_single_fold": EXECUTE_SINGLE_FOLD,
        "single_fold_index": SINGLE_FOLD_INDEX if EXECUTE_SINGLE_FOLD else None,
    }

    save_json(os.path.join(train_run_dir, "run_config.json"), run_config)
    save_run_config_txt(os.path.join(train_run_dir, "RUN_CONFIG.txt"), run_config)

    log(run_log, f"Loaded manifest: {args.master_manifest_csv}")
    log(run_log, f"Model={model_name}, Modality={modality}")
    log(run_log, f"RunTag={run_tag}")
    log(run_log, f"Train_valid exam units={len(train_valid_df)}, Test exam units={len(test_df)}")

    run_folds = [SINGLE_FOLD_INDEX] if EXECUTE_SINGLE_FOLD else list(range(1, NUM_FOLDS + 1))

    fold_results = []
    t0 = time.time()

    for fold_num in run_folds:
        train_fold_df, val_fold_df, _ = build_fold_dfs(df, fold_id=fold_num)

        print("\n" + "=" * 80)
        print(f"FOLD {fold_num}/{NUM_FOLDS}")
        print("=" * 80)

        log(run_log, f"Fold {fold_num}: train={len(train_fold_df)}, val={len(val_fold_df)}")

        res = train_one_fold(
            fold_num=fold_num,
            train_df=train_fold_df,
            val_df=val_fold_df,
            device=device,
            model_name=model_name,
            modality=modality,
            ckpt_run_dir=ckpt_run_dir,
            train_run_dir=train_run_dir,
            per_fold_oof_dir=per_fold_oof_dir,
            args=args,
        )

        fold_results.append(res)

    # choose best fold by val_bce
    best_fold_num = None
    if len(fold_results) > 0:
        best = min(fold_results, key=lambda r: r["best_val_key"])
        best_fold_num = best["fold"]

        src_ckpt = os.path.join(ckpt_run_dir, "Kfold", f"fold{best_fold_num}")
        dst_ckpt = os.path.join(ckpt_run_dir, f"Best_fold{best_fold_num}")
        if os.path.exists(dst_ckpt):
            shutil.rmtree(dst_ckpt)
        shutil.copytree(src_ckpt, dst_ckpt)

        src_train = os.path.join(train_run_dir, "Kfold", f"fold{best_fold_num}")
        dst_train = os.path.join(train_run_dir, f"Best_fold{best_fold_num}")
        if os.path.exists(dst_train):
            shutil.rmtree(dst_train)
        shutil.copytree(src_train, dst_train)

    all_folds_oof_csv = merge_all_fold_oof(
        per_fold_oof_dir=per_fold_oof_dir,
        final_oof_csv=os.path.join(oof_run_dir, "all_folds_oof.csv")
    )

    total_min = (time.time() - t0) / 60.0

    summary = {
        "configuration": {
            "model_name": model_name,
            "backbone_name": get_backbone_name(model_name),
            "modality": modality,
            "num_folds": NUM_FOLDS,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "weighted_sampler": USE_WEIGHTED_SAMPLER,
            "temperature_scaling": USE_TEMPERATURE_SCALING,
            "unfreeze_mode": UNFREEZE_MODE,
            "backbone_lr_mult": BACKBONE_LR_MULT,
            "llrd_decay": LLRD_DECAY,
            "focal_alpha": FOCAL_LOSS_ALPHA[modality],
            "focal_gamma": FOCAL_LOSS_GAMMA,
            "run_tag": run_tag,
            "execute_single_fold": EXECUTE_SINGLE_FOLD,
            "single_fold_index": SINGLE_FOLD_INDEX if EXECUTE_SINGLE_FOLD else None,
        },
        "fold_results": fold_results,
        "best_fold_num": best_fold_num,
        "all_folds_oof_csv": all_folds_oof_csv,
        "training_time_minutes": total_min,
    }

    save_training_summary(summary, train_run_dir)

    if os.path.exists(per_fold_oof_dir):
        shutil.rmtree(per_fold_oof_dir)

    log(run_log, f"Best fold={best_fold_num}")
    log(run_log, f"All folds OOF CSV={all_folds_oof_csv}")
    log(run_log, f"Total training time={total_min:.2f} min")
    run_log.close()


if __name__ == "__main__":
    main()