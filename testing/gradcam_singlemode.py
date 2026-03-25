# gradcam_singlemode.py

"""
Reads test_preds.csv produced by test_singlemode.py, groups samples by
confusion-matrix quadrant (TP / TN / FP / FN), randomly selects one image
per quadrant, and generates Grad-CAM overlays for the predicted class.

Outputs  ->  <TEST_EVAL_DIR>/gradcam/<timestamp>/
  GradCAM_4x3_panel.png   4x3: original | Grad-CAM class-0 | Grad-CAM class-1
  TP_panel.png  FP_panel.png  FN_panel.png  TN_panel.png
  gradcam_samples.csv     per-sample record (quadrant, filename, GT, Pred, ...)
  gradcam_log.txt

Algorithm (Grad-CAM for Swin-Tiny, binary BCEWithLogits)
---------------------------------------------------------
Swin-Tiny specifics (no cls_token, hierarchical patch merging):
  - Token output of the last stage: [B, N, C] where N = (H/32)*(W/32) = 7*7=49
    for 224x224 input.  (patch_size=4, 4 stages -> 4x merging -> 32x downscale)
  - Target layer  : model.norm  (Global LayerNorm after all 4 Swin stages)
    norm1 captures intra-window local features -> fragmented, background-biased.
    model.norm integrates globally -> coherent heatmap focused on lesion.
  - Threshold     : CAM_THRESHOLD applied at 7x7 before upsampling.
    Suppresses low-contribution pixels (black borders, specular artefacts).
  - Upsample      : bicubic 7x7 -> 224x224  (smoother than bilinear)
  - reshape_transform: [B, N, C] or [B, H, W, C] -> [B, C, 7, 7]
  - Binary scoring:
      class=1 (active)   ->  score = logit
      class=0 (inactive) ->  score = -logit
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG  --  Edit only this section
# ==============================================================================

# 1. Directory that contains test_preds.csv (produced by test_singlemode.py).
#    Structure: TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/
TEST_EVAL_DIR = (
    "/data/Irene/SwinTransformer/Swin_Meta/outputs/test_evaluation/"
    "swin_tiny/OCTA3/"
    "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6/"
    "Best_fold2"
)
# OCT0: BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9
# OCT1: BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8
# OCTA3: BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6

# 2. Checkpoint root (same as CHECKPOINT_ROOT in test_singlemode.py).
#    Leave "" to auto-detect as PROJECT_ROOT/checkpoints.
CHECKPOINT_ROOT = ""

# 3. Model switch -- enable one backbone at a time.
#    When adding VGG16 / EfficientNet, add entries to TIMM_MODEL_MAP and
#    _get_target_layer() only.
ACTIVE_MODEL = "swin_tiny"   # "swin_tiny" | "vgg16" | "efficientnet_b0"

TIMM_MODEL_MAP: Dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    # "vgg16":           "vgg16",
    # "efficientnet_b0": "efficientnet_b0",
}

# 4. Visual settings
IMG_SIZE       = 224     # must match training
OVERLAY_ALPHA  = 0.50    # 0 = original only, 1 = heatmap only
COLORMAP       = "jet"   # matplotlib colormap for heatmap
CLASS_NAMES    = ["inactive", "active"]

# CAM post-processing applied at 7x7 feature-map resolution (before upsample).
# Pixels below CAM_THRESHOLD * max are zeroed out.
# Suppresses low-contribution background noise (black borders, artefacts).
# Set 0.0 to disable.  Ref: common in medical imaging Grad-CAM papers.
CAM_THRESHOLD  = 0.30

# 5. Random seed.
#    None = time-based (different sample every run).
#    Integer (e.g. 42) = fixed, reproducible selection.
N_RANDOM_SEED = None

# Output figure style
_TITLE_SIZE      = 14
_AXIS_LABEL_SIZE = 11
_FIG_DPI         = 220

# ==============================================================================


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def log_print(logf, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if logf:
        logf.write(line + "\n")
        logf.flush()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def resolve_ckpt_root(cfg: str, project_root: Path) -> str:
    return cfg.strip() if cfg.strip() else str(project_root / "checkpoints")


def parse_test_eval_dir(test_eval_dir: str) -> dict:
    """
    Parse model_name, modality, run_tag, best_fold, project_root
    from the TEST_EVAL_DIR path.

    Required structure:
      PROJECT_ROOT/outputs/test_evaluation/<model>/<modality>/<run_tag>/Best_fold{N}
    """
    p = Path(test_eval_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"TEST_EVAL_DIR not found: {test_eval_dir}")

    leaf = p.name
    if not leaf.startswith("Best_fold"):
        raise ValueError(f"TEST_EVAL_DIR must end with 'Best_fold{{N}}', got: {leaf}")
    best_fold = int(leaf.replace("Best_fold", ""))

    run_tag      = p.parents[0].name   # <run_tag>
    modality     = p.parents[1].name   # e.g. OCT0
    model_name   = p.parents[2].name   # e.g. swin_tiny
    project_root = p.parents[5]        # PROJECT_ROOT

    return {
        "best_fold":    best_fold,
        "run_tag":      run_tag,
        "modality":     modality,
        "model_name":   model_name,
        "project_root": project_root,
    }


# ------------------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    """Build timm model with num_classes=1 (BCEWithLogits head)."""
    if model_name not in TIMM_MODEL_MAP:
        raise NotImplementedError(
            f"model_name='{model_name}' not in TIMM_MODEL_MAP. "
            f"Registered: {list(TIMM_MODEL_MAP.keys())}"
        )
    return timm.create_model(TIMM_MODEL_MAP[model_name], pretrained=False, num_classes=1)


def load_checkpoint_and_temperature(
    model: nn.Module, ckpt_path: str, device: torch.device,
) -> Tuple[nn.Module, float]:
    """
    Load model_best.pth saved by train_singlemode_oof.py.
    Returns (model, temperature).  temperature defaults to 1.0 if not stored.
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        state_dict  = raw.get("model_state_dict", raw.get("state_dict", raw))
        temperature = float(raw.get("temperature", 1.0))
    else:
        state_dict  = raw
        temperature = 1.0
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, temperature


# ------------------------------------------------------------------------------
# Grad-CAM (manual hook, no external library dependency)
# ------------------------------------------------------------------------------

def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the Grad-CAM hook target layer.

    Swin-Tiny: model.norm  (Global LayerNorm, after all 4 stages)
      norm1 captures intra-window LOCAL features -> fragmented heatmap.
      model.norm integrates features GLOBALLY before the classifier head
      -> coherent, semantically meaningful heatmap focused on lesion regions.
      Output shape (timm >= 0.6): [B, H, W, C] = [B, 7, 7, 768]
      Output shape (timm <  0.6): [B, N, C]    = [B, 49, 768]
      Both handled by _reshape_swin_tokens().
    """
    if model_name == "swin_tiny":
        return model.norm
    raise NotImplementedError(
        f"Target layer not defined for '{model_name}'. "
        "Add an entry to _get_target_layer()."
    )


def _reshape_swin_tokens(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert Swin hook output to [B, C, H_feat, W_feat].

    timm >= 0.6 outputs [B, H, W, C] (NHWC, 4D).
    timm <  0.6 outputs [B, N, C]    (flat tokens, 3D).
    """
    if tensor.ndim == 4:
        # [B, H, W, C] -> [B, C, H, W]
        return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.ndim == 3:
        # [B, N, C] -> [B, C, sqrt(N), sqrt(N)]
        B, N, C = tensor.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"N={N} is not a perfect square"
        return tensor.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        f"Unexpected tensor ndim={tensor.ndim}, shape={tensor.shape}. "
        "Expected 3D [B,N,C] or 4D [B,H,W,C]."
    )


class SwinGradCAM:
    """
    Grad-CAM for timm Swin-Tiny binary (BCEWithLogits) model.

    Grad-CAM formula (Selvaraju et al. 2017):
      alpha_k  = (1/Z) sum_{i,j} dY/dA_k_{ij}   [channel importance weight]
      L        = ReLU( sum_k alpha_k * A_k )      [class activation map]

    Binary scoring:
      class=1 (active)   -> score = logit
      class=0 (inactive) -> score = -logit
    """

    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self.model      = model
        self.model_name = model_name
        self.device     = device
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        target = _get_target_layer(model, model_name)
        self._fwd_hook = target.register_forward_hook(self._save_activation)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        if grad_out and grad_out[0] is not None:
            self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx: int,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate Grad-CAM heatmap for class_idx.

        Returns
        -------
        cam : np.ndarray  [H, W]  values in [0, 1]
            Pipeline: Grad-CAM [7x7] -> normalize -> threshold(CAM_THRESHOLD)
                      -> bicubic upsample [224x224] -> clip
        prob_active : float  sigmoid(logit / T*)
        """
        self.model.eval()
        self.model.zero_grad()
        self._activations = None
        self._gradients   = None

        logit       = self.model(img_tensor)                         # [1, 1]
        logit_calib = logit / max(temperature, 1e-6)
        prob_active = float(torch.sigmoid(logit_calib[0, 0]).item())

        score = logit[0, 0] if class_idx == 1 else -logit[0, 0]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture data. "
                "Check that the target layer is correct."
            )

        A = _reshape_swin_tokens(self._activations)   # [1, C, 7, 7]
        G = _reshape_swin_tokens(self._gradients)     # [1, C, 7, 7]

        # Eq.1: alpha_k = (1/Z) sum_{i,j} dY/dA_k_{ij}
        weights = G.mean(dim=(2, 3), keepdim=True)           # [1, C, 1, 1]
        # Eq.2: L = ReLU( sum_k alpha_k * A_k )
        cam     = (weights * A).sum(dim=1, keepdim=True)      # [1, 1, 7, 7]
        cam     = F.relu(cam)

        # Normalise to [0, 1] at 7x7 feature-map resolution
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Threshold: zero out pixels below CAM_THRESHOLD * max.
        # Applied at 7x7 before upsampling so low-contribution background
        # regions (black borders, specular artefacts) are suppressed.
        if CAM_THRESHOLD > 0.0:
            cam = cam * (cam >= CAM_THRESHOLD).float()

        # Bicubic upsample 7x7 -> 224x224.
        # Provides smoother transitions than bilinear for 32x upscaling.
        cam_up = F.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE),
            mode="bicubic", align_corners=False,
        )[0, 0].cpu().numpy()
        cam_up = np.clip(cam_up, 0.0, 1.0)

        return cam_up, prob_active


# ------------------------------------------------------------------------------
# Image transform  (must match test_singlemode.py)
# ------------------------------------------------------------------------------

def get_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------------------
# Overlay
# ------------------------------------------------------------------------------

def overlay_heatmap(
    orig_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    colormap: str = COLORMAP,
) -> np.ndarray:
    """Blend original image and Grad-CAM heatmap. Returns [H, W, 3] float32."""
    img_np  = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE),
                                        Image.BILINEAR).convert("RGB"),
                       dtype=np.float32) / 255.0
    heatmap = plt.get_cmap(colormap)(cam)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * img_np + alpha * heatmap, 0.0, 1.0)


# ------------------------------------------------------------------------------
# CSV record
# ------------------------------------------------------------------------------

def save_samples_csv(
    samples: Dict[str, Optional[dict]],
    out_path: str,
    run_meta: dict,
) -> None:
    """
    Save per-sample Grad-CAM record to CSV.

    Columns: run_timestamp, model_name, modality, run_tag, best_fold,
             temperature, target_layer,
             quadrant, exam_key, filename, image_path,
             gt_label, gt_class, pred_label, pred_class,
             prob_active, prob_inactive, logit_uncal, logit_calib,
             cam1_max, cam0_max, panel_file, note
    """
    rows = []
    for quadrant, data in samples.items():
        base = {
            "run_timestamp": run_meta["timestamp"],
            "model_name":    run_meta["model_name"],
            "modality":      run_meta["modality"],
            "run_tag":       run_meta["run_tag"],
            "best_fold":     run_meta["best_fold"],
            "temperature":   run_meta["temperature"],
            "target_layer":  run_meta["target_layer"],
            "quadrant":      quadrant,
        }
        if data is None:
            base.update({
                "exam_key": "", "filename": "", "image_path": "",
                "gt_label": "", "gt_class": "",
                "pred_label": "", "pred_class": "",
                "prob_active": "", "prob_inactive": "",
                "logit_uncal": "", "logit_calib": "",
                "cam1_max": "", "cam0_max": "",
                "panel_file": f"(no {quadrant} sample)",
                "note": "no sample in this quadrant",
            })
        else:
            base.update({
                "exam_key":    data["exam_key"],
                "filename":    data["filename"],
                "image_path":  data["img_path"],
                "gt_label":    data["gt"],
                "gt_class":    CLASS_NAMES[data["gt"]],
                "pred_label":  data["pred"],
                "pred_class":  CLASS_NAMES[data["pred"]],
                "prob_active":   round(data["prob_active"],   6),
                "prob_inactive": round(1.0 - data["prob_active"], 6),
                "logit_uncal":   round(data["logit_uncal"],   6),
                "logit_calib":   round(data["logit_calib"],   6),
                "cam1_max":  round(float(data["cam1"].max()), 4),
                "cam0_max":  round(float(data["cam0"].max()), 4),
                "panel_file": f"{quadrant}_panel.png",
                "note": "",
            })
        rows.append(base)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------------------

def _set_style() -> None:
    plt.rcParams.update({
        "font.size":       10,
        "axes.titlesize":  _TITLE_SIZE,
        "axes.labelsize":  _AXIS_LABEL_SIZE,
        "figure.dpi":      _FIG_DPI,
        "savefig.dpi":     _FIG_DPI,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def save_single_panel(
    orig_pil: Image.Image,
    cam0: np.ndarray, cam1: np.ndarray,
    quadrant: str, gt: int, pred: int, prob_active: float,
    out_path: str,
) -> None:
    """1x3: original | Grad-CAM class-0 | Grad-CAM class-1"""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(
        f"{quadrant}  |  GT={CLASS_NAMES[gt]}  Pred={CLASS_NAMES[pred]}"
        f"  P(active)={prob_active:.3f}",
        fontsize=_TITLE_SIZE, fontweight="bold",
    )
    axes[0].imshow(np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")))
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(overlay_heatmap(orig_pil, cam0))
    axes[1].set_title("Grad-CAM\nClass 0 (inactive)", fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(overlay_heatmap(orig_pil, cam1))
    axes[2].set_title("Grad-CAM\nClass 1 (active)", fontweight="bold")
    axes[2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_4x3_panel(
    samples: Dict[str, Optional[dict]],
    out_path: str, modality: str, model_name: str,
) -> None:
    """4x3: rows=TP/FP/FN/TN, cols=original/class-0/class-1"""
    _set_style()
    row_keys = ["TP", "FP", "FN", "TN"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(
        f"Grad-CAM -- {model_name.upper()} / {modality}  (Test Set)",
        fontsize=_TITLE_SIZE + 2, fontweight="bold",
    )
    for c, ct in enumerate(["Original",
                             "Grad-CAM: Class 0 (inactive)",
                             "Grad-CAM: Class 1 (active)"]):
        axes[0, c].set_title(ct, fontsize=_TITLE_SIZE, fontweight="bold", pad=6)

    for r, key in enumerate(row_keys):
        data = samples.get(key)
        for c in range(3):
            ax = axes[r, c]
            ax.axis("off")
            if data is None:
                ax.text(0.5, 0.5, f"No {key} sample",
                        ha="center", va="center", fontsize=10,
                        transform=ax.transAxes)
                continue
            orig  = data["orig_pil"]
            cam0  = data["cam0"]
            cam1  = data["cam1"]
            gt    = data["gt"]
            pred  = data["pred"]
            p_act = data["prob_active"]
            fname = data["filename"]
            if c == 0:
                ax.imshow(np.array(orig.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")))
                ax.set_title(
                    f"{key}  GT={CLASS_NAMES[gt]}  Pred={CLASS_NAMES[pred]}\n"
                    f"P(active)={p_act:.3f}  {fname}",
                    fontsize=9, fontweight="bold", pad=4,
                )
            elif c == 1:
                ax.imshow(overlay_heatmap(orig, cam0))
            else:
                ax.imshow(overlay_heatmap(orig, cam1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    # Seed: None = time-based (different sample every run)
    if N_RANDOM_SEED is None:
        random.seed(int(time.time() * 1000) % (2 ** 31))
    else:
        random.seed(N_RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 0: Parse paths
    parsed       = parse_test_eval_dir(TEST_EVAL_DIR)
    model_name   = ACTIVE_MODEL
    modality     = parsed["modality"]
    run_tag      = parsed["run_tag"]
    best_fold    = parsed["best_fold"]
    project_root = parsed["project_root"]

    ckpt_root = resolve_ckpt_root(CHECKPOINT_ROOT, project_root)
    ckpt_path = os.path.join(
        ckpt_root, model_name, modality, run_tag,
        f"Best_fold{best_fold}", "model_best.pth",
    )
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(
            ckpt_root, model_name, modality, run_tag,
            "Kfold", f"fold{best_fold}", "model_best.pth",
        )

    preds_csv = os.path.join(TEST_EVAL_DIR, "test_preds.csv")
    ts        = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(TEST_EVAL_DIR, "gradcam", ts)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "gradcam_log.txt"),
                "w", buffering=1, encoding="utf-8")

    log_print(logf, "=" * 62)
    log_print(logf, "GRAD-CAM  SINGLE-MODALITY  mCNV CLASSIFICATION")
    log_print(logf, "=" * 62)
    log_print(logf, f"model      : {model_name}  ({TIMM_MODEL_MAP[model_name]})")
    log_print(logf, f"modality   : {modality}")
    log_print(logf, f"run_tag    : {run_tag}")
    log_print(logf, f"best_fold  : {best_fold}")
    log_print(logf, f"device     : {device}")
    log_print(logf, f"checkpoint : {ckpt_path}")
    log_print(logf, f"preds_csv  : {preds_csv}")
    log_print(logf, f"out_dir    : {out_dir}")
    log_print(logf, f"target_layer : model.norm  (Global LayerNorm, all 4 stages)")
    log_print(logf, f"cam_threshold: {CAM_THRESHOLD}  (zero pixels < threshold*max)")
    log_print(logf, f"random_seed  : {N_RANDOM_SEED} (None=time-based, diff every run)")
    log_print(logf, f"reference    : Selvaraju et al. ICCV 2017 (Grad-CAM)")

    # Step 1: Load test_preds.csv
    log_print(logf, "-" * 62)
    log_print(logf, "Step 1: Load test_preds.csv")

    if not os.path.isfile(preds_csv):
        raise FileNotFoundError(
            f"test_preds.csv not found:\n  {preds_csv}\n"
            "Run test_singlemode.py first."
        )

    df = pd.read_csv(preds_csv)
    required = {"exam_key", "y_true", "logit_uncal", "logit_calib",
                 "prob_calib", "temperature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test_preds.csv missing columns: {missing}")

    df["y_true"]       = df["y_true"].astype(int)
    df["y_pred_calib"] = (df["prob_calib"] >= 0.5).astype(int)
    temperature = float(df["temperature"].iloc[0])

    log_print(logf, f"Loaded {len(df)} samples  temperature T*={temperature:.6f}")
    log_print(logf, "Using calibrated predictions (prob_calib, threshold=0.5)")

    # Step 2: Group by confusion-matrix quadrant
    log_print(logf, "-" * 62)
    log_print(logf, "Step 2: Group by confusion-matrix quadrant")

    def _idx(mask):
        return df.index[mask].tolist()

    idx_TP = _idx((df["y_true"] == 1) & (df["y_pred_calib"] == 1))
    idx_FP = _idx((df["y_true"] == 0) & (df["y_pred_calib"] == 1))
    idx_FN = _idx((df["y_true"] == 1) & (df["y_pred_calib"] == 0))
    idx_TN = _idx((df["y_true"] == 0) & (df["y_pred_calib"] == 0))

    log_print(logf, f"TP={len(idx_TP)}  FP={len(idx_FP)}  "
                    f"FN={len(idx_FN)}  TN={len(idx_TN)}")

    selected = {
        "TP": random.choice(idx_TP) if idx_TP else None,
        "FP": random.choice(idx_FP) if idx_FP else None,
        "FN": random.choice(idx_FN) if idx_FN else None,
        "TN": random.choice(idx_TN) if idx_TN else None,
    }
    for key, idx in selected.items():
        if idx is not None:
            row = df.loc[idx]
            log_print(logf, f"  {key}: exam_key={row['exam_key']}"
                            f"  gt={int(row['y_true'])}"
                            f"  pred={int(row['y_pred_calib'])}"
                            f"  prob_calib={float(row['prob_calib']):.4f}")
        else:
            log_print(logf, f"  {key}: no sample available")

    # Step 3: Build model + load checkpoint
    log_print(logf, "-" * 62)
    log_print(logf, "Step 3: Build model and load checkpoint")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"model_best.pth not found:\n  {ckpt_path}\n"
            "Check CHECKPOINT_ROOT."
        )

    model, ckpt_temperature = load_checkpoint_and_temperature(
        build_model(model_name), ckpt_path, device
    )
    log_print(logf, f"Checkpoint T*={ckpt_temperature:.6f}  "
                    f"(using CSV T*={temperature:.6f})")

    target_layer_name = "model.norm"
    gradcam = SwinGradCAM(model, model_name, device)
    log_print(logf, f"Grad-CAM engine ready  target={target_layer_name}")

    tfm = get_test_transform()

    # Step 4: Generate Grad-CAM
    log_print(logf, "-" * 62)
    log_print(logf, "Step 4: Generate Grad-CAM heatmaps")

    # Modality -> image path column in master_manifest.csv
    MODALITY_IMG_COL = {
        "OCT0":  "oct0_image_path",
        "OCT1":  "oct1_image_path",
        "OCTA3": "octa3_image_path",
    }
    img_col = MODALITY_IMG_COL[modality]

    # Cache manifest if needed (loaded once for all quadrants)
    _manifest_df: Optional[pd.DataFrame] = None

    def _get_img_path(row) -> Optional[str]:
        nonlocal _manifest_df
        # Attempt 1: image path column already in test_preds.csv
        if img_col in df.columns:
            return str(row[img_col])
        # Attempt 2: look up master_manifest.csv
        if _manifest_df is None:
            candidates = [
                project_root / "outputs" / "manifests" / "master_split" / "master_manifest.csv",
                project_root / "outputs" / "manifests" / "master_manifest.csv",
                project_root / "data_splits" / "master_manifest.csv",
            ]
            for c in candidates:
                if c.is_file():
                    _manifest_df = pd.read_csv(str(c), usecols=["exam_key", img_col])
                    _manifest_df["exam_key"] = _manifest_df["exam_key"].astype(str)
                    break
        if _manifest_df is None:
            return None
        match = _manifest_df[_manifest_df["exam_key"] == str(row["exam_key"])]
        return str(match.iloc[0][img_col]) if not match.empty else None

    samples_for_panel: Dict[str, Optional[dict]] = {}

    for quadrant, row_idx in selected.items():
        if row_idx is None:
            log_print(logf, f"  [{quadrant}] skipped -- no sample")
            samples_for_panel[quadrant] = None
            continue

        row      = df.loc[row_idx]
        gt       = int(row["y_true"])
        pred     = int(row["y_pred_calib"])

        img_path = _get_img_path(row)
        if img_path is None:
            log_print(logf, f"  [{quadrant}] WARN: cannot locate master_manifest.csv -- skipping")
            samples_for_panel[quadrant] = None
            continue
        if not os.path.isfile(img_path):
            log_print(logf, f"  [{quadrant}] WARN: image not found: {img_path} -- skipping")
            samples_for_panel[quadrant] = None
            continue

        orig_pil   = Image.open(img_path).convert("RGB")
        img_tensor = tfm(orig_pil).unsqueeze(0).to(device)

        cam0, _          = gradcam.generate(img_tensor, class_idx=0, temperature=temperature)
        cam1, prob_active = gradcam.generate(img_tensor, class_idx=1, temperature=temperature)

        samples_for_panel[quadrant] = {
            "orig_pil":    orig_pil,
            "cam0":        cam0,
            "cam1":        cam1,
            "gt":          gt,
            "pred":        pred,
            "prob_active": prob_active,
            "filename":    os.path.basename(img_path),
            "img_path":    img_path,
            "exam_key":    str(row["exam_key"]),
            "logit_uncal": float(row.get("logit_uncal", float("nan"))),
            "logit_calib": float(row.get("logit_calib", float("nan"))),
        }

        log_print(logf, f"  [{quadrant}] {os.path.basename(img_path)}"
                        f"  gt={CLASS_NAMES[gt]}  pred={CLASS_NAMES[pred]}"
                        f"  P(active)={prob_active:.4f}"
                        f"  cam0_max={cam0.max():.3f}  cam1_max={cam1.max():.3f}")

        panel_path = os.path.join(out_dir, f"{quadrant}_panel.png")
        save_single_panel(orig_pil, cam0, cam1, quadrant,
                          gt, pred, prob_active, panel_path)
        log_print(logf, f"    saved -> {panel_path}")

    gradcam.remove_hooks()

    # Step 5: Save combined 4x3 panel
    log_print(logf, "-" * 62)
    log_print(logf, "Step 5: Save combined 4x3 panel")
    big_path = os.path.join(out_dir, "GradCAM_4x3_panel.png")
    save_4x3_panel(samples_for_panel, big_path, modality, model_name)
    log_print(logf, f"Saved -> {big_path}")

    # Step 6: Save CSV record
    log_print(logf, "-" * 62)
    log_print(logf, "Step 6: Save CSV record")
    run_meta = {
        "timestamp":    ts,
        "model_name":   model_name,
        "modality":     modality,
        "run_tag":      run_tag,
        "best_fold":    best_fold,
        "temperature":  temperature,
        "target_layer": target_layer_name,
    }
    csv_path = os.path.join(out_dir, "gradcam_samples.csv")
    save_samples_csv(samples_for_panel, csv_path, run_meta)
    log_print(logf, f"Saved -> {csv_path}")

    # Done
    log_print(logf, "=" * 62)
    log_print(logf, "GRAD-CAM COMPLETE")
    log_print(logf, f"  Output: {out_dir}")
    log_print(logf, "  Files : GradCAM_4x3_panel.png")
    for q in ["TP", "FP", "FN", "TN"]:
        if samples_for_panel.get(q) is not None:
            log_print(logf, f"          {q}_panel.png")
    log_print(logf, "          gradcam_samples.csv")
    log_print(logf, "          gradcam_log.txt")
    log_print(logf, "=" * 62)
    logf.close()


if __name__ == "__main__":
    main()