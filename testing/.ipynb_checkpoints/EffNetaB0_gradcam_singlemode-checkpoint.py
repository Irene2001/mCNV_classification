# EffNetaB0_gradcam_singlemode.py

"""
EfficientNetB0 Grad-CAM visualisation for the mCNV single-modality base model.

Mirrors the output format of VGG16_gradcam_singlemode.py exactly:
  GradCAM_4x3_panel.png   4×3: original | Grad-CAM class-0 | Grad-CAM class-1
  TP_panel.png  FP_panel.png  FN_panel.png  TN_panel.png
  gradcam_samples.csv     per-sample record (quadrant, filename, GT, Pred, ...)
  gradcam_log.txt

Algorithm (Grad-CAM for EfficientNetB0, binary BCEWithLogits)
--------------------------------------------------------------
EfficientNetB0 feature map structure (timm, 224×224 input):
  conv_stem              : 3×3 stride-2 conv  -> [B, 32,  112, 112]
  bn1                    :
  blocks[0]  MBConv1 3×3 × 1  -> [B, 16,  112, 112]  (stage 1)
  blocks[1]  MBConv6 3×3 × 2  -> [B, 24,   56,  56]  (stage 2)
  blocks[2]  MBConv6 5×5 × 2  -> [B, 40,   28,  28]  (stage 3)
  blocks[3]  MBConv6 3×3 × 3  -> [B, 80,   14,  14]  (stage 4)
  blocks[4]  MBConv6 5×5 × 3  -> [B, 112,  14,  14]  (stage 5)  ← FROZEN (Partial_B4_6)
  blocks[5]  MBConv6 5×5 × 4  -> [B, 192,   7,   7]  (stage 6)  ← TRAINABLE
  blocks[6]  MBConv6 3×3 × 1  -> [B, 320,   7,   7]  (stage 7)  ← TRAINABLE
  conv_head              : 1×1 -> [B, 1280,  7,   7]             ← TRAINABLE
  bn2 + act2             :
  global_avg_pool        : [B, 1280]
  classifier             : Linear(1280, 1)                        ← TRAINABLE

Target layer for Grad-CAM: model.blocks[5]
  blocks[5] is the last MBConv block that still has spatial resolution > 1×1
  before the classification head. It outputs [B, 192, 7, 7], the highest-level
  spatial feature map before global pooling collapses spatial information.

  Alternative: model.conv_head  → [B, 1280, 7, 7]  (projection layer)
  Both are standard choices for EfficientNet Grad-CAM.
  We use blocks[5] (last MBConv body block) to capture richer semantic activation.

  Note on timm EfficientNetB0 block indexing:
    blocks is nn.Sequential of 7 MBConv stages (indices 0..6).
    The Grad-CAM hook targets the LAST Sub-module of blocks[5] to capture
    the output feature map at 7×7 resolution.

Grad-CAM formula (Selvaraju et al. 2017, Eq. 1-2):
  alpha_k  = (1/Z) Σ_{i,j} dY^c / dA^k_{ij}   (global avg of gradients)
  L^c      = ReLU( Σ_k alpha_k × A^k )          (weighted activation map)

Binary BCEWithLogits scoring:
  class=1 (active)   -> score = logit
  class=0 (inactive) -> score = -logit

CAM post-processing (applied at 7×7 before upsample):
  1. Normalize to [0, 1].
  2. Zero pixels below CAM_THRESHOLD × max  (suppress background noise).
  3. Bicubic upsample 7×7 -> 224×224 (32× upscaling).
  4. Clip to [0, 1].

EfficientNetB0 model loading:
  Checkpoint saved by correct_EffNetB0_train_singlemode_oof.py:
    {model_state_dict: ..., temperature: T*, unfreeze_mode: ..., drop_rate: ..., ...}
  Model built via EffNetB0_model_factory.create_model():
    timm.create_model("efficientnet_b0", num_classes=1, drop_rate=0.2)

Unfreeze strategy (mirrors correct_EffNetB0_train_singlemode_oof.py):
  EFFNET_FROZEN_BLOCK_INDICES    = [0, 1, 2, 3, 4]
  EFFNET_TRAINABLE_BLOCK_INDICES = [5, 6]
  Grad-CAM target: blocks[5] — last body block in the trainable range.
  This is the deepest spatially-resolved trained feature map before global pooling.
"""

import os
import sys
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FACTORY_DIR = "/data/Irene/SwinTransformer/Swin_Meta/training"

if FACTORY_DIR not in sys.path:
    sys.path.append(FACTORY_DIR)
    
try:
    from EffNetB0_model_factory import (
        create_model,
        normalize_model_name,
        get_backbone_name,
    )
except ImportError:
    try:
        from training.EffNetB0_model_factory import (
            create_model,
            normalize_model_name,
            get_backbone_name,
        )
    except ImportError:
        from model_factory import (
            create_model,
            normalize_model_name,
            get_backbone_name,
        )


# ==============================================================================
# CONFIG  --  Edit only this section
# ==============================================================================

# 1. Directory that contains test_preds.csv (produced by EffNetB0_test_singlemode.py).
#    Structure: TEST_EVAL_ROOT/<model>/<strategy>/<modality>/<run_tag>/Best_fold{N}/
TEST_EVAL_DIR = (
    "/data/Irene/SwinTransformer/Swin_Meta/EffNetB0_outputs/test_evaluation/"
    "efficientnet_b0/Partial_B5_6/OCTA3/"
    "BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.13_0.87_2_WSon_1_2.6/"
    "Best_fold2"
)

# OCT0: BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.11_0.89_2_WSon_1_2.9 (Best_fold2)
# OCT1: BS16_EP100_LR2e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.113_0.887_2_WSon_1_2.8 (Best_fold4)
# OCTA3: BS16_EP100_LR3e-05_WD0.01_PARTIAL_FINETUNE_DR0.2_FL0.13_0.87_2_WSon_1_2.6 (Best_fold2)

# 2. Checkpoint root (same as CHECKPOINT_ROOT in correct_EffNetB0_train_singlemode_oof.py).
#    Leave "" to auto-detect as PROJECT_ROOT/checkpoints (shared) OR
#    EffNetB0_outputs/checkpoints (recommended, isolated).
CHECKPOINT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta/checkpoints"

# 3. Model name (fixed for this script).
ACTIVE_MODEL = "efficientnet_b0"

# 4. Visual settings
IMG_SIZE      = 224
OVERLAY_ALPHA = 0.50
COLORMAP      = "jet"
CLASS_NAMES   = ["inactive", "active"]

# CAM post-processing threshold applied at 7×7 feature-map resolution (before upsample).
# Pixels below CAM_THRESHOLD × max are zeroed to suppress low-contribution background.
# Set 0.0 to disable.  0.20-0.30 is a reasonable starting point for EffNetB0.
CAM_THRESHOLD = 0.0

# 5. Random seed.
#    None = time-based (different sample every run).
#    Integer = fixed, reproducible.
N_RANDOM_SEED = None

# drop_rate used during training (must match correct_EffNetB0_train_singlemode_oof.py).
DROP_RATE = 0.2

# Output figure style
_TITLE_SIZE      = 14
_AXIS_LABEL_SIZE = 11
_FIG_DPI         = 220

# ==============================================================================


# EfficientNetB0 Grad-CAM target: blocks[5] last sub-module.
# blocks[5] = MBConv6 5×5 ×4, outputs [B, 192, 7, 7].
# This is the deepest body block in the trainable range (EFFNET_TRAINABLE_BLOCK_INDICES=[4,5,6])
# that has spatial resolution > 1×1. Captures highest-level semantic features
# before global average pooling collapses spatial structure.
_EFFNET_TARGET_BLOCK_IDX = 5     # model.blocks[5]
_EFFNET_FEAT_MAP_SIZE    = 7     # spatial resolution at blocks[5]: 7×7
_EFFNET_FEAT_CHANNELS    = 192   # output channels of blocks[5]

# Frozen / trainable blocks matching correct_EffNetB0_train_singlemode_oof.py
_EFFNET_FROZEN_BLOCK_INDICES    = [0, 1, 2, 3, 4]
_EFFNET_TRAINABLE_BLOCK_INDICES = [5, 6]


# ------------------------------------------------------------------------------
# Utilities  (UNCHANGED from VGG16 version)
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
    if cfg.strip():
        return cfg.strip()
    # EffNetB0 checkpoints are in EffNetB0_outputs/checkpoints (isolated)
    effnet_base = project_root / "EffNetB0_outputs"
    effnet_ckpt = effnet_base / "checkpoints"
    if effnet_ckpt.exists():
        return str(effnet_ckpt)
    # Fallback: shared checkpoints directory
    return str(project_root / "checkpoints")


def parse_test_eval_dir(test_eval_dir: str) -> dict:
    """
    Parse model_name, strategy, modality, run_tag, best_fold, project_root
    from TEST_EVAL_DIR path.

    Required structure:
      <EffNetB0_outputs>/test_evaluation/<model>/<strategy>/<modality>/<run_tag>/Best_fold{N}

    Path index from leaf (Best_fold{N}):
      parents[0] = run_tag
      parents[1] = modality
      parents[2] = strategy       e.g. Partial_B4_6
      parents[3] = model_name     e.g. efficientnet_b0
      parents[4] = test_evaluation
      parents[5] = EffNetB0_outputs
      parents[6] = Swin_Meta  (PROJECT_ROOT)
    """
    p = Path(test_eval_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"TEST_EVAL_DIR not found: {test_eval_dir}")

    leaf = p.name
    if not leaf.startswith("Best_fold"):
        raise ValueError(
            f"TEST_EVAL_DIR must end with 'Best_fold{{N}}', got: {leaf}"
        )
    best_fold  = int(leaf.replace("Best_fold", ""))
    run_tag    = p.parents[0].name
    modality   = p.parents[1].name
    strategy   = p.parents[2].name
    model_name = p.parents[3].name

    # Find project_root by locating test_evaluation anchor
    anchor = None
    for parent in p.parents:
        if parent.name == "test_evaluation":
            anchor = parent
            break
    if anchor is None:
        raise ValueError(
            "Could not find 'test_evaluation' in TEST_EVAL_DIR path.\n"
            f"Path: {test_eval_dir}"
        )
    # EffNetB0_outputs -> project_root
    project_root = anchor.parent.parent

    return {
        "best_fold":    best_fold,
        "run_tag":      run_tag,
        "modality":     modality,
        "strategy":     strategy,
        "model_name":   model_name,
        "project_root": project_root,
    }


# ------------------------------------------------------------------------------
# Model builder for EfficientNetB0
# ------------------------------------------------------------------------------

def _build_effnetb0_for_inference(drop_rate: float = 0.2) -> nn.Module:
    """
    Build EfficientNetB0 via EffNetB0_model_factory with inplace=False on all
    activations to support Grad-CAM backward hooks.

    Architecture matches correct_EffNetB0_train_singlemode_oof.py:
      timm.create_model("efficientnet_b0", num_classes=1, drop_rate=0.2)
    """
    model = create_model(
        model_name="efficientnet_b0",
        num_classes=1,
        pretrained=False,   # weights loaded from checkpoint
        drop_rate=drop_rate,
    )

    # Disable all inplace operations to prevent Grad-CAM hook conflicts
    # (same fix applied in VGG16_gradcam_singlemode.py)
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False

    return model


def load_checkpoint_and_temperature(
    model: nn.Module, ckpt_path: str, device: torch.device,
) -> Tuple[nn.Module, float]:
    """
    Load model_best.pth saved by correct_EffNetB0_train_singlemode_oof.py.
    Returns (model, temperature).

    Checkpoint dict keys:
      model_state_dict | temperature | val_nll_uncal | val_acc | val_auc
      nll_beforeTS | nll_afterTS | fold | model_name | modality
      unfreeze_mode | drop_rate
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
# Grad-CAM for EfficientNetB0 (timm, manual hook)
# ------------------------------------------------------------------------------

class EffNetB0GradCAM:
    """
    Grad-CAM for timm EfficientNetB0 binary (BCEWithLogits) model.

    Target layer: model.blocks[5]  (last sub-module of MBConv stage 6)
      Output shape: [B, 192, 7, 7]  — deepest spatially-resolved trained feature map.

    Why blocks[5] and not conv_head?
      conv_head (1×1 projection, 1280ch) compresses channel information without
      adding spatial selectivity. blocks[5] retains richer per-channel semantics
      from the full MBConv body (depthwise conv + SE attention), making the
      resulting heatmap more interpretable for mCNV pathology localisation.

    Grad-CAM formula (Selvaraju et al. 2017, Eq. 1-2):
      alpha_k  = (1/Z) Σ_{i,j} dY^c / dA^k_{ij}
      L^c      = ReLU( Σ_k alpha_k × A^k )

    Binary BCEWithLogits scoring:
      class=1 (active)   -> score = logit
      class=0 (inactive) -> score = -logit

    CAM pipeline:
      1. Grad-CAM at 7×7  (blocks[5] output resolution).
      2. Normalize to [0, 1].
      3. Threshold: zero pixels < CAM_THRESHOLD × max.
      4. Bicubic upsample 7 -> 224  (32× upscaling).
      5. Clip to [0, 1].

    Reference: Selvaraju et al. (2017). ICCV.
      https://doi.org/10.1109/ICCV.2017.74

    EfficientNetB0 GitHub reference (Grad-CAM on blocks):
      https://github.com/jacobgil/pytorch-grad-cam
      (recommends last feature block before global pooling as target layer)
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model
        self.device = device

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Hook on the last sub-module of blocks[5]
        # blocks[5] is nn.Sequential of MBConv sub-layers;
        # hooking on the whole block captures the final output feature map.
        target = model.blocks[_EFFNET_TARGET_BLOCK_IDX]
        self._fwd_hook = target.register_forward_hook(self._save_activation)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        # out: [B, 192, 7, 7] for blocks[5]
        self._activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        if grad_out and grad_out[0] is not None:
            # grad_out[0]: [B, 192, 7, 7]
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

        Parameters
        ----------
        img_tensor : torch.Tensor  [1, 3, H, W]  (on device)
        class_idx  : int   0=inactive, 1=active
        temperature: float T* for logit calibration

        Returns
        -------
        cam_upsampled : np.ndarray  [224, 224]  values in [0, 1]
            Pipeline: Grad-CAM [7×7] -> normalize -> threshold ->
                      bicubic upsample [224×224] -> clip
        prob_active   : float  sigmoid(logit / T*)
        """
        self.model.eval()
        self.model.zero_grad()
        self._activations = None
        self._gradients   = None

        # Forward pass
        logit       = self.model(img_tensor)                          # [1, 1]
        logit_calib = logit / max(temperature, 1e-6)
        prob_active = float(torch.sigmoid(logit_calib[0, 0]).item())

        # Backward: binary scoring
        score = logit[0, 0] if class_idx == 1 else -logit[0, 0]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients.\n"
                f"Check that model.blocks[{_EFFNET_TARGET_BLOCK_IDX}] is the correct target."
            )

        # A: [1, C, 7, 7],  G: [1, C, 7, 7]
        A = self._activations
        G = self._gradients

        # Eq.1: alpha_k = (1/Z) Σ_{i,j} dY/dA^k_{ij}
        weights = G.mean(dim=(2, 3), keepdim=True)     # [1, C, 1, 1]

        # Eq.2: L = ReLU( Σ_k alpha_k * A^k )
        cam = (weights * A).sum(dim=1, keepdim=True)   # [1, 1, 7, 7]
        cam = F.relu(cam)

        # Normalize to [0, 1] at 7×7
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Threshold: suppress background noise at 7×7 resolution
        if CAM_THRESHOLD > 0.0:
            cam = cam * (cam >= CAM_THRESHOLD).float()

        # Bicubic upsample 7×7 -> 224×224 (32× upscaling)
        cam_up = F.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE),
            mode="bicubic", align_corners=False,
        )[0, 0].cpu().numpy()
        cam_up = np.clip(cam_up, 0.0, 1.0)

        return cam_up, prob_active


# ------------------------------------------------------------------------------
# Image transform  (UNCHANGED — must match training)
# ------------------------------------------------------------------------------

def get_test_transform() -> transforms.Compose:
    """ImageNet eval transform — identical to training validation transform."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------------------
# Overlay  (UNCHANGED)
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
# CSV record  (UNCHANGED from VGG16 version)
# ------------------------------------------------------------------------------

def save_samples_csv(
    samples: Dict[str, Optional[dict]],
    out_path: str,
    run_meta: dict,
) -> None:
    """Save per-sample Grad-CAM record to CSV. Format identical to VGG16 version."""
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
                "cam1_max": round(float(data["cam1"].max()), 4),
                "cam0_max": round(float(data["cam0"].max()), 4),
                "panel_file": f"{quadrant}_panel.png",
                "note": "",
            })
        rows.append(base)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------------------
# Plot helpers  (UNCHANGED from VGG16 version)
# ------------------------------------------------------------------------------

def _set_style() -> None:
    plt.rcParams.update({
        "font.size":         10,
        "axes.titlesize":    _TITLE_SIZE,
        "axes.labelsize":    _AXIS_LABEL_SIZE,
        "figure.dpi":        _FIG_DPI,
        "savefig.dpi":       _FIG_DPI,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def save_single_panel(
    orig_pil: Image.Image,
    cam0: np.ndarray, cam1: np.ndarray,
    quadrant: str, gt: int, pred: int, prob_active: float,
    out_path: str,
) -> None:
    """1×3: original | Grad-CAM class-0 | Grad-CAM class-1"""
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
    """4×3: rows=TP/FP/FN/TN, cols=original/class-0/class-1"""
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
    if N_RANDOM_SEED is None:
        random.seed(int(time.time() * 1000) % (2 ** 31))
    else:
        random.seed(N_RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 0: Parse paths
    parsed       = parse_test_eval_dir(TEST_EVAL_DIR)
    model_name   = ACTIVE_MODEL         # "efficientnet_b0"
    strategy     = parsed["strategy"]   # "Partial_B4_6"
    modality     = parsed["modality"]
    run_tag      = parsed["run_tag"]
    best_fold    = parsed["best_fold"]
    project_root = parsed["project_root"]

    ckpt_root = resolve_ckpt_root(CHECKPOINT_ROOT, project_root)
    ckpt_path = os.path.join(
        ckpt_root, model_name, strategy, modality, run_tag,
        f"Best_fold{best_fold}", "model_best.pth",
    )
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(
            ckpt_root, model_name, strategy, modality, run_tag,
            "Kfold", f"fold{best_fold}", "model_best.pth",
        )

    preds_csv = os.path.join(TEST_EVAL_DIR, "test_preds.csv")

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(TEST_EVAL_DIR, "gradcam_effnetb0", ts)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "gradcam_log.txt"),
                "w", buffering=1, encoding="utf-8")

    target_layer_name = f"model.blocks[{_EFFNET_TARGET_BLOCK_IDX}]"

    log_print(logf, "=" * 66)
    log_print(logf, "GRAD-CAM  EfficientNetB0  SINGLE-MODALITY  mCNV CLASSIFICATION")
    log_print(logf, "=" * 66)
    log_print(logf, f"model        : {model_name}  (timm efficientnet_b0)")
    log_print(logf, f"strategy     : {strategy}")
    log_print(logf, f"modality     : {modality}")
    log_print(logf, f"run_tag      : {run_tag}")
    log_print(logf, f"best_fold    : {best_fold}")
    log_print(logf, f"device       : {device}")
    log_print(logf, f"checkpoint   : {ckpt_path}")
    log_print(logf, f"preds_csv    : {preds_csv}")
    log_print(logf, f"out_dir      : {out_dir}")
    log_print(logf, f"target_layer : {target_layer_name}"
                    f"  (last MBConv body block, "
                    f"{_EFFNET_FEAT_MAP_SIZE}×{_EFFNET_FEAT_MAP_SIZE}, "
                    f"{_EFFNET_FEAT_CHANNELS}ch)")
    log_print(logf, f"cam_threshold: {CAM_THRESHOLD}"
                    f"  (zero pixels < threshold×max at "
                    f"{_EFFNET_FEAT_MAP_SIZE}×{_EFFNET_FEAT_MAP_SIZE})")
    log_print(logf, f"upsample     : bicubic "
                    f"{_EFFNET_FEAT_MAP_SIZE}×{_EFFNET_FEAT_MAP_SIZE}"
                    f" -> {IMG_SIZE}×{IMG_SIZE}  "
                    f"({IMG_SIZE // _EFFNET_FEAT_MAP_SIZE}×)")
    log_print(logf, f"drop_rate    : {DROP_RATE}  (EfficientNetB0 default)")
    log_print(logf, f"frozen_blocks: {_EFFNET_FROZEN_BLOCK_INDICES}")
    log_print(logf, f"trainable_blocks: {_EFFNET_TRAINABLE_BLOCK_INDICES}")
    log_print(logf, f"random_seed  : {N_RANDOM_SEED}  (None=time-based)")
    log_print(logf, f"reference    : Selvaraju et al. ICCV 2017 (Grad-CAM)")
    log_print(logf, f"reference    : Tan & Le ICML 2019 (EfficientNet)")
    log_print(logf, f"reference    : https://github.com/jacobgil/pytorch-grad-cam")

    # Step 1: Load test_preds.csv
    log_print(logf, "-" * 66)
    log_print(logf, "Step 1: Load test_preds.csv")

    if not os.path.isfile(preds_csv):
        raise FileNotFoundError(
            f"test_preds.csv not found:\n  {preds_csv}\n"
            "Run EffNetB0_test_singlemode.py first."
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
    log_print(logf, "-" * 66)
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
    log_print(logf, "-" * 66)
    log_print(logf, "Step 3: Build EfficientNetB0 model and load checkpoint")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"model_best.pth not found:\n  {ckpt_path}\n"
            "Check CHECKPOINT_ROOT or run EffNetB0_train_singlemode_oof.py first."
        )

    model = _build_effnetb0_for_inference(drop_rate=DROP_RATE)
    model, ckpt_temperature = load_checkpoint_and_temperature(model, ckpt_path, device)

    log_print(logf, f"Checkpoint T*={ckpt_temperature:.6f}  "
                    f"(using CSV T*={temperature:.6f})")
    log_print(logf, f"Model: timm EfficientNetB0  classifier=Linear(1280,1)")
    log_print(logf, f"  blocks total: {len(model.blocks)}")
    log_print(logf, f"  target block: blocks[{_EFFNET_TARGET_BLOCK_IDX}]"
                    f"  -> [B, {_EFFNET_FEAT_CHANNELS},"
                    f" {_EFFNET_FEAT_MAP_SIZE}, {_EFFNET_FEAT_MAP_SIZE}]")

    gradcam = EffNetB0GradCAM(model, device)
    log_print(logf, f"Grad-CAM engine ready  target={target_layer_name}")

    tfm = get_test_transform()

    # Step 4: Generate Grad-CAM
    log_print(logf, "-" * 66)
    log_print(logf, "Step 4: Generate Grad-CAM heatmaps")

    MODALITY_IMG_COL = {
        "OCT0":  "oct0_image_path",
        "OCT1":  "oct1_image_path",
        "OCTA3": "octa3_image_path",
    }
    img_col = MODALITY_IMG_COL[modality]

    _manifest_df: Optional[pd.DataFrame] = None

    def _get_img_path(row) -> Optional[str]:
        nonlocal _manifest_df
        if img_col in df.columns:
            return str(row[img_col])
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

        row  = df.loc[row_idx]
        gt   = int(row["y_true"])
        pred = int(row["y_pred_calib"])

        img_path = _get_img_path(row)
        if img_path is None:
            log_print(logf,
                f"  [{quadrant}] WARN: cannot locate master_manifest.csv -- skipping")
            samples_for_panel[quadrant] = None
            continue
        if not os.path.isfile(img_path):
            log_print(logf,
                f"  [{quadrant}] WARN: image not found: {img_path} -- skipping")
            samples_for_panel[quadrant] = None
            continue

        orig_pil   = Image.open(img_path).convert("RGB")
        img_tensor = tfm(orig_pil).unsqueeze(0).to(device)

        cam0, _           = gradcam.generate(img_tensor, class_idx=0, temperature=temperature)
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

    # Step 5: Save combined 4×3 panel
    log_print(logf, "-" * 66)
    log_print(logf, "Step 5: Save combined 4×3 panel")
    big_path = os.path.join(out_dir, "GradCAM_4x3_panel.png")
    save_4x3_panel(samples_for_panel, big_path, modality, model_name)
    log_print(logf, f"Saved -> {big_path}")

    # Step 6: Save CSV record
    log_print(logf, "-" * 66)
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
    log_print(logf, "=" * 66)
    log_print(logf, "GRAD-CAM EfficientNetB0 COMPLETE")
    log_print(logf, f"  Output: {out_dir}")
    log_print(logf, "  Files : GradCAM_4x3_panel.png")
    for q in ["TP", "FP", "FN", "TN"]:
        if samples_for_panel.get(q) is not None:
            log_print(logf, f"          {q}_panel.png")
    log_print(logf, "          gradcam_samples.csv")
    log_print(logf, "          gradcam_log.txt")
    log_print(logf, "=" * 66)
    logf.close()


if __name__ == "__main__":
    main()