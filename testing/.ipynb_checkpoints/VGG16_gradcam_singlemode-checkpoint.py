# VGG16_gradcam_singlemode.py

"""
VGG16 Grad-CAM visualisation for the mCNV single-modality base model.

Mirrors the output format of gradcam_singlemode.py (Swin-Tiny version) exactly:
  GradCAM_4x3_panel.png   4x3: original | Grad-CAM class-0 | Grad-CAM class-1
  TP_panel.png  FP_panel.png  FN_panel.png  TN_panel.png
  gradcam_samples.csv     per-sample record (quadrant, filename, GT, Pred, ...)
  gradcam_log.txt

Algorithm (Grad-CAM for VGG16, binary BCEWithLogits)
-----------------------------------------------------
VGG16 feature map structure (torchvision, 224x224 input):
  Block 1 (64ch)  : features[0..4]    -> 224x224
  Block 2 (128ch) : features[5..9]    -> 112x112
  Block 3 (256ch) : features[10..16]  ->  56x56
  Block 4 (512ch) : features[17..23]  ->  28x28
  Block 5 (512ch) : features[24..30]  ->  14x14  (trained in FIXED_BACKBONE mode)
  avgpool                              ->   7x7
  classifier[0..6]                     ->   1 logit

Target layer: model.features[28]
  The last Conv2d (3rd Conv in Block 5) at 14x14 spatial resolution.
  Captures the highest-level convolutional semantics available before
  spatial information is collapsed by avgpool.
  Upscaling ratio: 224/14 = 16x  (smoother than VGG4's 32x).

  Reference for VGG16 Grad-CAM target layer choice:
    Selvaraju et al. (2017). Grad-CAM. ICCV.
    https://doi.org/10.1109/ICCV.2017.74
    (Section 3: "the last convolutional layer before pooling")

Grad-CAM formula (Selvaraju et al. 2017, Eq. 1-2):
  alpha_k  = (1/Z) sum_{i,j} dY^c / dA^k_{ij}   (global avg of gradients)
  L^c      = ReLU( sum_k alpha_k * A^k )           (weighted activation map)

Binary BCEWithLogits scoring:
  class=1 (active)   -> score = logit        (positive -> active)
  class=0 (inactive) -> score = -logit       (flip sign -> active for inactive class)

CAM post-processing (applied at 14x14 before upsample):
  1. Normalize to [0,1].
  2. Zero pixels below CAM_THRESHOLD * max  (suppress background noise).
  3. Bicubic upsample 14x14 -> 224x224 (smooth 16x upscaling).
  4. Clip to [0,1].

VGG16 model loading:
  Checkpoint saved by VGG16_train_singlemode_oof.py:
    {model_state_dict: ..., temperature: T*, ...}
  Model structure: torchvision vgg16 with classifier[6] replaced by
    Linear(4096, 1)  for BCEWithLogitsLoss.
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
import torchvision.models as tv_models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG  --  Edit only this section
# ==============================================================================

# 1. Directory that contains test_preds.csv (produced by test_singlemode.py).
#    Structure: TEST_EVAL_ROOT/<model>/<modality>/<run_tag>/Best_fold{N}/
TEST_EVAL_DIR = (
    "/data/Irene/SwinTransformer/Swin_Meta/VGG16_outputs/test_evaluation/"
    "vgg16/Partial_B5/OCT0/"
    "BS16_EP99_LR3e-05_WD0.01_DR0.5_FIXED_BACKBONE_FL0.11_0.89_2_WSon_1_2.9/"
    "Best_fold5"
)


# OCT0: BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.11_0.89_2_WSon_1_2.9 (Best_fold2)
# OCT1: BS16_EP100_LR9e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.113_0.887_2_WSon_1_2.8 (Best_fold5)
# OCTA3: BS16_EP100_LR8e-06_WD0.01_DR0.5_FIXED_BACKBONE_FL0.13_0.87_2_WSon_1_2.6 (Best_fold1)

# 2. Checkpoint root (same as CHECKPOINT_ROOT in VGG16_train_singlemode_oof.py).
#    Leave "" to auto-detect as PROJECT_ROOT/checkpoints.
CHECKPOINT_ROOT = ""

# 3. Model name.  Only "vgg16" is supported here.
ACTIVE_MODEL = "vgg16"

# 4. Visual settings
IMG_SIZE      = 224      # must match training
OVERLAY_ALPHA = 0.50     # 0 = original only, 1 = heatmap only
COLORMAP      = "jet"    # matplotlib colormap for heatmap
CLASS_NAMES   = ["inactive", "active"]

# CAM post-processing applied at 14x14 feature-map resolution (before upsample).
# Pixels below CAM_THRESHOLD * max are zeroed out.
# Suppresses low-contribution background noise (black borders, artefacts).
# Set 0.0 to disable.
# Note: VGG16 14x14 CAM has finer spatial detail than Swin 7x7;
#       0.25-0.35 is a reasonable starting point.
CAM_THRESHOLD = 0.0

# 5. Random seed.
#    None = time-based (different sample every run).
#    Integer (e.g. 42) = fixed, reproducible selection.
N_RANDOM_SEED = None

# Dropout rate used during training (must match VGG16_train_singlemode_oof.py).
# Used only to reconstruct the model architecture for checkpoint loading.
DROP_RATE = 0.5

# Output figure style
_TITLE_SIZE      = 14
_AXIS_LABEL_SIZE = 11
_FIG_DPI         = 220

# ==============================================================================


# VGG16 target layer index in model.features.
# features[28] = last Conv2d in Block 5 (14x14 spatial, 512 channels).
# This is the standard Grad-CAM target for VGG-family networks.
# Ref: Selvaraju et al. 2017, Section 3: "last convolutional layer before pooling".
_VGG16_TARGET_FEAT_IDX = 28
_VGG16_FEAT_MAP_SIZE   = 14   # spatial resolution at features[28]: 14x14


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

    Required structure (mirrors VGG16_train_singlemode_oof.py output tree):
      PROJECT_ROOT/outputs/test_evaluation/<model>/<sub>/<modality>/<run_tag>/Best_fold{N}
      or (without sub-dir):
      PROJECT_ROOT/outputs/test_evaluation/<model>/<modality>/<run_tag>/Best_fold{N}

    The function finds Best_fold{N} and walks up to reconstruct key fields.
    """
    p = Path(test_eval_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"TEST_EVAL_DIR not found: {test_eval_dir}")

    leaf = p.name
    if not leaf.startswith("Best_fold"):
        raise ValueError(
            f"TEST_EVAL_DIR must end with 'Best_fold{{N}}', got: {leaf}"
        )
    best_fold = int(leaf.replace("Best_fold", ""))

    # Walk up: Best_fold{N} / run_tag / modality / [sub /] model / ...
    run_tag  = p.parents[0].name   # run_tag
    modality = p.parents[1].name   # OCT0 / OCT1 / OCTA3
    strategy = p.parents[2].name   # Partial_B5
    model_name = p.parents[3].name # vgg16

    # Search upward for the "outputs" anchor to find PROJECT_ROOT
    # and model_name (the dir right below test_evaluation/)
    anchor = None
    for parent in p.parents:
        if parent.name == "test_evaluation":
            anchor = parent
            break

    if anchor is None:
        raise ValueError(
            "Could not find 'test_evaluation' in TEST_EVAL_DIR path. "
            f"Path: {test_eval_dir}"
        )

    project_root = anchor.parent.parent   # two levels up: outputs -> PROJECT_ROOT
    model_name   = p.relative_to(anchor).parts[0]   # first subdir under test_evaluation

    return {
        "best_fold":    best_fold,
        "run_tag":      run_tag,
        "modality":     modality,
        "strategy":     strategy,
        "model_name":   model_name,
        "project_root": project_root,
    }


# ------------------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------------------

def _build_vgg16_for_inference(drop_rate: float = 0.5) -> nn.Module:
    """
    建立 VGG16 模型並禁用所有的 inplace 操作以支援 Grad-CAM。
    """
    model = tv_models.vgg16(weights=None)
    
    # 1. 替換分類頭 (與訓練時一致)
    in_features = model.classifier[6].in_features   
    model.classifier[6] = nn.Linear(in_features, 1)
    
    if drop_rate != 0.5 and drop_rate >= 0.0:
        model.classifier[2] = nn.Dropout(p=drop_rate)
        model.classifier[5] = nn.Dropout(p=drop_rate)
        
    # --- 核心修正點：禁用所有 ReLU 的 inplace 屬性 ---
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False  # 改為 False，避免干擾梯度 Hook
            
    return model


def load_checkpoint_and_temperature(
    model: nn.Module, ckpt_path: str, device: torch.device,
) -> Tuple[nn.Module, float]:
    """
    Load model_best.pth saved by VGG16_train_singlemode_oof.py.
    Returns (model, temperature).  temperature defaults to 1.0 if not stored.

    Checkpoint dict keys (saved by VGG16_train_singlemode_oof.py):
      model_state_dict | temperature | val_nll_uncal | val_acc | val_auc
      nll_beforeTS | nll_afterTS | fold | model_name | modality
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
# Grad-CAM for VGG16 (manual hook, no external library)
# ------------------------------------------------------------------------------

class VGG16GradCAM:
    """
    Grad-CAM for torchvision VGG16 binary (BCEWithLogits) model.

    Target layer: model.features[28]
      The last Conv2d in Block 5 at 14x14 spatial resolution.
      Output shape: [B, 512, 14, 14]  (standard 4D CNN feature map, no reshape needed).

    Grad-CAM formula (Selvaraju et al. 2017, Eq. 1-2):
      alpha_k  = (1/Z) sum_{i,j} dY^c / dA^k_{ij}
      L^c      = ReLU( sum_k alpha_k * A^k )

    Binary BCEWithLogits scoring:
      class=1 (active)   -> score = logit
      class=0 (inactive) -> score = -logit

    CAM pipeline:
      1. Grad-CAM at 14x14 (features[28] output resolution).
      2. Normalize to [0,1].
      3. Threshold: zero pixels < CAM_THRESHOLD * max.
      4. Bicubic upsample 14->224 (16x, smoother than bilinear).
      5. Clip to [0,1].

    Reference: Selvaraju et al. (2017). ICCV.
    https://doi.org/10.1109/ICCV.2017.74
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model
        self.device = device

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Hook on features[28] = last Conv2d of Block 5
        target = model.features[_VGG16_TARGET_FEAT_IDX]
        self._fwd_hook = target.register_forward_hook(self._save_activation)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        # out: [B, 512, 14, 14]  (CNN feature map, already BCHW)
        self._activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        if grad_out and grad_out[0] is not None:
            # grad_out[0]: [B, 512, 14, 14]
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
            Pipeline: Grad-CAM [14x14] -> normalize -> threshold ->
                      bicubic upsample [224x224] -> clip
        prob_active   : float  sigmoid(logit / T*)
        """
        self.model.eval()
        self.model.zero_grad()
        self._activations = None
        self._gradients   = None

        # Forward pass
        logit       = self.model(img_tensor)                        # [1, 1]
        logit_calib = logit / max(temperature, 1e-6)
        prob_active = float(torch.sigmoid(logit_calib[0, 0]).item())

        # Backward pass: binary scoring
        #   class=1 -> maximize logit; class=0 -> maximize -logit
        score = logit[0, 0] if class_idx == 1 else -logit[0, 0]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients.\n"
                "Check that model.features[28] is the correct target layer."
            )

        # A: [1, 512, 14, 14],  G: [1, 512, 14, 14]
        A = self._activations   # no reshape needed -- VGG16 is already BCHW
        G = self._gradients

        # Eq.1: alpha_k = (1/Z) sum_{i,j} dY/dA^k_{ij}
        weights = G.mean(dim=(2, 3), keepdim=True)          # [1, 512, 1, 1]

        # Eq.2: L = ReLU( sum_k alpha_k * A^k )
        cam = (weights * A).sum(dim=1, keepdim=True)         # [1, 1, 14, 14]
        cam = F.relu(cam)

        # Normalize to [0, 1] at 14x14 feature-map resolution
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        # Threshold: zero out pixels below CAM_THRESHOLD * max.
        # Applied at 14x14 to suppress low-contribution background regions.
        if CAM_THRESHOLD > 0.0:
            cam = cam * (cam >= CAM_THRESHOLD).float()
       
        # Bicubic upsample 14x14 -> 224x224 (16x upscaling).
        # Bicubic provides smoother transitions than bilinear.
        cam_up = F.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE),
            mode="bicubic", align_corners=False,
        )[0, 0].cpu().numpy()
        cam_up = np.clip(cam_up, 0.0, 1.0)

        return cam_up, prob_active


# ------------------------------------------------------------------------------
# Image transform  (must match VGG16_train_singlemode_oof.py)
# ------------------------------------------------------------------------------

def get_test_transform() -> transforms.Compose:
    """ImageNet eval transform -- identical to training validation transform."""
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
# CSV record  (mirrors gradcam_singlemode.py save_samples_csv exactly)
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
                "cam1_max": round(float(data["cam1"].max()), 4),
                "cam0_max": round(float(data["cam0"].max()), 4),
                "panel_file": f"{quadrant}_panel.png",
                "note": "",
            })
        rows.append(base)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------------------
# Plot helpers  (identical style to gradcam_singlemode.py)
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
    model_name   = ACTIVE_MODEL          # "vgg16"
    strategy     = parsed["strategy"]    # "Partial_B5"
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
        # Fallback: Kfold layout
        ckpt_path = os.path.join(
            ckpt_root, model_name, strategy, modality, run_tag,
            "Kfold", f"fold{best_fold}", "model_best.pth",
        )

    preds_csv = os.path.join(TEST_EVAL_DIR, "test_preds.csv")

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(TEST_EVAL_DIR, "gradcam_vgg16", ts)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "gradcam_log.txt"),
                "w", buffering=1, encoding="utf-8")

    log_print(logf, "=" * 66)
    log_print(logf, "GRAD-CAM  VGG16  SINGLE-MODALITY  mCNV CLASSIFICATION")
    log_print(logf, "=" * 66)
    log_print(logf, f"model        : {model_name}  (torchvision vgg16)")
    log_print(logf, f"modality     : {modality}")
    log_print(logf, f"run_tag      : {run_tag}")
    log_print(logf, f"best_fold    : {best_fold}")
    log_print(logf, f"device       : {device}")
    log_print(logf, f"checkpoint   : {ckpt_path}")
    log_print(logf, f"preds_csv    : {preds_csv}")
    log_print(logf, f"out_dir      : {out_dir}")
    log_print(logf, f"target_layer : model.features[{_VGG16_TARGET_FEAT_IDX}]"
                    f"  (last Conv2d Block5, {_VGG16_FEAT_MAP_SIZE}x{_VGG16_FEAT_MAP_SIZE}, 512ch)")
    log_print(logf, f"cam_threshold: {CAM_THRESHOLD}"
                    f"  (zero pixels < threshold*max at {_VGG16_FEAT_MAP_SIZE}x{_VGG16_FEAT_MAP_SIZE})")
    log_print(logf, f"upsample     : bicubic {_VGG16_FEAT_MAP_SIZE}x{_VGG16_FEAT_MAP_SIZE}"
                    f" -> {IMG_SIZE}x{IMG_SIZE}  ({IMG_SIZE // _VGG16_FEAT_MAP_SIZE}x)")
    log_print(logf, f"drop_rate    : {DROP_RATE}")
    log_print(logf, f"random_seed  : {N_RANDOM_SEED}  (None=time-based, diff every run)")
    log_print(logf, f"reference    : Selvaraju et al. ICCV 2017 (Grad-CAM)")
    log_print(logf, f"reference    : Simonyan & Zisserman arXiv:1409.1556 (VGG)")

    # Step 1: Load test_preds.csv
    log_print(logf, "-" * 66)
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
    log_print(logf, "Step 3: Build VGG16 model and load checkpoint")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"model_best.pth not found:\n  {ckpt_path}\n"
            "Check CHECKPOINT_ROOT."
        )

    model = _build_vgg16_for_inference(drop_rate=DROP_RATE)
    model, ckpt_temperature = load_checkpoint_and_temperature(model, ckpt_path, device)

    log_print(logf, f"Checkpoint T*={ckpt_temperature:.6f}  "
                    f"(using CSV T*={temperature:.6f})")
    log_print(logf, f"Model: torchvision VGG16  classifier[6]=Linear(4096,1)")

    target_layer_name = f"model.features[{_VGG16_TARGET_FEAT_IDX}]"
    gradcam = VGG16GradCAM(model, device)
    log_print(logf, f"Grad-CAM engine ready  target={target_layer_name}")
    log_print(logf, f"  Feature map shape at target: [B, 512, "
                    f"{_VGG16_FEAT_MAP_SIZE}, {_VGG16_FEAT_MAP_SIZE}]")

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
                    _manifest_df = pd.read_csv(
                        str(c), usecols=["exam_key", img_col]
                    )
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

    # Step 5: Save combined 4x3 panel
    log_print(logf, "-" * 66)
    log_print(logf, "Step 5: Save combined 4x3 panel")
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
    log_print(logf, "GRAD-CAM VGG16 COMPLETE")
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