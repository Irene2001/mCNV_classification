# test_scorecam_singlemode.py

"""
Score-CAM visualisation for the Swin-Tiny single-modality mCNV base model.

Outputs  ->  <TEST_EVAL_DIR>/scorecam/<timestamp>/
  ScoreCAM_4x3_panel.png   4x3: original | Score-CAM class-0 | Score-CAM class-1
  TP_panel.png  FP_panel.png  FN_panel.png  TN_panel.png
  scorecam_samples.csv     per-sample record (quadrant, filename, GT, Pred, ...)
  scorecam_log.txt

Algorithm (Score-CAM for Swin-Tiny, binary BCEWithLogits)
----------------------------------------------------------
Score-CAM is gradient-free.  It replaces gradient-based channel weights with
a forward-pass score, which is more faithful (no vanishing/exploding gradient
issues) and produces cleaner, less noisy saliency maps.

Algorithm (Wang et al. 2020, Algorithm 1) with medical imaging corrections:
  1. Forward pass: extract feature maps A  [C, H_feat, W_feat] at target layer.
  2. For each channel k = 1..C:
       a. Normalise A_k to [0,1]:  M_k = (A_k - min) / (max - min + eps)
       b. Upsample M_k to input size [H, W].
       c. Perturb input:  X_k = X * M_k + BG * (1 - M_k)
          BG = ImageNet-normalised black image = (0 - mean) / std per channel.
          CRITICAL FIX: naive X*M sends masked regions to 0 (ImageNet mean),
          not to true black, creating spurious activations on background areas.
          Using BG fill ensures masked regions are the model's true "nothing" baseline.
       d. Forward pass X_k; score s_k = logit(X_k) / T* (not sigmoid).
          Using raw logit gives wider dynamic range for softmax differentiation.
  3. Channel weights:  w = softmax(s)   [C]
     class=1 (active)   ->  s_k = logit_k / T*
     class=0 (inactive) ->  s_k = -logit_k / T*  (flip sign)
  4. Final CAM = ReLU( sum_k w_k * A_k )
  5. Normalise to [0, 1], threshold, bicubic upsample to [H, W].

Swin-Tiny specifics:
  - Target layer  : model.norm  (Global LayerNorm after all 4 Swin stages)
    Captures globally integrated semantic features.
  - Feature map   : [B, H_feat, W_feat, C] (NHWC, timm >= 0.6)
                    [B, N, C]               (flat tokens, timm <  0.6)
    C=768, H_feat=W_feat=7 for 224x224 input.
  - BG baseline   : tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]) per channel
                    (pixel value 0 after ImageNet normalisation)

Computational cost:
  Score-CAM runs C=768 forward passes per image per class.
  Each pass is light (no backward), but 768x2 = 1536 forward passes per sample.
  On GPU this takes ~5-30 s/image depending on hardware.
  BATCH_SIZE_SCORECAM controls how many masks are processed simultaneously.

Compatibility:
  Checkpoint format: {"model_state_dict": state_dict, "temperature": T*, ...}
  test_preds.csv   : exam_key | y_true | logit_uncal | prob_uncal |
                     temperature | logit_calib | prob_calib
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
    "swin_tiny/OCT1/"
    "BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8/"
    "Best_fold1"
)
# OCT0:  BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9
# OCT1:  BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8
# OCTA3: BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6

# 2. Checkpoint root (same as CHECKPOINT_ROOT in test_singlemode.py).
#    Leave "" to auto-detect as PROJECT_ROOT/checkpoints.
CHECKPOINT_ROOT = ""

# 3. Model switch
ACTIVE_MODEL = "swin_tiny"   # "swin_tiny" | "vgg16" | "efficientnet_b0"

TIMM_MODEL_MAP: Dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    # "vgg16":           "vgg16",
    # "efficientnet_b0": "efficientnet_b0",
}

# 4. Visual settings
IMG_SIZE      = 224     # must match training
OVERLAY_ALPHA = 0.50    # 0 = original only, 1 = heatmap only
COLORMAP      = "jet"
CLASS_NAMES   = ["inactive", "active"]

# 5. Score-CAM settings
# Number of channel masks processed in one forward batch.
# Larger = faster but more GPU memory.  C=768 for Swin-Tiny.
BATCH_SIZE_SCORECAM = 32

# CAM post-processing: zero pixels below CAM_THRESHOLD * max (before upsample).
CAM_THRESHOLD = 0.20   # slightly lower than Grad-CAM (Score-CAM is already focused)

# 6. Random seed
#    None = time-based (different sample every run)
#    Integer = fixed, reproducible
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
    Parse model_name, modality, run_tag, best_fold, project_root from path.
    Required: PROJECT_ROOT/outputs/test_evaluation/<model>/<modality>/<run_tag>/Best_fold{N}
    """
    p = Path(test_eval_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"TEST_EVAL_DIR not found: {test_eval_dir}")
    leaf = p.name
    if not leaf.startswith("Best_fold"):
        raise ValueError(f"TEST_EVAL_DIR must end with 'Best_fold{{N}}', got: {leaf}")
    return {
        "best_fold":    int(leaf.replace("Best_fold", "")),
        "run_tag":      p.parents[0].name,
        "modality":     p.parents[1].name,
        "model_name":   p.parents[2].name,
        "project_root": p.parents[5],
    }


# ------------------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    if model_name not in TIMM_MODEL_MAP:
        raise NotImplementedError(f"model_name='{model_name}' not in TIMM_MODEL_MAP.")
    return timm.create_model(TIMM_MODEL_MAP[model_name], pretrained=False, num_classes=1)


def load_checkpoint_and_temperature(
    model: nn.Module, ckpt_path: str, device: torch.device,
) -> Tuple[nn.Module, float]:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        state_dict  = raw.get("model_state_dict", raw.get("state_dict", raw))
        temperature = float(raw.get("temperature", 1.0))
    else:
        state_dict, temperature = raw, 1.0
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, temperature


# ------------------------------------------------------------------------------
# Score-CAM (gradient-free)
# ------------------------------------------------------------------------------

def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the feature extraction hook target layer.

    Swin-Tiny: model.norm  (Global LayerNorm, after all 4 stages)
      Produces globally integrated 7x7 feature maps; same rationale as Grad-CAM.
      Output (timm >= 0.6): [B, H, W, C] = [B, 7, 7, 768]
      Output (timm <  0.6): [B, N, C]    = [B, 49, 768]
    """
    if model_name == "swin_tiny":
        return model.norm
    raise NotImplementedError(
        f"Target layer not defined for '{model_name}'. "
        "Add an entry to _get_target_layer()."
    )


def _to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert Swin hook output to [B, C, H_feat, W_feat].
    Handles both 4D NHWC (timm >= 0.6) and 3D flat-token (timm < 0.6) formats.
    """
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()     # [B,H,W,C] -> [B,C,H,W]
    elif tensor.ndim == 3:
        B, N, C = tensor.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"N={N} is not a perfect square"
        return tensor.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


class SwinScoreCAM:
    """
    Score-CAM for timm Swin-Tiny binary (BCEWithLogits) model.

    Algorithm (Wang et al. CVPR Workshop 2020) with medical imaging corrections:
      For each channel k in the target feature map:
        1. Normalise channel A_k to [0,1]                  -> M_k  [H_feat, W_feat]
        2. Upsample M_k to input resolution [H, W]
        3. Masked input: X_k = X * M_k + BG * (1 - M_k)
           BG = ImageNet-normalised black image (pixel 0 in original space).
           Fix: naive X*M sends masked pixels to 0 = ImageNet mean (~grey),
           producing spurious background activations in medical images.
        4. Forward pass: s_k = logit(X_k) / T*              (no gradient needed)
      Weights  : w = softmax(s)                             [C]
      CAM      = ReLU( sum_k w_k * A_k )                   [H_feat, W_feat]

    Binary BCEWithLogits (single logit output [B,1]):
      class=1 (active)   ->  s_k = logit_k / T*   (higher logit = more active)
      class=0 (inactive) ->  s_k = -logit_k / T*  (lower logit = more inactive)
      Using logit (not sigmoid) gives wider dynamic range for softmax weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device,
        batch_size: int = BATCH_SIZE_SCORECAM,
    ):
        self.model      = model
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size

        self._activations: Optional[torch.Tensor] = None
        target = _get_target_layer(model, model_name)
        self._hook = target.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o.detach())
        )

    def remove_hooks(self) -> None:
        self._hook.remove()

    @torch.no_grad()
    def generate(
        self,
        img_tensor: torch.Tensor,      # [1, 3, H, W]
        class_idx: int,                # 0=inactive, 1=active
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate Score-CAM heatmap for class_idx.  No gradient is computed.

        Masking: X_k = X * M_k + BG * (1-M_k)  where BG = normalised black.
        Scoring: logit-based (not sigmoid) for wider softmax dynamic range.

        Returns
        -------
        cam : np.ndarray  [H, W]  values in [0, 1]
        prob_active : float  sigmoid(logit / T*)  from full (unmasked) image
        """
        self.model.eval()

        # Step 1: Extract feature maps via forward pass
        self._activations = None
        logit = self.model(img_tensor)                           # [1, 1]
        assert self._activations is not None, "Hook did not capture activations."

        A = _to_spatial(self._activations)[0]                    # [C, Hf, Wf]
        C, Hf, Wf = A.shape

        T = max(temperature, 1e-6)
        prob_active = float(torch.sigmoid(logit[0, 0] / T).item())

        # Step 2: Normalise each channel to [0, 1]  (Wang et al. 2020 Eq.4)
        A_min = A.flatten(1).min(dim=1).values.view(C, 1, 1)
        A_max = A.flatten(1).max(dim=1).values.view(C, 1, 1)
        A_norm = (A - A_min) / (A_max - A_min + 1e-8)           # [C, Hf, Wf]

        # Upsample all masks to input size  -> [C, 1, H, W]
        masks = F.interpolate(
            A_norm.unsqueeze(1),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )                                                         # [C, 1, H, W]

        # Step 3: Baseline bg = ImageNet-normalised black pixel (value 0 in original space).
        # CRITICAL: naive X*M sends masked pixels to 0 = ImageNet mean (grey ~0.45),
        # not to true black, creating strong false activations on OCT background borders.
        # Correct masked input: X_k = X * M_k + BG * (1 - M_k)
        # BG_c = (0 - mean_c) / std_c per channel.
        bg = torch.tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            dtype=torch.float32, device=self.device,
        ).view(1, 3, 1, 1)                                       # [1, 3, 1, 1]

        # Step 4: Batched forward passes; score each channel mask
        scores = torch.zeros(C, dtype=torch.float32, device=self.device)

        for start in range(0, C, self.batch_size):
            end     = min(start + self.batch_size, C)
            m_batch = masks[start:end]                           # [B, 1, H, W]

            # img_tensor [1,3,H,W] broadcasts with m_batch [B,1,H,W]
            masked_inputs = img_tensor * m_batch + bg * (1.0 - m_batch)  # [B, 3, H, W]

            logits_batch = self.model(masked_inputs)[:, 0]      # [B]
            # Score using logit (not sigmoid): wider dynamic range -> sharper softmax.
            # class=1 (active):   higher logit = more active -> positive score
            # class=0 (inactive): lower logit = more inactive -> flip sign
            if class_idx == 1:
                scores[start:end] = logits_batch / T
            else:
                scores[start:end] = -logits_batch / T

        # Step 5: Softmax channel weights  (Wang et al. 2020 Eq.5)
        weights = torch.softmax(scores, dim=0)                   # [C]

        # Step 6: Weighted sum + ReLU  (Wang et al. 2020 Eq.6)
        cam = (weights.view(C, 1, 1) * A).sum(dim=0, keepdim=True)  # [1, Hf, Wf]
        cam = F.relu(cam)

        # Step 7: Normalise at feature-map resolution
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Step 8: Threshold + bicubic upsample to input resolution
        if CAM_THRESHOLD > 0.0:
            cam = cam * (cam >= CAM_THRESHOLD).float()

        cam_up = F.interpolate(
            cam.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bicubic",
            align_corners=False,
        )[0, 0].cpu().numpy()
        return np.clip(cam_up, 0.0, 1.0), prob_active


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
    Save per-sample Score-CAM record to CSV.
    Columns: run_timestamp, model_name, modality, run_tag, best_fold,
             temperature, target_layer, cam_method, batch_size_scorecam,
             quadrant, exam_key, filename, image_path,
             gt_label, gt_class, pred_label, pred_class,
             prob_active, prob_inactive, logit_uncal, logit_calib,
             cam1_max, cam0_max, panel_file, note
    """
    rows = []
    for quadrant, data in samples.items():
        base = {
            "run_timestamp":       run_meta["timestamp"],
            "model_name":          run_meta["model_name"],
            "modality":            run_meta["modality"],
            "run_tag":             run_meta["run_tag"],
            "best_fold":           run_meta["best_fold"],
            "temperature":         run_meta["temperature"],
            "target_layer":        run_meta["target_layer"],
            "cam_method":          "Score-CAM",
            "batch_size_scorecam": BATCH_SIZE_SCORECAM,
            "cam_threshold":       CAM_THRESHOLD,
            "quadrant":            quadrant,
        }
        if data is None:
            base.update({
                "exam_key": "", "filename": "", "image_path": "",
                "gt_label": "", "gt_class": "", "pred_label": "", "pred_class": "",
                "prob_active": "", "prob_inactive": "",
                "logit_uncal": "", "logit_calib": "",
                "cam1_max": "", "cam0_max": "",
                "panel_file": f"(no {quadrant} sample)",
                "note": "no sample in this quadrant",
            })
        else:
            base.update({
                "exam_key":      data["exam_key"],
                "filename":      data["filename"],
                "image_path":    data["img_path"],
                "gt_label":      data["gt"],
                "gt_class":      CLASS_NAMES[data["gt"]],
                "pred_label":    data["pred"],
                "pred_class":    CLASS_NAMES[data["pred"]],
                "prob_active":   round(data["prob_active"], 6),
                "prob_inactive": round(1.0 - data["prob_active"], 6),
                "logit_uncal":   round(data["logit_uncal"], 6),
                "logit_calib":   round(data["logit_calib"], 6),
                "cam1_max":      round(float(data["cam1"].max()), 4),
                "cam0_max":      round(float(data["cam0"].max()), 4),
                "panel_file":    f"{quadrant}_panel.png",
                "note":          "",
            })
        rows.append(base)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------------------
# Plot helpers
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
    """1x3: original | Score-CAM class-0 | Score-CAM class-1"""
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
    axes[1].set_title("Score-CAM\nClass 0 (inactive)", fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(overlay_heatmap(orig_pil, cam1))
    axes[2].set_title("Score-CAM\nClass 1 (active)", fontweight="bold")
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
        f"Score-CAM -- {model_name.upper()} / {modality}  (Test Set)",
        fontsize=_TITLE_SIZE + 2, fontweight="bold",
    )
    for c, ct in enumerate(["Original",
                             "Score-CAM: Class 0 (inactive)",
                             "Score-CAM: Class 1 (active)"]):
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
    # scorecam folder, distinct from gradcam to avoid mixing results
    out_dir   = os.path.join(TEST_EVAL_DIR, "test_scorecam", ts)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "scorecam_log.txt"),
                "w", buffering=1, encoding="utf-8")

    log_print(logf, "=" * 62)
    log_print(logf, "SCORE-CAM  SINGLE-MODALITY  mCNV CLASSIFICATION")
    log_print(logf, "=" * 62)
    log_print(logf, f"model          : {model_name}  ({TIMM_MODEL_MAP[model_name]})")
    log_print(logf, f"modality       : {modality}")
    log_print(logf, f"run_tag        : {run_tag}")
    log_print(logf, f"best_fold      : {best_fold}")
    log_print(logf, f"device         : {device}")
    log_print(logf, f"checkpoint     : {ckpt_path}")
    log_print(logf, f"preds_csv      : {preds_csv}")
    log_print(logf, f"out_dir        : {out_dir}")
    log_print(logf, f"target_layer   : model.norm  (Global LayerNorm, all 4 stages)")
    log_print(logf, f"cam_method     : Score-CAM (gradient-free)")
    log_print(logf, f"masking        : X*M + BG*(1-M)  BG=normalised-black")
    log_print(logf, f"scoring        : logit/T (not sigmoid), class0=-logit/T")
    log_print(logf, f"batch_scorecam : {BATCH_SIZE_SCORECAM}  "
                    f"(channels per forward batch, C=768 total)")
    log_print(logf, f"cam_threshold  : {CAM_THRESHOLD}")
    log_print(logf, f"random_seed    : {N_RANDOM_SEED} (None=time-based)")
    log_print(logf, f"reference      : Wang et al. CVPR Workshop 2020 (Score-CAM)")

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
    scorecam = SwinScoreCAM(model, model_name, device,
                            batch_size=BATCH_SIZE_SCORECAM)
    log_print(logf, f"Score-CAM engine ready  target={target_layer_name}  "
                    f"C=768  batch={BATCH_SIZE_SCORECAM}")
    log_print(logf, f"  Est. forward passes per image: 768x2 classes = 1536")

    tfm = get_test_transform()

    # Step 4: Generate Score-CAM
    log_print(logf, "-" * 62)
    log_print(logf, "Step 4: Generate Score-CAM heatmaps")

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
            log_print(logf, f"  [{quadrant}] WARN: cannot locate master_manifest.csv -- skipping")
            samples_for_panel[quadrant] = None
            continue
        if not os.path.isfile(img_path):
            log_print(logf, f"  [{quadrant}] WARN: image not found: {img_path} -- skipping")
            samples_for_panel[quadrant] = None
            continue

        orig_pil   = Image.open(img_path).convert("RGB")
        img_tensor = tfm(orig_pil).unsqueeze(0).to(device)

        t0 = time.time()
        cam0, _          = scorecam.generate(img_tensor, class_idx=0, temperature=temperature)
        cam1, prob_active = scorecam.generate(img_tensor, class_idx=1, temperature=temperature)
        elapsed = time.time() - t0

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
                        f"  cam0_max={cam0.max():.3f}  cam1_max={cam1.max():.3f}"
                        f"  time={elapsed:.1f}s")

        panel_path = os.path.join(out_dir, f"{quadrant}_panel.png")
        save_single_panel(orig_pil, cam0, cam1, quadrant,
                          gt, pred, prob_active, panel_path)
        log_print(logf, f"    saved -> {panel_path}")

    scorecam.remove_hooks()

    # Step 5: Save combined 4x3 panel
    log_print(logf, "-" * 62)
    log_print(logf, "Step 5: Save combined 4x3 panel")
    big_path = os.path.join(out_dir, "ScoreCAM_4x3_panel.png")
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
    csv_path = os.path.join(out_dir, "scorecam_samples.csv")
    save_samples_csv(samples_for_panel, csv_path, run_meta)
    log_print(logf, f"Saved -> {csv_path}")

    # Done
    log_print(logf, "=" * 62)
    log_print(logf, "SCORE-CAM COMPLETE")
    log_print(logf, f"  Output: {out_dir}")
    log_print(logf, "  Files : ScoreCAM_4x3_panel.png")
    for q in ["TP", "FP", "FN", "TN"]:
        if samples_for_panel.get(q) is not None:
            log_print(logf, f"          {q}_panel.png")
    log_print(logf, "          scorecam_samples.csv")
    log_print(logf, "          scorecam_log.txt")
    log_print(logf, "=" * 62)
    logf.close()


if __name__ == "__main__":
    main()