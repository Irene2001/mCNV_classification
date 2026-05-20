# Swin_focus_scorecam_multimodal.py
#
# Score-CAM visualisation for the Swin-Tiny multimodal mCNV base model.
# Input: 224×448 combined image (OCT top 224×224 / OCTA3 bottom 224×224).
#
# 與 Swin_focus_scorecam_singlemode.py 的差異（最小改動）：
#   ① TEST_EVAL_DIR 指向 Multimodal_outputs/test_evaluation
#   ② CHECKPOINT_ROOT 指向 Multimodal_outputs/checkpoints
#   ③ parse_test_eval_dir()：p.parents[4] 取 multimodal_out_root（非 p.parents[5]）
#      同時加入 training_dir 驗證，確認路徑結構正確
#   ④ IMG_SIZE 改為 (IMG_H, IMG_W) = (448, 224)，非正方形
#   ⑤ build_model() 傳入 img_size=(IMG_H, IMG_W)
#   ⑥ get_test_transform() resize 改為 (IMG_H, IMG_W)
#   ⑦ _to_spatial() 不假設正方形：H_feat, W_feat 從 N 直接計算
#   ⑧ img_col 固定為 oct0_image_path（合併影像路徑），不需查 master_manifest
#   ⑨ SwinScoreCAM.generate() 的 masked_inputs 用 (IMG_H, IMG_W) upsample
#   其餘（Score-CAM 演算法、overlay、panel、CSV、log）完全不變
#
# Algorithm (Score-CAM for Swin-Tiny, binary BCEWithLogits)
# ----------------------------------------------------------
# Score-CAM is gradient-free.  It replaces gradient-based channel weights with
# a forward-pass score, which is more faithful (no vanishing/exploding gradient
# issues) and produces cleaner, less noisy saliency maps.
#
# Algorithm (Wang et al. 2020, Algorithm 1) with medical imaging corrections:
#   1. Forward pass: extract feature maps A  [C, H_feat, W_feat] at target layer.
#   2. For each channel k = 1..C:
#        a. Normalise A_k to [0,1]:  M_k = (A_k - min) / (max - min + eps)
#        b. Upsample M_k to input size [H, W].
#        c. Perturb input:  X_k = X * M_k + BG * (1 - M_k)
#           BG = ImageNet-normalised black image = (0 - mean) / std per channel.
#           CRITICAL FIX: naive X*M sends masked regions to 0 (ImageNet mean),
#           not to true black, creating spurious activations on background areas.
#           Using BG fill ensures masked regions are the model's true "nothing" baseline.
#        d. Forward pass X_k; score s_k = logit(X_k) / T* (not sigmoid).
#           Using raw logit gives wider dynamic range for softmax differentiation.
#   3. Channel weights:  w = softmax(s)   [C]
#      class=1 (active)   ->  s_k = logit_k / T*
#      class=0 (inactive) ->  s_k = -logit_k / T*  (flip sign)
#   4. Final CAM = ReLU( sum_k w_k * A_k )
#   5. Normalise to [0, 1], threshold, bicubic upsample to [H, W].
#
# Swin-Tiny multimodal specifics:
#   - Input size    : 224×448 (W×H), non-square
#   - Target layer  : model.norm  (Global LayerNorm after all 4 Swin stages)
#   - Feature map   : [B, H_feat, W_feat, C] (NHWC, timm >= 0.6)
#                     [B, N, C]               (flat tokens, timm <  0.6)
#     C=768, H_feat=14, W_feat=7 for 448×224 input (patch_size=32 → 14×7=98 tokens)
#   - BG baseline   : tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]) per channel

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

# ① TEST_EVAL_DIR → Multimodal_outputs/test_evaluation
#    Structure: MULTIMODAL_OUTPUT_ROOT/test_evaluation/<model>/MULTIMODAL/<run_tag>/Best_fold{N}/
TEST_EVAL_DIR = (
    "/data/Irene/SwinTransformer/Swin_Meta/Multimodal_outputs/test_evaluation/"
    "swin_tiny/MULTIMODAL/"
    "MULTIMODAL_BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6/"
    "Best_fold2"
)

# ② CHECKPOINT_ROOT → Multimodal_outputs/checkpoints
CHECKPOINT_ROOT = (
    "/data/Irene/SwinTransformer/Swin_Meta/Multimodal_outputs/checkpoints"
)

ACTIVE_MODEL = "swin_tiny"

TIMM_MODEL_MAP: Dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

# ④ 影像尺寸：非正方形，與訓練腳本一致
IMG_W = 224
IMG_H = 448    # OCT 上 224 / OCTA3 下 224

OVERLAY_ALPHA = 0.50
COLORMAP      = "jet"
CLASS_NAMES   = ["inactive", "active"]
MODALITY      = "MULTIMODAL"

# Score-CAM settings（與 singlemode 完全相同）
BATCH_SIZE_SCORECAM = 32
CAM_THRESHOLD       = 0.20

N_RANDOM_SEED = None

_TITLE_SIZE      = 14
_AXIS_LABEL_SIZE = 11
_FIG_DPI         = 300

# ==============================================================================


# ------------------------------------------------------------------------------
# Utilities（與 singlemode 完全相同）
# ------------------------------------------------------------------------------

def log_print(logf, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if logf:
        logf.write(line + "\n")
        logf.flush()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ------------------------------------------------------------------------------
# ③ parse_test_eval_dir — 修正 parents 層級，加入 training_dir 驗證
# ------------------------------------------------------------------------------

def parse_test_eval_dir(test_eval_dir: str) -> dict:
    """
    從 TEST_EVAL_DIR 解析路徑元件。

    Required structure:
      <MULTIMODAL_OUTPUT_ROOT>/test_evaluation/<model>/MULTIMODAL/<run_tag>/Best_fold{N}

    Path parents mapping：
      p              = .../Multimodal_outputs/test_evaluation/swin_tiny/MULTIMODAL/<run_tag>/Best_foldN
      p.parents[0]   = <run_tag>
      p.parents[1]   = MULTIMODAL
      p.parents[2]   = swin_tiny
      p.parents[3]   = test_evaluation
      p.parents[4]   = Multimodal_outputs   ← ③ 正確的 root
    """
    p = Path(test_eval_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"TEST_EVAL_DIR not found: {test_eval_dir}")

    leaf = p.name
    if not leaf.startswith("Best_fold"):
        raise ValueError(
            f"TEST_EVAL_DIR must end with 'Best_fold{{N}}', got: '{leaf}'"
        )

    best_fold         = int(leaf.replace("Best_fold", ""))
    run_tag           = p.parents[0].name   # <run_tag>
    modality_from_path = p.parents[1].name  # should be "MULTIMODAL"
    model_name        = p.parents[2].name   # e.g. "swin_tiny"
    eval_dir_name     = p.parents[3].name   # should be "test_evaluation"
    multimodal_root   = p.parents[4]        # ③ Multimodal_outputs

    if eval_dir_name != "test_evaluation":
        raise ValueError(
            f"TEST_EVAL_DIR path structure error: expected 'test_evaluation', "
            f"got '{eval_dir_name}'.\n"
            f"Required: <MULTIMODAL_OUTPUT_ROOT>/test_evaluation/<model>/"
            f"{MODALITY}/<run_tag>/Best_fold{{N}}"
        )
    if modality_from_path != MODALITY:
        raise ValueError(
            f"Parsed modality='{modality_from_path}' from path, "
            f"but script MODALITY='{MODALITY}'."
        )

    return {
        "best_fold":         best_fold,
        "run_tag":           run_tag,
        "modality":          MODALITY,
        "model_name":        model_name,
        "multimodal_root":   multimodal_root,
    }


# ------------------------------------------------------------------------------
# ⑤ Model builder — 傳入 img_size=(IMG_H, IMG_W)
# ------------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    if model_name not in TIMM_MODEL_MAP:
        raise NotImplementedError(
            f"model_name='{model_name}' not in TIMM_MODEL_MAP."
        )
    # ⑤ img_size=(IMG_H, IMG_W) = (448, 224)，讓 Swin patch_embed 正確建立非正方形
    return timm.create_model(
        TIMM_MODEL_MAP[model_name],
        pretrained=False,
        num_classes=1,
        img_size=(IMG_H, IMG_W),
    )


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
# Score-CAM（演算法與 singlemode 相同，僅 upsample size 改為非正方形）
# ------------------------------------------------------------------------------

def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Swin-Tiny: model.norm (Global LayerNorm, after all 4 stages)
    Output (timm >= 0.6): [B, H_feat, W_feat, C]
    Output (timm <  0.6): [B, N, C]
    For 448×224 input: H_feat=14, W_feat=7 (patch_size=32 → 14×7=98 tokens).
    """
    if model_name == "swin_tiny":
        return model.norm
    raise NotImplementedError(
        f"Target layer not defined for '{model_name}'."
    )


def _to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert Swin hook output to [B, C, H_feat, W_feat].
    ⑦ 修正：非正方形輸入時 H_feat ≠ W_feat，不能假設 N 為完全平方數。
       對 448×224 輸入，patch_size=32：H_feat=448/32=14, W_feat=224/32=7, N=98。
       使用 IMG_H // 32 × IMG_W // 32 推算 H_feat, W_feat。
    """
    if tensor.ndim == 4:
        # timm >= 0.6: [B, H_feat, W_feat, C] -> [B, C, H_feat, W_feat]
        return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.ndim == 3:
        B, N, C = tensor.shape
        # ⑦ 非正方形：從訓練 img_size 推算 patch grid
        H_feat = IMG_H // 32   # Swin patch_size=4, window 7→ downscale ×32
        W_feat = IMG_W // 32
        assert H_feat * W_feat == N, (
            f"Token count mismatch: N={N}, expected H_feat×W_feat="
            f"{H_feat}×{W_feat}={H_feat*W_feat}.\n"
            f"Check IMG_H={IMG_H}, IMG_W={IMG_W} and Swin stride settings."
        )
        return tensor.view(B, H_feat, W_feat, C).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


class SwinScoreCAM:
    """
    Score-CAM for timm Swin-Tiny binary (BCEWithLogits) model.
    ⑨ Upsample mask size 改為 (IMG_H, IMG_W) = (448, 224)（非正方形）。
    演算法邏輯與 singlemode 完全相同。
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
        img_tensor: torch.Tensor,      # [1, 3, IMG_H, IMG_W]
        class_idx: int,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate Score-CAM heatmap for class_idx.
        Returns: (cam [IMG_H, IMG_W], prob_active float)
        """
        self.model.eval()
        self._activations = None
        logit = self.model(img_tensor)
        assert self._activations is not None, "Hook did not capture activations."

        A = _to_spatial(self._activations)[0]    # [C, H_feat, W_feat]
        C, Hf, Wf = A.shape

        T = max(temperature, 1e-6)
        prob_active = float(torch.sigmoid(logit[0, 0] / T).item())

        # Normalise each channel to [0, 1]
        A_min = A.flatten(1).min(dim=1).values.view(C, 1, 1)
        A_max = A.flatten(1).max(dim=1).values.view(C, 1, 1)
        A_norm = (A - A_min) / (A_max - A_min + 1e-8)    # [C, Hf, Wf]

        # ⑨ Upsample to (IMG_H, IMG_W) = (448, 224) — non-square
        masks = F.interpolate(
            A_norm.unsqueeze(1),
            size=(IMG_H, IMG_W),      # ⑨ non-square upsample
            mode="bilinear",
            align_corners=False,
        )                                                  # [C, 1, IMG_H, IMG_W]

        # BG = ImageNet-normalised black pixel (value 0 in original space)
        bg = torch.tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            dtype=torch.float32, device=self.device,
        ).view(1, 3, 1, 1)

        scores = torch.zeros(C, dtype=torch.float32, device=self.device)

        for start in range(0, C, self.batch_size):
            end     = min(start + self.batch_size, C)
            m_batch = masks[start:end]                    # [B, 1, IMG_H, IMG_W]

            masked_inputs = img_tensor * m_batch + bg * (1.0 - m_batch)
            logits_batch  = self.model(masked_inputs)[:, 0]

            if class_idx == 1:
                scores[start:end] = logits_batch / T
            else:
                scores[start:end] = -logits_batch / T

        weights = torch.softmax(scores, dim=0)

        cam = (weights.view(C, 1, 1) * A).sum(dim=0, keepdim=True)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        if CAM_THRESHOLD > 0.0:
            cam = cam * (cam >= CAM_THRESHOLD).float()

        # ⑨ Bicubic upsample to (IMG_H, IMG_W) — non-square
        cam_up = F.interpolate(
            cam.unsqueeze(0),
            size=(IMG_H, IMG_W),      # ⑨ non-square
            mode="bicubic",
            align_corners=False,
        )[0, 0].cpu().numpy()
        return np.clip(cam_up, 0.0, 1.0), prob_active


# ------------------------------------------------------------------------------
# ⑥ Image transform — resize to (IMG_H, IMG_W) = (448, 224)
# ------------------------------------------------------------------------------

def get_test_transform() -> transforms.Compose:
    """⑥ resize 改為 (IMG_H, IMG_W)，與訓練 tf_val 完全一致。"""
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),           # ⑥ (448, 224)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------------------
# Overlay（與 singlemode 完全相同，但 resize 到 IMG_H×IMG_W）
# ------------------------------------------------------------------------------

def overlay_heatmap(
    orig_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    colormap: str = COLORMAP,
) -> np.ndarray:
    img_np  = np.array(
        orig_pil.resize((IMG_W, IMG_H), Image.BILINEAR).convert("RGB"),
        dtype=np.float32,
    ) / 255.0
    heatmap = plt.get_cmap(colormap)(cam)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * img_np + alpha * heatmap, 0.0, 1.0)


# ------------------------------------------------------------------------------
# CSV record（與 singlemode 完全相同）
# ------------------------------------------------------------------------------

def save_samples_csv(
    samples: Dict[str, Optional[dict]],
    out_path: str,
    run_meta: dict,
) -> None:
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
# Plot helpers（與 singlemode 完全相同）
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
    """1×3: original | Score-CAM class-0 | Score-CAM class-1"""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(
        f"{quadrant}  |  GT={CLASS_NAMES[gt]}  Pred={CLASS_NAMES[pred]}"
        f"  P(active)={prob_active:.3f}",
        fontsize=_TITLE_SIZE, fontweight="bold",
    )
    axes[0].imshow(np.array(
        orig_pil.resize((IMG_W, IMG_H)).convert("RGB")
    ))
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
    """4×3: rows=TP/FP/FN/TN, cols=original/class-0/class-1"""
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
                ax.imshow(np.array(orig.resize((IMG_W, IMG_H)).convert("RGB")))
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
    parsed     = parse_test_eval_dir(TEST_EVAL_DIR)
    model_name = ACTIVE_MODEL
    modality   = parsed["modality"]
    run_tag    = parsed["run_tag"]
    best_fold  = parsed["best_fold"]

    # Checkpoint path（與 singlemode 相同邏輯，路徑已指向 Multimodal_outputs）
    ckpt_path = os.path.join(
        CHECKPOINT_ROOT, model_name, modality, run_tag,
        f"Best_fold{best_fold}", "model_best.pth",
    )
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(
            CHECKPOINT_ROOT, model_name, modality, run_tag,
            "Kfold", f"fold{best_fold}", "model_best.pth",
        )

    preds_csv = os.path.join(TEST_EVAL_DIR, "test_preds.csv")
    ts        = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(TEST_EVAL_DIR, "new_scorecam", ts)
    ensure_dir(out_dir)

    logf = open(os.path.join(out_dir, "scorecam_log.txt"),
                "w", buffering=1, encoding="utf-8")

    log_print(logf, "=" * 62)
    log_print(logf, "SCORE-CAM  MULTIMODAL  mCNV CLASSIFICATION")
    log_print(logf, "=" * 62)
    log_print(logf, f"model          : {model_name}  ({TIMM_MODEL_MAP[model_name]})")
    log_print(logf, f"modality       : {modality}  (OCT top {IMG_H//2}px / OCTA3 bottom {IMG_H//2}px)")
    log_print(logf, f"img_size       : {IMG_W}×{IMG_H}  (W×H, non-square)")
    log_print(logf, f"run_tag        : {run_tag}")
    log_print(logf, f"best_fold      : {best_fold}")
    log_print(logf, f"device         : {device}")
    log_print(logf, f"checkpoint     : {ckpt_path}")
    log_print(logf, f"preds_csv      : {preds_csv}")
    log_print(logf, f"out_dir        : {out_dir}")
    log_print(logf, f"target_layer   : model.norm  (Global LayerNorm, all 4 stages)")
    log_print(logf, f"feat_map_size  : H_feat={IMG_H//32}, W_feat={IMG_W//32}, C=768")
    log_print(logf, f"cam_method     : Score-CAM (gradient-free)")
    log_print(logf, f"masking        : X*M + BG*(1-M)  BG=normalised-black")
    log_print(logf, f"scoring        : logit/T (not sigmoid), class0=-logit/T")
    log_print(logf, f"batch_scorecam : {BATCH_SIZE_SCORECAM}  (C=768 total)")
    log_print(logf, f"cam_threshold  : {CAM_THRESHOLD}")
    log_print(logf, f"random_seed    : {N_RANDOM_SEED} (None=time-based)")
    log_print(logf, f"reference      : Wang et al. CVPR Workshop 2020 (Score-CAM)")

    # Step 1: Load test_preds.csv
    log_print(logf, "-" * 62)
    log_print(logf, "Step 1: Load test_preds.csv")

    if not os.path.isfile(preds_csv):
        raise FileNotFoundError(
            f"test_preds.csv not found:\n  {preds_csv}\n"
            "Run New_2class_Swin_test_multimodal.py first."
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

    # Dummy forward 確認非正方形 input 可通過
    model.eval()
    with torch.no_grad():
        dummy     = torch.randn(1, 3, IMG_H, IMG_W, device=device)
        dummy_out = model(dummy)
    if dummy_out.shape != (1, 1):
        raise RuntimeError(
            f"Dummy forward shape 不符：{dummy_out.shape}，期望 (1,1)。"
            f"請確認 timm model 支援 img_size=({IMG_H},{IMG_W})。"
        )

    log_print(logf, f"Checkpoint T*={ckpt_temperature:.6f}  "
                    f"(using CSV T*={temperature:.6f})")

    target_layer_name = "model.norm"
    scorecam = SwinScoreCAM(model, model_name, device,
                            batch_size=BATCH_SIZE_SCORECAM)
    log_print(logf, f"Score-CAM engine ready  target={target_layer_name}  "
                    f"H_feat={IMG_H//32} W_feat={IMG_W//32} C=768  "
                    f"batch={BATCH_SIZE_SCORECAM}")
    log_print(logf, f"  Est. forward passes per image: 768×2 classes = 1536")

    tfm = get_test_transform()

    # Step 4: Generate Score-CAM
    log_print(logf, "-" * 62)
    log_print(logf, "Step 4: Generate Score-CAM heatmaps")

    # ⑧ 合併影像路徑直接從 test_preds.csv 的 oct0_image_path 欄取得，
    #    不需再查 master_manifest.csv
    MULTIMODAL_IMG_COL = "oct0_image_path"

    samples_for_panel: Dict[str, Optional[dict]] = {}

    for quadrant, row_idx in selected.items():
        if row_idx is None:
            log_print(logf, f"  [{quadrant}] skipped -- no sample")
            samples_for_panel[quadrant] = None
            continue

        row  = df.loc[row_idx]
        gt   = int(row["y_true"])
        pred = int(row["y_pred_calib"])

        # ⑧ 直接讀合併影像路徑
        if MULTIMODAL_IMG_COL not in df.columns:
            log_print(logf,
                      f"  [{quadrant}] WARN: '{MULTIMODAL_IMG_COL}' not in "
                      f"test_preds.csv -- skipping.\n"
                      f"  Please re-run New_2class_Swin_test_multimodal.py to "
                      f"regenerate test_preds.csv with source tracing columns.")
            samples_for_panel[quadrant] = None
            continue

        img_path = str(row[MULTIMODAL_IMG_COL])
        if not os.path.isfile(img_path):
            log_print(logf, f"  [{quadrant}] WARN: image not found: {img_path} -- skipping")
            samples_for_panel[quadrant] = None
            continue

        orig_pil   = Image.open(img_path).convert("RGB")
        img_tensor = tfm(orig_pil).unsqueeze(0).to(device)

        t0 = time.time()
        cam0, _           = scorecam.generate(img_tensor, class_idx=0, temperature=temperature)
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

    # Step 5: Save combined 4×3 panel
    log_print(logf, "-" * 62)
    log_print(logf, "Step 5: Save combined 4×3 panel")
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
    log_print(logf, "SCORE-CAM MULTIMODAL COMPLETE")
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