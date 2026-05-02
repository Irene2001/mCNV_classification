# SaveIndivpic_Swin_focus_scorecam_singlemode.py

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

# 1. Directory that contains test_preds.csv
TEST_EVAL_DIR = (
    "/data/Irene/SwinTransformer/Swin_Meta/outputs/test_evaluation/"
    "swin_tiny/OCT0/"
    "BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9/"
    "Best_fold2"
)

# "OCT0":  "BS16_EP100_LR2e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9" (Best_fold2),
# "OCT1":  "BS16_EP100_LR4e-06_WD0.01_FULL_FINETUNE_FL0.113_0.887_2_WSon_1_2.8" (Best_fold1),
# "OCTA3": "BS16_EP100_LR3e-06_WD0.01_FULL_FINETUNE_FL0.13_0.87_2_WSon_1_2.6" (Best_fold2),


# 2. Checkpoint root
CHECKPOINT_ROOT = ""

# 3. Model switch
ACTIVE_MODEL = "swin_tiny"

TIMM_MODEL_MAP: Dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

# 4. Visual settings
IMG_SIZE      = 224
OVERLAY_ALPHA = 0.50
COLORMAP      = "jet"
CLASS_NAMES   = ["inactive", "active"]

# 5. Score-CAM settings
BATCH_SIZE_SCORECAM = 32
CAM_THRESHOLD = 0.20

# 6. Random seed
N_RANDOM_SEED = None

# Output figure style
_TITLE_SIZE      = 14
_AXIS_LABEL_SIZE = 11
_FIG_DPI         = 300

# ==============================================================================

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

def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "swin_tiny":
        return model.norm
    raise NotImplementedError(f"Target layer not defined for '{model_name}'.")

def _to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.ndim == 3:
        B, N, C = tensor.shape
        H = W = int(N ** 0.5)
        return tensor.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

class SwinScoreCAM:
    def __init__(self, model: nn.Module, model_name: str, device: torch.device, batch_size: int = BATCH_SIZE_SCORECAM):
        self.model      = model
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self._activations: Optional[torch.Tensor] = None
        target = _get_target_layer(model, model_name)
        self._hook = target.register_forward_hook(lambda m, i, o: setattr(self, "_activations", o.detach()))

    def remove_hooks(self) -> None:
        self._hook.remove()

    @torch.no_grad()
    def generate(self, img_tensor: torch.Tensor, class_idx: int, temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        self.model.eval()
        self._activations = None
        logit = self.model(img_tensor)
        A = _to_spatial(self._activations)[0]
        C, Hf, Wf = A.shape
        T = max(temperature, 1e-6)
        prob_active = float(torch.sigmoid(logit[0, 0] / T).item())

        A_min = A.flatten(1).min(dim=1).values.view(C, 1, 1)
        A_max = A.flatten(1).max(dim=1).values.view(C, 1, 1)
        A_norm = (A - A_min) / (A_max - A_min + 1e-8)

        masks = F.interpolate(A_norm.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        bg = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        scores = torch.zeros(C, dtype=torch.float32, device=self.device)
        for start in range(0, C, self.batch_size):
            end = min(start + self.batch_size, C)
            m_batch = masks[start:end]
            masked_inputs = img_tensor * m_batch + bg * (1.0 - m_batch)
            logits_batch = self.model(masked_inputs)[:, 0]
            scores[start:end] = (logits_batch / T) if class_idx == 1 else (-logits_batch / T)

        weights = torch.softmax(scores, dim=0)
        cam = (weights.view(C, 1, 1) * A).sum(dim=0, keepdim=True)
        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        if CAM_THRESHOLD > 0.0:
            cam = cam * (cam >= CAM_THRESHOLD).float()

        cam_up = F.interpolate(cam.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        return np.clip(cam_up, 0.0, 1.0), prob_active

def get_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225]),
    ])

def overlay_heatmap(orig_pil: Image.Image, cam: np.ndarray, alpha: float = OVERLAY_ALPHA, colormap: str = COLORMAP) -> np.ndarray:
    img_np = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).convert("RGB"), dtype=np.float32) / 255.0
    heatmap = plt.get_cmap(colormap)(cam)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * img_np + alpha * heatmap, 0.0, 1.0)

def save_samples_csv(samples: Dict[str, Optional[dict]], out_path: str, run_meta: dict) -> None:
    rows = []
    for quadrant, data in samples.items():
        base = {**run_meta, "cam_method": "Score-CAM", "batch_size_scorecam": BATCH_SIZE_SCORECAM, "cam_threshold": CAM_THRESHOLD, "quadrant": quadrant}
        if data:
            base.update({
                "exam_key": data["exam_key"], "filename": data["filename"], "image_path": data["img_path"],
                "gt_label": data["gt"], "gt_class": CLASS_NAMES[data["gt"]], "pred_label": data["pred"], "pred_class": CLASS_NAMES[data["pred"]],
                "prob_active": round(data["prob_active"], 6), "prob_inactive": round(1.0 - data["prob_active"], 6),
                "logit_uncal": round(data["logit_uncal"], 6), "logit_calib": round(data["logit_calib"], 6),
                "cam1_max": round(float(data["cam1"].max()), 4), "cam0_max": round(float(data["cam0"].max()), 4),
                "panel_file": f"{quadrant}_panel.png", "note": "",
            })
        else:
            base.update({"panel_file": f"(no {quadrant} sample)", "note": "no sample in this quadrant"})
        rows.append(base)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

def _set_style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": _TITLE_SIZE, "axes.labelsize": _AXIS_LABEL_SIZE, "figure.dpi": _FIG_DPI, "savefig.dpi": _FIG_DPI})

def save_single_panel(orig_pil: Image.Image, cam0: np.ndarray, cam1: np.ndarray, quadrant: str, gt: int, pred: int, prob_active: float, out_path: str) -> None:
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(f"{quadrant} | GT={CLASS_NAMES[gt]} Pred={CLASS_NAMES[pred]} P(active)={prob_active:.3f}", fontweight="bold")
    axes[0].imshow(np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")))
    axes[0].set_title("Original", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(overlay_heatmap(orig_pil, cam0))
    axes[1].set_title("Class 0 (inactive)", fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(overlay_heatmap(orig_pil, cam1))
    axes[2].set_title("Class 1 (active)", fontweight="bold"); axes[2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def save_4x3_panel(samples: Dict[str, Optional[dict]], out_path: str, modality: str, model_name: str) -> None:
    _set_style()
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(f"Score-CAM -- {model_name.upper()} / {modality}", fontsize=_TITLE_SIZE + 2, fontweight="bold")
    for r, key in enumerate(["TP", "FP", "FN", "TN"]):
        data = samples.get(key)
        for c in range(3):
            ax = axes[r, c]; ax.axis("off")
            if not data: continue
            if c == 0:
                ax.imshow(np.array(data["orig_pil"].resize((IMG_SIZE, IMG_SIZE)).convert("RGB")))
                ax.set_title(f"{key} GT={CLASS_NAMES[data['gt']]} Pred={CLASS_NAMES[data['pred']]}\n{data['filename']}", fontsize=9, fontweight="bold")
            elif c == 1: ax.imshow(overlay_heatmap(data["orig_pil"], data["cam0"]))
            else: ax.imshow(overlay_heatmap(data["orig_pil"], data["cam1"]))
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def main() -> None:
    random.seed(N_RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed = parse_test_eval_dir(TEST_EVAL_DIR)
    model_name, project_root = ACTIVE_MODEL, parsed["project_root"]
    ckpt_path = os.path.join(resolve_ckpt_root(CHECKPOINT_ROOT, project_root), model_name, parsed["modality"], parsed["run_tag"], f"Best_fold{parsed['best_fold']}", "model_best.pth")
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(TEST_EVAL_DIR, "New_ScoreCAM", ts)
    ensure_dir(out_dir)
    
    # [新增] 建立個別檔案儲存路徑
    individual_dir = os.path.join(out_dir, "individual_images")
    ensure_dir(individual_dir)

    logf = open(os.path.join(out_dir, "scorecam_log.txt"), "w", buffering=1, encoding="utf-8")
    log_print(logf, f"Out dir: {out_dir}")

    df = pd.read_csv(os.path.join(TEST_EVAL_DIR, "test_preds.csv"))
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred_calib"] = (df["prob_calib"] >= 0.5).astype(int)
    temperature = float(df["temperature"].iloc[0])

    indices = {q: df.index[(df["y_true"] == (1 if q[0]=='T' else 0)) & (df["y_pred_calib"] == (1 if q[1]=='P' else 0))].tolist() for q in ["TP", "FP", "FN", "TN"]}
    selected = {k: random.choice(v) if v else None for k, v in indices.items()}

    model, _ = load_checkpoint_and_temperature(build_model(model_name), ckpt_path, device)
    scorecam = SwinScoreCAM(model, model_name, device)
    tfm = get_test_transform()

    samples_for_panel = {}
    img_col = {"OCT0": "oct0_image_path", "OCT1": "oct1_image_path", "OCTA3": "octa3_image_path"}[parsed["modality"]]

    for quadrant, idx in selected.items():
        if idx is None: 
            samples_for_panel[quadrant] = None
            continue
        row = df.loc[idx]
        img_path = str(row[img_col])
        if not os.path.isfile(img_path): continue

        orig_pil = Image.open(img_path).convert("RGB")
        img_tensor = tfm(orig_pil).unsqueeze(0).to(device)
        cam0, _ = scorecam.generate(img_tensor, class_idx=0, temperature=temperature)
        cam1, prob_active = scorecam.generate(img_tensor, class_idx=1, temperature=temperature)

        # --- [核心修正：儲存個別高解析圖檔] ---
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # 1. Original (224x224)
        orig_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(os.path.join(individual_dir, f"{quadrant}_0_original_{base_name}.png"))
        # 2. Class 0 Heatmap
        plt.imsave(os.path.join(individual_dir, f"{quadrant}_1_class0_inactive_{base_name}.png"), overlay_heatmap(orig_pil, cam0))
        # 3. Class 1 Heatmap
        plt.imsave(os.path.join(individual_dir, f"{quadrant}_2_class1_active_{base_name}.png"), overlay_heatmap(orig_pil, cam1))

        samples_for_panel[quadrant] = {
            "orig_pil": orig_pil, "cam0": cam0, "cam1": cam1, "gt": int(row["y_true"]), 
            "pred": int(row["y_pred_calib"]), "prob_active": prob_active, "filename": os.path.basename(img_path),
            "img_path": img_path, "exam_key": str(row["exam_key"]), "logit_uncal": float(row["logit_uncal"]), "logit_calib": float(row["logit_calib"])
        }
        save_single_panel(orig_pil, cam0, cam1, quadrant, int(row["y_true"]), int(row["y_pred_calib"]), prob_active, os.path.join(out_dir, f"{quadrant}_panel.png"))

    scorecam.remove_hooks()
    save_4x3_panel(samples_for_panel, os.path.join(out_dir, "ScoreCAM_4x3_panel.png"), parsed["modality"], model_name)
    save_samples_csv(samples_for_panel, os.path.join(out_dir, "scorecam_samples.csv"), {"run_timestamp": ts, "model_name": model_name, "modality": parsed["modality"], "run_tag": parsed["run_tag"], "best_fold": parsed["best_fold"], "temperature": temperature, "target_layer": "model.norm"})
    log_print(logf, "DONE")
    logf.close()

if __name__ == "__main__":
    main()