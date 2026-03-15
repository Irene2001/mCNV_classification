# test_singlemode.py

"""
Evaluate one single-modality base learner on the independent test set.

Design
------
- Load fold-specific calibrated checkpoints saved by train_singlemode_oof.py
- Use all 5 fold models for test-time ensemble
- Apply per-checkpoint temperature if present
- Export image-level test predictions and metrics

Example
-------
python test_singlemode.py --model_name swin_tiny --modality OCT0
python test_singlemode.py --model_name swin_tiny --modality OCT1
python test_singlemode.py --model_name swin_tiny --modality OCTA3
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

from training.model_factory import create_model, get_backbone_name, normalize_model_name


# ===================== CONFIG =====================
PROJECT_ROOT_DIR = "/data/Irene/SwinTransformer/Swin_OOF"

MANIFEST_CSV = os.path.join(
    PROJECT_ROOT_DIR,
    "outputs",
    "manifests",
    "global_patient_fold_split",
    "all_images_manifest_with_fold.csv",
)

CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT_DIR, "checkpoints")
RESULTS_ROOT = os.path.join(PROJECT_ROOT_DIR, "outputs", "results")
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

MODALITY_TO_IMAGE_SLOT = {
    "OCT0": "OCT0",
    "OCT1": "OCT1",
    "OCTA3": "OCTA3",
}

MODALITY_TO_CHECKPOINT_DIR = {
    "OCT0": "OCT0_Horizontal",
    "OCT1": "OCT1_Vertical",
    "OCTA3": "OCTA3",
}


# ===================== UTILS =====================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ===================== DATASET =====================
class ManifestTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        meta = {
            "dataset_index": int(row["dataset_index"]),
            "patient_id": str(row["patient_id"]),
            "pair_id": int(row["pair_id"]) if "pair_id" in row and pd.notna(row["pair_id"]) else -1,
            "pair_key": str(row["pair_key"]) if "pair_key" in row and pd.notna(row["pair_key"]) else "",
            "image_path": str(row["image_path"]),
            "image_slot": str(row["image_slot"]),
            "y_true": int(row["y_true"]),
        }
        return image, int(row["y_true"]), meta


# ===================== HELPERS =====================
def load_test_manifest_for_modality(manifest_csv: str, modality: str) -> pd.DataFrame:
    if modality not in MODALITY_TO_IMAGE_SLOT:
        raise ValueError(f"Unsupported modality={modality}")

    image_slot = MODALITY_TO_IMAGE_SLOT[modality]
    df = pd.read_csv(manifest_csv)

    required_cols = [
        "dataset_index", "split_set", "image_path", "image_slot",
        "y_true", "patient_id", "pair_key", "fold_id"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    if "pair_id" not in df.columns:
        df["pair_id"] = -1

    df = df[
        (df["split_set"] == "test") &
        (df["fold_id"] == 0) &
        (df["image_slot"] == image_slot)
    ].copy()

    if len(df) == 0:
        raise RuntimeError(f"No test rows found for modality={modality}")

    df["y_true"] = df["y_true"].astype(int)
    return df.reset_index(drop=True)


def get_checkpoint_dir(model_name: str, modality: str) -> str:
    model_name = normalize_model_name(model_name)
    ckpt_modality_dir = MODALITY_TO_CHECKPOINT_DIR[modality]
    return os.path.join(CHECKPOINT_ROOT, model_name, ckpt_modality_dir)


def build_test_loader(df: pd.DataFrame, img_size: int, batch_size: int, num_workers: int) -> DataLoader:
    tf_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = ManifestTestDataset(df, transform=tf_test)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def load_fold_models(model_name: str, modality: str, device: torch.device) -> List[Tuple[nn.Module, float, str]]:
    ckpt_dir = get_checkpoint_dir(model_name, modality)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    fold_dirs = sorted([
        d for d in os.listdir(ckpt_dir)
        if d.startswith("fold") and os.path.isdir(os.path.join(ckpt_dir, d))
    ])
    if len(fold_dirs) == 0:
        raise RuntimeError(f"No fold directories found in: {ckpt_dir}")

    models = []
    for fold_name in fold_dirs:
        fold_path = os.path.join(ckpt_dir, fold_name)

        # prefer calibrated checkpoint, fallback to uncalibrated
        calibrated_path = os.path.join(fold_path, "best_model_calibrated.pth")
        uncalibrated_path = os.path.join(fold_path, "best_model_uncalibrated.pth")

        if os.path.isfile(calibrated_path):
            ckpt_path = calibrated_path
        elif os.path.isfile(uncalibrated_path):
            ckpt_path = uncalibrated_path
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {fold_path}. "
                f"Expected best_model_calibrated.pth or best_model_uncalibrated.pth"
            )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = create_model(
            model_name=model_name,
            num_classes=1,
            pretrained=False,
            drop_rate=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        temperature = float(ckpt.get("temperature", 1.0))
        models.append((model, temperature, ckpt_path))

    return models


def run_test_inference(
    models: List[Tuple[nn.Module, float, str]],
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows = []

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device, non_blocking=True)

            probs_per_model = []
            logits_per_model = []

            for model, temperature, _ in models:
                z = model(x) / float(temperature)
                p = torch.sigmoid(z.view(-1))
                probs_per_model.append(p.detach().cpu().numpy())
                logits_per_model.append(z.view(-1).detach().cpu().numpy())

            probs_per_model = np.stack(probs_per_model, axis=0)   # [n_models, B]
            logits_per_model = np.stack(logits_per_model, axis=0) # [n_models, B]

            prob_mean = probs_per_model.mean(axis=0)
            logit_mean = logits_per_model.mean(axis=0)
            pred_mean = (prob_mean >= 0.5).astype(int)

            batch_size = len(prob_mean)
            for i in range(batch_size):
                rows.append({
                    "dataset_index": int(meta["dataset_index"][i]),
                    "patient_id": str(meta["patient_id"][i]),
                    "pair_id": int(meta["pair_id"][i]),
                    "pair_key": str(meta["pair_key"][i]),
                    "image_path": str(meta["image_path"][i]),
                    "image_slot": str(meta["image_slot"][i]),
                    "y_true": int(meta["y_true"][i]),
                    "ensemble_logit": float(logit_mean[i]),
                    "ensemble_prob": float(prob_mean[i]),
                    "ensemble_pred": int(pred_mean[i]),
                    "num_fold_models": int(len(models)),
                })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["dataset_index"]).reset_index(drop=True)
    return out_df


# ===================== MAIN =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["swin_tiny", "vgg16", "efficientnet_b0"])
    parser.add_argument("--modality", type=str, required=True, choices=["OCT0", "OCT1", "OCTA3"])
    parser.add_argument("--manifest_csv", type=str, default=MANIFEST_CSV)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.model_name)
    modality = args.modality

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(RESULTS_ROOT, model_name)
    ensure_dir(results_dir)

    test_df = load_test_manifest_for_modality(args.manifest_csv, modality)
    test_loader = build_test_loader(test_df, args.img_size, args.batch_size, args.num_workers)
    models = load_fold_models(model_name, modality, device)

    pred_df = run_test_inference(models, test_loader, device)

    metrics = compute_binary_metrics(
        pred_df["y_true"].astype(int).values,
        pred_df["ensemble_prob"].astype(float).values,
        threshold=0.5,
    )

    prediction_csv = os.path.join(results_dir, f"{model_name}_{modality.lower()}_test_predictions.csv")
    metrics_json = os.path.join(results_dir, f"{model_name}_{modality.lower()}_test_metrics.json")

    pred_df.to_csv(prediction_csv, index=False, encoding="utf-8-sig")

    summary = {
        "model_name": model_name,
        "backbone_name": get_backbone_name(model_name),
        "modality": modality,
        "num_test_images": int(len(pred_df)),
        "num_test_patients": int(pred_df["patient_id"].astype(str).nunique()),
        "num_fold_models": int(len(models)),
        "checkpoint_paths": [x[2] for x in models],
        "metrics": metrics,
        "prediction_csv": prediction_csv,
    }
    save_json(metrics_json, summary)

    print("\n[SAVED]")
    print(f"Prediction CSV : {prediction_csv}")
    print(f"Metrics JSON   : {metrics_json}")


if __name__ == "__main__":
    main()