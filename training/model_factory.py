# model_factory.py

"""
Factory functions for building image classification backbones.

Supported models
----------------
- swin_tiny
- vgg16
- efficientnet_b0

Notes
-----
- All models output a single logit for binary classification.
- Used by train_singlemode_oof.py / test_singlemode.py
"""

from typing import List

import timm
import torch.nn as nn


MODEL_NAME_MAP = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "vgg16": "vgg16",
    "efficientnet_b0": "efficientnet_b0",
}


def list_supported_models() -> List[str]:
    return sorted(MODEL_NAME_MAP.keys())


def normalize_model_name(model_name: str) -> str:
    if model_name is None:
        raise ValueError("model_name cannot be None")
    model_name = str(model_name).strip().lower()
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unsupported model_name={model_name}. "
            f"Supported: {list_supported_models()}"
        )
    return model_name


def get_backbone_name(model_name: str) -> str:
    model_name = normalize_model_name(model_name)
    return MODEL_NAME_MAP[model_name]


def create_model(
    model_name: str,
    num_classes: int = 1,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    model_name = normalize_model_name(model_name)
    backbone_name = MODEL_NAME_MAP[model_name]

    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model