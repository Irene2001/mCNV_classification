# VGG16_model_factory.py

"""
Model factory for mCNV single-modality binary classification.
Supports swin_tiny (FULL_FINETUNE / LLRD_FULL) and vgg16 (FIXED_BACKBONE partial unfreeze).

Exported symbols (mirrors original model_factory.py interface):
    create_model(model_name, num_classes, pretrained, drop_rate) -> nn.Module
    normalize_model_name(name)  -> str
    get_backbone_name(name)     -> str

VGG16 architecture reference:
    Simonyan & Zisserman (2014). Very Deep Convolutional Networks for Large-Scale
    Image Recognition. arXiv:1409.1556. https://arxiv.org/abs/1409.1556

VGG16 torchvision implementation:
    https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html

VGG16 partial-unfreeze strategy (Block5 + classifier):
    - Freeze: features[0..23]  (blocks 1-4, low-level texture/edge features)
    - Train : features[24..30] (block 5, high-level semantic features) + classifier
    - classifier[6] replaced: Linear(4096, 1000) -> Linear(4096, num_classes)
    - Head replacement must be done AFTER loading pretrained weights to avoid
      shape mismatch. (ref: github.com/pytorch/vision/issues/2919)

Usage in train_singlemode_oof.py:
    try:
        from training.VGG16_model_factory import (
            create_model, normalize_model_name, get_backbone_name,
        )
    except Exception:
        from VGG16_model_factory import (
            create_model, normalize_model_name, get_backbone_name,
        )
"""

import timm
import torch.nn as nn
import torchvision.models as tv_models


# ==============================================================================
# Name registry
# ==============================================================================

_NAME_MAP = {
    "swin_tiny":       "swin_tiny",
    "swintiny":        "swin_tiny",
    "swin":            "swin_tiny",
    "vgg16":           "vgg16",
    "vgg":             "vgg16",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet":    "efficientnet_b0",
}

_BACKBONE_DISPLAY = {
    "swin_tiny":       "swin_tiny_patch4_window7_224",
    "vgg16":           "vgg16 (torchvision)",
    "efficientnet_b0": "efficientnet_b0 (timm)",
}


def normalize_model_name(name: str) -> str:
    """Normalize user-supplied model name to canonical internal key."""
    key = str(name).lower().strip()
    if key not in _NAME_MAP:
        raise ValueError(
            f"Unknown model_name='{name}'. "
            f"Supported: {sorted(set(_NAME_MAP.values()))}"
        )
    return _NAME_MAP[key]


def get_backbone_name(name: str) -> str:
    """Return human-readable backbone string for logging."""
    return _BACKBONE_DISPLAY.get(normalize_model_name(name), name)


# ==============================================================================
# Model builders
# ==============================================================================

def _build_swin_tiny(num_classes: int, pretrained: bool, drop_rate: float) -> nn.Module:
    """
    Swin-Tiny via timm.
    Head: Linear(768, num_classes) — compatible with BCEWithLogitsLoss.
    """
    return timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def _build_vgg16(num_classes: int, pretrained: bool, drop_rate: float) -> nn.Module:
    """
    VGG16 via torchvision with head replacement for BCEWithLogitsLoss.

    Strategy (avoids shape mismatch from github.com/pytorch/vision/issues/2919):
      1. Load pretrained weights with original 1000-class head.
      2. Replace classifier[6]: Linear(4096, 1000) -> Linear(4096, num_classes).
      3. Optionally replace Dropout layers if drop_rate != 0.5.

    Partial-unfreeze (FIXED_BACKBONE mode, applied in train_singlemode_oof.py):
      - Freeze  : model.features[0..23]  (blocks 1-4)
      - Unfreeze: model.features[24..30] (block 5) + model.classifier[0..6]
    This function does NOT apply freezing; freezing is handled by apply_unfreeze_mode().

    VGG16 features index mapping (ref: torchvision vgg.py):
      Block 1 (64ch)  : features[0..4]
      Block 2 (128ch) : features[5..9]
      Block 3 (256ch) : features[10..16]
      Block 4 (512ch) : features[17..23]
      Block 5 (512ch) : features[24..30]  <- unfreeze this
      classifier      : [Lin(25088,4096), ReLU, Drop, Lin(4096,4096), ReLU, Drop, Lin(4096,n)]
    """
    weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tv_models.vgg16(weights=weights)

    # Replace output head (must be done after weight loading to avoid mismatch)
    in_features = model.classifier[6].in_features  # 4096
    model.classifier[6] = nn.Linear(in_features, num_classes)

    # Optionally adjust dropout rate (default torchvision VGG16 uses 0.5)
    if drop_rate != 0.5 and drop_rate >= 0.0:
        model.classifier[2] = nn.Dropout(p=drop_rate)
        model.classifier[5] = nn.Dropout(p=drop_rate)

    return model


# ==============================================================================
# Public factory
# ==============================================================================

def create_model(
    model_name: str,
    num_classes: int = 1,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    """
    Build and return a model for mCNV binary classification.

    Parameters
    ----------
    model_name  : str   Canonical or alias name (swin_tiny | vgg16 | efficientnet_b0).
    num_classes : int   Output dimension. Use 1 for BCEWithLogitsLoss (binary).
    pretrained  : bool  Load ImageNet pretrained weights.
    drop_rate   : float Dropout probability for applicable layers.

    Returns
    -------
    nn.Module  Model with output shape [B, num_classes], ready for BCEWithLogitsLoss.
    """
    name = normalize_model_name(model_name)

    if name == "swin_tiny":
        return _build_swin_tiny(num_classes, pretrained, drop_rate)

    if name == "vgg16":
        return _build_vgg16(num_classes, pretrained, drop_rate)

    raise NotImplementedError(
        f"create_model: model_name='{name}' is recognized but not yet implemented. "
        "Add a builder function and register it here."
    )