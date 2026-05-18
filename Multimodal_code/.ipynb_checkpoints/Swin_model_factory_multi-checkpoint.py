# Swin_model_factory_multi.py
#
# Multimodal-specific model factory.
# Purpose:
#   Build models for 224×448 OCT–OCTA3 vertically concatenated images.
#
# Important:
#   For Swin Tiny, img_size must support non-square input:
#       img_size=(448, 224)  # height, width
#
# This file is separated from the original singlemode factory to avoid
# affecting OCT0 / OCT1 / OCTA3 single-modality experiments.

import timm
import torch.nn as nn


def normalize_model_name(model_name: str) -> str:
    name = str(model_name).lower().strip()

    alias = {
        "swin": "swin_tiny",
        "swin_tiny": "swin_tiny",
        "swin-tiny": "swin_tiny",
        "swin_t": "swin_tiny",

        "vgg16": "vgg16",
        "vgg": "vgg16",

        "efficientnet_b0": "efficientnet_b0",
        "efficientnet-b0": "efficientnet_b0",
        "effb0": "efficientnet_b0",
    }

    if name not in alias:
        raise ValueError(
            f"Unsupported model_name={model_name}. "
            f"Supported: swin_tiny, vgg16, efficientnet_b0"
        )

    return alias[name]


def get_backbone_name(model_name: str) -> str:
    name = normalize_model_name(model_name)

    if name == "swin_tiny":
        return "swin_tiny_patch4_window7_224"
    if name == "vgg16":
        return "vgg16"
    if name == "efficientnet_b0":
        return "efficientnet_b0"

    raise ValueError(f"Unsupported model_name={model_name}")


def _replace_classifier_if_needed(model: nn.Module, num_classes: int):
    """
    Safety replacement for classifier/head/fc if timm does not already
    adapt it through num_classes.
    """
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        if model.head.out_features != num_classes:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)

    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        if model.classifier.out_features != num_classes:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != num_classes:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    return model


def create_model(
    model_name: str,
    num_classes: int = 1,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    img_size=224,
):
    """
    Create model for multimodal OCT–OCTA3 image classification.

    Parameters
    ----------
    model_name:
        swin_tiny / vgg16 / efficientnet_b0

    num_classes:
        For binary BCEWithLogitsLoss, use num_classes=1.

    pretrained:
        Use ImageNet pretrained weights if True.

    drop_rate:
        Dropout rate passed to timm where supported.

    img_size:
        For Swin multimodal input, use:
            img_size=(448, 224)
        where format is (height, width).
    """
    name = normalize_model_name(model_name)

    if name == "swin_tiny":
        # Important:
        # timm Swin checks input H/W against model.patch_embed.img_size.
        # Therefore img_size must be set to (448, 224) for multimodal images.
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            img_size=img_size,
        )
        return _replace_classifier_if_needed(model, num_classes)

    if name == "vgg16":
        model = timm.create_model(
            "vgg16",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        return _replace_classifier_if_needed(model, num_classes)

    if name == "efficientnet_b0":
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        return _replace_classifier_if_needed(model, num_classes)

    raise ValueError(f"Unsupported model_name={model_name}")