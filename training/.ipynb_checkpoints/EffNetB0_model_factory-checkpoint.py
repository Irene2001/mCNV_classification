# EffNetB0_model_factory.py
"""
Factory functions for building image classification backbones.
Enhanced for EfficientNet-B0 Partial Finetuning.
"""

from typing import List, Optional
import timm
import torch.nn as nn

# Model name mapping ensures that the model points to the implementation of the TIMM standard.
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
    drop_path_rate: Optional[float] = None, # Random depth for EfficientNet
) -> nn.Module:
    model_name = normalize_model_name(model_name)
    backbone_name = MODEL_NAME_MAP[model_name]

    # For EfficientNetB0, timm will structure it into model.blocks.
    # Drop_rate corresponds to the dropout of the classifier head.
    # Drop_path_rate corresponds to the Stochastic Depth within MBConv.
    
    kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_rate": drop_rate,
    }
    
    if "efficientnet" in model_name and drop_path_rate is not None:
        kwargs["drop_path_rate"] = drop_path_rate

    model = timm.create_model(backbone_name, **kwargs)

    return model