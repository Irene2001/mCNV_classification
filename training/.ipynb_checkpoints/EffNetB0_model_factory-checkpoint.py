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
    drop_path_rate: Optional[float] = None, # 針對 EfficientNet 的隨機深度
) -> nn.Module:
    model_name = normalize_model_name(model_name)
    backbone_name = MODEL_NAME_MAP[model_name]

    # 針對 EfficientNetB0，timm 會將其結構化為 model.blocks
    # 這裡的 drop_rate 對應的是 classifier head 的 dropout
    # 這裡的 drop_path_rate 對應的是 MBConv 內部的 Stochastic Depth
    
    kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_rate": drop_rate,
    }
    
    # 如果是 EfficientNet 且有提供隨機深度設定（醫療影像建議 0.1-0.2）
    if "efficientnet" in model_name and drop_path_rate is not None:
        kwargs["drop_path_rate"] = drop_path_rate

    model = timm.create_model(backbone_name, **kwargs)

    return model