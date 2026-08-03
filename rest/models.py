from typing import Optional, Set

import torch.nn as nn
import torchvision.models as tvm

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


CNN_MODELS = {
    "googlenet":    lambda: tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT, aux_logits=True),
    "mobilenet_v2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT),
    "mnasnet1_0":   lambda: tvm.mnasnet1_0(weights=tvm.MNASNet1_0_Weights.DEFAULT),
    "densenet121":  lambda: tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT),
    "densenet169":  lambda: tvm.densenet169(weights=tvm.DenseNet169_Weights.DEFAULT),
    "densenet201":  lambda: tvm.densenet201(weights=tvm.DenseNet201_Weights.DEFAULT),
    "resnet34":     lambda: tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT),
    "resnet50":     lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT),
    "resnet101":    lambda: tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT),
    "resnet152":    lambda: tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT),
}

# self-supervised transformers are loaded head-less (num_classes=0)
SSL_VIT_KEYS = {
    "mae_vitb16", "mae_vitl16", "dino_vitb16", "dino_vits8", "dino_vits16",
    "mocov3_vitb16", "mocov3_vits16", "simmim_vitb16",
}

# each entry is a list of timm identifiers tried in order
VIT_MODELS = {
    "mae_vitb16":    ["vit_base_patch16_224.mae"],
    "mae_vitl16":    ["vit_large_patch16_224.mae"],
    "dino_vitb16":   ["vit_base_patch16_224.dino"],
    "dino_vits8":    ["vit_small_patch8_224.dino"],
    "dino_vits16":   ["vit_small_patch16_224.dino"],
    "mocov3_vitb16": ["vit_base_patch16_224.mocov3", "vit_base_patch16_224.dino"],
    "mocov3_vits16": ["vit_small_patch16_224.mocov3", "vit_small_patch16_224.dino"],
    "simmim_vitb16": ["vit_base_patch16_224.simmim", "vit_base_patch16_224.mae"],
    "vit_t_16":      ["vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_tiny_patch16_224"],
    "vit_s_16":      ["vit_small_patch16_224.augreg_in21k_ft_in1k", "vit_small_patch16_224"],
    "vit_b_16":      ["vit_base_patch16_224.augreg_in21k_ft_in1k", "vit_base_patch16_224"],
    "pvtv2_b2":      ["pvt_v2_b2"],
    "pvt_t":         ["pvt_v2_b0", "pvt_tiny"],
    "pvt_s":         ["pvt_v2_b1", "pvt_small"],
    "pvt_m":         ["pvt_v2_b3", "pvt_medium"],
    "swin_t":        ["swin_tiny_patch4_window7_224.ms_in1k", "swin_tiny_patch4_window7_224"],
}


def load_model(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name in CNN_MODELS:
        model = CNN_MODELS[name]()
        if name == "googlenet":
            model.aux_logits = False
        return model
    if name in VIT_MODELS:
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for transformer models: pip install timm")
        kwargs = {"pretrained": True}
        if name in SSL_VIT_KEYS:
            kwargs["num_classes"] = 0
        last_error = None
        for candidate in VIT_MODELS[name]:
            try:
                return timm.create_model(candidate, **kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"could not load '{model_name}' from timm: {last_error}")
    raise ValueError(f"Unknown model: {model_name}")


def find_module_name(model: nn.Module, target: nn.Module) -> Optional[str]:
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def resolve_classifier(model: nn.Module) -> Optional[nn.Linear]:
    """Final linear head across torchvision CNNs, torchvision ViTs, timm ViTs and timm Swin/PVT."""
    heads = getattr(model, "heads", None)
    if heads is not None and isinstance(getattr(heads, "head", None), nn.Linear):
        return heads.head
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return head
    if head is not None and isinstance(getattr(head, "fc", None), nn.Linear):
        return head.fc
    if isinstance(getattr(model, "fc", None), nn.Linear):
        return model.fc
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return classifier
    if isinstance(classifier, nn.Sequential) and len(classifier) and isinstance(classifier[-1], nn.Linear):
        return classifier[-1]
    return None


def resolve_classifier_names(model: nn.Module) -> Set[str]:
    module = resolve_classifier(model)
    name = find_module_name(model, module) if module is not None else None
    return {name} if name else set()


def find_last_mlp_fc2(model: nn.Module) -> Optional[nn.Module]:
    """Penultimate weight matrix for head-less (self-supervised) transformers."""
    candidates = [
        m for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and "mlp" in n.lower()
        and ("fc2" in n.lower() or n.lower().endswith(".3"))
    ]
    return candidates[-1] if candidates else None


def find_last_weighted_module(model: nn.Module, exclude: Set[str]):
    candidates = [
        (n, m) for n, m in model.named_modules()
        if n not in exclude and isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
    ]
    return (candidates[-1][1], candidates[-1][0]) if candidates else (None, None)
