"""
Model loading and architecture utilities.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from typing import Optional, Set, Tuple


def load_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Load a pre-trained model from torchvision.

    Args:
        model_name: Name of the model architecture
        pretrained: Whether to load pre-trained weights

    Returns:
        PyTorch model
    """
    name = model_name.lower()

    if name == "inception_v3":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None,
            aux_logits=True
        )
        model.aux_logits = False
        return model

    if name == "googlenet":
        model = models.googlenet(
            weights=models.GoogLeNet_Weights.DEFAULT if pretrained else None,
            aux_logits=True
        )
        model.aux_logits = False
        return model

    model_map = {
        "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        "mnasnet1_0": (models.mnasnet1_0, models.MNASNet1_0_Weights.DEFAULT),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        "densenet169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
        "densenet201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
    }

    if name in model_map:
        model_fn, weights = model_map[name]
        return model_fn(weights=weights if pretrained else None)

    raise ValueError(f"Unknown model: {model_name}")


def default_transform(model_name: str) -> T.Compose:
    """
    Get default image transformation for a model.

    Args:
        model_name: Name of the model

    Returns:
        Torchvision Compose transform
    """
    size = (299, 299) if model_name == "inception_v3" else (224, 224)
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _find_module_name(model: nn.Module, target: nn.Module) -> Optional[str]:
    """Find the name of a module in a model."""
    for name, mod in model.named_modules():
        if mod is target:
            return name
    return None


def resolve_final_classifier_names(model: nn.Module) -> Set[str]:
    """
    Identify the names of final classifier layers in a model.

    Args:
        model: PyTorch model

    Returns:
        Set of classifier module names
    """
    names: Set[str] = set()

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        n = _find_module_name(model, model.fc)
        if n:
            names.add(n)

    if hasattr(model, "classifier"):
        clf = getattr(model, "classifier")
        if isinstance(clf, nn.Linear):
            n = _find_module_name(model, clf)
            if n:
                names.add(n)
        elif isinstance(clf, nn.Sequential) and len(clf) > 0:
            last = clf[-1]
            if isinstance(last, nn.Linear):
                n = _find_module_name(model, last)
                if n:
                    names.add(n)

    return names


def resolve_final_classifier_module(model: nn.Module) -> Optional[nn.Linear]:
    """
    Get the final classifier module from a model.

    Args:
        model: PyTorch model

    Returns:
        Final Linear layer or None
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc

    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            return clf
        if isinstance(clf, nn.Sequential) and len(clf) > 0 and isinstance(clf[-1], nn.Linear):
            return clf[-1]

    return None


def find_last_weighted_module(
        model: nn.Module,
        exclude_names: Set[str]
) -> Tuple[Optional[nn.Module], Optional[str]]:
    """
    Find the last module with weights that is not in the excluded set.

    Args:
        model: PyTorch model
        exclude_names: Set of module names to exclude

    Returns:
        Tuple of (module, module_name) or (None, None)
    """
    last_mod, last_name = None, None

    for name, m in model.named_modules():
        if name in exclude_names:
            continue
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            W = getattr(m, "weight", None)
            if isinstance(W, torch.Tensor):
                last_mod, last_name = m, name

    return last_mod, last_name