"""
REST (Rank-based Estimation of Transferability) metric.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_nested(d: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    cur = d
    for k in keys:
        if cur is None or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def min_dim_from_shape(shape) -> Optional[int]:
    """Get minimum of first two dimensions from shape."""
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    return int(min(shape[0], shape[1]))


def sr_activation(model_dict: dict, layer_key: str) -> Optional[float]:
    """Extract stable rank from activation layer array."""
    arr = model_dict.get(layer_key)
    if isinstance(arr, (list, tuple)) and len(arr) >= 1:
        return arr[0]
    return None


def compute_rest_score(
        target_data: dict,
        source_data: dict,
        model_name: str,
        gamma: float = 0.21,
        alpha: float = 0.51
) -> Optional[float]:
    """
    Compute REST score for a single model.

    REST = (1 - gamma) * G + gamma * L
    where:
        G = alpha * pen_before_weight + (1 - alpha) * clf_before_weight
        L = alpha * pen_act + (1 - alpha) * clf_act

    Args:
        target_data: Target dataset features for the model
        source_data: Source dataset features for the model
        model_name: Name of the model
        gamma: Weight between geometry (G) and activation shift (L)
        alpha: Weight between penultimate and classifier features

    Returns:
        REST score or None if features are missing
    """
    tgt_model = target_data.get(model_name)
    src_model = source_data.get(model_name)

    if tgt_model is None or src_model is None:
        return None

    # Activation dimensions from target
    act_pen_shape = get_nested(tgt_model, "activation_space_shape", "penultimate_layer")
    act_cls_shape = get_nested(tgt_model, "activation_space_shape", "classifier_layer")
    act_pen_dim = min_dim_from_shape(act_pen_shape)
    act_cls_dim = min_dim_from_shape(act_cls_shape)

    # Activation stable ranks
    tgt_pen_sr = sr_activation(tgt_model, "penultimate_layer")
    src_pen_sr = sr_activation(src_model, "penultimate_layer")
    tgt_cls_sr = sr_activation(tgt_model, "classifier_layer")
    src_cls_sr = sr_activation(src_model, "classifier_layer")

    required = {
        "act_pen_dim": act_pen_dim,
        "act_cls_dim": act_cls_dim,
        "tgt_pen_sr": tgt_pen_sr,
        "src_pen_sr": src_pen_sr,
        "tgt_cls_sr": tgt_cls_sr,
        "src_cls_sr": src_cls_sr,
    }

    if any(v is None for v in required.values()):
        return None

    # Activation shifts (normalized by target dims)
    pen_act = (tgt_pen_sr / act_pen_dim) - (src_pen_sr / act_pen_dim)
    clf_act = (tgt_cls_sr / act_cls_dim) - (src_cls_sr / act_cls_dim)

    # Weight ranks (before adaptation)
    b_pen_sr = get_nested(tgt_model, "weight_penultimate", "stable_rank")
    b_pen_shape = get_nested(tgt_model, "weight_penultimate", "matrix_shape")
    b_pen_dim = min_dim_from_shape(b_pen_shape)

    b_cls_sr = get_nested(tgt_model, "weight_classifier", "stable_rank")
    b_cls_shape = get_nested(tgt_model, "weight_classifier", "matrix_shape")
    b_cls_dim = min_dim_from_shape(b_cls_shape)

    if b_pen_sr is None or b_pen_dim in (None, 0) or b_cls_sr is None or b_cls_dim in (None, 0):
        return None

    pen_before_weight = b_pen_sr / b_pen_dim
    clf_before_weight = b_cls_sr / b_cls_dim

    # Compute REST score
    G = alpha * pen_before_weight + (1 - alpha) * clf_before_weight
    L = alpha * pen_act + (1 - alpha) * clf_act
    rest_score = (1 - gamma) * G + gamma * L

    return rest_score


def compute_rest_for_dataset(
        target_file: str,
        source_file: str,
        gamma: float = 0.21,
        alpha: float = 0.51
) -> Dict[str, float]:
    """
    Compute REST scores for all models in a target dataset.

    Args:
        target_file: Path to target dataset JSON
        source_file: Path to source dataset JSON
        gamma: Weight parameter for REST
        alpha: Weight parameter for REST

    Returns:
        Dictionary mapping model names to REST scores
    """
    with open(target_file, 'r') as f:
        target_data = json.load(f)

    with open(source_file, 'r') as f:
        source_data = json.load(f)

    scores = {}
    common_models = set(target_data.keys()) & set(source_data.keys())

    for model_name in common_models:
        score = compute_rest_score(target_data, source_data, model_name, gamma, alpha)
        if score is not None:
            scores[model_name] = score

    return scores