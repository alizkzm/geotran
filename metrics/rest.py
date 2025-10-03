"""
REST (Rank-based Estimation of Transferability) metric.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


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


def compute_features_for_model(
    target_data: dict,
    source_data: dict,
    model_name: str
) -> Optional[Dict[str, float]]:
    """
    Extract all REST features for a single model.
    
    Returns:
        Dictionary with keys: pen_act, clf_act, pen_before_weight, clf_before_weight
        or None if features are missing
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

    return {
        "pen_act": pen_act,
        "clf_act": clf_act,
        "pen_before_weight": pen_before_weight,
        "clf_before_weight": clf_before_weight
    }


def compute_rest_for_dataset(
    target_file: str,
    source_file: str,
    gamma: float = 0.21,
    alpha: float = 0.51
) -> Dict[str, float]:
    """
    Compute REST scores for all models in a target dataset with z-score normalization.
    
    Args:
        target_file: Path to target dataset JSON
        source_file: Path to source dataset JSON
        gamma: Weight parameter for REST (between geometry G and activation shift L)
        alpha: Weight parameter for REST (between penultimate and classifier)
        
    Returns:
        Dictionary mapping model names to REST scores
    """
    with open(target_file, 'r') as f:
        target_data = json.load(f)

    with open(source_file, 'r') as f:
        source_data = json.load(f)

    # Extract features for all models
    features_dict = {}
    common_models = set(target_data.keys()) & set(source_data.keys())
    
    for model_name in common_models:
        features = compute_features_for_model(target_data, source_data, model_name)
        if features is not None:
            features_dict[model_name] = features
    
    if len(features_dict) < 2:
        # Need at least 2 models for z-score normalization
        return {}
    
    # Collect all feature values for z-score normalization
    feature_names = ["pen_act", "clf_act", "pen_before_weight", "clf_before_weight"]
    feature_arrays = {name: [] for name in feature_names}
    model_list = []
    
    for model_name, features in features_dict.items():
        model_list.append(model_name)
        for name in feature_names:
            feature_arrays[name].append(features[name])
    
    # Convert to numpy arrays and compute z-scores
    z_scores = {}
    for name in feature_names:
        arr = np.array(feature_arrays[name])
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)  # Use sample std (n-1)
        
        if std < 1e-10:  # Avoid division by zero
            z_scores[name] = np.zeros_like(arr)
        else:
            z_scores[name] = (arr - mean) / std
    
    # Compute REST scores using z-normalized features
    rest_scores = {}
    for idx, model_name in enumerate(model_list):
        # Get z-scored features for this model
        pen_act_z = z_scores["pen_act"][idx]
        clf_act_z = z_scores["clf_act"][idx]
        pen_weight_z = z_scores["pen_before_weight"][idx]
        clf_weight_z = z_scores["clf_before_weight"][idx]
        
        # Compute REST score
        G = alpha * pen_weight_z + (1 - alpha) * clf_weight_z
        L = alpha * pen_act_z + (1 - alpha) * clf_act_z
        rest_score = (1 - gamma) * G + gamma * L
        
        rest_scores[model_name] = float(rest_score)
    
    return rest_scores
