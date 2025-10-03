"""
Utility functions for transferability estimation.
"""

from .rank_metrics import (
    compute_stable_rank_from_svals,
    compute_effective_rank_from_svals,
    compute_ranks,
    feature_downproj
)

from .model_utils import (
    load_model,
    default_transform,
    resolve_final_classifier_names,
    resolve_final_classifier_module,
    find_last_weighted_module
)

from .data_utils import (
    get_dataset,
    create_balanced_subset,
    HFDatasetTorchWrapper
)

from .ground_truth import FINETUNE_ACC, get_finetune_accuracy

__all__ = [
    # Rank metrics
    'compute_stable_rank_from_svals',
    'compute_effective_rank_from_svals',
    'compute_ranks',
    'feature_downproj',
    # Model utils
    'load_model',
    'default_transform',
    'resolve_final_classifier_names',
    'resolve_final_classifier_module',
    'find_last_weighted_module',
    # Data utils
    'get_dataset',
    'create_balanced_subset',
    'HFDatasetTorchWrapper',
    # Ground truth
    'FINETUNE_ACC',
    'get_finetune_accuracy',
]