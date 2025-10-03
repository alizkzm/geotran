#!/usr/bin/env python3
"""
Rank Measure Script - Extract transferability features from pre-trained models.

Usage:
    python rank_measure.py -s 2 -t cifar100
"""

import os
import sys
import json
import time
import warnings
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import get_dataset, create_balanced_subset
from utils.model_utils import (
    load_model, default_transform, resolve_final_classifier_names,
    resolve_final_classifier_module, find_last_weighted_module, _find_module_name
)
from utils.rank_metrics import compute_ranks, feature_downproj

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActivationExtractor:
    """Extract activations from specific layers during forward pass."""

    def __init__(
            self,
            model: nn.Module,
            device: str = "cuda",
            max_feats_per_layer: Optional[int] = 4096,
            final_exclude_names: Optional[set] = None,
            classifier_module: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        self.max_feats_per_layer = max_feats_per_layer
        self.final_exclude_names = final_exclude_names or set()
        self.penult_name: Optional[str] = None
        self.classifier_module = classifier_module
        self.classifier_inputs: List[torch.Tensor] = []
        self.classifier_outputs: List[torch.Tensor] = []

    def _keep_module_for_hook(self, name: str, module: nn.Module) -> bool:
        """Determine if a module should have a hook attached."""
        if name in self.final_exclude_names:
            return False
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.Identity)):
            return False
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            return False
        return True

    def register_hooks(self):
        """Register forward hooks on relevant modules."""

        def hook_fn(name):
            def _hook(_module, _input, output):
                x = output
                if isinstance(output, (list, tuple)):
                    x = output[0] if len(output) > 0 else None
                if not isinstance(x, torch.Tensor):
                    return
                self.penult_name = name
                B = x.shape[0]
                flat = x.view(B, -1)
                flat = feature_downproj(flat, self.max_feats_per_layer)
                self.activations[name] = flat.detach()

            return _hook

        for name, module in self.model.named_modules():
            if self._keep_module_for_hook(name, module):
                self.hooks.append(module.register_forward_hook(hook_fn(name)))

        if self.classifier_module is not None:
            def pre_hook(_m, x):
                x0 = x[0]
                if isinstance(x0, torch.Tensor):
                    B = x0.shape[0]
                    flat = x0.view(B, -1)
                    flat = feature_downproj(flat, self.max_feats_per_layer)
                    self.classifier_inputs.append(flat.detach().cpu())

            def classifier_hook(_m, _input, output):
                if isinstance(output, torch.Tensor):
                    B = output.shape[0]
                    flat = output.view(B, -1)
                    self.classifier_outputs.append(flat.detach().cpu())
                elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    out = output[0]
                    B = out.shape[0]
                    flat = out.view(B, -1)
                    self.classifier_outputs.append(flat.detach().cpu())

            self.hooks.append(self.classifier_module.register_forward_pre_hook(pre_hook))
            self.hooks.append(self.classifier_module.register_forward_hook(classifier_hook))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract_activations(self, dataloader, max_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract activations by running the model on the dataloader.

        Args:
            dataloader: DataLoader for input data
            max_samples: Maximum number of samples to process

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.model.eval()
        buf: Dict[str, List[torch.Tensor]] = defaultdict(list)
        seen = 0
        self.classifier_inputs = []
        self.classifier_outputs = []

        for images, _ in dataloader:
            if (max_samples is not None) and (seen >= max_samples):
                break
            self.activations.clear()
            images = images.to(self.device, non_blocking=True)
            _ = self.model(images)
            for lname, act in self.activations.items():
                if act is not None:
                    buf[lname].append(act.cpu())
            seen += images.size(0)

        final: Dict[str, torch.Tensor] = {}
        for lname, chunks in buf.items():
            if chunks:
                final[lname] = torch.cat(chunks, dim=0)
        return final


def _weight_matrix_from_module(mod: nn.Module):
    """Extract weight matrix from a module."""
    W = getattr(mod, "weight", None)
    if not isinstance(W, torch.Tensor):
        return None, None, None

    orig_shape = list(W.shape)

    if isinstance(mod, nn.Linear):
        mat = W.detach().float()
    elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        out = W.shape[0]
        mat = W.detach().float().view(out, -1)
    else:
        return None, orig_shape, None

    matrix_shape = list(mat.shape)
    return mat.cpu(), orig_shape, matrix_shape


def compute_weight_ranks(mod: Optional[nn.Module]) -> Dict[str, Any]:
    """Compute ranks for module weights."""
    if mod is None:
        return {"stable_rank": None, "effective_rank": None, "shape": None, "matrix_shape": None}

    mat, orig_shape, mat_shape = _weight_matrix_from_module(mod)
    if mat is None:
        return {"stable_rank": None, "effective_rank": None, "shape": orig_shape, "matrix_shape": mat_shape}

    sr, er = compute_ranks(mat)
    return {"stable_rank": sr, "effective_rank": er, "shape": orig_shape, "matrix_shape": mat_shape}


def calculate_transferability_scores(
        model_name: str,
        dataset_name: str,
        images_per_class: int = 2,
        device: str = "cuda",
        batch_size: int = 32,
        max_feats_per_layer: Optional[int] = 4096,
        max_samples: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calculate transferability scores for a model on a target dataset.

    Args:
        model_name: Name of the pre-trained model
        dataset_name: Name of the target dataset
        images_per_class: Number of images per class to use
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for data loading
        max_feats_per_layer: Maximum features per layer (for downprojection)
        max_samples: Maximum samples to process

    Returns:
        Dictionary containing transferability scores and metadata
    """
    print(f"Processing {model_name} on {dataset_name}...")

    transform = default_transform(model_name)
    model = load_model(model_name, pretrained=True)

    final_exclude_names = resolve_final_classifier_names(model)
    clf_mod = resolve_final_classifier_module(model)

    dataset = get_dataset(dataset_name, transform, split="train")
    subset = create_balanced_subset(dataset, images_per_class=images_per_class)

    dataloader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda")
    )

    extractor = ActivationExtractor(
        model,
        device=device,
        max_feats_per_layer=max_feats_per_layer,
        final_exclude_names=final_exclude_names,
        classifier_module=clf_mod,
    )
    extractor.register_hooks()

    try:
        activations = extractor.extract_activations(dataloader, max_samples=max_samples)
        penult_name = extractor.penult_name
    finally:
        extractor.remove_hooks()

    if not activations and not extractor.classifier_inputs and not extractor.classifier_outputs:
        print(f"No activations extracted for {model_name} on {dataset_name}")
        return None

    # Penultimate activations
    penult_source = "last_non_excluded"
    penult = None
    if extractor.classifier_inputs:
        penult = torch.cat(extractor.classifier_inputs, dim=0)
        penult_source = "classifier_pre_hook"
    else:
        if penult_name and (penult_name in activations):
            penult = activations[penult_name]
        else:
            last_key = next(reversed(activations))
            penult = activations[last_key]

    C = (clf_mod.out_features if clf_mod is not None else None)
    if C is not None and penult is not None and penult.shape[1] == C:
        print(f"[WARN] Penultimate dims ({penult.shape[1]}) == num_classes ({C}); likely logits.")

    penult_stable_rank, penult_effective_rank = 0.0, 0.0
    if penult is not None:
        X_pen = penult.float().T
        penult_stable_rank, penult_effective_rank = compute_ranks(X_pen)

    # Classifier logits activations
    classifier_stable_rank, classifier_effective_rank = 0.0, 0.0
    classifier_source = "none"
    classifier_tensor = None
    if extractor.classifier_outputs:
        classifier_tensor = torch.cat(extractor.classifier_outputs, dim=0)
        classifier_source = "classifier_hook"
        X_clf = classifier_tensor.float().T
        classifier_stable_rank, classifier_effective_rank = compute_ranks(X_clf)

    # Weight ranks (penultimate & classifier)
    penult_weight_mod, penult_weight_name = find_last_weighted_module(model, final_exclude_names)
    weight_penult = compute_weight_ranks(penult_weight_mod)
    weight_classifier = compute_weight_ranks(clf_mod)

    result = {
        "penultimate_layer": [penult_stable_rank, penult_effective_rank],
        "classifier_layer": [classifier_stable_rank, classifier_effective_rank],
        "penultimate_layer_source": penult_source,
        "penultimate_weight_layer_name": penult_weight_name,
        "classifier_layer_source": classifier_source,
        "images_used_count": len(subset),
        "activation_space_shape": {
            "penultimate_layer": list(penult.shape) if penult is not None else None,
            "classifier_layer": list(classifier_tensor.shape) if classifier_tensor is not None else None,
        },
        "classifier_layer_name": _find_module_name(model, clf_mod) if clf_mod is not None else None,
        "penultimate_layer_name": penult_name,
        "weight_classifier": weight_classifier,
        "weight_penultimate": weight_penult,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract transferability rank measures")
    parser.add_argument("-s", "--samples", type=int, default=2,
                        help="Images per class (default: 2)")
    parser.add_argument("-t", "--target", type=str, default="cifar100",
                        help="Target dataset (default: cifar100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--max-feats", type=int, default=4096,
                        help="Max features per layer (default: 4096)")

    args = parser.parse_args()

    set_seed(123)

    models_hub = [
        "mobilenet_v2", "mnasnet1_0",
        "densenet121", "densenet169", "densenet201",
        "resnet34", "resnet50", "resnet101", "resnet152",
        "googlenet",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Target dataset: {args.target}")
    print(f"Images per class: {args.samples}")

    os.makedirs("transferability_results", exist_ok=True)

    dataset_results: Dict[str, Any] = {}

    for model_name in models_hub:
        try:
            start = time.time()
            result = calculate_transferability_scores(
                model_name,
                args.target,
                images_per_class=args.samples,
                device=device,
                batch_size=args.batch_size,
                max_feats_per_layer=args.max_feats,
                max_samples=None,
            )

            if result is not None:
                dataset_results[model_name] = result

                wclf = result["weight_classifier"]
                wpen = result["weight_penultimate"]

                print(
                    f"  {model_name}: "
                    f"Act-Pen({result['penultimate_layer_source']}): SR={result['penultimate_layer'][0]:.3f}, "
                    f"ER={result['penultimate_layer'][1]:.3f} | "
                    f"Act-Clf({result['classifier_layer_source']}): SR={result['classifier_layer'][0]:.3f}, "
                    f"ER={result['classifier_layer'][1]:.3f}\n"
                    f"      Weights: "
                    f"Penult sr/er={wpen['stable_rank']:.3f},{wpen['effective_rank']:.3f} "
                    f"shape={wpen['shape']} flat={wpen['matrix_shape']} | "
                    f"Classifier sr/er={wclf['stable_rank']:.3f},{wclf['effective_rank']:.3f} "
                    f"shape={wclf['shape']} flat={wclf['matrix_shape']}"
                )

            elapsed = time.time() - start
            print(f"  Duration: {elapsed:.2f}s\n")

        except Exception as e:
            print(f"  Error processing {model_name} on {args.target}: {e}\n")
            continue

    if dataset_results:
        out_file = f"transferability_results/{args.target}_transferability_scores.json"
        with open(out_file, "w") as f:
            json.dump(dataset_results, f, indent=2)
        print(f"Saved results to {out_file}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()