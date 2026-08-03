from collections import defaultdict
from typing import Optional, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import create_random_subset, default_transform, get_dataset
from .models import (
    find_last_mlp_fc2, find_last_weighted_module, load_model,
    resolve_classifier, resolve_classifier_names,
)


def feature_downproj(x: torch.Tensor, max_feat: Optional[int] = 4096) -> torch.Tensor:
    """Random projection to at most `max_feat` dimensions (deterministic)."""
    if max_feat is None:
        return x
    _, dim = x.shape
    if dim <= max_feat:
        return x
    generator = torch.Generator(device=x.device).manual_seed(12345)
    projection = torch.randn(dim, max_feat, generator=generator, device=x.device) / np.sqrt(max_feat)
    return x @ projection


def stable_rank(matrix: torch.Tensor, eps: float = 1e-8) -> float:
    """||X||_F^2 / ||X||_2^2."""
    if matrix.numel() == 0:
        return 0.0
    svals = torch.linalg.svdvals(matrix)
    return (torch.sum(svals ** 2) / (torch.max(svals) ** 2 + eps)).item()


def weight_ranks(module: Optional[nn.Module]) -> dict:
    if module is None:
        return {"stable_rank": None, "shape": None, "matrix_shape": None}
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return {"stable_rank": None, "shape": None, "matrix_shape": None}
    if isinstance(module, nn.Linear):
        matrix = weight.detach().float()
    else:
        matrix = weight.detach().float().reshape(weight.shape[0], -1)
    return {
        "stable_rank": stable_rank(matrix.cpu()),
        "shape": list(weight.shape),
        "matrix_shape": list(matrix.shape),
    }


class ActivationExtractor:
    """Collects penultimate (classifier input) and classifier-output activations."""

    def __init__(self, model, device, max_feats=4096, exclude=None, classifier=None, penultimate=None):
        self.model = model.to(device)
        self.device = device
        self.max_feats = max_feats
        self.exclude: Set[str] = exclude or set()
        self.classifier = classifier
        self.penultimate = penultimate
        self.activations = {}
        self.hooks = []
        self.penult_name = None
        self.classifier_inputs = []
        self.classifier_outputs = []
        self.labels = None
        self.no_head = classifier is None

    def _keep(self, name, module):
        if name in self.exclude:
            return False
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.Identity)):
            return False
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            return False
        return True

    def register_hooks(self):
        def make_hook(name):
            def hook(_module, _inputs, output):
                x = output[0] if isinstance(output, (list, tuple)) and output else output
                if not isinstance(x, torch.Tensor):
                    return
                self.penult_name = name
                self.activations[name] = feature_downproj(
                    x.reshape(x.shape[0], -1), self.max_feats).detach()
            return hook

        for name, module in self.model.named_modules():
            if self._keep(name, module):
                self.hooks.append(module.register_forward_hook(make_hook(name)))

        if self.penultimate is not None:
            def penult_hook(_module, _inputs, output):
                x = output[0] if isinstance(output, (list, tuple)) else output
                if isinstance(x, torch.Tensor):
                    token = x[:, 0, :] if x.dim() == 3 else x
                    self.classifier_inputs.append(
                        feature_downproj(token.reshape(token.shape[0], -1), self.max_feats).detach().cpu())
            self.hooks.append(self.penultimate.register_forward_hook(penult_hook))

        if self.classifier is not None:
            def pre_hook(_module, inputs):
                x = inputs[0]
                if isinstance(x, torch.Tensor):
                    token = x[:, 0, :] if x.dim() == 3 else x
                    self.classifier_inputs.append(
                        feature_downproj(token.reshape(token.shape[0], -1), self.max_feats).detach().cpu())

            def post_hook(_module, _inputs, output):
                out = output[0] if isinstance(output, (list, tuple)) and output else output
                if isinstance(out, torch.Tensor):
                    self.classifier_outputs.append(out.reshape(out.shape[0], -1).detach().cpu())

            self.hooks.append(self.classifier.register_forward_pre_hook(pre_hook))
            self.hooks.append(self.classifier.register_forward_hook(post_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract(self, dataloader):
        self.model.eval()
        buffer = defaultdict(list)
        self.classifier_inputs, self.classifier_outputs = [], []
        pooled, labels = [], []

        for images, targets in dataloader:
            self.activations.clear()
            images = images.to(self.device, non_blocking=True)
            output = self.model(images)
            if self.no_head and isinstance(output, torch.Tensor):
                pooled.append((output.mean(dim=1) if output.dim() == 3 else output).detach().cpu())
            for name, activation in self.activations.items():
                if activation is not None:
                    buffer[name].append(activation.cpu())
            labels.append(targets)

        if pooled:
            self.classifier_outputs = [torch.cat(pooled, 0)]
        self.labels = torch.cat(labels, 0) if labels else None
        return {name: torch.cat(chunks, 0) for name, chunks in buffer.items() if chunks}


def calculate_transferability_scores(model_name, dataset_name, num_samples=200, sample_seed=1234,
                                     device="cuda", batch_size=32, max_feats=4096, data_root="./data"):
    """Return the raw stable-rank record for one (model, dataset) pair."""
    transform = default_transform(model_name)
    model = load_model(model_name)
    classifier = resolve_classifier(model)
    exclude = resolve_classifier_names(model)
    penultimate = find_last_mlp_fc2(model) if classifier is None else None

    dataset = get_dataset(dataset_name, transform, "train", root=data_root)
    subset = create_random_subset(dataset, num_samples=num_samples, seed=sample_seed)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=(device == "cuda"))

    extractor = ActivationExtractor(model, device, max_feats, exclude, classifier, penultimate)
    extractor.register_hooks()
    try:
        activations = extractor.extract(dataloader)
        penult_name = extractor.penult_name
    finally:
        extractor.remove_hooks()

    if not activations and not extractor.classifier_inputs and not extractor.classifier_outputs:
        return None

    if extractor.classifier_inputs:
        penult = torch.cat(extractor.classifier_inputs, 0)
    elif penult_name and penult_name in activations:
        penult = activations[penult_name]
    else:
        penult = activations[next(reversed(activations))]

    classifier_sr = 0.0
    if extractor.classifier_outputs:
        classifier_sr = stable_rank(torch.cat(extractor.classifier_outputs, 0).float().T)

    if classifier is None:
        penult_weight_module = penultimate
        classifier_weights = {"stable_rank": None, "shape": None, "matrix_shape": None}
    else:
        penult_weight_module, _ = find_last_weighted_module(model, exclude)
        classifier_weights = weight_ranks(classifier)

    return {
        "no_head": classifier is None,
        "penultimate_layer": [stable_rank(penult.float().T)],
        "classifier_layer": [classifier_sr],
        "weight_penultimate": weight_ranks(penult_weight_module),
        "weight_classifier": classifier_weights,
    }
