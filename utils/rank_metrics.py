"""
Rank computation utilities for transferability estimation.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_stable_rank_from_svals(s: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute stable rank from singular values.

    Stable rank = (Frobenius norm)^2 / (largest singular value)^2

    Args:
        s: Singular values tensor
        eps: Small constant for numerical stability

    Returns:
        Stable rank as float
    """
    if s.numel() == 0:
        return 0.0
    fro2 = torch.sum(s ** 2)
    smax2 = torch.max(s) ** 2
    return (fro2 / (smax2 + eps)).item()


def compute_effective_rank_from_svals(s: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute effective rank from singular values using entropy.

    Effective rank = exp(H(p)) where p is the normalized distribution of singular values

    Args:
        s: Singular values tensor
        eps: Small constant for numerical stability

    Returns:
        Effective rank as float
    """
    if s.numel() == 0:
        return 0.0
    p = s / (torch.sum(s) + eps)
    ent = -torch.sum(p * torch.log(p + eps))
    return torch.exp(ent).item()


def compute_ranks(matrix: torch.Tensor) -> Tuple[float, float]:
    """
    Compute both stable and effective ranks for a matrix.

    Args:
        matrix: Input matrix tensor

    Returns:
        Tuple of (stable_rank, effective_rank)
    """
    if matrix.numel() == 0:
        return 0.0, 0.0
    s = torch.linalg.svdvals(matrix)
    return compute_stable_rank_from_svals(s), compute_effective_rank_from_svals(s)


def feature_downproj(x: torch.Tensor, max_feat: Optional[int] = 4096) -> torch.Tensor:
    """
    Downsample features via random projection if dimensionality exceeds max_feat.

    Args:
        x: Feature tensor of shape (B, D)
        max_feat: Maximum feature dimension

    Returns:
        Downprojected features of shape (B, max_feat) or original if D <= max_feat
    """
    if max_feat is None:
        return x
    B, D = x.shape
    if D <= max_feat:
        return x
    gen = torch.Generator(device=x.device).manual_seed(12345)
    P = torch.randn(D, max_feat, generator=gen, device=x.device) / np.sqrt(max_feat)
    return x @ P