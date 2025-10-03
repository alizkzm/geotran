"""
Dataset loading and preprocessing utilities.
"""

import torch
import torchvision.datasets as tvds
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Subset
from typing import List


class HFDatasetTorchWrapper(torch.utils.data.Dataset):
    """Wrapper for HuggingFace datasets to work with PyTorch DataLoader."""

    def __init__(self, hf_ds, transform: T.Compose):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[int(idx)]
        img = example["image"]
        label = int(example["label"])
        img = self.transform(img)
        return img, label


def _simple_loader_wrapper(dataset, transform):
    """Wrapper for datasets without standard labels."""

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, _target = self.base[idx]
            img = self.transform(img)
            return img, 0

    return WrappedDataset(dataset, transform)


def _load_hf_mini_imagenet(split: str, transform: T.Compose):
    """Load Mini-ImageNet from HuggingFace."""
    try:
        from datasets import load_dataset
    except Exception:
        raise RuntimeError("Install: pip install datasets pillow")

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("timm/mini-imagenet", split=hf_split, trust_remote_code=True)
    return HFDatasetTorchWrapper(ds, transform)


def get_dataset(dataset_name: str, transform: T.Compose, split: str = "train"):
    """
    Load a dataset by name.

    Args:
        dataset_name: Name of the dataset
        transform: Torchvision transforms to apply
        split: 'train' or 'test'

    Returns:
        PyTorch Dataset
    """
    name = dataset_name.lower()

    if name == "cifar10":
        return tvds.CIFAR10(root="./data", train=(split == "train"), transform=transform, download=True)

    if name == "cifar100":
        return tvds.CIFAR100(root="./data", train=(split == "train"), transform=transform, download=True)

    if name == "caltech101":
        return tvds.Caltech101(root="./data", transform=transform, download=True)

    if name == "dtd":
        part = "train" if split == "train" else "test"
        return tvds.DTD(root="./data", split=part, transform=transform, download=True)

    if name == "pets":
        part = "trainval" if split == "train" else "test"
        return tvds.OxfordIIITPet(root="./data", split=part, target_types="category", transform=transform,
                                  download=True)

    if name == "flowers":
        return tvds.Flowers102(root="./data", split=split, transform=transform, download=True)

    if name == "food":
        part = "train" if split == "train" else "test"
        return tvds.Food101(root="./data", split=part, transform=transform, download=True)

    if name == "sun397":
        return tvds.SUN397(root="./data", transform=transform, download=True)

    if name == "cars":
        part = "train" if split == "train" else "test"
        return tvds.StanfordCars(root="./data", split=part, transform=transform, download=True)

    if name == "aircraft":
        if hasattr(tvds, "FGVCAircraft"):
            part = "train" if split == "train" else "test"
            return tvds.FGVCAircraft(root="./data", split=part, annotation_level="variant", transform=transform,
                                     download=True)
        raise ValueError("FGVCAircraft not available in your torchvision version.")

    if name == "voc2007":
        base = tvds.VOCDetection(root="./data", year="2007", image_set=("train" if split == "train" else "val"),
                                 download=True)
        return _simple_loader_wrapper(base, transform)

    if name in {"tiny_imagenet", "tiny-imagenet", "mini_imagenet", "mini-imagenet"}:
        return _load_hf_mini_imagenet(split, transform)

    raise ValueError(f"Unknown or unsupported dataset: {dataset_name}")


def create_balanced_subset(dataset, images_per_class: int = 2, fallback_max: int = 256):
    """
    Create a balanced subset of a dataset with specified images per class.

    Args:
        dataset: PyTorch Dataset
        images_per_class: Number of images to sample per class
        fallback_max: Maximum images if class extraction fails

    Returns:
        Subset of the dataset
    """
    targets = None

    try:
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        else:
            tmp = []
            for i in range(min(len(dataset), 1024)):
                y = dataset[i][1]
                if isinstance(y, (int, np.integer, torch.Tensor)) and (not isinstance(y, dict)):
                    tmp.append(int(y) if not isinstance(y, torch.Tensor) else int(y.item()))
                else:
                    tmp = None
                    break
            if tmp is not None and len(tmp) > 0:
                targets = [dataset[i][1] for i in range(len(dataset))]
    except Exception:
        targets = None

    if targets is None:
        idxs = list(range(min(fallback_max, len(dataset))))
        return Subset(dataset, idxs)

    if not isinstance(targets, torch.Tensor):
        try:
            targets = torch.tensor(targets)
        except Exception:
            idxs = list(range(min(fallback_max, len(dataset))))
            return Subset(dataset, idxs)

    targets = targets.view(-1)
    classes = torch.unique(targets)
    idxs: List[int] = []
    g = torch.Generator().manual_seed(1234)

    for c in classes.tolist():
        c_idx = torch.where(targets == c)[0]
        if len(c_idx) == 0:
            continue
        if len(c_idx) > images_per_class:
            perm = torch.randperm(len(c_idx), generator=g)[:images_per_class]
            sel = c_idx[perm].tolist()
        else:
            sel = c_idx.tolist()
        idxs.extend(sel)

    if not idxs:
        idxs = list(range(min(fallback_max, len(dataset))))

    return Subset(dataset, idxs)