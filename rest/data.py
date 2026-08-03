import random

import numpy as np
import torch
from torch.utils.data import Subset
import torchvision.datasets as tvds
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_transform(model_name: str = None) -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[int(idx)]
        return self.transform(example["image"].convert("RGB")), int(example["label"])


def _load_mini_imagenet(split, transform, root="./data"):
    from datasets import load_dataset
    hf_split = "train" if split == "train" else "validation"
    return HFDatasetWrapper(load_dataset("timm/mini-imagenet", split=hf_split), transform)


def get_dataset(dataset_name: str, transform, split: str = "train", root: str = "./data"):
    name = dataset_name.lower()

    if name == "cifar10":
        return tvds.CIFAR10(root=root, train=(split == "train"), transform=transform, download=True)
    if name == "cifar100":
        return tvds.CIFAR100(root=root, train=(split == "train"), transform=transform, download=True)
    if name == "mini_imagenet":
        return _load_mini_imagenet(split, transform, root)

    if name == "aircraft":
        return tvds.FGVCAircraft(root=root, split=("train" if split == "train" else "test"),
                                 annotation_level="variant", transform=transform, download=True)
    if name == "caltech101":
        return tvds.Caltech101(root=root, transform=transform, download=True)
    if name == "cars":
        return tvds.StanfordCars(root=root, split=("train" if split == "train" else "test"),
                                 transform=transform, download=True)
    if name == "dtd":
        return tvds.DTD(root=root, split=("train" if split == "train" else "test"),
                        transform=transform, download=True)
    if name == "flowers":
        return tvds.Flowers102(root=root, split=split, transform=transform, download=True)
    if name == "food":
        return tvds.Food101(root=root, split=("train" if split == "train" else "test"),
                            transform=transform, download=True)
    if name == "pets":
        return tvds.OxfordIIITPet(root=root, split=("trainval" if split == "train" else "test"),
                                  target_types="category", transform=transform, download=True)
    if name == "sun397":
        return tvds.SUN397(root=root, transform=transform, download=True)
    if name == "voc2007":
        base = tvds.VOCDetection(root=root, year="2007",
                                 image_set=("train" if split == "train" else "val"), download=True)
        class _VOC(torch.utils.data.Dataset):
            def __len__(self): return len(base)
            def __getitem__(self, i): return transform(base[i][0]), 0
        return _VOC()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_random_subset(dataset, num_samples: int = 200, seed: int = 1234) -> Subset:
    n = len(dataset)
    k = min(num_samples, n)
    if seed is None:
        perm = torch.randperm(n)
    else:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)
    return Subset(dataset, perm[:k].tolist())
