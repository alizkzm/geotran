from dataclasses import dataclass, field
from typing import List, Optional
import os
import yaml


@dataclass
class Config:
    seed: int = 123
    gamma: float = 0.2
    num_samples: int = 200
    sample_seed: int = 1234
    batch_size: int = 32
    max_feats: int = 4096
    use_clf: bool = True

    model_hub: List[str] = field(default_factory=list)
    source_dataset: str = "mini_imagenet"
    target_datasets: List[str] = field(default_factory=lambda: ["cifar10", "cifar100"])

    out_dir: str = "outputs/rest_json"
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


def record_path(out_dir: str, dataset: str) -> str:
    """Path of a dataset's record file.

    Returns an existing file if one is present under any accepted name, otherwise
    the canonical ``<dataset>.json``.
    """
    candidates = [
        f"{dataset}.json",
        f"{dataset}_transferability_scores.json",
        f"{dataset}_transferability_scores_overall_layers.json",
    ]
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(out_dir, candidates[0])


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {k: v for k, v in raw.items() if k in Config.__dataclass_fields__}
    cfg = Config(**known)
    os.makedirs(cfg.out_dir, exist_ok=True)
    return cfg
