"""Extract the stable-rank records for every (model, dataset) pair."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rest.config import load_config, record_path
from rest.data import set_seed
from rest.extract import calculate_transferability_scores
from rest.ground_truth import get_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Extract ReST stable-rank records.")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--hub", default="cnn", choices=["cnn", "vit"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--force", action="store_true", help="re-extract even if a JSON exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    ground_truth = get_ground_truth(args.hub)
    print(f"device: {device}")

    for dataset_name in [cfg.source_dataset] + cfg.target_datasets:
        path = record_path(cfg.out_dir, dataset_name)
        if os.path.exists(path) and not args.force:
            print(f"=== {dataset_name}: cached ({path})")
            continue

        print(f"\n=== {dataset_name} ===")
        records = {}
        for model_name in cfg.model_hub:
            try:
                record = calculate_transferability_scores(
                    model_name, dataset_name,
                    num_samples=cfg.num_samples, sample_seed=cfg.sample_seed,
                    device=device, batch_size=cfg.batch_size, max_feats=cfg.max_feats,
                    data_root=args.data_root,
                )
            except Exception as exc:
                print(f"  [skip] {model_name}: {exc}")
                continue
            if record is None:
                print(f"  [skip] {model_name}: no activations")
                continue
            record["finetune_accuracy"] = ground_truth.get(dataset_name, {}).get(model_name)
            records[model_name] = record
            clf_weight = record["weight_classifier"]["stable_rank"]
            print(f"  {model_name:<14} pen_sr={record['penultimate_layer'][0]:.3f} "
                  f"clf_sr={record['classifier_layer'][0]:.3f} "
                  f"w_pen={record['weight_penultimate']['stable_rank']:.3f} "
                  f"w_clf={'None' if clf_weight is None else f'{clf_weight:.3f}'}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
