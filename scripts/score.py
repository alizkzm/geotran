"""Compute the ReST score from extracted records and evaluate it."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rest.config import load_config, record_path
from rest.ground_truth import get_ground_truth
from rest.score import build_elements, evaluate, rest_score


def main():
    parser = argparse.ArgumentParser(description="Compute and evaluate the ReST score.")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--hub", default="cnn", choices=["cnn", "vit"])
    parser.add_argument("--gamma", type=float, default=None, help="override the config gamma")
    parser.add_argument("--save-csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gamma = args.gamma if args.gamma is not None else cfg.gamma

    all_json = {}
    for dataset_name in [cfg.source_dataset] + cfg.target_datasets:
        path = record_path(cfg.out_dir, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run scripts/extract.py first")
        with open(path, "r", encoding="utf-8") as f:
            all_json[dataset_name] = json.load(f)

    df = build_elements(all_json, cfg.source_dataset, cfg.target_datasets,
                        get_ground_truth(args.hub), use_clf=cfg.use_clf)
    df = rest_score(df, gamma=gamma)
    taus = evaluate(df, cfg.target_datasets)

    print(f"ReST (gamma={gamma}) | source={cfg.source_dataset} | use_clf={cfg.use_clf}")
    print("-" * 34)
    for dataset_name in cfg.target_datasets:
        print(f"  {dataset_name:<14}{taus[dataset_name]:>8.4f}")
    print("-" * 34)
    print(f"  {'MEAN':<14}{taus['MEAN']:>8.4f}")

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print("saved", args.save_csv)


if __name__ == "__main__":
    main()
