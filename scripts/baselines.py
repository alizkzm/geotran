"""Run LEEP, LogME and ETran in label-independent or label-dependent mode."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rest.baselines import clip_pseudo_labels, etran, extract_features_logits, leep, logme
from rest.config import load_config
from rest.ground_truth import get_ground_truth
from rest.score import weighted_kendall


def main():
    parser = argparse.ArgumentParser(description="Transferability baselines.")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--hub", default="cnn", choices=["cnn", "vit"])
    parser.add_argument("--label-independent", action="store_true",
                        help="use num_samples with CLIP pseudo-labels instead of the full labeled split")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--save-csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.resolved_device()
    ground_truth = get_ground_truth(args.hub)
    n_samples = cfg.num_samples if args.label_independent else None

    rows = []
    for dataset_name in cfg.target_datasets:
        pseudo = None
        for model_name in cfg.model_hub:
            features, logits, y_true, subset, dataset = extract_features_logits(
                model_name, dataset_name, n_samples, cfg.sample_seed,
                device=device, batch_size=cfg.batch_size, max_feats=cfg.max_feats,
                data_root=args.data_root,
            )
            if args.label_independent:
                if pseudo is None:
                    pseudo = clip_pseudo_labels(subset, dataset.classes, device=device)
                y = pseudo
            else:
                y = y_true
            rows.append({
                "target dataset": dataset_name,
                "pre-trained model": model_name,
                "LEEP": leep(logits, y),
                "LogME": logme(features, y),
                "ETran": etran(features, logits, y),
                "fine-tune accuracy": ground_truth[dataset_name][model_name],
            })
            print(f"  {dataset_name}/{model_name} done")

    df = pd.DataFrame(rows)
    mode = "200 samples + CLIP pseudo-labels" if args.label_independent else "full split + true labels"
    print(f"\nBaselines ({mode})")
    print("-" * 42)
    print(f"{'dataset':<14}{'LEEP':>9}{'LogME':>9}{'ETran':>9}")
    for dataset_name, sub in df.groupby("target dataset"):
        print(f"{dataset_name:<14}" + "".join(
            f"{weighted_kendall(sub['fine-tune accuracy'], sub[m]):>9.3f}"
            for m in ["LEEP", "LogME", "ETran"]))

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print("saved", args.save_csv)


if __name__ == "__main__":
    main()
