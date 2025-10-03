#!/usr/bin/env python3
"""
Transferability Estimation Metric Calculator

Computes transferability scores using various methods.

Usage:
    python te_metric.py -t cifar100 -te rest
    python te_metric.py -t cifar100 -te logme
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.rest import compute_rest_for_dataset


def main():
    parser = argparse.ArgumentParser(description="Calculate transferability estimation metrics")
    parser.add_argument("-t", "--target", type=str, default="cifar100",
                        help="Target dataset (default: cifar100)")
    parser.add_argument("-te", "--method", type=str, default="rest",
                        choices=["rest", "logme", "leep", "energy", "lda", "sfda"],
                        help="Transferability estimation method (default: rest)")
    parser.add_argument("--source", type=str, default="tiny_imagenet",
                        help="Source dataset for REST (default: tiny_imagenet)")
    parser.add_argument("--gamma", type=float, default=0.21,
                        help="Gamma parameter for REST (default: 0.21)")
    parser.add_argument("--alpha", type=float, default=0.51,
                        help="Alpha parameter for REST (default: 0.51)")

    args = parser.parse_args()

    results_dir = Path("transferability_results")
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist. Run rank_measure.py first.")
        return

    target_file = results_dir / f"{args.target}_transferability_scores.json"

    if not target_file.exists():
        print(f"Error: {target_file} does not exist. Run rank_measure.py for {args.target} first.")
        return

    print(f"Computing {args.method.upper()} scores for {args.target}...")

    if args.method == "rest":
        source_file = results_dir / f"{args.source}_transferability_scores.json"

        # Check if source file exists, if not, process it automatically
        if not source_file.exists():
            print(f"\nSource dataset ({args.source}) not found. Processing source dataset first...")
            import subprocess

            # Get samples per class from target file if available
            with open(target_file, 'r') as f:
                target_data = json.load(f)

            # Extract images_per_class from any model entry
            samples = 2  # default
            for model_data in target_data.values():
                if "images_used_count" in model_data:
                    # This is approximate, but reasonable
                    samples = 2
                    break

            print(f"Running: python scripts/rank_measure.py -s {samples} -t {args.source}")
            result = subprocess.run(
                ["python", "scripts/rank_measure.py", "-s", str(samples), "-t", args.source],
                cwd=Path(__file__).parent.parent
            )

            if result.returncode != 0:
                print(f"Error: Failed to process source dataset {args.source}")
                return

            # Check again
            if not source_file.exists():
                print(f"Error: {source_file} still does not exist after processing.")
                return

        scores = compute_rest_for_dataset(
            str(target_file),
            str(source_file),
            gamma=args.gamma,
            alpha=args.alpha
        )

        print(f"\nREST Scores (gamma={args.gamma}, alpha={args.alpha}):")
        print(f"Source: {args.source}")
        print(f"Target: {args.target}")
        print()
        for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:20s}: {score:.4f}")

        # Save scores
        output_file = results_dir / f"{args.target}_rest_scores.json"
        output_data = {
            "method": "rest",
            "target": args.target,
            "source": args.source,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "scores": scores
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {output_file}")

    elif args.method == "logme":
        print("LogME implementation placeholder - to be added")
        # from metrics.logme import compute_logme_for_dataset
        # scores = compute_logme_for_dataset(str(target_file))

    elif args.method == "leep":
        print("LEEP implementation placeholder - to be added")
        # from metrics.leep import compute_leep_for_dataset
        # scores = compute_leep_for_dataset(str(target_file))

    elif args.method == "energy":
        print("Energy implementation placeholder - to be added")
        # from metrics.energy import compute_energy_for_dataset
        # scores = compute_energy_for_dataset(str(target_file))

    elif args.method == "lda":
        print("LDA implementation placeholder - to be added")
        # from metrics.lda import compute_lda_for_dataset
        # scores = compute_lda_for_dataset(str(target_file))

    elif args.method == "sfda":
        print("SFDA implementation placeholder - to be added")
        # from metrics.sfda import compute_sfda_for_dataset
        # scores = compute_sfda_for_dataset(str(target_file))


if __name__ == "__main__":
    main()