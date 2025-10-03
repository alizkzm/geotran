#!/usr/bin/env python3
"""
Transferability Estimation Performance Evaluator

Computes correlation between transferability scores and fine-tuning accuracy.

Usage:
    python te_performance.py -te rest -t cifar100
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import weightedtau, kendalltau, spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ground_truth import FINETUNE_ACC


def load_scores(method: str, target: str, results_dir: Path) -> dict:
    """Load transferability scores from JSON file."""
    score_file = results_dir / f"{target}_{method}_scores.json"

    if not score_file.exists():
        raise FileNotFoundError(f"{score_file} not found. Run te_metric.py first.")

    with open(score_file, 'r') as f:
        data = json.load(f)

    # Handle both old format (just scores) and new format (with metadata)
    if "scores" in data:
        return data["scores"]
    return data


def evaluate_correlation(scores: dict, ground_truth: dict, exclude_models=None):
    """
    Evaluate correlation between scores and ground truth accuracy.

    Args:
        scores: Dictionary of model -> transferability score
        ground_truth: Dictionary of model -> fine-tune accuracy

    Returns:
        Dictionary with correlation metrics
    """
    if exclude_models is None:
        exclude_models = []

    # Filter common models
    common_models = set(scores.keys()) & set(ground_truth.keys())
    common_models = [m for m in common_models if m not in exclude_models]

    if len(common_models) < 3:
        return {
            "error": "Insufficient models for correlation",
            "n_models": len(common_models)
        }

    # Prepare data
    score_values = [scores[m] for m in common_models]
    acc_values = [ground_truth[m] for m in common_models]

    # Compute correlations
    wtau, wtau_p = weightedtau(score_values, acc_values)
    ktau, ktau_p = kendalltau(score_values, acc_values)
    rho, rho_p = spearmanr(score_values, acc_values)

    return {
        "weighted_kendall": wtau,
        "weighted_kendall_pvalue": wtau_p,
        "kendall_tau": ktau,
        "kendall_tau_pvalue": ktau_p,
        "spearman_rho": rho,
        "spearman_pvalue": rho_p,
        "n_models": len(common_models),
        "models": common_models
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate transferability estimation performance")
    parser.add_argument("-te", "--method", type=str, default="rest",
                        choices=["rest", "logme", "leep", "energy", "lda", "sfda"],
                        help="Transferability estimation method (default: rest)")
    parser.add_argument("-t", "--target", type=str, default="cifar100",
                        help="Target dataset (default: cifar100)")
    parser.add_argument("--exclude", type=str, nargs="+", default=["inception_v3"],
                        help="Models to exclude from evaluation (default: inception_v3)")

    args = parser.parse_args()

    results_dir = Path("transferability_results")

    # Check if ground truth exists for target
    if args.target not in FINETUNE_ACC:
        print(f"Error: No ground truth accuracy available for {args.target}")
        print(f"Available datasets: {list(FINETUNE_ACC.keys())}")
        return

    ground_truth = FINETUNE_ACC[args.target]

    try:
        scores = load_scores(args.method, args.target, results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = evaluate_correlation(scores, ground_truth, exclude_models=args.exclude)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    print(f"\nCorrelation Results:")
    print(f"  Weighted Kendall Tau: {results['weighted_kendall']:.4f} (p={results['weighted_kendall_pvalue']:.4e})")
    print(f"  Kendall Tau:          {results['kendall_tau']:.4f} (p={results['kendall_tau_pvalue']:.4e})")
    print(f"  Spearman Rho:         {results['spearman_rho']:.4f} (p={results['spearman_pvalue']:.4e})")

    # Save detailed results
    output_file = results_dir / f"{args.target}_{args.method}_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create comparison DataFrame
    comparison_data = []
    for model in results['models']:
        comparison_data.append({
            "model": model,
            f"{args.method}_score": scores[model],
            "finetune_accuracy": ground_truth[model]
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values(f"{args.method}_score", ascending=False)

    csv_file = results_dir / f"{args.target}_{args.method}_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved comparison to {csv_file}")

    print("\nModel Rankings:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
