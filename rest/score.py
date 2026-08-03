import numpy as np
import pandas as pd
from scipy.stats import weightedtau

FEATURES = ["pen_act", "clf_act", "pen_before_weight", "clf_before_weight"]


def weighted_kendall(a, b) -> float:
    return weightedtau(np.array(a, dtype=float, copy=True),
                       np.array(b, dtype=float, copy=True)).correlation


def zscore_nan(values: np.ndarray) -> np.ndarray:
    """z-score that ignores NaN; missing entries stay NaN."""
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(values)
    out = np.full_like(values, np.nan)
    if mask.sum() < 2:
        return out
    mean, std = values[mask].mean(), values[mask].std()
    out[mask] = 0.0 if std == 0 else (values[mask] - mean) / std
    return out


def build_elements(all_json, source_dataset, target_datasets, ground_truth, use_clf=True) -> pd.DataFrame:
    """The four ReST elements for every (model, target) pair."""
    source = all_json[source_dataset]
    rows = []
    for dataset_name in target_datasets:
        for model_name, target in all_json[dataset_name].items():
            if model_name not in source:
                continue
            src = source[model_name]
            clf_weight = target["weight_classifier"]["stable_rank"]
            rows.append({
                "target dataset": dataset_name,
                "pre-trained model": model_name,
                "pen_act": target["penultimate_layer"][0] - src["penultimate_layer"][0],
                "clf_act": target["classifier_layer"][0] - src["classifier_layer"][0],
                "pen_before_weight": (target["weight_penultimate"]["stable_rank"]
                                      / min(target["weight_penultimate"]["matrix_shape"])),
                "clf_before_weight": (clf_weight / min(target["weight_classifier"]["matrix_shape"])
                                      if clf_weight is not None else np.nan),
                "fine-tune accuracy": ground_truth.get(dataset_name, {}).get(model_name),
            })
    df = pd.DataFrame(rows)
    if not use_clf:
        df["clf_act"] = np.nan
        df["clf_before_weight"] = np.nan
    return df


def rest_score(df: pd.DataFrame, gamma: float = 0.2) -> pd.DataFrame:
    """z-score each element within a target dataset, then ReST = (1-gamma) * G + gamma * L."""
    df = df.copy()
    for _, sub in df.groupby("target dataset"):
        df.loc[sub.index, FEATURES] = sub[FEATURES].apply(
            lambda col: zscore_nan(col.to_numpy(dtype=float)))

    g = df["pen_before_weight"].fillna(0.0) + df["clf_before_weight"].fillna(0.0)
    l = df["pen_act"].fillna(0.0) + df["clf_act"].fillna(0.0)
    df["ReST"] = (1 - gamma) * g + gamma * l
    return df


def evaluate(df: pd.DataFrame, target_datasets=None, score_column="ReST") -> dict:
    """Weighted Kendall tau against fine-tuning accuracy, per target dataset."""
    targets = target_datasets or sorted(df["target dataset"].unique())
    taus = {}
    for dataset_name in targets:
        sub = df[(df["target dataset"] == dataset_name) & df["fine-tune accuracy"].notna()]
        taus[dataset_name] = (weighted_kendall(sub["fine-tune accuracy"], sub[score_column])
                              if len(sub) >= 2 else np.nan)
    values = [v for v in taus.values() if not np.isnan(v)]
    taus["MEAN"] = float(np.mean(values)) if values else np.nan
    return taus
