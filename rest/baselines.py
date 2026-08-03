"""LEEP, LogME and ETran, in label-dependent and label-independent settings."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .data import create_random_subset, default_transform, get_dataset
from .extract import ActivationExtractor
from .models import load_model, resolve_classifier, resolve_classifier_names

CLIP_ARCH = "ViT-L-14"
CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"


def leep(logits: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    prob = softmax(logits, axis=1)
    classes = np.unique(y)
    pyz = np.zeros((len(classes), prob.shape[1]))
    for i, c in enumerate(classes):
        pyz[i] = prob[y == c].sum(axis=0) / n
    py_given_z = pyz / (pyz.sum(axis=0) + 1e-12)
    py_x = prob @ py_given_z.T
    idx = np.searchsorted(classes, y)
    return float(np.sum(py_x[np.arange(n), idx]) / n)


def logme(features: np.ndarray, y: np.ndarray, max_iter: int = 11) -> float:
    features = features.astype(np.float64)
    n, d = features.shape
    u, s, _ = np.linalg.svd(features, full_matrices=False)
    sigma = s ** 2
    evidences = []
    for c in np.unique(y):
        target = (y == c).astype(np.float64).reshape(-1, 1)
        x = u.T @ target
        x2 = (x ** 2).ravel()
        res_x2 = float((target ** 2).sum() - x2.sum())
        alpha, beta = 1.0, 1.0
        m2 = res2 = 0.0
        for _ in range(max_iter):
            t = alpha / beta
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / ((t + sigma) ** 2)).sum())
            res2 = float((x2 / ((1 + sigma / t) ** 2)).sum() + res_x2)
            alpha = gamma / (m2 + 1e-12)
            beta = (n - gamma) / (res2 + 1e-12)
            if abs(alpha / beta - t) / t <= 1e-3:
                break
        evidence = (d / 2 * np.log(alpha) + n / 2 * np.log(beta)
                    - 0.5 * float(np.sum(np.log(alpha + beta * sigma)))
                    - beta / 2 * res2 - alpha / 2 * m2 - n / 2 * np.log(2 * np.pi))
        evidences.append(evidence / n)
    return float(np.mean(evidences))


def etran(features: np.ndarray, logits: np.ndarray, y: np.ndarray, percent: float = 0.5) -> float:
    """Energy score + LDA classification score (ETran Sen+Scls)."""
    energy = np.log(np.exp(logits).sum(axis=1) + 1e-12)
    k = max(1, int(percent * 10) * len(energy) // 1000)
    energy_score = float(np.sort(energy)[:k].mean())

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(features, y)
    prob = lda.predict_proba(features)
    idx = np.searchsorted(np.unique(y), y)
    classification_score = float(np.sum(prob[np.arange(len(y)), idx]) / len(y))
    return energy_score + classification_score


@torch.no_grad()
def clip_pseudo_labels(subset, class_names, device="cuda", batch_size=64):
    """Zero-shot labels from OpenCLIP ViT-L-14 / datacomp_xl_s13b_b90k."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
    model = model.to(device).eval()

    tokens = tokenizer([f"a photo of a {name}" for name in class_names]).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    base = subset.dataset
    original_transform = base.transform
    base.transform = preprocess
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    predictions = []
    for images, _ in loader:
        image_features = model.encode_image(images.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        predictions.append((image_features @ text_features.T).argmax(dim=-1).cpu())
    base.transform = original_transform

    del model
    torch.cuda.empty_cache()
    return torch.cat(predictions).numpy()


@torch.no_grad()
def extract_features_logits(model_name, dataset_name, num_samples, sample_seed=1234,
                            device="cuda", batch_size=32, max_feats=4096, data_root="./data"):
    """Penultimate features, classifier logits and labels. num_samples=None uses the full split."""
    transform = default_transform(model_name)
    model = load_model(model_name)
    classifier = resolve_classifier(model)
    exclude = resolve_classifier_names(model)

    dataset = get_dataset(dataset_name, transform, "train", root=data_root)
    subset = create_random_subset(dataset, num_samples, sample_seed) if num_samples else dataset
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    extractor = ActivationExtractor(model, device, max_feats, exclude, classifier, None)
    extractor.register_hooks()
    try:
        extractor.extract(loader)
    finally:
        extractor.remove_hooks()

    features = torch.cat(extractor.classifier_inputs, 0).float().numpy()
    logits = torch.cat(extractor.classifier_outputs, 0).float().numpy()
    return features, logits, extractor.labels.numpy(), subset, dataset
