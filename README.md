# ReST — Remarkably Simple Transferability Estimation

Label-free transferability estimation for pre-trained model selection.

Given a hub of pre-trained models and an unlabeled target dataset, ReST predicts which model will
fine-tune best — without fine-tuning anything, without target labels, and without knowing how many
classes the target has. It reduces to computing the **stable rank** of the final two layers from
**200 random unlabeled samples**.

```
ReST(f, D_T) = (1 − γ) · G(f) + γ · L(f, D_S, D_T)
```

| Term | Meaning | Source |
|---|---|---|
| `G` | intrinsic generalization capacity | stable rank of the penultimate and classifier **weight** matrices |
| `L` | adaptation flexibility | shift in stable rank of the penultimate and classifier **activations**, target vs. source |

For a matrix `X`, the stable rank is `‖X‖_F² / ‖X‖_2²` — a continuous, noise-robust measure of
effective dimensionality.

---

## Installation

```bash
git clone <this-repo>
cd REST-GitHub
pip install -r requirements.txt
```

Python ≥ 3.9. A GPU is recommended for extraction but not required.

---

## Quick start

```bash
# 1. extract the stable-rank records for every (model, dataset) pair
python scripts/extract.py --config configs/cnn.yaml --hub cnn

# 2. compute the ReST score and evaluate it
python scripts/score.py --config configs/cnn.yaml --hub cnn
```

```
ReST (gamma=0.2) | source=mini_imagenet | use_clf=True
----------------------------------
  cifar10         0.9214
  cifar100        0.8727
----------------------------------
  MEAN            0.8971
```

Transformer hub:

```bash
python scripts/extract.py --config configs/vit.yaml --hub vit
python scripts/score.py   --config configs/vit.yaml --hub vit
```

---

## Repository layout

```
.
├── configs/
│   ├── cnn.yaml            # supervised CNN hub, γ = 0.21
│   └── vit.yaml            # 8 self-supervised + 8 supervised transformers, γ = 0.65
├── rest/
│   ├── config.py           # YAML-backed Config
│   ├── models.py           # model hubs, head/penultimate resolution
│   ├── data.py             # datasets, transforms, random subsets
│   ├── extract.py          # activation hooks, stable rank, per-pair records
│   ├── score.py            # the four elements, ReST score, weighted Kendall
│   ├── baselines.py        # LEEP, LogME, ETran, CLIP pseudo-labels
│   └── ground_truth.py     # fine-tuning accuracies
├── scripts/
│   ├── extract.py          # CLI: stage 1
│   ├── score.py            # CLI: stage 2
│   └── baselines.py        # CLI: baselines
├── notebooks/
│   ├── ReST.ipynb          # CNN walkthrough
│   └── ReST_vit.ipynb      # transformer walkthrough
└── requirements.txt
```

---

## Configuration

Everything is driven by a YAML file:

```yaml
gamma: 0.2                 # G vs L balance (0.21 CNNs, 0.65 transformers)
num_samples: 200           # unlabeled target samples
sample_seed: 1234
use_clf: true              # false drops both classifier terms for every model
source_dataset: mini_imagenet
target_datasets: [cifar10, cifar100]
model_hub: [resnet50, densenet121, ...]
out_dir: outputs/rest_json_cnn
```

Override γ at the command line without editing the file:

```bash
python scripts/score.py --config configs/cnn.yaml --gamma 0.3
```

---

## Model hubs

**CNNs** (`configs/cnn.yaml`) — ResNet-{34,50,101,152}, DenseNet-{121,169,201}, MNASNet-1.0,
MobileNetV2, GoogLeNet.

**Transformers** (`configs/vit.yaml`) — self-supervised MAE-ViT-{B/16,L/16}, DINO-ViT-{S/8,S/16,B/16},
MoCo-v3-ViT-{S/16,B/16}, SimMIM-ViT-B/16; supervised ViT-{T,S,B}/16, PVT-{T,S,M}, PVTv2-B2, Swin-T.

Self-supervised backbones are loaded head-less. They have no classifier weight, so `G` falls back to
the penultimate weight alone; the classifier *activation* is taken from the mean-pooled output and the
penultimate activation from the last block's `mlp.fc2`.

---

## Datasets

`cifar10`, `cifar100` and `mini_imagenet` (the source domain) are enabled by default. The remaining
benchmark targets — Aircraft, Caltech-101, Cars, DTD, Flowers, Food-101, Pets, SUN397, VOC2007 —
are present in `rest/data.py` as commented loaders; uncomment the one you need and add it to
`target_datasets`.

Ground-truth fine-tuning accuracies for all eleven targets ship in `rest/ground_truth.py`
(source: SFDA, Shao et al., ECCV 2022).

---

## Baselines

LEEP, LogME and ETran can be run in two settings:

```bash
# label-dependent: the original protocol (full labeled target training set)
python scripts/baselines.py --config configs/cnn.yaml

# label-independent: 200 samples, pseudo-labels from OpenCLIP ViT-L-14 / DataComp-XL
python scripts/baselines.py --config configs/cnn.yaml --label-independent
```

In the label-independent setting the pseudo-labels are generated **once per target dataset**,
independently of the models being ranked, and the same labels are given to every method — so no
method is scored on labels derived from its own representation.

---

## Method details

**Layer selection.** The penultimate layer is the input to the final classifier (captured with a
forward pre-hook; for transformers the `[CLS]` token). The classifier layer is the final linear head,
whose activations are the logits. `W_p` is the last convolutional or linear layer before the head
(`mlp.fc2` for transformers); `W_c` is the head weight.

**Normalization.** Weight stable ranks are divided by the smaller matrix dimension. Each of the four
elements is z-scored across the model hub *within each target dataset* before being combined, so the
score is a ranking over models for a fixed target.

**Metric.** Weighted Kendall τ against fine-tuning accuracy, which emphasises the top of the ranking —
the part that matters for model selection.

---

## Citation

```bibtex
@inproceedings{rest,
  title  = {ReST: Remarkably Simple Transferability Estimation},
  author = {Anonymous},
  year   = {2026}
}
```
