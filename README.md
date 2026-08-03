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


## Repository layout

```
.
├── configs/
│   ├── cnn.yaml            
│   └── vit.yaml            
├── rest/
│   ├── config.py           
│   ├── models.py           
│   ├── data.py             
│   ├── extract.py          
│   ├── score.py            
│   ├── baselines.py       
│   └── ground_truth.py     
├── scripts/
│   ├── extract.py          # CLI: stage 1
│   ├── score.py            # CLI: stage 2
│   └── baselines.py        
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


## Baselines

Can be run in two settings:

```bash
# label-dependent: the original protocol (full labeled target training set)
python scripts/baselines.py --config configs/cnn.yaml

# label-independent: 200 samples, pseudo-labels 
python scripts/baselines.py --config configs/cnn.yaml --label-independent
```

In the label-independent setting the pseudo-labels are generated **once per target dataset**,
independently of the models being ranked, and the same labels are given to every method — so no
method is scored on labels derived from its own representation.

---
