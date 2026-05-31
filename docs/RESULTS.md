# Reported Results (CCPL-default)

All numbers below were obtained under **automatically generated fallback splits** (`split_zhou_*.json`) in our local experiments, **not** the official Zhou Google Drive splits. Use them for within-protocol comparison only.

**Default CCPL configuration:** `lambda_text=0.5`, `lambda_kl=0.0`, `concept_dropout=0.3`, `ensemble_alpha=0.1`, 4-shot, seed=1, 50 epochs, ViT-B/16.

Config file: [`configs/trainers/CCPL/vit_b16_ep50_safe.yaml`](../configs/trainers/CCPL/vit_b16_ep50_safe.yaml)

## Main Base-to-New Results

| Dataset | Method | Base | New | H | ΔH vs CoOp |
|---------|--------|------|-----|---|------------|
| DTD | CoOp | 72.3 | 55.2 | 62.6 | — |
| DTD | CCPL-default | 73.8 | 55.3 | 63.2 | +0.6 |
| EuroSAT | CoOp | 79.8 | 56.3 | 66.0 | — |
| EuroSAT | CCPL-default | 79.7 | 60.6 | 68.9 | +2.9 |
| OxfordPets | CoOp | 94.9 | 97.7 | 96.3 | — |
| OxfordPets | CCPL-default | 94.9 | 97.5 | 96.2 | -0.1 |

## EuroSAT Ablation (Base-to-New)

| Variant | λ_text | α | Base | New | H |
|---------|--------|---|------|-----|---|
| CoOp | — | — | 79.8 | 56.3 | 66.0 |
| CCPL-default | 0.5 | 0.1 | 79.7 | 60.6 | 68.9 |
| no text regularization | 0.0 | 0.1 | 79.6 | 58.8 | 67.6 |
| no inference ensemble | 0.5 | 0.0 | 79.9 | 56.6 | 66.3 |
| stronger ensemble | 0.5 | 0.4 | 75.6 | 66.1 | 70.5 |

## Supporting Evidence

| Setting | Dataset/Seed | CoOp | CCPL-default | Gain |
|---------|--------------|------|--------------|------|
| All-class | DTD (seed1) | 60.8 | 62.9 | +2.1 |
| All-class | EuroSAT (seed1) | 70.9 | 71.0 | +0.1 |
| All-class | DTD (seed2) | 60.1 | 60.9 | +0.8 |
| Base-to-new H | DTD (seed1) | 62.6 | 63.2 | +0.6 |
| Base-to-new H | DTD (seed2) | 59.5 | 63.4 | +3.9 |

## Notes

- Gains are **dataset-dependent**; EuroSAT improvement is mainly from new-class accuracy.
- OxfordPets is near-neutral under CCPL-default.
- We compare primarily against **CoOp** under identical protocol; broader SOTA comparisons are future work.
