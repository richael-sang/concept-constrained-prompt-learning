# Concept-Constrained Prompt Learning (CCPL)

Official implementation of **Concept-Constrained Prompt Learning for Few-Shot CLIP Adaptation**.

CCPL regularizes learnable class prompts with **frozen concept-level text prototypes** built from a hand-crafted concept bank. The CLIP image/text encoders remain frozen; only shared context tokens are optimized. The default configuration uses text-space cosine regularization and weak concept-guided inference fusion.

This repository extends the [PromptSRC](https://github.com/muzairkhattak/PromptSRC) codebase with a new trainer `CCPL` and a concept bank. Baseline trainers (CoOp, Co-CoOp, MaPLe, PromptSRC, etc.) from the upstream project are retained for comparison.

## Highlights

- **Shared context tokens** (CoOp-style) with class names appended at inference for unseen classes
- **Frozen concept prototypes** from [`concept_bank/ccpl_concepts.json`](concept_bank/ccpl_concepts.json)
- **Text-space consistency loss** on base classes during training
- **Concept dropout** during training; full concept set at inference
- **Optional concept-guided logit fusion** with weight `ENSEMBLE_ALPHA`

Default CCPL settings (`CCPL-default`):

| Parameter | Value |
|-----------|-------|
| `LAMBDA_TEXT` | 0.5 |
| `LAMBDA_KL` | 0.0 |
| `CONCEPT_DROPOUT` | 0.3 |
| `ENSEMBLE_ALPHA` | 0.1 |
| Shots | 4 |
| Epochs | 50 |
| Backbone | ViT-B/16 |

See [`configs/trainers/CCPL/vit_b16_ep50_safe.yaml`](configs/trainers/CCPL/vit_b16_ep50_safe.yaml).

## Installation

### 1. Environment

```bash
conda create -y -n ccpl python=3.10
conda activate ccpl

# Install PyTorch matching your CUDA (example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Dassl

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -r requirements.txt
pip install -e .
cd ..
```

### 3. This repository

```bash
git clone https://github.com/YOUR_USERNAME/concept-constrained-prompt-learning.git
cd concept-constrained-prompt-learning
pip install -r requirements.txt
pip install ftfy regex tqdm yacs scipy scikit-learn gdown tabulate
pip install git+https://github.com/openai/CLIP.git
```

Detailed upstream install notes: [`docs/INSTALL.md`](docs/INSTALL.md).

## Dataset Preparation

Follow [`docs/DATASETS.md`](docs/DATASETS.md) to download and organize datasets under a single root, e.g.:

```text
/path/to/data/
  dtd/
  eurosat/
  oxford_pets/
  ...
```

Place Zhou-style split JSON files as required by Dassl (e.g. `split_zhou_DescribableTextures.json`). If split files are missing, the codebase may auto-generate fallback splits (fixed seed for reproducibility).

**Note:** Reported results in [`docs/RESULTS.md`](docs/RESULTS.md) use locally generated fallback splits, not official Zhou Google Drive splits.

## Training & Evaluation

### CCPL base-to-new (recommended)

```bash
export CUDA_VISIBLE_DEVICES=0
DATA=/path/to/data

# Train on base classes
bash scripts/ccpl/base2new_train.sh dtd 1 ${DATA}
bash scripts/ccpl/base2new_train.sh eurosat 1 ${DATA}

# Evaluate base / new
bash scripts/ccpl/base2new_test.sh dtd 1 base ${DATA}
bash scripts/ccpl/base2new_test.sh dtd 1 new ${DATA}
```

### Direct `train.py` example

```bash
python train.py \
  --root /path/to/data \
  --seed 1 \
  --trainer CCPL \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/CCPL/vit_b16_ep50_safe.yaml \
  --output-dir output/ccpl_dtd_4shot \
  DATASET.NUM_SHOTS 4 \
  DATASET.SUBSAMPLE_CLASSES base
```

### CoOp baseline (same protocol)

```bash
python train.py \
  --root /path/to/data \
  --seed 1 \
  --trainer CoOp \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/CoOp/vit_b16_ep50.yaml \
  --output-dir output/coop_dtd_4shot \
  DATASET.NUM_SHOTS 4 \
  DATASET.SUBSAMPLE_CLASSES base
```

## Concept Bank

Concept phrases are stored in [`concept_bank/ccpl_concepts.json`](concept_bank/ccpl_concepts.json).

- **EuroSAT:** class-specific scene attributes (hand-crafted)
- **DTD / OxfordPets:** template-based phrases such as `{class_name} texture`

Prompt template used in code:

```text
a photo of a {class_name}, which contains {concept}.
```

## Project Structure

```text
train.py                  # Entry point
trainers/ccpl.py          # CCPL trainer and model
concept_bank/             # Concept JSON
configs/trainers/CCPL/    # CCPL configs
scripts/ccpl/             # Base-to-new helper scripts
docs/RESULTS.md           # Reported experimental numbers
```

## Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for main base-to-new numbers, ablations, and protocol caveats.

## License

This project inherits the [PromptSRC LICENSE](LICENSE). Please also respect licenses of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), [CLIP](https://github.com/openai/CLIP), and dataset providers.

## Acknowledgements

Built on:

- [PromptSRC](https://github.com/muzairkhattak/PromptSRC) (ICCV 2023)
- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
- [CoOp / Co-CoOp](https://github.com/KaiyangZhou/CoOp)
- [OpenAI CLIP](https://github.com/openai/CLIP)

Upstream README: [`docs/PROMPTSRC_README.md`](docs/PROMPTSRC_README.md)

## Citation

If you use this code, please cite CCPL (paper link TBD) and the PromptSRC paper:

```bibtex
@inproceedings{khattak2023promptsrc,
  title={Self-regulating Prompts: Foundational Model Adaptation without Forgetting},
  author={Khattak, Muhammad Uzair and others},
  booktitle={ICCV},
  year={2023}
}
```
