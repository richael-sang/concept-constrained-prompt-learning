# Release Manifest

This directory is the **public open-source release** of CCPL, filtered from the local experiment workspace.

## Included

- CCPL trainer: `trainers/ccpl.py`
- Concept bank: `concept_bank/ccpl_concepts.json`
- CCPL configs: `configs/trainers/CCPL/`
- Upstream PromptSRC code (trainers, datasets, configs, docs, scripts)
- Helper scripts: `scripts/ccpl/`
- Documentation: `README.md`, `docs/RESULTS.md`, `docs/INSTALL.md`, `docs/DATASETS.md`

## Excluded (not in this repo)

- `data/` — raw datasets and split archives
- `outputs/`, `logs/` — experiment checkpoints and logs
- `paper/` — LaTeX manuscript
- `notes/`, `CCPL_PROGRESS.md` — internal experiment notes
- Model weights (`.pth`, `.pth.tar`, `.pt`, `.ckpt`)

## Audit (2026-05-31)

- No absolute home paths or API keys found in release tree
- No checkpoint or dataset archives included
- Largest files: `clip/bpe_simple_vocab_16e6.txt.gz` (required), `docs/main_figure.png` (upstream doc asset)
