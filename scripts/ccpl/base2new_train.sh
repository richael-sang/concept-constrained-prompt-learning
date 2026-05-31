#!/bin/bash
# Train CCPL on base classes (base-to-new protocol).
#
# Usage:
#   bash scripts/ccpl/base2new_train.sh dtd 1 /path/to/data
#   bash scripts/ccpl/base2new_train.sh eurosat 1 /path/to/data

set -e

DATASET=${1:?dataset name, e.g. dtd, eurosat, oxford_pets}
SEED=${2:-1}
DATA=${3:-/path/to/data}

TRAINER=CCPL
CFG=vit_b16_ep50_safe
SHOTS=4

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

echo "Training CCPL on base classes: dataset=${DATASET}, seed=${SEED}"
echo "Output directory: ${DIR}"

python train.py \
  --root "${DATA}" \
  --seed "${SEED}" \
  --trainer "${TRAINER}" \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/CCPL/${CFG}.yaml \
  --output-dir "${DIR}" \
  DATASET.NUM_SHOTS "${SHOTS}" \
  DATASET.SUBSAMPLE_CLASSES base
