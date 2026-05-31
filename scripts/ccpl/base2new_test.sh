#!/bin/bash
# Evaluate CCPL on base or new classes (base-to-new protocol).
#
# Usage:
#   bash scripts/ccpl/base2new_test.sh dtd 1 base /path/to/data
#   bash scripts/ccpl/base2new_test.sh dtd 1 new /path/to/data
#
# Requires a trained checkpoint from base2new_train.sh.

set -e

DATASET=${1:?dataset name}
SEED=${2:-1}
SUB=${3:?base or new}
DATA=${4:-/path/to/data}

TRAINER=CCPL
CFG=vit_b16_ep50_safe
SHOTS=4

TRAIN_DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
TEST_DIR=output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ ! -d "${TRAIN_DIR}" ]; then
  echo "Missing trained model: ${TRAIN_DIR}"
  echo "Run: bash scripts/ccpl/base2new_train.sh ${DATASET} ${SEED} ${DATA}"
  exit 1
fi

echo "Evaluating CCPL on ${SUB} classes: dataset=${DATASET}, seed=${SEED}"
echo "Loading weights from: ${TRAIN_DIR}"
echo "Output directory: ${TEST_DIR}"

python train.py \
  --root "${DATA}" \
  --seed "${SEED}" \
  --trainer "${TRAINER}" \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/CCPL/${CFG}.yaml \
  --output-dir "${TEST_DIR}" \
  --model-dir "${TRAIN_DIR}" \
  --load-epoch 50 \
  --eval-only \
  DATASET.NUM_SHOTS "${SHOTS}" \
  DATASET.SUBSAMPLE_CLASSES "${SUB}"
