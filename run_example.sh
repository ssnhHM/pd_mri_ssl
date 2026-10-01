#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-data/ppmi_ssl_severity_manifest_3T_scanner_corrected.csv}"
CONFIG="${CONFIG:-}"
PHASE="${PHASE:-all}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-3407 3408 3409 3410 3411}"
FOLDS="${FOLDS:-1 2 3 4 5}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best}"
OUT="${OUT:-results/e1_e9_3t}"
TRAIN_ROOT="${TRAIN_ROOT:-$OUT/training}"
EMBEDDING_ROOT="${EMBEDDING_ROOT:-$OUT/embeddings}"
PROBING_ROOT="${PROBING_ROOT:-}"
FORCE="${FORCE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

case "$PHASE" in all|train|extract|nested) ;; *) echo "Unknown PHASE=$PHASE" >&2; exit 2 ;; esac
if [[ "$CHECKPOINT_NAME" != checkpoint_best ]]; then
  echo "The current-paper pipeline uses checkpoint_best." >&2
  exit 2
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Set MANIFEST to the private PPMI analysis manifest." >&2
  exit 2
fi

configs=(configs/*.yml)
if [[ -n "$CONFIG" ]]; then configs=("$CONFIG"); fi
experiments=()
train_cmd=("$PYTHON_BIN" train_global_ssl.py --manifest "$MANIFEST"
           --device "$DEVICE" --output-root "$TRAIN_ROOT")
if [[ "$FORCE" == 1 ]]; then train_cmd+=(--force); fi
echo "[PD MRI SSL] 3T device=$DEVICE seeds=$SEEDS folds=$FOLDS"

for config in "${configs[@]}"; do
  experiment="$("$PYTHON_BIN" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"])' "$config")"
  experiments+=("$experiment")
  echo "[PD MRI SSL] experiment=$experiment config=$config phase=$PHASE"
  for seed in $SEEDS; do
    if [[ "$PHASE" == all || "$PHASE" == train ]]; then
      for fold in $FOLDS; do
        "${train_cmd[@]}" --config "$config" --seed "$seed" --fold "$fold"
      done
    fi

    if [[ "$PHASE" == all || "$PHASE" == extract ]]; then
      "$PYTHON_BIN" scripts/extract_global_embeddings.py \
        --config "$config" \
        --manifest "$MANIFEST" \
        --checkpoint-root "$TRAIN_ROOT" \
        --output-root "$EMBEDDING_ROOT" \
        --seed "$seed" \
        --folds $FOLDS \
        --checkpoint-names "$CHECKPOINT_NAME" \
        --device "$DEVICE"
    fi
  done
done

if [[ "$PHASE" == all || "$PHASE" == nested ]]; then
  if [[ -z "$PROBING_ROOT" ]]; then
    PROBING_ROOT="$OUT/r2/nested"
    if [[ -n "$CONFIG" ]]; then PROBING_ROOT="$PROBING_ROOT/${experiments[0]}"; fi
  fi
  "$PYTHON_BIN" scripts/frozen_probe.py \
    --manifest "$MANIFEST" \
    --embedding-root "$EMBEDDING_ROOT" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --experiments "${experiments[@]}" \
    --seeds $SEEDS --folds $FOLDS \
    --endpoints putamen_sbr_min \
    --cohort pd_only \
    --output-dir "$PROBING_ROOT"
fi

echo "[PD MRI SSL] finished phase=$PHASE"
