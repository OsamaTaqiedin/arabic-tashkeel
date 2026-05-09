#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/project}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/output_strict}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/gru_seq2seq_runpod}"

cd "${PROJECT_ROOT}"

python train_tashkeel_seq2seq.py \
  --train-path "${DATA_ROOT}/train.jsonl" \
  --validation-path "${DATA_ROOT}/validation.jsonl" \
  --test-path "${DATA_ROOT}/test.jsonl" \
  --checkpoint-dir "${ARTIFACT_ROOT}" \
  --device cuda \
  --use-amp \
  --resume-from-latest \
  "$@"
