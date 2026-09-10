#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/venv/main/bin/python}"
DATA="${DATA:-${ROOT}/work/support/data/sft.jsonl}"
VOCAB_PRUNED="${VOCAB_PRUNED:-${ROOT}/work/support/merged}"
RUN_DIR="${RUN_DIR:-${ROOT}/work/support/distill-10l}"
TEACHER="${TEACHER:-${RUN_DIR}/teacher}"
STUDENT="${STUDENT:-${RUN_DIR}/student}"
HEALED="${HEALED:-${RUN_DIR}/healed}"
HOLDOUT="${HOLDOUT:-${RUN_DIR}/holdout_ids.txt}"

mkdir -p "${RUN_DIR}"

"${PYTHON}" "${ROOT}/sft_teacher.py" \
  --model "${VOCAB_PRUNED}" \
  --data "${DATA}" \
  --holdout-out "${HOLDOUT}" \
  --out "${TEACHER}" \
  --epochs 2 \
  --lr 2e-5 \
  --augment 8 \
  --probe-rows 300 \
  --max-len 1536

"${PYTHON}" "${ROOT}/prune_layers.py" \
  --model "${TEACHER}" \
  --data "${DATA}" \
  --holdout "${HOLDOUT}" \
  --keep 10 \
  --force-keep 0,1,27 \
  --strategy per-layer \
  --max-seqs 600 \
  --out "${STUDENT}"

"${PYTHON}" "${ROOT}/heal_kd.py" \
  --student "${STUDENT}" \
  --teacher "${TEACHER}" \
  --data "${DATA}" \
  --holdout "${HOLDOUT}" \
  --epochs 3 \
  --lr 1e-4 \
  --alpha 0.5 \
  --temperature 2.0 \
  --augment 8 \
  --probe-rows 300 \
  --max-len 1536 \
  --out "${HEALED}"

printf 'Healed checkpoint: %s\n' "${HEALED}"
