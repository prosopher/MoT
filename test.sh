#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configurable settings
# =========================
CUDA_DEVICE=0

CHECKPOINT_BASE="output_checkpoints/_scalability/bfloat16_Qwen2.5-0.5B→Qwen2.5-0.5B"

C2C_CHECKPOINT="${CHECKPOINT_BASE}/c2c_20260430_210719"
INTERLAT_CHECKPOINT="${CHECKPOINT_BASE}/interlat_20260430_214051"

MOT_CHECKPOINT_PREFIX="${CHECKPOINT_BASE}/mot_topk-sparse-attn="

OUTPUT_BASE="outputs"

BATCH_SIZE=2
MAX_EXAMPLES_PER_DATASET=100000
DATASET_FILTER="hotpotqa_e"

CONTEXT_BUDGETS=(16384 8192 4096)
TOPK_SPARSE_ATTN_LIST=(64 32 16)

# =========================
# Sweep
# =========================
for CONTEXT_BUDGET in "${CONTEXT_BUDGETS[@]}"; do
  echo "========================================"
  echo "Running context budget: ${CONTEXT_BUDGET}"
  echo "========================================"

  echo "[c2c] context=${CONTEXT_BUDGET}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python eval.py c2c \
    --checkpoint-dir-path "${C2C_CHECKPOINT}" \
    --output-path "${OUTPUT_BASE}/c2c_${DATASET_FILTER}_ctx=${CONTEXT_BUDGET}" \
    --batch-size "${BATCH_SIZE}" \
    --max-examples-per-dataset "${MAX_EXAMPLES_PER_DATASET}" \
    --generation-dataset-filter "${DATASET_FILTER}" \
    --generation-context-budgets "${CONTEXT_BUDGET}"

  echo "[interlat] context=${CONTEXT_BUDGET}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python eval.py interlat \
    --checkpoint-dir-path "${INTERLAT_CHECKPOINT}" \
    --output-path "${OUTPUT_BASE}/interlat_${DATASET_FILTER}_ctx=${CONTEXT_BUDGET}" \
    --batch-size "${BATCH_SIZE}" \
    --max-examples-per-dataset "${MAX_EXAMPLES_PER_DATASET}" \
    --generation-dataset-filter "${DATASET_FILTER}" \
    --generation-context-budgets "${CONTEXT_BUDGET}"

  for TOPK_SPARSE_ATTN in "${TOPK_SPARSE_ATTN_LIST[@]}"; do
    echo "[mot] context=${CONTEXT_BUDGET}, topk_sparse_attn=${TOPK_SPARSE_ATTN}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python eval.py mot \
      --checkpoint-dir-path "${MOT_CHECKPOINT_PREFIX}${TOPK_SPARSE_ATTN}" \
      --output-path "${OUTPUT_BASE}/mot_topk${TOPK_SPARSE_ATTN}_${DATASET_FILTER}_ctx=${CONTEXT_BUDGET}" \
      --batch-size "${BATCH_SIZE}" \
      --max-examples-per-dataset "${MAX_EXAMPLES_PER_DATASET}" \
      --generation-dataset-filter "${DATASET_FILTER}" \
      --generation-context-budgets "${CONTEXT_BUDGET}"
  done
done

curl -X POST -H 'Content-type: application/json' --data '{"text":"<@U0ACKM9LA10> finished Test(LongContext sweep) on GPU 0 at Code 2"}' https://hooks.slack.com/services/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
