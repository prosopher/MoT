#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

COMMON_ARGS=("$@")
VARIANT_VALUES=(mot)
TOP_K_VALUES=(2)
NUM_TRANSLATORS_VALUES=(4)
OUTPUT_ROOT=outputs/correction

for variant in "${VARIANT_VALUES[@]}"; do
  for top_k in "${TOP_K_VALUES[@]}"; do
    for num_translators in "${NUM_TRANSLATORS_VALUES[@]}"; do
      if (( top_k >= num_translators )); then
        echo "[CorrectionSweep] skipping Variant=${variant} TopK=${top_k} NumT=${num_translators} because TopK must be < NumT"
        continue
      fi

      study_id="Variant=${variant}_TopK=${top_k}_NumT=${num_translators}"
      echo "[CorrectionSweep] running OUTPUT_ROOT=${OUTPUT_ROOT} STUDY_ID=${study_id}"

      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      STUDY_ID="${study_id}" \
      bash exp/correction.sh \
        --variant "${variant}" \
        --mot-top-k "${top_k}" \
        --mot-num-translators "${num_translators}" \
        "${COMMON_ARGS[@]}"
    done
  done
done