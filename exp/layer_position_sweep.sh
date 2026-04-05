#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/layer_position}
SWEEP_ID=${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}
COMMON_ARGS=("$@")

for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  case "${arg}" in
    --study-id|--study-id=*|--benchmark-mode|--benchmark-mode=*|--injection-window-size|--injection-window-size=*)
      echo "layer_position_sweep.sh controls --study-id, --benchmark-mode, and --injection-window-size automatically" >&2
      exit 1
      ;;
  esac
done

BENCHMARK_MODES=(gen_qa)
INJECTION_WINDOW_SIZES=(5)

for benchmark_mode in "${BENCHMARK_MODES[@]}"; do
  for injection_window_size in "${INJECTION_WINDOW_SIZES[@]}"; do
    study_id="${SWEEP_ID}_${benchmark_mode}_iws${injection_window_size}"

    echo "[LayerPositionSweep] ===== ${study_id} ====="
    bash exp/layer_position.sh \
      --study-id "${study_id}" \
      --benchmark-mode "${benchmark_mode}" \
      --injection-window-size "${injection_window_size}" \
      "${COMMON_ARGS[@]}"
  done
done

echo "[LayerPositionSweep] done"
echo "[LayerPositionSweep] output_path=${OUTPUT_ROOT}"
echo "[LayerPositionSweep] sweep_id=${SWEEP_ID}"
