#!/usr/bin/env bash
set -euo pipefail

# FastWAM already exposes NUM_INFERENCE_STEPS in its untouched RobotWin
# websocket evaluator.  This wrapper only loops over that public knob.

TOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "${TOY_ROOT}/.." && pwd)"
FAST_ROOT="${FAST_ROOT:-${PROJECT_ROOT}/FastWAM}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/absolute/path/to/FastWAM checkpoint}"
DATASET_STATS="${DATASET_STATS:?Set DATASET_STATS=/absolute/path/to/dataset_stats.json}"
TASK_CONFIG="${TASK_CONFIG:-robotwin_uncond_3cam_384_1e-4_clean50}"
EVAL_MODE="${EVAL_MODE:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-50}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-27900}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NFE_VALUES_STR="${NFE_VALUES:-1 2 4 8 16}"
OUT_ROOT="${OUT_ROOT:-${TOY_ROOT}/results/closed_loop_nfe/fastwam_robotwin}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATASET_STATS}" ]] || { echo "Dataset stats not found: ${DATASET_STATS}" >&2; exit 2; }
read -r -a NFE_VALUES_ARR <<< "${NFE_VALUES_STR}"
mkdir -p "${OUT_ROOT}"
echo "FastWAM original RobotWin evaluator default: 10 (run_robotwin_websocket_all.sh)"
echo "NFE sweep: ${NFE_VALUES_ARR[*]}"

index=0
for nfe in "${NFE_VALUES_ARR[@]}"; do
  [[ "${nfe}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid NFE: ${nfe}" >&2; exit 2; }
  run_dir="${OUT_ROOT}/nfe_${nfe}"
  mkdir -p "${run_dir}"
  echo "[fastwam] NFE=${nfe} output=${run_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'NUM_INFERENCE_STEPS=%s BASE_PORT=%s bash %q/experiments/robotwin/run_robotwin_websocket_all.sh\n' "${nfe}" "$((BASE_PORT + index * 100))" "${FAST_ROOT}"
  else
    (
      cd "${FAST_ROOT}"
      export CHECKPOINT DATASET_STATS FASTWAM_PYTHON ROBOTWIN_PYTHON
      export TASK_CONFIG EVAL_MODE NUM_EPISODES NUM_GPUS REPLAN_STEPS
      export NUM_INFERENCE_STEPS="${nfe}" BASE_PORT=$((BASE_PORT + index * 100))
      export OUTPUT_DIR="${run_dir}"
      bash experiments/robotwin/run_robotwin_websocket_all.sh
    ) 2>&1 | tee "${run_dir}/run.log"
  fi
  index=$((index + 1))
done
