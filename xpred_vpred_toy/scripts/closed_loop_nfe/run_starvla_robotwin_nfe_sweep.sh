#!/usr/bin/env bash
set -euo pipefail

# Closed-loop RobotWin NFE sweep for StarVLA.  This wrapper leaves the original
# StarVLA evaluator intact and uses its optional STARVLA_CONFIG_OVERRIDE hook.

TOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "${TOY_ROOT}/.." && pwd)"
STAR_ROOT="${STAR_ROOT:-${PROJECT_ROOT}/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/starVLA/bin/python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/bin/python}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-${PROJECT_ROOT}/FastWAM/third_party/RoboTwin}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/absolute/path/to/StarVLA checkpoint}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
ROBOTWIN_SEED="${ROBOTWIN_SEED:-42}"
NUM_EPISODES="${NUM_EPISODES:-50}"
ROBOTWIN_JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-26900}"
NFE_VALUES_STR="${NFE_VALUES:-1 2 4 8 16}"
TASKS_SPEC="${TASKS:-all}"
OUT_ROOT="${OUT_ROOT:-${TOY_ROOT}/results/closed_loop_nfe/starvla_robotwin}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ "${TASK_CONFIG}" != demo_clean && "${TASK_CONFIG}" != demo_randomized ]]; then
  echo "TASK_CONFIG must be demo_clean or demo_randomized" >&2
  exit 2
fi
read -r -a NFE_VALUES_ARR <<< "${NFE_VALUES_STR}"
read -r -a TASK_ARGS <<< "${TASKS_SPEC}"
(( ${#NFE_VALUES_ARR[@]} > 0 )) || { echo "NFE_VALUES is empty" >&2; exit 2; }
(( ${#TASK_ARGS[@]} > 0 )) || { echo "TASKS is empty" >&2; exit 2; }

mkdir -p "${OUT_ROOT}"
echo "StarVLA checkpoint solver default: 4 (checkpoint config num_inference_timesteps)"
echo "NFE sweep: ${NFE_VALUES_ARR[*]}"
echo "RobotWin GPUs: ${GPU_IDS} (override with GPU_IDS=... if needed)"

index=0
for nfe in "${NFE_VALUES_ARR[@]}"; do
  [[ "${nfe}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid NFE: ${nfe}" >&2; exit 2; }
  run_dir="${OUT_ROOT}/nfe_${nfe}"
  mkdir -p "${run_dir}"
  echo "[starvla] NFE=${nfe} output=${run_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'STARVLA_CONFIG_OVERRIDE=framework.action_model.num_inference_timesteps=%s ' "${nfe}"
    printf 'ROBOTWIN_BASE_PORT=%s bash %q' "$((BASE_PORT + index * 100))" "${STAR_ROOT}"
    printf '/examples/simBenchmarks/Robotwin/eval_files/start_eval.sh '
    printf -- '-m %q -n %q -s %q -c %q' "${TASK_CONFIG}" "starvla_nfe${nfe}" "${ROBOTWIN_SEED}" "${CHECKPOINT}"
    printf ' %q' "${TASK_ARGS[@]}"
    printf '\n'
  else
    (
      # StarVLA checkpoint configs contain repository-relative VLM paths such
      # as ./playground/Pretrained_models/..., so the server must inherit the
      # StarVLA repository as its working directory.
      cd "${STAR_ROOT}"
      export STARVLA_PYTHON ROBOTWIN_PYTHON ROBOTWIN_PATH
      export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
      export STARVLA_CONFIG_OVERRIDE="framework.action_model.num_inference_timesteps=${nfe}"
      export ROBOTWIN_BASE_PORT=$((BASE_PORT + index * 100))
      export ROBOTWIN_LOG_ROOT="${run_dir}/logs"
      export ROBOTWIN_SEED ROBOTWIN_JOBS_PER_GPU
      export ROBOTWIN_NUM_EPISODES="${NUM_EPISODES}"
      bash "${STAR_ROOT}/examples/simBenchmarks/Robotwin/eval_files/start_eval.sh" \
        -m "${TASK_CONFIG}" -n "starvla_nfe${nfe}" -s "${ROBOTWIN_SEED}" \
        -c "${CHECKPOINT}" "${TASK_ARGS[@]}"
    ) 2>&1 | tee "${run_dir}/run.log"
  fi
  index=$((index + 1))
done
