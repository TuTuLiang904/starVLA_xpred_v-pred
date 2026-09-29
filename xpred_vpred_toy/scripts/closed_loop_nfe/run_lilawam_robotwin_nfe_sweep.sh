#!/usr/bin/env bash
set -euo pipefail

# LiLaWAM NFE sweep.  The evaluator receives LILAWAM_NUM_INFERENCE_STEPS and
# robotwin_infer.py applies it only at inference time; the checkpoint is never
# modified.

TOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "${TOY_ROOT}/.." && pwd)"
LILA_ROOT="${LILA_ROOT:-${PROJECT_ROOT}/LiLa-WAM}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/bin/python}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${PROJECT_ROOT}/FastWAM/third_party/RoboTwin}"
CKPT_SETTING="${CKPT_SETTING:?Set CKPT_SETTING, e.g. sft_full_vpred_stage2}"
CHECKPOINT_EP="${CHECKPOINT_EP:-4}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
TEST_NUM="${TEST_NUM:-50}"
SEED="${SEED:-42}"
NFE_VALUES_STR="${NFE_VALUES:-1 2 4 8 16}"
OUT_ROOT="${OUT_ROOT:-${TOY_ROOT}/results/closed_loop_nfe/lilawam_robotwin}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${LILA_ROOT}/checkpoints_vla/${CKPT_SETTING}/checkpoint_epoch_${CHECKPOINT_EP}.pt" ]] || {
  echo "LiLaWAM checkpoint not found under ${LILA_ROOT}/checkpoints_vla/${CKPT_SETTING}" >&2; exit 2;
}
read -r -a NFE_VALUES_ARR <<< "${NFE_VALUES_STR}"
mkdir -p "${OUT_ROOT}"
echo "LiLaWAM original RobotWin evaluator default: 10 (configs/robotwin_all.yaml common.num_inference_steps)"
echo "NFE sweep: ${NFE_VALUES_ARR[*]}"

index=0
for nfe in "${NFE_VALUES_ARR[@]}"; do
  [[ "${nfe}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid NFE: ${nfe}" >&2; exit 2; }
  run_dir="${OUT_ROOT}/nfe_${nfe}"
  mkdir -p "${run_dir}"
  echo "[lilawam] NFE=${nfe} output=${run_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'LILAWAM_NUM_INFERENCE_STEPS=%s LILAWAM_SAVE_ROOT=%q bash %q/eval_8gpu_all_tasks.sh\n' "${nfe}" "${run_dir}" "${LILA_ROOT}"
  else
    (
      cd "${LILA_ROOT}"
      export ROBOTWIN_PYTHON ROBOTWIN_ROOT CKPT_SETTING CHECKPOINT_EP TASK_CONFIG
      export GPUS PROCS_PER_GPU TEST_NUM SEED
      export LILAWAM_NUM_INFERENCE_STEPS="${nfe}" LILAWAM_SAVE_ROOT="${run_dir}"
      bash eval_8gpu_all_tasks.sh
    ) 2>&1 | tee "${run_dir}/run.log"
  fi
  index=$((index + 1))
done
