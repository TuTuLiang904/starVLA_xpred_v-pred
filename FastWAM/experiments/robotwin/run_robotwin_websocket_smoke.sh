#!/usr/bin/env bash
# One-task FastWAM/RoboTwin smoke test using separate policy and simulator envs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_PATH:-${ROOT_DIR}/third_party/RoboTwin}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/bin/python}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/absolute/path/to/step_xxx.pt}"
DATASET_STATS="${DATASET_STATS:?Set DATASET_STATS=/absolute/path/to/dataset_stats.json}"
TASK_CONFIG="${TASK_CONFIG:-robotwin_uncond_3cam_384_1e-4}"
TASK_NAME="${TASK_NAME:-click_alarmclock}"
EVAL_MODE="${EVAL_MODE:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-1}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"

SERVER_LOG="${ROOT_DIR}/evaluate_results/robotwin_smoke/server_${TASK_NAME}_${EVAL_MODE}_${PORT}.log"
EVAL_LOG="${ROOT_DIR}/evaluate_results/robotwin_smoke/eval_${TASK_NAME}_${EVAL_MODE}_${PORT}.log"
mkdir -p "$(dirname "${SERVER_LOG}")"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  "${FASTWAM_PYTHON}" -u "${ROOT_DIR}/experiments/robotwin/fastwam_policy_server.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats "${DATASET_STATS}" \
  --task "${TASK_CONFIG}" \
  --project-root "${ROOT_DIR}" \
  --port "${PORT}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --replan-steps "${REPLAN_STEPS}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if "${FASTWAM_PYTHON}" -c "import socket; s=socket.create_connection(('127.0.0.1', ${PORT}), timeout=1); s.close()" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    cat "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  cat "${SERVER_LOG}" >&2
  exit 1
fi

cd "${ROBOTWIN_ROOT}"
PYTHONPATH="${ROBOTWIN_ROOT}:${ROOT_DIR}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  "${ROBOTWIN_PYTHON}" -u script/eval_policy.py \
  --config "${ROOT_DIR}/experiments/robotwin/fastwam_policy/deploy_policy_server.yml" \
  --policy_ckpt_path "${CHECKPOINT}" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${EVAL_MODE}" \
  --ckpt_setting "${CHECKPOINT}" \
  --seed 42 \
  --policy_name fastwam_policy \
  --eval_num_episodes "${NUM_EPISODES}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --replan_steps "${REPLAN_STEPS}" \
  --skip_get_obs_within_replan true |& tee "${EVAL_LOG}"
