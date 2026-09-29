#!/usr/bin/env bash
# Run all RoboTwin tasks with one FastWAM server per GPU and RoboTwin clients
# in the untouched RoboTwin environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_PATH:-${ROOT_DIR}/third_party/RoboTwin}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/absolute/path/to/step_xxx.pt}"
DATASET_STATS="${DATASET_STATS:?Set DATASET_STATS=/absolute/path/to/dataset_stats.json}"
TASK_CONFIG="${TASK_CONFIG:-robotwin_uncond_3cam_384_1e-4}"
EVAL_MODE="${EVAL_MODE:-demo_clean}"
NUM_EPISODES="${NUM_EPISODES:-100}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-16994}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"

if [[ "${EVAL_MODE}" != demo_clean && "${EVAL_MODE}" != demo_randomized ]]; then
  echo "EVAL_MODE must be demo_clean or demo_randomized" >&2; exit 2
fi
if (( NUM_GPUS < 1 || NUM_GPUS > 8 )); then echo "NUM_GPUS must be 1..8" >&2; exit 2; fi

RUN_DIR="${OUTPUT_DIR:-${ROOT_DIR}/evaluate_results/robotwin_websocket/$(basename "${CHECKPOINT%.*}")_${EVAL_MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_DIR}/servers" "${RUN_DIR}/tasks"
SERVER_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_port() {
  local port="$1"
  for _ in $(seq 1 180); do
    if "${ROBOTWIN_PYTHON}" -c "import socket; s=socket.create_connection(('127.0.0.1',${port}),1); s.close()" 2>/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  port=$((BASE_PORT + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
    "${FASTWAM_PYTHON}" -u "${ROOT_DIR}/experiments/robotwin/fastwam_policy_server.py" \
    --checkpoint "${CHECKPOINT}" --dataset-stats "${DATASET_STATS}" \
    --task "${TASK_CONFIG}" --project-root "${ROOT_DIR}" --port "${port}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" --replan-steps "${REPLAN_STEPS}" \
    >"${RUN_DIR}/servers/gpu${gpu}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  wait_port $((BASE_PORT + gpu)) || { echo "FastWAM server $gpu did not start; see ${RUN_DIR}/servers/gpu${gpu}.log" >&2; exit 1; }
done

mapfile -t TASKS < <(awk -F: 'NF >= 2 && $1 !~ /^#/ {gsub(/[[:space:]]/,"",$1); print $1}' "${ROOT_DIR}/third_party/RoboTwin/task_config/_eval_step_limit.yml")
printf 'mode=%s episodes=%s tasks=%s gpus=%s\n' "${EVAL_MODE}" "${NUM_EPISODES}" "${#TASKS[@]}" "${NUM_GPUS}" | tee "${RUN_DIR}/run.log"

PIDS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  (
    port=$((BASE_PORT + gpu))
    for ((i=gpu; i<${#TASKS[@]}; i+=NUM_GPUS)); do
      task="${TASKS[$i]}"
      echo "[gpu${gpu}] starting ${task}" | tee -a "${RUN_DIR}/run.log"
      (
        cd "${ROBOTWIN_ROOT}"
        PYTHONPATH="${ROBOTWIN_ROOT}:${ROOT_DIR}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
          "${ROBOTWIN_PYTHON}" -u script/eval_policy.py \
          --config "${ROOT_DIR}/experiments/robotwin/fastwam_policy/deploy_policy_server.yml" \
          --policy_ckpt_path "${CHECKPOINT}" --overrides \
          --task_name "${task}" --task_config "${EVAL_MODE}" --ckpt_setting "${CHECKPOINT}" \
          --seed "${SEED}" --policy_name fastwam_policy --eval_num_episodes "${NUM_EPISODES}" \
          --host 127.0.0.1 --port "${port}" --replan_steps "${REPLAN_STEPS}" \
          --skip_get_obs_within_replan true |& tee "${RUN_DIR}/tasks/${task}.log"
      )
    done
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
echo "Evaluation logs and RoboTwin results: ${RUN_DIR}" | tee -a "${RUN_DIR}/run.log"
exit "${status}"
