#!/usr/bin/env bash
set -euo pipefail

# One FastWAM policy server and one non-overlapping LIBERO-plus shard per GPU.
FASTWAM_DIR="${FASTWAM_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-}"
DATASET_STATS="${DATASET_STATS:-}"
TASK_CONFIG="${TASK_CONFIG:-libero_uncond_2cam224_1e-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-9900}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-4}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
OUTPUT_DIR="${OUTPUT_DIR:-${FASTWAM_DIR}/evaluate_results/libero_plus/$(date +"%Y%m%d_%H%M%S")}"

for required in LIBERO_HOME CHECKPOINT DATASET_STATS; do
  if [[ -z "${!required}" ]]; then
    echo "${required} is required." >&2
    exit 2
  fi
done
if [[ ! -f "${LIBERO_HOME}/libero/config.yaml" || ! -f "${CHECKPOINT}" || ! -f "${DATASET_STATS}" ]]; then
  echo "LIBERO_HOME, CHECKPOINT, or DATASET_STATS points to a missing file." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ ${#GPUS[@]} -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight comma-separated GPU IDs." >&2
  exit 2
fi

cd "${FASTWAM_DIR}"
mkdir -p "${OUTPUT_DIR}/servers" "${OUTPUT_DIR}/workers" "${OUTPUT_DIR}/shards"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${FASTWAM_DIR}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
# Keep local policy-server traffic off any inherited HTTP(S) proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam_libero_plus_numba_cache}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam_libero_plus_mpl_cache}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export TOKENIZERS_PARALLELISM=false

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local pid="$1" port="$2"
  local attempts=$(((SERVER_START_TIMEOUT_SECONDS + 1) / 2))
  for _ in $(seq 1 "${attempts}"); do
    kill -0 "${pid}" 2>/dev/null || return 1
    (: > "/dev/tcp/127.0.0.1/${port}") 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

SUITES=(libero_spatial libero_object libero_goal libero_10)
SIZES=(2402 2518 2591 2519)
for suite_index in "${!SUITES[@]}"; do
  for shard_index in 0 1; do
    worker_index=$((suite_index * 2 + shard_index))
    suite="${SUITES[$suite_index]}"
    size="${SIZES[$suite_index]}"
    start_idx=$((size * shard_index / 2))
    end_idx=$((size * (shard_index + 1) / 2))
    port=$((BASE_PORT + worker_index))
    gpu_id="${GPUS[$worker_index]}"
    name="${suite}_${start_idx}_${end_idx}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${FASTWAM_PYTHON}" -m experiments.libero_plus.fastwam_policy_server \
      --checkpoint "${CHECKPOINT}" --dataset-stats "${DATASET_STATS}" --task-config "${TASK_CONFIG}" \
      --port "${port}" --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      >"${OUTPUT_DIR}/servers/${name}.log" 2>&1 &
    SERVER_PIDS+=("$!")
    echo "server ${name}: GPU ${gpu_id}, port ${port}, tasks [${start_idx}, ${end_idx})"
  done
done

for worker_index in "${!SERVER_PIDS[@]}"; do
  wait_for_server "${SERVER_PIDS[$worker_index]}" "$((BASE_PORT + worker_index))" || {
    echo "FastWAM server ${worker_index} failed to start; inspect ${OUTPUT_DIR}/servers." >&2
    exit 1
  }
done

for suite_index in "${!SUITES[@]}"; do
  for shard_index in 0 1; do
    worker_index=$((suite_index * 2 + shard_index))
    suite="${SUITES[$suite_index]}"
    size="${SIZES[$suite_index]}"
    start_idx=$((size * shard_index / 2))
    end_idx=$((size * (shard_index + 1) / 2))
    port=$((BASE_PORT + worker_index))
    gpu_id="${GPUS[$worker_index]}"
    name="${suite}_${start_idx}_${end_idx}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_PYTHON}" -m experiments.libero_plus.eval_fastwam \
      --libero-home "${LIBERO_HOME}" --host 127.0.0.1 --port "${port}" --task-suite-name "${suite}" \
      --start-idx "${start_idx}" --end-idx "${end_idx}" --num-trials "${NUM_TRIALS}" \
      --replan-steps "${REPLAN_STEPS}" --log-path "${OUTPUT_DIR}/shards" \
      >"${OUTPUT_DIR}/workers/${name}.log" 2>&1 &
    WORKER_PIDS+=("$!")
  done
done

status=0
for pid in "${WORKER_PIDS[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || { echo "A LIBERO-plus worker failed; inspect ${OUTPUT_DIR}/workers." >&2; exit "${status}"; }
"${LIBERO_PYTHON}" -m experiments.libero_plus.aggregate_results --log-dir "${OUTPUT_DIR}/shards"
echo "Completed FastWAM LIBERO-plus evaluation: ${OUTPUT_DIR}/shards/summary.txt"
