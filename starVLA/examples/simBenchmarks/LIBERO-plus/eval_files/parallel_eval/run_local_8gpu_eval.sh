#!/usr/bin/env bash
# Evaluate one LIBERO-Plus checkpoint on eight local GPUs.
#
# Each GPU owns one StarVLA policy server and one non-overlapping task shard.
# This preserves the validated websocket evaluation path: checkpoint config,
# action horizon, and training-time action unnormalization are all loaded by
# deployment/model_server/server_policy.py.
set -euo pipefail

# This script is one level below eval_files, so five parents reach the
# StarVLA repository root.
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
your_ckpt="${your_ckpt:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-9883}"
USE_BF16="${USE_BF16:-1}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
# A policy server otherwise lets PyTorch/BLAS create roughly one CPU worker per
# host core.  Four simultaneous 8-GPU jobs then create thousands of runnable
# threads, which makes model startup (and shared-filesystem reads) stall.
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-4}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_parallel/$(date +"%Y%m%d_%H%M%S")}" 

if [[ -z "${LIBERO_HOME}" || ! -f "${LIBERO_HOME}/libero/config.yaml" ]]; then
  echo "Set LIBERO_HOME to the configured LIBERO-plus repository root." >&2
  exit 2
fi
if [[ -z "${your_ckpt}" || ! -f "${your_ckpt}" ]]; then
  echo "Set your_ckpt to checkpoints/steps_120000_pytorch_model.pt." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ ${#GPUS[@]} -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight comma-separated GPU IDs; got '${GPU_IDS}'." >&2
  exit 2
fi
if ! [[ "${CPU_THREADS_PER_PROCESS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS_PER_PROCESS must be a positive integer; got '${CPU_THREADS_PER_PROCESS}'." >&2
  exit 2
fi
if ! [[ "${SERVER_START_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SERVER_START_TIMEOUT_SECONDS must be a positive integer; got '${SERVER_START_TIMEOUT_SECONDS}'." >&2
  exit 2
fi

cd "${STARVLA_DIR}"
mkdir -p "${output_dir}/servers" "${output_dir}/workers"
LOG_DIR="${output_dir}/shards"
mkdir -p "${LOG_DIR}"

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export LIBERO_HOME
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL PYOPENGL_PLATFORM
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/starvla_libero_plus_numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/starvla_libero_plus_mpl_cache}"
# Keep total host-side parallelism bounded when several checkpoints are
# evaluated at once.  These variables are inherited by both policy servers and
# MuJoCo evaluators; they do not change inference results.
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export STARVLA_CPU_THREADS="${CPU_THREADS_PER_PROCESS}"
export TOKENIZERS_PARALLELISM=false

SERVER_PIDS=()
EVAL_PIDS=()

cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_servers EXIT INT TERM

wait_for_server() {
  local pid="$1" port="$2"
  local attempts=$(((SERVER_START_TIMEOUT_SECONDS + 1) / 2))
  for _ in $(seq 1 "${attempts}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Policy server on port ${port} exited during startup." >&2
      return 1
    fi
    if (echo > "/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out after ${SERVER_START_TIMEOUT_SECONDS}s waiting for policy server on port ${port}." >&2
  return 1
}

# Two near-equal shards per suite: eight workers in total.
SUITES=(libero_spatial libero_object libero_goal libero_10)
SIZES=(2402 2518 2591 2519)

for suite_index in "${!SUITES[@]}"; do
  suite="${SUITES[$suite_index]}"
  size="${SIZES[$suite_index]}"
  for shard_index in 0 1; do
    worker_index=$((suite_index * 2 + shard_index))
    start_idx=$((size * shard_index / 2))
    end_idx=$((size * (shard_index + 1) / 2))
    gpu_id="${GPUS[$worker_index]}"
    port=$((BASE_PORT + worker_index))
    worker_name="${suite}_${start_idx}_${end_idx}"
    server_log="${output_dir}/servers/${worker_name}.log"

    server_args=(deployment/model_server/server_policy.py --ckpt_path "${your_ckpt}" --port "${port}")
    if [[ "${USE_BF16}" == "1" ]]; then
      server_args+=(--use_bf16)
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${STARVLA_PYTHON}" "${server_args[@]}" >"${server_log}" 2>&1 &
    SERVER_PIDS+=("$!")
    echo "server ${worker_name}: GPU ${gpu_id}, port ${port}, tasks [${start_idx}, ${end_idx})"
  done
done

for worker_index in "${!SERVER_PIDS[@]}"; do
  wait_for_server "${SERVER_PIDS[$worker_index]}" "$((BASE_PORT + worker_index))"
done

for suite_index in "${!SUITES[@]}"; do
  suite="${SUITES[$suite_index]}"
  size="${SIZES[$suite_index]}"
  for shard_index in 0 1; do
    worker_index=$((suite_index * 2 + shard_index))
    start_idx=$((size * shard_index / 2))
    end_idx=$((size * (shard_index + 1) / 2))
    gpu_id="${GPUS[$worker_index]}"
    port=$((BASE_PORT + worker_index))
    worker_name="${suite}_${start_idx}_${end_idx}"
    worker_log="${output_dir}/workers/${worker_name}.log"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_PYTHON}" \
      ./examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.py \
      --args.pretrained-path "${your_ckpt}" \
      --args.host 127.0.0.1 \
      --args.port "${port}" \
      --args.task-suite-name "${suite}" \
      --args.num-trials-per-task 1 \
      --args.start-idx "${start_idx}" \
      --args.end-idx "${end_idx}" \
      --args.log-path "${LOG_DIR}" \
      --args.video-out-path "${output_dir}/videos/${worker_name}" \
      >"${worker_log}" 2>&1 &
    EVAL_PIDS+=("$!")
  done
done

status=0
for pid in "${EVAL_PIDS[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one shard failed. Inspect ${output_dir}/workers and ${output_dir}/servers." >&2
  exit "${status}"
fi

export LOG_DIR
"${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO-plus/eval_files/aggregate_results.py
echo "Completed 8-GPU evaluation. Results: ${LOG_DIR}/summary.txt"
