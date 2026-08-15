#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
SOURCE_DATASET="${SOURCE_DATASET:-/mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/FastWAM/data/robotwin2.0/robotwin2.0}"
SOURCE_STATS="${SOURCE_STATS:-/mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/FastWAM/data/robotwin2.0/dataset_stats.json}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/playground/Datasets/RoboTwin}"
PREDICTION_TYPE="${PREDICTION_TYPE:-x_prediction}"
TRAIN_SPLIT="${TRAIN_SPLIT:-clean}"
RUN_ID="${RUN_ID:-qwen_gr00t_robotwin_${PREDICTION_TYPE}_seed42}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${ROOT_DIR}/playground/Checkpoints}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-75000}"

case "${PREDICTION_TYPE}" in
  v_prediction|x_prediction|x_prediction_v_loss) ;;
  *) echo "Unsupported PREDICTION_TYPE=${PREDICTION_TYPE}" >&2; exit 2 ;;
esac
case "${TRAIN_SPLIT}" in
  clean) DATA_MIX="robotwin_clean_50" ;;
  full) DATA_MIX="robotwin_all_50_500" ;;
  *) echo "Unsupported TRAIN_SPLIT=${TRAIN_SPLIT}; use clean or full" >&2; exit 2 ;;
esac
for numeric_setting in NUM_PROCESSES PER_DEVICE_BATCH_SIZE MAX_TRAIN_STEPS; do
  value="${!numeric_setting}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${numeric_setting} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done

cd "${ROOT_DIR}"
python examples/simBenchmarks/Robotwin/train_files/prepare_robotwin_dataset.py \
  --source "${SOURCE_DATASET}" --source-stats "${SOURCE_STATS}" --output-root "${DATA_ROOT}"
GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE))
echo "[INFO] train_split=${TRAIN_SPLIT} data_mix=${DATA_MIX} prediction_type=${PREDICTION_TYPE}"
echo "[INFO] run_root_dir=${RUN_ROOT_DIR} per_device_batch_size=${PER_DEVICE_BATCH_SIZE} global_batch_size=${GLOBAL_BATCH_SIZE} max_train_steps=${MAX_TRAIN_STEPS}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/Robotwin/train_files/starvla_qwengr00t_robotwin.yaml \
  --framework.action_model.prediction_type "${PREDICTION_TYPE}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
