#!/bin/bash
# GR00T on LIBERO 4-in-1. Run from the starVLA repository root.

# GPUs visible to this training job. Keep NUM_PROCESSES equal to their count.
GPU_IDS=4,5,6,7
NUM_PROCESSES=4
MAIN_PROCESS_PORT=29503

# Modes: v_prediction (GR00T baseline), x_prediction (direct x-MSE), or
# x_prediction_v_loss (ABot AML: clean-action output + velocity-MSE).
PREDICTION_TYPE=${PREDICTION_TYPE:-v_prediction}
RUN_ID=${RUN_ID:-layers8_${PREDICTION_TYPE}_seed42_ft}

CONFIG_YAML=./examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
RUN_ROOT=./playground/Checkpoints

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export WANDB_MODE=disabled

OUTPUT_DIR=${RUN_ROOT}/${RUN_ID}
mkdir -p ${OUTPUT_DIR}
cp $0 ${OUTPUT_DIR}/

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${MAIN_PROCESS_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${CONFIG_YAML} \
  --framework.action_model.prediction_type ${PREDICTION_TYPE} \
  --run_root_dir ${RUN_ROOT} \
  --run_id ${RUN_ID}
