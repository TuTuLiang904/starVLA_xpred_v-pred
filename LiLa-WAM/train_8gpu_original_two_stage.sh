#!/usr/bin/env bash
set -euo pipefail

# LiLa-WAM original RoboTwin full-data recipe:
#   data_mode=both (50 clean + 500 randomized, excluding prepared outliers)
#   stage 1: 12 epochs @ 2e-4
#   stage 2: 4 epochs  @ 4e-5, model weights only
#
# Examples:
#   ACTION_PREDICTION_TYPE=v_prediction bash train_8gpu_original_two_stage.sh
#   ACTION_PREDICTION_TYPE=x_prediction RUN_ID=full_xpred bash train_8gpu_original_two_stage.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GPUS="${GPUS:-8}"
CONFIG="${CONFIG:-$ROOT/configs/robotwin_all.yaml}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints_vla}"
RUN_ID="${RUN_ID:-full_8gpu_$(date +%Y%m%d_%H%M%S)}"
ACTION_PREDICTION_TYPE="${ACTION_PREDICTION_TYPE:-v_prediction}"

case "$ACTION_PREDICTION_TYPE" in
  v_prediction|x_prediction|x_prediction_v_loss) ;;
  *) echo "Unsupported ACTION_PREDICTION_TYPE=$ACTION_PREDICTION_TYPE" >&2; exit 2 ;;
esac

STAGE1_ID="${RUN_ID}_stage1"
STAGE2_ID="${RUN_ID}_stage2"
STAGE1_CONFIG="${TMPDIR:-/tmp}/lilawam_${RUN_ID}_stage1.yaml"
STAGE2_CONFIG="${TMPDIR:-/tmp}/lilawam_${RUN_ID}_stage2.yaml"

cp "$CONFIG" "$STAGE1_CONFIG"

# Override only the experiment-specific fields; all architecture settings stay
# in the repository config.
sed -i -E \
  -e 's/^([[:space:]]*data_mode:).*/\1 "both"/' \
  -e 's#^([[:space:]]*task_cond_dir:).*#\1 "./data-500-taskcond"#' \
  -e 's/^([[:space:]]*action_prediction_type:).*/\1 "'"$ACTION_PREDICTION_TYPE"'"/' \
  -e 's/^([[:space:]]*learning_rate:).*/\1 2e-4/' \
  -e 's/^([[:space:]]*lr_min:).*/\1 5.0e-5/' \
  -e 's/^([[:space:]]*epochs:).*/\1 12/' \
  "$STAGE1_CONFIG"

echo "[LiLa-WAM] Original full-data Stage 1"
echo "  prediction=$ACTION_PREDICTION_TYPE, data_mode=both, epochs=12, lr=2e-4"
RUN_ID="$STAGE1_ID" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  torchrun --standalone --nproc_per_node="$GPUS" \
  train.py --config "$STAGE1_CONFIG" --save_dir "$SAVE_DIR"

STAGE1_DIR="$SAVE_DIR/sft_${STAGE1_ID}"
STAGE1_CKPT="$STAGE1_DIR/checkpoint_epoch_12.pt"
test -f "$STAGE1_CKPT"

# Crucially, derive Stage 2 from the saved Stage-1 config so prediction mode,
# data mode, task conditions, and model architecture cannot silently change.
cp "$STAGE1_DIR/config.yaml" "$STAGE2_CONFIG"
sed -i -E \
  -e 's/^([[:space:]]*learning_rate:).*/\1 4e-5/' \
  -e 's/^([[:space:]]*lr_min:).*/\1 1.0e-5/' \
  -e 's/^([[:space:]]*epochs:).*/\1 4/' \
  "$STAGE2_CONFIG"

echo "[LiLa-WAM] Original full-data Stage 2"
echo "  prediction=$ACTION_PREDICTION_TYPE, data_mode=both, epochs=4, lr=4e-5"
RUN_ID="$STAGE2_ID" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  torchrun --standalone --nproc_per_node="$GPUS" \
  train.py --config "$STAGE2_CONFIG" --save_dir "$SAVE_DIR" \
  --init_from "$STAGE1_CKPT"

echo "Training complete. Final checkpoint directory: $SAVE_DIR/sft_${STAGE2_ID}"
