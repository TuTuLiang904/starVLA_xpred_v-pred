#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GPUS="${GPUS:-8}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$ROOT/configs/robotwin_all.yaml}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints_vla}"
RUN_ID="${RUN_ID:-clean_8gpu_$(date +%Y%m%d_%H%M%S)}"
ACTION_PREDICTION_TYPE="${ACTION_PREDICTION_TYPE:-}"
STAGE1_ID="${RUN_ID}_stage1"
STAGE2_ID="${RUN_ID}_stage2"
STAGE2_CONFIG="${TMPDIR:-/tmp}/lilawam_${RUN_ID}_stage2.yaml"

if [[ -n "$ACTION_PREDICTION_TYPE" ]]; then
  case "$ACTION_PREDICTION_TYPE" in
    v_prediction|x_prediction|x_prediction_v_loss) ;;
    *) echo "Unsupported ACTION_PREDICTION_TYPE=$ACTION_PREDICTION_TYPE" >&2; exit 2 ;;
  esac
  CONFIG_FOR_STAGE1="${TMPDIR:-/tmp}/lilawam_${RUN_ID}_stage1.yaml"
  cp "$CONFIG" "$CONFIG_FOR_STAGE1"
  sed -i -E "s/^([[:space:]]*action_prediction_type:).*/\\1 \\\"$ACTION_PREDICTION_TYPE\\\"/" "$CONFIG_FOR_STAGE1"
else
  CONFIG_FOR_STAGE1="$CONFIG"
fi

echo "[LiLa-WAM] Stage 1: clean data, 15 epochs, lr=1e-4"
RUN_ID="$STAGE1_ID" torchrun --standalone --nproc_per_node="$GPUS" \
  train.py --config "$CONFIG_FOR_STAGE1" --save_dir "$SAVE_DIR"

STAGE1_DIR="$SAVE_DIR/sft_${STAGE1_ID}"
STAGE1_CKPT="$STAGE1_DIR/checkpoint_epoch_15.pt"
test -f "$STAGE1_CKPT"

cp "$STAGE1_DIR/config.yaml" "$STAGE2_CONFIG"
sed -i 's/learning_rate: 1e-4/learning_rate: 4e-5/; s/lr_min: 5.0e-5/lr_min: 1.0e-5/; s/epochs: 15/epochs: 5/' "$STAGE2_CONFIG"

echo "[LiLa-WAM] Stage 2: initialize from $STAGE1_CKPT, 5 epochs, lr=4e-5"
RUN_ID="$STAGE2_ID" torchrun --standalone --nproc_per_node="$GPUS" \
  train.py --config "$STAGE2_CONFIG" --save_dir "$SAVE_DIR" \
  --init_from "$STAGE1_CKPT"

echo "Training complete. Stage-2 checkpoints: $SAVE_DIR/sft_${STAGE2_ID}"
