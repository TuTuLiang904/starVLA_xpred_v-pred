#!/usr/bin/env bash
set -euo pipefail
VLA_ROOT="$(cd "$(dirname "$0")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/pfs/pg4hw0/mobile/qiwei/mobile/FastWAM/third_party/RoboTwin}"
PYTHON="${PYTHON:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/bin/python}"
RT_SITE="${RT_SITE:-/mnt/pfs/pg4hw0/conda_envs/RoboTwin/lib/python3.10/site-packages}"
OVERLAY="${OVERLAY:-/tmp/lila_overlay}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
CKPT_SETTING="${CKPT_SETTING:?Set CKPT_SETTING}"
CHECKPOINT_EP="${CHECKPOINT_EP:?Set CHECKPOINT_EP (Stage-2 final is 4)}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"; TEST_NUM="${TEST_NUM:-50}"; SEED="${SEED:-0}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
TOTAL_WORKERS=$(( ${#GPU_LIST[@]} * PROCS_PER_GPU ))
TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch)
test -f "$ROBOTWIN_ROOT/eval_vla_bridge.py"
test -f "$VLA_ROOT/checkpoints_vla/$CKPT_SETTING/checkpoint_epoch_${CHECKPOINT_EP}.pt"
SAVE_ROOT="${LILAWAM_SAVE_ROOT:-$VLA_ROOT/eval_result/$CKPT_SETTING/${TASK_CONFIG}_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$SAVE_ROOT/logs" "$SAVE_ROOT/bridge_results"
echo "${#TASKS[@]} tasks, $TOTAL_WORKERS workers ($PROCS_PER_GPU per GPU), checkpoint=$CKPT_SETTING epoch=$CHECKPOINT_EP"
pids=()
for ((w=0; w<TOTAL_WORKERS; w++)); do
  gpu="${GPU_LIST[$((w % ${#GPU_LIST[@]}))]}"; chunk=()
  for ((i=w; i<${#TASKS[@]}; i+=TOTAL_WORKERS)); do chunk+=("${TASKS[$i]}"); done
  ((${#chunk[@]})) || continue
  (
    cd "$ROBOTWIN_ROOT"
    PYTHONPATH="$OVERLAY:$RT_SITE:$ROBOTWIN_ROOT" CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROBOTWIN_ROOT/eval_vla_bridge.py" --task_names "${chunk[@]}" --task_config "$TASK_CONFIG" --instruction_type unseen --ckpt_setting "$CKPT_SETTING" --checkpoint_ep "$CHECKPOINT_EP" --model_base_path "$VLA_ROOT" --norm_stats_path utils/stat-500-all.json --config_path configs/robotwin_all.yaml --vla_root "$VLA_ROOT" --save_root "$SAVE_ROOT" --result_json "$SAVE_ROOT/bridge_results/worker_${w}.json" --seed "$((SEED+w))" --test_num "$TEST_NUM" 2>&1 | tee "$SAVE_ROOT/logs/worker_${w}.log"
  ) & pids+=("$!")
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "Evaluation workers finished. Results: $SAVE_ROOT"
exit "$fail"
