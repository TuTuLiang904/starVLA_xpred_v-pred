# 三套模型闭环 NFE sweep

这里的 NFE 指一次 action chunk 从纯噪声积分到动作 endpoint 时，action model 被调用的次数。脚本只覆盖推理，不修改 checkpoint 和训练配置。

## 已核实的原版步数

| 模型 | 原版步数 | 依据 |
|---|---:|---|
| StarVLA | 4 | RobotWin checkpoint `config.yaml` 中 `framework.action_model.num_inference_timesteps: 4`；默认 action head 也是 4 |
| FastWAM | 10 | `FastWAM/configs/train.yaml` 的 `eval_num_inference_steps: 10`，RobotWin websocket script 默认 `NUM_INFERENCE_STEPS=10` |
| LiLaWAM | 10 | `LiLa-WAM/configs/robotwin_all.yaml` 的 `common.num_inference_steps: 10` |

因此 `1,2,4,8,16` 中，StarVLA 的原版点是 4，FastWAM/LiLaWAM 的原版点是 10（不在这组 sweep 网格中）。如果需要严格包含原版，还应额外跑 NFE=10。

## StarVLA

脚本：[`run_starvla_robotwin_nfe_sweep.sh`](../scripts/closed_loop_nfe/run_starvla_robotwin_nfe_sweep.sh)

它通过 StarVLA server 已支持的 OmegaConf dotlist 覆盖：

```text
framework.action_model.num_inference_timesteps=N
```

原版 `run_policy_server.sh` 增加了一个可选环境变量 `STARVLA_CONFIG_OVERRIDE`。不设置该变量时，原有命令行为不变。

Smoke test（只跑一个 task，先验证服务器和 NFE 覆盖）：

```bash
cd /mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/xpred_vpred_toy
export CHECKPOINT="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_v_prediction_seed42/final_model/pytorch_model.pt"
export NFE_VALUES="1 2"
export TASKS="click_alarmclock"
export GPU_IDS=0
export NUM_EPISODES=1
export ROBOTWIN_JOBS_PER_GPU=1
scripts/closed_loop_nfe/run_starvla_robotwin_nfe_sweep.sh
```

正式运行默认会跑 `all` task；也可以用 `TASKS="click_alarmclock open_laptop"` 限定任务：

```bash
scripts/closed_loop_nfe/run_starvla_robotwin_nfe_sweep.sh
```

StarVLA 默认使用 `GPU_IDS=0,1,2,3,4,5,6,7`，并把它传给原版 `start_eval.sh` 的 `CUDA_VISIBLE_DEVICES`。如果终端原先有 `CUDA_VISIBLE_DEVICES=4`，不会再意外退化成单卡；需要缩小规模时显式设置 `GPU_IDS=4` 或 `GPU_IDS=0,1`。

每个 NFE 的日志在：

```text
xpred_vpred_toy/results/closed_loop_nfe/starvla_robotwin/nfe_N/
```

StarVLA wrapper 将 `TASKS` 原样传给原版 `start_eval.sh`，因此可以使用单个 task、多个 task 或 `all`。`NUM_EPISODES` 默认是 50，smoke 时应设为 1。

## FastWAM

脚本：[`run_fastwam_robotwin_nfe_sweep.sh`](../scripts/closed_loop_nfe/run_fastwam_robotwin_nfe_sweep.sh)

FastWAM 原版 RobotWin websocket evaluator 已经有：

```bash
NUM_INFERENCE_STEPS=10 bash experiments/robotwin/run_robotwin_websocket_all.sh
```

因此 wrapper 不改 FastWAM 代码，只循环设置 `NUM_INFERENCE_STEPS`。

```bash
cd /mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/xpred_vpred_toy
export CHECKPOINT="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/checkpoints/weights/step_021295.pt"
export DATASET_STATS="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/dataset_stats.json"
export NUM_GPUS=8 NUM_EPISODES=50 EVAL_MODE=demo_clean
scripts/closed_loop_nfe/run_fastwam_robotwin_nfe_sweep.sh
```

结果在：

```text
xpred_vpred_toy/results/closed_loop_nfe/fastwam_robotwin/nfe_N/
```

严格对照原版时，建议另外运行：

```bash
NFE_VALUES="10" scripts/closed_loop_nfe/run_fastwam_robotwin_nfe_sweep.sh
```

## LiLaWAM

脚本：[`run_lilawam_robotwin_nfe_sweep.sh`](../scripts/closed_loop_nfe/run_lilawam_robotwin_nfe_sweep.sh)

LiLaWAM 原评估脚本没有暴露 NFE 环境变量，所以只增加了两个 inference-only hook：

- `LILAWAM_NUM_INFERENCE_STEPS`：覆盖 `RobotWinInference.num_inference_steps`；
- `LILAWAM_SAVE_ROOT`：为不同 NFE 固定输出目录。

未设置时，原配置仍然是 10 步。

```bash
cd /mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/xpred_vpred_toy
export CKPT_SETTING=sft_full_vpred_stage2
export CHECKPOINT_EP=4
export GPUS=0,1,2,3,4,5,6,7
export TEST_NUM=50 SEED=42 TASK_CONFIG=demo_clean
scripts/closed_loop_nfe/run_lilawam_robotwin_nfe_sweep.sh
```

结果在：

```text
xpred_vpred_toy/results/closed_loop_nfe/lilawam_robotwin/nfe_N/
```

## 运行建议

完整 sweep 的计算量很大。建议顺序：

1. 每个模型先 `NFE_VALUES="1 4"`、单 task 或 `TEST_NUM=1` 做 smoke test；
2. 检查日志中实际打印的 NFE、checkpoint、GPU 和任务数；
3. 再跑 `1 2 4 8 16`；
4. 最后补跑原版点：StarVLA 的 4，FastWAM/LiLaWAM 的 10；
5. 固定 `replan_steps`、seed、task_config 和 episode 数，不能在不同 NFE 间改变这些条件。

注意：这是真实闭环仿真 NFE，不是之前 `real_mechanism` 的离线 replay NFE。最终应按每个 NFE 报告：overall success、每 task success、失败类型、episode length，并与离线 MSE/jerk 曲线分开分析。
