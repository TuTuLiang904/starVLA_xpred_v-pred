# FastWAM

Official codebase for **Fast-WAM: Do World Action Models Need Test-time Future Imagination?**

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

This repository contains the training and evaluation code for FastWAM on LIBERO / RoboTwin.

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Inference with Released Checkpoints](#inference-with-released-checkpoints)
- [Training](#training)
- [Inference with Your Trained Checkpoints](#inference-with-your-trained-checkpoints)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

```text
FastWAM/
├── configs/
│   ├── data/                 # Dataset configs (LIBERO, RoboTwin, etc.)
│   ├── model/                # Model architecture and component configs
│   └── task/                 # Task-level configs (training task names)
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh        # Deepspeed zero1 training entrypoint
│   ├── preprocess_action_dit_backbone.py  # Preprocess ActionDiT backbone before training
│   └── precompute_text_embeds.py  # Precompute T5 text embedding cache before training
├── experiments/
│   ├── libero/
│   │   └── run_libero_manager.py
│   └── robotwin/
│       └── run_robotwin_manager.py
├── src/fastwam/              # Core code
├── runs/                     # Training outputs (ckpt, logs)
├── checkpoints/              # Pretrained or external checkpoints
├── data/                     # Data directory
└── evaluate_results/         # Inference / evaluation results
```

## Environment Setup

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Model Preparation

This step is required before both training and inference.

Step 1: set the Wan model directory first (opional, default `./checkpoints`):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Step 2: pre-generate the ActionDiT backbone (interpolated from Wan22 DiT):

```bash
# uncond (fastwam)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Dataset Download

### LIBERO

The preprocessed LIBERO dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

Download all compressed files first, then extract them all:

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# Run after downloading all 4 tar.gz files
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

The extracted directory structure should be:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

The preprocessed RoboTwin dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Download all split archive files first, then concatenate and extract:

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# Run after downloading all robotwin2.0.tar.gz.part-* files
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

The extracted directory structure should be:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

If you also keep:

```text
data/robotwin2.0/dataset_stats.json
```

in the root directory, it can be used directly as the statistics file for the current configs in this repo. You can also recompute it.

## Inference with Released Checkpoints

The released checkpoints and their corresponding dataset stats are available on [Hugging Face](https://huggingface.co/yuanty/fastwam).

Optional: download released checkpoints and dataset stats from Hugging Face:

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

After downloading, the local directory is expected to contain:

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

Before running the `LIBERO` benchmark, install the official LIBERO environment first
from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Then run this final step:

```bash
pip install mujoco==3.3.2
```

The `mujoco` environment should ideally stay consistent with the LIBERO data version.

We have already copied the `RoboTwin` evaluation-related code into `third_party/RoboTwin`.
You still need to follow the official RoboTwin instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin) to finish environment installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Optional: evaluate released LIBERO checkpoint:

The released `LIBERO` / `RoboTwin` evaluation managers default to `8` GPUs
(`MULTIRUN.num_gpus=8` in `configs/sim_libero.yaml` and `configs/sim_robotwin.yaml`).
If you want to evaluate with fewer GPUs, pass a smaller value such as
`MULTIRUN.num_gpus=4`.

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Optional: evaluate released RoboTwin checkpoint:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

For faster RoboTwin evaluation, we have enabled `EVALUATION.skip_get_obs_within_replan=true` in [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml).
This skips RGB rendering while consecutively executing an action chunk within one replan window, which speeds up evaluation but makes the saved video look very low-FPS.
Set it to `false` if you want to save a fully rendered video.

**Note:** We evaluate with **unseen** instructions, following Motus. [Lingbot-VA](https://github.com/Robbyant/lingbot-va/blob/661d52a59dc634a650efcd10a79d06bbb17ea81f/evaluation/robotwin/eval_polict_client_openpi.py#L308) uses **seen** instructions instead. You can try `EVALUATION.instruction_type=seen` to use **seen** instructions, which should theoretically improve performance by one or two points.

## Training

### 1) Precompute T5 embedding cache before training

Use `scripts/precompute_text_embeds.py` to precompute embeddings for each training task:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### 2) Training (using `fastwam` as an example)

When running a new task for the first time, set `pretrained_norm_stats` in the corresponding `configs/data/*.yaml` to `null` first.
After one training run, a `dataset_stats.json` file will be generated in the current run directory (for example, `runs/{task_name}/{run_id}/dataset_stats.json`).
You can then update `pretrained_norm_stats` to that file path for subsequent runs.

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4

# RoboTwin clean-action prediction (x-pred + action MSE)
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4_xpred

# RoboTwin clean-action prediction (x-pred + velocity-field MSE)
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4_xpred_vloss
```

For LIBERO, we train on a single node with 8 GPUs. For RoboTwin, we use 64 GPUs to accelerate training. You can try reducing the GPU count or training epochs.

The RoboTwin release has 921,032 unique language instructions. Its required
T5 cache therefore needs roughly 0.88 TiB for tensors alone (plan for about
1 TiB including per-file filesystem overhead). The 64-GPU setup is eight
8-GPU nodes; set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` when
launching `scripts/train_zero1.sh` on every node.

For an 8-GPU, reproducible 50%-episode experiment, first create the included
complete-episode subset (13,750 / 27,500 demonstrations). It reuses the
official full-dataset cache, so first run the original cache command, then
train for the original five epochs:

```bash
python scripts/create_robotwin_episode_subset.py
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=robotwin_uncond_3cam_384_1e-4
RUN_ID=robotwin_half_xpred \
  bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4_half_xpred \
  gradient_accumulation_steps=8
```

The cache is mandatory with the default training setup: text encoder loading is
disabled during training and every sample reads its precomputed
`context/context_mask`. Full-cache preparation is model-agnostic and can be
reused by v-pred, x-pred, x-pred/v-loss, and any episode subset. Use
`robotwin_uncond_3cam_384_1e-4_half_xpred_vloss` for x-pred/v-loss, or
`robotwin_uncond_3cam_384_1e-4_half` for original v-pred. If disk is limited,
you may generate only the half-subset cache by overriding both train/val cache
paths to a new empty directory; it needs about 0.6 TiB in practice.

RoboTwin evaluation reads `model.action_prediction_type` from the `config.yaml`
saved with a training checkpoint, so v-pred, x-pred, and x-pred/v-loss
checkpoints automatically use the matching inference update rule.

### Training monitoring and action metrics

Each run directory automatically contains:

- `train.log`: the terminal training logs, including epoch, step, loss terms, learning rate, throughput, and ETA;
- `metrics.jsonl`: line-delimited JSON scalar records for post-processing;
- `loss_curve.svg`: local training/validation loss curves without a matplotlib dependency;
- `eval/action_errors/step_*.svg` and `step_*.npz`: a rank-0 evaluation sample's absolute-action-error histogram plus raw prediction, target, and error arrays.

The LIBERO task runs an actual policy-sampling evaluation every `2000` steps by default. It records `action_l1`, `action_l2`, error percentiles (`p50/p90/p95`), `action_smoothness` (L2 first difference of denormalized commands), `action_jerk` (L2 second difference), `action_infer_latency_ms` (action-only policy), and `rollout_infer_latency_ms` (full video-and-action rollout). A lower smoothness value is not automatically better; interpret it with task success and action error.

The local-monitoring switches live under `monitoring` in `configs/train.yaml`; the evaluation cadence is `eval_every` in the task YAML. Set `eval_every: 999999` to disable periodic real sampling while retaining loss logs, or reduce it for denser policy metrics at the cost of evaluation time.

### Action head prediction modes

`model.action_prediction_type` controls only the action head; the video branch remains the original velocity prediction.

- `v_prediction` (default): original FastWAM behavior, predicts the flow velocity and uses the original time-weighted velocity MSE.
- `x_prediction`: predicts clean normalized actions and uses unweighted clean-action MSE.
- `x_prediction_v_loss`: predicts clean normalized actions, then converts the prediction to a flow velocity and uses velocity-space MSE. `model.action_x_pred_v_loss_eps` (default `0.05`) clamps the denominator near the clean-action endpoint.

For example:

```bash
# Clean-action x-pred with direct action MSE
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 \
  model.action_prediction_type=x_prediction

# Clean-action x-pred with AML-style velocity MSE
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 \
  model.action_prediction_type=x_prediction_v_loss \
  model.action_x_pred_v_loss_eps=0.05
```

The released checkpoints use `v_prediction`. Train a checkpoint for either x-pred mode before evaluating it with that mode; changing only the evaluation configuration changes the meaning of the action-head output.

LIBERO also provides two fixed-mode task YAMLs:

- `libero_uncond_2cam224_1e-4_xpred`: `x_prediction`
- `libero_uncond_2cam224_1e-4_xpred_vloss`: `x_prediction_v_loss`

You can also edit `model.action_prediction_type` in `configs/task/libero_uncond_2cam224_1e-4.yaml` directly.

## Inference with Your Trained Checkpoints

The `mujoco` environment should ideally stay consistent with the LIBERO data version. Then run LIBERO evaluation:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

### LIBERO-plus (FastWAM policy server)

LIBERO-plus runs the simulator in the `libero-plus` environment and FastWAM in
the `fastwam` environment. This repository reuses LIBERO-plus task sharding and
category aggregation, but does not use the StarVLA checkpoint/server format.
The launcher starts one FastWAM server and one simulator worker per GPU (8 GPUs
by default):

```bash
cd /mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/FastWAM

LIBERO_HOME=/mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred/LIBERO-plus \
FASTWAM_PYTHON=/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python \
LIBERO_PYTHON=/mnt/pfs/pg4hw0/conda_envs/libero-plus/bin/python \
CHECKPOINT="$PWD/runs/libero_uncond_2cam224_1e-4/libero_vpred/checkpoints/weights/step_021700.pt" \
DATASET_STATS="$PWD/runs/libero_uncond_2cam224_1e-4/libero_vpred/dataset_stats.json" \
OUTPUT_DIR="$PWD/evaluate_results/libero_plus/libero_vpred" \
bash experiments/libero_plus/run_local_8gpu_eval.sh
```

Run a first end-to-end smoke test with `NUM_TRIALS=1`. The FastWAM server
automatically uses `dataset_stats.json`, receives both camera views and the
8-D proprio state, and preserves FastWAM's 32-action horizon, 10-step replanning,
and gripper semantics.

The launcher defaults to `NUM_TRIALS=1`; use `NUM_TRIALS=50` for the full
per-task rollout count. Each worker writes progress, cumulative success rate,
elapsed time, and ETA to `OUTPUT_DIR/workers/*.log`. After all shards finish,
the category table is in `OUTPUT_DIR/shards/summary.txt` and raw counts are in
`OUTPUT_DIR/shards/overall_results.json`.

The policy server reads `model.action_prediction_type` from the training
`config.yaml` adjacent to the checkpoint. This keeps v-pred, x-pred, and
x-pred/v-loss checkpoints on their respective inference paths automatically.

We have already copied the `RoboTwin` evaluation-related code into `third_party/RoboTwin`.
You still need to follow the official RoboTwin instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).
Finish installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Then run RoboTwin evaluation:

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```

Common `task_name` examples:

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## Acknowledgements

The RoboTwin evaluation code in this repository is adapted from the official [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin). We thank the RoboTwin team for releasing their codebase and assets.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
