# 第二阶段机制实验：中文运行与解读手册

这里的代码独立于 `starVLA`、`FastWAM`、`LiLa-WAM` 的训练和仿真评测目录：**不会修改**训练配置、scheduler、checkpoint 或原有 eval 入口。它回答的不是“哪一个成功率更高”（这是第一阶段已经有的结论），而是：

> 为什么在 VLA 中 x-pred 往往优于 v-pred，而在 WAM 中 v-pred 往往更好？

本阶段的证据链分三步：先证明数据中 clean action 的确更低维、更平滑；再直接测模型从不同噪声强度恢复 clean action 的能力；最后测不同去噪步数下的离线轨迹误差。三条证据连起来，才可以讨论机制，而不是只报告现象。

## 先看总流程

| 线 | 输入 | 输出 | 要回答的问题 |
|---|---|---|---|
| A. 数据几何 | RobotWin 示范动作，不加载模型 | `global_task.csv`、`local_anchor.csv` | clean action `x` 是否真的低维、局部平滑？`ε` 和 `v` 是否更复杂？ |
| B. endpoint recovery | 已训练 checkpoint；固定真实动作和噪声 | `*_recovery.npz/csv` | 在给定噪声强度 `σ` 时，x/v/x-vloss 谁一步恢复终点更准？ |
| C. offline NFE | 已训练 checkpoint；从高斯噪声采样 | `*_nfe.npz/csv` | NFE=1/2/4/8/16 时，哪种预测方式更快收敛？更多步数是否反转优势？ |

建议顺序：**先 A，再 B，最后 C**。A 完全不需要 GPU；B 最直接验证“高噪声恢复”假设；C 才把它连接到实际的少步/多步推理现象。

## 符号与直觉

- `x`：数据集中的干净动作 chunk。例如 RobotWin 是未来 32 或 50 步、每步 14 维的动作。
- `ε`：与 `x` 同形状的标准高斯噪声。
- `v`：速度场目标。此处在统一比较坐标中可理解为 `x - ε`。
- `σ`：噪声比例。统一的加噪动作是 `zσ=(1-σ)x+σε`。
  - `σ≈0`：输入接近干净动作，恢复很容易。
  - `σ≈1`：输入接近纯噪声，恢复最困难，也是少步推理最关心的区域。
- NFE（number of function evaluations）：去噪/ODE 更新次数。NFE=1 最便宜也最困难；NFE=16 更接近充分迭代。

注意：StarVLA 使用原生 h50；FastWAM 和 LiLa-WAM 使用原生 h32。**绝不能把 h50 截成 h32，或给 h32 补零到 h50**，否则会制造假的低维结构和假的模型差异。

## A：数据几何指标到底代表什么？

每个动作 chunk 先展平成一个向量，例如 h32×14=448 维。几何脚本在每个物理任务内分别比较 `x`、`ε`、`v`；它不使用任何模型自己的 normalization，因此是 training-free 的公平诊断。

| CSV 列 | 它测量什么 | 值小/值大代表什么 | 在本文中的用途 |
|---|---|---|---|
| `stable_rank` | 奇异值能量分散到多少有效方向 | 小：主要变化集中在少数方向；大：能量分散、较高维 | clean `x` 应显著小于 `ε`；`v` 通常介于两者或更接近高维端 |
| `participation_ratio` | 另一种“有效维数” | 同上，且比单个阈值更连续 | 用来确认 stable rank 的结论不是单一指标偶然造成 |
| `rank90/95/99` | 保留 90/95/99% 方差所需的主成分数 | 小：少数方向足以还原动作；大：需要很多方向 | 最直观地报告“99% 能量需要几维” |
| `two_nn_id` | 近邻距离比得到的 intrinsic dimension 估计 | 小：样本附近近似位于低维流形；大：局部空间更高维 | 不是 PCA 的重复证据，而是局部几何证据 |
| `linear_residual` | 用局部切平面重建邻居后的残差 | 小：邻域接近线性切平面，局部平坦；大：局部非线性/噪声强 | clean `x` 若小，说明“低维”不仅是全局 PCA 现象 |
| `tangent_variation` | 相邻局部切平面之间的主角度差异 | 小：沿轨迹移动时切空间稳定、流形平滑；大：弯曲快或方向不稳定 | 支持 clean action 是平滑流形，而不是一团随机点 |

应报告的模式是：在**同一 benchmark、同一 horizon、同一任务内**，`x` 的有效维数、局部残差、切空间变化均低于 `ε`，并通常低于 `v`。这支持“预测 clean action 与低维、平滑的目标对齐”。

不能这样解释：

- 不能跨 RobotWin (14-D) 和 LIBERO (7-D) 直接比较绝对 rank。
- 不能仅凭 `ε` 高维就声称 x-pred 必然胜出；这是数据前提，模型机制仍需 B/C 验证。
- `v` 是 `x-ε`，它高维是预期现象；关键是它是否与 checkpoint 的恢复/NFE 劣势相对应。

当前 medium sanity run 已出现合理趋势：多数任务内 `x` stable rank 约 1–2，`ε` 约 7，`v` 介于两者。这只是程序 sanity check，不应作为论文数字；论文使用下方 50 task × 40 anchor 的正式输出。脚本默认局部近邻数 `k=16`，因为正式配置每任务只有 40 个 anchor；如自行减少 anchor 数，应保证 `k < 每任务 anchor 数`。

## B：endpoint recovery 测什么？

对于每个真实目标动作 `x`、每个固定噪声种子和每个 `σ`，构造同一个 `zσ`。然后让每种 checkpoint **只调用一次自己的 action head**，再转换到它预测的 clean endpoint `x_hat`。最终记录：

`endpoint_mse = mean((x_hat - x)^2)`。

| 输出列 | 含义 | 如何解读 |
|---|---|---|
| `sigma` | 输入的噪声强度 | 横轴；越接近 1 越困难 |
| `endpoint_mse` / `endpoint_rmse` | 一步终点恢复误差 | 越小越好；重点看 `σ=0.8/0.95/0.99` |
| `task_sem` | 各 anchor 误差均值的标准误 | 越小表示估计越稳定；正式图应画 error bar |
| `endpoint_seed_variance` | 同一 anchor 在不同初始噪声下输出差异 | 小表示对噪声实现更稳定；它不是准确率，需与 MSE 一起看 |

这条线直接检验你的假设：**若 x-pred 在高 `σ` 的 endpoint MSE 显著低于 v-pred，说明从“几乎纯噪声”直接找回 clean action 时，x-pred 更有利。** 这是预期解释 VLA 少步优势的最直接证据。

反过来，若 WAM 的 v-pred 在高 `σ` 更低，或者它的 `action_only` 与 `joint` 曲线差异很大，则说明 WAM 的优势不能仅由 clean-action 低维解释，而可能来自 v 目标与视频/动作联合动力学的对齐。

## C：offline NFE curve 测什么？

该线从固定高斯初始噪声开始，按每个模型自己的 sampling rule 运行 NFE=1/2/4/8/16 次，得到动作 chunk `a_hat`，与示范 chunk `a` 对比。它是**离线 replay 诊断**，不是机器人仿真成功率。

| 输出列 | 含义 | 如何解读 |
|---|---|---|
| `nfe` | 去噪步数 | 横轴；成本随它上升 |
| `replay_mse` | 采样动作与演示动作的 MSE | 越小越好；看不同预测类型曲线是否交叉 |
| `task_sem` | replay MSE 的 anchor 标准误 | 正式图的误差条 |
| `seed_variance` | 同一观测、不同初始噪声输出的差异 | 反映采样稳定性；不是替代 MSE |
| `action_delta_sq` | 相邻动作时刻的平方变化量 | 过大可能意味着动作过跳；太小也可能是不动。必须结合 replay MSE |
| `jerk_sq` | 二阶差分平方，即动作变化的变化 | 过大表示不平滑/抖动；不能单独当作性能指标 |

你最关心的图是 `replay_mse` 对 `NFE`：

1. 若 VLA 的 x-pred 在 NFE=1/2 更好，并在更高 NFE 仍好或逐步接近 v-pred，支持“x-pred 对有限步的终点恢复更友好”。
2. 若在 NFE=16 曲线交叉或反弹，不能直接叫“x-pred 失效”；应和 B 的高 `σ` recovery 曲线一起看，判断是多步累计误差、solver 匹配还是模型本身差异。
3. 若 WAM 中 v-pred 低 NFE 就更好，且 FastWAM 的 `joint` 与 `action_only` 差异大，优先检验“视频—动作联合流使 velocity 目标受益”的机制。

最终仍须在 RobotWin clean/random 的 NFE=1/4/16 仿真成功率上验证：离线 replay 是机制诊断，不能代替泛化或闭环控制结论。

## 输出文件与维度

- recovery archive：`endpoint [M,S,R,N,T,D]`
- NFE archive：`sample [M,K,R,N,T,D]`

其中：`M`=预测类型/模型数，`S`=9 个 `σ`，`K`=5 个 NFE，`R`=配对噪声种子数，`N`=anchor 数，`T`=horizon，`D`=动作维数。三个预测类型必须用相同 manifest 和相同 seed bank，才可以作逐点公平比较。

## RobotWin-clean：正式运行顺序

以下都从 `xpred_vpred_toy` 执行。

```bash
export ROOT=/mnt/pfs/pg4hw0/mobile/qiwei/mobile/starVLA_xpred_v-pred
export TOY="$ROOT/xpred_vpred_toy"
cd "$TOY"
export PYTHONPATH="$PWD"

# A. 训练无关几何：50 个物理任务 × 每任务 40 anchor，h32
python real_mechanism/scripts/build_robotwin_manifest.py \
  --horizon 32 --anchors-per-task 40 --cache-actions \
  --output real_mechanism/outputs/robotwin_hdf5_h32.csv
python real_mechanism/scripts/run_geometry.py \
  --manifest real_mechanism/outputs/robotwin_hdf5_h32.csv \
  --output-dir real_mechanism/outputs/geometry_robotwin_h32

# B/C. StarVLA/FastWAM 必须用原生三视角 LeRobot；分别建 h32 和 h50。
python real_mechanism/scripts/build_lerobot_manifest.py \
  --horizon 32 --anchors-per-task 40 --cache-actions \
  --output real_mechanism/outputs/lerobot_robotwin_h32.csv
python real_mechanism/scripts/build_lerobot_manifest.py \
  --horizon 50 --anchors-per-task 40 --cache-actions \
  --output real_mechanism/outputs/lerobot_robotwin_h50.csv
```

首次请给每个 checkpoint 命令附上 `--max-samples 8 --noise-seeds 2`；确认生成 `.npz` 后再移除这两个参数，得到正式的 2,000 anchor × 8 seeds 结果。FastWAM 全量会依次加载三个 12GB checkpoint，故意串行以控制 A800 显存。

```bash
# StarVLA：h50；先 recovery，再 NFE
export CUDA_VISIBLE_DEVICES=0
/mnt/pfs/pg4hw0/conda_envs/starVLA/bin/python real_mechanism/scripts/run_starvla_checkpoint_probe.py --kind recovery \
  --manifest real_mechanism/outputs/lerobot_robotwin_h50.csv \
  --checkpoint vpred="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_v_prediction_seed42/final_model/pytorch_model.pt" \
  --checkpoint xpred="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_x_prediction_seed42/final_model/pytorch_model.pt" \
  --checkpoint xpred_vloss="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_x_prediction_v_loss_seed42/final_model/pytorch_model.pt" \
  --output real_mechanism/outputs/starvla_robotwin_recovery.npz
/mnt/pfs/pg4hw0/conda_envs/starVLA/bin/python real_mechanism/scripts/run_starvla_checkpoint_probe.py --kind nfe \
  --manifest real_mechanism/outputs/lerobot_robotwin_h50.csv \
  --checkpoint vpred="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_v_prediction_seed42/final_model/pytorch_model.pt" \
  --checkpoint xpred="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_x_prediction_seed42/final_model/pytorch_model.pt" \
  --checkpoint xpred_vloss="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_x_prediction_v_loss_seed42/final_model/pytorch_model.pt" \
  --output real_mechanism/outputs/starvla_robotwin_nfe.npz

# LiLa-WAM：h32。其训练环境缺 matplotlib，而官方 robotwin_infer.py 强制导入它，故使用此可运行环境。
export CUDA_VISIBLE_DEVICES=1
/opt/conda/bin/python real_mechanism/scripts/run_lilawam_checkpoint_probe.py --kind recovery \
  --manifest real_mechanism/outputs/robotwin_hdf5_h32.csv \
  --checkpoint vpred="$ROOT/LiLa-WAM/checkpoints_vla/sft_clean_vpred_stage2/checkpoint_epoch_5.pt" \
  --checkpoint xpred="$ROOT/LiLa-WAM/checkpoints_vla/sft_clean_xpred_stage2/checkpoint_epoch_5.pt" \
  --checkpoint xpred_vloss="$ROOT/LiLa-WAM/checkpoints_vla/sft_clean_xpred_vloss_stage2/checkpoint_epoch_5.pt" \
  --output real_mechanism/outputs/lilawam_robotwin_recovery.npz
/opt/conda/bin/python real_mechanism/scripts/run_lilawam_checkpoint_probe.py --kind nfe \
  --manifest real_mechanism/outputs/robotwin_hdf5_h32.csv \
  --checkpoint vpred="$ROOT/LiLa-WAM/checkpoints_vla/sft_clean_vpred_stage2/checkpoint_epoch_5.pt" \
  --checkpoint xpred="$ROOT/LiLa-WAM/checkpoints_vla/sft_clean_xpred_stage2/checkpoint_epoch_5.pt" \
  --checkpoint xpred_vloss="$ROOT/LiLa-WAM/checkpoints_vla/sft_clean_xpred_vloss_stage2/checkpoint_epoch_5.pt" \
  --output real_mechanism/outputs/lilawam_robotwin_nfe.npz

# FastWAM：h32。三个 clean50 run 的 dataset_stats.json 已核验相同。
export CUDA_VISIBLE_DEVICES=2
/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python real_mechanism/scripts/run_fastwam_nfe_probe.py --kind recovery --mode action_only \
  --manifest real_mechanism/outputs/lerobot_robotwin_h32.csv \
  --checkpoint vpred="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/checkpoints/weights/step_021295.pt" \
  --checkpoint xpred="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred/robotwin_clean50_xpred/checkpoints/weights/step_021295.pt" \
  --checkpoint xpred_vloss="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred_vloss/robotwin_clean50_xpred_vloss/checkpoints/weights/step_021295.pt" \
  --stats "$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/dataset_stats.json" \
  --output real_mechanism/outputs/fastwam_robotwin_recovery.npz

# FastWAM 的两种 NFE 机制：action_only 是部署时静态视频 cache；joint 是视频和动作共同演化。
/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python real_mechanism/scripts/run_fastwam_nfe_probe.py --kind nfe --mode action_only \
  --manifest real_mechanism/outputs/lerobot_robotwin_h32.csv \
  --checkpoint vpred="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/checkpoints/weights/step_021295.pt" \
  --checkpoint xpred="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred/robotwin_clean50_xpred/checkpoints/weights/step_021295.pt" \
  --checkpoint xpred_vloss="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred_vloss/robotwin_clean50_xpred_vloss/checkpoints/weights/step_021295.pt" \
  --stats "$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/dataset_stats.json" \
  --output real_mechanism/outputs/fastwam_robotwin_nfe_action_only.npz
/mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python real_mechanism/scripts/run_fastwam_nfe_probe.py --kind nfe --mode joint \
  --manifest real_mechanism/outputs/lerobot_robotwin_h32.csv \
  --checkpoint vpred="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/checkpoints/weights/step_021295.pt" \
  --checkpoint xpred="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred/robotwin_clean50_xpred/checkpoints/weights/step_021295.pt" \
  --checkpoint xpred_vloss="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred_vloss/robotwin_clean50_xpred_vloss/checkpoints/weights/step_021295.pt" \
  --stats "$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/dataset_stats.json" \
  --output real_mechanism/outputs/fastwam_robotwin_nfe_joint.npz
```

将每个 `.npz` 转为易读 CSV：

```bash
python real_mechanism/scripts/run_probe_stats.py --kind recovery \
  --archive RESULTS_RECOVERY.npz --output recovery_summary.csv
python real_mechanism/scripts/run_probe_stats.py --kind nfe \
  --archive RESULTS_NFE.npz --output nfe_summary.csv
```

## 论文图和结论的最低标准

1. 图 A：每个任务或任务平均的 `x/ε/v` effective dimension 与 `tangent_variation`；不要只画一个任务。
2. 图 B：endpoint MSE–`σ` 曲线，x/v/x-vloss 共用 seeds、并画 task SEM。重点标出高噪声端。
3. 图 C：replay MSE–NFE 曲线，并报告平滑性指标，但不要用平滑性替代准确性。
4. 图 D：RobotWin clean 与 random 上 NFE=1/4/16 的闭环成功率，用于确认 B/C 的离线机制在真实控制中成立。
5. 对 FastWAM，`action_only` vs `joint` 是额外消融；它能区分“纯 action 参数化效应”和“联合视频—动作流的效应”。

## StarVLA 加速：三张 GPU 并行

不要把三个 StarVLA checkpoint 放在同一个进程中配合 `CUDA_VISIBLE_DEVICES=0` 跑；那会让三个模型串行使用 GPU0。推荐一个进程对应一个 checkpoint，三张卡并行。先停止旧的单卡命令（终端按 `Ctrl-C`），再执行：

为避免长路径被复制成带换行的字符串，建议先把路径保存为变量，再写
`--checkpoint "vpred=$V_CKPT"`；不要在 checkpoint 路径的中间手动换行。

```bash
cd "$TOY"
export PYTHONPATH="$PWD"

CUDA_VISIBLE_DEVICES=0 /mnt/pfs/pg4hw0/conda_envs/starVLA/bin/python real_mechanism/scripts/run_starvla_checkpoint_probe.py --kind recovery \
  --manifest real_mechanism/outputs/lerobot_robotwin_h50.csv \
  --checkpoint vpred="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_v_prediction_seed42/final_model/pytorch_model.pt" \
  --output real_mechanism/outputs/starvla_robotwin_recovery.vpred.npz \
  --max-samples 8 --noise-seeds 2 > /tmp/starvla_recovery_vpred.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 /mnt/pfs/pg4hw0/conda_envs/starVLA/bin/python real_mechanism/scripts/run_starvla_checkpoint_probe.py --kind recovery \
  --manifest real_mechanism/outputs/lerobot_robotwin_h50.csv \
  --checkpoint xpred="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_x_prediction_seed42/final_model/pytorch_model.pt" \
  --output real_mechanism/outputs/starvla_robotwin_recovery.xpred.npz \
  --max-samples 8 --noise-seeds 2 > /tmp/starvla_recovery_xpred.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 /mnt/pfs/pg4hw0/conda_envs/starVLA/bin/python real_mechanism/scripts/run_starvla_checkpoint_probe.py --kind recovery \
  --manifest real_mechanism/outputs/lerobot_robotwin_h50.csv \
  --checkpoint xpred_vloss="$ROOT/starVLA/playground/Checkpoints/qwen_gr00t_robotwin_x_prediction_v_loss_seed42/final_model/pytorch_model.pt" \
  --output real_mechanism/outputs/starvla_robotwin_recovery.xpred_vloss.npz \
  --max-samples 8 --noise-seeds 2 > /tmp/starvla_recovery_xvloss.log 2>&1 &
wait
```

先观察日志：

```bash
tail -f /tmp/starvla_recovery_vpred.log
```

脚本现在每 10 个 anchor 打印一次进度，并在每个 checkpoint 完成后写一个中间 archive。三卡任务完成后可以合并为统一 archive：

```bash
python real_mechanism/scripts/merge_probe_archives.py --kind recovery \
  --archive vpred=real_mechanism/outputs/starvla_robotwin_recovery.vpred.npz \
  --archive xpred=real_mechanism/outputs/starvla_robotwin_recovery.xpred.npz \
  --archive xpred_vloss=real_mechanism/outputs/starvla_robotwin_recovery.xpred_vloss.npz \
  --output real_mechanism/outputs/starvla_robotwin_recovery.npz
python real_mechanism/scripts/run_probe_stats.py --kind recovery \
  --archive real_mechanism/outputs/starvla_robotwin_recovery.npz \
  --output real_mechanism/outputs/starvla_robotwin_recovery.csv
```

上面的 `8 anchors × 2 seeds` 只是快速验证。确认日志和 archive 正常后，将三个命令中的 `--max-samples 8 --noise-seeds 2` 删除，重新跑正式结果。NFE 只需把三个命令中的 `--kind recovery` 改成 `--kind nfe`，并修改输出文件名。

## FastWAM 加速：不要把三个 checkpoint 放进一个进程

FastWAM 的完整 checkpoint 很大，而且 `joint` 每个 NFE 都比
`action_only` 慢。旧命令把三个 checkpoint 串行放在一个进程里，因此几个
小时没有最终文件并不代表死锁，而是还没有完成第一个完整输出。建议先终止旧
进程，再一个 checkpoint 一个 GPU 地运行。`joint` 默认不解码最终视频，因为
本实验只保存 action；如果确实需要视频输出，再加 `--decode-video`。

```bash
# 先在运行 FastWAM 的终端按 Ctrl-C；若是后台进程，使用 ps 找到对应 PID 后：
# kill <旧的 action_only PID> <旧的 joint PID>

V_CKPT="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/checkpoints/weights/step_021295.pt"
X_CKPT="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred/robotwin_clean50_xpred/checkpoints/weights/step_021295.pt"
XL_CKPT="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50_xpred_vloss/robotwin_clean50_xpred_vloss/checkpoints/weights/step_021295.pt"
STATS="$ROOT/FastWAM/runs/robotwin_uncond_3cam_384_1e-4_clean50/robotwin_clean50_vpred/dataset_stats.json"

# 先做 action-only；GPU 0/1/2 分别对应 v/x/x-vloss
CUDA_VISIBLE_DEVICES=0 /mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python real_mechanism/scripts/run_fastwam_nfe_probe.py --kind nfe --mode action_only --manifest real_mechanism/outputs/lerobot_robotwin_h32.csv --checkpoint "vpred=$V_CKPT" --stats "$STATS" --output real_mechanism/outputs/fastwam_nfe_action_only.vpred.npz --max-samples 8 --noise-seeds 2 > /tmp/fastwam_action_vpred.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 /mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python real_mechanism/scripts/run_fastwam_nfe_probe.py --kind nfe --mode action_only --manifest real_mechanism/outputs/lerobot_robotwin_h32.csv --checkpoint "xpred=$X_CKPT" --stats "$STATS" --output real_mechanism/outputs/fastwam_nfe_action_only.xpred.npz --max-samples 8 --noise-seeds 2 > /tmp/fastwam_action_xpred.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 /mnt/pfs/pg4hw0/conda_envs/fastwam/bin/python real_mechanism/scripts/run_fastwam_nfe_probe.py --kind nfe --mode action_only --manifest real_mechanism/outputs/lerobot_robotwin_h32.csv --checkpoint "xpred_vloss=$XL_CKPT" --stats "$STATS" --output real_mechanism/outputs/fastwam_nfe_action_only.xpred_vloss.npz --max-samples 8 --noise-seeds 2 > /tmp/fastwam_action_xvloss.log 2>&1 &
wait

# joint 模式同理，换三个 GPU；不要加 --decode-video。
```

日志应每 10 个 anchor 显示一次进度；每个 checkpoint 结束时会先写入一个
同名的中间 archive。确认 smoke 通过后删除 `--max-samples 8 --noise-seeds 2`，
再跑正式 2,000 anchor × 8 seed。三个单 checkpoint archive 可用
`merge_probe_archives.py --kind nfe` 沿 model 维合并。

`SyntheticBackend` 和 `smoke_backend.py` 只验证 archive 格式，永远不能用于论文结果。单 anchor 冒烟输出也仅证明脚本正确，不能用于任何模型优劣结论。
