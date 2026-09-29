# x-pred vs v-pred toy

独立、可复现的教学仓库：在**双流 arm–base 动作专家**里，比较 rectified flow 的两种输出参数化——预测干净动作 `x`，还是预测速度 `v`。

三种参数化在无限容量下代数等价；本仓库要回答的是：**有限宽度、无 skip connection 的双流专家，会把表征容量花在噪声上，还是花在可执行动作和跨流信息上。**

```
低维干净动作
    → x-pred 学到更简单的输入–输出映射
        → 高噪声处更稳（少步采样从这里起步）
            → 少 NFE 时更接近可执行流形
```

仓库自包含：合成数据、闭式 Bayes 去噪器、配对训练、探针、生成指标、画图脚本。不依赖真实机器人数据，也不依赖完整 VLA 训练栈。

---

## 最快上手（约 5–15 分钟，一块 GPU）

```bash
cd xpred_vpred_toy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python examples/inspect_manifold.py
python scripts/demo.py
```

`demo.py` 默认是 **smoke 预算**（宽 128、2 个 block、约 600 step）。它会：

1. 不训练，先打印干净 `x`、速度 `v`、噪声 `ε` 的有效秩；
2. 用**同一个 seed** 训练 `x-pred` 与 `v-pred`（mode 给定）；
3. 打印 Jacobian 秩、噪声/partner 探针、1/5 步 off-manifold。

论文趋势请用更长预算：

```bash
python scripts/demo.py --full --steps 4000
```

smoke 预算下部分指标可能反转，这是正常的，不要据此写结论。

---

## 仓库结构

```
velatoy/                 核心库（数据、流、模型、训练、探针）
scripts/
  demo.py                教学入口：谱 + 一对训练
  train_one.py           训一个 arm，写一条 JSON
  run_study.py           完整实验矩阵（多 GPU）
  report.py              从 JSON 汇总配对统计
  export_figure_data.py  把图画用的序列导出 CSV
figures/                 论文图（含 headline 四联图）
examples/                只看数据、不训练
docs/                    教学文档（先读这个）
figures/reference/       一份已画好的 headline 图，便于对照
```

建议阅读顺序：

| 文档 | 内容 |
|---|---|
| [docs/01_idea.md](docs/01_idea.md) | 问题、claim、为什么代数等价还要比较 |
| [docs/02_data.md](docs/02_data.md) | 合成流形：输入张量、mode、validity mask |
| [docs/03_model_and_loss.md](docs/03_model_and_loss.md) | 网络、无 skip、三种 head、同一个 loss |
| [docs/04_metrics.md](docs/04_metrics.md) | 每个指标怎么算、对照是什么 |
| [docs/05_experiments.md](docs/05_experiments.md) | 实验矩阵、配对、公平性 |
| [docs/06_reproduce.md](docs/06_reproduce.md) | 从 demo 到全量复现的命令 |

---

## 输入 / 输出（一句话版）

**数据（每个样本）**

- 条件 `c ∈ R¹²`
- 本体 id `e ∈ {0,…,5}`
- 干净 chunk：`x_manip ∈ R¹⁶ˣ⁵⁸`，`x_aux ∈ R¹⁶ˣ²²`
- 监督 mask 与 `aux_active`
- 协调 mode `α ∈ {0.35, 0.65}`，且 `α_manip + α_aux = 1`

**网络输入**

- 加噪后的两路 chunk `z_t = (1-t)ε + t x`
- 8 个只读 condition token
- AdaLN 的时间 `t`
- （主对比）共享 mode token

**网络输出**

- 每个动作 token 一个与该 token 同宽的向量
- **没有从 `z_t` 到输出的 skip**；三种参数化只是对这个向量的解读不同

**训练目标（所有 arm 同一个标量）**

```
x̂ = net                         # x-pred
x̂ = z_t + σ(t) · net            # v-pred
x̂ = (z_t − σ(t)·net) / max(t, σ_min)   # ε-pred

L = E[ w(t) ‖x̂ − x‖² ] ,  w(t) = 1/σ(t)² ,  σ(t)=max(1-t, 0.05)
```

---

## 论文主实验怎么跑

主对比是 **mode 给定、58 维/token、velocity-space loss**：

```bash
# 一个 seed 的一对（建议先这样确认环境）
python scripts/train_one.py --arm x-pred/v+mode --seed 0 --steps 12000 --out results
python scripts/train_one.py --arm v-pred/v+mode --seed 0 --steps 12000 --out results

# 完整矩阵（多卡）。先跑 headline：
python scripts/run_study.py --out results --only modewidth,noise --seeds 8 --hd-seeds 8 --noise-seeds 16

python scripts/report.py --results results --out results/report.md
python figures/fig_story.py --results results --out figures/out/fig_story
```

全量矩阵（含宽度扫描、loss 加权 3×2、关 joint attention 等）见 [docs/06_reproduce.md](docs/06_reproduce.md)。一块 A100 上完整研究是数百个 job；教学请从 `demo.py` 和 `modewidth` 的 `g=1` 开始。

---

## 论文里应对齐的数字（mode 给定、58D/token、12k step）

这些来自原始完整实验，**不是** `demo.py` 的 smoke 数字：

| 量 | x-pred | v-pred |
|---|---:|---:|
| Jacobian PR | 3.44 | 36.97 |
| Jacobian 99% 能量秩 | 13.5 | 51.7 |
| own-noise R² | 0.617 | 0.943 |
| partner-action R² | 0.293 | 0.212 |
| t=0.01 excess-risk 比 v/x | 1.64× | — |
| t=0.01 噪声不变性比 v/x | 23× | — |
| NFE=1 off-manifold | 0.090 | 0.127 |
| invalid-DoF leakage | 0.181 | 0.375 |
| excess denoising risk（聚合） | 0.0378 | **0.0373** |

最后一行是刻意保留的：x-pred **并不**靠训得更低的回归误差取胜，而是生成更可执行的动作。

---

## 设计上必须说清的三件事

1. **无 skip。** 若 `v`-head 可以在网外写 `x̂ = z_t + (1-t)v̂` 且残差流不必搬运 `ε`，本实验的机制不成立。本仓库的 decoder 只读残差流。
2. **同一个 loss。** 比较的是 head 发什么，不是优化两个不同目标。
3. **主对比给定 mode。** 不定 mode 时，MSE 最优是两个合法计划的中点；x-pred 会为此付“不够决断”的代价。真实动作专家通常有语言/目标条件，对应 toy 里的 mode token。

---

## License

MIT。合成数据，无真人图像或语音。
