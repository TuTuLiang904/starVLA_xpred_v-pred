# 6. 复现命令

假设已 `pip install -r requirements.txt && pip install -e .`，且当前目录是仓库根。

## 0. 环境

- Python ≥ 3.10
- PyTorch 2.x，一块 GPU 即可跑 demo；全量研究建议多卡
- 不需要数据集下载

```bash
export CUDA_VISIBLE_DEVICES=0   # 按机器改
```

`run_study.py` 会按可见 GPU 起 worker；默认每卡 2 个进程。

## 1. 看懂数据（CPU 即可）

```bash
python examples/inspect_manifold.py
```

## 2. 教学 demo（推荐所有人先跑）

```bash
python scripts/demo.py                 # smoke，数分钟
python scripts/demo.py --full --steps 4000
```

输出 JSON 默认 `results/demo.json`。

## 3. 论文设定下的一对（1 seed）

```bash
python scripts/train_one.py --arm x-pred/v+mode --seed 0 --steps 12000 --eval-samples 4096 --out results
python scripts/train_one.py --arm v-pred/v+mode --seed 0 --steps 12000 --eval-samples 4096 --out results
```

单卡 12k step 量级大约数十分钟（视 GPU）。JSON 落在 `results/single__...json`，可把 `--experiment modewidth` 改成与矩阵一致的名字。若要和 `run_study` 的 modewidth 文件名对齐，用：

```bash
python scripts/run_study.py --out results --only modewidth --hd-seeds 1 --steps 12000
```

这会跑 `g ∈ {1,2,4,8,16}` × `{x,v,ε}` × 1 seed。只要 58D 一对：

编辑太重的话，直接两次 `train_one.py` 再自己看 `final` 字段即可。

## 4. Headline 矩阵（论文图 (b)(c)(d)）

```bash
python scripts/run_study.py --out results \
  --only modewidth,noise \
  --hd-seeds 8 --noise-seeds 16 --steps 12000
```

- `modewidth`：宽度扫描 + 给定 mode，8 seed，12k（`noise` 也在 headline 列表里，同样 12k）
- `noise`：仅 `g=1`，16 seed，密 `t` 网格

然后：

```bash
python scripts/report.py --results results --out results/report.md
python figures/fig_story.py --results results --out figures/out/fig_story
```

`fig_story.py` 会从 JSON **现场读数**；数字和文档差太多会打印 `!!` 警告。若你只跑了 1 个 seed 或短预算，警告是预期行为，图仍会画，只是不要和论文表格逐格对拍。

已提交的对照图在 `figures/reference/fig_story.{pdf,png,svg}`。

附录图：

```bash
python figures/figA_mechanism.py --results results --out figures/out/figA
python figures/figB_payoff.py    --results results --out figures/out/figB
python figures/figC_distributions.py --results results --out figures/out/figC
python figures/figD_width.py     --results results --out figures/out/figD
```

缺实验时对应 panel 可能为空或报错，按 `docs/05_experiments.md` 补跑。

## 5. 全量研究

```bash
python scripts/run_study.py --out results --seeds 8 --hd-seeds 8 --noise-seeds 16
```

这是数百个 job。失败的 JSON 不会写入；删掉坏文件后重跑同一命令会跳过已完成的 job。

## 6. 预算对照（教学用）

| 级别 | 命令 | 用途 |
|---|---|---|
| smoke | `python scripts/demo.py` | 检查代码路径 |
| 短论文趋势 | `demo.py --full --steps 4000` | 单 seed，可见 PR / 噪声 R² 方向 |
| 论文一对 | `train_one` 12k × 2 | 与表中数量级可比 |
| 论文图 | `run_study --only modewidth,noise` | 可画 fig_story |
| 全量 | `run_study` 无 `--only` | 所有对照 |

## 常见坑

1. **demo 数字 ≠ 论文表。** 宽度和 step 不够时 Jacobian / 生成指标会抖。
2. **`noise` 实验必须 12k。** 旧调度曾把它当成 4k sweep；本仓库已把 `noise` 放进 headline。
3. **零初始化 decoder 对 v-pred 不公平。** 不要改回 DiT 式 zero-init 还声称公平比较。
4. **比值口径。** 正文用均值比；按 seed 求比再平均会得到约 1.65 和 24 而不是 1.64 和 23.5。
5. **图 (a) 的 SVD 样本数要多于 928 列。** `fig_story.py` 已按 embodiment 过滤后采足够行，否则噪声谱会在样本数处假性截断。
