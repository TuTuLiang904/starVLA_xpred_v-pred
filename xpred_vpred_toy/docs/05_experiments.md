# 5. 实验矩阵

调度：`scripts/run_study.py`，条件：`velatoy/arms.py`。

## 主对比（写进论文的那组）

- 实验名：`modewidth`（宽度扫描 + 给定 mode）和 `noise`（58D/token、加密 `t` 网格、16 seed）
- arm：`x-pred/v+mode@g1` vs `v-pred/v+mode@g1`（以及 ε 作为 skip 极限）
- `g=1` ⇒ 58 个数/token
- 12k step，batch 256，AdamW `3e-4`，wd `1e-4`，warmup 200，EMA 0.999
- 评估 4096 样本；NFE ∈ {1,2,4,5,8,16}

## 其它实验（附录 / 机制对照）

| 名字 | 问什么 |
|---|---|
| `main` | **不**给 mode：会把 x-pred 的噪声不变性收成 inter-mode 插值代价 |
| `main_hd` | `token_group=4`（232/token），机制数字更干净 |
| `tokengroup` / `tokengroup_long` | 每 token 动作维扫描 |
| `seqlen` | 拆开“更宽”和“更短序列” |
| `width` | 网络宽度扫描 |
| `fix` | mode token vs 仅 mode 监督 |
| `schedule` | 把训练 `t` 质量往高噪声挪，看排序变不变 |
| `intrinsic` | 把流形加厚（增大 style_dim），流形假设变差时优势应收缩 |
| 3×2 factorial | `{x,v,ε}` × `{v-weight, unweighted}`：效应跟参数化走，不跟权重走 |
| `+cons` | 给 v-pred 加显式两端一致性罚，看能否买到 x-pred 的表征 |
| `nojoint` | 关掉跨流 attention |

## 怎么读结果文件

每个 job 一个 JSON：

```
results/{experiment}__{arm}__s{seed}.json
```

`arm` 名里的 `/` 会写成 `-`。  
`final` 是标量指标；`dynamics` 是训练过程探针。

汇总：

```bash
python scripts/report.py --results results --out results/report.md
```

配对统计：对共享 seed 做差，bootstrap 95% 区间，精确 Wilcoxon。比值用 **两臂均值之比**（与 `velatoy.stats.paired` 一致），不要和“先按 seed 求比再平均”混用。
