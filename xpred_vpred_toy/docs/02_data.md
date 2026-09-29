# 2. 数据：合成流形的输入输出

代码：`velatoy/data.py`，配置：`velatoy/config.py` 的 `DataConfig`。

先看一个真实 batch：

```bash
python examples/inspect_manifold.py
```

## 为什么要合成数据

真实机器人轨迹没有闭式 `E[x | z_t, c, e]`。这里把生成过程写死，于是可以报告 **excess risk = 模型 MSE − Bayes 最优 MSE**，把“任务本身的多模态ambiguity”和“参数化带来的表征差距”分开。

## 每个样本有什么

| 字段 | shape | 含义 |
|---|---|---|
| `cond` | `(B, 12)` | 观测/条件，toy 里当作 VLM readout |
| `embod` | `(B,)` | 6 个本体之一 |
| `tau` | `(B, 8)` | 共享任务 latent，由 `cond` 决定 |
| `alpha` / `mode` | `(B,)` | 协调分配；mode 独立于 cond 抽样 |
| `style_manip`, `style_aux` | `(B, 4)` | 各流私有，partner 看不见 |
| `x_manip` | `(B, 16, 58)` | 干净 manipulation chunk |
| `x_aux` | `(B, 16, 22)` | 干净 body chunk |
| `mask_*` | 同 chunk | 该本体真正有标签的通道 |
| `aux_active` | `(B,)` | 有的本体没有 whole-body 标签 |

干净动作由 **13 维 latent** 线性张成（8 维任务份额 + 1 维 mode witness + 4 维私有 style），再乘上平滑时间基。因此 **全局内在维是设计出来的 13**，不是拟合出来的。图 (a) 验证的是这件事，不是某个训练好的网络。

## 协调 mode

```
α_manip ∈ {0.35, 0.65}
α_aux = 1 − α_manip
```

同一 `cond` 下两种计划都合法，它们的平均 `0.5` **不合法**。  
Witness 故意不对称：`witness_aux = 1.60`，`witness_manip = 0.12`。manipulation 流要接近 Bayes 最优，就必须通过 joint attention 读 body 流。这让“跨流信息”可测，而不是假定网络“应该交互”。

## Validity mask

不同本体丢掉不同通道（关节 vs EEF、有无灵巧手、有无 body）。生成指标里的 **invalid-DoF leakage** 就是：模型在这些永远无标签的维度上放了多少能量。

## 网络实际吃进去的张量

训练一步：

1. `t ~ Beta(1.5, 1)`（偏向干净端）；
2. `ε ~ N(0, I)`，与 `x` 同形；
3. `z_t = (1−t)ε + t x`；
4. 模型输入：`(z_manip, z_aux, t, cond, embod, masks, aux_active[, mode])`。

**没有单独的“动作标签文件”。** 标签就是刚采样的 `x_manip` / `x_aux`。

## 默认 tokenization

- `token_group=1`：每步一个 token，manipulation 每 token **58** 个数（MM-ABC 部署点）；
- `token_group=4`：每 token **232** 个数；图 (a) 的 928 是 **整段 manipulation chunk** `58×16`，与切成几个 token 无关。
