# 4. 指标：每个数字是什么

代码：`velatoy/probes.py`，`velatoy/metrics.py`，`velatoy/train.py` 的 `probe_suite`。

## 表征

### Jacobian participation ratio

`J = ∂x̂ / ∂z_t`（对当前 arm 的 **endpoint 映射**，不是 raw head）。  
奇异值 `s_i`：

```
PR = (∑ s_i²)² / ∑ s_i⁴
```

低 PR = 有效输入方向少。另报 **覆盖 99% Jacobian 能量所需方向数**（`jac_rank99`）。  
13 是数据全局内在维；PR≈3.4 是**局部 token 映射**的有效维，二者不应被要求相等。

未训练对照：三种参数化的 PR 一开始接近，差距是训练出来的。

### Own-noise R²

最后一个 block 的 per-token hidden → 该流自己的 `ε`。  
Ridge `λ=10⁻³`，特征标准化，70/30 划分，**留出集** R²。  
高 = 残差流还在记这次抽到的噪声。

打乱配对对照约 `−0.09`，用来挡住“特征维数高就会虚高 R²”。

### Partner-action R²

同样探针，目标换成 **partner 流的干净 chunk**。  
措辞只能是 “encodes more partner information”，不要写成更依赖通道。

天花板：两流都给闭式 Bayes 去噪器时，partner chunk 的 R² ≈ **0.84**。测到的 0.29 vs 0.21 远低于天花板，所以差别是“编码了多少”，不是“已经把通道用满”。

## 去噪

### Excess risk

`risk_model − risk_bayes_oracle`。Oracle 是知道 mode 的闭式 `E[x|z_t,c,e,mode]`。  
表里的 0.0378 / 0.0373 是对训练时间分布 `Beta(1.5,1)` 的**聚合**。  
图 (c) 是 **固定 t** 上的比值 `excess_v / excess_x`，来自更密的 `t` 网格和 16 个 seed。不要把两个量当成同一个点。

### Noise invariance (NIV)

固定干净 chunk，抽 24 个独立 `ε`，看 `x̂` 移动多少。Bayes 去噪器也会动，作为下限。  
`t=0.01` 处 v/x ≈ 23×：x-pred 的端点估计对噪声实现不敏感得多。

## 生成

Euler 从纯噪声积到干净端，再打分。

| 名字 | 定义 | 方向 |
|---|---|---|
| off-manifold mass | 生成 chunk 在已知动作子空间正交分量上的相对 RMS | 低好 |
| invalid-DoF leakage | 平方能量落在该本体永不标注的通道上的比例 | 低好 |
| coordination violation | `\|a_manip + a_aux − 1\|`（由已知 readout 读回分配） | 低好 |
| task residual | 实现的 `a` 到**较近合法 mode** 的距离 | 低好 |
| mode mismatch | 两流提交了不同的离散 mode | 低好 |

**不要**用相对记录轨迹的 RMSE 当主指标：mode 不定时，两个合法计划的平均在 RMSE 上更好，但物理上非法。
