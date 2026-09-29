# 3. 模型与损失

代码：`velatoy/model.py`，`velatoy/flow.py`，`velatoy/train.py`。

## 网络（`ToyJointExpert`）

微型双流 transformer，对应完整模型里的 joint action expert：

| 项 | 默认 |
|---|---|
| blocks | 4 |
| width | 512 |
| heads | 8 |
| FFN | 4×width |
| 每流 | 自己的 encoder / QKV / FFN / decoder |
| attention | 两流 token **拼接后** 一次共享 SDPA |
| condition | 8 个只读 token：只有 K、V，没有 Q，没有 FFN |
| time | AdaLN-zero |
| near/far | near token 不能读 far token |

### 关键：没有 skip

`model.py` 写得很明确：从 `z_t` 到预测的每一条路径都穿过宽 512 的残差流。  
因此 `v`-pred 若要输出 `x−ε`，必须在残差里保留关于 `ε` 的信息；`x`-pred 不必。

如果有人把 skip 加回去，本实验的机制证据链就断了。不要在没改文档的情况下加残差直连。

## 三种参数化，一个目标

`RectifiedFlow.to_endpoint`：

```
σ(t) = max(1 − t, 0.05)

x-pred:    x̂ = u
v-pred:    x̂ = z_t + σ(t) u
ε-pred:    x̂ = (z_t − σ(t) u) / max(t, 0.05)
```

`u` 是 head 的原始输出。然后所有 arm 都优化

```
L = E_{t,ε}[ w(t) ‖x̂ − x‖² ]
w(t) = 1/σ(t)²     # loss_space = "v"，再按 batch 归一化到均值 1
w(t) = 1           # loss_space = "x"，控制实验
```

在 `v`-pred + `w=1/σ²` 下，这就是标准 velocity MSE：`L = ‖u − v*‖²`。  
所以 **x-pred/v 与 v-pred/v 是同一个标量目标，只是 head 发的量不同。**

两流 loss 先各自按 mask 平均，再对活跃的 head 取平均，避免 58D 完全淹没 22D。

## 配对公平性

同一 `seed` 下，所有 arm 共享：

- 同一个生成流形对象（`DataConfig` seed 1234）；
- 同一套初始化（`torch.manual_seed(10000 + seed)`）；
- 同一 batch / 噪声 / `t` 序列（训练循环里的 generator 只由 seed 决定）。

解码器 **不用 zero-init**：对 `v`-pred，`u=0` 给出 `x̂=z_t` 是合理热启动；对 `x`-pred，`u=0` 是预测全零。对称小高斯初始化，避免一上来偏向 skip 约定。

## Mode token

`mode_conditioning=True` 时，训练用 **真实 mode**，推理从均匀先验抽样同一个 token 给两流。这是 controlled intervention，用来隔离“预测目标”和“多模态打结”。它不是可部署的完整方法（真实模型没有 oracle mode）。

## 采样

Euler，`t: 0 → 1`，默认部署 NFE=5。每一步：

```
x̂ = endpoint_fn(z, t)
z ← z + Δt · (x̂ − z) / σ(t)
```
