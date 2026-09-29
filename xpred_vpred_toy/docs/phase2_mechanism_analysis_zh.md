# x-pred / v-pred 第二阶段机制分析

本文档整理当前第一阶段准确率、toy story、geometry、recovery 和 NFE probe 的结果，目标是回答三个问题：

1. x-pred、v-pred 和 x-pred-v-loss 分别在学习什么；
2. 为什么局部 recovery、离线 NFE 和最终仿真准确率的排序不一致；
3. 下一步应该如何把现象收敛成可以写进论文的机制结论，并为第三阶段融合做准备。

数据记录表见 [`action maniflow learning数据.xlsx`](../action%20maniflow%20learning%E6%95%B0%E6%8D%AE.xlsx)，本次 probe 输出位于 [`real_mechanism/outputs`](../real_mechanism/outputs)。所有 checkpoint 使用 seed42。

> **重要更正**：本文档第一版把当前 StarVLA recovery 的排序写成了“高噪声下 v 略好”，并将它直接用于解释 toy story。这是不准确的。你同事 paper 中的高噪声结论是：在受控 toy 中，固定 flow time 后，x-pred 的 **excess risk** 明显低于 v-pred；而我们当前真实模型 probe 计算的是原始 endpoint MSE，且不是同一个实验设置。下面明确区分两者。

---

## 1. 研究背景：我们究竟在比较什么

扩散/flow matching 动作策略在噪声状态 `z_sigma` 上预测一个量。三种参数化的含义不同：

- **x-pred**：直接预测干净动作终点 `x`；
- **v-pred**：预测噪声和干净动作之间的方向/位移场 `v`；具体符号随实现而变，但都可以转换回干净动作端点；
- **x-pred-v-loss**：输出仍然是 `x`，但训练损失不是普通的 `||x_hat-x||²`，而是把误差按噪声尺度重新加权。

以 StarVLA 的实现为例，训练路径是：

```text
z_t = (1-t) * noise + t * action
```

因此 `t=0` 是高噪声，`t=1` 是干净动作。v-pred 的目标近似为 `action-noise`，x-pred 的目标是 `action`。

x-pred-v-loss 实际上计算了：

```text
((x_hat - z_t) / (1-t))  vs  ((action - z_t) / (1-t))
```

这等价于对 x 误差施加大约 `1/sigma²` 的权重，其中 `sigma=1-t`。因此它不是“第三种输出空间”，而是：

> 仍然预测 x，但显著强调低噪声、接近最终动作的区域，弱化高噪声区域。

这点是解释后续结果的核心。

---

## 2. Toy story：clean action 是低维流形，noise 和 flow field 更高维

### 2.0 paper 的结论和我们的 probe 不是同一个实验

同事 paper 第 3 节的 toy study 使用的是一个可解析的合成系统：

- clean action 由已知的低维 latent 生成；
- 有闭式 Bayes denoiser，可以计算不可约 Bayes risk；
- x/v 两个模型共享初始化、数据、batch 顺序、噪声和 flow time；
- 两者只改变 output parameterization，主对比使用相同的 velocity-space loss；
- 通过 `excess risk = model MSE - Bayes risk` 消除任务本身的多模态不确定性；
- NFE 指标是已知合法动作子空间上的 off-manifold mass，而不是普通 replay MSE。

paper 的“高噪声 x 更好”具体指固定 `t` 时的 excess-risk 比值：

```text
excess_v(t) / excess_x(t) > 1
```

在 `t≈0`（几乎纯噪声）处约为 1.64；随着 `t` 接近 1，二者接近持平，低噪声处甚至可能略微反转。这里的高噪声参数是 `t` 小，而不是把所有噪声层的平均 MSE 汇总后比较。

我们的 `real_mechanism/scripts/run_*_checkpoint_probe.py` 只借鉴了路径和 endpoint 转换，**没有复现 toy 的 Bayes risk、paired training、Jacobian probe、own-noise/partner-information probe 和 off-manifold 指标**。所以应该说“real probe 受 toy 启发”，不能说“已经复现了 paper 的 toy experiment”。

本阶段的 toy story 不是简单地说“x 比 v 好”，而是区分两种结构：

### 2.1 干净动作是低维、结构化的终点

机器人动作序列虽然表示为 `H × D` 个数，但实际受任务语义、机器人运动学、接触关系和时间结构约束，真正自由变化的维度远少于原始坐标维度。因此 clean action `x` 更像低维流形上的点。

### 2.2 噪声是高维扰动

独立或近似独立的噪声会填充大量原本不存在于动作流形上的方向。它的谱更平、更高维，不能直接代表有意义的控制变化。

### 2.3 v 是从噪声到动作的方向场

v-pred 不直接给出流形终点，而是给出当前点应该往哪里走。方向场需要在整个噪声路径上保持一致，因此它可能比 x 更高维、但更适合做连续积分。

由此得到两个可能互相竞争的性质：

- x：终点语义清晰、动作结构低维、适合最终校正；
- v：方向场平滑、全局一致、适合多步积分和轨迹连续性。

这正是为什么不能用单一的 recovery MSE 判断最终控制性能。

---

## 3. Geometry 指标：它们测量什么

当前 geometry 使用 RobotWin clean action、噪声和 v 表示进行谱及 intrinsic dimension 分析。当前完整结果在 [`geometry_robotwin_h32/global_task.csv`](../real_mechanism/outputs/geometry_robotwin_h32/global_task.csv)，局部 anchor 结果在 [`local_anchor.csv`](../real_mechanism/outputs/geometry_robotwin_h32/local_anchor.csv)。

### 3.1 Singular value spectrum

对每个任务的动作矩阵做 SVD，观察奇异值能量是否集中在少数方向。

- 谱下降快：数据集中在少数主方向；
- 谱下降慢：表示包含更多独立变化方向。

### 3.2 Stable rank

`stable_rank = ||A||_F² / ||A||_2²`。

它是有效秩，越小表示能量越集中在头部奇异方向。它不像硬 rank 那样受数值阈值影响，适合比较不同表示的谱集中程度。

### 3.3 Participation ratio

`PR = (sum(s_i²))² / sum(s_i^4)`。

它近似表示“有多少个方向共同承担能量”。越小表示有效维度越低。

### 3.4 rank99

达到总谱能量 99% 所需的奇异方向数。它更直观，但对尾部噪声和阈值比较敏感。

### 3.5 TwoNN intrinsic dimension

使用最近邻距离估计局部 intrinsic dimension。它不是线性秩，而是局部几何维度估计。由于高维噪声和有限样本会使该估计偏高，它适合看相对趋势，不应被当成绝对真实维度。

### 3.6 当前结果

按所有 global task 行取均值：

| 表示 | stable rank | participation ratio | rank99 | TwoNN ID |
|---|---:|---:|---:|---:|
| clean x | 1.695 | 2.469 | 11.56 | 2.63 |
| noise ε | 24.07 | 35.81 | 39.00 | 80.27 |
| v | 3.911 | 10.84 | 38.04 | 41.37 |

结果强烈支持以下较弱、但可靠的结论：

> 在当前 RobotWin 数据上，clean action 的有效维度明显低于 noise 和 v；v 不是噪声本身，但其几何复杂度显著高于 clean action。

这解释了为什么直接预测 x 可能更符合动作流形；同时也说明 v 的学习问题不是简单的低维回归，而是高维向量场估计。

注意：geometry 只能说明表示的几何结构，不能单独推出哪种表示的闭环成功率一定更高。局部切空间变化、曲率和 tangent variation 还需要进一步汇总后，才能讨论“流形是否弯曲、v 是否在切空间内稳定运动”。

---

## 4. Recovery 指标：测量局部一次预测能力

Recovery probe 的流程是：

1. 从真实动作序列取一个 anchor；
2. 按指定 `sigma` 加噪；
3. 给模型真实条件输入和 noisy action；
4. 只调用一次模型；
5. 把 v 转换为 x endpoint 后与真实 clean action 计算 MSE。

因此 recovery 主要测量：

> 在给定噪声层、给定真实状态条件下，模型一次把 noisy point 拉回真实动作端点的能力。

它不是完整的 policy rollout，也不包含模型自身造成的状态分布偏移。MSE 越低越好。

### 4.1 StarVLA recovery

高噪声区域的结果约为：

| sigma | v | x | x-v-loss |
|---:|---:|---:|---:|
| 0.80 | 0.181695 | 0.181743 | 0.231245 |
| 0.95 | 0.180661 | 0.181545 | 0.233088 |
| 0.99 | 0.180908 | 0.181592 | 0.232929 |

这里在当前实现的**原始 endpoint MSE**下，v 略好于 x，但差距非常小；x-v-loss 明显更差。

正确的表述应当是：

> 当前 StarVLA real probe 没有复现 toy 中的“高噪声 x 优势”；它只显示 v 和 x 的 raw endpoint MSE 几乎打平，v 有轻微优势。这个结果不能推翻 toy 结论，因为我们没有计算 toy 定义的 excess risk，也没有在相同初始化和相同训练轨迹下配对训练真实模型。

另外，x-v-loss 不能和 paper 的 x-pred 直接等同。paper 的主对比是 x-pred 与 v-pred 在同一个 velocity-space objective 下的 parameterization 对比；我们的 x-v-loss 是另一种训练目标，在 StarVLA 中近似使用 `1/sigma²` 重新加权 x 误差。它在高噪声区域受到的训练权重更低，因此 recovery 差并不意外。

### 4.1.1 为什么三条曲线形状完全不同

当前脚本的定义为：

```text
sigma = 1 - t
z = sigma * eps + (1-sigma) * x
```

所以 `sigma=0.01` 是几乎干净的输入，`sigma=0.99` 是几乎纯噪声的输入。三种头的 endpoint 写法不同：

```text
v-pred endpoint:      x_hat = z + sigma * v_hat
x-pred endpoint:      x_hat = x_hat_direct
x-pred-v-loss:        x_hat = x_hat_direct
```

这直接导致曲线不能按同一种直觉阅读：

- **v-pred 在低 sigma 处接近零不是纯粹的模型能力**。即使 `v_hat=0`，输出也是 `z`；当 `sigma=0.01` 时，`z` 已接近真实 x。v 的直连项在低噪声区提供了一个很强的“免费 baseline”。
- **普通 x-pred 从 0.01 到 0.99 都约为 0.18**，表示该 head 的 endpoint 大体由视觉/语言/state 条件决定，几乎没有有效利用当前 noisy action 的噪声程度。它不是“在每个 sigma 都同样会去噪”，而是近似输出同类 conditional action guess。
- **x-pred-v-loss 从约 0.176 增到约 0.233**，与 `1/sigma²` 的低噪声加权一致：低 sigma 时比普通 x 稍好，高 sigma 时缺少足够监督，输入接近纯噪声后误差上升。

因此，v 在 sigma 从 0.01 上升到 0.99 时从近零增长到约 0.18，主要是因为“可直接复制的 z”逐步被纯噪声替代；不是说明 v 的网络在 sigma=0.01 神奇地恢复了所有动作。

在真正可比较的高噪声端（sigma=0.80/0.95/0.99），v 和普通 x 仍只有约 0%–0.5% 的差异。以 sigma=0.99 为例，配对的 `x-v` MSE 差约为 `+0.000685 ± 0.000037`（正号表示 x 较差），所以当前 checkpoint 并不支持“真实 StarVLA 的 x 在高噪声 raw MSE 上优于 v”。

### 4.2 FastWAM recovery

FastWAM 中大致是：

- x 最好；
- v 次之；
- x-v-loss 最差。

这说明 recovery 排序已经依赖模型架构，不能把 StarVLA 的结果推广到 WAM。

### 4.3 LiLaWAM recovery

LiLaWAM 中 x-v-loss 的 recovery 最好，说明它在不同 WAM 实现中并不总是高噪声恢复差。这个差异可能来自数据规模、动作归一化、网络结构和训练阶段，而不是预测参数化单独决定的。

---

## 5. Excel 中的闭环准确率：测量任务成功，而不是动作 MSE

Excel 中的准确率/成功率是仿真环境中的任务级指标，数值越高越好。它包含了：

- 模型反复推理；
- 预测动作影响下一个状态；
- replan 和 action chunk 执行；
- 接触、抓取、碰撞和终止条件；
- 任务成功的非线性阈值。

因此它和 recovery MSE 之间没有理由保持同样排序。

### 5.1 StarVLA

StarVLA RobotWin 的均值约为：

| 设置 | v | x | x-v-loss |
|---|---:|---:|---:|
| clean | 0.4288 | 0.4292 | 0.5752 |
| random | 0.0704 | 0.0816 | 0.1084 |

LIBERO fine-tune 的总成绩也是 x-v-loss > x > v。

这和当前 raw recovery 的 v 略优并不矛盾，原因是：

1. **recovery 是全维度平均误差**：所有时间步和动作维度近似等权；
2. **任务成功由少数关键量决定**：例如末端接近、gripper、接触瞬间和最后几个 action step；
3. **x-v-loss 强调低噪声区域**：它可能牺牲高噪声 MSE，但把最终动作校正得更适合执行；
4. **闭环会累积小误差**：单次误差很小不代表重复积分后轨迹仍稳定；
5. **x-v-loss 可能改变动作的语义方向**：平均误差变大，但关键维度的方向更正确。

因此当前最合理、且不混淆 toy 与真实模型的解释是：

> 在真实 StarVLA checkpoint 上，v 与普通 x 的 raw recovery 几乎相同；x-v-loss 的低噪声加权则可能牺牲高噪声 raw MSE，换取任务最后阶段的关键动作修正，所以闭环成功率更高。toy 中的高噪声 x 优势需要用 excess risk 和受控 paired training 重新验证，不能从当前 raw MSE 直接推出或否定。

这是一种“global MSE”和“task-critical error”的差异，而不是实验矛盾。

### 5.2 FastWAM 和 LiLaWAM

FastWAM RobotWin 多数设置是 v > x > x-v-loss，LiLaWAM 也基本是 v ≥ x。可是 FastWAM offline recovery 中 x 又略好。这说明 WAM 的任务成功率更可能受到：

- 动作时间连续性；
- jerk 和轨迹平滑度；
- 接触动力学；
- 状态反馈与误差累积；
- action chunk 的整体一致性；

的影响，而不是单次 endpoint MSE。

FastWAM 当前 action-only 和 joint NFE probe 几乎一致，因此现阶段不能宣称“视频/partner 信息耦合”是 v 优势的原因。

---

## 6. NFE 指标：测量离线多步积分，不是仿真准确率

NFE 是 number of function evaluations，即在一次从噪声到动作的离线去噪中调用模型多少次。当前 NFE probe 仍然使用固定的离线 anchor 和真实条件，输出的是 action MSE、delta 或 jerk 等统计量。

它能回答：

> 这个预测头产生的场，经过多次数值积分后是否保持一致、平滑、接近真实动作？

它不能直接回答：

> 机器人在真实闭环仿真中成功率是多少？

### 6.1 StarVLA NFE

StarVLA 中 v 在 NFE=1、4、16 都是最好的，x 略差，x-v-loss 随 NFE 增加而恶化。x-v-loss 的 trajectory jerk 也明显更大。

这支持一个重要机制假设：

> StarVLA 的 x-v-loss 在低噪声终点附近有较强的任务校正能力，但其预测在不同噪声层之间不够全局一致，反复积分时会产生抖动和误差累积。

也就是说，它更像“好的 endpoint corrector”，而不是“好的 global vector field”。

更具体地说，NFE=1 从纯噪声 `t=0` 开始，因此是 recovery 高噪声端的直接补充。StarVLA 的 raw replay MSE 为：

| NFE | v | x | x-v-loss |
|---:|---:|---:|---:|
| 1 | 0.18047 | 0.18165 | 0.23291 |
| 4 | 0.18010 | 0.18165 | 0.24260 |
| 16 | 0.18025 | 0.18183 | 0.24762 |

v 与 x 的绝对差仅约 `0.0012–0.0029`（约 0.7%–1.6%），故可称为“数值上非常接近、v 系统性略低”，不能把它夸大为两者机制已经分离。增加 NFE 也几乎不能降低两者的 replay MSE，说明当前误差主要是 conditional action ambiguity/模型表示误差，而不是 Euler 离散误差。

x-v-loss 则从 NFE=1 的 0.23291 上升到 NFE=16 的 0.24762，且 seed variance、action delta 和 jerk 都远大于 x/v。它的起点正是训练权重最弱的高噪声区域；首步偏差之后，多个不同 time 的 endpoint 预测又不一致，于是更多 NFE 不能修正，反而把不一致的场累积起来。这是“field consistency 不足”的证据，但仍需直接的 path-consistency probe 才能作为最终机制结论。

### 6.2 LiLaWAM NFE

LiLaWAM 中 x-v-loss 在 NFE=1 最好，但 NFE=4/16 后 x 更好。这说明一步 recovery 的优势不等于多步积分优势。

### 6.3 FastWAM NFE

FastWAM action-only NFE 中 x 最好，但闭环 Excel 中 v 多数更好。这进一步说明 WAM 的闭环优势可能来自轨迹和状态反馈，而不是离线 endpoint MSE。

---

## 7. 当前可以写进论文的结论

### 7.1 已经有证据支持的结论

1. clean action 的有效几何维度显著低于 noise 和 v；
2. x、v 和 x-v-loss 优化的是不同性质：终点结构、方向场一致性和低噪声校正；
3. recovery、NFE 和 closed-loop success 的排序可以不同，这是由评价对象不同造成的；
4. StarVLA 当前 raw probe 中 v 与 x 的局部 recovery 几乎打平，v 在该指标上略好；多步离线积分 v 更稳定，但 x-v-loss 的闭环任务成功率更高；
5. WAM 中 v 通常有更好的闭环表现，但并不总是有更好的单步 recovery；
6. 因此不存在与模型无关的“x 永远优于 v”或“v 永远优于 x”的结论。

### 7.2 现在还不能写死的结论

以下结论目前证据还不够：

- 不能说 clean action 低维就必然应该使用 x-pred；
- 不能说 v 一定代表更平滑的真实控制轨迹，需要直接测量 temporal smoothness 和 field consistency；
- 不能说 x-v-loss 的高准确率完全来自低噪声加权，需要做 loss ablation；
- 不能把 toy 中“高噪声 x 的 excess-risk 优势”直接外推到真实 StarVLA；当前 real probe 没有 Bayes reference，也没有复现 toy 的 paired finite-capacity setup；
- 不能把不同模型的绝对 recovery MSE 横向比较，因为归一化统计量和动作空间不同；
- 不能把 FastWAM old/new 两套 checkpoint 的结果直接合并，它们属于不同评估配置。

---

## 8. 下一步实验顺序

### 第一优先级：闭环 NFE 曲线

在完全相同的 RobotWin task、seed、checkpoint 和 replan 设置下，分别跑 NFE=1、2、4、8、16 的仿真成功率，并记录：

- overall success；
- 每个 task 的 success；
- episode length；
- 碰撞/抓取失败类型；
- action jerk 和终点误差。

这一步用于判断 offline NFE 曲线是否真正预测 closed-loop 曲线。

### 第二优先级：分解 recovery 误差

不要只报告一个 global MSE，至少拆成：

- sigma=0.1/0.3/0.5/0.7/0.9/0.99；
- 前 1、4、8、16 个 action step；
- position、rotation、gripper；
- 每个动作维度；
- clean/random task；
- mean、median 和 worst-case。

重点检验：x-v-loss 是否在低噪声和关键 gripper/接触维度上明显更好。

同时要把 paper 的定义单独复现出来：

1. 用同一个 `(anchor, noise seed, sigma)` 产生配对输入；
2. 分别计算 `MSE_x(sigma)` 和 `MSE_v(sigma)`；
3. 明确报告 raw MSE，不要把它叫 excess risk；
4. 如果能建立 Bayes/teacher 或局部 oracle，再报告 `MSE - oracle risk`；
5. 横轴使用 paper 的 `t`，并明确 `sigma=1-t`，避免把“高噪声”方向写反。

### 第三优先级：向量场一致性

建议增加四类指标：

1. **endpoint consistency**：不同 NFE 最终 x 的差异；
2. **one-step/two-step consistency**：先走一步再走一步，是否等于直接走两步；
3. **path dependence/cycle error**：不同积分路径是否得到相近终点；
4. **temporal smoothness**：动作 delta、加速度和 jerk。

这一步可以直接验证“v 是更好的向量场、x-v-loss 是更好的终点校正器”这一机制假设。

### 第四优先级：验证 x-v-loss 的加权机制

补充三个损失版本：

- 普通 x MSE；
- `1/sigma` 加权；
- `1/sigma²` 加权；

并分别画出每个 sigma 区间的训练 loss、recovery 和闭环成功率。这样才能把“观察到 x-v-loss 好”推进到“证明低噪声加权导致关键动作改善”。

---

## 9. 第三阶段融合的设计启发

不建议直接平均 `x_hat` 和 `v_hat`。首先把 v 转为 endpoint：

```text
x_from_v = z - sigma * v_hat
```

然后做动态融合：

```text
x_fused = alpha(sigma, state, action_dim) * x_hat \
        + (1-alpha(sigma, state, action_dim)) * x_from_v
```

初始假设可以是：

- 高噪声阶段更多使用 v；
- 低噪声阶段更多使用 x 或 x-v-loss；
- 关键 gripper/接触维度使用更偏向 x-v-loss 的权重；
- WAM 增加 jerk 和 temporal consistency 正则；
- StarVLA 增加 endpoint accuracy 和 task-critical action loss。

融合权重不能只用固定常数。应该在 validation task 上根据以下目标学习：

```text
endpoint error
+ field consistency
+ temporal smoothness
+ task-critical action error
+ closed-loop success
```

最终可以把第三阶段概括为：

> v-pred 提供全局、稳定的运动方向；x-pred 提供低维、语义化的动作终点；融合模块根据噪声阶段和动作维度动态选择两者。

---

## 10. 一句话总结

当前结果不是说明某一种预测方式绝对更好，而是说明它们在不同层次上优化了不同目标：

> toy 中 clean x 更低维，x-pred 在高噪声 excess recovery 和少步 off-manifold 质量上更有优势；真实 StarVLA 当前 raw endpoint probe 尚未复现这一点，只显示 v 与 x 近乎打平、v 略优。x-v-loss 是额外的低噪声加权训练目标，不能与 toy 的普通 x-pred 等同。recovery 测局部 raw 误差，NFE 测离线积分一致性，仿真准确率测闭环任务成功，因此必须把 toy、raw probe 和 closed-loop 结果分开叙述。
