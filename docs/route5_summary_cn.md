# Route5 总结报告

## 1. 一句话结论

`route5` 已经在当前原型下实现了

```text
H_min > 2
```

而且这条路线不是依赖单光子探测器，而是

```text
高速 CV 前端 + 数字 coarse-graining + route2 单设备 SDP
```

因此它从原理上规避了“离散单光子探测器死时间把采样速率卡到 MHz”这一瓶颈。

当前最好结果来自：

- `radii = [0.0, 0.85, 1.25]`
- `8` 相位
- `6 x 2` IQ 分区
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`
- `num_quadrature_nodes = 12`
- 认证前 `3` 个最有希望的 target inputs
- `MOSEK`

得到

```text
H_min ≈ 2.11639
```

结果文件见：
[route5_local_refine_queue_mosek_v1.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/route5_local_refine_queue_mosek_v1.json)

## 2. Route5 的原理

### 2.1 核心思想

`route5` 的目的，是把两类原本看起来冲突的要求拼到一起：

- 硬件上要保留高速连续变量接收机，避免单光子计数那种低速瓶颈；
- 安全分析上仍想保留 `route2` 那种单设备 MDI SDP 认证框架。

因此，`route5` 采用的是下面这个混合结构：

```text
广义 coherent trusted alphabet
+ beam splitter + dual-homodyne / IQ 测量
+ 数字离散化输出
+ route2 单设备 guessing-probability SDP
```

它和旧路线的关系可以这样理解：

- 相对 `route3`：
  不再只用“固定振幅、只扫相位”的 alphabet，而是允许同时扫振幅和相位。
- 相对 `route2`：
  安全模型还是离散输出的 prepare-and-measure 单设备 SDP；
  但物理前端不再是“自由离散 POVM + 单光子探测器”，而是高速可实现的 CV / IQ 接收机。

### 2.2 为什么这能解决“离散方法采样速率慢”的担心

这里最关键的区分是：

- “安全证明里最终输出是离散的”
- 和
- “物理硬件必须用低速单光子探测器做离散计数”

这两件事不是一回事。

`route5` 的做法是：

1. 先用高速 IQ / dual-homodyne 前端得到连续模拟量；
2. 再在数字端做 coarse-graining，把连续样本映射成有限个离散输出；
3. 最后把这个离散输出送进 `route2` 的 SDP。

因此：

- 输出在安全模型里依然是离散的；
- 但采样前端仍然可以是高速连续变量硬件。

也就是说，`route5` 解决的正是之前最担心的那件事：

```text
“离散安全模型” 并不等于 “必须使用 MHz 级的单光子探测器离散采样”。
```

## 3. 为什么它绕开了单光子死时间瓶颈

导师那句解释：

> 离散方法用单光子的偏振或者相位编码，需要用单光子探测器去测，单光子探测器有死时间，所以最大探测速率是死时间的倒数，大概是 MHz 量级。

如果说的是“传统基于单光子编码和 APD/SNSPD 计数的离散方案”，这句话是对的。

但对 `route5` 来说，这个限制不是主瓶颈，因为 `route5` 的测量前端不是：

- 单光子偏振计数
- 单光子相位计数
- APD click/no-click 读出

而是：

- beam splitter
- 双路 homodyne / IQ 检测
- 线性光电探测器
- 模拟前端
- ADC 采样
- 数字分箱

所以对 `route5` 更准确的说法是：

```text
APD / 单光子探测器的死时间瓶颈被规避了，
但系统仍然会受到 CV 接收机带宽、ADC、调制器和校准开销的限制。
```

换句话说：

- “MHz 级单光子死时间上限”不是 `route5` 的主限制；
- 但“完全没有速率限制”也不对。

`route5` 仍然会受到这些因素限制：

- 平衡探测器和 TIA 的模拟带宽
- ADC 采样率、ENOB、抖动和饱和
- LO 功率与 shot-noise clearance
- IQ 通道失衡、相位漂移与校准频率
- trusted coherent alphabet 的调制速度
- 测试轮占比、参数估计和提取器损耗

因此，结论应当表述成：

```text
route5 可以从原理上避开“单光子死时间导致的 MHz 级采样率上限”，
但它仍然需要面对高速 CV 接收机的工程带宽约束。
```

## 4. 实现流程

当前代码流程是：

1. 构造 trusted alphabet
   - 同时扫描半径和相位，形成广义 coherent alphabet。
2. 做局域支撑降维
   - 只在 trusted inputs 的精确支撑子空间里工作，压缩 SDP 维度。
3. 固定中央测量为物理受限 IQ 结构
   - beam splitter + dual-homodyne。
4. 在数字端搜索 IQ 分区
   - 当前限制为 axis-aligned 的 `x/p` 分箱。
5. 先按 raw 指标粗筛
   - 先看观测分布上的 raw 最小熵。
6. 再做正式 SDP 认证
   - 对 top-k 候选做单设备 guessing-probability SDP。
7. 不只认证 raw-best 输入
   - 对前若干个最有希望的 target inputs 都做认证，再取最好者。

实现上后来又补了两项加速：

- 复用单个 SDP 模型，只更换目标态参数，而不是每个 target 都重建问题；
- 给 `route5` CLI 和精修队列补上 `MOSEK` 参数与线程设置。

关键代码位置：

- [common.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/common.py)
- [hybrid_iq.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/route5/hybrid_iq.py)
- [main.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/route5/main.py)
- [refine_queue.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/route5/refine_queue.py)

## 5. 参数调整过程

### 5.1 先验搜索阶段

最开始的经验是：

- `12` 输出通常优于 `16` 输出；
- 最有效的结构不是更对称的 `4 x 4`，
  而是把更多分辨率放在更有信息的那一轴上，即 `6 x 2`；
- 只看 raw 分布时，某些 `16` 输出点可以很平，
  但 formal certification 未必更高。

因此，后续搜索基本收敛到：

- `num_x_bins = 6`
- `num_p_bins = 2`

### 5.2 早期 SCS 阶段

早期快跑里曾出现：

- `[0.0, 1.0, 1.2]`
- `range = 2.0`
- `gamma = 1.25`
- `nodes = 10`

对应

```text
H_min ≈ 1.8188
```

但后来更严格地重跑后发现，这个值偏乐观。

对照结果：

- [route5_single_12out_candidate_newbest_r1012_fastscs.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/route5_single_12out_candidate_newbest_r1012_fastscs.json)
  `H_min ≈ 1.8188`
- [route5_single_12out_r1012_nodes20_strict.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/route5_single_12out_r1012_nodes20_strict.json)
  `H_min ≈ 1.5636`
- [route5_single_12out_r0911_range185_gamma105_nodes20.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/route5_single_12out_r0911_range185_gamma105_nodes20.json)
  `H_min ≈ 1.5689`

这说明：

- 快速 `SCS` 结果适合做方向判断；
- 但不适合直接拿来作为最终结论。

### 5.3 转向 MOSEK 阶段

在确认本机 `generic` 环境里可用 `MOSEK` 且许可证正常后，
搜索策略升级为：

- 仍用 raw scout 做粗筛；
- 但 formal certification 改用 `MOSEK`；
- 且每个候选认证前 `3` 个最有希望的 target inputs。

这一步非常关键，因为我们观察到：

```text
最优 formal target 不一定等于 raw-best target。
```

例如：

- `[0.0, 0.9, 1.05]` 这个点里，
  raw-best 是 `[15, 7]`，
  但 formal 最优实际是 `[15, 15]`，
  从而把 `H_min` 推到了 `2.0806`。

## 6. 最终结果

当前 `MOSEK` 队列已经完成 top-8 候选的 formal certification。

最优结果如下：

| 排名 | radii | H_min | target_input |
| --- | --- | ---: | --- |
| 1 | `[0.0, 0.85, 1.25]` | `2.11639` | `[15, 7]` |
| 2 | `[0.0, 0.9, 1.15]` | `2.11030` | `[11, 3]` |
| 3 | `[0.0, 0.85, 1.2]` | `2.10055` | `[11, 3]` |
| 4 | `[0.0, 0.85, 1.1]` | `2.08283` | `[11, 11]` |
| 5 | `[0.0, 0.9, 1.05]` | `2.08058` | `[15, 15]` |
| 6 | `[0.0, 0.9, 1.1]` | `2.07990` | `[15, 15]` |
| 7 | `[0.0, 0.85, 1.05]` | `2.07243` | `[15, 15]` |
| 8 | `[0.0, 0.85, 1.15]` | `2.06937` | `[15, 7]` |

完整结果见：
[route5_local_refine_queue_mosek_v1.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/route5_local_refine_queue_mosek_v1.json)

这个结果意味着：

```text
route5 已经在当前原型下实质上达到 H_min >= 2 的目标。
```

## 7. 是否还有必要继续改进

如果当前目标只是：

```text
找到一条既能保留高速 CV 前端、又能把认证最小熵推到 2 bit 以上的路线
```

那么现在已经达标，可以停止作为“必须继续优化”的主任务。

继续优化当然还可以做，但属于锦上添花，不再是“是否可行”的必要验证。

因此我的判断是：

- 从“达到目标”这个角度：已经够了。
- 从“继续冲更高数值”这个角度：可以停，也可以作为后续附加工作慢慢做。

## 8. 代码逻辑检查结论

我对当前 `route5` 的实现做了针对性复核，没有发现会推翻结论的明显逻辑 bug。

主要依据是：

1. `single` 与 `partition-search` 的概率生成都走同一条中央测量路径；
2. `route5` 的 formal certification 结果在切换 `SCS -> MOSEK` 后表现出更一致、更可信的趋势；
3. 最终 `MOSEK` 结果为 `optimal`，而不是 `optimal_inaccurate`；
4. top-8 候选全部完成后，结果排序是连贯的，不像数值偶然或某个点单独爆掉；
5. 多 target-input 认证能解释为什么某些点 formal 最优不等于 raw-best，这和模型结构是相容的。

### 8.1 当前我没有看到的明显逻辑问题

- 没有证据表明 `16` 输出低于 `12` 输出是代码 bug；
- 没有证据表明 `MOSEK > 2` 只是因为某个文件被写错；
- 没有证据表明 `route5` 的新队列在复用 SDP 后改变了目标函数含义。

### 8.2 仍需保留的实现风险

虽然没有发现明显 bug，但仍有几项风险要如实保留：

1. 当前最优值是在 `num_quadrature_nodes = 12` 下得到的。
   - 这比最早的 `10` 已经更稳；
   - 但还不是“无限精度”。
2. 目前只认证了每个候选的前 `3` 个 target inputs。
   - 这已经明显优于只看 `1` 个；
   - 但原则上仍可能存在更好的 target 没被扫到。
3. 还没有做有限尺寸分析。
4. 还没有把测试轮占比、参数估计开销和提取器损耗合进最终平均认证速率。
5. 当前还是协议-硬件共设计原型，不是完整实验闭环。

因此，当前更合适的结论不是“所有问题都已完全解决”，而是：

```text
route5 已经完成了可行性验证，并在当前数值原型中实现了 H_min > 2；
但若要作为最终实验方案，还需要做更严格的数值与工程收口。
```

## 9. Route5 的优点

相对于此前路线，`route5` 的优势可以概括为：

1. 它把“高速硬件”和“离散安全模型”统一起来了。
2. 它规避了单光子死时间导致的 MHz 级瓶颈。
3. 它保留了 `route2` 的单设备 SDP 认证结构。
4. 它比 `route3` 的 phase-only alphabet 更强。
5. 它的中央测量结构比“自由高输出 POVM”更接近真实可实现的实验前端。
6. 它允许把连续前端和数字 coarse-graining 清晰分层，工程解释更自然。

## 10. 仍然存在的风险

最后再把最重要的残余风险单独列出来。

### 10.1 数值风险

- `nodes = 12` 仍然不是最终收敛性证明；
- 候选认证只覆盖了 top-3 target inputs；
- 目前没有有限尺寸与统计波动分析。

### 10.2 实验风险

- 高速 CV 前端的模拟带宽、ADC 和平衡探测器仍然会限制真实速率；
- trusted alphabet 的高速调制与稳定性仍需要工程验证；
- IQ 分区参数是否能在真实噪声、失衡和漂移下保持最优，还需要实验校准。

### 10.3 速率解释风险

`route5` 解决的是：

```text
“离散安全模型不必等于低速单光子采样”
```

而不是：

```text
“系统从此没有任何带宽限制”
```

这一点在对外表述时必须说清楚。

## 11. 最终建议

当前建议是：

1. 把 `route5` 作为“已达到 H_min >= 2 的主可行路线”写入总报告。
2. 对外表述时明确：
   - 它绕开了单光子死时间瓶颈；
   - 但仍受 CV 接收机带宽和工程噪声限制。
3. 如果只是当前阶段收口，可以先停止大规模继续搜索。
4. 如果后续要做更严谨的论文或实验推进，再补：
   - 更高节点数复核
   - 更多 target-input 复核
   - 有限尺寸与平均速率分析
   - 硬件误差模型

就当前目标而言，我的结论是：

```text
route5 已经足够证明这条混合路线是可行的，
而且它确实解决了之前最担心的“离散方案采样速率太慢”问题。
```
