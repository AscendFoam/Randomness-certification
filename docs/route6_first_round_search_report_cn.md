# Route6 第一轮参数搜索结果报告

## 1. 这轮搜索想回答什么

这份报告总结 `route6` 的第一轮系统参数搜索结果。

`route6` 的定义是：

- trusted input 允许多半径相干态字母表；
- 输出离散化采用 route5 风格的 axis-aligned IQ 矩形分箱；
- 概率直接使用解析高斯矩形积分；
- SDP 中 trusted states 采用 coherent alphabet 的 Gram 表示，而不是截断 Fock 近似。

这轮搜索想回答的核心问题不是“route6 理论上有没有意义”，而是更直接的一句：

```text
在实际数值搜索里，route6 能不能比旧 route3 更明显地把 formal H_min 往上推？
```

## 2. 先给结论

这轮第一阶段的结论可以概括为三点：

1. `route6` 确实能把 `raw_H_min` 推得很高，第一轮最优 raw 值普遍达到
   `raw_H_min ≈ 3.2013`。
2. 但 `raw_H_min` 很高并不意味着 formal 认证值也会高。
   在这轮 route6 搜索里，formal `H_min` 普遍明显回落。
3. 第一轮最好的 formal 结果是：
   - 非真空搜索最佳点：`H_min ≈ 0.60994`
   - 含真空搜索最佳点：`H_min ≈ 0.42507`

因此，第一轮 route6 的总体判断是：

```text
它目前还没有显示出比旧 route3 更强的正式认证优势，
更没有接近 route5 当前已经超过 2 bit 的水平。
```

## 3. 本轮实际完成了哪些搜索

这次真正落盘并完成的第一轮结果主要包括以下几类：

- 非真空 alphabet-search：
  [route6_first_round_nonvacuum_scs.json](../output/qrng_routes/route6_first_round_nonvacuum_scs.json)
- 含真空 alphabet-search：
  [route6_first_round_withvacuum_scs.json](../output/qrng_routes/route6_first_round_withvacuum_scs.json)
- 非真空 raw scout 摘要：
  [route6_first_round_raw_scout_nonvacuum_top10.json](../output/qrng_routes/route6_first_round_raw_scout_nonvacuum_top10.json)
- 两个代表性手动 formal 点：
  [route6_formal_r08_r12_2phase_6x2_scs.json](../output/qrng_routes/route6_formal_r08_r12_2phase_6x2_scs.json)
  [route6_formal_r04_r12_2phase_6x2_scs.json](../output/qrng_routes/route6_formal_r04_r12_2phase_6x2_scs.json)
- 一个关键的单半径对照点：
  [route6_formal_single_radius12_4phase_6x2_scs.json](../output/qrng_routes/route6_formal_single_radius12_4phase_6x2_scs.json)
  [route6_formal_single_radius12_4phase_6x2_mosek.json](../output/qrng_routes/route6_formal_single_radius12_4phase_6x2_mosek.json)
- 两个分区搜索 probe：
  [route6_probe_single_radius4phase_scs.json](../output/qrng_routes/route6_probe_single_radius4phase_scs.json)
  [route6_probe_two_radius2phase_scs.json](../output/qrng_routes/route6_probe_two_radius2phase_scs.json)

另外，我检查过后台进程，本轮 `route6` 任务已经全部结束，没有遗留运行中的 route6 程序。

## 4. 第一轮最关键的数值结果

### 4.1 非真空搜索最佳点

非真空 alphabet-search 的最佳结果来自：

- 半径：`[0.4]`
- 相位：`[0, π/2, π, 3π/2]`
- 本地态数：`4`
- 最佳分区：`6 x 2`
- `x_range = 1.5`
- `x_gamma = 1.5`
- `p` 轴只有符号分箱

对应结果见
[route6_first_round_nonvacuum_scs.json](../output/qrng_routes/route6_first_round_nonvacuum_scs.json)：

- `raw_best_H_min ≈ 3.201316`
- `formal H_min ≈ 0.609943`
- `p_guess ≈ 0.655223`

这说明在第一轮搜索里，最好的非真空点并不是“多半径字母表”，而是一个比较简单的单半径四相位字母表，而且这个半径还偏小。

### 4.2 含真空搜索最佳点

含真空 alphabet-search 的最佳结果来自：

- 半径：`[0.0, 0.8, 1.2]`
- 相位：`[0, π]`
- 本地态数：`5`
- 最佳分区同样是 `6 x 2`
- 同样落在 `x_range = 1.5`, `x_gamma = 1.5`

对应结果见
[route6_first_round_withvacuum_scs.json](../output/qrng_routes/route6_first_round_withvacuum_scs.json)：

- `raw_best_H_min ≈ 3.201316`
- `formal H_min ≈ 0.425072`
- `p_guess ≈ 0.744801`

和非真空最佳点相比，虽然 raw 值同样很高，但 formal 认证值更低。

### 4.3 手动 probe 与对照点

为了把“单半径 vs 多半径”的趋势看得更清楚，我还单独跑了几个代表性 formal 点。

1. 两半径两相位 `([0.8, 1.2], {0, π})`
   对应
   [route6_probe_two_radius2phase_scs.json](../output/qrng_routes/route6_probe_two_radius2phase_scs.json)
   得到：
   - `raw_H_min ≈ 3.112300`
   - `formal H_min ≈ 0.389428`

2. 两半径两相位 `([0.8, 1.2], {π/2, 3π/2})`
   对应
   [route6_formal_r08_r12_2phase_6x2_scs.json](../output/qrng_routes/route6_formal_r08_r12_2phase_6x2_scs.json)
   得到：
   - `raw_H_min ≈ 3.201316`
   - `formal H_min ≈ 0.251895`

3. 两半径两相位 `([0.4, 1.2], {π/2, 3π/2})`
   对应
   [route6_formal_r04_r12_2phase_6x2_scs.json](../output/qrng_routes/route6_formal_r04_r12_2phase_6x2_scs.json)
   得到：
   - `raw_H_min ≈ 3.201316`
   - `formal H_min ≈ 0.144645`

4. 单半径四相位 `([1.2], {0, π/2, π, 3π/2})`
   对应
   [route6_formal_single_radius12_4phase_6x2_scs.json](../output/qrng_routes/route6_formal_single_radius12_4phase_6x2_scs.json)
   和
   [route6_formal_single_radius12_4phase_6x2_mosek.json](../output/qrng_routes/route6_formal_single_radius12_4phase_6x2_mosek.json)
   得到：
   - `SCS: H_min ≈ 0.04903`
   - `MOSEK: H_min ≈ 0.04931`

这个对照点很重要，因为它说明：

```text
不是“单半径四相位”这个结构本身好，
而是“单半径 + 半径不要太大”这个方向更稳。
```

半径从 `0.4` 拉到 `1.2` 后，formal `H_min` 反而几乎塌掉。

## 5. 这一轮最清楚的现象

### 5.1 最佳 IQ 分箱几乎固定

无论是 raw scout、非真空搜索还是含真空搜索，最优分区都高度集中在同一类方案上：

- `num_x_bins = 6`
- `num_p_bins = 2`
- `x_range = 1.5`
- `x_gamma = 1.5`
- `p_bounds = [-∞, 0, +∞]`

这说明在 route6 当前设置下，最值得关注的不是“分区有没有选错”，而是：

```text
trusted alphabet 本身对 formal SDP 的约束到底强不强。
```

### 5.2 `raw_H_min` 高度一致，但 `formal H_min` 差别很大

本轮大量候选的 `raw_best_H_min` 都在
`≈ 3.20` 附近，差别非常小。

但是 formal `H_min` 可以从

- `≈ 0.61`

一路掉到

- `≈ 0.05`

这说明 route6 当前最大的现象不是“输出分布不够宽”，而是：

```text
很多 alphabet 在 raw 层面看起来非常平，
但一进 formal SDP，Eve 还能利用剩余自由度把猜测概率拉高。
```

这和之前对 route3 / route4 看到的“raw 和 formal 之间存在明显落差”是一致的。

## 6. 和旧 route3、route5 的比较

### 6.1 和旧 route3 的比较

此前对 route3 的重新分析文档
[non_route2_hmin_target_reanalysis_cn.md](../docs/non_route2_hmin_target_reanalysis_cn.md)
里给出的代表性结果是：

- `certified H_min ≈ 0.6671`

也就是说，旧 route3 虽然离 `2` 很远，但它目前的最好 formal 值仍然高于这轮 route6 的最好值 `≈ 0.6099`。

所以从这轮第一批结果看：

```text
route6 还没有表现出“比旧 route3 明显更强”的 formal 优势。
```

### 6.2 和 route5 的比较

route5 目前的代表性结果见
[route5_principle_and_feasibility_cn.md](../docs/route5_principle_and_feasibility_cn.md)：

- 自由搜索最好点：`H_min ≈ 2.11639`
- 受限光强 `[0,80,160]`：`H_min ≈ 2.10102`

和这些结果相比，route6 第一轮的最好 formal 结果 `≈ 0.61` 仍然差得非常远。

因此第一轮 route6 更适合被理解为：

```text
沿着导师建议做出的一个新方向原型，
而不是已经显示出能取代 route5 的高熵方案。
```

## 7. 我对结果的解释

### 7.1 为什么小半径反而更好

第一轮最佳点是单半径 `r = 0.4` 四相位，而不是更大的 `r = 1.2`。

一个直观解释是：

- 半径太大时，各输入态在解析高斯前端下更容易变得“分布上非常可区分”；
- 这会把 raw 熵做得很好看；
- 但在 formal SDP 里，Eve 同样能更容易利用这些强可区分结构来提高猜测概率；
- 最终认证值不一定升，反而可能下降。

所以对 route6 而言，

```text
更亮的 coherent alphabet 不等于更好的 formal 随机性。
```

### 7.2 为什么多半径没有自动带来提升

从设计直觉上说，多半径应该比单半径更强，因为它给 trusted input 带来了更多方向。

但第一轮结果表明，这个提升没有自动转化为更高的 formal `H_min`。

我认为原因主要有两层：

1. 这轮最优点的本地态数仍然不大。
   - 非真空最佳点是 `4` 个本地态；
   - 含真空最佳点是 `5` 个本地态。
   这仍然是一组相对较小的 trusted alphabet。
2. 虽然 route6 用的是 Gram 表示，不再受 Fock cutoff 近似限制，
   但 SDP 的可认证强度最终仍取决于这组输入态本身能提供多少“不可伪装的测量约束”。

也就是说，route6 的新表示法修掉了 route3 的一部分数值建模问题，
但没有自动修掉“trusted input 约束强度仍然不足”的结构问题。

### 7.3 为什么 raw 很高但 formal 没跟上

这是这轮最重要的总体认识。

route6 当前最显著的现象是：

```text
解析高斯 bin 概率可以很容易给出非常平的离散输出分布，
但这并不等于这些离散输出已经被单设备 SDP 牢固地认证了。
```

因此，第一轮 route6 的结果更像是在提醒我们：

- 解析积分概率本身没有问题；
- Gram 表示本身也没有问题；
- 真正的难点仍然是“trusted alphabet 到底能给 SDP 多强的约束”。

## 8. 结论

这轮 route6 第一轮参数搜索可以总结为：

1. route6 已经作为一个完整新原型跑通。
2. 它的最佳 raw 结果很高，说明“多半径 + IQ 矩形离散化 + 解析高斯概率”在分布层面很有潜力。
3. 但 formal 认证值目前最好只有 `H_min ≈ 0.60994`，还没有超过旧 route3 代表性结果 `≈ 0.6671`，更远低于 route5 的 `> 2`。
4. 第一轮最优点反而是一个较小半径的单半径四相位 alphabet，而不是更复杂的多半径字母表。
5. 因此，当前还不能把 route6 视为“已经明显优于 route3 的升级版”，更不能把它视为 route5 的替代者。

更准确的表述是：

```text
route6 已经证明这条新建模路线可计算、可搜索、可认证；
但在第一轮系统搜索里，它还没有显示出 formal H_min 上的明显优势。
```

## 9. 下一步建议

如果后续还要继续推进 route6，我建议优先做下面三件事，而不是盲目扩大搜索范围。

1. 固定第一轮已知最优分区族
   - 继续以 `6 x 2`, `x_range = 1.5`, `x_gamma = 1.5` 为主线，
   - 不再把大量算力浪费在分区扫描上。

2. 重点扩大“本地态数”，而不是只换半径
   - 第一轮最优点的 `num_local_states` 仍然只有 `4` 或 `5`；
   - 如果 route6 真要超越 route3，下一轮更值得尝试 `6 - 8` 个本地态的 alphabet。

3. 重点检查“约束强度”而不是只看 raw 熵
   - route6 第一轮最明显的问题是 `raw` 和 `formal` 脱钩；
   - 因此第二轮应直接围绕 formal `H_min` 排序，而不是再被 raw `3.2 bit` 的表面值误导。

