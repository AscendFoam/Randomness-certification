# Route4-ex 在 `Probability.mat` 的 `[100,120,140]` 窗口上的一轮结果简报

## 1. 目的

这份短报告总结了 `route4-ex` 在外部概率表
`src/matlab/Probability.mat`
上的一轮局部扫描结果。当前只聚焦于此前已经发现可行的三输入窗口：

- 选取光强行：`[100,120,140]`
- 对应 `row_indices = [5,6,7]`

本轮问题是：

1. 在这个已知可行窗口内，`H_min` 能否从此前的约 `0.0596` 继续抬高；
2. 哪些参数方向更有希望：
   - `num_outputs`
   - 相位图样
   - `max_abs_alpha`
   - `cutoff`

## 2. 扫描口径

本轮都使用 `route4-ex external` 的 formal full-primal 认证值作为主要判断指标。

共同设置：

- 概率数据：`src/matlab/Probability.mat`
- 变量名：`Probability`
- 输入强度：`[100,120,140]`
- 行索引：`[5,6,7]`
- 概率粗粒化：由外部 `256` 维直方图压缩为少量离散输出
- 求解器：`SCS`

比较过的主要结果文件：

- `output/qrng_routes/route4_ex_external_probabilitymat_stageA_outputs3_scan_100_120_140.json`
- `output/qrng_routes/route4_ex_external_probabilitymat_stageC_outputs2_scan_100_120_140.json`
- `output/qrng_routes/route4_ex_external_probabilitymat_outputs2_refine_alpha062_068_cutoff6.json`

## 3. 结果概览

### 3.1 原始基线

此前已知可行点是：

- 相位：`[0, π/2, π]`
- `max_abs_alpha = 0.9`
- `cutoff = 8`
- `num_outputs = 3`
- `H_min ≈ 0.05957`

这一点来自更早的小规模可行性检查，只能说明 `route4-ex` 在 `Probability.mat` 上不是完全不可行。

### 3.2 在 `3` 输出主线上的改进

在 `num_outputs = 3` 主线上，当前最好点提升到了：

- 相位图样：`0_pi2_pi`
- `max_abs_alpha = 0.75`
- `cutoff = 6`
- `H_min ≈ 0.09198`

对应文件：
`output/qrng_routes/route4_ex_external_probabilitymat_stageA_outputs3_scan_100_120_140.json`

这说明即使仍然坚持 `3` 输出，也能比原来的 `0.0596` 提高不少。

### 3.3 更关键的现象：`2` 输出优于 `3` 输出

继续扫描后发现，当前窗口里更优的不是更细的输出分箱，而是更粗的 `2` 输出。

在 `2` 输出 broad scan 中，当前最好点是：

- 相位图样：`0_pi3_2pi3`
- `max_abs_alpha = 0.8`
- `cutoff = 4`
- `num_outputs = 2`
- `H_min ≈ 0.25022`

对应文件：
`output/qrng_routes/route4_ex_external_probabilitymat_stageC_outputs2_scan_100_120_140.json`

这已经远高于 `3` 输出主线的 `0.09198`。

### 3.4 进一步精修后的当前最好点

在更贴近当前最优谷底的精修里，沿着

- `num_outputs = 2`
- 相位：`[0, π/2, π]`
- `cutoff = 6`

继续细扫 `max_abs_alpha`，得到目前全局最好点：

- 相位图样：`0_pi2_pi`
- `max_abs_alpha = 0.63`
- `cutoff = 6`
- `num_outputs = 2`
- `distribution-only H_min ≈ 0.28055`
- `formal H_min ≈ 0.27622`

对应文件：
`output/qrng_routes/route4_ex_external_probabilitymat_outputs2_refine_alpha062_068_cutoff6.json`

这意味着当前 formal 值已经非常接近 distribution-only 上界，说明在这个局部窗口里，认证损失已经被压得比较低。

## 4. 当前最重要的结论

### 4.1 `num_outputs = 2` 是当前主线

在 `[100,120,140]` 这个窗口里，当前证据清楚地表明：

- `2` 输出比 `3` 输出更有希望；
- 更细分箱并没有带来更高的 formal `H_min`；
- 相反，较粗的输出把约束“集中”得更强，从而更有利于 full-primal 认证。

所以接下来的主线应切到：

- `num_outputs = 2`

### 4.2 最优 `max_abs_alpha` 不在更大值，而在一个窄窗口

在当前最优相位线 `0_pi2_pi` 上，`max_abs_alpha` 的趋势不是“越大越好”：

- `0.62` 时已经变成 infeasible；
- `0.63` 最好；
- `0.64, 0.65, 0.66, ...` 会逐步回落。

因此当前局部最优谷大致位于：

- `max_abs_alpha ≈ 0.63`

### 4.3 `cutoff` 在这里不是主导瓶颈

在 `2` 输出、`max_abs_alpha ≈ 0.65` 一带，`cutoff = 4, 6, 8` 的 formal 结果差别很小，说明当前最主要的增益并不是来自继续增大 Fock 截断，而更像是来自：

- 合适的输入态半径；
- 合适的相位图样；
- 更有利的输出粗粒化。

## 5. 为什么 `2` 输出反而更好

目前的理解是：

1. `Probability.mat` 原始数据本身是固定实验条件下的计数直方图。
2. 当我们把它 coarse-grain 成更多输出时，表面上分布会更细，但同时 full-primal 需要满足的约束也会更分散、更难兼容。
3. 在当前这组三输入的非对角 trusted states 下，`2` 输出似乎更容易把“可区分性”集中到少数几个关键统计特征上，因此 formal `H_min` 更高。

这不意味着“输出越少越好”是普适规律，而只是说明：

- 在当前 `[100,120,140]` 这个窗口里，
- 对当前这组三输入映射与概率表而言，
- `2` 输出恰好更适合 full-primal 认证。

## 6. 与目标 `H_min >= 1` 的距离

这轮结果已经把 `H_min` 从约 `0.0596` 提高到了约 `0.2762`，提升是显著的。

但也要如实说明：

- 当前还远没有到 `H_min >= 1`；
- 目前最好的结果大约是 `0.2762`；
- 因此 `route4-ex + Probability.mat + 当前三输入窗口` 虽然已经展示出改进空间，但还没有显示出“很快就能冲到 1 bit”的信号。

## 7. 后续精修补充：偏置 `q_selected` 的结果

在完成上面这轮均匀权重扫描后，又额外做了一轮 follow-up 精修：

- 固定相位：`0_pi2_pi`
- 固定 `num_outputs = 2`
- 固定 `cutoff = 6`
- 固定 `max_abs_alpha = 0.63`
- 扫描不同的 `q_selected`

对应结果文件：

- `output/qrng_routes/route4_ex_external_probabilitymat_outputs2_qbias_alpha063_cutoff6.json`

最重要的结论是：`q_selected` 的影响非常大。

几个代表点如下：

- 均匀权重 `q = [1,1,1]` 时，`H_min ≈ 0.27622`
- 偏向第一个输入 `q = [2,1,1]` 时，`H_min ≈ 0.40405`
- 更强偏置 `q = [5,1,1]` 时，`H_min ≈ 0.58683`
- 极端单输入 `q = [1,0,0]` 时，`H_min ≈ 0.87328`

而另外两个单输入情形明显差很多：

- `q = [0,1,0]` 时，`H_min ≈ 0.10007`
- `q = [0,0,1]` 时，`H_min ≈ 0.00185`

这说明在当前 `[100,120,140]` 窗口里，三个输入并不对称：

- 第一个输入明显是“好输入”；
- 第二个输入中等；
- 第三个输入最差。

因此，均匀 `q_selected` 会把好输入和差输入平均掉，从而显著拉低 formal `H_min`。

随后又围绕最佳偏置方向 `q = [1,0,0]` 做了一轮 `max_abs_alpha` 精修：

- 文件：`output/qrng_routes/route4_ex_external_probabilitymat_outputs2_alpha_scan_q100_alpha060_066.json`

结果表明：

- `max_abs_alpha = 0.63` 仍然是当前最好点；
- 对应 `H_min ≈ 0.87328`
- 当 `max_abs_alpha` 从 `0.635` 到 `0.66` 增大时，结果会逐步回落；
- 当 `max_abs_alpha <= 0.625` 时，full-primal 重新变成 infeasible

所以到目前为止，当前最强结果已经不再是 `0.276`，而是：

- 相位：`0_pi2_pi`
- `num_outputs = 2`
- `cutoff = 6`
- `max_abs_alpha = 0.63`
- `q_selected = [1,0,0]`
- `formal H_min ≈ 0.87328`

## 8. 下一步建议

下一轮最值得继续做的是：

1. 固定主线
   - `num_outputs = 2`
   - 相位优先看 `0_pi2_pi`
   - `max_abs_alpha` 在 `0.63` 附近做更细精修

2. 开始扫偏置输入分布 `q_selected`
   - 当前大多数结果都采用均匀权重
   - full-primal 的目标函数直接依赖 `q_selected`
   - 因此偏置 `q_selected` 可能进一步把 formal `H_min` 往上推

3. 继续保留一个对照相位族
   - `0_pi3_2pi3`
   - 因为它在 broad scan 中给出了很高的 `0.2502`

## 9. 一句话总结

在 `Probability.mat` 的 `[100,120,140]` 三输入窗口上，`route4-ex` 已经从最初的
`H_min ≈ 0.0596`
先提升到均匀权重下最好的
`H_min ≈ 0.2762`，
再在偏置 `q_selected` 后进一步提升到
`H_min ≈ 0.8733`。

当前最清晰的结论是：

- 主线应该切到 `2` 输出；
- `max_abs_alpha` 最优区在 `0.63` 左右；
- `q_selected` 不是次要细节，而是当前最关键的放大器；
- 在目前这组数据上，如果允许把生成权重明显偏向第一个输入，那么 `route4-ex` 已经可以接近 `H_min = 1` 的目标。 
