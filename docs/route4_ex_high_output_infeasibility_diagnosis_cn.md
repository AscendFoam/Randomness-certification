# Route4-ex 中高熵 `3/4` 输出边界为何会 formal infeasible

## 1. 问题是什么

在 `route4-ex` 里，我们已经看到一个很突出的现象：

- 如果只看目标输入那一行的 coarse-grained 分布，某些 `3/4` 输出边界看起来非常好；
- 但把同一组边界放回 formal full-primal SDP 后，结果不是很低，就是直接 `infeasible`。

这份短文档总结当前证据，并解释这更像是什么问题。

## 2. 先区分两种“熵”

这里必须先区分：

1. `distribution-only H_min`
   - 只看目标输入那一行 coarse-grained 分布的最大桶概率；
   - 它相当于“如果完全不考虑多输入共同约束，单看这一行有多平”。

2. formal `H_min`
   - 要求存在一套统一的测量/策略变量；
   - 同时解释所有 trusted inputs 对应的概率约束；
   - 然后再求 adversary 的 guessing probability。

因此：

- `distribution-only` 高，不等于 formal 一定高；
- 特别是在多输入 SDP 里，两者经常会分离。

## 3. 已有证据一：目标行自身的高熵边界确实存在

对 `Probability.mat` 的第一个目标行（光强 `100` 对应的那一行），我们已经做过边界扫描：

- [`../output/qrng_routes/route4_ex_probabilitymat_row100_edge_search_summary.json`](../output/qrng_routes/route4_ex_probabilitymat_row100_edge_search_summary.json)

其中代表性的高熵边界是：

- `3` 输出：`[0,121,132,256]`
  - `distribution-only H_min ≈ 1.538`
- `4` 输出：`[0,118,127,136,256]`
  - `distribution-only H_min ≈ 1.924`

这说明“高熵边界不存在”不是问题所在。恰恰相反，问题正是：

- 单看目标输入行，这些边界太好了；
- 但它们并不容易和其它输入一起被同一套 trusted-state 模型兼容。

## 4. 已有证据二：subset 诊断说明，三输入约束是主要塌陷点

我们已经把这两组高熵边界拿出来，做了单输入、两输入、三输入的 subset 诊断：

- [`../output/qrng_routes/route4_ex_high_output_subset_diagnosis.json`](../output/qrng_routes/route4_ex_high_output_subset_diagnosis.json)

结论非常清楚。

### 4.1 `3` 输出高熵边界 `[0,121,132,256]`

- 单输入 `[100]`
  - `distribution-only H_min ≈ 1.538`
  - formal `H_min ≈ 0`
- 两输入 `[100,120]`
  - formal `H_min ≈ 0.218`
- 两输入 `[100,140]`
  - formal `H_min ≈ 0.280`
- 三输入 `[100,120,140]`
  - `infeasible`

### 4.2 `4` 输出高熵边界 `[0,118,127,136,256]`

- 单输入 `[100]`
  - `distribution-only H_min ≈ 1.924`
  - formal `H_min ≈ 0`
- 两输入 `[100,120]`
  - formal `H_min ≈ 0.233`
- 两输入 `[100,140]`
  - formal `H_min ≈ 0.294`
- 三输入 `[100,120,140]`
  - `infeasible`

这组结果说明了两件事：

1. 单输入时，即便目标行 coarse-graining 很平，formal 认证也几乎给不出熵。
2. 加入第二个输入后，formal 值反而上升；但加入第三个输入后，整个问题塌成 infeasible。

所以问题不是一句“输出太多了”就能解释的。更准确地说：

- 单输入太弱，无法真正约束 adversary；
- 三输入又太强，把当前 trusted-state 模型和高熵边界同时逼得不相容。

## 5. 已有证据三：不是所有 `3` 输出都不行

联合兼容性搜索已经说明：

- 不是“`3` 输出天然不行”；
- 而是“某些高熵边界不行”。

已有结果文件：

- [`../output/qrng_routes/route4_ex_joint_compat_search_round1.json`](../output/qrng_routes/route4_ex_joint_compat_search_round1.json)

其中一个代表性可行点是：

- 窗口 `[100,120,140]`
- `3` 输出等覆盖边界 `[0,85,170,256]`
- 相位 `0_pi3_2pi3`
- `max_abs_alpha = 0.66`
- `q = [1,1,1]`
- formal `H_min ≈ 0.247`

这说明：

- `3` 输出 formal 可行并非不可能；
- 但“让目标行单独很平”的高熵窄边界，反而更容易和多输入约束冲突。

## 6. 当前最合理的物理解释

根据现有证据，当前问题更像是下面这三件事叠加。

### 6.1 高熵窄边界过度贴合了目标行

像

- `[0,121,132,256]`
- `[0,118,127,136,256]`

这类边界，本质上是在围着目标行最“好看”的局部峰/谷去切。

对单个目标行来说，这当然能把最大桶概率压低；
但对另外两行来说，这些边界未必还能保持同样的几何关系。

结果就是：

- 目标行看起来很随机；
- 但三行合起来却不再像是同一个物理测量下得到的结果。

### 6.2 当前 trusted-state 模型可能过硬

当前 `route4-ex external` 用的是：

- 外部概率表中的若干行；
- 一组精确截断相干态作为 trusted inputs；
- 然后要求这组 inputs 共同解释同一张 coarse-grained 概率表。

对应代码入口：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

实例构造在：

- [`../src/python/qrng_routes/route4_ex/prototype.py#L490-L540`](../src/python/qrng_routes/route4_ex/prototype.py#L490-L540)

而目前最常用的输入映射还是：

- 强度 `I` 经 `sqrt(I/I_max)` 映成半径；
- 再乘上预设 phase pattern。

如果实验真实输入和这套理想化相干态模型存在偏差，那么高熵窄边界就更容易首先暴露出“不兼容”。

### 6.3 第三个输入很可能是主要冲突源

subset 诊断最直观地表明：

- 单输入不是 infeasible，而是 formal 值接近 0；
- 两输入不是 infeasible，而是还能给出 `0.2-0.29 bit`；
- 三输入才塌成 infeasible。

所以目前最像“主犯”的，不是输出数本身，而是：

- 第三个输入加入后，
- 要求同一套 trusted-state + 测量变量同时解释三行高熵窄边界，
- 这个要求已经过于紧。

## 7. 这意味着下一步该怎么改

如果继续沿着当前主线推进，最值得做的不是再抠 `2` 输出峰值，而是直接对下面两类改法做定向搜索。

### 7.1 改输入窗口

也就是不默认固定在 `[100,120,140]`，而是比较：

- `[80,100,120]`
- `[100,120,140]`
- `[100,140,160]`
- `[120,140,160]`

看哪一个三输入窗口更容易和高熵 `3/4` 输出边界兼容。

### 7.2 改 trusted-state 模型

也就是不再死守

- “半径严格按 `sqrt(I)` 缩放”

这一个模型，而是放松成：

- 仍保持三个输入按半径递增；
- 但半径可以自由搜索；
- 相位图样也允许比较。

这类搜索已经单独写成脚本：

- [`../src/python/qrng_routes/route4_ex/high_output_model_window_search.py`](../src/python/qrng_routes/route4_ex/high_output_model_window_search.py)

它的目的不是再优化 `2` 输出，而是直接回答：

- 高熵 `3/4` 输出的 formal infeasible，究竟更像是“窗口不对”，还是“输入模型太硬”。

## 8. 当前阶段的结论边界

在写这份文档的同时，还启动了一轮更直接的定向搜索：

- [`../src/python/qrng_routes/route4_ex/high_output_model_window_search.py`](../src/python/qrng_routes/route4_ex/high_output_model_window_search.py)
- 当前输出文件：
  - [`../output/qrng_routes/route4_ex_high_output_model_window_search_q100.json`](../output/qrng_routes/route4_ex_high_output_model_window_search_q100.json)

截至这份文档写下时，已落盘的早期结果已经给出一个很重要的信号：

- 在窗口 `[100,120,140]`
- 高熵 `3` 输出边界仍取 `[0,121,132,256]`
- 相位仍取 `0_pi2_pi`
- `q=[1,0,0]`

时，

- `max_abs_alpha = 0.60, 0.63` 都仍然 `infeasible`
- 但 `max_abs_alpha = 0.69` 已经变成 `optimal`
- 且 formal `H_min ≈ 0.687`

这说明：

1. “高熵 3 输出边界”不是绝对不可能。
2. 当前的 infeasible 现象至少部分取决于 trusted-state 模型所处的参数区间。
3. 问题更像是“当前三输入模型太硬、工作点太窄”，而不一定是“只要是高熵 3 输出就必死”。

目前可以较有把握地说：

1. 高熵 `3/4` 输出边界确实存在。
2. 这些边界在单输入 formal 问题里几乎不给熵。
3. 加第二个输入后可行性和 formal 值会上升。
4. 加第三个输入后，在当前窗口和当前 trusted-state 主线下容易直接变成 infeasible。
5. 因此，主问题不在于“原始概率表没有高熵结构”，而在于“高熵边界与多输入 trusted-state 约束不兼容”。

但目前还不能最终断言：

- 根因是输入窗口选错；
- 还是 `sqrt(I)` 映射过硬；
- 还是两者同时在起作用。

这正是下一轮定向搜索要回答的问题。

## 9. 一句话总结

`route4-ex` 里高熵 `3/4` 输出边界的 formal infeasible，不像是“输出越多越差”这么简单；更像是：

- 这些边界对目标输入过于友好，
- 但无法被当前三输入 trusted-state 模型同时解释。

因此下一步主线应该转向：

- 改输入窗口；
- 放松 trusted-state 半径模型；

而不是继续围着现在的 `2` 输出最优点反复精修。
