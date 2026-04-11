# Route4-ex 两个关键问题的说明

## 1. 说明目的

本文档集中回答两个问题：

1. `route4-ex` 当前最优约 `H_min ≈ 1.54` 的结果中，`rho_x^{diag}` 到底对应哪几个输入态？它们是不是“强度为 `100,120,140` 的三个相干态”？
2. `route4-ex-constrained` 和 `route4-ex` 最终结果里使用的 `M=6, N=3` 会不会太小？如果把 `N` 提高到 `8` 左右、把 `M` 也继续增大，是否有希望把最小熵推到 `2 bit` 以上？

---

## 2. 问题一：`1.54 bit` 最优点中的 `rho_x^{diag}` 到底是什么

### 2.1 先给结论

`route4-ex` 当前最优约 `H_min ≈ 1.54395` 的结果，确实使用了 `Probability.mat` 中标签为 `[100,120,140]` 的三行实验概率数据；但进入 full primal 的 trusted input states 并不是“直接把 `100,120,140` 当成三个相干态强度构造出来”的。

更准确地说：

1. 概率表部分使用的是窗口 `[100,120,140]` 对应的三行实验数据；
2. trusted input 部分使用的是另外指定并精修得到的三个截断相干态
   $$
   \rho_x = |\alpha_x\rangle\langle\alpha_x| ;
   $$
3. 这三个相干态与三个标签一一对应，但它们的振幅不是 `100,120,140` 本身。

对应当前最优稳定点，三组复振幅大约为

$$
\alpha_1 \approx 0.537954,
\qquad
\alpha_2 \approx 0.662046\, i,
\qquad
\alpha_3 \approx -0.717954.
$$

后续可参考：

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)
- [`../output/qrng_routes/route4_ex_residual_diag_q419over1024.json`](../output/qrng_routes/route4_ex_residual_diag_q419over1024.json)

### 2.2 `rho_x` 和 `rho_x^{diag}` 的区别

`route4-ex` 和原始 `route4` 的一个核心区别就在于：

- 原始 `route4` 只使用 Fock 对角输入；
- `route4-ex` 使用完整的 non-diagonal coherent trusted states。

程序里这一点对应：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

其中 `build_coherent_density_matrices(...)` 明确构造的是

$$
\rho_x = |\alpha_x\rangle\langle\alpha_x|,
$$

而不是只取对角部分。

不过，如果单独问“这个点对应的 `rho_x^{diag}` 是什么”，那答案是：

$$
\rho_x^{\mathrm{diag}}(n)
\propto
e^{-|\alpha_x|^2}\frac{|\alpha_x|^{2n}}{n!},
\qquad n=0,\dots,M-1,
$$

也就是这三个完整相干态在 Fock 基下的对角投影。

### 2.3 这三个 `rho_x^{diag}` 对应的平均光子数是多少

由上面的三个振幅可得：

$$
|\alpha_1|^2 \approx 0.28939,
\qquad
|\alpha_2|^2 \approx 0.43830,
\qquad
|\alpha_3|^2 \approx 0.51546.
$$

因此，这个最优点对应的三组 `rho_x^{diag}`，本质上是三个 Poisson 型对角分布，其参数分别接近

$$
\mu_1 \approx 0.28939,\quad
\mu_2 \approx 0.43830,\quad
\mu_3 \approx 0.51546.
$$

所以，不能把它表述成“`1.54` 结果就是强度 `100,120,140` 的三个相干态”；更准确的表述应当是：

> `1.54 bit` 结果使用了 `[100,120,140]` 这三行实验概率数据，但 SDP 中的 trusted coherent states 是另外选定并精修的一组三个复振幅，它们分别对应这三个标签。

### 2.4 这件事为什么有意义

这正是 `route4-ex` 的核心设计思想：

- 实验观测概率仍然来自原始 `Probability.mat`；
- 但 trusted input 模型不再退回到原始 `route4` 的纯对角 Poisson 输入，而是升级成完整相干态输入；
- 正式提升到 `1.54 bit` 的关键，恰恰来自这一“non-diagonal trusted states + same probability table”的组合。

---

## 3. 问题二：`M=6, N=3` 会不会太小

### 3.1 先给结论

1. `M=6` 对当前 `route4-ex` / `route4-ex-constrained` 主结果而言，并不算明显偏小；
2. `N=3` 也不是随便取的保守小值，而更像是在当前 `Probability.mat + 3输入 trusted-state` 主线下，一个 formal 可行性和认证值之间的折中最优区间；
3. 单纯把 `N` 提高到 `8` 左右、再把 `M` 增大，并没有明确证据表明就能把 formal `H_min` 推到 `2 bit` 以上；
4. 相反，从已有证据看，这样做更可能先带来规模爆炸和 formal 不可行，而不是自然跃升到 `2 bit`。

这里也参考了外部AI给出的一个分析框架。其核心思路整体上是合理的，尤其是以下三点：

1. 对当前这组三个小振幅 trusted coherent states 而言，`M=6` 在物理表示上已经基本够用；
2. 真正更危险的不是 `M` 太小，而是 `N` 增大后策略数和 full primal 规模指数爆炸；
3. `route4-ex` 的正式提升核心来自 non-diagonal trusted input + full primal，而不是“输出 bin 越多越好”。

### 3.2 `M` 和 `N` 在这两条路线中的确切含义

这里区分两个参数：

- `M` 是 Fock 空间截断维数，也就是 trusted coherent state 在数值上只保留到
  $$
  |0\rangle,\dots,|M-1\rangle ;
  $$
- `N` 是 coarse-graining 之后的输出数，也就是把 `256` 维原始直方图压成多少个离散输出区间。

在 [`../src/matlab/guessprobprimal_route4_ex_constrained.m`](../src/matlab/guessprobprimal_route4_ex_constrained.m) 里，这两个量对应得很直接：

- `M = 6`
- `N = length(custom_edges)-1 = 3`

因此：

- `M` 不是采样数，也不是输入数；
- `N` 不是光强数，也不是策略数。

### 3.3 为什么说 `M=6` 基本是合理的

对当前 `route4-ex` 最优点而言，三组 trusted 振幅模长约为

$$
0.537954,\ 0.662046,\ 0.717954.
$$

因此其平均光子数只有

$$
0.28939,\ 0.43830,\ 0.51546.
$$

这意味着相干态在 Fock 基上的主要权重集中在很低的光子数上。对这三组参数，截断到 `n=0,\dots,5` 之后，Poisson 尾部漏掉的总概率大约只有：

- `6.65e-7`
- `6.54e-6`
- `1.73e-5`

换句话说，`M=6` 已经把这些 trusted coherent states 的主要概率质量基本吃住了。

如果想把单纯的截断尾部误差进一步压到极小，比如接近 `10^{-10}` 量级，那么确实可能需要把 `M` 提高到大约 `9-11` 左右。但这类改进更像是“数值表示更保守、更干净”，而不是“物理模型本身发生显著变化”。对当前 `route4-ex` 来说，影响正式 `H_min` 的主导误差来源，并不太可能是这一级别的截断尾部，而更可能是：

1. trusted-state 模型与实验概率表之间的兼容性；
2. coarse-graining 结构；
3. SDP 在窄稳定前沿附近的可行性与病态程度。

因此：

- 把 `M` 从 `6` 增到 `8` 或 `10`，当然会更保险；
- 但它更像是“稳健性复核”或“数值更保守的表示”，而不像是一个能把 `1.54` 一口气抬到 `2.0+` 的决定性杠杆。

### 3.4 为什么说 `N=3` 也不是随便取小了

`route4-ex` 的 formal full primal 难度对 `N` 极其敏感。若输入数记为 `D`，则策略数是

$$
N^{D+1},
$$

而 full primal 的 Hermitian 标量规模大致与

$$
N^{D+2} M^2
$$

同阶增长。

对当前主线，`D=3`。于是有：

- `N=3, M=6` 时，
  - `num_strategies = 3^4 = 81`
  - `hermitian_scalar_count = 8748`
- `N=8, M=6` 时，
  - `num_strategies = 8^4 = 4096`
  - `hermitian_scalar_count = 1,179,648`
- `N=8, M=10` 时，
  - `hermitian_scalar_count = 3,276,800`

也就是说，仅仅把 `N` 从 `3` 提高到 `8`，full primal 的变量规模就会暴涨约 `135` 倍；若再同时增大 `M`，规模会进一步急剧上升。

若直接拿当前最优主线和一个“更大 `N`、更大 `M`”的典型点比较，则：

- 基线：`D=3, N=3, M=6`
  - `num_strategies = 81`
  - `hermitian_scalar_count = 8748`
- 放大型：`D=3, N=8, M=10`
  - `num_strategies = 4096`
  - `hermitian_scalar_count = 3,276,800`

后者相对于前者，Hermitian 标量规模约增加

$$
\frac{3,276,800}{8,748} \approx 375
$$

倍。这已经不是“再多花一点时间就行”的量级，而是会明显改变问题是否还能现实求解。

这说明一个事实：

> 对 `route4-ex` 来说，增大 `N` 不是“微调”，而是会显著改变求解难度和数值病态程度的结构性操作。

### 3.5 更大的 `N` 为什么不一定更好

直觉上，更多输出 bin 似乎意味着：

- 输出分布更细；
- raw 熵可能更高；
- 因而 formal 熵也可能更高。

但 `route4-ex` 的已有结果并不支持“输出越多越好”这个简单结论。

现有诊断文档已经多次表明：

- 某些 `3/4` 输出高熵边界在 raw 层面看起来很好；
- 但一旦送回 formal full primal，就可能直接 `infeasible`，或者非常不稳定。

可参考：

- [`./route4_ex_high_output_infeasibility_diagnosis_cn.md`](./route4_ex_high_output_infeasibility_diagnosis_cn.md)
- [`./route4_ex_stage_report_cn.md`](./route4_ex_stage_report_cn.md)

这说明，在当前 `Probability.mat + 3输入 trusted-state` 主线上，增加输出数的后果往往是：

1. 统计分布表面上更“花”；
2. 但多输入兼容性约束更难满足；
3. formal 问题更容易失稳或不可行。

这一点也可以换一个角度理解。统计匹配约束的数量随 `N` 线性增长：

$$
\mathrm{Tr}(\rho_x \bar M_c) = p(c|x),
\qquad \forall x,c,
$$

其方程数是

$$
D \times N.
$$

当前 `D=3, N=3` 时共有 `9` 个统计约束；若升到 `N=8`，则变成 `24` 个。约束越多，并不自动意味着“认证更强”；它同样意味着可行域更小。如果实验概率和 trusted input 模型之间本来就只是在一条很窄的兼容边界上，那么更细的分箱更可能先把问题压成 `infeasible`，而不是自动给出更高的 `H_min`。

因此，`N=3` 当前更像是一个“formal 可行且值高”的工作点，而不只是一个“没来得及往上加”的小参数。

### 3.6 对 `route4-ex-constrained` 的判断

`route4-ex-constrained` 的定位本来就不是“冲最高熵”，而是：

- 更贴近 Matlab 单文件；
- 更贴近导师熟悉的原始 route4 结构；
- 在尽量少改动物理叙事的前提下，保留 non-diagonal trusted input 的核心提升。

它目前在

- `M=6`
- `N=3`

下已经给出

$$
H_{\min} \approx 1.22750.
$$

对这条线来说，限制更主要来自：

- 固定窗口 `[100,120,140]`
- 固定 `custom_edges`
- 固定相位图样
- 固定 `alpha_values`

而不是单纯因为 `M` 或 `N` 太小。

因此，若把 `N` 直接推到 `8`，更可能是把 constrained 版本变得更不稳定、更难解释，而不是自然得到 `2 bit`。

### 3.7 对 `route4-ex` 的判断

当前 `route4-ex` 最优正式结果约为

$$
H_{\min} \approx 1.54395.
$$

这是在：

- `3` 输入
- `3` 输出
- `M=6`
- `free_monotone_radii`
- `q=[1,0,0]`

这条主线上取得的。

这说明当前提升最关键的来源是：

1. non-diagonal trusted coherent inputs；
2. 合适的三输出 coarse-graining；
3. 局部半径精修；
4. MOSEK 对高值稳定带的正式确认。

而不是“大 `M`”或“大 `N`”本身。

若把当前几条相关主线放在一起看，这一点会更明显：

- 原始 `route4` 在较大 `N` 的 Matlab 兼容口径下，正式最好结果仍只有约 `0.53 bit`
- `route4-ex-constrained` 在 `N=3` 下已达到约 `1.23 bit`
- `route4-ex` 在同样 `N=3` 下，通过 non-diagonal trusted input 与局部精修进一步达到约 `1.54 bit`

因此，从项目现有证据看，把认证值从 `0.5` 提升到 `1.2-1.5` 的关键，不是输出分辨率越做越细，而是 trusted input 模型从对角输入升级到了 non-diagonal coherent input，并允许 full primal 去真正利用这些非对角结构。

因此，从已有证据看，把这条线从 `1.54` 推到 `2.0+` 的主要瓶颈并不在 `M=6` 或 `N=3` 太小，而在于：

- 当前三输入结构已经接近一条窄稳定前沿；
- 再提高输出复杂度，更容易掉入 formal infeasible；
- 当前路线更像“强 `>1 bit` 路线”，不像“只差一点就能到 `2 bit`”。

---

## 4. 两个问题合在一起的汇报

> 当前 `route4-ex` 最优约 `1.54 bit` 的结果，使用的是 `Probability.mat` 中窗口 `[100,120,140]` 的三行实验概率数据，但 trusted input states 不是把 `100,120,140` 直接当成相干态强度，而是另外精修得到的三组复振幅对应的完整相干态。  
> 其中 `rho_x^{diag}` 只是这三个相干态在 Fock 基下的对角部分。  
> 同时，结果里使用的 `M=6, N=3` 并不能简单理解成“参数取太小了所以熵上不去”：对当前这些小振幅 coherent states，`M=6` 已经足够表示主要 Fock 权重；而 `N=3` 则是当前 formal 可行性和认证值之间的一个较优折中。  
> 单纯把 `N` 提到 `8`、把 `M` 加大，并没有明确证据会把 formal `H_min` 推到 `2 bit` 以上，反而更可能先导致规模暴涨和 formal 不可行。

---

## 5. 最终结论

对这两个问题，最终可以压缩成四句话：

1. `1.54 bit` 那个点确实使用了 `[100,120,140]` 这三行实验概率数据；
2. 但 trusted input states 不是“强度 `100,120,140` 的三个相干态”，而是另外精修得到的三组复振幅对应的完整相干态；
3. 其中 `rho_x^{diag}` 只是这些相干态的 Fock 对角投影；
4. `M=6, N=3` 对当前 `route4-ex` / `route4-ex-constrained` 主结果而言并不显得过小，单纯增大它们也没有明显证据会把 formal 最小熵直接推到 `2 bit` 以上。
