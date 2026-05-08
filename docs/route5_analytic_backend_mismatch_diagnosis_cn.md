# Route5 解析概率后端失配诊断简报

## 1. 问题背景

最近对 `route5` 做了一次关键复核：把原先的 `trace_povm` 概率后端，替换为更接近理想 CV Bell / dual-homodyne 理论公式的解析高斯积分后端 `analytic_gaussian_rectangles`，检查此前主线

\[
\text{radii}=[0,0.85,1.25]
\]

在解析后端下是否仍能给出正式的

\[
H_{\min}>2.
\]

这一步的意义很直接：

- 如果解析后端下 formal 仍然大于 `2`，那么 `route5` 还能维持“可向实验室正式汇报”的状态；
- 如果解析后端下 formal 失效，那么此前 `route5` 的高熵结果只能视为“旧后端下的数值探索结果”，不能再直接作为正式主结果。

---

## 2. 本轮测试配置

本轮复核使用的是当前 `route5` 的历史强点主线：

- `radius_values = [0, 0.85, 1.25]`
- `phase_values = 8` 个均匀相位
- `cutoff = 4`
- `num_x_bins = 6`
- `num_p_bins = 2`
- `num_outputs = 12`
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`
- `max_inputs = 3`
- solver = `MOSEK`
- probability engine = `analytic_gaussian_rectangles`

对应输出文件：

- [route5_node_convergence_scan_analytic_r085125.json](../output/qrng_routes/route5_node_convergence_scan_analytic_r085125.json)
- [route5_analytic_backend_diagnostics_r085125.json](../output/qrng_routes/route5_analytic_backend_diagnostics_r085125.json)

---

## 3. 主结论

结论非常明确：

1. 在解析概率后端下，这条 `route5` 主线目前不能作为正式结果汇报。
2. 原因不是 “formal 值从 `>2` 稍微掉一点”，而是 formal 直接变成了 `infeasible`。
3. 因而此前基于 `trace_povm` 得到的 `formal > 2`，不能被自动视为理想解析 CV Bell 模型下也成立。

更具体地说，本轮解析后端 formal 复核得到：

- `raw_best_H_min ≈ 2.7277`
- 最强 raw 目标输入变成了 `(0,0)`、`(12,10)`、`(11,11)` 这一类
- 但对 top-3 raw 目标，formal 全部返回 `infeasible`

这说明问题已经不是“raw 很强，但 formal 稍微保守一些”，而是：

\[
\text{解析概率表} \quad+\quad \text{当前 trusted-state/SDP 模型}
\]

这两部分在数学上已经不自洽。

---

## 4. 直接数值证据

### 4.1 formal 复核结果

对主线配置，解析后端给出的 top-3 raw 目标为：

- `(0,0)`
- `(12,10)`
- `(11,11)`

它们的 raw 最小熵都约为

\[
H_{\min}^{\mathrm{raw}} \approx 2.7277.
\]

但 formal 认证时，这三个点全部返回：

- `status = infeasible`

因此目前不能再说“这条 `route5` 主线 formal 超过 2 bit”。

### 4.2 trace 后端与解析后端的偏差

用诊断脚本比较两种概率后端，可得到下表：

| cutoff | local rank | mean TV(trace, analytic) | analytic 线性拟合残差 max | 结论 |
| --- | ---: | ---: | ---: | --- |
| 4 | 4 | 0.1542 | 0.1413 | 明显失配 |
| 5 | 5 | 0.0880 | 0.0209 | 仍明显失配 |
| 6 | 6 | 0.0541 | \(1.6\times 10^{-14}\) | 线性上可拟合，但仍有更深层 incompatibility |
| 8 | 8 | 0.0354 | \(\sim 10^{-14}\) | 与解析后端更接近，但未完全一致 |
| 12 | 12 | 0.0338 | \(\sim 10^{-14}\) | 进入平台区 |
| 16 | 16 | 0.0338 | \(\sim 10^{-14}\) | 平台区仍存在 |

这张表给出两个非常关键的事实：

1. `cutoff=4/5` 时，解析概率列向量甚至不在当前 trusted-state 模型诱导出的线性像空间里。
2. 即使把 `cutoff` 增到 `6` 以上，旧 `trace_povm` 后端和解析后端之间仍然保留了大约 `3.4e-2` 量级的平均 TV 差异，没有继续收敛到零。

### 4.3 截断尾概率本身已经不小

对于最大半径 `|alpha| = 1.25`，Poisson 尾概率为：

- `cutoff=4` 时：`0.07373`
- `cutoff=5` 时：`0.02167`
- `cutoff=6` 时：`0.00540`

也就是说，仅在单模层面，`cutoff=4` 已经丢掉了约 `7.4%` 的概率质量。

如果两个输入都处在大半径层，那么对应双模态保留在截断子空间内的概率质量大约只有

\[
0.92627^2 \approx 0.8580,
\]

也就是还有大约 `14.2%` 的双模概率质量在截断外。

这已经不是“很小的高阶修正”。

### 4.4 当前本地态模型维数明显偏小

当前主线的本地 alphabet 一共有 `17` 个相干态：

- `1` 个真空态
- `8` 个半径 `0.85` 的相位态
- `8` 个半径 `1.25` 的相位态

但在当前截断模型中，本地支持维数只有：

- `cutoff=4` 时 `local_rank = 4`
- `cutoff=5` 时 `local_rank = 5`
- `cutoff=6` 时 `local_rank = 6`
- `cutoff=16` 时 `local_rank = 16`
- 直到 `cutoff=17` 才达到 `local_rank = 17`

这意味着：在 `cutoff=4` 的正式主线里，我们实际上是在用一个只有 `4` 维的 trusted-state 子空间，去描述一个由 `17` 个固定相位相干态生成的真实 alphabet。

这类压缩本身并不一定非法，但一旦概率层切换到“无限维理想 coherent + 解析 Gaussian bin”模型，双方很容易失去自洽。

---

## 5. 为什么解析后端会把 formal 推成 infeasible

### 5.1 当前 formal 问题真正求解的对象

解析概率后端给出的统计，本质上是

\[
p(c|s)=\mathrm{Tr}\!\bigl(M_c^{(\infty)} \rho_s^{(\infty)}\bigr),
\]

其中：

- \(\rho_s^{(\infty)}\) 是无限维精确相干态输入；
- \(M_c^{(\infty)}\) 是理想 CV Bell / dual-homodyne 的矩形 coarse-grained POVM。

但当前 `route5` formal SDP 并不是在这个模型上求解，而是在下面这个模型上求解：

\[
\rho_s^{(d)}=
\frac{P_d \rho_s^{(\infty)} P_d}
{\mathrm{Tr}(P_d \rho_s^{(\infty)})},
\qquad
\tilde\rho_s^{(d)} = B_d^\dagger \rho_s^{(d)} B_d,
\]

其中：

- \(P_d\) 是 Fock 截断投影；
- \(d=\texttt{cutoff}\)；
- \(B_d\) 是由截断态张成的支持空间基；
- SDP 要找的是有限维 POVM 元 \(E_c\)，满足

\[
E_c \succeq 0,\qquad \sum_c E_c = I,
\qquad
\mathrm{Tr}(E_c \tilde\rho_s^{(d)}) = p(c|s).
\]

也就是说，当前 formal 在问的是：

“是否存在一个有限维 POVM，能让这批截断/投影后的 trusted states 精确复现无限维解析概率表？”

这件事不是自动成立的。

### 5.2 `cutoff=4/5`：连线性可实现性都失败

对每一个输出 \(c\)，若想存在某个算符 \(E_c\) 使

\[
\mathrm{Tr}(E_c \tilde\rho_s^{(d)}) = p(c|s),
\]

那么对应的概率列向量 \(p_c(s)\) 至少必须落在当前 states 诱导出的线性像空间内。

诊断结果表明：

- `cutoff=4` 时最大列残差约 `0.1413`
- `cutoff=5` 时最大列残差约 `0.0209`

这说明在这两个 cutoff 下，问题甚至不是 “找不到正定 POVM”，而是更基础地：

\[
\text{连线性算符都不可能精确复现这张解析概率表。}
\]

因此 formal `infeasible` 是必然的，不是求解器偶然失稳。

### 5.3 `cutoff=6`：线性上能拟合，但物理 POVM 仍不存在

到 `cutoff=6` 时，线性残差已经降到机器精度量级，这意味着：

- 每一列概率向量单独看，都可以由某个线性算符拟合出来；

但这还不够，因为真实 POVM 还必须同时满足：

\[
E_c \succeq 0,\qquad \sum_c E_c = I.
\]

此前已经对普通 POVM 可行性做过单独复核：即使不进入 guessing SDP 的二级拆分，只要求

\[
E_c \succeq 0,\qquad \sum_c E_c = I,\qquad
\mathrm{Tr}(E_c \tilde\rho_s^{(d)}) = p(c|s),
\]

在 `cutoff=4/5/6` 下也仍然是 `infeasible`。

这说明问题已经上升到：

\[
\text{解析概率表与当前 finite-dimensional trusted-state 几何本身不兼容。}
\]

也就是说，不是 guessing SDP 太强，而是普通 POVM 层面就已经对不上。

### 5.4 更深层原因：当前态表示不是“解析 coherent 概率”的自洽态表示

如果我们真的想在解析高斯概率模型下做 formal，那么最自然的态表示应当也是“精确 coherent-state 几何”。

这时应该用的是相干态的 Gram 矩阵：

\[
G_{ij}=\langle \alpha_i | \alpha_j \rangle=
\exp\!\left(
-\frac{|\alpha_i|^2+|\alpha_j|^2}{2}
+\alpha_i^* \alpha_j
\right),
\]

再从 \(G\) 中提取 exact support 表示。

而当前 `route5` 仍然采用的是：

- 先做 Fock cutoff；
- 再对截断态重新归一化；
- 再投影到截断态支持子空间；
- 最后用这个低维模型去匹配无限维解析概率。

这就是这次失配的根本来源。

---

## 6. 现阶段应如何解读 route5 结果

现阶段最稳妥的解读是：

1. `trace_povm` 后端下的 `route5` 高熵结果，仍然有数值研究价值；
2. 但它们不能再直接表述为“理想解析 CV Bell 概率模型下正式 confirmed 的 >2 bit 结果”；
3. 在解析后端 formal 重新变得自洽之前，`route5` 暂时不应恢复到“可向实验室正式汇报”的状态。

换句话说，现在最重要的不是再继续报更高的旧 formal 数值，而是先把模型口径统一。

---

## 7. 建议的解决方案

### 方案 A：短期保守方案

直接把当前 `analytic_gaussian_rectangles` 视为诊断后端，而不是正式后端。

具体做法：

- `trace_povm` 结果保留为“旧工作流下的探索性结果”
- 解析后端只用于检查“当前 route5 与理想理论模型差得有多远”
- 暂时停止把 `trace_povm > 2` 作为正式可汇报主结论

这个方案最稳，但代价是：`route5` 暂时失去正式主结果资格。

### 方案 B：真正自洽的解析后端方案

如果希望解析后端重新成为正式主线，那么必须把 trusted-state 模型也改成与之配套的“精确 coherent-state 支撑表示”。

技术上更合理的路线是：

1. 不再用 Fock cutoff 构造 trusted states；
2. 直接用相干态 Gram 矩阵
   \[
   G_{ij}=\langle \alpha_i|\alpha_j\rangle
   \]
   构造本地 exact support；
3. 在这个 support 上构造联合输入态；
4. 概率层继续用解析 Gaussian rectangles；
5. 再在这个完全一致的模型上做 SDP。

这实际上已经很接近当前仓库中 `route6` 的思路。

优点：

- 态模型和概率模型完全一致；
- 不再有 Fock 截断尾概率失配；
- 理论口径最干净。

难点：

- 当前 17 个本地态对应的 exact local support 维数接近 `17`；
- 联合维数将接近 `17^2=289`；
- 以现有 primal SDP 直接做 formal，规模会非常大。

因此如果走这条路，往往还需要同时做：

- alphabet 缩减
- 对称性化简
- 或更换/推导更适合的 dual SDP

### 方案 C：折中方案，不再用任意 Fock cutoff，而改做 Gram-threshold 压缩

本地 17 态的精确 Gram 矩阵虽然数学上是满秩的，但数值上比较病态，最小特征值很小。

这说明一个更合理的折中方式是：

- 不是随意指定 `cutoff=4`
- 而是先构造 exact coherent Gram matrix
- 再按特征值阈值 `gram_tol` 压缩成一个“受控低秩近似”

这样得到的是：

- 与真实 coherent overlap 一致的低维近似；
- 比 “先 Fock 截断再重归一化” 更有物理解释；
- 也更适合作为后续误差分析对象。

这同样更接近 `route6` 现有框架。

### 方案 D：若仍想沿 `route5` 当前框架补救，至少不要再停留在 `cutoff=4`

如果坚持保留 `route5` 当前 Fock 截断主框架，那么最低限度也应该承认：

- `cutoff=4` 对 `|alpha|=1.25` 明显过小；
- `cutoff=6` 也还不足以恢复普通 POVM 可行性；
- 单纯把 `cutoff` 从 `4` 提到 `6` 并不能解决问题。

而要等本地支持维数真正追到 alphabet 的 `17` 态规模，至少要到 `cutoff \ge 17` 附近。

但这时 formal SDP 的尺寸又会急剧变大，所以“继续硬抬 cutoff”并不是当前最优解法。

---

## 8. 推荐的下一步主线

我建议后续按下面的优先级推进：

1. 先明确停止把当前 `route5 trace_povm > 2` 作为正式可汇报结果。
2. 把当前结论写清楚：解析后端下，现有 `route5` 主线 formal `infeasible`，根源是态模型与概率模型不自洽。
3. 如果还想保 `route5`，应优先转向“Gram 精确态表示 + 解析概率”的自洽版本，而不是继续在 `cutoff=4` 上抠数值。
4. 如果算力受限，则优先尝试：
   - 缩小 alphabet
   - 减少相位数
   - 或在 `route6` 式 exact-support 框架下做受控低秩近似

---

## 9. 一句话总结

本轮复核表明：`route5` 当前的 `analytic_gaussian_rectangles` 概率后端，与 `cutoff=4` 的 truncated/projected trusted-state formal 模型并不自洽，因此 formal 会直接变成 `infeasible`。问题的核心不是求解器，也不是 guessing SDP 太强，而是“无限维理想 coherent 概率”与“低维截断重归一化 trusted states”之间存在结构性失配。要修复它，正确方向不是继续依赖 `cutoff=4`，而是转向与解析概率口径一致的 exact/Gram-support 态表示。
