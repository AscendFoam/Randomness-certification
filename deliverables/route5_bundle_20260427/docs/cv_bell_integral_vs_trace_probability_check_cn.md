# CV Bell 积分公式与当前 `Tr(M_c \rho)` 概率实现的对比说明

## 1. 背景

把 [SDP_solve.tex](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/docs/SDP_solve.tex) 里的解析积分公式，与当前代码中通过

\[
P(c|x,y)=\mathrm{Tr}(M_c \rho_{xy})
\]

得到概率的方法做一次正面对比，看看两者是否本质一致，尤其是这个思路能否从 route3 推广到 route5。

我这次的判断思路是分四步走的：

1. 先看两种方法描述的物理前端是不是同一个对象。
2. 再把符号、归一化和输出口约定对齐，避免“其实是同一件事但写法不同”。
3. 先在 route3 的 tex-compatible 特例上做强校验。
4. 再在 route5 的固定光强受限 alphabet 上做一次数值诊断，判断它能不能直接替代现有实现。

结论先说：

- 对 route3 而言，`SDP_solve.tex` 的积分公式和当前 `Tr(M_c \rho)` 实现，在对齐约定后是高度一致的。
- 对 route5 而言，也可以做同类对比，而且确实能看出两者在高 cutoff 下逐步接近；但在当前 route5 正式流程常用的低 cutoff 条件下，它更适合作为“收敛/建模诊断工具”，暂时不建议直接替代当前概率引擎。

## 2. 两种概率计算方法分别在做什么

### 2.1 `SDP_solve.tex` 的解析积分公式

[SDP_solve.tex](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/docs/SDP_solve.tex) 采用的是理想化 CV Bell 测量模型：

- 输入是两模相干态 \(|\alpha_1\rangle \otimes |\alpha_2\rangle\)。
- 中央测量是 50:50 分束器之后，对两个可对易联合正交量做测量。
- 连续输出在 \(x\) 轴和 \(p\) 轴上做轴对齐分箱。
- 因为相干态下输出是高斯分布，所以每个 bin 的概率可以写成误差函数 `erf` 的闭式表达。

tex 中写的是 \(X_+=X_1+X_2\)、\(P_-=P_1-P_2\) 这两个变量，因此最后的离散概率本质上是“二维高斯分布落入矩形 bin 的质量”。

### 2.2 当前代码中的 `Tr(M_c \rho)` 方法

当前 route3 / route5 的概率是用更“算符化”的方式算出来的：

- 先在截断 Fock 空间里构造输入态。
- 再构造“平衡分束器 + 两个正交相位 coarse-graining”的 POVM。
- 然后把 POVM 投影到 trusted alphabet 的支持子空间。
- 最后用
  \[
  P(c|x,y)=\mathrm{Tr}(M_c \rho_{xy})
  \]
  得到每个输入下每个离散输出的概率。

相关代码位置如下：

- route3 的 CV Bell POVM 与概率计算在 [cv_four_phase.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/route3/cv_four_phase.py#L119) 和 [cv_four_phase.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/route3/cv_four_phase.py#L175)
- 分束器与正交相位 POVM 的底层构造在 [common.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/common.py#L194) 、[common.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/common.py#L341) 、[common.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/common.py#L446)
- route5 复用了 route3 的这套概率内核，入口在 [hybrid_iq.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/route5/hybrid_iq.py#L18)

从物理含义上说，这两种方法并不是“一个是物理方法，一个是纯数学方法”。它们描述的是同一类 CV Bell 前端，只是：

- `SDP_solve.tex` 走的是理想高斯分布的解析积分；
- 当前代码走的是截断 Hilbert 空间里的 POVM 迹公式。

## 3. 为什么我认为两者可以直接比较

我先检查的是：两边描述的测量链条是否相同。答案基本是“相同，只是坐标约定不同”。

在当前仓库的分束器和测量约定下，route3 / route5 这套离散化双正交测量可以理解成：

- \(x\) 轴均值对应 \(\mathrm{Re}(\alpha_1+\alpha_2)\)
- \(p\) 轴均值对应 \(\mathrm{Im}(\alpha_1-\alpha_2)\)
- 两个轴的方差都对应 \(1/2\)

因此在当前代码约定下，如果 \(x\) 轴分箱边界为 \(\{c_k\}\)，\(p\) 轴分箱边界为 \(\{d_l\}\)，那么理想解析概率可以直接写成

\[
P_{kl}(\alpha_1,\alpha_2)
=\frac{1}{2}\left[\mathrm{erf}(c_k-\mu_x)-\mathrm{erf}(c_{k-1}-\mu_x)\right]
\cdot
\frac{1}{2}\left[\mathrm{erf}(d_l-\mu_p)-\mathrm{erf}(d_{l-1}-\mu_p)\right],
\]

其中

\[
\mu_x=\mathrm{Re}(\alpha_1+\alpha_2), \qquad
\mu_p=\mathrm{Im}(\alpha_1-\alpha_2).
\]

这和 tex 中的 \(X_+,P_-\) 描述是同一物理结构，只差两点：

1. 一个整体的 \(\sqrt{2}\) 归一化缩放。
2. 第二个轴会受到输出口编号 / 坐标方向约定影响，本质上是 \(P_-\) 的等价写法。

也就是说，真正要比的不是“一个对、一个错”，而是“在统一坐标后，它们是不是给出相同的 bin 概率”。

基于这个对齐，我写了一个专门的验证脚本：

- [verify_cv_bell_probabilities.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/verify_cv_bell_probabilities.py)

对应的结果保存在：

- [cv_bell_probability_formula_check.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/cv_bell_probability_formula_check.json)

## 4. route3 上的强校验

### 4.1 为什么先用 route3

`SDP_solve.tex` 最直接对应的是“相位只取 \(0,\pi\) 的实输入”情形，这正好是 route3 的一个特例。因此 route3 是最合适的第一检查点。

### 4.2 校验设置

我选的是：

- `mu = 0.05`
- `cutoff = 12`
- `num_phases = 2`
- `num_x_bins = 2`
- `num_p_bins = 2`
- `quadrature_range = 3.0`

在这个设置下：

- 解析公式按 tex 思路计算每个矩形 bin 的高斯积分；
- 当前代码则按相同输入、相同 bin 边界去算 `Tr(M_c \rho)`。

### 4.3 结果

数值结果非常接近：

- `max_abs_error ≈ 1.06e-4`
- `mean_abs_error ≈ 5.31e-5`

而且有两个输入点的误差已经到机器精度量级（约 `1e-15`）。

### 4.4 解释

这个量级的误差已经足够说明：在 route3 的 tex-compatible 特例中，解析积分公式与当前 `Tr(M_c \rho)` 概率实现本质一致。

残余误差主要来自数值实现细节，例如：

- 正交相位 POVM 是通过 Gauss-Hermite 求积构造的；
- 构造后又做了一次数值白化来强制 POVM 完备性；
- 这些步骤都会带来 \(10^{-4}\) 量级的数值差。

所以对 route3，可以把 `SDP_solve.tex` 里的公式视为当前概率实现的一个强交叉验证。

## 5. route5 上的对比与判断

### 5.1 为什么 route5 也值得比较

route5 仍然是同一类前端：

- 输入仍然是两路相干态；
- 中央仍然是平衡分束器 + 双正交测量；
- 输出仍然是 IQ 平面的轴对齐 coarse-graining。

因此从物理上说，解析积分公式完全可以推广到 route5。区别只在于：

- route3 只用了较简单的输入字母表；
- route5 用的是更一般的 coherent alphabet；
- route5 的正式 SDP 目前常用较低的 Fock cutoff，并把态与 POVM 都投影到受限支持空间。

### 5.2 我这次做的 route5 检查

为了和最近讨论过的“固定光强受限 alphabet”一致，我选了强度菜单：

- `[0, 80, 160]`

并沿用 route5 当前内部的归一化映射，把它们转成半径：

- `[0.0, 0.8485, 1.2]`

这里需要强调，这只是为了在当前 route5 数值框架里做一致性检查，不是新的实验标定结论。

其余设置为：

- 8 个相位
- `6 x 2` 个 IQ 输出 bin
- `quadrature_range = 1.8`
- 比较 `cutoff = 4, 8, 12`

### 5.3 结果

结果如下：

| cutoff | max abs error | mean abs error | 解释 |
| --- | ---: | ---: | --- |
| 4 | 0.32396 | 0.08276 | 与理想解析模型差距较大 |
| 8 | 0.02579 | 0.01344 | 明显接近 |
| 12 | 0.02123 | 0.01328 | 继续接近，但已进入平台区 |

### 5.4 这些结果说明什么

这组数据说明两点。

第一，route5 的解析积分推广是有意义的。因为随着 cutoff 从 4 提高到 8、12，`Tr(M_c \rho)` 和理想高斯积分之间的差距显著下降，这说明两者确实在朝同一个物理对象收敛。

第二，当前 route5 正式流程里如果还停留在低 cutoff，解析积分不能直接拿来“替换”现有概率。原因不是公式本身错，而是模型层级不同：

- 解析积分对应的是理想无限维 coherent-state 高斯模型；
- 当前正式 route5 SDP 用的是低 cutoff、有限支持、投影后的有效态模型；
- 如果只把概率表换成理想解析值，但 trusted states 仍然沿用低 cutoff 投影态，那么整个 SDP 约束未必自洽。

更直白地说：

- route5 里“态的模型”和“概率的模型”目前是绑在一起的；
- 直接只换概率，不同步提升态模型，可能会把 SDP 变成一个内部不一致的问题。

## 6. 我对“能不能替代现有方法”的最终判断

我的结论是分成三层。

### 6.1 对 route3

可以认为已经验证成功。

如果后续只是想说明“当前代码的概率是不是和 tex 里的公式一致”，那么 route3 这里可以直接回答：一致，误差只剩数值求积层面的微小差别。

### 6.2 对 route5

可以比较，也值得比较，但当前更适合当作以下两类工具：

- 截断 cutoff 是否足够的收敛诊断；
- 当前 route5 数值概率是否偏离理想 CV Bell 模型太多的物理 sanity check。

它暂时不适合作为 route5 正式概率引擎的直接替代品，至少在当前低 cutoff 工作流下不适合。

### 6.3 如果后续想把它真正纳入正式流程

我觉得可以按下面的顺序推进：

1. 固定同一个 alphabet 和同一个 bin 方案，系统检查 cutoff 提升时概率是否收敛。
2. 再检查对应的 SDP 认证值是否也随 cutoff 收敛。
3. 只有当“态表示”和“概率表示”在更高 cutoff 下都稳定后，才考虑把解析公式更深地嵌入正式流程。

## 7. 简明结论

可以概括成下面这段：

“我把 `SDP_solve.tex` 里的 CV Bell 解析积分公式，与当前代码中通过 `Tr(M_c \rho)` 计算概率的方法做了对比。结论是：在 route3 的 tex-compatible 特例里，两者在对齐归一化和输出轴约定后高度一致，最大误差约 `1e-4`，因此可把该积分公式视为当前概率实现的强交叉验证。对于 route5，这个积分公式也可以推广使用，并且数值上随着 Fock cutoff 提升会逐步接近当前 `Tr(M_c \rho)` 结果；但由于 route5 当前正式 SDP 使用的是低 cutoff、投影到有限支持空间的有效态模型，所以该积分公式目前更适合作为收敛诊断和物理 sanity check，而不宜直接替代现有概率引擎。” 

## 8. 本次产出

- 验证脚本：[verify_cv_bell_probabilities.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/src/python/qrng_routes/verify_cv_bell_probabilities.py)
- 数值结果：[cv_bell_probability_formula_check.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/Randomness-certification/output/qrng_routes/cv_bell_probability_formula_check.json)

如果后续需要，我可以在这份说明的基础上再整理一版“更适合发老师微信/邮件”的短版摘要。
