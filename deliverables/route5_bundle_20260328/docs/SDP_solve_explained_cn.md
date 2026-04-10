# 连续变量Bell态测量与MDIQRNG详解

> 本文档是《SDP_solve.tex》的详细解读，面向不具备量子力学背景的学生。我们将逐步解释每个物理概念、符号和公式。

---

## 一、背景引入：什么是MDIQRNG？

### 1.1 随机数的重要性

随机数在密码学、模拟计算、博彩游戏等领域至关重要。理想的随机数必须是**不可预测的**，而量子力学提供了真正的随机性来源——因为量子测量结果本质上是随机的。

### 1.2 设备无关量子随机数生成（DIQRNG / MDIQRNG）

- **DIQRNG（Device-Independent QRNG）**：即使你不信任测量设备，也能产生真随机数。但实际实现非常困难。
- **MDIQRNG（Measurement-Device-Independent QRNG）**：一种介于两者之间的方案。它不需要信任测量设备的内部工作原理，但假设光源是可信的。这种方案更加实用。

### 1.3 连续变量（CV）与离散变量（DV）

在量子光学中，有两种描述光场的方式：
- **离散变量（DV）**：以单个光子为基本单位
- **连续变量（CV）**：以光的振幅（如位置和动量）为连续变量来描述

本文档采用**连续变量**方案，因为它更适合使用标准的光学元件（如分束器）实现Bell测量。

---

## 二、预备知识：什么是相干态？

### 2.1 简化的光场描述

在量子力学中，**相干态（Coherent State）**是描述激光最常用的量子态。简单理解：
- 它是一种最"经典"的光场状态
- 它的量子涨落最小
- 它可以用一个复数 $\alpha$ 完全描述

**相干态的标记**：$|\alpha\rangle$
- $\alpha$ 可以写成 $\alpha = \sqrt{\mu} e^{i\phi}$
- $\mu = |\alpha|^2$ 是**平均光子数**（光的强度）
- $\phi$ 是**相位**（光波的振动方向）

### 2.2 振幅算符与正交分量

我们引入两个**正交振幅算符**来描述单模光场：

$$\hat{X} = \frac{\hat{a} + \hat{a}^\dagger}{\sqrt{2}}, \quad \hat{P} = \frac{\hat{a} - \hat{a}^\dagger}{i\sqrt{2}}$$

**直观理解**：
- $\hat{X}$ 类似于光波的"位置"或"余弦分量"
- $\hat{P}$ 类似于光波的"动量"或"正弦分量"
- 它们统称为** quadrature 算符**（正交振幅算符）

**算符与期望值**：
对于相干态 $|\alpha\rangle$：
- $\langle \hat{X} \rangle = \sqrt{2}\,\text{Re}(\alpha) = \sqrt{2\mu}\cos\phi$
- $\langle \hat{P} \rangle = \sqrt{2}\,\text{Im}(\alpha) = \sqrt{2\mu}\sin\phi$

**方差**（量子涨落）：
$$\text{Var}(\hat{X}) = \text{Var}(\hat{P}) = \frac{1}{2}$$

这意味着即使是最"安静"的相干态（真空态），也有固有的量子涨落。这满足**海森堡不确定性原理**：
$$\Delta X \cdot \Delta P \geq \frac{1}{2}$$

---

## 三、本文的输入态：双模相干态

### 3.1 什么是双模状态？

**双模**意味着我们同时处理两束光（两个"模式"的光场）。本文中：
- **模式1**（Mode 1）：第一束光
- **模式2**（Mode 2）：第二束光

双模相干态是两个单模相干态的直积：
$$|\psi_\sigma\rangle = |\alpha_1\rangle \otimes |\alpha_2\rangle$$

### 3.2 相位的特殊选择：$\phi \in \{0, \pi\}$

本文做了一个简化假设：**相位只有两个取值**：0或π。

**为什么这样假设？**
- 当 $\phi = 0$ 时：$\alpha_1 = +\sqrt{\mu_1}$（正的实数）
- 当 $\phi = \pi$ 时：$\alpha_1 = -\sqrt{\mu_1}$（负的实数）

这相当于光的振幅要么是正向，要么是反向，正好相反。

引入**符号参数** $s_1, s_2 \in \{+1, -1\}$：
- $s_1 = +1$ 对应 $\phi_1 = 0$（正向振幅）
- $s_1 = -1$ 对应 $\phi_1 = \pi$（负向振幅）
- 同理 $s_2$ 对应模式2

### 3.3 四种输入态

由于 $s_1, s_2$ 各有两种取值，组合起来有**四种输入态**：

| 状态标记 $\sigma$ | 模式1 | 模式2 | 物理含义 |
|-----------------|-------|-------|---------|
| $(+1, +1)$ | $+\alpha$ | $+\alpha$ | 两束光振幅都正向 |
| $(+1, -1)$ | $+\alpha$ | $-\alpha$ | 模式1正向，模式2负向 |
| $(-1, +1)$ | $-\alpha$ | $+\alpha$ | 模式1负向，模式2正向 |
| $(-1, -1)$ | $-\alpha$ | $-\alpha$ | 两束光振幅都负向 |

**重要性质**：由于相位只有0或π，光的动量期望值始终为0：
$$\langle \hat{P}_1 \rangle = \langle \hat{P}_2 \rangle = 0$$

---

## 四、连续变量Bell测量

### 4.1 为什么要做Bell测量？

Bell测量是实现**纠缠检测**和**量子关联**的关键。在MDIQRNG中，我们需要测量两个光场之间的量子关联，从而产生不可预测的随机数。

### 4.2 联合正交算符

对于双模光场，我们构造两个**联合正交算符**：

$$X_+ = \hat{X}_1 + \hat{X}_2, \quad P_- = \hat{P}_1 - \hat{P}_2$$

**直观理解**：
- $X_+$：两束光的"位置"相加
- $P_-$：两束光的"动量"相减

**关键性质**：$X_+$ 和 $P_-$ 是**对易的**，即 $[X_+, P_-] = 0$，这意味着它们可以同时被精确测量。

**为什么选择这种组合？**
这种特定的组合（一个相加、一个相减）能够产生最大程度的量子关联，是实现CV Bell态测量的标准选择。

---

## 五、测量结果的离散化

### 5.1 为什么需要离散化？

实际的探测器只能分辨有限数量的输出值。因此，我们需要把连续的测量结果 $x_+$ 和 $p_-$ 映射到有限的"档位"。

### 5.2 离散化方案

**$X_+$ 的离散化**：
- 连续值 $x_+ \in \mathbb{R}$ 被分成 $n$ 个区间
- 第 $k$ 个区间：$I_{+k} = [c_{k-1}, c_k)$，其中 $k = 1, 2, \ldots, n$
- $c_0 \to -\infty$，$c_n \to +\infty$（边界延拓到无穷）

**$P_-$ 的离散化**：
- 类似地，$p_- \in \mathbb{R}$ 被分成 $n$ 个区间
- 第 $l$ 个区间：$I_{-l} = [d_{l-1}, d_l)$

**离散化结果**：
- 测量outcome是一个**二元组** $(k, l)$
- 总共有 $n^2$ 种可能的离散结果

**简单例子**（$n=2$的情况）：
- $X_+$ 分成两个区间：$(-\infty, 0)$ 和 $[0, +\infty)$
- $P_-$ 分成两个区间：$(-\infty, 0)$ 和 $[0, +\infty)$
- 结果有4种：$(1,1), (1,2), (2,1), (2,2)$

---

## 六、条件概率的数学推导

### 6.1 什么是条件概率？

条件概率 $P((k,l)|\sigma)$ 表示：**给定输入态是 $\sigma$ 的情况下，测量到离散结果 $(k,l)$ 的概率**。

用积分表示：
$$P((k,l)|\sigma) = \iint_{I_{+k} \times I_{-l}} f(x_+, p_-|\sigma) \, dx_+ dp_-$$

其中 $f(x_+, p_-|\sigma)$ 是给定输入态 $\sigma$ 时，连续变量 $(x_+, p_-)$ 的**联合概率密度**。

### 6.2 联合概率密度是高斯分布

由于 $X_+$ 和 $P_-$ 都是正交算符的线性组合，而相干态的量子态是高斯型的，因此**联合概率密度是高斯分布**。

**均值（期望值）**：
- 对于 $X_+$：
$$\mu_{+\sigma} = \langle X_+ \rangle_\sigma = \sqrt{2}(s_1\sqrt{\mu_1} + s_2\sqrt{\mu_2})$$

**推导过程**：
$$\mu_{+\sigma} = \langle \hat{X}_1 \rangle + \langle \hat{X}_2 \rangle = \sqrt{2}\,\text{Re}(\alpha_1) + \sqrt{2}\,\text{Re}(\alpha_2) = \sqrt{2}(s_1\sqrt{\mu_1} + s_2\sqrt{\mu_2})$$

- 对于 $P_-$：
$$\mu_{-\sigma} = \langle P_- \rangle_\sigma = \langle \hat{P}_1 \rangle - \langle \hat{P}_2 \rangle = 0 - 0 = 0$$

**方差**（涨落）：
$$\text{Var}(X_+) = 1, \quad \text{Var}(P_-) = 1$$

这是因为两个模式的方差是**独立可加**的：
$$\text{Var}(X_+) = \text{Var}(\hat{X}_1) + \text{Var}(\hat{X}_2) = \frac{1}{2} + \frac{1}{2} = 1$$

### 6.3 高斯概率密度函数

给定输入态 $\sigma$ 的条件下，联合概率密度为：

$$f(x_+, p_-|\sigma) = \frac{1}{2\pi} \exp\left( -\frac{(x_+ - \mu_{+\sigma})^2}{2} - \frac{p_-^2}{2} \right)$$

**解释**：
- 这是一个二维高斯分布
- 均值在 $(\mu_{+\sigma}, 0)$ 处
- 两个方向的方差都是1
- $2\pi$ 是归一化因子

**可分离性**（重要！）：
$$f(x_+, p_-|\sigma) = f_+(x_+|\sigma) \cdot f_-(p_-)$$

这意味着 $X_+$ 和 $P_-$ 的测量结果是**统计独立**的。

### 6.4 通过误差函数计算积分

现在计算条件概率：
$$P((k,l)|\sigma) = \left( \int_{c_{k-1}}^{c_k} f_+(x_+|\sigma) \, dx_+ \right) \cdot \left( \int_{d_{l-1}}^{d_l} f_-(p_-) \, dp_- \right)$$

**误差函数（Error Function）**定义：
$$\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z \exp(-t^2) \, dt$$

误差函数可以将高斯积分转化为函数值的差。

**$X_+$ 的积分结果**：
$$\int_{c_{k-1}}^{c_k} \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(x_+ - \mu_{+\sigma})^2}{2} \right) dx_+ = \frac{1}{2} \left[ \text{erf}\left( \frac{c_k - \mu_{+\sigma}}{\sqrt{2}} \right) - \text{erf}\left( \frac{c_{k-1} - \mu_{+\sigma}}{\sqrt{2}} \right) \right]$$

**$P_-$ 的积分结果**：
$$\int_{d_{l-1}}^{d_l} \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{p_-^2}{2} \right) dp_- = \frac{1}{2} \left[ \text{erf}\left( \frac{d_l}{\sqrt{2}} \right) - \text{erf}\left( \frac{d_{l-1}}{\sqrt{2}} \right) \right]$$

### 6.5 最终公式

将 $\mu_{+\sigma} = \sqrt{2}(s_1\sqrt{\mu_1} + s_2\sqrt{\mu_2})$ 代入，得到**条件概率的完整公式**：

$$P((k,l)|s_1,s_2) = \frac{1}{4} \cdot \left[ \text{erf}\left( \frac{c_k}{\sqrt{2}} - s_1\sqrt{\mu_1} - s_2\sqrt{\mu_2} \right) - \text{erf}\left( \frac{c_{k-1}}{\sqrt{2}} - s_1\sqrt{\mu_1} - s_2\sqrt{\mu_2} \right) \right] \cdot \left[ \text{erf}\left( \frac{d_l}{\sqrt{2}} \right) - \text{erf}\left( \frac{d_{l-1}}{\sqrt{2}} \right) \right]$$

**参数说明**：
- $(s_1, s_2) \in \{(+1,+1), (+1,-1), (-1,+1), (-1,-1)\}$：四种输入态
- $k, l \in \{1, 2, \ldots, n\}$：离散结果的索引
- $c_{k-1}, c_k$：$X_+$ 离散化的边界
- $d_{l-1}, d_l$：$P_-$ 离散化的边界

**关于参数选择的备注**：
- $\mu_1$ 和 $\mu_2$ 是平均光子数，可以取相同值，也可以在 $[0, 10]$ 范围内优化
- 边界值 $|c_1|, |c_{n-1}|, |d_1|, |d_{n-1}|$ 通常设为10（约10个标准差）

---

## 七、态的矩阵表示（向量形式）

### 7.1 为什么要用向量表示态？

在量子力学中，量子态可以用**向量**（更准确地说，是**态矢量**）来表示。这种表示使得计算变得简单明确。

在有限维子空间中，我们可以把态写成列向量：

$$|\psi\rangle = \begin{pmatrix} \text{分量1} \\ \text{分量2} \\ \vdots \end{pmatrix}$$

### 7.2 单模的二维子空间

对于单模，我们定义一个**二维正交归一基** $\{|0\rangle, |1\rangle\}$：
- $|0\rangle \equiv |\alpha\rangle$：相干态 $|\alpha\rangle$（相位为0）
- $|1\rangle$：与 $|0\rangle$ 正交归一化的另一个基向量

**注意**：$|1\rangle$ **不是**数光子数的那个 $|1\rangle$（Fock态），而是为了在由 $|\alpha\rangle$ 和 $|-\alpha\rangle$ 张成的子空间中构造正交基而引入的。

### 7.3 展开系数 $\delta$ 的计算

由于 $|-\alpha\rangle$（相位为π）也在这个子空间中，它可以展开为：
$$|-\alpha\rangle = \delta |0\rangle + \sqrt{1-\delta^2} |1\rangle$$

其中 $\delta = \langle 0 | -\alpha \rangle = \langle \alpha | -\alpha \rangle$ 是内积。

**计算 $\delta$**：
两个相干态的内积为：
$$\langle \alpha | \beta \rangle = \exp\left( -\frac{1}{2}(|\alpha|^2 + |\beta|^2 - 2\alpha^*\beta) \right)$$

设 $\beta = -\alpha$，且 $\alpha$ 是实数（因为相位只有0或π）：
$$\delta = \langle \alpha | -\alpha \rangle = \exp\left( -\frac{1}{2}(\mu + \mu + 2\mu) \right) = e^{-2\mu}$$

**物理意义**：$\delta$ 表示 $|\alpha\rangle$ 和 $|-\alpha\rangle$ 之间的**重叠程度**。
- 当 $\mu$ 很小时（弱光），$\delta \approx 1$（两态几乎相同）
- 当 $\mu$ 很大时（强光），$\delta \approx 0$（两态几乎正交）

### 7.4 双模态的向量表示

双模空间是单模空间的张量积。我们用4维基：
$$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$
其中 $|ij\rangle = |i\rangle_1 \otimes |j\rangle_2$。

**四种输入态的向量表示**（假设 $\mu_1 = \mu_2 = \mu$，即两束光强度相同）：

**态1：$\sigma = (+1, +1)$ → $|\alpha\rangle_1 \otimes |\alpha\rangle_2$**
$$|\psi_{(+1,+1)}\rangle = |00\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

**态2：$\sigma = (+1, -1)$ → $|\alpha\rangle_1 \otimes |-\alpha\rangle_2$**
\begin{align*}
|\alpha\rangle_1 \otimes |-\alpha\rangle_2 &= |0\rangle_1 \otimes (\delta |0\rangle_2 + \sqrt{1-\delta^2}|1\rangle_2) \\
&= \delta |00\rangle + \sqrt{1-\delta^2} |01\rangle
\end{align*}
$$|\psi_{(+1,-1)}\rangle = \begin{pmatrix} \delta \\ \sqrt{1-\delta^2} \\ 0 \\ 0 \end{pmatrix}$$

**态3：$\sigma = (-1, +1)$ → $|-\alpha\rangle_1 \otimes |\alpha\rangle_2$**
\begin{align*}
|-\alpha\rangle_1 \otimes |\alpha\rangle_2 &= (\delta |0\rangle_1 + \sqrt{1-\delta^2}|1\rangle_1) \otimes |0\rangle_2 \\
&= \delta |00\rangle + \sqrt{1-\delta^2} |10\rangle
\end{align*}
$$|\psi_{(-1,+1)}\rangle = \begin{pmatrix} \delta \\ 0 \\ \sqrt{1-\delta^2} \\ 0 \end{pmatrix}$$

**态4：$\sigma = (-1, -1)$ → $|-\alpha\rangle_1 \otimes |-\alpha\rangle_2$**
\begin{align*}
|-\alpha\rangle_1 \otimes |-\alpha\rangle_2 &= (\delta |0\rangle_1 + \sqrt{1-\delta^2}|1\rangle_1) \otimes (\delta |0\rangle_2 + \sqrt{1-\delta^2}|1\rangle_2) \\
&= \delta^2 |00\rangle + \delta\sqrt{1-\delta^2} |01\rangle + \delta\sqrt{1-\delta^2} |10\rangle + (1-\delta^2) |11\rangle
\end{align*}
$$|\psi_{(-1,-1)}\rangle = \begin{pmatrix} \delta^2 \\ \delta\sqrt{1-\delta^2} \\ \delta\sqrt{1-\delta^2} \\ 1-\delta^2 \end{pmatrix}$$

---

## 八、物理直觉总结

### 8.1 整个方案的物理图景

1. **发送端**：Alice准备四种双模相干态之一（由参数 $s_1, s_2$ 标记）
2. **信道**：两束光通过光纤到达测量端
3. **测量端**：Charlie对两束光做CV Bell测量（测量 $X_+$ 和 $P_-$），得到离散结果 $(k, l)$
4. **后处理**：根据测量结果和输入态的关联，产生随机数

### 8.2 关键物理要点

| 概念 | 物理意义 |
|------|---------|
| $\hat{X}, \hat{P}$ | 描述光场的两个正交分量，类似位置和动量 |
| 相干态 $|\alpha\rangle$ | 最接近经典光场的量子态 |
| $X_+ = \hat{X}_1 + \hat{X}_2$ | 两束光位置之和的联合测量 |
| $P_- = \hat{P}_1 - \hat{P}_2$ | 两束光动量之差的联合测量 |
| 高斯分布 | 量子涨落导致的测量结果统计分布 |
| $\delta = e^{-2\mu}$ | $\|+\alpha\rangle$ 和 $\|-\alpha\rangle$ 的重叠程度 |

### 8.3 公式的物理意义

- **均值公式**：$\mu_{+\sigma} = \sqrt{2}(s_1\sqrt{\mu_1} + s_2\sqrt{\mu_2})$ 表示不同输入态在 $X_+$ 测量中的平均输出不同
- **条件概率公式**：给出了每种输入态产生每种测量结果的概率，这是计算安全性的基础
- **态的向量表示**：将量子态编码为计算机可处理的向量形式，便于数值计算

---

## 九、进一步阅读建议

如果想深入学习相关内容，推荐阅读：
1. **量子光学基础**：了解相干态、光的 quadratures、不确定性原理
2. **Bell不等式与量子纠缠**：理解量子关联的本质
3. **连续变量量子信息**：CV Bell态测量、CV量子密钥分发
4. **半定规划（SDP）**：优化理论，用于计算安全速率

---

*本文档为《SDP_solve.tex》的详细解读，旨在帮助非量子物理专业的学生理解连续变量Bell态测量与MDIQRNG的原理。*
