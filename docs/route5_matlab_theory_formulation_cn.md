# Route5 对应 Matlab 脚本的理论表述、算法流程与变量说明

## 摘要

本文档面向 [`guessprobprimal_route5_hybrid_iq.m`](../src/matlab/guessprobprimal_route5_hybrid_iq.m) 这一份 Matlab 单文件参考脚本，采用偏理论论文的写法，对其背后的物理模型、概率生成方法、半定规划认证问题以及程序执行流程做统一说明。

这份脚本对应的是 `route5` 的“单点正式认证主流程”，而不是 Python 版搜索器的完整翻译。更准确地说，它实现的是：

1. 构造 `generalized coherent alphabet`；
2. 在截断 Fock 空间中构造相干态并压缩到有效支持空间；
3. 通过 `balanced beamsplitter + dual-homodyne / IQ coarse-graining` 生成离散输出概率；
4. 用单设备 prepare-and-measure MDI SDP 计算正式猜测概率；
5. 输出 `raw_H_min`、`p_guess` 与 `H_min`。

在当前脚本默认参数下，它对齐的是 Python `route5` 主线上一条已经 formal 验证过的强点：

- `cutoff = 4`
- `radii = [0.0, 0.85, 1.25]`
- `phase_values = 8` 个均匀相位
- `num_x_bins = 6`
- `num_p_bins = 2`
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`
- `max_inputs_to_certify = 3`

对应的 Python 结果文件可参考：

- [`route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json`](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)
- [`route5_fixed_intensity_080160_scale120.json`](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)

从项目结果看，前者对应的 formal 结果约为

$$
H_{\min} \approx 2.11639,
$$

后者对应固定光强主线时的 formal 结果约为

$$
H_{\min} \approx 2.10102.
$$

因此，`guessprobprimal_route5_hybrid_iq.m` 的理论意义，不是“随意构造一个新的 Matlab 脚本”，而是把 `route5` 已经在 Python 中验证过的核心协议，压缩成一份更便于导师和实验室直接检查的 Matlab 主流程版本。

---

## 1. 统一记号与脚本定位

### 1.1 本文档讨论的对象

本文只讨论以下脚本：

- [`guessprobprimal_route5_hybrid_iq.m`](../src/matlab/guessprobprimal_route5_hybrid_iq.m)

不讨论 Python 搜索器如何大范围扫描 `alphabet` 和 `IQ partition`。这些外围调度逻辑主要在：

- [`hybrid_iq.py`](../src/python/qrng_routes/route5/hybrid_iq.py)
- [`main.py`](../src/python/qrng_routes/route5/main.py)
- [`refine_queue.py`](../src/python/qrng_routes/route5/refine_queue.py)
- [`intensity_menu_search.py`](../src/python/qrng_routes/route5/intensity_menu_search.py)

本文的核心目标是回答三个问题：

1. 这份 Matlab 脚本到底在解什么数学问题；
2. 它的每个主要程序块分别对应什么物理对象；
3. 脚本里的变量赋值应当如何理解。

### 1.2 Route5 的协议层级

`route5` 可以看成一个“连续变量前端 + 离散输出认证”的混合协议。其结构不是纯粹的离散变量方案，也不是从头到尾的纯连续变量安全证明，而是如下四层组合：

1. 可信输入层：一组受信任的相干态字母表；
2. 物理测量层：分束器后的双正交连续变量测量；
3. 数字离散化层：在 IQ 平面上做轴对齐分箱；
4. 安全认证层：对离散输出概率做单设备 MDI SDP。

因此，`route5` 的最终认证对象虽然是离散输出，但其概率不是“手工指定”的，而是从一个连续变量测量模型出发生成的。

### 1.3 输入、输出与目标量

设本地可信字母表为

$$
\mathcal A = \{ \lvert \alpha_1 \rangle, \dots, \lvert \alpha_L \rangle \},
$$

其中每个 \(\alpha_\ell \in \mathbb C\) 是一个相干态振幅。脚本里：

- `local_alphas` 对应 \(\{\alpha_\ell\}_{\ell=1}^L\)；
- `num_local_states = L`。

双边输入采用笛卡尔积，因此联合输入索引记为

$$
s = (x,y), \qquad x,y \in \{1,\dots,L\}.
$$

对应联合输入态为

$$
\rho_{xy}
=
\lvert \alpha_x \rangle \langle \alpha_x \rvert
\otimes
\lvert \alpha_y \rangle \langle \alpha_y \rvert .
$$

脚本里：

- `joint_states` 存放所有 \(\rho_{xy}\)；
- `labels` 记录每个联合输入对应的零基标签 `[x-1, y-1]`；
- `num_inputs = L^2`。

中央测量的离散输出记为

$$
c \in \mathcal C = \{1,\dots,C\},
$$

其中

$$
C = N_x N_p,
$$

而脚本中：

- `num_x_bins = N_x`
- `num_p_bins = N_p`
- `num_outputs = C`

最终进入安全证明的观测概率表写为

$$
P(c \mid x,y).
$$

在脚本中，这就是 `probabilities(s,c)`。

正式认证的目标量是最大猜测概率

$$
p_{\mathrm{guess}},
$$

及相应的最小熵

$$
H_{\min} = -\log_2 p_{\mathrm{guess}}.
$$

---

## 2. Generalized Coherent Alphabet 的理论定义

### 2.1 相干态与复振幅

在单模 Fock 空间中，相干态定义为

$$
\lvert \alpha \rangle
=
e^{-|\alpha|^2/2}
\sum_{n=0}^{\infty}
\frac{\alpha^n}{\sqrt{n!}}
\lvert n \rangle .
$$

这里的 \(\alpha\) 是复振幅，可以写成极坐标形式

$$
\alpha = r e^{i\phi},
$$

其中：

- \(r = |\alpha|\) 是振幅模长；
- \(\phi = \arg(\alpha)\) 是相位；
- 平均光子数是
  $$
  \mu = |\alpha|^2 = r^2.
  $$

因此，脚本中的：

- `radii` 对应一组候选模长 \(r\)；
- `phase_values` 对应一组候选相位 \(\phi\)；
- `alpha_values` 或 `local_alphas` 对应真正进入计算的 \(\alpha = r e^{i\phi}\)。

### 2.2 字母表的生成方式

脚本支持两种输入方式。

第一种是直接给定复振幅列表：

$$
\{ \alpha_1, \dots, \alpha_L \}.
$$

第二种是由半径网格与相位网格生成：

$$
\alpha_{m,n} = r_m e^{i\phi_n},
\qquad
r_m \in \mathcal R,\ \phi_n \in \Phi.
$$

然后去掉重复值，得到最终字母表。

程序上对应：

- `build_alpha_values_from_grid(radii, phase_values)`
- `deduplicate_alphas(alpha_values)`

其数学作用可以概括为

$$
\mathcal A
=
\mathrm{unique}\bigl( \{ r e^{i\phi} : r\in\mathcal R,\ \phi\in\Phi \} \bigr).
$$

### 2.3 为什么叫 generalized coherent alphabet

在很多更传统的 CV 离散化方案里，输入态往往固定在：

- 单一半径；
- 少数几个等角相位；
- 或者单一物理自由度。

`route5` 则允许：

1. 多个半径；
2. 多个相位；
3. 最终以一组离散的相干态字母表进入同一个有限维安全模型。

因此它不是“只在一个圆环上取几个相位点”，而是更一般地允许多半径、多相位的 `coherent alphabet`。这就是 `generalized coherent alphabet` 的含义。

---

## 3. 截断 Fock 表示与有效支持空间

### 3.1 截断表示

Matlab 脚本不在无限维 Hilbert 空间中直接计算，而是在 Fock 截断维数

$$
d = \texttt{cutoff}
$$

下工作。于是相干态被替换为截断态

$$
\lvert \alpha ; d \rangle
=
\frac{1}{\mathcal N_d(\alpha)}
\sum_{n=0}^{d-1}
e^{-|\alpha|^2/2}
\frac{\alpha^n}{\sqrt{n!}}
\lvert n \rangle,
$$

其中归一化因子

$$
\mathcal N_d(\alpha)
=
\left(
\sum_{n=0}^{d-1}
e^{-|\alpha|^2}
\frac{|\alpha|^{2n}}{n!}
\right)^{1/2}.
$$

脚本中对应函数为：

- `build_truncated_coherent_density(alpha, M)`

它输出三样对象：

1. 截断 ket：`ket`
2. 截断密度矩阵：`rho_i = ket * ket'`
3. 其对角线：`rho_diag_i`

不过对于 `route5`，真正进入后续支持压缩和 SDP 的是完整密度矩阵，而不是仅仅对角线。

### 3.2 有效支持空间的动机

虽然每个输入态先在 \(d\) 维截断空间中表示，但所有本地输入态通常只张成一个较小的子空间。若把这些截断态矢量记为

$$
\lvert \psi_1 \rangle,\dots,\lvert \psi_L \rangle,
$$

则可以定义它们张成的支持子空间

$$
\mathcal H_{\mathrm{supp}}
=
\mathrm{span}\{ \lvert \psi_1 \rangle,\dots,\lvert \psi_L \rangle \}.
$$

Matlab 脚本通过 SVD 求得该子空间的正交归一基

$$
B_{\mathrm{loc}} \in \mathbb C^{d \times r},
$$

其中 \(r\) 是局部支持秩。

程序上对应：

- `support_basis_from_vectors(local_kets)`

这样做的理论意义是：

1. 不改变这些输入态之间的 Gram 结构；
2. 把后续 SDP 维数从原始截断维数 \(d\) 压缩到真实可达的支持维数 \(r\)；
3. 避免在“根本不会被输入态访问到”的方向上浪费变量。

### 3.3 投影后的本地输入态

对每个输入态 \(\rho_x\)，脚本计算其在支持基上的投影

$$
\tilde \rho_x
=
B_{\mathrm{loc}}^\dagger \rho_x B_{\mathrm{loc}}.
$$

对应程序：

- `project_density_to_basis(rho, basis)`

于是本地输入态由 \(\rho_x\) 变成 \(\tilde \rho_x\)，但对输入态之间的相对几何关系而言，这是精确保留的。

### 3.4 联合输入态

双边联合输入采用张量积：

$$
\tilde \rho_{xy}
=
\tilde \rho_x \otimes \tilde \rho_y.
$$

脚本中对应：

```matlab
joint_states{row_counter} = kron(reduced_local_states{x_idx}, reduced_local_states{y_idx});
```

若局部支持维数为 \(r\)，则联合空间维数变为

$$
r_{\mathrm{joint}} = r^2.
$$

联合支持基是

$$
B_{\mathrm{joint}} = B_{\mathrm{loc}} \otimes B_{\mathrm{loc}}.
$$

其后，所有中央测量 POVM 也会投影到这一联合支持上。

---

## 4. IQ 测量与离散化的理论结构

### 4.1 分束器变换

设两路输入模式的湮灭算符分别为 \(a\) 和 \(b\)。50:50 平衡分束器由幺正算符

$$
U_{\mathrm{BS}}
=
\exp \left[
\frac{\pi}{4}
\left(
a^\dagger b - a b^\dagger
\right)
\right]
$$

实现。

脚本里对应函数：

- `balanced_beamsplitter_unitary_route5(dimension)`

其内部通过产生算符和湮灭算符矩阵构造生成元，再调用矩阵指数 `expm` 得到 \(U_{\mathrm{BS}}\)。

### 4.2 双正交测量与 IQ 输出

`route5` 的中央测量不是一个简单的单模离散点击器，而是：

1. 输入的双模联合态先经过分束器；
2. 一路测 \(X\)；
3. 另一路测 \(P\)；
4. 得到连续输出 \((x,p)\)。

因此，理想连续 POVM 可以形式化写成

$$
M(x,p)
=
U_{\mathrm{BS}}^\dagger
\left(
\lvert x \rangle \langle x \rvert
\otimes
\lvert p \rangle \langle p \rvert
\right)
U_{\mathrm{BS}}.
$$

脚本不直接处理连续值 \((x,p)\)，而是对其做离散分箱。

### 4.3 轴对齐矩形分箱

设 \(X\) 方向边界为

$$
-\infty = b_0^{(x)} < b_1^{(x)} < \cdots < b_{N_x}^{(x)} = +\infty,
$$

设 \(P\) 方向边界为

$$
-\infty = b_0^{(p)} < b_1^{(p)} < \cdots < b_{N_p}^{(p)} = +\infty.
$$

则第 \((i,j)\) 个 IQ 矩形输出定义为

$$
R_{ij}
=
[b_{i-1}^{(x)}, b_i^{(x)}) \times [b_{j-1}^{(p)}, b_j^{(p)}).
$$

总输出数为

$$
C = N_x N_p.
$$

脚本中：

- `x_bounds` 和 `p_bounds` 对应上述边界；
- `output_labels(row,:) = [i-1, j-1]` 记录每个输出矩形的二维标签。

### 4.4 Power-spaced 边界的定义

若用户不直接给边界，脚本通过 `power_spaced_bounds(num_bins, finite_range, gamma)` 自动生成对称边界。其思路是：

先在区间 \([-1,1]\) 上等间距取点

$$
t_k = -1 + \frac{2k}{N}, \qquad k=0,\dots,N,
$$

然后映射为

$$
b_k
=
\mathrm{sign}(t_k)\, |t_k|^\gamma \, R,
$$

其中：

- \(R = \texttt{finite\_range}\)
- \(\gamma = \texttt{boundary\_gamma}\)

最后把最外层边界替换为 \(\pm\infty\)。

于是：

- 当 \(\gamma = 1\) 时，是有限边界内的等间距方案；
- 当 \(\gamma > 1\) 时，边界更聚集在中心；
- 当 \(\gamma < 1\) 时，边界更聚集在边缘。

这正是脚本里 `quadrature_range` 与 `boundary_gamma` 的数学含义。

---

## 5. 单模 Quadrature POVM 的数值构造

### 5.1 连续 quadrature POVM

对单模相位为 \(\theta\) 的 quadrature

$$
\hat X_\theta
=
\frac{1}{\sqrt 2}
\left(
a e^{-i\theta} + a^\dagger e^{i\theta}
\right),
$$

若第 \(k\) 个 bin 对应实轴区间 \(I_k\)，则理想 POVM 元可写为

$$
F_k^{(\theta)}
=
\int_{x \in I_k}
\lvert x_\theta \rangle \langle x_\theta \rvert\, dx.
$$

这里 \(\lvert x_\theta \rangle\) 是 \(\hat X_\theta\) 的广义本征态。

### 5.2 高斯-厄米特求积

脚本不直接符号积分，而是使用高斯-厄米特数值求积。设节点与权重为

$$
\{x_j, w_j\}_{j=1}^{K},
$$

则可把积分近似写成离散和。由于 Fock 基下的 quadrature 波函数由厄米函数给出，脚本先构造归一化厄米函数值

$$
\varphi_n(x_j),
\qquad
n=0,\dots,d-1.
$$

程序上由：

- `roots_hermite_golub_welsch(num_nodes)`
- `quadrature_hermite_data_route5(dimension, num_nodes)`

完成。

在离散化之后，一个 bin 的未修正 POVM 元可以理解为

$$
F_k^{(0)}
\approx
\sum_{j=1}^{K}
m_{k,j}\, w_j\,
\lvert x_j \rangle \langle x_j \rvert,
$$

其中 \(m_{k,j}\in\{0,1\}\) 表示第 \(j\) 个节点是否落在第 \(k\) 个 bin 中。

### 5.3 节点掩码与矩阵元

脚本中 `quadrature_povms_from_bounds_route5(...)` 先根据边界构造掩码矩阵

$$
M_{k j} = m_{k,j},
$$

然后在 `quadrature_povms_from_node_masks_route5(...)` 中生成 base elements。

若记

$$
V_{n j} = \varphi_n(x_j)\sqrt{w_j},
$$

则第 \(k\) 个 base element 可以理解为

$$
\bigl[F_k^{(0)}\bigr]_{nm}
\approx
\sum_{j=1}^{K}
V_{n j} V_{m j} m_{k,j}.
$$

这正对应了脚本中

```matlab
masked_values = weighted_values .* sqrt(masks(idx, :));
base_elements{idx} = masked_values * masked_values.';
```

所实现的外积结构。

### 5.4 从 \(X\) 到任意相位 \(\theta\)

Fock 基下，quadrature 旋转可由数算符相位因子给出，因此有

$$
\bigl[F_k^{(\theta)}\bigr]_{nm}
=
e^{-i\theta(n-m)}
\bigl[F_k^{(0)}\bigr]_{nm}.
$$

脚本中令

$$
\eta_n = e^{-i\theta n},
$$

则矩阵元旋转等价于

$$
F_k^{(\theta)}
=
\left( \eta \eta^\dagger \right) \odot F_k^{(0)},
$$

其中 \(\odot\) 表示逐元素乘法。

程序上对应：

```matlab
phase = exp(-1i * theta * number_indices);
rotated{idx} = (phase * phase') .* element;
```

这正好实现了

$$
(\eta_n \overline{\eta_m})_{nm}
=
e^{-i\theta n} e^{i\theta m}
=
e^{-i\theta(n-m)}.
$$

### 5.5 POVM 完备性的白化修正

由于数值求积与截断误差，简单相加通常只有

$$
S = \sum_k F_k^{(\theta)} \approx I,
$$

而不一定严格等于单位算符。脚本因此引入白化修正：

$$
\tilde F_k^{(\theta)}
=
S^{-1/2}
F_k^{(\theta)}
S^{-1/2}.
$$

于是自动满足

$$
\sum_k \tilde F_k^{(\theta)} = I.
$$

程序上对应：

- `complete_povm_via_whitening_route5(povm)`

这是一个数值修正步骤，其目的不是改变物理模型，而是把近似积分得到的一组正半定元素重新拉回严格完备的 POVM 集。

---

## 6. 从单模 POVM 到双模 IQ POVM

### 6.1 张量积输出效应

设 \(X\) 方向 coarse POVM 为 \(\{F_i^{(x)}\}_{i=1}^{N_x}\)，设 \(P\) 方向 coarse POVM 为 \(\{F_j^{(p)}\}_{j=1}^{N_p}\)，则分束器后的双模离散输出效应为

$$
E_{ij}
=
U_{\mathrm{BS}}^\dagger
\left(
F_i^{(x)} \otimes F_j^{(p)}
\right)
U_{\mathrm{BS}}.
$$

脚本中：

- `x_povms = quadrature_povms_from_bounds_route5(..., 0.0, ...)`
- `p_povms = quadrature_povms_from_bounds_route5(..., pi/2.0, ...)`
- `output_effect = kron(x_povms{x_idx}, p_povms{p_idx})`
- `povm{row_counter} = beamsplitter' * output_effect * beamsplitter`

因此，`route5` 的离散输出 `c` 本质上对应一个二维 bin \((i,j)\)。

### 6.2 投影到联合支持空间

由于输入态实际只活在支持子空间 \(\mathcal H_{\mathrm{supp}}^{\otimes 2}\) 中，因此脚本进一步把 \(E_{ij}\) 投影成

$$
\tilde E_{ij}
=
B_{\mathrm{joint}}^\dagger E_{ij} B_{\mathrm{joint}}.
$$

程序上对应：

- `project_povm_to_basis_list(povm, joint_basis)`

这样做之后，概率仍可精确写成 Born 形式：

$$
P(c \mid x,y)
=
\mathrm{Tr}\left( \tilde E_c \tilde \rho_{xy} \right).
$$

---

## 7. 概率表、Raw 熵与候选输入筛选

### 7.1 Born 概率

对每个联合输入态 \(\tilde \rho_{xy}\) 和每个离散输出 \(c\)，脚本计算

$$
P(c \mid x,y)
=
\mathrm{Tr}\left( \tilde E_c \tilde \rho_{xy} \right).
$$

程序上对应：

- `measurement_probabilities_from_states(states, povm)`

这一步得到完整概率表 `probabilities`。

### 7.2 Raw 最小熵

在不考虑更强兼容性约束时，对每个输入 \(s=(x,y)\) 可定义一个分布级别的 raw 猜测概率：

$$
p_{\mathrm{guess}}^{\mathrm{raw}}(s)
=
\max_c P(c \mid s).
$$

相应的 raw 最小熵为

$$
H_{\min}^{\mathrm{raw}}(s)
=
-\log_2 \max_c P(c \mid s).
$$

脚本中：

```matlab
raw_h = -log2(max(max(probabilities, [], 2), 1e-15));
```

然后取其中最大者，得到

$$
H_{\min,\mathrm{best}}^{\mathrm{raw}}
=
\max_s H_{\min}^{\mathrm{raw}}(s).
$$

这只是一个“哪几个输入看起来最随机”的初筛指标。

### 7.3 为什么要只认证前若干个目标输入

由于正式 SDP 成本高，脚本不必对全部 \(L^2\) 个联合输入都做 full certification，而是先按 `raw_h` 从高到低排序，再只保留前

$$
K = \texttt{max\_inputs\_to\_certify}
$$

个候选。

程序上对应：

- `sort_target_indices_desc(raw_h)`
- `candidate_order = candidate_order(1:min(...))`

这一设计对应 Python `route5` 的真实工作流：先用 raw 熵做便宜的候选筛选，再把算力集中在最有希望的 target input 上。

---

## 8. Single-device Guessing SDP 的理论模型

### 8.1 SDP 的索引结构

对某个固定的目标输入 \(s^\star\)，脚本求解的不是一个纯粹的 classical LP，而是一个单设备 prepare-and-measure MDI SDP。

定义：

- 真实输出索引 \(c \in \{1,\dots,C\}\)；
- Eve 的猜测索引 \(e \in \{1,\dots,C\}\)；
- SDP 变量是一组厄米半正定算符
  $$
  M_{c,e} \succeq 0.
  $$

程序里通过三维变量

```matlab
variable M_ops(dimension, dimension, num_outputs * num_outputs) hermitian semidefinite
```

来承载这些算符，并用

```matlab
op_idx = operator_index_route5(c_idx, e_idx, num_outputs);
```

实现 \((c,e)\) 到一维切片索引的映射。

### 8.2 目标函数

若目标输入记为 \(\rho_{s^\star}\)，则最大猜测概率写成

$$
p_{\mathrm{guess}}(s^\star)
=
\max
\sum_{c=1}^{C}
\mathrm{Tr}\left( M_{c,c} \rho_{s^\star} \right).
$$

直观解释是：当真实输出也是 \(c\)，而 Eve 也猜 \(c\) 时，才算猜对，因此只累加对角块 \(M_{c,c}\)。

程序中对应：

```matlab
for c_idx = 1:num_outputs
    op_idx = operator_index_route5(c_idx, c_idx, num_outputs);
    objective_value = objective_value + real(trace(M_ops(:, :, op_idx) * rho_star));
end
maximize(objective_value)
```

### 8.3 统计一致性约束

对每个输入态 \(\rho_s\) 和每个真实输出 \(c\)，必须满足

$$
\sum_{e=1}^{C}
\mathrm{Tr}(M_{c,e}\rho_s)
=
P(c \mid s).
$$

这表示：不管 Eve 最后猜什么，真实测量落在 \(c\) 的总概率必须和观测统计一致。

程序上对应：

```matlab
for s_idx = 1:num_inputs
    for c_idx = 1:num_outputs
        stats_sum == probabilities(s_idx, c_idx);
    end
end
```

### 8.4 完备性约束

对每个猜测标签 \(e\)，一组 \(\{M_{c,e}\}_c\) 的和必须正比于单位算符：

$$
\sum_{c=1}^{C} M_{c,e} = p_e I,
$$

其中 \(p_e \ge 0\)，且

$$
\sum_{e=1}^{C} p_e = 1.
$$

这组约束对应单设备 guessing SDP 的标准结构，保证每个猜测分支本身构成一个合法的测量分解。

程序里对应：

```matlab
variable p_e(num_outputs) nonnegative
...
complete_sum == p_e(e_idx) * identity_matrix;
sum(p_e) == 1;
```

### 8.5 正式最小熵

若 SDP 最优值为

$$
\hat p_{\mathrm{guess}},
$$

则正式认证的最小熵为

$$
\hat H_{\min}
=
-\log_2 \hat p_{\mathrm{guess}}.
$$

脚本中：

```matlab
result_struct.p_guess = cvx_optval;
result_struct.H_min = -log2(cvx_optval);
```

这就是 `route5` 真正关心的最终量。

### 8.6 `raw_H_min` 与 formal `H_min` 的区别

需要特别强调：

$$
H_{\min}^{\mathrm{raw}}
\neq
H_{\min}^{\mathrm{formal}}.
$$

前者只看单个输入对应的输出分布是否平坦；后者是在所有与观测统计兼容的量子攻击策略中，求一个最坏情况上仍能保证的随机性下界。

因此，脚本的工作流是：

1. 用 raw 熵找候选 target；
2. 用正式 SDP 给出真正可报告的 `H_min`。

---

## 9. 程序流程与理论对象的一一对应

### 9.1 第一步：参数解析与字母表构造

脚本开始时定义：

- `cutoff`
- `alpha_values`
- `radii`
- `phase_values`
- `num_x_bins`
- `num_p_bins`
- `quadrature_range`
- `boundary_gamma`
- `num_quadrature_nodes`
- `max_inputs_to_certify`

理论上，这一步是在选定协议实例

$$
\Pi
=
\left(
\mathcal A,\ d,\ N_x,\ N_p,\ R,\ \gamma,\ K
\right).
$$

### 9.2 第二步：构造本地截断相干态

对每个 \(\alpha_\ell\) 计算：

$$
\rho_\ell = \lvert \alpha_\ell; d \rangle \langle \alpha_\ell; d \rvert.
$$

对应函数：

- `build_truncated_coherent_density`

### 9.3 第三步：做支持压缩

从 \(\{\rho_\ell\}\) 对应的 ket 向量构造支持基 \(B_{\mathrm{loc}}\)，并投影得到 \(\tilde \rho_\ell\)。

对应函数：

- `support_basis_from_vectors`
- `project_density_to_basis`

### 9.4 第四步：构造全部联合输入

对所有 \((x,y)\) 构造

$$
\tilde \rho_{xy} = \tilde \rho_x \otimes \tilde \rho_y.
$$

对应脚本的双重循环与 `kron(...)`。

### 9.5 第五步：生成 IQ 边界与 POVM

若 `x_bounds` / `p_bounds` 为空，则调用 `power_spaced_bounds` 自动生成；随后构造：

1. \(X\) 方向粗粒化 POVM；
2. \(P\) 方向粗粒化 POVM；
3. 分束器后的双模离散 POVM；
4. 投影到联合支持上的 POVM。

对应函数链：

- `power_spaced_bounds`
- `quadrature_povms_from_bounds_route5`
- `quadrature_povms_from_node_masks_route5`
- `complete_povm_via_whitening_route5`
- `dual_homodyne_povm_route5`
- `project_povm_to_basis_list`

### 9.6 第六步：生成概率表

对每个输入态、每个输出效应计算

$$
P(c \mid x,y) = \mathrm{Tr}(\tilde E_c \tilde \rho_{xy}).
$$

对应函数：

- `measurement_probabilities_from_states`

### 9.7 第七步：raw 粗筛

计算每个联合输入对应的 raw 熵

$$
H_{\min}^{\mathrm{raw}}(x,y)
=
-\log_2 \max_c P(c \mid x,y),
$$

然后保留前 `max_inputs_to_certify` 个候选。

### 9.8 第八步：正式 SDP 认证

对每个候选目标输入，调用

- `solve_single_device_guessing_route5`

求出正式 \(p_{\mathrm{guess}}\) 与 \(H_{\min}\)，再在所有候选中选出 formal 最好的那个目标输入。

---

## 10. 默认参数的物理意义与当前结果

### 10.1 默认参数实例

脚本默认参数为：

$$
d = 4,
\qquad
\mathcal R = \{0,\ 0.85,\ 1.25\},
\qquad
\Phi = \left\{ \frac{2\pi k}{8} : k=0,\dots,7 \right\},
$$

并取

$$
N_x = 6,\qquad N_p = 2,
$$

因此总输出数为

$$
C = 12.
$$

边界自动生成参数为

$$
R = 1.8,
\qquad
\gamma = 1.0.
$$

这是一条已经被 Python 主线 formal 验证过的 `route5` 强点配置，而不是随意给出的示例数值。

### 10.2 自由主线与固定光强主线

从项目现有结果看：

1. 自由搜索强点
   $$
   \texttt{radii} = [0.0,\ 0.85,\ 1.25]
   $$
   对应 formal 结果约为
   $$
   H_{\min} \approx 2.11639.
   $$

2. 固定光强主线 `[0,80,160]`
   在当前 `scale120` 映射下，对应半径近似为
   $$
   [0.0,\ 0.8485,\ 1.2],
   $$
   formal 结果约为
   $$
   H_{\min} \approx 2.10102.
   $$

这说明：

- `route5` 不只是自由搜索时偶然超过 `2 bit`；
- 即便把字母表收回到固定光强主线上，formal 结果仍能保持在 `2 bit` 以上。

### 10.3 为什么 Route5 能超过 2 bit

从模型结构上看，`route5` 的优势主要来自三点：

1. 输入字母表比传统单半径相位编码更丰富；
2. 中央测量不是低速单光子点击，而是高速 CV 前端；
3. 最终离散输出由二维 IQ 分区提供，因此在相同输出数下，几何结构比一维粗粒化更灵活。

当然，这并不意味着任何参数都能自动得到 `H_min > 2`。真正起作用的是“输入几何 + IQ 分区 + SDP 约束”三者的匹配。

---

## 11. 主要变量的数学含义

下面把脚本中最关键的变量与数学对象做一次集中对应。

### 11.1 输入态相关

- `cutoff`
  对应截断维数 \(d\)。

- `radii`
  对应相干态模长集合 \(\mathcal R\)。

- `phase_values`
  对应相干态相位集合 \(\Phi\)。

- `alpha_values`
  对应显式输入的复振幅列表 \(\{\alpha_\ell\}\)。

- `local_alphas`
  对应去重后的最终本地字母表。

- `local_states`
  对应每个 \(\rho_\ell = \lvert \alpha_\ell; d \rangle\langle \alpha_\ell; d \rvert\)。

- `local_basis`
  对应本地支持子空间基 \(B_{\mathrm{loc}}\)。

- `reduced_local_states`
  对应投影后的 \(\tilde \rho_\ell\)。

### 11.2 联合输入相关

- `joint_states`
  对应全部联合输入 \(\tilde \rho_{xy}\)。

- `labels`
  对应每个输入的 `(x,y)` 标签。

- `joint_basis`
  对应 \(B_{\mathrm{joint}} = B_{\mathrm{loc}} \otimes B_{\mathrm{loc}}\)。

### 11.3 IQ 分区相关

- `num_x_bins = N_x`
- `num_p_bins = N_p`
- `x_bounds`
  对应 \(X\) 方向边界；
- `p_bounds`
  对应 \(P\) 方向边界；
- `quadrature_range = R`
- `boundary_gamma = \gamma`

### 11.4 概率与熵相关

- `probabilities`
  对应 \(P(c \mid x,y)\)。

- `raw_h`
  对应每个输入的 raw 熵
  $$
  -\log_2 \max_c P(c \mid x,y).
  $$

- `raw_best_H_min`
  对应 raw 最优输入的分布级熵。

- `p_guess`
  对应正式 SDP 最优值。

- `H_min`
  对应正式认证熵
  $$
  -\log_2 p_{\mathrm{guess}}.
  $$

### 11.5 认证候选相关

- `max_inputs_to_certify`
  对应正式进入 SDP 的候选输入数 \(K\)。

- `raw_best_target`
  raw 熵最高的联合输入标签。

- `certified_best_target`
  formal 认证后真正最优的联合输入标签。

---

## 12. 这份 Matlab 脚本与 Python 搜索器的关系

需要明确写清：

1. 这份 Matlab 脚本实现的是 `route5` 的单点主流程；
2. 它并不包含 Python 中的大规模 `alphabet-search` 和 `partition-search`；
3. Python 搜索器负责“找参数”；
4. Matlab 脚本负责“给定参数后，把物理模型和正式 SDP 主流程说清楚并跑出来”。

因此，不能把“这份脚本没有整个搜索器”理解成它不完整。恰恰相反，它更适合作为导师审阅的协议核心说明文件。

---

## 13. 理论边界与使用注意

### 13.1 这不是实验数据版脚本

当前脚本内部直接按物理模型生成

$$
P(c \mid x,y)=\mathrm{Tr}(E_c \rho_{xy}),
$$

并不从实验文件中读取 IQ 原始数据。因此它目前对应的是：

- 理论原型；
- 数值验证；
- 与 Python 主线一致的 formal 主流程。

如果后续实验室提供真实 route5 概率数据，这份脚本的理论框架仍然成立，但“概率生成”那一段应被实验概率表替换。

### 13.2 截断与求积误差

脚本包含两类数值近似：

1. Fock 截断 `cutoff = d`；
2. 高斯-厄米特求积节点数 `num_quadrature_nodes = K`。

因此，任何数值结果都应理解为：

$$
\text{给定 } d \text{ 与 } K \text{ 时的数值认证结果}.
$$

不过脚本已经通过：

- 支持压缩；
- POVM 白化；
- 与 Python 现有主线的一致配置

来尽量减少这些近似带来的额外误差。

### 13.3 当前环境中的验证边界

本仓库当前工作流里，这份 Matlab 脚本已经完成静态对齐与文档化，但在当前终端环境中没有 Matlab/Octave 运行器，因此本文档的定位仍应是：

- 理论解释正确；
- 与 Python 主线结构对齐；
- Matlab 数值运行应在导师或实验室本机环境中进一步复核。

这并不影响本文对其理论原理与程序结构的说明。

---

## 14. 一句话总结

[`guessprobprimal_route5_hybrid_iq.m`](../src/matlab/guessprobprimal_route5_hybrid_iq.m) 的理论本质可以概括为：

$$
\text{generalized coherent alphabet}
\;+\;
\text{balanced beamsplitter + dual-homodyne}
\;+\;
\text{IQ coarse-graining}
\;+\;
\text{single-device MDI SDP}.
$$

它把 `route5` 中最关键的物理与安全证明主线压缩进一份 Matlab 单文件里，使导师或实验室可以不依赖 Python 搜索器，也能直接检查：

1. 输入态是怎么定义的；
2. 中央测量概率是怎么来的；
3. 离散输出是怎么构造的；
4. 正式最小熵是怎么由 SDP 算出来的。

从现有项目结果看，这条主线已经在理论数值层面给出：

$$
H_{\min} > 2
$$

的清晰证据；而这份 Matlab 脚本，就是对这条主线核心机制的单文件化表达。
