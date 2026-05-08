# Route5 理论表述、算法流程与变量说明

<a id="toc"></a>
## 目录

- [Route5 理论表述、算法流程与变量说明](#route5-理论表述算法流程与变量说明)
  - [目录](#目录)
  - [摘要](#摘要)
  - [1. 统一记号与脚本定位](#1-统一记号与脚本定位)
    - [1.1 本文档讨论的对象](#11-本文档讨论的对象)
    - [1.2 Route5 的协议层级](#12-route5-的协议层级)
    - [1.3 输入、输出与目标量](#13-输入输出与目标量)
  - [2. Generalized Coherent Alphabet 的理论定义](#2-generalized-coherent-alphabet-的理论定义)
    - [2.1 相干态与复振幅](#21-相干态与复振幅)
    - [2.2 字母表的生成方式](#22-字母表的生成方式)
    - [2.3 为什么叫 generalized coherent alphabet](#23-为什么叫-generalized-coherent-alphabet)
  - [3. 截断 Fock 表示与有效支持空间](#3-截断-fock-表示与有效支持空间)
    - [3.1 截断表示](#31-截断表示)
    - [3.2 有效支持空间的动机](#32-有效支持空间的动机)
    - [3.3 投影后的本地输入态](#33-投影后的本地输入态)
    - [3.4 联合输入态](#34-联合输入态)
  - [4. IQ 测量与离散化的理论结构](#4-iq-测量与离散化的理论结构)
    - [4.1 分束器变换](#41-分束器变换)
    - [4.2 双正交测量与 IQ 输出](#42-双正交测量与-iq-输出)
    - [4.3 轴对齐矩形分箱](#43-轴对齐矩形分箱)
    - [4.4 Power-spaced 边界的定义](#44-power-spaced-边界的定义)
  - [5. 单模 Quadrature POVM 的数值构造](#5-单模-quadrature-povm-的数值构造)
    - [5.1 连续 quadrature POVM](#51-连续-quadrature-povm)
    - [5.2 高斯-厄米特求积](#52-高斯-厄米特求积)
    - [5.3 节点掩码与矩阵元](#53-节点掩码与矩阵元)
    - [5.4 从 (X) 到任意相位 (\\theta)](#54-从-x-到任意相位-theta)
    - [5.5 POVM 完备性的白化修正](#55-povm-完备性的白化修正)
  - [6. 从单模 POVM 到双模 IQ POVM](#6-从单模-povm-到双模-iq-povm)
    - [6.1 张量积输出效应](#61-张量积输出效应)
    - [6.2 投影到联合支持空间](#62-投影到联合支持空间)
    - [6.3 解析 Gaussian 矩形概率后端](#63-解析-gaussian-矩形概率后端)
  - [7. 概率表、Raw 熵与候选输入筛选](#7-概率表raw-熵与候选输入筛选)
    - [7.1 Trace POVM 的 Born 概率](#71-trace-povm-的-born-概率)
    - [7.2 解析 Gaussian 概率](#72-解析-gaussian-概率)
    - [7.3 Raw 最小熵](#73-raw-最小熵)
    - [7.4 为什么要只认证前若干个目标输入](#74-为什么要只认证前若干个目标输入)
  - [8. Single-device Guessing SDP 的理论模型](#8-single-device-guessing-sdp-的理论模型)
    - [8.1 SDP 的索引结构](#81-sdp-的索引结构)
    - [8.2 目标函数](#82-目标函数)
    - [8.3 统计一致性约束](#83-统计一致性约束)
    - [8.4 完备性约束](#84-完备性约束)
    - [8.5 正式最小熵](#85-正式最小熵)
    - [8.6 `raw_H_min` 与 formal `H_min` 的区别](#86-raw_h_min-与-formal-h_min-的区别)
  - [9. 程序流程与理论对象的一一对应](#9-程序流程与理论对象的一一对应)
    - [9.1 第一步：参数解析与字母表构造](#91-第一步参数解析与字母表构造)
    - [9.2 第二步：构造本地截断相干态](#92-第二步构造本地截断相干态)
    - [9.3 第三步：做支持压缩](#93-第三步做支持压缩)
    - [9.4 第四步：构造全部联合输入](#94-第四步构造全部联合输入)
    - [9.5 第五步：生成 IQ 边界与 POVM](#95-第五步生成-iq-边界与-povm)
    - [9.6 第六步：生成概率表](#96-第六步生成概率表)
    - [9.7 第七步：raw 粗筛](#97-第七步raw-粗筛)
    - [9.8 第八步：正式 SDP 认证](#98-第八步正式-sdp-认证)
  - [10. 默认参数的物理意义与当前结果](#10-默认参数的物理意义与当前结果)
    - [10.1 默认参数实例](#101-默认参数实例)
    - [10.2 Trace 主线与固定光强主线](#102-trace-主线与固定光强主线)
    - [10.3 解析概率版本的结果与诊断](#103-解析概率版本的结果与诊断)
    - [10.4 为什么 Route5 在旧 Trace 主线下能超过 2 bit](#104-为什么-route5-在旧-trace-主线下能超过-2-bit)
  - [11. 主要变量的数学含义](#11-主要变量的数学含义)
    - [11.1 输入态相关](#111-输入态相关)
    - [11.2 联合输入相关](#112-联合输入相关)
    - [11.3 IQ 分区相关](#113-iq-分区相关)
    - [11.4 概率与熵相关](#114-概率与熵相关)
    - [11.5 认证候选相关](#115-认证候选相关)
  - [12. Python 主线与 Matlab 参考脚本的关系](#12-python-主线与-matlab-参考脚本的关系)
  - [13. 理论边界、实验可行性与使用注意](#13-理论边界实验可行性与使用注意)
    - [13.1 这不是实验数据版脚本](#131-这不是实验数据版脚本)
    - [13.2 实验室需要完成的工作包](#132-实验室需要完成的工作包)
    - [13.3 实验室应如何给出数据](#133-实验室应如何给出数据)
    - [13.4 推荐的数据文件结构与必须附带的元数据](#134-推荐的数据文件结构与必须附带的元数据)
    - [13.5 所需物理设备与典型实验方法](#135-所需物理设备与典型实验方法)
    - [13.6 Route5 在实验上是可行的，但应视为一条新的 IQ 路线](#136-route5-在实验上是可行的但应视为一条新的-iq-路线)
    - [13.7 固定光强结果的真实含义与风险边界](#137-固定光强结果的真实含义与风险边界)
    - [13.8 字母表规模、统计负担与认证覆盖范围](#138-字母表规模统计负担与认证覆盖范围)
    - [13.9 主要实验风险与建议检查项](#139-主要实验风险与建议检查项)
    - [13.10 截断、求积与后端误差](#1310-截断求积与后端误差)
    - [13.11 解析概率版本的理论意义与当前限制](#1311-解析概率版本的理论意义与当前限制)
    - [13.12 当前环境中的验证边界](#1312-当前环境中的验证边界)
  - [14. 一句话总结](#14-一句话总结)

<a id="summary"></a>
## 摘要

本文档以 Python 主线实现 [`hybrid_iq.py`](../src/python/qrng_routes/route5/hybrid_iq.py) 为主要讨论对象，采用偏理论论文的写法，对 `route5` 背后的物理模型、概率生成方法、半定规划认证问题以及程序执行流程做统一说明；与此同时，也把 [`guessprobprimal_route5_hybrid_iq.m`](../src/matlab/guessprobprimal_route5_hybrid_iq.m) 视为一份便于导师直接阅读的 Matlab 参考实现。

如果只想集中查看“概率是怎么算出来的”和“SDP 是怎么构造并求解的”，可另见专题补充说明：
[`route5_probability_sdp_explanation_cn.md`](./route5_probability_sdp_explanation_cn.md)。

当前 Python 主线支持两类概率后端：

1. `trace_povm`：在截断 Fock 空间中构造双 Homodyne POVM，再以 Born 迹公式生成离散概率；
2. `analytic_gaussian_rectangles`：把输入视为理想无限维相干态，直接用解析高斯积分计算每个 IQ 矩形 bin 的概率。

这两类后端共用同一套输入字母表、IQ 分箱和 single-device guessing SDP，但它们对“概率层”采取的建模口径不同。因此，本文不仅解释 `route5` 的主协议逻辑，也会把这两种概率版本的理论关系、数值结果与当前限制一并写清。

若只看当前 Python 工作流的正式主线，它更准确地实现的是：

1. 构造 `generalized coherent alphabet`；
2. 在截断 Fock 空间中构造相干态并压缩到有效支持空间；
3. 通过 `balanced beamsplitter + dual-homodyne / IQ coarse-graining` 生成离散输出概率；
4. 用单设备 prepare-and-measure MDI SDP 计算正式猜测概率；
5. 输出 `raw_H_min`、`p_guess` 与 `H_min`。

在当前 Python 主线默认强点配置下，较重要的一组参数是：

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

因此，这份文档的理论意义，不是“单独解释一个 Matlab 单文件”，而是把 `route5` 已经在 Python 中实现和探索过的核心协议，整理成一份更适合导师、实验同学和后续接手者统一查阅的总说明；其中 Matlab 文件只是 Python 主线的一个单点参考镜像。

同时也需要明确：当前 `route5` 给出的

$$
H_{\min} > 2
$$

仍属于“理论模型 + 数值认证”结论，而不是“当前实验已经闭环完成”的结论。更准确地说，`route5` 更像一条独立的 `IQ / dual-homodyne` 新实验路线：它的认证层已经清楚，但若要闭合成实验结果，还需要 route5 自己的二维 IQ / histogram / coarse-grained `P_{\mathrm{exp}}(c|x,y)` 数据，以及把固定光强版中的半径映射关系改成实验独立标定，而不是继续作为搜索参数。

此外，最近补入的解析概率版本复核表明：在当前 `cutoff=4`、截断支持压缩的 trusted-state 模型下，`analytic_gaussian_rectangles` 后端会把历史强点推成 `formal infeasible`。这说明 `route5` 目前应被理解为：

- `trace_povm` 后端下，已经出现了 `H_{\min} > 2` 的强候选窗口；
- `analytic_gaussian_rectangles` 后端下，当前 formal 模型与概率层尚未完全自洽；
- 因而“哪一条结果可以直接当正式实验口径汇报”这件事，还需要结合数值收敛、模型一致性和实验标定三方面一起判断。

---

<a id="sec-1"></a>
## 1. 统一记号与脚本定位

<a id="sec-1-1"></a>
### 1.1 本文档讨论的对象

本文主讨论以下 Python 文件：

- [`hybrid_iq.py`](../src/python/qrng_routes/route5/hybrid_iq.py)
- [`main.py`](../src/python/qrng_routes/route5/main.py)
- [`refine_queue.py`](../src/python/qrng_routes/route5/refine_queue.py)
- [`intensity_menu_search.py`](../src/python/qrng_routes/route5/intensity_menu_search.py)
- [`node_convergence_scan.py`](../src/python/qrng_routes/route5/node_convergence_scan.py)
- [`analytic_backend_diagnostics.py`](../src/python/qrng_routes/route5/analytic_backend_diagnostics.py)

同时，把以下 Matlab 文件视为辅助参考实现：

- [`guessprobprimal_route5_hybrid_iq.m`](../src/matlab/guessprobprimal_route5_hybrid_iq.m)

本文的核心目标是回答三个问题：

1. Python 主线里的 `route5` 到底在解什么数学问题；
2. `trace_povm` 与 `analytic_gaussian_rectangles` 两个概率版本分别对应什么理论对象；
3. 代码里的核心变量、算法步骤和当前数值结果应当如何理解。

<a id="sec-1-2"></a>
### 1.2 Route5 的协议层级

`route5` 可以看成一个“连续变量前端 + 离散输出认证”的混合协议。其结构不是纯粹的离散变量方案，也不是从头到尾的纯连续变量安全证明，而是如下四层组合：

1. 可信输入层：一组受信任的相干态字母表；
2. 物理测量层：分束器后的双正交连续变量测量；
3. 数字离散化层：在 IQ 平面上做轴对齐分箱；
4. 安全认证层：对离散输出概率做单设备 MDI SDP。

因此，`route5` 的最终认证对象虽然是离散输出，但其概率不是“手工指定”的，而是从一个连续变量测量模型出发生成的。

**通俗理解：** 可以把 `route5` 想成“先做一次高速的连续变量测量，得到很多 IQ 散点；再把这些散点按区域编号；最后不再直接研究散点本身，而是研究每个编号出现的概率，并对这些编号做随机性认证”。

<a id="sec-1-3"></a>
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
\rho_{xy}=
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

**通俗理解：** 这一节其实只是在回答三个最基础的问题：“我们往设备里送什么态”“设备会吐出什么离散结果”“最后到底想优化哪个数字”。其中 \(H_{\min}\) 越大，表示攻击者越难猜中输出。

---

<a id="sec-2"></a>
## 2. Generalized Coherent Alphabet 的理论定义

<a id="sec-2-1"></a>
### 2.1 相干态与复振幅

在单模 Fock 空间中，相干态定义为

$$
\lvert \alpha \rangle=
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

**通俗理解：** 如果把相空间想成平面，那么 \(\alpha\) 就是平面上的一个点；`radii` 决定这个点离原点多远，`phase_values` 决定它朝哪个方向。平均光子数 \(\mu\) 本质上就是“离原点的距离平方”。

<a id="sec-2-2"></a>
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
\mathcal A=
\mathrm{unique}\bigl( \{ r e^{i\phi} : r\in\mathcal R,\ \phi\in\Phi \} \bigr).
$$

**通俗理解：** 这一步可以理解为“先挑几圈半径，再在每一圈上挑几个角度，把这些点全列出来，重复的删掉”，最后得到真正送进协议的一组测试态。

<a id="sec-2-3"></a>
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

<a id="sec-3"></a>
## 3. 截断 Fock 表示与有效支持空间

<a id="sec-3-1"></a>
### 3.1 截断表示

Matlab 脚本不在无限维 Hilbert 空间中直接计算，而是在 Fock 截断维数

$$
d = \texttt{cutoff}
$$

下工作。于是相干态被替换为截断态

$$
\lvert \alpha ; d \rangle=
\frac{1}{\mathcal N_d(\alpha)}
\sum_{n=0}^{d-1}
e^{-|\alpha|^2/2}
\frac{\alpha^n}{\sqrt{n!}}
\lvert n \rangle,
$$

其中归一化因子

$$
\mathcal N_d(\alpha)=
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

**通俗理解：** `cutoff` 可以粗略理解为“我们只保留 Fock 空间前 \(d\) 层来做数值计算”。这像是把无限长的展开式截成前面几项来近似，只要主要概率质量都落在前几层，近似就会比较稳。

<a id="sec-3-2"></a>
### 3.2 有效支持空间的动机

虽然每个输入态先在 \(d\) 维截断空间中表示，但所有本地输入态通常只张成一个较小的子空间。若把这些截断态矢量记为

$$
\lvert \psi_1 \rangle,\dots,\lvert \psi_L \rangle,
$$

则可以定义它们张成的支持子空间

$$
\mathcal H_{\mathrm{supp}}=
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

**通俗理解：** 这里是在做一次“瘦身”。虽然我们一开始把每个态都写在 \(d\) 维空间里，但这些态真正活动的方向往往没有那么多，所以可以先找出它们真正用到的那一小块空间，再在那里面解 SDP，会省很多算力。

<a id="sec-3-3"></a>
### 3.3 投影后的本地输入态

对每个输入态 \(\rho_x\)，脚本计算其在支持基上的投影

$$
\tilde \rho_x=
B_{\mathrm{loc}}^\dagger \rho_x B_{\mathrm{loc}}.
$$

对应程序：

- `project_density_to_basis(rho, basis)`

于是本地输入态由 \(\rho_x\) 变成 \(\tilde \rho_x\)，但对输入态之间的相对几何关系而言，这是精确保留的。

<a id="sec-3-4"></a>
### 3.4 联合输入态

双边联合输入采用张量积：

$$
\tilde \rho_{xy}=
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

<a id="sec-4"></a>
## 4. IQ 测量与离散化的理论结构

<a id="sec-4-1"></a>
### 4.1 分束器变换

设两路输入模式的湮灭算符分别为 \(a\) 和 \(b\)。50:50 平衡分束器由幺正算符

$$
U_{\mathrm{BS}}=
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

<a id="sec-4-2"></a>
### 4.2 双正交测量与 IQ 输出

`route5` 的中央测量不是一个简单的单模离散点击器，而是：

1. 输入的双模联合态先经过分束器；
2. 一路测 \(X\)；
3. 另一路测 \(P\)；
4. 得到连续输出 \((x,p)\)。

因此，理想连续 POVM 可以形式化写成

$$
M(x,p)=
U_{\mathrm{BS}}^\dagger
\left(
\lvert x \rangle \langle x \rvert
\otimes
\lvert p \rangle \langle p \rvert
\right)
U_{\mathrm{BS}}.
$$

脚本不直接处理连续值 \((x,p)\)，而是对其做离散分箱。

**通俗理解：** 设备原本会给出一个连续坐标点 \((x,p)\)。但为了后面的离散随机性认证，我们不直接记这个精确坐标，而是只记它落进了哪个小格子里。

<a id="sec-4-3"></a>
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
R_{ij}=
[b_{i-1}^{(x)}, b_i^{(x)}) \times [b_{j-1}^{(p)}, b_j^{(p)}).
$$

总输出数为

$$
C = N_x N_p.
$$

脚本中：

- `x_bounds` 和 `p_bounds` 对应上述边界；
- `output_labels(row,:) = [i-1, j-1]` 记录每个输出矩形的二维标签。

**通俗理解：** 这一节就是在把 IQ 平面切成若干个矩形“格子”。以后每次测量不再输出一个连续小数点，而是输出“落在第几个格子里”。

<a id="sec-4-4"></a>
### 4.4 Power-spaced 边界的定义

若用户不直接给边界，脚本通过 `power_spaced_bounds(num_bins, finite_range, gamma)` 自动生成对称边界。其思路是：

先在区间 \([-1,1]\) 上等间距取点

$$
t_k = -1 + \frac{2k}{N}, \qquad k=0,\dots,N,
$$

然后映射为

$$
b_k=
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

**通俗理解：** `quadrature_range` 决定我们大致看多宽的 IQ 范围，`\gamma` 决定这些格子是“中间切得更细”还是“边上切得更细”。它们共同决定了 coarse-graining 的形状。

---

<a id="sec-5"></a>
## 5. 单模 Quadrature POVM 的数值构造

<a id="sec-5-1"></a>
### 5.1 连续 quadrature POVM

对单模相位为 \(\theta\) 的 quadrature

$$
\hat X_\theta=
\frac{1}{\sqrt 2}
\left(
a e^{-i\theta} + a^\dagger e^{i\theta}
\right),
$$

若第 \(k\) 个 bin 对应实轴区间 \(I_k\)，则理想 POVM 元可写为

$$
F_k^{(\theta)}=
\int_{x \in I_k}
\lvert x_\theta \rangle \langle x_\theta \rvert\, dx.
$$

这里 \(\lvert x_\theta \rangle\) 是 \(\hat X_\theta\) 的广义本征态。

<a id="sec-5-2"></a>
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

**通俗理解：** 高斯-厄米特求积可以理解成“用一批精心挑选的采样点和权重去近似连续积分”。也就是说，我们没有真的把整条实轴无穷细地积分，而是用少量高质量采样点把它近似出来。

<a id="sec-5-3"></a>
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

<a id="sec-5-4"></a>
### 5.4 从 \(X\) 到任意相位 \(\theta\)

Fock 基下，quadrature 旋转可由数算符相位因子给出，因此有

$$
\bigl[F_k^{(\theta)}\bigr]_{nm}=
e^{-i\theta(n-m)}
\bigl[F_k^{(0)}\bigr]_{nm}.
$$

脚本中令

$$
\eta_n = e^{-i\theta n},
$$

则矩阵元旋转等价于

$$
F_k^{(\theta)}=
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
(\eta_n \overline{\eta_m})_{nm}=
e^{-i\theta n} e^{i\theta m}=
e^{-i\theta(n-m)}.
$$

<a id="sec-5-5"></a>
### 5.5 POVM 完备性的白化修正

由于数值求积与截断误差，简单相加通常只有

$$
S = \sum_k F_k^{(\theta)} \approx I,
$$

而不一定严格等于单位算符。脚本因此引入白化修正：

$$
\tilde F_k^{(\theta)}=
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

**通俗理解：** 白化修正有点像“数值上的最后校准”。前面因为近似积分和截断，所有 POVM 元加起来可能不是严格的单位算符；白化这一步就是把它们轻微修正一下，让整套测量重新严格满足概率完备性。

---

<a id="sec-6"></a>
## 6. 从单模 POVM 到双模 IQ POVM

<a id="sec-6-1"></a>
### 6.1 张量积输出效应

设 \(X\) 方向 coarse POVM 为 \(\{F_i^{(x)}\}_{i=1}^{N_x}\)，设 \(P\) 方向 coarse POVM 为 \(\{F_j^{(p)}\}_{j=1}^{N_p}\)，则分束器后的双模离散输出效应为

$$
E_{ij}=
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

**通俗理解：** 前面我们已经把 \(X\) 轴和 \(P\) 轴各自切格子了；这里就是把“X 落在哪格”和“P 落在哪格”合并起来，形成最终的二维输出标签。

<a id="sec-6-2"></a>
### 6.2 投影到联合支持空间

由于输入态实际只活在支持子空间 \(\mathcal H_{\mathrm{supp}}^{\otimes 2}\) 中，因此脚本进一步把 \(E_{ij}\) 投影成

$$
\tilde E_{ij}=
B_{\mathrm{joint}}^\dagger E_{ij} B_{\mathrm{joint}}.
$$

程序上对应：

- `project_povm_to_basis_list(povm, joint_basis)`

这样做之后，概率仍可精确写成 Born 形式：

$$
P(c \mid x,y)=
\mathrm{Tr}\left( \tilde E_c \tilde \rho_{xy} \right).
$$

**通俗理解：** 这一步的意思是：虽然我们前面做了支持压缩，但并没有改变真正的物理预测方式。概率仍然是“态乘测量再取迹”，只是把计算搬到了更小、更高效的子空间里。

<a id="sec-6-3"></a>
### 6.3 解析 Gaussian 矩形概率后端

在 Python 主线中，`route5` 还提供第二类概率后端 `analytic_gaussian_rectangles`。它不再先在截断 Fock 空间中构造 quadrature POVM，而是直接利用理想相干态经过平衡分束器后的解析高斯分布，计算每个 IQ 矩形 bin 的概率。

设联合输入仍为

$$
\rho_{xy}^{(\infty)}=
|\alpha_x\rangle\langle \alpha_x|
\otimes
|\alpha_y\rangle\langle \alpha_y|.
$$

由于平衡分束器对相干态有封闭作用，

$$
U_{\mathrm{BS}}
\left(
|\alpha_x\rangle \otimes |\alpha_y\rangle
\right)=
|\gamma_{xy}\rangle \otimes |\delta_{xy}\rangle,
$$

其中

$$
\gamma_{xy}=\frac{\alpha_x+\alpha_y}{\sqrt 2},
\qquad
\delta_{xy}=\frac{\alpha_x-\alpha_y}{\sqrt 2}.
$$

随后对第一路测 \(X\)，对第二路测 \(P\)。在本项目采用的约定

$$
X=\frac{a+a^\dagger}{\sqrt 2},
\qquad
P=\frac{a-a^\dagger}{i\sqrt 2}
$$

下，相干态 \(|\gamma\rangle\) 与 \(|\delta\rangle\) 的 quadrature 分布均为方差 \(1/2\) 的高斯分布，并满足

$$
\mu_x=
\mathbb E[X]=
\sqrt 2\,\mathrm{Re}(\gamma_{xy})=
\mathrm{Re}(\alpha_x+\alpha_y),
$$

$$
\mu_p=
\mathbb E[P]=
\sqrt 2\,\mathrm{Im}(\delta_{xy})=
\mathrm{Im}(\alpha_x-\alpha_y).
$$

若 \(X\) 方向第 \(i\) 个 bin 为

$$
I_i=[x_{i-1},x_i),
$$

而 \(P\) 方向第 \(j\) 个 bin 为

$$
J_j=[p_{j-1},p_j),
$$

则一维区间概率分别为

$$
P_X(i|x,y)=
\frac12
\left[
\operatorname{erf}(x_i-\mu_x)-
\operatorname{erf}(x_{i-1}-\mu_x)
\right],
$$

$$
P_P(j|x,y)=
\frac12
\left[
\operatorname{erf}(p_j-\mu_p)-
\operatorname{erf}(p_{j-1}-\mu_p)
\right].
$$

由于两路输出是相干态张量积，且分别在不同模式上测 \(X\) 与 \(P\)，因此二维矩形 bin 的概率直接分解为

$$
P_{\mathrm{analytic}}\bigl((i,j)\mid x,y\bigr)=
P_X(i|x,y)\,P_P(j|x,y).
$$

这就是 Python 中 `analytic_iq_probabilities(...)` 所实现的解析后端。它保留了：

1. 同一套本地 alphabet；
2. 同一套 `x_bounds` / `p_bounds`；
3. 同一套 top-\(K\) 候选筛选；
4. 同一套 single-device guessing SDP；

但它在“概率层”不再经过有限维 POVM 数值积分，而是直接采用理想无限维相干态的解析 Gaussian 矩形概率公式。

**通俗理解：** 这一版相当于跳过“先把连续测量离散化成 POVM 再取迹”的数值近似步骤，直接用理想双 Homodyne 理论告诉我们“这个 IQ 小矩形里本来应该落多少概率”。

---

<a id="sec-7"></a>
## 7. 概率表、Raw 熵与候选输入筛选

<a id="sec-7-1"></a>
### 7.1 Trace POVM 的 Born 概率

对 `trace_povm` 后端，完整概率表由截断支持空间上的 Born 迹公式给出：

$$
P_{\mathrm{trace}}(c \mid x,y)=
\mathrm{Tr}\left( \tilde E_c \tilde \rho_{xy} \right).
$$

这里：

- \(\tilde \rho_{xy}\) 是由 `cutoff` 截断、支持压缩后得到的联合 trusted state；
- \(\tilde E_c\) 是由 quadrature 数值积分、白化修正、分束器变换和联合支持投影得到的离散 IQ POVM 元。

程序上对应：

- `dual_homodyne_probabilities(...)`
- `measurement_probabilities_from_states(states, povm)`

因此，这个后端的概率不仅依赖输入 alphabet 和 IQ 边界，还依赖：

1. `cutoff`
2. `num_quadrature_nodes`
3. POVM 白化与支持投影

等数值建模步骤。

**通俗理解：** `trace_povm` 这一版是“先认真搭出一套有限维近似测量算符，再让态去撞这套算符并取迹”。

<a id="sec-7-2"></a>
### 7.2 解析 Gaussian 概率

对 `analytic_gaussian_rectangles` 后端，概率表不再通过 \(\mathrm{Tr}(\rho E)\) 数值生成，而是直接使用第 6.3 节的解析公式：

$$
P_{\mathrm{analytic}}\bigl((i,j)\mid x,y\bigr)=
P_X(i|x,y)\,P_P(j|x,y).
$$

这意味着：

1. 它与 `trace_povm` 共用同一套输入标签；
2. 共用同一套 `x_bounds`、`p_bounds` 和输出标签 \((i,j)\)；
3. 但不再显式依赖 `num_quadrature_nodes`，因为概率层已经不走高斯-厄米特求积。

在 Python 中，这一分支由

- `route5_iq_probabilities(..., probability_engine="analytic_gaussian_rectangles")`

统一调度。

需要特别提醒的是：虽然解析后端不再用有限维 POVM 生成概率，但当前 formal SDP 仍然沿用截断/投影后的 trusted-state 表示。因此，后面出现的 `formal infeasible`，并不意味着解析公式本身有错，而是意味着“解析概率层”和“当前有限维 trusted-state 认证层”之间可能不再完全自洽。

**通俗理解：** 两个后端的区别不在于输入或分箱变了，而在于“概率是先数值构造 POVM 再算出来”，还是“直接用理想 Gaussian 公式一口气算出来”。

<a id="sec-7-3"></a>
### 7.3 Raw 最小熵

无论使用哪一类概率后端，只要概率表 `probabilities` 已生成，就可以对每个输入 \(s=(x,y)\) 定义分布级别的 raw 猜测概率

$$
p_{\mathrm{guess}}^{\mathrm{raw}}(s)=
\max_c P(c \mid s),
$$

以及对应的 raw 最小熵

$$
H_{\min}^{\mathrm{raw}}(s)=
-\log_2 \max_c P(c \mid s).
$$

脚本中：

```matlab
raw_h = -log2(max(max(probabilities, [], 2), 1e-15));
```

然后在全部输入上取最优值

$$
H_{\min,\mathrm{best}}^{\mathrm{raw}}=
\max_s H_{\min}^{\mathrm{raw}}(s).
$$

它只是一个“哪几个输入表面上最平坦、最值得进一步 formal 认证”的初筛指标。

**通俗理解：** `raw_H_min` 更像是输出分布的第一眼印象分，不涉及更深的量子兼容性最坏情况。

<a id="sec-7-4"></a>
### 7.4 为什么要只认证前若干个目标输入

由于正式 SDP 代价高，`route5` 不会对全部 \(L^2\) 个联合输入都逐个做 full certification，而是先按 `raw_h` 从高到低排序，只保留前

$$
K = \texttt{max\_inputs\_to\_certify}
$$

个候选目标输入。

程序上对应：

- `sort_target_indices_desc(raw_h)`
- `candidate_order = candidate_order(1:min(...))`

同时，JSON 输出里还会保存 `raw_top_targets`，用来记录 raw 层面最强的一批候选输入，便于后续复核“为什么 formal 最强点会落在这里”。

这一设计对应 Python `route5` 的真实工作流：先用 raw 熵做便宜粗筛，再把最昂贵的 MOSEK / SDP 认证算力集中到最有希望的几个 target input 上。

**通俗理解：** 这和先打预赛再打决赛是一个思路。不是所有输入都不重要，而是先把最可能出高正式熵的几个点筛出来。

---

<a id="sec-8"></a>
## 8. Single-device Guessing SDP 的理论模型

<a id="sec-8-1"></a>
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

**通俗理解：** 这里的 \(M_{c,e}\) 可以把它读成：“真实结果是 \(c\)，而 Eve 猜的是 \(e\) 时，对应的那一块算符变量。” 所以 SDP 并不是只在优化一个数字，而是在同时优化整张“真实结果/猜测结果”的算符表。

<a id="sec-8-2"></a>
### 8.2 目标函数

若目标输入记为 \(\rho_{s^\star}\)，则最大猜测概率写成

$$
p_{\mathrm{guess}}(s^\star)=
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

**通俗理解：** 目标函数只把 \(M_{c,c}\) 加起来，是因为只有“真实结果和 Eve 猜的一样”才算猜中。这个目标就是在问：在所有满足实验统计的量子策略里，Eve 最多能猜对多少次？

<a id="sec-8-3"></a>
### 8.3 统计一致性约束

对每个输入态 \(\rho_s\) 和每个真实输出 \(c\)，必须满足

$$
\sum_{e=1}^{C}
\mathrm{Tr}(M_{c,e}\rho_s)=
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

**通俗理解：** 这条约束是在说：“不管 Eve 私下怎么分解和猜测，最终对外表现出来的真实统计，必须和我们观测到的概率表完全一致。” 它把 SDP 紧紧绑在实验/理论概率表上。

<a id="sec-8-4"></a>
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

**通俗理解：** 可以把每个 \(e\) 看成 Eve 预先选中的一种“猜测分支”。这些分支各自对应一套合法测量，而且所有分支的权重加起来必须是 1，才能代表一个完整的物理策略。

<a id="sec-8-5"></a>
### 8.5 正式最小熵

若 SDP 最优值为

$$
\hat p_{\mathrm{guess}},
$$

则正式认证的最小熵为

$$
\hat H_{\min}=
-\log_2 \hat p_{\mathrm{guess}}.
$$

脚本中：

```matlab
result_struct.p_guess = cvx_optval;
result_struct.H_min = -log2(cvx_optval);
```

这就是 `route5` 真正关心的最终量。

**通俗理解：** 一旦算出 \(p_{\mathrm{guess}}\)，最小熵只是再做一次 \(-\log_2\)。所以本质上，整个正式认证都在围绕“把 Eve 的最优猜中率上界卡到多低”这件事展开。

<a id="sec-8-6"></a>
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

**通俗理解：** 一个输入“看起来随机”，不等于它“真的已经被安全证明随机”。`raw` 更像肉眼观察分布图后的印象分，`formal` 才是把所有量子兼容性和最坏情况都算进去后的正式分数。

---

<a id="sec-9"></a>
## 9. 程序流程与理论对象的一一对应

<a id="sec-9-1"></a>
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
\Pi=
\left(
\mathcal A,\ d,\ N_x,\ N_p,\ R,\ \gamma,\ K
\right).
$$

<a id="sec-9-2"></a>
### 9.2 第二步：构造本地截断相干态

对每个 \(\alpha_\ell\) 计算：

$$
\rho_\ell = \lvert \alpha_\ell; d \rangle \langle \alpha_\ell; d \rvert.
$$

对应函数：

- `build_truncated_coherent_density`

<a id="sec-9-3"></a>
### 9.3 第三步：做支持压缩

从 \(\{\rho_\ell\}\) 对应的 ket 向量构造支持基 \(B_{\mathrm{loc}}\)，并投影得到 \(\tilde \rho_\ell\)。

对应函数：

- `support_basis_from_vectors`
- `project_density_to_basis`

<a id="sec-9-4"></a>
### 9.4 第四步：构造全部联合输入

对所有 \((x,y)\) 构造

$$
\tilde \rho_{xy} = \tilde \rho_x \otimes \tilde \rho_y.
$$

对应脚本的双重循环与 `kron(...)`。

<a id="sec-9-5"></a>
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

**通俗理解：** 这一步是整条路线里最“物理前端”的部分。它把“怎么切 IQ 平面”真正翻译成一组可以和量子态做 Born 迹运算的 POVM 元。

<a id="sec-9-6"></a>
### 9.6 第六步：生成概率表

在 Python 主线里，这一步由 `route5_iq_probabilities(...)` 统一调度，并显式分成两条分支。

第一条是 `trace_povm` 分支。此时对每个输入态、每个输出效应计算

$$
P_{\mathrm{trace}}(c \mid x,y) = \mathrm{Tr}(\tilde E_c \tilde \rho_{xy}).
$$

对应函数：

- `dual_homodyne_probabilities`
- `measurement_probabilities_from_states`

第二条是 `analytic_gaussian_rectangles` 分支。此时不再显式构造有限维 POVM，而是直接调用

- `analytic_iq_probabilities`

按解析 Gaussian 矩形积分得到

$$
P_{\mathrm{analytic}}(c \mid x,y).
$$

因此，本步更准确地说不是“固定一种概率生成公式”，而是“在同一套输入 alphabet 和 IQ 分箱之上，选择一个概率后端来产出同形状的概率表”。

需要注意：

1. `num_quadrature_nodes` 只对 `trace_povm` 分支有意义；
2. 当前 Matlab 参考脚本本质上只镜像了 `trace_povm` 的单点流程；
3. 解析后端目前主要用于 Python 中的一致性诊断与理论对照。

<a id="sec-9-7"></a>
### 9.7 第七步：raw 粗筛

计算每个联合输入对应的 raw 熵

$$
H_{\min}^{\mathrm{raw}}(x,y)=
-\log_2 \max_c P(c \mid x,y),
$$

然后保留前 `max_inputs_to_certify` 个候选。

<a id="sec-9-8"></a>
### 9.8 第八步：正式 SDP 认证

对每个候选目标输入，调用

- `solve_single_device_guessing_route5`

求出正式 \(p_{\mathrm{guess}}\) 与 \(H_{\min}\)，再在所有候选中选出 formal 最好的那个目标输入。

---

<a id="sec-10"></a>
## 10. 默认参数的物理意义与当前结果

<a id="sec-10-1"></a>
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

<a id="sec-10-2"></a>
### 10.2 Trace 主线与固定光强主线

如果先只看历史 `trace_povm` 主线，那么 `route5` 目前最重要的两类结果如下。

第一类是自由半径主线。对应参数为

$$
\texttt{radii}=[0.0,\ 0.85,\ 1.25],
\qquad
|\Phi|=8,
\qquad
\texttt{cutoff}=4,
\qquad
N_x=6,\ N_p=2,
$$

并在当前强点记录中取

$$
\texttt{num\_quadrature\_nodes}=12.
$$

在这组参数下，Python 结果文件
[`route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json`](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)
给出

$$
H_{\min}^{\mathrm{formal}} \approx 2.11639,
\qquad
p_{\mathrm{guess}} \approx 0.230623.
$$

对应的 raw 最强输入分数约为

$$
H_{\min}^{\mathrm{raw,best}} \approx 2.99218.
$$

从扫描记录看，formal 最强点与其对称点分别落在目标输入 `[15,7]` 与 `[11,3]` 一带，它们对应的是两层非零半径上的固定相位相干态组合。

第二类是固定光强主线。对光强菜单 `[0,80,160]`，当前 Python 脚本采用

$$
r(\mu)=r_{\max}\sqrt{\frac{\mu}{\mu_{\max}}},
\qquad
r_{\max}=1.2,
\qquad
\mu_{\max}=160,
$$

从而把固定光强映射成内部半径

$$
[0.0,\ 0.8485,\ 1.2].
$$

对应结果文件
[`route5_fixed_intensity_080160_scale120.json`](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)
给出

$$
H_{\min}^{\mathrm{formal}} \approx 2.10102.
$$

这说明：在历史 `trace_povm` 口径下，`route5` 的高熵并不完全依赖于自由搜索出来的任意 alphabet；即便把输入收回到固定光强菜单约束下，formal 熵仍曾保持在 `2 bit` 以上。

但这部分结果必须连同一个关键数值事实一起汇报：同样的自由主线几何参数，如果把

$$
\texttt{num\_quadrature\_nodes}
$$

从 `12` 提高到 `20`，则对应文件
[`route5_node_convergence_scan_probe20.json`](../output/qrng_routes/route5_node_convergence_scan_probe20.json)
给出的 formal 结果会下降到

$$
H_{\min}^{\mathrm{formal}} \approx 1.77989,
$$

尽管其 raw 最强值反而上升到

$$
H_{\min}^{\mathrm{raw,best}} \approx 3.34343.
$$

因此，当前应更谨慎地把这部分结论表述为：

1. `trace_povm` 历史主线已经给出过 `H_{\min} > 2` 的强候选窗口；
2. 固定光强约束下也出现过 `H_{\min} > 2` 的候选窗口；
3. 但这些结果对 quadrature 数值离散参数仍然敏感，还不能简单视为“完全稳定、与后端选择无关的最终结论”。

**通俗理解：** 历史 trace 主线确实跑出过很亮眼的 `2 bit` 以上结果，但这更像“已经看到了很强信号”，还不是“所有数值口径都已经钉死”的最终定稿版。

<a id="sec-10-3"></a>
### 10.3 解析概率版本的结果与诊断

当保持同一套几何参数

$$
\texttt{radii}=[0,0.85,1.25],\quad
|\Phi|=8,\quad
\texttt{cutoff}=4,\quad
N_x=6,\ N_p=2
$$

不变，仅把概率后端切换为 `analytic_gaussian_rectangles` 后，结果会发生明显变化。

对应文件
[`route5_node_convergence_scan_analytic_r085125.json`](../output/qrng_routes/route5_node_convergence_scan_analytic_r085125.json)
显示：

$$
H_{\min}^{\mathrm{raw,best}} \approx 2.72772,
$$

raw 最强输入落在 `[0,0]`、`[12,10]`、`[11,11]` 等一批对称点上；但对 top-3 raw 候选做 formal 认证时，三者全部返回

$$
\texttt{status}=\texttt{infeasible}.
$$

这说明问题不是“formal 比 raw 保守一点”，而是：

$$
\text{解析概率表}
\;+\;
\text{当前截断 trusted-state formal 模型}
$$

在现阶段并不完全自洽。

更细的诊断来自文件
[`route5_analytic_backend_diagnostics_r085125.json`](../output/qrng_routes/route5_analytic_backend_diagnostics_r085125.json)。
其中最重要的一组摘要如下：

| cutoff | local rank | mean TV(trace, analytic) | 最大线性拟合残差 | plain POVM feasibility |
| --- | ---: | ---: | ---: | --- |
| 4 | 4 | 0.15417 | 0.14127 | infeasible |
| 5 | 5 | 0.08799 | 0.02093 | infeasible |
| 6 | 6 | 0.05414 | \(1.74\times 10^{-14}\) | infeasible |
| 8 | 8 | 0.03539 | \(2.13\times 10^{-14}\) | 未继续单独复核 |
| 12 | 12 | 0.03382 | \(4.30\times 10^{-14}\) | 未继续单独复核 |
| 16 | 16 | 0.03382 | \(9.92\times 10^{-14}\) | 未继续单独复核 |

这张表反映出三层信息。

第一，`cutoff=4/5` 时，解析概率列向量甚至不完全落在当前 trusted states 诱导的线性像空间里，因此 formal infeasible 并不奇怪。

第二，到 `cutoff=6` 时，线性拟合残差已经降到机器精度量级，说明“单列概率向量能由某个线性算符拟合”这一步基本没问题；但 plain POVM feasibility 仍然是 `infeasible`，这表明问题已经不只是线性代数层面的拟合，而是更深的正定性/完备性兼容失败。

第三，即便 `cutoff` 继续增大，`trace_povm` 与解析后端之间的平均 TV 距离也没有收敛到零，而是在约

$$
3.38\times 10^{-2}
$$

附近进入平台区。这说明两条概率后端并不只是“低 cutoff 下稍微不同”，而是对应了两种真正不同的数值/建模口径。

因此，当前更合适的口径是：

1. `analytic_gaussian_rectangles` 是一个非常重要的一致性诊断工具；
2. 它揭示出当前 `trace_povm` 主线与“理想无限维 coherent 解析概率”之间仍有模型失配；
3. 在失配没有被解决前，解析后端不应被包装成已经正式超过 `2 bit` 的主结果线。

**通俗理解：** 解析后端像是一把更锋利的尺子。它不是把原结果轻微修正一下，而是直接告诉我们：当前这套有限维 trusted-state 认证框架，和理想 Gaussian 概率之间还没有完全扣上。

<a id="sec-10-4"></a>
### 10.4 为什么 Route5 在旧 Trace 主线下能超过 2 bit

下面这个解释针对的是“为什么历史 `trace_povm` 主线曾经跑出 `H_{\min} > 2`”，而不意味着解析后端下也必然保持同样结论。

从模型结构上看，`route5` 的优势主要来自三点：

1. 输入字母表比传统单半径相位编码更丰富；
2. 中央测量不是低速单光子点击，而是高速 CV 前端；
3. 最终离散输出由二维 IQ 分区提供，因此在相同输出数下，几何结构比一维粗粒化更灵活。

真正起作用的是“输入几何 + IQ 分区 + SDP 约束”三者的匹配，而不是某一个参数单独神奇地拉高结果。

**通俗理解：** `route5` 能在旧 trace 主线下拉高熵，核心靠的是一整套几何结构协同，而不是因为某个参数被调到了特殊数值。

---

<a id="sec-11"></a>
## 11. 主要变量的数学含义

下面把脚本中最关键的变量与数学对象做一次集中对应。

<a id="sec-11-1"></a>
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

<a id="sec-11-2"></a>
### 11.2 联合输入相关

- `joint_states`
  对应全部联合输入 \(\tilde \rho_{xy}\)。

- `labels`
  对应每个输入的 `(x,y)` 标签。

- `joint_basis`
  对应 \(B_{\mathrm{joint}} = B_{\mathrm{loc}} \otimes B_{\mathrm{loc}}\)。

<a id="sec-11-3"></a>
### 11.3 IQ 分区相关

- `num_x_bins = N_x`
- `num_p_bins = N_p`
- `x_bounds`
  对应 \(X\) 方向边界；
- `p_bounds`
  对应 \(P\) 方向边界；
- `quadrature_range = R`
- `boundary_gamma = \gamma`

<a id="sec-11-4"></a>
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

<a id="sec-11-5"></a>
### 11.5 认证候选相关

- `max_inputs_to_certify`
  对应正式进入 SDP 的候选输入数 \(K\)。

- `raw_best_target`
  raw 熵最高的联合输入标签。

- `certified_best_target`
  formal 认证后真正最优的联合输入标签。

---

<a id="sec-12"></a>
## 12. Python 主线与 Matlab 参考脚本的关系

现在需要把这层关系讲得更清楚一些。

第一，当前 `route5` 的主实现权重已经转到 Python。真正定义协议细节、概率后端分支、搜索流程和诊断流程的，是：

- [`hybrid_iq.py`](../src/python/qrng_routes/route5/hybrid_iq.py)
- [`main.py`](../src/python/qrng_routes/route5/main.py)
- [`refine_queue.py`](../src/python/qrng_routes/route5/refine_queue.py)
- [`intensity_menu_search.py`](../src/python/qrng_routes/route5/intensity_menu_search.py)
- [`node_convergence_scan.py`](../src/python/qrng_routes/route5/node_convergence_scan.py)
- [`analytic_backend_diagnostics.py`](../src/python/qrng_routes/route5/analytic_backend_diagnostics.py)

这些文件共同构成了：

1. alphabet 定义；
2. `trace_povm` / `analytic_gaussian_rectangles` 双后端；
3. raw 粗筛；
4. formal SDP 认证；
5. 参数搜索与一致性诊断；
6. 结果 JSON 落盘。

第二，Matlab 文件
[`guessprobprimal_route5_hybrid_iq.m`](../src/matlab/guessprobprimal_route5_hybrid_iq.m)
的定位更像是一份“单点参考镜像”。它的优点是：

1. 单文件、易阅读；
2. 便于导师直接在 Matlab 环境里跑一个固定参数实例；
3. 适合对照理论主流程理解 route5。

但它并不覆盖：

1. Python 中的大规模 `alphabet-search`；
2. `num_quadrature_nodes` 收敛扫描；
3. 解析概率后端诊断；
4. 多个结果 JSON 的统一汇总。

因此，更准确的表述应是：

1. Python 主线是当前项目中 `route5` 的权威实现；
2. Matlab 脚本是一个便于阅读和单点复核的参考实现；
3. 若两边出现口径差异，应先检查参数是否完全一致，再以 Python 主线和对应 JSON 结果作为当前阶段的主要工作依据。

这并不是说 Matlab 不重要，而是因为：

- 最新的概率后端分叉；
- 收敛性复核；
- 解析概率一致性诊断；

都首先在 Python 中发展出来了。

**通俗理解：** 现在的关系不是“Matlab 才是本体，Python 只是翻译”，而是“Python 已经是主线工作台，Matlab 更像一份更适合老师直接打开看的单文件说明版”。

---

<a id="sec-13"></a>
## 13. 理论边界、实验可行性与使用注意

<a id="sec-13-1"></a>
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

这里还需要说得更具体一些。`route5` 需要的不是当前 `route4` 那种一维 `Probability.mat` 点击概率表，而是与 `IQ / dual-homodyne` 结构相匹配的数据，例如：

1. 原始 IQ 样本；
2. 二维直方图；
3. 或者至少是已经按固定分箱整理好的
   $$
   P_{\mathrm{exp}}(c \mid x,y).
   $$

因此，`route5` 在实验上并不是“直接复用现有 APD 数据就能完成”的小修版，而是一条需要自己数据接口的独立实验路线。现有代码在认证层结构上已经可以复用，但还没有整理成一个专门面向实验文件的整洁接入入口。

**通俗理解：** 如果把 route4 的 `Probability.mat` 直接塞给 route5，那就像把“点击计数表”硬塞给一个本来想看“二维 IQ 散点图”的流程，数据格式和物理对象都对不上。

<a id="sec-13-2"></a>
### 13.2 实验室需要完成的工作包

如果实验室后续要把 `route5` 做成真正可接入 SDP 的实验流程，至少需要把工作拆成下面几个明确的工作包。

1. 先冻结一版协议参数
   需要先固定：
   - 本地 trusted alphabet 的定义；
   - 选哪些半径 / 光强；
   - 每个非零半径配几个相位；
   - IQ 分箱的 `x_bounds` 与 `p_bounds`；
   - 生成轮使用哪些目标输入，测试轮覆盖哪些输入。

2. 做输入态标定
   对每个本地输入态，都需要给出它在实验中的实际标签与物理参数，例如：
   - 输入编号 `x` 或 `y`；
   - 对应的强度档位或平均光子数 \(\mu\)；
   - 对应的目标相位 \(\phi\)；
   - 若已做完独立标定，还应给出复振幅 \(\alpha\) 或等效 `I/Q` 位移坐标；
   - 对应的不确定度或漂移范围。
   若当前协议采用固定相位 coherent alphabet，则实验端不应再额外做未建模的相位随机化或相位平均，否则 trusted input 的物理含义会被改掉。

3. 做中央接收机标定
   需要明确：
   - 50:50 分束器是否足够接近平衡；
   - 两路正交测量是否真正对应 \(X\) 与 \(P\)；
   - ADC 输出如何归一化成脚本使用的无量纲 quadrature 坐标；
   - 真空参考或 shot-noise 单位是如何定义的；
   - 是否做去直流、滤波、重采样、积分窗口截取。

4. 采集按输入标签分组的实验数据
   对每个输入对 \((x,y)\)，都应能回溯到：
   - 该输入对采了多少轮；
   - 每轮对应的 IQ 输出是什么；
   - 每轮样本是否有效；
   - 数据来自哪个采集批次、哪个时间段。

5. 输出可认证的数据文件
   至少要输出：
   - 每个输入对、每个离散输出的计数；
   - 总轮数；
   - 所用分箱边界；
   - 输入态定义与映射规则。

6. 做独立的 formal 认证
   最后才是把实验版
   $$
   P_{\mathrm{exp}}(c \mid x,y)
   $$
   代入同一套 SDP，得到实验版 `p_guess` 与 `H_min`。

**通俗理解：** 对实验室来说，真正要做的并不是“把一堆 IQ 数据测出来”这么简单，而是要把“输入是什么、测量怎么做、数据怎么归一化、分箱怎么定、每个概率从哪来”这整条链条一起固定住，否则后面的 SDP 没法有清晰的物理含义。

<a id="sec-13-3"></a>
### 13.3 实验室应如何给出数据

从可复核性和后续认证的角度，实验数据最好按三个层级准备。

第一层是最理想的数据：逐轮原始 IQ 数据。  
对每个输入对 \(s=(x,y)\)，记录每一轮的原始或归一化后样本

$$
(I_t, Q_t), \qquad t=1,\dots,N_s.
$$

这时固定分箱 \(R_c\) 之后，实验概率可以直接写成

$$
P_{\mathrm{exp}}(c \mid s)=
\frac{1}{N_s}
\sum_{t=1}^{N_s}
\mathbf 1\!\left[(I_t,Q_t)\in R_c\right].
$$

第二层是次优但仍然很好用的数据：二维直方图计数。  
若实验室不方便保存所有逐轮样本，则至少对每个输入对给出二维 histogram

$$
H_s(i,j),
$$

其中 \((i,j)\) 对应 IQ 平面上的一个矩形 bin。此时

$$
N_{s,c} = H_s(i,j),
\qquad
P_{\mathrm{exp}}(c \mid s)=\frac{N_{s,c}}{\sum_{c'} N_{s,c'}}.
$$

第三层是最小可用数据：离散概率表加计数表。  
若连二维 histogram 都不能保留，则至少应给出：

1. `counts[s,c] = N_{s,c}`
2. `totals[s] = N_s`
3. `probabilities[s,c] = N_{s,c}/N_s`
4. `x_bounds`
5. `p_bounds`
6. 输入标签 `labels[s] = (x,y)`
7. 每个本地输入对应的物理参数说明。

需要特别强调：只给一张已经归一化的概率表而不给计数，通常是不够的，因为后续若要做有限尺寸分析，就必须知道真实样本数。

<a id="sec-13-4"></a>
### 13.4 推荐的数据文件结构与必须附带的元数据

为了让 route5 后续可以真正自动接入，实验室最好按“主数据 + 元数据”一起打包。
从工程实现上看，`HDF5`、`MAT`、`NPZ` 这类既能存多维数组又能存元数据字段的格式会比较合适。

推荐的主数据内容如下。

1. 若保存逐轮数据，则主文件至少应包含字段：
   - `shot_id`
   - `batch_id`
   - `input_x`
   - `input_y`
   - `I_raw`
   - `Q_raw`
   - `I_calibrated`
   - `Q_calibrated`
   - `valid_flag`
   - `timestamp`

2. 若保存 histogram，则主文件至少应包含：
   - `counts[x,y,i,j]`
   - `x_bounds`
   - `p_bounds`
   - `total_shots[x,y]`

3. 若只保存 coarse-grained 概率，则至少应包含：
   - `counts[x,y,c]`
   - `probabilities[x,y,c]`
   - `total_shots[x,y]`
   - 输出标签 `c <-> (i,j)` 的映射。

元数据文件则建议至少包含：

1. 输入态定义
   - 每个输入编号对应的 \(\mu\)、相位、目标 \(\alpha\)；
   - 这些值是“理论设定值”还是“实验标定值”；
   - 若有漂移补偿，补偿方式是什么。

2. 接收机标定
   - LO 功率或等效参考条件；
   - \(I/Q\) 通道增益；
   - 直流偏置扣除方式；
   - ADC 量程；
   - 饱和样本是否丢弃。

3. 轮次定义
   - 一轮样本是如何从连续波形中截取的；
   - 每轮是否做积分窗口；
   - 不同轮之间是否有保护间隔；
   - 是否存在 post-selection。

4. 数据处理口径
   - 是否做低通或带通滤波；
   - 是否做重采样；
   - `x_bounds` 和 `p_bounds` 是事先固定还是事后从数据决定；
   - 若有真空校准，归一化规则是什么。

这里最重要的一条原则是：只要某一步处理会改变 IQ 点云的位置或形状，就应当在元数据里明确写出来。

**通俗理解：** 对后续程序来说，最怕的不是数据少，而是“看起来有数据，但不知道这些数是怎么处理出来的”。只要不知道某一步滤波、归一化或边界设置，后面的 formal 认证就很难说清楚。

<a id="sec-13-5"></a>
### 13.5 所需物理设备与典型实验方法

从物理实现上看，`route5` 至少需要以下几类核心设备。

1. 稳定的相干光源
   - 用来制备 trusted coherent states；
   - 若协议包含多个相位点，则要求相位参考稳定；
   - 若协议包含多个强度点，则要求强度菜单可重复调用。

2. 输入态调制模块
   - 用于控制振幅 / 强度；
   - 用于控制相位；
   - 在实验上可以是独立调制器，也可以是等效的复振幅调制链路。

3. 双路输入结构
   - 因为 `route5` 的联合输入是
     $$
     |\alpha_x\rangle \otimes |\alpha_y\rangle,
     $$
     因此实验上要么有双路独立输入支路，要么有能等效产生双路输入的时分 / 空分结构。

4. 中央 50:50 分束器
   - 用于实现理论模型中的 `balanced beamsplitter`。

5. Dual-homodyne / IQ 接收机
   - 可以理解为：分束器后分别测量两个正交 quadrature；
   - 实验实现上，可以是两路平衡 homodyne，且参考相位差约为 \(\pi/2\)；
   - 也可以是等效的 IQ coherent receiver / optical hybrid 方案。
   - 无论采用哪种实现，都需要给出本振参考和正交相位关系是如何建立与维持的。

6. 低噪声探测与电子学链路
   - 平衡探测器；
   - 放大器；
   - ADC / 示波器；
   - 时钟同步与触发模块。

7. 数据采集与控制软件
   - 需要能把“当前输入标签 \((x,y)\)”和“当前测得的 IQ 样本”正确关联起来；
   - 还要能导出适合后续认证的数据文件。

典型实验方法可以按下列步骤理解。

1. 预先固定一版 trusted alphabet
   例如先固定：
   - 真空点；
   - 两个非零半径层；
   - 每层 8 个相位。

2. 按输入标签生成双路相干输入
   对每一轮，实验控制系统决定当前使用的 \((x,y)\)，并在两路输入端生成相应 coherent states。

3. 在中央做分束干涉与 IQ 读出
   两路输入在 50:50 分束器上干涉，随后对两个正交 quadrature 做同步测量，得到连续 \((I,Q)\) 或 \((X,P)\) 样本。

4. 对连续样本做统一的数据处理
   包括：
   - 去直流；
   - 增益归一化；
   - 真空参考标定；
   - 轮次切分。

5. 用固定分箱做 coarse-graining
   按预先锁定的 `x_bounds`、`p_bounds` 把每轮样本映射成离散输出 `c`。

6. 汇总为实验概率表
   计算每个输入对 \((x,y)\) 下各个输出 `c` 的频率，得到实验版
   $$
   P_{\mathrm{exp}}(c \mid x,y).
   $$

7. 送入同一套 SDP 做正式认证
   这一步原则上不改模型主线，只把“理论生成概率”替换成“实验测得概率”。

**通俗理解：** 物理上可以把 route5 看成“先把两束调好振幅和相位的光拿去干涉，再用 IQ 接收机把结果读成二维点云，最后把点云切格子并统计每个格子的概率”。

<a id="sec-13-6"></a>
### 13.6 Route5 在实验上是可行的，但应视为一条新的 IQ 路线

从物理结构上看，`route5` 并不是黑箱式的数值构造，而是由

$$
\text{相干态字母表}
\;+\;
\text{分束器}
\;+\;
\text{dual-homodyne / IQ}
\;+\;
\text{数字 coarse-graining}
\;+\;
\text{单设备 SDP}
$$

这几步串起来的。

因此，它在实验上并非“不现实”，相反，作为一条新的连续变量前端路线，它的实验含义是清楚的：

1. 中央测量对象明确，就是 `IQ / dual-homodyne`；
2. 安全证明对象明确，就是 coarse-grained 后的离散输出概率；
3. 高速前端与 formal 认证可以共存，不必退回到低速 APD 点击模型。

但与此同时，也不能把它表述成“当前 route4 实验的小改版”。更准确的说法应是：

- `route4` 适合当前 APD / `Probability.mat` 主线；
- `route5` 更适合下一阶段专门设计的 IQ 实验主线。

<a id="sec-13-7"></a>
### 13.7 固定光强结果的真实含义与风险边界

当前最贴实验限制的一条结果来自固定光强菜单

$$
\{0, 80, 160\},
$$

其 formal 结果约为

$$
H_{\min} \approx 2.10102.
$$

对应的猜测概率为

$$
p_{\mathrm{guess}}=
2^{-2.101017214340893}
\approx
0.23309.
$$

若以 `2 bit` 门槛

$$
H_{\min} \ge 2
\quad\Longleftrightarrow\quad
p_{\mathrm{guess}} \le \frac{1}{4}
$$

为参照，则当前余量约为

$$
0.25 - 0.23309 \approx 0.01691.
$$

这说明两件事：

1. 这条固定光强主线确实已经给出正余量，不是卡在 `2 bit` 门槛上；
2. 但这条余量也不算特别宽，因此若后续实验标定、真实噪声、有限尺寸统计或模型失配带来额外偏差，formal 值是有可能被压回 `2 bit` 以下的。

更关键的是，当前固定光强版仍采用

$$
r(\mu)=r_{\max}\sqrt{\mu/\mu_{\max}}
$$

这一搜索型映射。因此它的物理解释应当是“受固定光强菜单约束的理论高分窗口”，而不是“实验已经独立标定完毕的正式输入态设置”。后续若要做正式实验闭环，需要把 \(\mu \mapsto \alpha\) 的换算独立固定下来，避免把这一步继续留在事后搜索里。

**通俗理解：** 现在这条 `2.101 bit` 结果已经很有价值，因为它说明“贴着实验限制时，route5 也不是一下子就掉到 2 以下”。但它离最终实验报告还差半步，那半步就是把强度到振幅的映射单独标定清楚。

<a id="sec-13-8"></a>
### 13.8 字母表规模、统计负担与认证覆盖范围

对默认强点参数

$$
\mathcal R = \{0,\ 0.85,\ 1.25\},
\qquad
|\Phi| = 8,
$$

由于真空态只保留一次，因此本地字母表大小为

$$
L = 1 + 2\times 8 = 17,
$$

联合输入总数为

$$
|\mathcal S| = L^2 = 289.
$$

这说明 `route5` 的优势不是“实验工作量小”，而是“用更大的输入字母表与更丰富的 IQ 几何结构，换取 formal `H_min > 2` 的可能性”。与之对应，实验上至少有三类压力：

1. 输入态标定压力更大；
2. 测试轮与统计采样量不会轻；
3. 有限尺寸分析会比当前理论概率版更复杂。

还必须强调一个容易被忽略的边界：当前代码并不是对全部 `289` 个联合输入都做 formal 认证，而是先按 raw 指标排序，再只对前

$$
K = \texttt{max\_inputs\_to\_certify} = 3
$$

个候选输入做正式 SDP。

因此，当前结果的准确表述应当是：

- 在当前 alphabet 与 IQ 分区下，已经找到了 formal 认证超过 `2 bit` 的优质目标输入；
- 但这并不等价于“所有输入对都同样提供了超过 `2 bit` 的正式认证随机性”。

若后续要给实验室一份真正可执行的协议说明，还应把“哪些输入用于测试、哪些输入用于正式随机数生成”预先写清楚。

**通俗理解：** `17` 个本地态、`289` 个联合输入，意味着 route5 换来高熵的代价之一，就是实验准备和数据采集会更重。它不是最省事的路线，而是“更复杂，但更有机会把正式熵做高”的路线。

<a id="sec-13-9"></a>
### 13.9 主要实验风险与建议检查项

从实验闭环角度看，`route5` 的主要风险至少有以下几类。

1. 输入态失配风险
   理论上假定的是一组已知 coherent alphabet，但实验中真正生成出来的态可能在振幅或相位上有偏差。若这一偏差没有独立标定并写进元数据，就会破坏 trusted input 的前提。

2. 未建模相位处理风险
   若理论上使用的是固定相位 alphabet，但实验端实际上做了相位漂移平均、随机相位抖动或额外相位随机化，那么进入 SDP 的 trusted states 就已经不是原来那组固定相位相干态了。

3. IQ 归一化不一致风险
   若不同批次数据使用了不同增益、不同去直流口径或不同真空参考，则同样的 `x_bounds` / `p_bounds` 可能不再代表同一个物理区域。

4. LO 相位与正交性漂移风险
   dual-homodyne / IQ 接收要求两个读出通道足够接近正交 quadrature。若相位差漂移较大，则实验读出的 `I/Q` 与理论的 `X/P` 对应关系会变差。

5. 探测器非理想风险
   例如：
   - 分束器不平衡；
   - 平衡探测器失衡；
   - 电子噪声过大；
   - ADC 饱和或裁剪；
   - 带宽不足导致波形失真。

6. 时间漂移与样本相关性风险
   若长时间采集时系统状态缓慢漂移，或者相邻轮次之间存在显著相关性，则简单频率估计未必能直接代表理想的 i.i.d. 概率模型。

7. 边界后选风险
   若看到数据后再频繁调整 `x_bounds`、`p_bounds`，则容易把“设计阶段搜索”与“正式认证阶段”混在一起。严格做法应当是：
   - 先用探索数据决定 protocol；
   - 再用独立数据集做正式报告。

8. 每输入对样本数不足的风险
   因为联合输入很多，若总采样预算固定，则平均到每个输入对的轮数会下降，导致统计误差增大。

建议实验室至少做以下检查。

1. 真空输入重复测量，用来检查 IQ 零点与噪声底。
2. 固定某个输入对重复测量多个批次，用来检查漂移。
3. 给出同一批数据在不同归一化口径下的对比，确认 `P_{\mathrm{exp}}(c|x,y)` 稳定。
4. 对容易饱和的高功率点单独做线性度检查。
5. 对最终报告使用的数据，保存未分箱的原始 IQ 样本或至少保存二维 histogram。

**通俗理解：** `route5` 的风险并不主要在“SDP 会不会跑”，而在“实验端给 SDP 的那张概率表，到底是不是和理论里说的是同一个物理对象”。只要这条数据链没钉牢，后面结果再高也会被质疑。

<a id="sec-13-10"></a>
### 13.10 截断、求积与后端误差

目前 `route5` 至少存在三类必须单独汇报的数值/建模误差来源。

第一类是 Fock 截断误差。对相干态振幅 \(|\alpha|\) 而言，截断到 `cutoff=d` 后丢失的单模 Poisson 尾概率为

$$
\epsilon_{\mathrm{tail}}(|\alpha|,d)=
1-
e^{-|\alpha|^2}
\sum_{n=0}^{d-1}
\frac{|\alpha|^{2n}}{n!}.
$$

对当前主线中的最大半径 \(|\alpha|=1.25\)，诊断结果给出：

$$
\epsilon_{\mathrm{tail}}(1.25,4)\approx 0.07373,
$$

$$
\epsilon_{\mathrm{tail}}(1.25,5)\approx 0.02167,
\qquad
\epsilon_{\mathrm{tail}}(1.25,6)\approx 0.00540.
$$

这说明 `cutoff=4` 并不是一个“尾部已经完全可忽略”的极深截断。

第二类是 quadrature 数值积分误差。历史 `trace_povm` 主线对 `num_quadrature_nodes` 明显敏感：

1. `nodes=12` 时，强点 formal 约为
   $$
   H_{\min}\approx 2.11639;
   $$
2. `nodes=20` 时，在相同几何参数下，formal 下降为
   $$
   H_{\min}\approx 1.77989.
   $$

这说明 `route5` 当前的 `trace_povm` 高熵结果，不能脱离 `num_quadrature_nodes` 的具体取值来单独引用。

第三类是概率后端失配误差。即使把 `cutoff` 提高到 `8,10,12,16`，`trace_povm` 与解析后端之间的平均 TV 距离仍稳定在约

$$
3.38\times 10^{-2}
$$

附近，而不会继续收敛到零。这意味着：

$$
\texttt{trace\_povm}
\neq
\texttt{analytic\_gaussian\_rectangles}
$$

并不只是“低精度数值误差”，而是两种不同的概率建模口径。

还要补一句：POVM 白化修正确实能把

$$
\sum_c E_c \approx I
$$

拉回严格完备，但它只能修正 POVM 完备性，不会自动消除“截断 trusted states”和“理想解析概率”之间的模型失配。

因此，当前任何 `route5` 结果都至少应连同以下参数一起汇报：

1. `cutoff`
2. `num_quadrature_nodes`
3. `probability_engine`
4. `x_bounds` / `p_bounds`
5. 是否为自由半径主线还是固定光强主线

**通俗理解：** 这里的误差不只是“求解器最后差了几个小数点”，而是“你到底在用哪一套有限维近似和哪一套概率后端讲这个故事”。

<a id="sec-13-11"></a>
### 13.11 解析概率版本的理论意义与当前限制

尽管解析后端目前把 formal 推成了 `infeasible`，但它的理论意义仍然很强，原因有三点。

第一，它更接近理想 dual-homodyne / IQ 理论公式。  
在这条后端里，概率直接由理想无限维相干态的 Gaussian 矩形积分给出，不再掺入 quadrature 节点数值积分误差。因此它是一个很自然的“上层物理模型一致性检查器”。

第二，它帮助我们区分“数值离散误差”和“模型本身不自洽”。  
如果解析后端只是把 `H_{\min}` 从 `2.11` 轻微改成 `2.05`，那我们会更倾向于把问题理解成普通数值修正；但现在解析后端直接给出 formal `infeasible`，说明更核心的问题在于：

$$
\text{理想无限维 coherent 概率}
\quad\text{vs.}\quad
\text{当前截断/投影 trusted-state formal 模型}
$$

并不是同一个闭合模型。

第三，它为下一步真正的“模型闭环”指出了两个可能方向。

方向 A：保持有限维 trusted-state / SDP 框架，但把概率层也改成与同一截断模型严格一致的有限维版本。  
这条路更偏向“让概率层向当前 formal 模型靠拢”，逻辑上更自洽，但未必还能保住 `H_{\min} > 2`。

方向 B：保留理想解析概率层，同时把 trusted-state 表示升级成更接近精确 coherent-state 几何的模型，例如基于相干态 Gram 结构的表示。  
这条路更偏向“让 formal 模型向理想物理层靠拢”，理论上更漂亮，但实现与求解会更难。

因此，在当前阶段，`analytic_gaussian_rectangles` 更适合作为：

1. 一致性诊断工具；
2. 理想 Gaussian 理论参照；
3. 判断历史 `trace_povm` 结果稳不稳的一把尺子；

而不适合直接被表述成“已经形成正式主结果的认证后端”。

**通俗理解：** 解析后端现在最重要的作用，不是替我们直接拿到更高分，而是提醒我们“这条 `>2 bit` 的故事，在哪一层模型上还没完全闭合”。

<a id="sec-13-12"></a>
### 13.12 当前环境中的验证边界

截至当前这份报告，已经能够在本仓库中直接核对和复查的是：

1. Python 主线代码结构；
2. `trace_povm` 与 `analytic_gaussian_rectangles` 的实现逻辑；
3. 相关 JSON 结果文件；
4. `num_quadrature_nodes` 收敛扫描与解析后端诊断结果；
5. 这份理论报告与 Python 主线之间的对应关系。

而当前终端环境中并没有 Matlab 运行器，因此 Matlab 参考脚本在本环境下仍主要承担：

1. 可读性说明；
2. 参数/流程镜像；
3. 交给导师本地 Matlab 复核的参考入口；

这意味着：

- 本文对 Matlab 脚本的理论解释与流程说明可以是完整的；
- 但若要做最终数值定稿，仍应以实验室或导师本机 Matlab 运行结果，以及 Python 主线对应的 JSON 复核一起交叉确认。

---

<a id="sec-14"></a>
## 14. 一句话总结

现在更准确地说，`route5` 的理论与程序本体，已经不应只理解成一份 Matlab 单文件，而应理解成 Python 主线定义的整套协议：

$$
\text{generalized coherent alphabet}
\;+\;
\text{balanced beamsplitter + dual-homodyne}
\;+\;
\text{IQ coarse-graining}
\;+\;
\text{single-device MDI SDP}.
$$

在这套框架下，当前需要并行记住两条结论：

1. 历史 `trace_povm` 主线已经给出过 `H_{\min} > 2` 的强候选窗口，其中自由半径主线约为 `2.11639`，固定光强主线约为 `2.10102`；
2. 新加入的 `analytic_gaussian_rectangles` 一致性检查表明，在当前截断 trusted-state formal 模型下，同一组历史强点会变成 `formal infeasible`，从而暴露出仍待闭合的模型一致性问题。

因此，今天这份报告最适合作为一份“可交导师的完整理论与程序说明书”：

- Python 是主线实现；
- Matlab 是单点参考镜像；
- `trace_povm` 给出了历史上的高熵候选；
- 解析后端给出了当前最重要的模型一致性提醒；

而若后续要把 `route5` 真正打包成实验主结果，还需要继续完成：

1. 概率层与 trusted-state formal 层的自洽闭环；
2. route5 专用真实 IQ / coarse-grained 概率数据接入；
3. 固定光强到相干振幅映射的独立实验标定。
