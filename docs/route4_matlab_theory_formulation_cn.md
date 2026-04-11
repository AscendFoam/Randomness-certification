# Route4(APD思路) 对应 Matlab 和 Python 脚本的理论表述、算法流程与变量说明

## 摘要

本文档从“理论论文”的写法出发，对以下**三条路线**分别对应的三个 Matlab 脚本进行统一的数学化说明(按最终结果由低到高排序)：

- [`guessprobprimal_phaseinsensitive.m`(原始)](../src/matlab/guessprobprimal_phaseinsensitive.m)
- [`guessprobprimal_route4_ex_constrained.m`(保守拓展)](../src/matlab/guessprobprimal_route4_ex_constrained.m)
- [`guessprobprimal_route4_ex.m`(拓展)](../src/matlab/guessprobprimal_route4_ex.m)

文档的目标不是逐句复述代码，而是将这三个脚本背后的优化问题、统计模型、输入态模型、粗粒化方法、策略索引结构以及最小熵计算方式写成公式化的形式，并解释脚本中每个主要变量的数学意义。整体上：

1. 原始 `guessprobprimal_phaseinsensitive.m` 描述的是一个**相位不敏感、Fock 对角、离散输出**的 primal SDP；
2. `guessprobprimal_route4_ex_constrained.m` 则在保留同一份 `Probability.mat` 与同一 primal 骨架的前提下，把 trusted input 从“仅使用对角 Poisson 分布”升级为“固定非对角截断相干态”，并把主问题从 diagonal primal 升级为 full primal。
3. `guessprobprimal_route4_ex.m` 进一步把 `route4-ex` 的 Python 主线统一压缩成 Matlab 单文件版，使同一套 non-diagonal trusted-input 模型可以对接三类概率后端：`toy`、`apdlike` 与 `external`。

为避免把“实验标签”和“理论参数”混为一谈，本文档默认遵循一个原则：

- **凡涉及脚本理论意义时，以脚本当前实际实现为准**；
- **若注释中出现与程序赋值并不完全一致的物理解释，则只做说明，不自行改写脚本含义。**

---

## 1. 统一记号与问题背景

### 1.1 输入、输出与观测概率

设实验端可选的输入集合记为

$$
\mathcal X = \{1,\dots,D\},
$$

其中第 \(x\) 个输入对应一个测试态。测量输出集合记为

$$
\mathcal C = \{1,\dots,N\}.
$$

实验中可观测的条件概率为

$$
p(c|x), \qquad x\in\mathcal X,\ c\in\mathcal C.
$$

脚本中：

- `D = length(selected_mu_list)`；
- `N` 为 coarse-graining 之后的输出数；
- `p(x,c)` 或 `p(i,k)` 存在变量 `p` 中；
- `q_selected(x)` 是输入 \(x\) 在目标函数中的权重。

### 1.2 猜测概率与最小熵

这三份脚本最终都以**最大猜测概率**

$$
p_{\mathrm{guess}}
$$

为优化目标，并通过

$$
H_{\min} = -\log_2 p_{\mathrm{guess}}
$$

计算认证最小熵。

因此，只要求解器返回最优值 \(\hat p_{\mathrm{guess}}\)，脚本就会输出

$$
\hat H_{\min} = -\log_2 \hat p_{\mathrm{guess}}.
$$

### 1.3 敌手策略的离散表示

这三份脚本都通过一个离散策略索引 \(\lambda\) 来表示敌手的确定性 guessing 规则。程序中使用

$$
\mathrm{LambdaIndices} \in \{1,\dots,N\}^{N^{D+1}\times (D+1)}
$$

枚举所有可能的策略组合。若把第 \(\ell\) 个策略写为

$$
\lambda^{(\ell)} = (\lambda^{(\ell)}_0,\lambda^{(\ell)}_1,\dots,\lambda^{(\ell)}_D),
$$

则脚本中真正进入目标函数的是

$$
\lambda^{(\ell)}_x \equiv \mathrm{LambdaIndices}(\ell, x+1), \qquad x=1,\dots,D.
$$

也就是说，在程序实现里，第 \(x\) 个输入对应的是 `LambdaIndices(:, x+1)` 这一列。

为了统一写法，下面定义

$$
g_x(\ell) := \mathrm{LambdaIndices}(\ell, x+1)\in\{1,\dots,N\},
$$

表示“第 \(\ell\) 个策略在输入 \(x\) 发生时所对应的输出标签”。

---

## 2. `guessprobprimal_phaseinsensitive.m` 的理论模型

## 2.1 物理假设：相位不敏感与 Fock 对角输入

原始脚本对应的核心假设是：

1. 输入态在理论上只通过其 **Fock 对角部分** 进入优化；
2. 测量被视为**相位不敏感**，因此只需要考虑 Fock 对角测量元。

脚本中首先构造

$$
\rho_x^{\mathrm{diag}} = \sum_{n=0}^{M-1} \rho_x(n)\,|n\rangle\langle n|,
$$

其中 \(M\) 是 Fock 截断维数。

若把第 \(x\) 个输入对应的平均光子数记为 \(\mu_x\)，则对角元被写成 Poisson 形式

$$
\rho_x(n) \propto e^{-\mu_x}\frac{\mu_x^n}{n!}, \qquad n=0,\dots,M-1.
$$

脚本采用对数域计算：

$$
\log \rho_x(n) = -\mu_x + n\log\mu_x - \log\Gamma(n+1),
$$

然后再做截断归一化：

$$
\rho_x(n)=
\frac{e^{-\mu_x}\mu_x^n/n!}{\sum_{m=0}^{M-1} e^{-\mu_x}\mu_x^m/m!}.
$$

### 2.1.1 关于 `selected_mu_list` 的程序含义

在这份脚本的当前实现中，`selected_mu_list` 被**直接代入**上式中的 \(\mu_x\)。因此从程序意义上讲，

$$
\mu_x = \texttt{selected\_mu\_list}(x).
$$

如果实验上希望把标签 `100,120,140` 理解为 `1.00,1.20,1.40`，则那属于**额外缩放约定**；当前脚本本身并没有做这一步除以 100 的变换。

### 2.1.2 为什么可以只保留对角部分

原始脚本的物理思想是：当输入态 \(\rho_x\) 本身已经 Fock 对角时，任何测量算符 \(M\) 在目标函数中只通过其对角部分出现，因为

$$
\mathrm{Tr}(\rho_x M)=
\sum_{n=0}^{M-1} \rho_x(n)\,\langle n|M|n\rangle.
$$

令去相干映射为

$$
\Delta(M) = \sum_{n=0}^{M-1} |n\rangle\langle n| M |n\rangle\langle n|,
$$

则对所有 Fock 对角 \(\rho_x\) 都有

$$
\mathrm{Tr}(\rho_x M)=\mathrm{Tr}(\rho_x \Delta(M)).
$$

并且若 \(M\succeq 0\)，则 \(\Delta(M)\succeq 0\)。因此在该模型下，只考虑对角测量元不会损失最优性。

---

## 2.2 观测概率的 coarse-graining

原始脚本从 `Probability.mat` 中读取一张 \(9\times 256\) 概率表。记原始 256 维输出索引为

$$
j\in\{1,\dots,256\}.
$$

脚本把它等宽分成 \(N\) 块。若

$$
b = \mathrm{round}(256/N),
$$

则第 \(c\) 个 coarse-grained 输出对应区间

$$
j \in \{(c-1)b+1,\dots, cb\}.
$$

于是最终输入到 SDP 的观测概率为

$$
p(c|x)=
\sum_{j=(c-1)b+1}^{cb} p_{\mathrm{raw}}(j|x).
$$

对于默认 `N = 4`，该脚本使用的是等宽粗粒化：

$$
[1,64],\ [65,128],\ [129,192],\ [193,256].
$$

需要额外强调的是：原始 Matlab 代码当前写法采用

$$
\texttt{block\_size}=\mathrm{round}(256/N)
$$

并按连续等宽块

$$
\{1,\dots,\texttt{block\_size}\},
\ \{\texttt{block\_size}+1,\dots,2\texttt{block\_size}\},
\ \dots
$$

做粗粒化。因此为了让索引精确覆盖 256 个原始输出而不越界、不遗漏，当前实现实际上要求

$$
N \mid 256.
$$

由于

$$
256 = 2^8,
$$

这在本脚本里就等价于

$$
N \in \{1,2,4,8,16,32,64,128,256\}.
$$

所以，从抽象理论上说，输出数 \(N\) 并不一定必须是 2 的幂；但从**这份原始 Matlab 代码的当前等宽分箱实现**来看，若继续沿用它现在的写法，则最安全、最一致的选择确实就是令 \(N\) 取 2 的幂。

---

## 2.3 原始脚本的 primal SDP

### 2.3.1 优化变量

在原始脚本中，每个策略 \(\ell\) 和每个输出 \(c\) 对应一个对角测量元。程序不直接存矩阵，而只存其对角线向量：

$$
m_{c,\ell}(n)\ge 0, \qquad n=0,\dots,M-1.
$$

脚本变量：

$$
\texttt{M\_elements}(n,c,\ell).
$$

### 2.3.2 目标函数

设输入分布为 \(q_x\)。当输入为 \(x\) 时，第 \(\ell\) 个策略“猜中”的输出标签是 \(g_x(\ell)\)。因此总 guessing probability 写为

$$
p_{\mathrm{guess}}^{\mathrm{diag}}=
\max_{\{m_{c,\ell}\}}
\sum_{x=1}^D q_x
\sum_{\ell}
\sum_{n=0}^{M-1}
\rho_x(n)\,
m_{g_x(\ell),\ell}(n).
$$

把满足 \(g_x(\ell)=c\) 的策略先聚合，则等价写成

$$
p_{\mathrm{guess}}^{\mathrm{diag}}=
\max
\sum_{x=1}^D q_x
\sum_{c=1}^N
\rho_x^\top
\left(
\sum_{\ell:\ g_x(\ell)=c}
m_{c,\ell}
\right).
$$

这正对应脚本里：

$$
\texttt{obj\_expr} = \sum_x q_x \sum_c \rho_x^\top M^{\mathrm{sum}}_{x,c}.
$$

### 2.3.3 归一化约束

脚本通过如下条件约束每个策略下所有输出算符之和是单位阵的倍数：

$$
\sum_{c=1}^N m_{c,\ell}(n)=
w_\ell,
\qquad
\forall n,\ \forall \ell.
$$

这里 \(w_\ell\ge 0\) 是与 \(n\) 无关的标量。程序中并没有显式引入 \(w_\ell\)，而是通过

$$
\sum_{c=1}^N m_{c,\ell}(n+1) = \sum_{c=1}^N m_{c,\ell}(n)
$$

逐分量相等的方式来隐式表达这一条件。

### 2.3.4 统计匹配约束

总测量元定义为

$$
\bar m_c(n) := \sum_{\ell} m_{c,\ell}(n).
$$

实验兼容性要求

$$
\sum_{n=0}^{M-1} \rho_x(n)\,\bar m_c(n) = p(c|x),
\qquad \forall x,c.
$$

这正是脚本中

$$
\texttt{rho\_diag(x,:) * M\_total(:,c) == p(x,c)}
$$

的数学含义。

### 2.3.5 最终优化问题

综上，原始脚本实现的 primal SDP 可以总结为

$$
\begin{aligned}
p_{\mathrm{guess}}^{\mathrm{diag}}
=\max_{\{m_{c,\ell}\}}
&\quad
\sum_{x=1}^{D} q_x
\sum_{\ell}
\sum_{n=0}^{M-1}
\rho_x(n)\,m_{g_x(\ell),\ell}(n)\\
\text{s.t.}
&\quad m_{c,\ell}(n)\ge 0,\\
&\quad \sum_{c=1}^{N} m_{c,\ell}(n)=w_\ell,\ \forall n,\ell,\\
&\quad \sum_{\ell}\sum_{n=0}^{M-1}\rho_x(n)m_{c,\ell}(n)=p(c|x),\ \forall x,c.
\end{aligned}
$$

最终输出

$$
H_{\min}^{\mathrm{diag}}=
-\log_2 p_{\mathrm{guess}}^{\mathrm{diag}}.
$$

---

## 2.4 原始脚本的程序流程

从程序角度看，`guessprobprimal_phaseinsensitive.m` 的流程是：

1. 读取 `selected_mu_list`、`q_selected`、`M`、`N`；
2. 构造 Fock 对角输入分布 `rho_diag`；
3. 从 `Probability.mat` 读取原始 256 维概率表；
4. 按等宽块合并成 `p(x,c)`；
5. 枚举全部 `LambdaIndices`；
6. 用 `M_elements(M,N,num_strategies)` 建 primal SDP；
7. 求出 `cvx_optval = p_guess`；
8. 输出 `H_min = -log2(p_guess)`。

---

## 2.5 原始脚本中主要变量的数学意义

| 脚本变量 | 数学对象 | 含义 |
|---|---|---|
| `selected_mu_list` | \(\{\mu_x\}_{x=1}^D\) | 被选入优化的输入标签/平均光子数参数 |
| `q_selected` | \(q_x\) | 输入 \(x\) 的先验权重 |
| `M` | \(M\) | Fock 截断维数 |
| `N` | \(N\) | coarse-graining 后输出数 |
| `rho_diag(i,:)` | \(\rho_x(n)\) | 输入 \(x\) 的 Fock 对角概率分布 |
| `p(i,k)` | \(p(c\|x)\) | 输入 \(x\) 下观测到 coarse 输出 \(c\) 的概率 |
| `LambdaIndices` | \(\lambda^{(\ell)}\) | 第 \(\ell\) 个离散策略 |
| `M_elements(:,:,k)` | \(m_{c,\ell}(n)\) | 第 \(\ell\) 个策略的对角测量元 |
| `cvx_optval` | \(p_{\mathrm{guess}}\) | 最优猜测概率 |
| `H_min` | \(-\log_2 p_{\mathrm{guess}}\) | 认证最小熵 |

---

## 3. `guessprobprimal_route4_ex_constrained.m` 的理论模型

## 3.1 模型动机

`guessprobprimal_route4_ex_constrained.m` 的核心目标是：

> 在保留 `Probability.mat` 作为实验数据入口、保留 primal SDP 骨架、保留同样的策略索引结构的前提下，把 trusted input 从“只用对角 Poisson 分布”升级为“固定非对角截断相干态”，并用 full primal 来利用这些非对角结构。

因此，这个脚本可以看成原始 route4 的一个 constrained 扩展：

1. **数据接口保持不变**；
2. **coarse-graining 不再等宽，而改为固定边界**；
3. **主结果从 diagonal primal 改成 full primal**。

---

## 3.2 trusted coherent alphabet 的定义

与原始脚本不同，constrained 脚本不再通过 `selected_mu_list` 自动构造 \(\alpha_x\)，而是直接固定一组相干态振幅：

$$
\alpha_x = r_x e^{i\theta_x},
$$

其中默认赋值为

$$
(r_1,r_2,r_3)=(0.54,\,0.66,\,0.72),
$$

$$
(\theta_1,\theta_2,\theta_3)=\left(0,\,\frac{\pi}{2},\,\pi\right),
$$

故而

$$
\alpha_1 = 0.54,\qquad
\alpha_2 = 0.66 i,\qquad
\alpha_3 = -0.72.
$$

脚本随后在 Fock 截断空间中构造

$$
|\alpha_x\rangle_M=
\frac{1}{\mathcal N_x}
\sum_{n=0}^{M-1}
e^{-|\alpha_x|^2/2}
\frac{\alpha_x^n}{\sqrt{n!}}
|n\rangle,
$$

其中 \(\mathcal N_x\) 是截断后的归一化因子。

于是 trusted input 为

$$
\rho_x = |\alpha_x\rangle_M\langle\alpha_x|_M.
$$

与原始脚本不同，这里主问题真正使用的是**完整矩阵 \(\rho_x\)**，而不仅是其对角部分。

### 3.2.1 截断后归一化

由于只保留 \(n=0,\dots,M-1\) 的分量，截断态必须重新归一化：

$$
\mathcal N_x^2=
\sum_{n=0}^{M-1}
e^{-|\alpha_x|^2}
\frac{|\alpha_x|^{2n}}{n!}.
$$

脚本通过对数域公式

$$
\log c_n = -\frac{|\alpha_x|^2}{2} + n\log\alpha_x - \frac{1}{2}\log\Gamma(n+1)
$$

来计算系数，以增强数值稳定性。

---

## 3.3 自定义 coarse-graining

constrained 脚本不再使用等宽 coarse-graining，而采用固定边界

$$
\texttt{custom\_edges} = [0,121,132,256].
$$

如果记原始 256 维输出为 \(j=1,\dots,256\)，则三个离散输出对应

$$
\mathcal B_1=\{1,\dots,121\},
$$

$$
\mathcal B_2=\{122,\dots,132\},
$$

$$
\mathcal B_3=\{133,\dots,256\}.
$$

因此

$$
p(c|x)=\sum_{j\in\mathcal B_c} p_{\mathrm{raw}}(j|x),\qquad c=1,2,3.
$$

若某些 coarse-grained 概率等于 0，脚本进一步施加

$$
p(c|x)\leftarrow \max\{p(c|x),\varepsilon\},\qquad \varepsilon = 10^{-12},
$$

然后逐行归一化。

---

## 3.4 constrained 脚本中的 diagonal primal（对照问题）

为了保留与原始 route4 的可比性，constrained 脚本首先仍然构造一份 diagonal primal：

$$
p_{\mathrm{guess}}^{\mathrm{diag\_cmp}}=
\max_{\{m_{c,\ell}\}}
\sum_{x=1}^{D} q_x
\sum_{\ell}
\mathrm{Tr}\!\left(\rho_x^{\mathrm{diag}} m_{g_x(\ell),\ell}\right).
$$

其约束与原始脚本相同：

$$
m_{c,\ell}(n)\ge 0,
$$

$$
\sum_{c=1}^N m_{c,\ell}(n)=w_\ell,\qquad \forall n,\ell,
$$

$$
\sum_\ell \sum_n \rho_x^{\mathrm{diag}}(n)\,m_{c,\ell}(n)=p(c|x).
$$

该问题的存在只是为了对照“如果仍然强迫测量保持对角，会发生什么”。在当前默认参数下，它通常是 infeasible。

---

## 3.5 constrained 脚本中的 full primal

这才是 constrained 脚本的正式主问题。

### 3.5.1 全矩阵优化变量

对每个输出 \(c\) 和每个策略 \(\ell\)，引入一个 \(M\times M\) 的 Hermitian PSD 矩阵：

$$
M_{c,\ell}\succeq 0.
$$

此外，对每个策略再引入一个非负标量

$$
s_\ell \ge 0,
$$

使得归一化约束可以写为

$$
\sum_{c=1}^{N} M_{c,\ell} = s_\ell I_M.
$$

脚本变量对应为：

- `M_full(:,:,op_idx)` 对应 \(M_{c,\ell}\)；
- `s_lambda(lambda_idx)` 对应 \(s_\ell\)。

### 3.5.2 目标函数

当输入为 \(x\) 时，第 \(\ell\) 个策略对应的“猜中输出”标签仍然是 \(g_x(\ell)\)。因此 full primal 的 guessing probability 为

$$
p_{\mathrm{guess}}^{\mathrm{full}}=
\max_{\{M_{c,\ell}\}}
\sum_{x=1}^{D} q_x
\sum_{\ell}
\mathrm{Tr}\!\left(
\rho_x\,M_{g_x(\ell),\ell}
\right).
$$

把满足 \(g_x(\ell)=c\) 的策略先聚合，可写成

$$
p_{\mathrm{guess}}^{\mathrm{full}}=
\max
\sum_{x=1}^{D} q_x
\sum_{c=1}^{N}
\mathrm{Tr}
\left[
\rho_x
\left(
\sum_{\ell:\ g_x(\ell)=c} M_{c,\ell}
\right)
\right].
$$

与原始脚本相比，关键差别在于：这里的 \(\rho_x\) 是**完整的非对角矩阵**，因此测量元的非对角部分会真实影响目标值与统计兼容性。

### 3.5.3 约束条件

full primal 的约束为：

1. **正定性**

$$
M_{c,\ell}\succeq 0,\qquad \forall c,\ell.
$$

2. **完备性/归一化**

$$
\sum_{c=1}^{N} M_{c,\ell}=s_\ell I_M,\qquad s_\ell\ge 0.
$$

3. **统计匹配**

定义总测量元

$$
\bar M_c := \sum_{\ell} M_{c,\ell},
$$

则实验兼容性要求

$$
\mathrm{Tr}(\rho_x \bar M_c)=p(c|x),\qquad \forall x,c.
$$

### 3.5.4 最终 full primal

因此 constrained 脚本的正式优化问题可以写为

$$
\begin{aligned}
p_{\mathrm{guess}}^{\mathrm{full}}
=\max_{\{M_{c,\ell}\},\{s_\ell\}}
&\quad
\sum_{x=1}^{D} q_x
\sum_{\ell}
\mathrm{Tr}\!\left(\rho_x M_{g_x(\ell),\ell}\right)\\
\text{s.t.}
&\quad M_{c,\ell}\succeq 0,\qquad \forall c,\ell,\\
&\quad \sum_{c=1}^{N} M_{c,\ell}=s_\ell I_M,\qquad \forall \ell,\\
&\quad s_\ell\ge 0,\qquad \forall \ell,\\
&\quad \mathrm{Tr}\!\left(\rho_x \sum_{\ell} M_{c,\ell}\right)=p(c|x),\qquad \forall x,c.
\end{aligned}
$$

最后输出

$$
H_{\min}^{\mathrm{full}}=
-\log_2 p_{\mathrm{guess}}^{\mathrm{full}}.
$$

---

## 3.6 为什么 constrained 脚本不再能直接退化为对角模型

在原始脚本中，由于 \(\rho_x\) 全部对角，可以使用去相干映射 \(\Delta\) 证明对角测量元足够。

但在 constrained 脚本里，

$$
\rho_x = |\alpha_x\rangle_M\langle\alpha_x|_M
$$

一般具有非零的非对角元：

$$
\langle n|\rho_x|m\rangle \neq 0,\qquad n\neq m.
$$

此时若把测量元强行去相干，

$$
\mathrm{Tr}(\rho_x M)
\neq
\mathrm{Tr}(\rho_x \Delta(M))
$$

一般不再成立，因为 \(\rho_x\) 本身会感受到 \(M\) 的非对角结构。因此：

- 原始脚本里“对角化测量元不损失最优性”的论证不再适用；
- 这正是 constrained 脚本需要 full primal 的根本原因。

---

## 3.7 constrained 脚本的程序流程

从程序实现看，`guessprobprimal_route4_ex_constrained.m` 的流程是：

1. 设定固定窗口 `selected_mu_list=[100,120,140]`；
2. 设定权重 `q_selected=[1,0,0]`；
3. 设定 fixed coherent alphabet：
   - `alpha_values = [0.54, 0.66i, -0.72]`；
4. 读取 `Probability.mat`；
5. 用 `custom_edges=[0,121,132,256]` 做 3 输出 coarse-graining；
6. 构造 `LambdaIndices`；
7. 先跑 diagonal primal 作为对照；
8. 再跑 full primal 作为正式结果；
9. 输出 `full_result` 与 `diagonal_result`。

---

## 3.8 constrained 脚本中主要变量的数学意义

| 脚本变量 | 数学对象 | 含义 |
|---|---|---|
| `selected_mu_list` | 输入标签集合 | 决定从 `Probability.mat` 读取哪些行 |
| `q_selected` | \(q_x\) | 目标函数中的输入权重 |
| `custom_edges` | \(\{\mathcal B_c\}\) | 从 256 原始输出到 \(N\) 个离散输出的边界 |
| `radii, phases` | \(r_x,\theta_x\) | 固定相干态字母表参数 |
| `alpha_values` | \(\alpha_x=r_xe^{i\theta_x}\) | trusted coherent amplitudes |
| `rho(:,:,x)` | \(\rho_x\) | 第 \(x\) 个完整截断相干态密度矩阵 |
| `rho_diag(x,:)` | \(\mathrm{diag}(\rho_x)\) | 第 \(x\) 个输入态的对角部分 |
| `p_raw, p` | \(p(c\|x)\) | coarse-grained 概率与正则化后概率 |
| `M_diag` | \(m_{c,\ell}(n)\) | diagonal primal 的对角测量元 |
| `M_full` | \(M_{c,\ell}\) | full primal 的 Hermitian PSD 测量元 |
| `s_lambda` | \(s_\ell\) | 满足 \(\sum_c M_{c,\ell}=s_\ell I\) 的策略权重 |
| `full_result.H_min` | \(H_{\min}^{\mathrm{full}}\) | 正式主结果 |

---

## 3.9 `route4-ex-constrained` 的更细理论解释

如果把 constrained 脚本进一步抽象，它实际上固定了一类三元组

$$
\bigl(\{\rho_x\}_{x=1}^D,\ \{p(c|x)\}_{x,c},\ q\bigr),
$$

然后把这组三元组送入与 route4-ex 相同的 full primal。

这里最关键的是：在 constrained 脚本里，“实验概率来自哪一行数据”和“trusted
input 的理论态是什么”被显式分成了两层。

先记 `Probability.mat` 的原始 256 维概率表为

$$
P_{\mathrm{mat}}(j|\mu),\qquad j=1,\dots,256,
$$

其中 \(\mu\) 只是文件中的行标签。若第 \(x\) 个输入对应的标签为 \(\mu_x\)，则
coarse-grained 实验概率被定义为

$$
p_{\mathrm{exp}}(c|x)=
\sum_{j\in \mathcal B_c} P_{\mathrm{mat}}(j|\mu_x).
$$

与此同时，trusted input 并不是由 \(\mu_x\) 直接生成，而是由另一组固定参数
\(\alpha_x\) 生成：

$$
\rho_x = |\alpha_x\rangle_M\langle \alpha_x|_M.
$$

因此 constrained 脚本真正求解的不是

$$
\text{“由 } \mu_x \text{ 唯一决定 } \rho_x \text{ 的模型”},
$$

而是

$$
\text{“由 } \mu_x \text{ 选择实验概率行，由 } \alpha_x \text{ 给出 trusted 输入态的模型”}.
$$

它们在 SDP 中通过同一组测量元 \(\bar M_c=\sum_\ell M_{c,\ell}\) 被耦合起来：

$$
\mathrm{Tr}(\rho_x \bar M_c)=p_{\mathrm{exp}}(c|x),\qquad \forall x,c.
$$

这正是 constrained 脚本的理论本质。换句话说：

1. `selected_mu_list` 决定从 `Probability.mat` 取哪些实验概率；
2. `alpha_values` 决定 trusted input 在 Fock 截断空间中的非对角结构；
3. `custom_edges` 决定把 256 维实验直方图如何压缩成 \(N=3\) 个离散输出；
4. full primal 则检查是否存在一组矩阵测量元同时兼容 trusted 态与实验概率。

从这个角度看，`route4-ex-constrained` 不是另一套安全证明，而是
`route4-ex external` 家族里的一条固定切片：

$$
\texttt{instance\_mode}=\texttt{external},
\qquad
\alpha=\text{fixed},
\qquad
\mathcal B=\text{fixed},
\qquad
q=\text{fixed}.
$$

它的价值主要在于两点：

1. 可以与原 Matlab route4 脚本共享同一份 `Probability.mat` 数据入口；
2. 可以把 route4-ex 的 non-diagonal trusted-input 思路压缩成一条非常便于导师逐项核对的固定实例。

---

## 4. `guessprobprimal_route4_ex.m` 的理论模型

[`guessprobprimal_route4_ex.m`](../src/matlab/guessprobprimal_route4_ex.m) 的角色，是把
`route4-ex` 的 Python 主线
[`prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)
压缩成一份 Matlab 单文件参考脚本。它的核心特征不是再引入第三种 trusted-input
理论，而是把**同一套 non-diagonal coherent trusted-input 模型**与三种不同的
概率后端统一起来：

1. `toy`
2. `apdlike`
3. `external`

并且允许在同一脚本中切换

1. `diagonal primal`
2. `full primal`
3. `compare`

三种求解模式。

### 4.1 设计定位

从理论上看，这个脚本的定位可以概括为：

> 保持 route4-ex 的 non-diagonal trusted coherent inputs 不变，只把概率模型层改造成可切换的后端，并把 diagonal/full primal 一并保留。

因此，它和 constrained 脚本的关系不是“完全不同的第三条路线”，而是：

- constrained 脚本对应于 general route4-ex 脚本在
  `instance_mode = external`
  且参数固定时的一条特化切片；
- general route4-ex 脚本则把 `toy`、`apdlike`、`external` 三个实例放进了同一个
  Matlab 骨架中。

---

### 4.2 统一的 trusted coherent input 模型

无论概率后端如何选择，`guessprobprimal_route4_ex.m` 都首先固定一组相干态振幅

$$
\alpha_x = r_x e^{i\theta_x},\qquad x=1,\dots,D,
$$

然后构造截断相干态

$$
|\alpha_x\rangle_M=
\frac{1}{\mathcal N_x}
\sum_{n=0}^{M-1}
e^{-|\alpha_x|^2/2}\frac{\alpha_x^n}{\sqrt{n!}}|n\rangle,
$$

并令

$$
\rho_x = |\alpha_x\rangle_M \langle \alpha_x|_M.
$$

这与 constrained 脚本在数学上是同一类 trusted-input 构造，只不过这里
`alpha_values`、`M`、`instance_mode` 都变成了更显式的顶层参数。

脚本中还同时记录

$$
\rho_x^{\mathrm{diag}} = \mathrm{diag}(\rho_x),
$$

这样就可以在同一个实例上同时运行：

1. 只看对角信息的 diagonal primal；
2. 使用完整矩阵结构的 full primal。

---

### 4.3 三类概率后端

#### 4.3.1 `toy` 模式

`toy` 模式使用一个二元 coherent-projector POVM：

$$
E_1 = |\beta\rangle_M\langle \beta|_M,
\qquad
E_2 = I_M - |\beta\rangle_M\langle \beta|_M,
$$

其中 \(\beta = \texttt{probe\_alpha}\)。

于是理论概率直接由

$$
p(c|x)=\mathrm{Tr}(\rho_x E_c)
$$

给出。

这一模式的主要作用不是贴近实验，而是提供一个最小结构例子，验证：

1. 当 trusted input 含有非对角元时；
2. diagonal primal 与 full primal 可能出现明显分叉。

#### 4.3.2 `apdlike` 模式

`apdlike` 模式更接近 route4-ex 的“理论版实验前端”。它先构造一个 Fock 对角
APD count POVM，再做位移共轭。

记原始输出 bin 数为 \(K\)，探测效率为 \(\eta\)，暗计数均值为 \(\nu\)。对固定输入
光子数 \(n\)，当输出 \(k=0,\dots,K-2\) 时，脚本使用

$$
q_{k|n}=
\sum_{t=0}^{\min(n,k)}
\binom{n}{t}\eta^t(1-\eta)^{n-t}
e^{-\nu}\frac{\nu^{k-t}}{(k-t)!}.
$$

最后一个输出 \(k=K-1\) 被当成 overflow bin，其条件概率为

$$
q_{K-1|n}=
1-\sum_{k=0}^{K-2} q_{k|n}.
$$

于是得到一个 Fock 对角 POVM

$$
F_k = \sum_{n=0}^{M-1} q_{k|n}|n\rangle\langle n|.
$$

再记位移振幅为 \(\delta = \texttt{displacement\_alpha}\)，则位移后的 POVM 元为

$$
E_k = D(\delta)^\dagger F_k D(\delta),
$$

其中

$$
D(\delta)=\exp(\delta a^\dagger - \delta^* a).
$$

原始 histogram 概率因此写为

$$
p_{\mathrm{raw}}(k|x)=\mathrm{Tr}(\rho_x E_k).
$$

随后脚本再对这张 raw histogram 做 coarse-graining。

#### 4.3.3 `external` 模式

`external` 模式对应当前最贴实验的数据接口。脚本从外部概率表中读取

$$
p_{\mathrm{raw}}(j|x),
$$

其中 `x` 由

1. `selected_mu_list` 在 `full_mu` 中的位置；
2. 或者 `external_row_indices_override`

决定。

当外部表尚未 coarse-grained 时，脚本再按等覆盖或自定义边界把 256 维原始输出
压缩为较少输出。若 `external_table_already_coarse = true`，则直接把外部表当作
最终 \(p(c|x)\) 使用。

这一模式与 constrained 脚本的关系最直接：

- constrained 脚本就是 external 模式下的一组固定默认参数；
- general route4-ex 脚本则允许更换边界、输入窗口、字母表和求解模式。

---

### 4.4 通用 coarse-graining 与正则化

设 raw output 的边界数组为

$$
e_0=0 < e_1 < \cdots < e_N = K,
$$

则第 \(c\) 个 coarse-grained 输出块为

$$
\mathcal B_c = \{e_{c-1}+1,\dots,e_c\}.
$$

脚本统一使用

$$
p(c|x)=\sum_{j\in\mathcal B_c} p_{\mathrm{raw}}(j|x)
$$

得到最终离散输出分布。

如果不显式给出边界，则采用等覆盖规则

$$
e_c = \left\lfloor \frac{cK}{N} \right\rfloor,\qquad c=0,\dots,N.
$$

若 coarse-grained 概率中出现零项，脚本进一步施加地板正则化

$$
\tilde p(c|x)=\max\{p(c|x),\varepsilon\},
\qquad
\varepsilon = \texttt{prob\_floor},
$$

并逐行归一化。

---

### 4.5 `guessprobprimal_route4_ex.m` 中的 diagonal primal

general route4-ex 脚本的 diagonal primal 与 constrained 脚本的对照问题本质相同，
只是其中的 \(p(c|x)\) 可以来自三种不同后端。

对每个输出 \(c\) 与策略 \(\ell\)，引入对角测量元

$$
m_{c,\ell}(n)\ge 0,\qquad n=0,\dots,M-1.
$$

目标函数为

$$
p_{\mathrm{guess}}^{\mathrm{diag}}=
\max_{\{m_{c,\ell}\}}
\sum_{x=1}^{D} q_x
\sum_{\ell}
\sum_{n=0}^{M-1}
\rho_x^{\mathrm{diag}}(n)\,
m_{g_x(\ell),\ell}(n).
$$

约束仍为

$$
\sum_{c=1}^{N} m_{c,\ell}(n)=w_\ell,\qquad \forall n,\ell,
$$

以及

$$
\sum_{\ell}\sum_{n=0}^{M-1}
\rho_x^{\mathrm{diag}}(n)\,m_{c,\ell}(n)=
p(c|x),\qquad \forall x,c.
$$

这个问题保留的意义，是用来回答一个结构性问题：

> 在给定 non-diagonal trusted inputs 的情况下，如果仍然强制测量元只能 Fock 对角，会损失多少认证能力？

---

### 4.6 `guessprobprimal_route4_ex.m` 中的 full primal

full primal 是该脚本的正式主问题。对每个输出 \(c\) 和策略 \(\ell\)，引入

$$
M_{c,\ell}\succeq 0,
$$

并对每个策略引入标量

$$
s_\ell \ge 0,
$$

使得

$$
\sum_{c=1}^{N} M_{c,\ell}=s_\ell I_M.
$$

目标函数为

$$
p_{\mathrm{guess}}^{\mathrm{full}}=
\max_{\{M_{c,\ell}\},\{s_\ell\}}
\sum_{x=1}^{D} q_x
\sum_{\ell}
\mathrm{Tr}\!\left(\rho_x M_{g_x(\ell),\ell}\right).
$$

实验兼容性约束为

$$
\mathrm{Tr}\!\left(\rho_x \sum_{\ell} M_{c,\ell}\right)=p(c|x),
\qquad
\forall x,c.
$$

与 constrained 脚本相比，这里的新意不在 primal 本身，而在于：

1. 同一份 full primal 可以对接 `toy`、`apdlike`、`external` 三种 \(p(c|x)\)；
2. 同一份 Matlab 脚本就能测试“概率模型变化”与“测量约束变化”这两类效应。

---

### 4.7 general route4-ex 脚本的程序流程

从程序角度看，`guessprobprimal_route4_ex.m` 的主流程是：

1. 读取 `instance_mode`、`solve_mode`、`alpha_values`、`q_selected`、`M`；
2. 构造完整截断相干态 `rho(:,:,x)` 与其对角 `rho_diag(x,:)`；
3. 按 `instance_mode` 生成概率表：
   - `toy`：直接算 `Tr(rho_x E_c)`；
   - `apdlike`：先算 raw histogram，再 coarse-graining；
   - `external`：从外部表取行，再 coarse-graining；
4. 做逐行归一化与 `prob_floor` 正则化；
5. 枚举全部 `LambdaIndices`；
6. 若需要，则先跑 diagonal primal；
7. 若需要，则再跑 full primal；
8. 输出包含配置、输入态诊断、概率表与 primal 结果的 `result` 结构体。

---

### 4.8 general route4-ex 脚本中主要变量的数学意义

| 脚本变量 | 数学对象 | 含义 |
|---|---|---|
| `instance_mode` | 概率后端选择 | 在 `toy`、`apdlike`、`external` 三类概率模型间切换 |
| `solve_mode` | 求解模式选择 | 在 `diagonal`、`full`、`compare` 间切换 |
| `alpha_values` | \(\alpha_x\) | trusted coherent amplitudes |
| `rho(:,:,x)` | \(\rho_x\) | 第 \(x\) 个完整截断相干态密度矩阵 |
| `rho_diag(x,:)` | \(\rho_x^{\mathrm{diag}}\) | 第 \(x\) 个 trusted input 的对角部分 |
| `probe_alpha` | \(\beta\) | toy coherent-projector POVM 的探针振幅 |
| `displacement_alpha` | \(\delta\) | APD-like 位移测量中的位移振幅 |
| `detection_efficiency` | \(\eta\) | APD-like 模型的探测效率 |
| `dark_count_mean` | \(\nu\) | APD-like 模型的暗计数均值 |
| `raw_probability_table_before_normalization` | \(p_{\mathrm{raw}}(j\|x)\) | coarse-graining 前的原始概率表 |
| `probabilities_raw` | \(p(c\|x)\) | coarse-graining 后、正则化前的概率 |
| `probabilities` | \(\tilde p(c\|x)\) | 正则化后的最终概率 |
| `LambdaIndices` | \(\lambda^{(\ell)}\) | 第 \(\ell\) 个离散策略 |
| `M_diag` | \(m_{c,\ell}(n)\) | diagonal primal 中的对角测量元 |
| `M_full` | \(M_{c,\ell}\) | full primal 中的 Hermitian PSD 测量元 |
| `s_lambda` | \(s_\ell\) | 满足 \(\sum_c M_{c,\ell}=s_\ell I\) 的策略权重 |
| `input_offdiagonal_metrics` | 非对角强度诊断量 | 用来量化 trusted input 的非对角程度 |

---

## 4.9 `route4-ex` 的更细理论解释

### 4.9.1 `route4-ex` 实际上是在扫描“实例族”，而不是改写 SDP 骨架

从更高一层看，general route4-ex 脚本固定的并不是某一个单点实例，而是一整个实例族

$$
\mathcal I =
\Bigl(
\{\rho_x\}_{x=1}^D,\ 
\{p(c|x)\}_{x,c},\ 
q,\ 
M,\ 
N
\Bigr).
$$

其中：

1. \(\rho_x\) 由 `alpha_values` 给定；
2. \(p(c|x)\) 由 `toy`、`apdlike` 或 `external` 后端给定；
3. \(q\) 由 `q_selected` 给定；
4. \(M\) 与 \(N\) 分别给出 Fock 截断和离散输出数。

在 route4-ex 的搜索过程中，被改变的主要是这组实例参数，而不是 primal SDP 的数学结构。
也就是说，route4-ex 的各种搜索并不是“不断换一套安全模型”，而是不断换

$$
\mathcal I \mapsto \mathcal I'
$$

之后，再反复求解同一类 full primal。

### 4.9.2 三类概率后端分别承担什么理论角色

在 route4-ex 中，

$$
p(c|x)
$$

可以由三种不同方式得到：

1. `toy`：用最小结构例子验证“non-diagonal trusted inputs 会不会让 diagonal 与 full 分叉”；
2. `apdlike`：用一个解析可控、但仍带有 APD 风味的理论前端产生概率；
3. `external`：直接从外部概率表读入，再做 coarse-graining。

因此三者的关系应理解为：

$$
\texttt{toy} \rightarrow \text{结构验证},
\qquad
\texttt{apdlike} \rightarrow \text{理论前端模型},
\qquad
\texttt{external} \rightarrow \text{实验接口模型}.
$$

其中只有 `external` 模式直接对应“把实验概率表送进 SDP”；`apdlike` 的结果不能直接当作实验结论，
它更适合被理解为“如果前端测量真接近某种 APD-like 响应，那么这条路线在理论上可能达到怎样的认证值”。

### 4.9.3 `route4-ex` 中真正起作用的非对角资源是什么

在 route4-ex 里，trusted input 的非对角性会通过

$$
\mathrm{Tr}(\rho_x M_{c,\ell})
$$

直接进入目标函数与统计匹配约束。为量化这种非对角资源，脚本还记录了
`input_offdiagonal_metrics`。如果定义去相干映射

$$
\Delta(\rho_x)=\sum_{n=0}^{M-1}|n\rangle\langle n|\rho_x|n\rangle\langle n|,
$$

则一个自然的非对角强度指标可以写为

$$
R_{\mathrm{off}}(\rho_x)=
\frac{\|\rho_x-\Delta(\rho_x)\|_F}{\|\rho_x\|_F}.
$$

route4-ex 输出里的 `offdiag_over_fro` 就是在数值上扮演这一角色。

因此，route4-ex 的 full primal 真正利用的是：

1. trusted input 的非对角矩阵结构；
2. 实验 coarse-grained 概率表中的不对称性；
3. 生成轮权重 \(q_x\) 对目标函数的放大作用。

### 4.9.4 `q_selected`、边界与半径搜索各自改变什么

在 route4-ex 中，有三类特别重要的可调对象：

1. `q_selected`
2. `custom_edges`
3. `alpha_values`，或等价地写成半径与相位 \((r_x,\theta_x)\)

它们分别改变的是不同层面：

$$
q_x \quad \text{改变目标函数的加权方式},
$$

$$
\mathcal B_c \quad \text{改变实验 raw histogram 被压缩成离散输出的方式},
$$

$$
\alpha_x=r_x e^{i\theta_x} \quad \text{改变 trusted input 的几何位置与非对角结构}.
$$

这三者同时变化时，route4-ex 实际上是在搜索一个联合兼容问题：

> 是否能找到一组 non-diagonal trusted inputs，使它们既匹配所选 coarse-grained 实验概率，又在目标输入分布 \(q\) 下尽量压低敌手的猜测概率。

这也解释了为什么 route4-ex 的增益不是简单来自“输出数更多”或者“光强更大”，而往往来自

$$
\text{trusted inputs} + \text{probability binning} + q
$$

的联合调优。

### 4.9.5 `SCS` 与 `MOSEK` 在 route4-ex 搜索中的分工

虽然 Matlab 单文件脚本本身通常直接跑正式求解，但 route4-ex 的 Python 主线在大规模搜索时采用了更细的分工：

1. 先用较快但相对粗糙的求解做初筛；
2. 再用 `MOSEK` 对高值候选做正式复核；
3. 最后对最强候选做病态边界定位和残差检查。

因此，route4-ex 的正式结论应始终以

$$
\texttt{MOSEK optimal}
$$

对应的 full-primal 结果为准，而不是以快速筛选值为准。

---

## 5. 三个脚本的本质差别

若用一句最短的话概括：

### 原始脚本

$$
\text{Probability.mat} + \text{Fock 对角输入模型} + \text{等覆盖 coarse-graining} + \text{diagonal primal}
$$

### constrained 脚本

$$
\text{Probability.mat} + \text{固定 non-diagonal coherent trusted inputs} + \text{固定 custom coarse-graining} + \text{full primal}
$$

### general route4-ex 脚本

$$
\text{non-diagonal coherent trusted inputs} + \text{toy / apdlike / external 概率后端} + \text{等覆盖或 custom coarse-graining} + \text{diagonal / full / compare}
$$

因此，三者的关系更准确地说是：

1. 原始脚本提供最保守的 phase-insensitive/Fock-diagonal primal 基线；
2. constrained 脚本提供一条固定参数的 route4-ex external 主线；
3. general route4-ex 脚本则把 route4-ex 的核心建模能力统一成了一个 Matlab 单文件框架。

---

## 5.1 三份 Matlab 脚本中可直接调整的主要参数

为了便于导师或实验室直接对照 Matlab 代码，本节把三份脚本里最主要的顶层可调参数统一列出。

### 原始 `guessprobprimal_phaseinsensitive.m`

这份脚本中，最直接影响结果口径的顶层参数包括：

1. `selected_mu_list`
2. `q_selected`
3. `M`
4. `N`
5. `full_mu`
6. `shift`
7. `Probability.mat` 本身

其中

$$
M \quad \text{控制 Fock 空间截断，}
\qquad
N \quad \text{控制输出 coarse-graining 的精度。}
$$

并且在当前这份 Matlab 脚本的等宽分箱实现下，`N` 还额外受到

$$
N \mid 256
$$

的实现约束。

### `guessprobprimal_route4_ex_constrained.m`

这份脚本的主要可调参数包括：

1. `selected_mu_list`
2. `q_selected`
3. `M`
4. `full_mu`
5. `shift`
6. `custom_edges`
7. `radii`
8. `phases`
9. `alpha_values`
10. `prob_floor`
11. `probability_variable_name`
12. `probability_filename`
13. `run_diagonal_primal`
14. `run_full_primal`
15. `save_result_mat`

其中最影响物理模型与 formal 结果的，是下面这组核心参数：

$$
(\texttt{selected\_mu\_list},\ \texttt{custom\_edges},\ \alpha,\ q,\ M).
$$

### `guessprobprimal_route4_ex.m`

这份 general route4-ex 脚本的顶层可调参数最多，可以按三层来分。

第一层是全局控制参数：

1. `instance_mode`
2. `solve_mode`
3. `preferred_solver`
4. `save_result_mat`

第二层是 route4-ex 共享物理参数：

1. `selected_mu_list`
2. `q_selected`
3. `M`
4. `full_mu`
5. `shift`
6. `alpha_values`
7. `prob_floor`

第三层是实例后端专属参数：

1. `toy` 模式
   - `probe_alpha`
2. `apdlike` 模式
   - `displacement_alpha`
   - `apdlike_raw_num_bins`
   - `apdlike_num_outputs`
   - `apdlike_custom_edges`
   - `detection_efficiency`
   - `dark_count_mean`
3. `external` 模式
   - `probability_filename`
   - `probability_variable_name`
   - `external_table_already_coarse`
   - `external_num_outputs`
   - `external_custom_edges`
   - `external_row_indices_override`

因此，对 general route4-ex 来说，真正定义一个具体实例的最小参数组通常是

$$
(\texttt{instance\_mode},\ \texttt{solve\_mode},\ \alpha,\ q,\ M,\ \text{coarse-graining 参数},\ \text{概率后端参数}).
$$

---

## 6. 现阶段三条路线的结果与搜索口径

本节不再只给“默认参数”单点结果，而是把截至当前已经确认的三条路线主结果、
搜索推进思路以及它们各自的理论含义一并整理出来。

## 6.1 原始 route4 的当前最好结果与搜索思路

截至当前，原始 route4 的结果需要区分成两种口径。

第一种是**严格原始 Matlab 兼容口径**。由于原始
[`../src/matlab/guessprobprimal_phaseinsensitive.m`](../src/matlab/guessprobprimal_phaseinsensitive.m)
当前采用等宽分箱写法，因此最自然、最稳妥的 `N` 选择应满足

$$
N \mid 256,
$$

也就是优先取 2 的幂。在这层口径下，目前最好 formal 点来自
[`../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json)
中的

$$
\texttt{selected\_mu\_list}=[140,160],\qquad
q=[0.5,0.5],\qquad
M=280,\qquad
N=16,
$$

对应

$$
H_{\min}\approx 0.5272804348158399.
$$

第二种是**Python 扩展实现口径**。在 Python 版 route4 中，等覆盖 coarse-graining
不是通过固定 `block_size=round(256/N)` 实现，而是通过严格覆盖 256 个 raw bins 的边界构造，
因此允许 `N=20` 这类非 2 幂输出数。在这层口径下，目前最好 formal 点同样来自
[`../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json)：

$$
\texttt{selected\_mu\_list}=[140,160],\qquad
q=[0.5,0.5],\qquad
M=280,\qquad
N=20,
$$

对应

$$
H_{\min}\approx 0.5549870213014914.
$$

因此，若强调“与原始 Matlab route4 完全同口径”，应把

$$
H_{\min}\approx 0.5273
$$

视为当前最好值；若允许采用 Python 版更一般的精确覆盖 coarse-graining，则可把

$$
H_{\min}\approx 0.5550
$$

视为扩展实现下的最好值。

如果把目前 route4 的搜索推进过程压缩成最核心的几步，可以概括为：

1. 先在 [`../output/qrng_routes/route4_summary.json`](../output/qrng_routes/route4_summary.json) 中做 `distribution-only` 粗筛，定位哪些输入窗口和输出数更值得正式认证；
2. 再围绕高光强窗口做 formal primal/dual 复核，发现两输入 `[140,160]` 和三输入 `[120,140,160]` 是最值得继续压榨的区域；
3. 然后把输出数从 \(N=12\) 推到 \(N=16\)、再推到 \(N=20\)，检查 formal `H_min` 是否随离散输出数增长；
4. 最后再围绕这些窗口测试偏置 \(q\)。

已经落盘的两输入定向扫描显示出很清楚的趋势：

$$
N=12 \Rightarrow H_{\min}^{\max}\approx 0.4507,
$$

$$
N=16 \Rightarrow H_{\min}^{\max}\approx 0.5273,
$$

$$
N=20 \Rightarrow H_{\min}^{\max}\approx 0.5550.
$$

这说明在原始 route4 的 phase-insensitive / diagonal 框架里，增大输出数 \(N\) 的确仍有帮助。
但另一方面，偏置 \(q\) 并没有像 route4-ex 那样成为主要增益来源。相反，在
`[140,160]` 这条两输入高光强主线上，当前最好的 formal 点始终是均匀权重

$$
q=[0.5,0.5].
$$

如果采用 Python 扩展 coarse-graining 口径，则在 \(N=20\) 时：

$$
q=[0.5,0.5] \Rightarrow H_{\min}\approx 0.5550,
$$

$$
q=[0.1,0.9] \Rightarrow H_{\min}\approx 0.5218,
$$

$$
q=[0.02,0.98] \Rightarrow H_{\min}\approx 0.5098.
$$

这说明对原始 route4 而言，生成分布偏置并没有带来 formal 认证上的净收益。

若只看严格 Matlab 兼容口径，则最应拿来汇报的仍是

$$
N=16,\qquad q=[0.5,0.5],\qquad H_{\min}\approx 0.5273.
$$

对于三输入窗口 `[120,140,160]`，定向扫描文件
[`../output/qrng_routes/route4_targeted_scan_triple_120_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_triple_120_140_160_v1.json)
截至本文档更新时仍在继续落盘，目前已完成的 \(N=12\) 四个点里，最好者为

$$
q=\left[\frac13,\frac13,\frac13\right],
\qquad
H_{\min}\approx 0.4585660416,
$$

暂时也没有显示出“更偏向最高光强输入会显著抬高 formal 值”的趋势。

因此，原始 route4 目前的阶段性判断可以概括为：

$$
\text{两输入高光强窗口} + \text{更大 } N
$$

仍是最有效的推进方向，而单纯偏置 \(q\) 并不是主要增长杆。

## 6.2 `route4-ex-constrained` 的主结果与搜索口径

对 constrained Matlab 脚本，在默认参数

$$
\texttt{selected\_mu\_list}=[100,120,140],\qquad
\texttt{q\_selected}=[1,0,0],
$$

$$
\texttt{custom\_edges}=[0,121,132,256],\qquad
\alpha=[0.54,\,0.66 i,\,-0.72],\qquad
M=6,\qquad
N=3
$$

下，你本机 Matlab 的输出为

$$
\texttt{Full primal status}=\texttt{Solved},
$$

$$
\texttt{Full primal } H_{\min}\approx 1.227498940472,
$$

$$
\texttt{Diagonal primal status}=\texttt{Infeasible}.
$$

这与 Python/MOSEK 主线结果

$$
H_{\min}\approx 1.227500864253
$$

之间只有约 \(10^{-6}\) 量级的差别，可视为正常的数值误差。

与之对应的 Python 结果文件包括：

- [`../output/qrng_routes/route4_ex_constrained_baseline_compare.json`](../output/qrng_routes/route4_ex_constrained_baseline_compare.json)
- [`../output/qrng_routes/route4_ex_constrained_matlab_style_compare.json`](../output/qrng_routes/route4_ex_constrained_matlab_style_compare.json)
- [`../output/qrng_routes/route4_ex_mosek_verify_3out_free_r054_066_072_q100.json`](../output/qrng_routes/route4_ex_mosek_verify_3out_free_r054_066_072_q100.json)

其中都给出了同一条核心结论：

$$
\text{full primal optimal},\qquad
H_{\min}\approx 1.2275,
$$

而对照用的 diagonal primal 则是 infeasible。

这条结果的理论含义是：一旦 trusted input 被替换为固定的 non-diagonal 截断相干态，
即便仍然使用同一份 `Probability.mat` 和同一组 coarse-grained 输出边界，原始 route4 那种
“只优化对角测量元”的表达已经不足以兼容实验概率；必须使用 full primal 才能得到可行的正式结果。

从搜索思路上看，`route4-ex-constrained` 并不是一条“重新做大范围参数搜索”的路线。
它更准确的定位是：

1. 先由更一般的 route4-ex 外部概率搜索确定一个有效窗口；
2. 再把这个窗口冻结成一条便于 Matlab 对照和导师逐项核查的核心切片；
3. 最后用这一切片验证“non-diagonal trusted input + external probability + full primal”
   这条模型链条是否稳定成立。

因此，`route4-ex-constrained` 的价值主要不在于“追求最高值”，而在于：

$$
\text{把 route4-ex 的核心机制压缩成一条最容易核对的固定主线。}
$$

## 6.3 一般 `route4-ex` 的当前最好结果与搜索思路

对新的 general route4-ex Matlab 脚本，如果保持其默认配置

$$
\texttt{instance\_mode}=\texttt{external},\qquad
\texttt{solve\_mode}=\texttt{compare},
$$

并继续使用同一组

$$
\texttt{selected\_mu\_list}=[100,120,140],\qquad
\texttt{q\_selected}=[1,0,0],
$$

$$
\texttt{external\_custom\_edges}=[0,121,132,256],\qquad
\alpha=[0.54,\,0.66 i,\,-0.72],\qquad
M=6,\qquad
N=3,
$$

那么它在理论上应当复现与 constrained 脚本同一条 `external + full primal` 主线；
差别只在于：

1. 它额外保留了 `toy` 与 `apdlike` 两类概率后端；
2. 它把 `diagonal/full/compare` 统一成了一个顶层切换接口；
3. 它把 coarse-graining 的等覆盖和自定义边界都纳入了同一脚本。

换句话说，`guessprobprimal_route4_ex.m` 的默认 external 配置，不是另一套
新结果口径，而是对 constrained 主线的一个更一般的 Matlab 包装。

不过，真正推动 route4-ex 达到当前最高值的，并不是这个固定默认点，而是后续在
Python 主线里完成的多轮局部精修。当前已正式确认的最强点来自
[`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)
与
[`../output/qrng_routes/route4_ex_residual_diag_q419over1024.json`](../output/qrng_routes/route4_ex_residual_diag_q419over1024.json)：

$$
\texttt{selected\_mu\_list}=[100,120,140],
$$

$$
\texttt{custom\_edges}=[0,121,132,256],
$$

$$
q=[1,0,0],
$$

$$
\theta = \left(0,\frac{\pi}{2},\pi\right),
$$

$$
r = [0.5379541015625,\ 0.6620458984375,\ 0.7179541015625],
$$

$$
\alpha = [0.5379541015625,\ 0.6620458984375 i,\ -0.7179541015625],
$$

对应

$$
H_{\min}\approx 1.5439508969460896.
$$

邻近的稳定旁证点
[`../output/qrng_routes/route4_ex_residual_diag_q209over512.json`](../output/qrng_routes/route4_ex_residual_diag_q209over512.json)
则给出

$$
H_{\min}\approx 1.5384598585291962,
$$

并具有同样量级的非对角强度与很小的约束残差。因此目前 route4-ex 的主结果不是“孤立幸运点”，而是一小段非常窄但数值上仍然干净的稳定前沿。

如果把 route4-ex 的推进过程按方法论拆开，大致经历了以下阶段：

1. 先在 `external` 模式下做小窗口可行性试探，确认 `Probability.mat` 驱动的外部概率表确实能与 non-diagonal trusted coherent inputs 一起进入 full primal；
2. 再比较 `2/3/4` 输出边界族，发现 `3` 输出边界

$$
[0,121,132,256]
$$

比更粗或更细的分法更容易维持 formal feasibility；
3. 然后比较相位图样，发现

$$
\left(0,\frac{\pi}{2},\pi\right)
$$

这一组相位更有利于把三个 trusted 输入拉开；
4. 再把输入权重偏向

$$
q=[1,0,0],
$$

使 guessing probability 的目标函数主要聚焦到最有利的那个生成输入上；
5. 在此基础上，从固定半径推进到 `free_monotone_radii` 局部精修，即在保持

$$
0 < r_1 \le r_2 \le r_3
$$

的前提下连续微调三组半径；
6. 最后使用病态边界扫描与残差体检，确认高值点没有明显约束失控。

这里特别值得强调的是：route4-ex 的主增益并不是简单来自“输出数增加”，也不是简单来自“光强更大”，而是来自以下三件事的协同：

$$
\text{非对角 trusted inputs}
\;+\;
\text{更合适的 coarse-graining 边界}
\;+\;
\text{生成轮权重偏置 } q.
$$

当前最高稳定点的 `input_offdiagonal_metrics` 还表明，这组 trusted input 的非对角成分并不弱。
在
[`../output/qrng_routes/route4_ex_residual_diag_q419over1024.json`](../output/qrng_routes/route4_ex_residual_diag_q419over1024.json)
中，最大的

$$
R_{\mathrm{off}}(\rho_x)
\approx 0.73624,
$$

说明 full primal 确实在利用 substantial 的非对角结构，而不是只在对角近似附近微调。

与此同时，该点的残差量级仍然保持在

$$
10^{-9}\sim 10^{-10}
$$

附近，因此可以作为当前阶段的正式主结果，而不只是探索性高值。

## 6.4 三条路线当前最好结果的并列比较

为方便后续汇报，可把三条路线当前已确认的最好 formal 结果总结为下表。

| 路线 | 当前最好 formal 点 | \(H_{\min}\) | 主要来源 |
|---|---|---:|---|
| 原始 `route4`（Matlab 兼容口径） | `selected_mu_list=[140,160]`, `q=[0.5,0.5]`, `M=280`, `N=16` | `0.527280` | [`../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json) |
| 原始 `route4`（Python 扩展口径） | `selected_mu_list=[140,160]`, `q=[0.5,0.5]`, `M=280`, `N=20` | `0.554987` | [`../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json) |
| `route4-ex-constrained` | `selected_mu_list=[100,120,140]`, `q=[1,0,0]`, `custom_edges=[0,121,132,256]`, `alpha=[0.54,0.66i,-0.72]`, `M=6`, `N=3` | `1.227501` | [`../output/qrng_routes/route4_ex_constrained_baseline_compare.json`](../output/qrng_routes/route4_ex_constrained_baseline_compare.json) |
| 一般 `route4-ex` | `selected_mu_list=[100,120,140]`, `q=[1,0,0]`, `custom_edges=[0,121,132,256]`, `phase=(0,\pi/2,\pi)`, `free_monotone_radii=[0.5379541,0.6620459,0.7179541]`, `M=6`, `N=3` | `1.543951` | [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json) |

从这张表可以直观看出三条路线的层级关系：

1. 原始 route4 无论按 Matlab 兼容口径还是 Python 扩展口径，仍保持最强的 phase-insensitive / diagonal 约束，因此结果最低；
2. `route4-ex-constrained` 通过 fixed non-diagonal trusted inputs 把 formal 值从约 `0.55` 抬到约 `1.23`；
3. 一般 `route4-ex` 再通过联合优化边界、相位、半径与 \(q\)，把 formal 值继续推到约 `1.54`。

## 6.5 这一阶段结果对理论模型的反向说明

把这些结果反过来读，其实也能得到三条重要的理论结论。

第一，原始 route4 的瓶颈并不只是“参数没扫够”，而更像是：

$$
\text{对角输入模型} + \text{对角测量表达}
$$

本身限制了它能利用的统计结构。因此即便把 \(N\) 从 \(12\) 推到 \(20\)，Python 扩展口径下目前最好 formal 结果也只到约 `0.555`；若坚持原始 Matlab 兼容口径，则最好值为约 `0.527`。

第二，`route4-ex-constrained` 的结果说明，只要 trusted input 改成 non-diagonal coherent states，full primal 就能显著释放认证能力；
而 diagonal primal 在同一实例上直接 infeasible，也从反面说明“去掉非对角信息之后，这一实例已经无法被原 route4 模型解释”。

第三，一般 `route4-ex` 的进一步提升说明：在 non-diagonal trusted-input 框架内，真正值得搜索的不是单一参数，而是

$$
(\alpha,\ \mathcal B,\ q)
$$

这三个层面的联合兼容性。也正因为如此，route4-ex 的最好结果并不是沿着原始 route4 的单一参数轴外推得到的，而是来自一套更细的联合搜索流程。

---

## 7. 总结

从理论上看，这三份 Matlab 脚本可视为同一问题族的三个层级：

1. [`guessprobprimal_phaseinsensitive.m`](../src/matlab/guessprobprimal_phaseinsensitive.m)
   给出最保守、最贴 phase-insensitive/Fock-diagonal 假设的 primal SDP；
2. [`guessprobprimal_route4_ex_constrained.m`](../src/matlab/guessprobprimal_route4_ex_constrained.m)
   给出一条固定参数的 non-diagonal trusted-input external 主线；
3. [`guessprobprimal_route4_ex.m`](../src/matlab/guessprobprimal_route4_ex.m)
   则把 route4-ex 的核心建模接口泛化成同一份 Matlab 单文件，实现
   `toy / apdlike / external` 三后端与 `diagonal / full / compare` 三求解模式的统一。

从程序骨架上看，它们仍然共享以下结构：

1. 都以离散输出概率 \(p(c|x)\) 为认证约束；
2. 都通过 `LambdaIndices` 枚举离散策略；
3. 都把 primal SDP 的最优值解释为 \(p_{\mathrm{guess}}\)；
4. 都最终输出

$$
H_{\min} = -\log_2 p_{\mathrm{guess}}.
$$

因此，如果导师想把 general route4-ex Matlab 脚本理解为

> “在原 route4 / constrained route4-ex 语法风格上，对 route4-ex Python 主线所做的一次统一封装”

这是准确的；但如果把它理解为“又引入了全新的一套安全模型”，则并不准确。

从现阶段结果看，这三条路线也给出了一个很清楚的层级结论：

1. 原始 route4 若按 Matlab 兼容口径，则当前最好 formal 结果约为 `0.5273 bit`；若按 Python 扩展口径，则约为 `0.5550 bit`；
2. `route4-ex-constrained` 在固定 Matlab 友好切片上可稳定达到约 `1.2275 bit`；
3. 一般 `route4-ex` 在完成边界、半径、相位与 \(q\) 的联合精修后，当前最好正式结果约为 `1.54395 bit`。

因此，若这份文档被用于导师讨论，比较稳妥的理解方式是：

$$
\text{route4} \subset \text{route4-ex-constrained} \subset \text{general route4-ex 的可表达实例族},
$$

其中越往右，模型表达能力越强、可搜索空间越大、能够利用的 non-diagonal 结构也越充分；
但与此同时，也越需要把“实验概率入口”“trusted-input 假设”“coarse-graining 规则”和“正式求解结果”分别说清楚。
