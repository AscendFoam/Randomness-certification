# Route4 与 Route4-ex-constrained 的 Matlab 算法形式化说明

## 摘要

本文对当前仓库中两份 Matlab 主脚本的算法与程序执行流程做形式化说明：

- route4 主脚本：
  [guessprobprimal_phaseinsensitive.m](../src/matlab/guessprobprimal_phaseinsensitive.m)
- route4-ex-constrained 主脚本：
  [guessprobprimal_route4_ex_constrained.m](../src/matlab/guessprobprimal_route4_ex_constrained.m)

目标不是重复注释，而是把两份脚本背后的数学对象、变量赋值、概率构造、策略枚举以及 SDP 写成接近论文推导的形式。文中将明确区分：

1. 代码层面的实际赋值；
2. 从这些赋值诱导出的数学模型；
3. 两条路线在 trusted input、coarse-graining 和 primal SDP 结构上的根本差异。

为避免歧义，本文中“route4”特指当前主用 Matlab 文件
[guessprobprimal_phaseinsensitive.m](../src/matlab/guessprobprimal_phaseinsensitive.m)；
它在程序结构上与历史文件
[guessprobprimal_phaseinsensitive_original.m](../src/matlab/guessprobprimal_phaseinsensitive_original.m)
同源，但对 `rho_diag` 的稳定构造更规范。

---

## 1. 记号与统一约定

### 1.1 输入、输出与截断维数

设

$$
D := \text{输入态数量}, \qquad
N := \text{coarse-graining 后的输出数}, \qquad
M := \text{Fock 截断维数}.
$$

对 Matlab 数组而言：

- 输入索引记为 $x \in \{1,\dots,D\}$；
- 输出索引记为 $y \in \{1,\dots,N\}$；
- Fock 数索引在物理上写作 $n \in \{0,\dots,M-1\}$，而在 Matlab 数组中对应位置 $n+1$。

### 1.2 概率表

设实验给出的原始概率表为

$$
\mathrm{ProbData} \in \mathbb{R}_{\ge 0}^{9 \times 256},
$$

其中每一行对应一个固定光强标签，每一列对应一个原始离散输出 bin。

完整光强菜单在两份 Matlab 文件中都写成

$$
\mathrm{full\_mu} = (0,20,40,60,80,100,120,140,160).
$$

若当前选定输入窗口为

$$
(\mu_1,\dots,\mu_D)=\mathrm{selected\_mu\_list},
$$

则对应行索引记为

$$
i_x = \operatorname{index}_{\mathrm{Matlab}}(\mu_x \in \mathrm{full\_mu}) + \mathrm{shift}.
$$

由此得到第 $x$ 个输入对应的原始 256 维行向量

$$
r_x(j) := \mathrm{ProbData}_{i_x,j}, \qquad j=1,\dots,256.
$$

若需要，程序会先做一行归一化：

$$
\hat r_x(j) = \frac{r_x(j)}{\sum_{k=1}^{256} r_x(k)}.
$$

### 1.3 策略集合

两份脚本都采用如下确定性策略索引：

$$
\Lambda = \{1,\dots,N\}^{D+1}.
$$

一个策略写成

$$
\lambda=(\lambda_0,\lambda_1,\dots,\lambda_D).
$$

Matlab 中用矩阵 `LambdaIndices` 存储全部策略，大小为

$$
|\Lambda| \times (D+1), \qquad |\Lambda| = N^{D+1}.
$$

代码目标函数实际使用的是列 $\lambda_x$，即 Matlab 中第 `x+1` 列；$\lambda_0$ 被保留是为了与通用 primal 编码习惯一致。

---

## 2. Route4 Matlab 脚本的形式化描述

### 2.1 默认参数赋值

在当前文件
[guessprobprimal_phaseinsensitive.m](../src/matlab/guessprobprimal_phaseinsensitive.m)
中，默认参数为

$$
(\mu_1,\mu_2,\mu_3) = (100,120,140),
$$

$$
q = (q_1,q_2,q_3) = \left(\tfrac14,\tfrac14,\tfrac12\right),
$$

$$
M = 280, \qquad N = 4, \qquad \mathrm{shift}=0.
$$

因此

$$
D=3,\qquad |\Lambda|=N^{D+1}=4^4=256.
$$

在当前默认参数下，diagonal primal 的标量变量数量为

$$
M \cdot N \cdot |\Lambda|
= 280 \times 4 \times 256
= 286{,}720.
$$

### 2.2 Trusted input 的代码级定义

在 route4 Matlab 文件中，实际被送入 SDP 的并不是完整密度矩阵，而是对角部分

$$
\rho_x^{\mathrm{diag}}(n), \qquad n=0,\dots,M-1.
$$

程序中直接使用的均值参数是

$$
\mu_x^{\mathrm{code}} := \texttt{selected\_mu\_list}(x).
$$

于是对 $n=0,\dots,M-1$，先构造未归一化的泊松权重

$$
\tilde \rho_x(n)
=
\begin{cases}
1, & \mu_x^{\mathrm{code}}=0,\ n=0,\\
0, & \mu_x^{\mathrm{code}}=0,\ n>0,\\
\exp(-\mu_x^{\mathrm{code}})
\dfrac{(\mu_x^{\mathrm{code}})^n}{n!},
& \mu_x^{\mathrm{code}}>0,
\end{cases}
$$

并在截断后重新归一化：

$$
\rho_x^{\mathrm{diag}}(n)
=
\frac{\tilde \rho_x(n)}
{\sum_{m=0}^{M-1}\tilde \rho_x(m)}.
$$

程序层面采用对数域公式

$$
\log \tilde \rho_x(n)
=
-
\mu_x^{\mathrm{code}}
+
n \log \mu_x^{\mathrm{code}}
-
\log \Gamma(n+1),
$$

以避免 `factorial(n)` 的数值溢出。

### 2.3 Coarse-graining

route4 默认采用等宽 coarse-graining。令

$$
b = \operatorname{round}(256/N).
$$

在当前默认参数 $N=4$ 时，有

$$
b = 64.
$$

因此第 $y$ 个输出区间对应

$$
B_y = \{(y-1)b+1,\dots,yb\}, \qquad y=1,\dots,N.
$$

程序构造的 coarse-grained 条件概率为

$$
p_{x,y}

=
\sum_{j \in B_y} \hat r_x(j).
$$

这一步在 Matlab 中由变量 `p(x,y)` 保存。

### 2.4 Strategy indexing

Route4 用 `ndgrid` 枚举

$$
\Lambda = \{1,\dots,N\}^{D+1},
$$

并得到

$$
\LambdaIndices \in \{1,\dots,N\}^{|\Lambda|\times(D+1)}.
$$

对固定输入 $x$，程序从第 `x+1` 列取出

$$
\lambda_x \in \{1,\dots,N\},
$$

用来判断哪些策略会把输入 $x$ 的输出猜成 $y$。

### 2.5 Route4 diagonal primal 的 SDP

Route4 的优化变量为

$$
M_{n,y,\lambda} \ge 0,
\qquad
n=0,\dots,M-1,\ y=1,\dots,N,\ \lambda\in\Lambda.
$$

这正对应 Matlab 变量

$$
\texttt{M\_elements}(n+1,y,\lambda).
$$

目标函数为

$$
\max
\sum_{x=1}^{D}
q_x
\sum_{y=1}^{N}
\sum_{\lambda:\lambda_x=y}
\sum_{n=0}^{M-1}
\rho_x^{\mathrm{diag}}(n)\, M_{n,y,\lambda}.
$$

程序中对固定 $x,y$ 先做

$$
\sum_{\lambda:\lambda_x=y} M_{n,y,\lambda},
$$

再与 $\rho_x^{\mathrm{diag}}$ 做内积。

约束分为两类。

第一类是“完备性 / 无信号”约束：

$$
\sum_{y=1}^{N} M_{n,y,\lambda}
=
\sum_{y=1}^{N} M_{n',y,\lambda},
\qquad
\forall n,n',\lambda.
$$

等价地，也可以写成存在标量 $s_\lambda \ge 0$，使得

$$
\sum_{y=1}^{N} M_{n,y,\lambda} = s_\lambda,
\qquad \forall n,\lambda.
$$

这正对应“对角 POVM 情况下

$$
\sum_y M_{y|\lambda} = s_\lambda I
$$

的对角元都相等”。

第二类是统计匹配约束：

$$
\sum_{\lambda\in\Lambda}
\sum_{n=0}^{M-1}
\rho_x^{\mathrm{diag}}(n)\, M_{n,y,\lambda}
=
p_{x,y},
\qquad
\forall x,y.
$$

最后输出

$$
p_{\mathrm{guess}} = \texttt{cvx\_optval},
\qquad
H_{\min} = -\log_2 p_{\mathrm{guess}}.
$$

### 2.6 Route4 的程序执行流程

按代码执行顺序，route4 Matlab 脚本可以概括为：

1. 固定 `selected_mu_list`、`q_selected`、`M`、`N`。
2. 计算 `selected_full_indices`，确定从 `Probability.mat` 取哪几行。
3. 构造截断并归一化后的 `rho_diag`。
4. 从 `Probability.mat` 读出 256 维原始分布并等宽 coarse-grain 成 `p`。
5. 用 `ndgrid` 枚举全部 `LambdaIndices`。
6. 建立 diagonal primal 并调用 CVX 求解。
7. 输出 `p_guess` 与 `H_min`。

---

## 3. Route4-ex-constrained Matlab 脚本的形式化描述

### 3.1 默认参数赋值

在
[guessprobprimal_route4_ex_constrained.m](../src/matlab/guessprobprimal_route4_ex_constrained.m)
中，默认参数为

$$
(\mu_1,\mu_2,\mu_3) = (100,120,140),
$$

$$
q=(1,0,0),
$$

$$
M=6,
\qquad
\mathrm{shift}=0.
$$

与 route4 不同，这里输出不是由整数 `N` 等宽划分，而是由固定边界

$$
E = (e_0,e_1,e_2,e_3) = (0,121,132,256)
$$

定义，因此

$$
N = |E|-1 = 3.
$$

程序中同时固定 trusted coherent alphabet：

$$
r = (0.54,0.66,0.72),
\qquad
\phi = \left(0,\frac{\pi}{2},\pi\right),
$$

$$
\alpha_x = r_x e^{i\phi_x},
\qquad x=1,2,3,
$$

即

$$
(\alpha_1,\alpha_2,\alpha_3)
=
(0.54,\ 0.66 i,\ -0.72).
$$

于是

$$
D=3,\qquad |\Lambda| = 3^4 = 81.
$$

若仅看 diagonal primal，则变量数量为

$$
M \cdot N \cdot |\Lambda|
= 6 \times 3 \times 81
= 1458.
$$

若看 full primal，则矩阵变量个数为

$$
N \cdot |\Lambda| = 3 \times 81 = 243,
$$

对应 Hermitian 标量数量

$$
243 \times M^2 = 243 \times 36 = 8748.
$$

### 3.2 Trusted coherent states 的构造

Route4-ex-constrained 的核心变化在于 trusted input 不再只保留对角部分。

对每个 $x$，先定义截断相干态系数

$$
\tilde c_{x,n}
=
\exp\!\left(-\frac{|\alpha_x|^2}{2}\right)
\frac{\alpha_x^n}{\sqrt{n!}},
\qquad n=0,\dots,M-1.
$$

程序中同样通过对数域写成

$$
\log \tilde c_{x,n}
=
-\frac{|\alpha_x|^2}{2}
+
n \log \alpha_x
-
\frac12 \log \Gamma(n+1),
$$

然后归一化：

$$
c_{x,n}
=
\frac{\tilde c_{x,n}}
\sqrt{\sum_{m=0}^{M-1} |\tilde c_{x,m}|^2}}.
$$

于是完整 trusted input 记为

$$
\ket{\alpha_x^{(M)}} = \sum_{n=0}^{M-1} c_{x,n}\ket{n},
$$

$$
\rho_x = \ket{\alpha_x^{(M)}}\bra{\alpha_x^{(M)}}.
$$

程序同时保留

$$
\rho_x^{\mathrm{diag}}(n) = |c_{x,n}|^2,
$$

但这里只是为了 optional diagonal primal 对照；正式主问题使用的是完整 $\rho_x$。

### 3.3 自定义 coarse-graining

与 route4 的等宽分块不同，route4-ex-constrained 使用固定边界 $E$。

定义

$$
B_y = \{e_{y-1}+1,\dots,e_y\},
\qquad y=1,2,3.
$$

在当前默认参数下即

$$
B_1 = \{1,\dots,121\},\quad
B_2 = \{122,\dots,132\},\quad
B_3 = \{133,\dots,256\}.
$$

于是原始 coarse-grained 概率为

$$
p_{x,y}^{\mathrm{raw}}
=
\sum_{j\in B_y} \hat r_x(j).
$$

为改善数值稳定性，程序还施加了概率下限 $\varepsilon=\texttt{prob\_floor}$：

$$
\tilde p_{x,y}
=
\max\bigl\{p_{x,y}^{\mathrm{raw}}, \varepsilon\bigr\},
$$

$$
p_{x,y}
=
\frac{\tilde p_{x,y}}
\sum_{y'=1}^{N} \tilde p_{x,y'}}.
$$

当前默认值为

$$
\varepsilon = 10^{-12}.
$$

### 3.4 Optional diagonal primal

新脚本先保留一份 diagonal primal 用于对照。其数学形式和上一节 route4 的 diagonal primal 相同，只是：

1. 输入态对角分布改成了由固定 $\alpha_x$ 导出的 $\rho_x^{\mathrm{diag}}$；
2. 输出概率改成了自定义边界下的 $p_{x,y}$；
3. 默认参数由 $(M,N,q)$ 的新取值给出。

因此该对照问题可写为

$$
\max
\sum_{x=1}^{D}
q_x
\sum_{y=1}^{N}
\sum_{\lambda:\lambda_x=y}
\sum_{n=0}^{M-1}
\rho_x^{\mathrm{diag}}(n)\, M_{n,y,\lambda},
$$

满足

$$
\sum_{y=1}^{N} M_{n,y,\lambda}
=
\sum_{y=1}^{N} M_{n',y,\lambda},
\qquad \forall n,n',\lambda,
$$

以及

$$
\sum_{\lambda\in\Lambda}
\sum_{n=0}^{M-1}
\rho_x^{\mathrm{diag}}(n)\, M_{n,y,\lambda}
=
p_{x,y}.
$$

在当前默认点，这个 diagonal primal 通常会返回 `Infeasible`。

### 3.5 Full primal：route4-ex-constrained 的主问题

这条路线真正关心的是 full primal。

对每个输出 $y$ 与策略 $\lambda$，引入矩阵变量

$$
M_{y,\lambda} \in \mathbb{C}^{M\times M},
\qquad
M_{y,\lambda} \succeq 0.
$$

同时为每个策略引入标量

$$
s_\lambda \ge 0.
$$

程序中矩阵变量通过三维数组 `M_full(:,:,op_idx)` 存放，索引映射是

$$
\mathrm{op\_idx} = (\lambda-1)N + y.
$$

目标函数为

$$
\max
\sum_{x=1}^{D}
q_x
\sum_{y=1}^{N}
\sum_{\lambda:\lambda_x=y}
\operatorname{Re}\!\bigl[\operatorname{Tr}(\rho_x M_{y,\lambda})\bigr].
$$

由于 $\rho_x$ 与 $M_{y,\lambda}$ 都是 Hermitian，理论上迹应为实数；代码中的 `real(...)` 只是数值保护。

第一组约束是完备性约束：

$$
\sum_{y=1}^{N} M_{y,\lambda}
=
s_\lambda I_M,
\qquad
\forall \lambda \in \Lambda.
$$

第二组约束是统计匹配约束：

$$
\sum_{\lambda\in\Lambda}
\operatorname{Re}\!\bigl[\operatorname{Tr}(\rho_x M_{y,\lambda})\bigr]
=
p_{x,y},
\qquad
\forall x,y.
$$

综上，full primal 可以完整写成

$$
\begin{aligned}
\max_{\{M_{y,\lambda}\},\{s_\lambda\}}
\quad &
\sum_{x=1}^{D}
q_x
\sum_{y=1}^{N}
\sum_{\lambda:\lambda_x=y}
\operatorname{Re}\!\bigl[\operatorname{Tr}(\rho_x M_{y,\lambda})\bigr] \\
\text{s.t.}\quad
&
M_{y,\lambda}\succeq 0,
\qquad \forall y,\lambda,\\
&
\sum_{y=1}^{N} M_{y,\lambda} = s_\lambda I_M,
\qquad \forall \lambda,\\
&
\sum_{\lambda\in\Lambda}
\operatorname{Re}\!\bigl[\operatorname{Tr}(\rho_x M_{y,\lambda})\bigr]
= p_{x,y},
\qquad \forall x,y,\\
&
s_\lambda \ge 0,
\qquad \forall \lambda.
\end{aligned}
$$

最后输出同样定义为

$$
p_{\mathrm{guess}} = \texttt{cvx\_optval},
\qquad
H_{\min} = -\log_2 p_{\mathrm{guess}}.
$$

### 3.6 Route4-ex-constrained 的程序执行流程

当前 Matlab 文件的实际执行顺序是：

1. 固定 `selected_mu_list`、`q_selected`、`custom_edges`、`alpha_values`、`M`。
2. 从 `full_mu` 计算 `selected_full_indices`。
3. 构造完整 trusted states $\rho_x$ 及其辅助对角部分 $\rho_x^{\mathrm{diag}}$。
4. 从 `Probability.mat` 读出原始 256 维概率，并按 `custom_edges` 得到 $p_{x,y}^{\mathrm{raw}}$。
5. 对 $p_{x,y}^{\mathrm{raw}}$ 做 `prob_floor` 正则化，得到 $p_{x,y}$。
6. 枚举全部 `LambdaIndices`。
7. 先求 diagonal primal 作为对照。
8. 再求 full primal 作为主结果。
9. 输出 `full_result`、`diagonal_result` 以及 `result.config` 中的关键配置。

---

## 4. 两条 Matlab 算法的形式化差异

若只从程序骨架看，两份脚本都可以概括为

$$
\text{参数初始化}
\to
\text{读 Probability.mat}
\to
\text{输出离散化}
\to
\text{枚举策略}
\to
\text{建 primal SDP}
\to
\text{输出 } H_{\min}.
$$

真正的结构性差异在于下列三点。

### 4.1 Trusted input 的层次不同

Route4：

$$
\rho_x \leadsto \rho_x^{\mathrm{diag}}
$$

程序最终只把对角部分送入 SDP。

Route4-ex-constrained：

$$
\rho_x \ \text{直接进入 full primal}.
$$

这意味着 non-diagonal coherent structure 在 constrained 版本中真正可见。

### 4.2 Coarse-graining 规则不同

Route4：

$$
B_y = \{(y-1)b+1,\dots,yb\}, \qquad b=\operatorname{round}(256/N).
$$

Route4-ex-constrained：

$$
B_y = \{e_{y-1}+1,\dots,e_y\},
\qquad
E=(0,121,132,256).
$$

前者是等宽压缩，后者是固定信息边界压缩。

### 4.3 主问题的优化变量不同

Route4 主问题：

$$
M_{n,y,\lambda}\in \mathbb{R}_{\ge 0}.
$$

Route4-ex-constrained 主问题：

$$
M_{y,\lambda}\in \mathbb{C}^{M\times M},\qquad M_{y,\lambda}\succeq 0.
$$

因此 constrained 版本显式允许一般 Hermitian PSD 测量元，而不再预先压成对角向量。

---

## 5. 默认参数下的结果口径

对于 route4 Matlab 主脚本，现有稳定报告给出的默认 `N=4` 结果位于

$$
H_{\min} \approx 0.1613
$$

的量级，见
[route4_matlab_vs_python_report_cn.md](./route4_matlab_vs_python_report_cn.md)。

对于 route4-ex-constrained Matlab 主脚本，在默认参数

$$
(\mu_1,\mu_2,\mu_3)=(100,120,140),\quad
q=(1,0,0),\quad
E=(0,121,132,256),\quad
\alpha=(0.54,0.66i,-0.72),\quad
M=6
$$

下，你本机 Matlab 已经得到

$$
H_{\min}^{\mathrm{full}}
=
1.227498940472,
$$

且 diagonal primal 返回 `Infeasible`。

这与 Python/MOSEK 主线

$$
H_{\min}^{\mathrm{Python}}
\approx
1.227500864253
$$

仅相差约 $10^{-6}$，属于正常数值误差。

---

## 6. 结论

从“代码对应的数学模型”角度看：

1. Route4 Matlab 主脚本实现的是一个**对角 trusted input + 对角 POVM 变量**的 primal SDP。
2. Route4-ex-constrained Matlab 主脚本则实现了一个**固定 coherent trusted alphabet + 自定义 coarse-graining + full Hermitian PSD primal** 的 constrained 扩展。
3. 两者共享同一份 `Probability.mat`、同一套策略枚举骨架与 `H_{\min}=-\log_2 p_{\mathrm{guess}}` 输出口径。
4. 因此，route4-ex-constrained 最准确的定位不是“完全脱离 route4 的新路线”，而是“在原 route4 Matlab 骨架上，把 trusted input 和主 SDP 测量模型同时强化后的 constrained 版本”。
