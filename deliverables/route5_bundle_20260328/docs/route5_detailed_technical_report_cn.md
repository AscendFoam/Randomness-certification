# Route5 独立技术详解报告

## 1. 执行摘要

本文只讨论 `route5`，不把其它路线作为正文主线。目标是给出一份可以直接审阅的技术说明，回答四个问题：

1. `route5` 的理论协议原型到底是什么。
2. 当前代码如何把这个原型落成可运行的搜索与认证流程。
3. 在固定光强约束下，当前数值结果到底说明了什么。
4. 后续通知实验室采集 `route5` 专用真实概率数据后，如何把实验数据接入现有流程。

当前最重要的两条结论是：

- 在自由搜索窗口内，当前正式 `MOSEK` 最优结果达到  
  `H_min ≈ 2.1163917383`，对应结果文件为  
  [r0.0000_0.8500_1.2500.json](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)。
- 在更贴近实验限制的固定光强主线 `[0,80,160]` 下，当前正式 `MOSEK` 结果仍达到  
  `H_min ≈ 2.1010172143`，对应结果文件为  
  [route5_fixed_intensity_080160_scale120.json](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)。

因此，如果更关心“贴实验限制时是否仍有希望过 2 bit”，应优先看固定光强主线，而不是只看自由搜索最好点。

本文的总体判断是：

- `route5` 已经在理论原型层面给出清晰的 `H_min > 2` 证据。
- 这个结论目前仍属于“理论模型 + 数值认证”层面，不应表述成“实验已经验证”。
- 后续当实验室按 `route5` 所需格式测得真实 IQ / coarse-grained 概率数据后，可以直接接入同一条 SDP 认证流程，形成实验版 `route5`。

## 2. Route5 的协议定义

### 2.1 协议的四个组成部分

`route5` 可以严格表述为如下四个模块的组合：

1. `trusted coherent alphabet`
2. `beam splitter + dual-homodyne / IQ measurement`
3. `digital coarse-graining`
4. `single-device prepare-and-measure MDI SDP`

这四部分在当前仓库中的主实现文件是：

- [hybrid_iq.py](../src/python/qrng_routes/route5/hybrid_iq.py)
- [main.py](../src/python/qrng_routes/route5/main.py)
- [refine_queue.py](../src/python/qrng_routes/route5/refine_queue.py)
- [intensity_menu_search.py](../src/python/qrng_routes/route5/intensity_menu_search.py)
- [common.py](../src/python/qrng_routes/common.py)

### 2.2 输入、输出与目标量

在 `route5` 中，可信输入不是单一振幅或单一相位，而是一组相干态字母表：

\[
\mathcal{A}=\{ |\alpha_x\rangle \}
\]

双边联合输入写成

\[
\rho_{xy}=|\alpha_x\rangle\langle \alpha_x| \otimes |\alpha_y\rangle\langle \alpha_y| .
\]

中央测量先产生连续输出，再在数字端离散化，因此最终进入安全证明的是离散输出变量 `c`。实验或理论统计形成条件概率表

\[
P(c|x,y).
\]

随后把这张概率表和可信输入态集合一起送入单设备 prepare-and-measure MDI SDP，得到攻击者最优猜测概率

\[
p_{\mathrm{guess}}
\]

并定义认证最小熵为

\[
H_{\min}=-\log_2 p_{\mathrm{guess}}.
\]

因此，`route5` 的最终目标不是“让连续变量分布本身看起来尽可能平”，而是让经过正式认证后的 `H_min` 尽可能高。

## 3. 理论基础

### 3.1 Route5 为什么是“CV 前端 + 离散输出认证”

`route5` 的核心思想是把“连续变量物理前端”和“离散输出安全认证”明确分层。

物理上，中央测量是连续变量结构：

- 两路输入相干态进入 50:50 分束器；
- 后面用双路 homodyne / IQ 接收读取两个连续量；
- 每轮原始输出本质上是 `(x,p)` 或 `(I,Q)` 这样的连续值。

安全证明上，最终送入 SDP 的却是离散变量：

- 在数字端对 `(x,p)` 做分箱；
- 每个二维分区对应一个离散输出 `c`；
- SDP 处理的是离散结果 `c` 的条件概率 `P(c|x,y)`。

因此，`route5` 不是“重新发明一个全新的连续变量安全证明”，而是：

```text
连续变量前端生成统计
+ 数字 coarse-graining
+ 离散输出的单设备 MDI 认证
```

这也是为什么 `route5` 从原理上避免了“离散安全模型就必须退回低速单光子点击探测”的误解。

### 3.2 Route5 不等于“纯连续变量安全证明”

如果一个方案从头到尾都把连续输出作为安全分析对象，那才更接近“纯连续变量安全证明”。`route5` 不是这样。

`route5` 的真实结构是：

- 统计生成阶段是 CV；
- 安全证明阶段是离散输出的 prepare-and-measure SDP。

因此它的优势是：

- 中央硬件结构仍然可以是高速 CV 接收机；
- 安全模型最终仍然是熟悉、可控、可数值实现的离散输出认证。

这也是当前代码设计最重要的理论出发点。

### 3.3 概率是如何由 `Tr(M_c rho_xy)` 得到的

`route5` 当前并不是直接从手写高斯公式开始算每个分箱概率，而是通过截断 Fock 空间中的测量算符计算：

\[
P(c|x,y)=\mathrm{Tr}(M_c \rho_{xy}) .
\]

这里：

- `rho_xy` 是可信联合输入态；
- `M_c` 是“分束器 + 双正交粗粒化测量 + 支撑投影”后得到的 POVM 元素。

这套概率生成逻辑复用了 route3 的 CV Bell 测量核心实现，关键位置在：

- [cv_four_phase.py:119](../src/python/qrng_routes/route3/cv_four_phase.py#L119)
- [cv_four_phase.py:175](../src/python/qrng_routes/route3/cv_four_phase.py#L175)

当前仓库中已经另外做过一次“解析积分公式 vs `Tr(M_c rho)`”的核对，结论记在：

- [cv_bell_integral_vs_trace_probability_check_cn.md](../docs/cv_bell_integral_vs_trace_probability_check_cn.md)

对 `route5` 而言，这个结论的意义是：

- 现有 `Tr(M_c rho)` 概率引擎确实对应同一类 CV Bell / IQ 前端物理对象；
- 在高一些的 cutoff 下，它会越来越接近理想解析模型；
- 但当前正式 `route5` 工作流仍应把它视为“截断模型上的正式概率引擎”，而不是直接拿理想解析公式替换全部数值过程。

### 3.4 SDP 中的 guessing probability 到底是什么

`route5` 最终调用的是项目里的单设备 prepare-and-measure MDI 猜测概率 SDP。它在代码中的核心类是：

- [common.py:722](../src/python/qrng_routes/common.py#L722)

这个 SDP 的思路是：

- 变量不是一个普通概率，而是一组满足半正定约束的算符 `M_{c,e}`；
- 这些算符必须与观测到的 `P(c|x,y)` 一致；
- 同时它们还必须满足测量完备性和概率归一化；
- 在所有满足约束的攻击策略中，最大化 Eve 对目标输入的正确猜测概率。

代码注释中给出的建模摘要非常清楚：

```python
- 变量：POVM元素M_{c,e}，其中c是真实结果，e是猜测结果
- 约束：
  1. M_{c,e} ≥ 0
  2. Σ_e Tr(M_{c,e} ρ_s) = P(s,c)
  3. Σ_c M_{c,e} = p_e I
  4. Σ_e p_e = 1
- 目标：最大化 Σ_c Tr(M_{c,c} ρ_target)
```

这意味着 `route5` 的 formal `H_min` 并不是“直方图最平就一定高”，而是已经把：

- 输入态一致性
- 中央测量可兼容性
- Eve 的最坏情况猜测

全部一起算进去了。

### 3.5 `raw_H_min` 与 `certified H_min` 的区别

当前 `route5` 结果文件里通常同时会出现两个值：

- `raw_best_H_min`
- `H_min`

它们的含义不同：

- `raw_best_H_min` 只基于单个输入下输出分布的最大概率，等价于
  \[
  -\log_2 \max_c P(c|x,y)
  \]
  它是“看起来有多随机”的指标。
- `H_min` 是通过正式 SDP 得到的认证熵，它是“在所有与观测统计一致的最坏情况下，仍能保证多少随机性”的指标。

以当前自由搜索最好点为例：

- `raw_best_H_min ≈ 2.99218`
- `H_min ≈ 2.11639`

这说明 route5 的真正价值不是“raw 分布很平”，而是“经过 formal 认证以后仍然稳定超过 2 bit”。

### 3.6 为什么 `raw-best target` 和 `formal best target` 可以不一致

这是 `route5` 里一个非常重要、而且完全正常的现象。

原因在于：

- `raw-best target` 只看某个输入自己的局部分布；
- `formal best target` 则要看该输入在整个 SDP 约束系统中的最坏情况表现。

因此一个输入即便 `raw` 看起来很平，也可能因为与其它输入一起放进 SDP 后给 Eve 留下更大自由度，从而 formal 值下降；反过来，一个 raw 不是最平的输入，也可能在正式认证里表现更好。

当前代码已经显式针对这个问题做了处理。核心逻辑在：

- [hybrid_iq.py:716](../src/python/qrng_routes/route5/hybrid_iq.py#L716)

对应的关键片段是：

```python
raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
indices = list(range(len(input_states))) if target_indices is None else list(target_indices)
reusable_problem = SingleDeviceGuessingProblem(input_states, probabilities)
...
for target_input in indices:
    current = reusable_problem.solve(...)
```

这段代码的作用是：

- 先根据 raw 指标挑出若干最有希望的 target；
- 再对这些 target 逐个正式认证；
- 最后选 formal `H_min` 最大的一个作为最终 target。

这一步不是装饰性的优化，而是当前 `route5` 得到 `H_min > 2` 的关键实现细节之一。

## 4. 代码架构总览

### 4.1 `hybrid_iq.py`：核心协议实现

文件：

- [hybrid_iq.py](../src/python/qrng_routes/route5/hybrid_iq.py)

这是 `route5` 的主引擎，负责：

- 构造广义 coherent alphabet；
- 支撑降维与联合输入构造；
- 生成物理受限 IQ 分区候选；
- 单次运行、分区搜索和 alphabet 搜索；
- target 输入认证逻辑。

可以把它理解成“route5 协议本体”。

### 4.2 `main.py`：命令行入口

文件：

- [main.py](../src/python/qrng_routes/route5/main.py)

它的作用不是重新实现算法，而是把 `route5` 的三种模式封装成可执行 CLI：

- `single`
- `partition-search`
- `alphabet-search`

因此它对应的是“用户如何调度 route5”，不是“route5 的理论核心”。

### 4.3 `refine_queue.py`：长时间本地精修队列

文件：

- [refine_queue.py](../src/python/qrng_routes/route5/refine_queue.py)

它负责：

- 在给定半径窗口内批量生成候选；
- 先做 raw scout；
- 再按 raw 排名，对前若干名做正式认证；
- 边跑边写总表和单点结果，适合长时间精修。

当前自由搜索最好结果实际上就是通过这条精修流程收口得到的。

### 4.4 `intensity_menu_search.py`：固定光强菜单搜索

文件：

- [intensity_menu_search.py](../src/python/qrng_routes/route5/intensity_menu_search.py)

它的职责是把实验给定的固定光强集合转成 `route5` 可搜索的半径池，再调用 `search_route5_alphabets(...)` 做受限搜索。

因此，这个脚本就是“理论原型如何贴实验强度约束”的桥梁。

### 4.5 `common.py`：正式认证的底层安全求解器

文件：

- [common.py](../src/python/qrng_routes/common.py)

对 `route5` 而言，最关键的不是整个 `common.py`，而是：

- `SingleDeviceGuessingProblem`
- solver 选择与回退逻辑

也就是说，`common.py` 在 `route5` 里承担的角色是“正式认证后端”，不是前端统计生成器。

### 4.6 核心接口清单

为了方便直接沿着代码核查，`route5` 当前最关键的接口可以整理为：

- [`run_route5(...)` in hybrid_iq.py:1007](../src/python/qrng_routes/route5/hybrid_iq.py#L1007)  
  给定一组 alphabet 和一组固定 IQ 分区，完成一次正式 route5 运行。
- [`search_route5_iq_partitions(...)` in hybrid_iq.py:1159](../src/python/qrng_routes/route5/hybrid_iq.py#L1159)  
  固定 alphabet，在物理受限 IQ 分区家族中做候选枚举、raw 排序和 top-k formal 认证。
- [`search_route5_alphabets(...)` in hybrid_iq.py:1332](../src/python/qrng_routes/route5/hybrid_iq.py#L1332)  
  从半径池和相位池生成系统 alphabet 候选，并对排名靠前者做后续分区认证。
- [`certify_target_inputs(...)` in hybrid_iq.py:716](../src/python/qrng_routes/route5/hybrid_iq.py#L716)  
  对一个或多个 target 输入做 formal SDP，并选出 formal 最优 target。
- [`intensity_menu_to_radii(...)` in intensity_menu_search.py:97](../src/python/qrng_routes/route5/intensity_menu_search.py#L97)  
  把实验给定的固定光强菜单映射成 route5 的半径池。
- [`SingleDeviceGuessingProblem.solve(...)` in common.py:806](../src/python/qrng_routes/common.py#L806)  
  在给定目标输入下求解正式 guessing-probability SDP，并返回 `p_guess` 与 `H_min`。

## 5. 核心运行流程

下面按真实代码执行顺序解释 `route5` 的核心 pipeline。

### 5.1 构造 coherent alphabet

对应函数：

- [hybrid_iq.py:211](../src/python/qrng_routes/route5/hybrid_iq.py#L211)

核心代码片段：

```python
if alpha_values is None:
    radii = DEFAULT_RADIUS_VALUES if radius_values is None else list(radius_values)
    phases = DEFAULT_PHASE_VALUES if phase_values is None else list(phase_values)
    alpha_values = [radius * np.exp(1j * phase) for radius in radii for phase in phases]

unique_alphas = _deduplicate_alphas([complex(alpha) for alpha in alpha_values])
states = [density_from_ket(coherent_state(cutoff, alpha)) for alpha in unique_alphas]
```

这一段负责什么：

- 把用户给定的 `radius_values x phase_values`，或直接给定的 `alpha_values`，转换成一组本地可信相干态。

它对应的物理对象是什么：

- `alpha = r e^{i\phi}` 是单模相干态的复振幅；
- `r` 决定振幅大小；
- `phi` 决定相位；
- 多个 `(r,\phi)` 共同构成 trusted alphabet。

它影响哪些结果：

- `num_local_states`
- `local_alphas`
- 后续所有联合输入数 `num_inputs = num_local_states^2`
- 最终的 `local_rank`、`local_operator_span_rank` 和 `H_min`

### 5.2 支撑降维与联合输入构造

对应函数：

- [hybrid_iq.py:500](../src/python/qrng_routes/route5/hybrid_iq.py#L500)

核心代码片段：

```python
local_basis = support_basis(local_kets)
reduced_local_states = [project_density_to_basis(rho, local_basis) for rho in local_states]

for x, rho_a in enumerate(reduced_local_states):
    for y, rho_b in enumerate(reduced_local_states):
        joint_states.append(kron(rho_a, rho_b))
        labels.append((x, y))
```

这一段负责什么：

- 先求 trusted alphabet 真正张成的局域支撑子空间；
- 再把所有局域态投影到这个更小的支撑里；
- 最后构造全部联合输入态 `rho_xy`。

它对应的物理/数学意义是什么：

- 这一步不是改协议，而是把“数学上冗余的高维表示”压缩掉；
- 保留的是 trusted states 真正可区分的支撑方向；
- 这样既减少 SDP 维度，也让 `local_rank` 和 `local_operator_span_rank` 成为有意义的结构诊断量。

它影响哪些结果：

- `local_rank`
- `joint_dim`
- `local_operator_span_rank`
- `operator_span_rank`

以当前自由最优点 `[0.0, 0.85, 1.25]` 为例：

- `num_local_states = 17`
- `local_rank = 4`
- `local_operator_span_rank = 13`
- `local_operator_space_dim = 16`
- 因此 `local_span_ratio = 13 / 16 = 0.8125`

这说明当前最佳 alphabet 在 `cutoff = 4` 的有效局域支撑里，已经覆盖了大部分局域算符空间。

### 5.3 生成物理受限的 IQ 分区边界

对应函数：

- [hybrid_iq.py:570](../src/python/qrng_routes/route5/hybrid_iq.py#L570)

核心代码片段：

```python
if num_bins == 2:
    return np.array([-np.inf, 0.0, np.inf], dtype=float)

normalized = np.linspace(-1.0, 1.0, num_bins + 1, dtype=float)
edges = np.sign(normalized) * (np.abs(normalized) ** gamma) * finite_range
edges[0] = -np.inf
edges[-1] = np.inf
```

这一段负责什么：

- 生成 `x` 轴或 `p` 轴的对称轴对齐分箱边界。

它对应的物理含义是什么：

- `finite_range` 决定中间有限区域的总宽度；
- `gamma` 决定边界是更均匀还是更偏向中心/边缘；
- 两端尾部始终并入 `±∞` 区间。

它影响哪些结果：

- `x_bounds`
- `p_bounds`
- `num_x_bins`
- `num_p_bins`
- `quadrature_range`
- `boundary_gamma`

例如当前最好点中：

- `num_x_bins = 6`
- `num_p_bins = 2`
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`

于是边界变成：

- `x_bounds = [-∞, -1.2, -0.6, 0, 0.6, 1.2, +∞]`
- `p_bounds = [-∞, 0, +∞]`

这对应“在 `x` 方向细分，在 `p` 方向只做正负两半”的 12 输出结构。

### 5.4 枚举 IQ 分区候选并做 raw 粗筛

对应函数：

- [hybrid_iq.py:913](../src/python/qrng_routes/route5/hybrid_iq.py#L913)

核心代码片段：

```python
for candidate_index, (x_candidate, p_candidate) in enumerate(...):
    probabilities, _, _, _ = dual_homodyne_probabilities(...)
    summary = _candidate_summary(...)
    if store_probabilities:
        summary["probabilities"] = probabilities
    raw_candidates.append(summary)

ranked_candidates = sorted(raw_candidates, key=lambda item: item["raw_best_H_min"], reverse=True)
```

这一段负责什么：

- 枚举很多种 `(x,p)` 分区候选；
- 对每个候选计算整张概率表 `P(c|x,y)`；
- 先根据 raw 指标对候选排序。

它对应的物理/数学对象是什么：

- 候选不是“任意 POVM”，而是被限制在 axis-aligned 的 IQ 分区家族里；
- 这是当前 route5 很重要的物理受限前提；
- raw 粗筛不是最终认证，只是为了把算力集中到最有希望的候选上。

它影响哪些结果：

- `raw_partition_ranking`
- 后续送进正式 SDP 的候选范围

### 5.5 正式运行 `run_route5(...)`

对应函数：

- [hybrid_iq.py:1007](../src/python/qrng_routes/route5/hybrid_iq.py#L1007)

核心代码片段：

```python
probabilities, output_labels, x_bounds_out, p_bounds_out = dual_homodyne_probabilities(...)

raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
candidate_order = list(np.argsort(-raw_h))

best, target_scan = certify_target_inputs(
    joint_states,
    probabilities,
    labels,
    local_alphas,
    target_indices=candidate_order,
    ...
)
```

这一段负责什么：

- 对一组给定 alphabet 和一组给定边界，完成整次 route5 正式运行。

它对应的数学含义是什么：

1. 先固定可信输入态和物理受限 IQ 分区；
2. 生成全部 `P(c|x,y)`；
3. 先做 raw 排序；
4. 再对若干候选 target 做 formal SDP；
5. 返回完整结果字典。

它影响哪些结果：

- `raw_best_H_min`
- `H_min`
- `p_guess`
- `certified_best_target`
- `target_scan`
- `x_bounds`
- `p_bounds`

### 5.6 正式认证 `certify_target_inputs(...)`

对应函数：

- [hybrid_iq.py:716](../src/python/qrng_routes/route5/hybrid_iq.py#L716)

这一段已经在第 3 节讲过原理，这里强调它在 pipeline 中的作用：

- 它是把“候选统计”变成“正式认证结果”的关键桥梁；
- 它通过复用同一个 `SingleDeviceGuessingProblem`，减少多 target 扫描时重复建模的开销。

因此，如果没有这一步的多 target formal 扫描，当前 `route5` 的最好结果未必能被稳定找出来。

### 5.7 底层正式求解器 `SingleDeviceGuessingProblem`

对应类：

- [common.py:722](../src/python/qrng_routes/common.py#L722)

核心代码片段：

```python
constraints.append(
    cp.sum([cp.real(cp.trace(operators[(c, e)] @ rho_s)) for e in range(self.num_outputs)])
    == probabilities[s, c]
)
...
objective = cp.Maximize(
    cp.sum([cp.real(cp.trace(operators[(c, c)] @ rho_star)) for c in range(self.num_outputs)])
)
```

这一段负责什么：

- 它把 route5 的可信输入态集合和整张概率表翻译成一个正式的 SDP。

它对应的物理/数学意义是什么：

- 任何可行攻击模型都必须与全部观测概率一致；
- 在这些模型里再最大化 Eve 对目标输入的猜测成功率；
- 这就是 formal `p_guess` 的来源。

它影响哪些结果：

- `p_guess`
- `H_min`
- `solver`
- `status`

### 5.8 固定光强映射 `intensity_menu_to_radii(...)`

对应函数：

- [intensity_menu_search.py:97](../src/python/qrng_routes/route5/intensity_menu_search.py#L97)

核心代码片段：

```python
max_intensity = max(positive) if len(positive) > 0 else 1.0
...
radius = float(max_radius) * math.sqrt(float(intensity) / max_intensity)
```

这一段负责什么：

- 把实验给定的固定光强菜单转成 route5 使用的半径池。

它对应的物理意义是什么：

- 在当前原型中，采取的是“保持光强比例”的归一化映射；
- 因为相干态满足 `|alpha|^2` 与平均光子数/光强成正比；
- 所以半径应按强度的平方根缩放。

它影响哪些结果：

- 固定光强搜索时的 `radius_pool`
- `[0,80,160]` 这条主线下的最终结果

### 5.9 本地精修队列 `refine_queue`

对应主流程：

- [refine_queue.py:177](../src/python/qrng_routes/route5/refine_queue.py#L177)

核心代码片段：

```python
for radii in radius_candidates:
    scout = search_route5_iq_partitions(... certify_top_k=0 ...)
...
ranked_scouts = sorted(scout_by_key.values(), key=lambda item: item["raw_best_H_min"], reverse=True)
...
result = run_route5(...)
```

这一段负责什么：

- 先做大批量 raw scout；
- 再对 raw 排名前列的候选做正式认证；
- 每完成一个候选就把结果写盘。

它对应的工程意义是什么：

- 把耗时最大的 formal SDP 只留给最有希望的少数候选；
- 使得 route5 可以在普通工作站上做长时间精修。

它影响哪些结果：

- 当前自由搜索 top-8 `MOSEK` 结果表；
- 自由最优点 `[0.0, 0.85, 1.25]`

## 6. 参数搜索逻辑

### 6.1 `single` 模式

`single` 模式对应“固定 alphabet + 固定分区”的单次正式运行。

它适合：

- 复核某个具体参数点；
- 比较不同 solver；
- 看 target scan 细节；
- 做单点严格重算。

### 6.2 `partition-search` 模式

`partition-search` 的目的是：

- 固定 alphabet；
- 系统搜索一组 `x/p` 分区候选；
- 先按 raw 指标排序；
- 再对 top-k 候选做 formal SDP。

因此它主要回答的问题是：

```text
在 trusted alphabet 已经给定时，
哪一种物理受限 IQ 分区最有希望把认证值推高？
```

### 6.3 `alphabet-search` 模式

`alphabet-search` 是更上层的搜索：

- 先从半径池和相位池系统地产生 alphabet 候选；
- 对每个 alphabet 做 partition 粗筛；
- 按结构指标与 raw 指标排序；
- 再对前若干个 alphabet 做正式后续搜索。

因此它主要回答的问题是：

```text
在允许的 coherent alphabet 家族里，
哪种输入字母表最值得继续精修？
```

### 6.4 为什么 `refine_queue` 必须存在

从计算上看，`route5` 的主要成本来自两个地方：

- 输入数一旦增大，`num_inputs = num_local_states^2` 很快上升；
- 输出数一旦增大，SDP 变量块数也会迅速增加。

因此，如果每个候选都直接做正式 SDP，代价会很高。`refine_queue` 的设计就是为了解决这个问题：

- 用 `search_route5_iq_partitions(... certify_top_k=0)` 做 raw scout；
- 只把最有希望的少数候选送进 `run_route5(...)` 做正式认证；
- 过程可中断、可恢复、可逐步落盘。

这也是当前最优自由点能够被稳定找到的工程前提。

### 6.5 固定光强搜索脚本为什么重要

`intensity_menu_search.py` 的意义在于，它不是简单改个参数，而是把：

- 理论上的半径/相位字母表

和

- 实验上真实给定的光强菜单

连接了起来。

它回答的问题是：

```text
如果实验室不能任意调半径，只能从固定光强集合里选，
route5 还能不能继续工作？
```

当前 `[0,80,160]` 的主线结果说明，答案是：能，而且 formal `H_min` 仍高于 2。

## 7. 结果分析

### 7.1 先说结果文件的证据链

当前 route5 结果最值得信任的证据链是：

1. `alphabet-search` 找出高潜力 alphabet 家族；
2. `partition-search` 把候选集中到少数高分分区；
3. `refine_queue + MOSEK` 对高潜力半径窗口做正式收口；
4. 固定光强脚本验证贴实验约束的主线仍然成立。

与此对应的关键结果文件是：

- [route5_local_refine_queue_mosek_v1.json](../output/qrng_routes/route5_local_refine_queue_mosek_v1.json)
- [route5_fixed_intensity_080160_scale120.json](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)
- [route5_single_12out_candidate_newbest_r1012_fastscs.json](../output/qrng_routes/route5_single_12out_candidate_newbest_r1012_fastscs.json)
- [route5_single_12out_r1012_nodes20_strict.json](../output/qrng_routes/route5_single_12out_r1012_nodes20_strict.json)

### 7.2 自由搜索最优点

当前自由搜索下的正式 `MOSEK` 最优点为：

- `radii = [0.0, 0.85, 1.25]`
- `8` 个等间隔相位
- `num_local_states = 17`
- `num_inputs = 289`
- `num_x_bins = 6`
- `num_p_bins = 2`
- `num_outputs = 12`
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`
- `num_quadrature_nodes = 12`
- `solver = MOSEK`
- `status = optimal`
- `target_input = [15, 7]`
- `p_guess ≈ 0.230623`
- `raw_best_H_min ≈ 2.99218`
- `H_min ≈ 2.11639`

结果文件：

- [r0.0000_0.8500_1.2500.json](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)

### 7.3 当前 top-8 正式结果

`route5_local_refine_queue_mosek_v1.json` 里当前共有 `29` 个 scout 候选、`8` 个已完成正式认证的候选。按 formal `H_min` 排序，前 8 个结果为：

| 排名 | radii | 输出结构 | range | gamma | solver | target_input | raw_best_H_min | H_min |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 1 | `[0.0, 0.85, 1.25]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[15, 7]` | `2.99218` | `2.11639` |
| 2 | `[0.0, 0.9, 1.15]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[11, 3]` | 见总表 | `2.11030` |
| 3 | `[0.0, 0.85, 1.2]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[11, 3]` | `2.98634` | `2.10055` |
| 4 | `[0.0, 0.85, 1.1]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[11, 11]` | 见总表 | `2.08283` |
| 5 | `[0.0, 0.9, 1.05]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[15, 15]` | 见总表 | `2.08058` |
| 6 | `[0.0, 0.9, 1.1]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[15, 15]` | 见总表 | `2.07990` |
| 7 | `[0.0, 0.85, 1.05]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[15, 15]` | 见总表 | `2.07243` |
| 8 | `[0.0, 0.85, 1.15]` | `6 x 2` | `1.8` | `1.0` | `MOSEK` | `[15, 7]` | 见总表 | `2.06937` |

这个结果表有两个明显特征：

- 高分点几乎全部集中在“三层半径、8 相位、`6 x 2` 分区、`range ≈ 1.8`、`gamma = 1.0`”附近；
- `H_min > 2` 不是单点偶然，而是一整个局部窗口都能稳定维持的现象。

### 7.4 为什么 `12` 输出优于 `16` 输出不一定是 bug

当前搜索里，“`12` 输出优于 `16` 输出”并不构成代码 bug 的证据。一个直接的例子是：

- `12` 输出最好自由点：`H_min ≈ 2.11639`
- 当前已有的 `16` 输出例子  
  [route5_single_16out_candidate17_fastscs.json](../output/qrng_routes/route5_single_16out_candidate17_fastscs.json)  
  给出 `H_min ≈ 0.89561`

这背后至少有三层原因。

第一，输出数更多并不保证 formal 认证更高。  
输出数增加时，raw 分布可能更平，但 SDP 中 Eve 的猜测结构也一起变大，formal 熵并不是单调上升量。

第二，当前中央测量被限制为 axis-aligned 的 IQ 分区，而不是任意自由 POVM。  
在这种物理受限家族里，输出类别更多可能只是把统计切得更碎，并不等于引入了更有效的可认证区分信息。

第三，当前高分窗口似乎把主要有效信息集中在一条轴上。  
这也是为什么 `6 x 2` 往往比 `4 x 4` 或其它更均匀的切法表现更好。

因此，当前更合理的说法是：

```text
在已扫描的 route5 物理受限 IQ 分区家族中，
12 输出的 6 x 2 结构比当前尝试过的 16 输出候选更适合 formal 认证。
```

而不是：

```text
16 输出比 12 输出差，说明代码一定错了。
```

### 7.5 为什么 `6 x 2` 比更平均的分法更好

从当前高分结果的集中趋势看，`6 x 2` 的优势并不是偶然。

其直观解释是：

- 有效区分信息并没有在 `x` 和 `p` 两轴上平均分布；
- 在更有信息的一轴上细分；
- 在另一轴上只做粗分；
- 反而更利于 formal 认证。

这和当前最好点的边界也一致：

- `x_bounds = [-∞, -1.2, -0.6, 0, 0.6, 1.2, +∞]`
- `p_bounds = [-∞, 0, +∞]`

这是一种非常明确的不对称设计，而不是对称地把资源平均分给两条轴。

### 7.6 为什么 `SCS` 快速值偏乐观，而 `MOSEK` 更适合做正式结论

`route5` 的搜索过程里，`SCS` 和 `MOSEK` 承担的角色不同。

`SCS` 更适合：

- 快速扫方向；
- 发现高潜力窗口；
- 做早期粗筛。

`MOSEK` 更适合：

- 对候选高分点做正式收口；
- 判断 `status = optimal` 的结果；
- 作为最终汇报值。

一个典型例子是：

- [route5_single_12out_candidate_newbest_r1012_fastscs.json](../output/qrng_routes/route5_single_12out_candidate_newbest_r1012_fastscs.json)  
  给出 `H_min ≈ 1.81877`
- 同一类点更严格重跑的  
  [route5_single_12out_r1012_nodes20_strict.json](../output/qrng_routes/route5_single_12out_r1012_nodes20_strict.json)  
  则回落到 `H_min ≈ 1.56357`

这说明：

- 快速 `SCS` 值适合做方向判断；
- 但不应直接拿来做最终结论。

当前文档中的正式结论，因此统一优先采用 `MOSEK` 结果。

## 8. 固定光强优先的实验主线

### 8.1 为什么这里要以固定光强为主线

从实验沟通角度，最重要的问题不是“完全自由搜索能做到多高”，而是：

```text
如果实验室的光强只能从有限菜单里选，
route5 还能不能工作，formal H_min 还能不能过 2？
```

这就是本文把固定光强 `[0,80,160]` 放在实验主线位置的原因。

### 8.2 强度到半径的映射

当前代码里，固定光强不是直接拿来代替 `alpha`，而是通过 `intensity_menu_to_radii(...)` 先映射为 route5 内部使用的半径。

采用的规则是：

\[
\text{radius}=\text{max\_radius}\cdot\sqrt{\frac{\text{intensity}}{\max(\text{intensity menu})}} .
\]

这条规则对应的代码在：

- [intensity_menu_search.py:97](../src/python/qrng_routes/route5/intensity_menu_search.py#L97)

它的物理依据是：

- 相干态满足 `|alpha|^2` 与平均光子数/光强成正比；
- 因此如果要保持强度比例关系，半径应按强度的平方根缩放。

### 8.3 `[0,80,160]` 对应的 route5 主线结果

在当前固定光强主线中，选择：

- `intensity_values = [0,80,160]`
- `max_radius = 1.2`

则映射得到：

- `radii = [0.0, 0.848528137423857, 1.2]`

在这个受限 alphabet 上，当前正式结果为：

- `num_x_bins = 6`
- `num_p_bins = 2`
- `num_outputs = 12`
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`
- `solver = MOSEK`
- `status = optimal`
- `target_input = [15, 7]`
- `raw_best_H_min ≈ 2.98296`
- `H_min ≈ 2.10102`

对应结果文件：

- [route5_fixed_intensity_080160_scale120.json](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)

### 8.4 它与自由搜索近邻点的一致性

这个固定光强点并不是孤立高点，它与自由搜索近邻点：

- `radii = [0.0, 0.85, 1.2]`
- `H_min ≈ 2.10055`

几乎一致。对应结果文件为：

- [r0.0000_0.8500_1.2000.json](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2000.json)

二者的比较是：

- 固定光强映射点：`H_min ≈ 2.10102`
- 自由搜索近邻点：`H_min ≈ 2.10055`

这说明：

- `[0,80,160]` 不是一个“勉强维持不崩”的受限点；
- 它本身就落在当前 route5 的高分稳定窗口附近。

### 8.5 这条结果为什么重要

这条结果真正重要的地方在于：

```text
当把实验限制显式加入之后，
route5 的 formal H_min 仍然没有掉回 2 以下。
```

因此更准确的结论是：

- 自由搜索最好点说明 route5 在理论原型上确实能过 2；
- 固定光强 `[0,80,160]` 主线说明，这个结论在更贴实验的约束下仍保留。

## 9. 实验准备与真实概率数据采集

### 9.1 当前阶段与下一阶段要明确区分

当前文档给出的结果属于第一阶段：

- 用理论输入态模型生成可信 alphabet；
- 用理论中央测量模型生成 `P(c|x,y)`；
- 再把这些统计送入正式 SDP。

这已经足以回答：

```text
route5 作为理论原型是否可行？
在固定光强约束下是否仍有机会实现 H_min > 2？
```

但它还没有回答：

```text
实验室真实数据代入之后，最终认证结果是多少？
```

这就是第二阶段。根据你刚补充的信息，后续会通知实验室去测量 `route5` 所需的真实实验概率数据。对 route5 来说，这是非常关键、而且逻辑上自然的一步：

- 第一阶段先用理论和数值把路线跑通；
- 第二阶段再把实验室实测的 route5 概率数据接入同一认证流程。

### 9.2 实验室后续需要测什么类型的数据

`route5` 真正需要的不是一张一维点击概率表，而是和它自身物理流程匹配的数据。最理想的数据形态是：

1. 每个测试输入对 `(x,y)` 的 IQ 原始样本序列。
2. 如果原始序列不方便保存，则至少要有每个 `(x,y)` 下的二维直方图或 coarse-grained 概率。
3. 更进一步，如果分箱已经固定，则至少应有实验版
   \[
   P_{\mathrm{exp}}(c|x,y).
   \]

也就是说，route5 最终需要的是“它自己的 IQ / coarse-grained 概率数据”，而不是任意路线共用的一张概率表。

### 9.3 Route5 需要的实验配套信息

如果实验室要真正把 route5 跑成实验认证，还至少需要以下几类配套信息。

#### 1. 输入态标定信息

必须知道每个 trusted input 实际对应的：

- 振幅
- 相位
- 或复振幅 `alpha`

否则 trusted alphabet 的前提无法严格成立。

#### 2. 中央测量与 ADC 轮次定义

必须明确：

- 每一轮样本如何从连续波形中截取；
- `x,p` 或 `I,Q` 是如何归一化的；
- 是否做了滤波、去直流、重采样；
- 分箱边界最终放在哪里。

这些处理都会直接改变 `P(c|x,y)`。

#### 3. 有限尺寸计数数据

当前 route5 原型使用的是理论概率，而实验版认证还需要：

- 每个输入输出组合的出现次数；
- 测试轮样本数；
- 参数估计误差范围。

否则只能做理想化概率认证，不能做有限尺寸实验认证。

### 9.4 为什么 `Probability.mat` 不是 route5 的直接数据格式

这里只简要说明边界，不展开历史路线讨论。

对 route5 来说，当前最关键的是：

- 它需要和 IQ / dual-homodyne 结构相匹配的数据；
- 需要保留输入标签 `(x,y)` 与离散输出 `c` 之间的对应关系；
- 最好还能追踪到分箱前的连续 IQ 样本。

因此，`Probability.mat` 不能直接作为 route5 的正式数据接口。更准确的表述应当是：

```text
route5 需要自己的 IQ / coarse-grained 概率数据。
```

### 9.5 实验闭环应该如何表述

最准确的一句话可以写成：

```text
当前文档给出的是 route5 的理论与数值设计图；后续当实验室测得相应概率数据后，
可直接替换理论概率表并进入同一 SDP 认证流程。
```

这句话很重要，因为它同时说明了：

- 当前成果不是实验数据结果；
- 但现有 route5 代码并不是“只能做理论玩具”，而是已经具备实验接入接口逻辑。

## 10. 风险、边界与可能质疑的问题

### 10.1 为什么当前结果仍属于理论原型，而不是实验结论

因为当前 route5 的 `H_min > 2` 主要来自：

- 理论态模型
- 理论中央测量模型
- 理论生成的 `P(c|x,y)`
- 然后再做 formal SDP

所以它已经足以说明“路线本身可行”，但还不能表述成“实验已经实现”。

更准确的口径应是：

```text
route5 已经完成理论原型层面的 formal 可行性验证；
实验版结果仍需等待 route5 专用真实概率数据接入。
```

### 10.2 `cutoff` 与 `num_quadrature_nodes` 的近似风险

当前最好点使用的是：

- `cutoff = 4`
- `num_quadrature_nodes = 12`

这已经足以给出稳定的高分窗口，但仍不是无限维极限，也不是严格的收敛证明。风险主要有两类：

1. 低 cutoff 可能使输入态和测量在截断表示中仍有近似误差。
2. 节点数有限意味着 quadrature POVM 仍是数值求积近似。

目前可以较稳妥地说：

- 当前值足以支持“route5 能过 2”的判断；
- 但若后续要做更严格论文化或实验闭环，需要继续做更高精度收敛复核。

### 10.3 输入数多带来的统计与标定压力

当前最佳 alphabet 是：

- 1 个真空点
- 2 个非零半径层
- 每层 8 个相位

因此本地共有 `17` 个状态，联合输入共有 `289` 个组合。

这带来两个直接后果：

1. 参数估计和标定工作量不小。
2. 有限尺寸实验认证时，测试轮开销不会很轻。

所以 route5 的优势不是“实验工作量最小”，而是“在较强输入字母表下，换来了 formal `H_min > 2` 的可能性”。

### 10.4 高速 CV 前端不等于自动得到最终高认证比特率

`route5` 解决的是：

```text
离散安全模型不必等于低速单光子点击探测。
```

但它没有自动解决：

- ADC 带宽与抖动
- 轮次定义
- 相邻样本相关性
- 测试轮占比
- 有限尺寸开销
- 提取器损耗

因此更准确的表述是：

```text
route5 使高速成为可能，但并不等于最终认证输出速率已经自动确定。
```

### 10.5 实验数据接入后下一步应检查什么

当实验室后续测得 route5 所需真实概率数据后，建议按以下顺序检查：

1. 输入态标定是否与假定 alphabet 一致。
2. 实验版 `P_exp(c|x,y)` 是否与理论高分窗口同量级。
3. 把实验概率替换理论概率后，formal `H_min` 是否仍维持在可接受水平。
4. 做有限尺寸与轮次相关性分析。
5. 再把认证熵换算为可实现的平均认证输出速率。

也就是说，实验数据接入不是“文档之外的新工作”，而是本文给出的 route5 设计图的自然下一步。

## 11. 附录

### 11.1 关键结果文件索引

- 自由搜索正式最优点：  
  [r0.0000_0.8500_1.2500.json](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)
- 固定光强 `[0,80,160]` 主线结果：  
  [route5_fixed_intensity_080160_scale120.json](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)
- 早期快速 `SCS` 高分点：  
  [route5_single_12out_candidate_newbest_r1012_fastscs.json](../output/qrng_routes/route5_single_12out_candidate_newbest_r1012_fastscs.json)
- 更严格重跑点：  
  [route5_single_12out_r1012_nodes20_strict.json](../output/qrng_routes/route5_single_12out_r1012_nodes20_strict.json)
- 自由窗口精修总表：  
  [route5_local_refine_queue_mosek_v1.json](../output/qrng_routes/route5_local_refine_queue_mosek_v1.json)

### 11.2 关键字段释义

- `num_local_states`  
  本地 trusted alphabet 的状态数。
- `num_inputs`  
  联合输入总数，通常等于 `num_local_states^2`。
- `local_rank`  
  局域支撑维数。
- `local_operator_span_rank`  
  局域输入态张成的算符空间秩。
- `operator_span_rank`  
  联合输入态张成的算符空间秩。
- `raw_best_H_min`  
  只看观测统计时最平输入的 raw 指标。
- `H_min`  
  通过正式 SDP 得到的认证熵。
- `target_input`  
  formal 最优目标输入。
- `raw_best_target`  
  raw 指标最优输入。
- `target_scan`  
  多个候选 target 的正式认证扫描记录。

### 11.3 常用术语表

- `coherent alphabet`  
  一组可信相干态输入集合。
- `radius`  
  复振幅 `alpha` 的模长，即 `|alpha|`。
- `phase`  
  复振幅 `alpha` 的相位角。
- `IQ / dual-homodyne`  
  同时读取两个正交分量的连续变量测量结构。
- `coarse-graining`  
  把连续输出按预设边界切成有限离散输出。
- `raw entropy`  
  不含最坏情况安全优化的表观熵指标。
- `certified entropy`  
  经过正式安全认证后仍能保证的最小熵。

## 12. 最终结论

如果把 route5 单独拿出来看，当前最合理、也最严谨的结论是：

1. `route5` 已经在理论原型层面给出清晰的 `H_min > 2` 证据。
2. 这个结论不仅存在于自由搜索最好点，也存在于固定光强 `[0,80,160]` 的更贴实验主线中。
3. `route5` 当前最大的价值，不是“已经完成实验”，而是“已经把一条可 formal 认证、且保持高速 CV 前端的路线跑通了”。
4. 后续当通知实验室采集 route5 所需真实概率数据后，这些数据可以直接接入现有 route5 流程，进入同一 SDP 认证框架。

因此，就当前阶段而言，`route5` 已经不是一个模糊的想法，而是一条结构、代码、结果和实验下一步都已经相对清楚的独立路线。
