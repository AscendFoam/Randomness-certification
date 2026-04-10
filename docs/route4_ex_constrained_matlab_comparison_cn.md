# `guessprobprimal_phaseinsensitive_original.m` 与 `guessprobprimal_route4_ex_constrained.m` 逐段对照

## 1. 文档目的

本文档用于把原始 Matlab 脚本

- `src/matlab/guessprobprimal_phaseinsensitive_original.m`

和新的 constrained Matlab 脚本

- `src/matlab/guessprobprimal_route4_ex_constrained.m`

做一个逐段对应说明，帮助快速判断：

1. 哪些部分是沿用原始 route4 的；
2. 哪些部分是 route4-ex-constrained 相对原始 route4 的关键修改；
3. 为什么新脚本仍然可以看作“贴近原 route4 结构的扩展版”，而不是完全另起炉灶。

---

## 2. 一句话总结

两份脚本的**总体骨架是相同的**：

1. 设定 `selected_mu_list / q_selected / M`
2. 读取 `Probability.mat`
3. 做输出 coarse-graining
4. 构造 `LambdaIndices`
5. 写 CVX primal
6. 输出 `p_guess` 和 `H_min`

真正的核心差异只有三条：

1. 原脚本的 trusted input 只在 SDP 中使用 **Fock 对角部分**；
2. constrained 脚本把 trusted input 改成了 **完整截断相干态**；
3. 原脚本的主问题是 **diagonal primal**，constrained 脚本的主结果改成 **full primal**。

---

## 3. 分段对应关系

### 3.1 文件开头说明

原脚本：

- 开头主要说明 CVX、`Probability.mat`、`selected_mu_list`、`q_selected`、`M` 的基本使用前提。

新脚本：

- 保留了相同的使用前提；
- 额外明确写出 constrained 主线的默认参数：
  - `selected_mu_list = [100, 120, 140]`
  - `q_selected = [1, 0, 0]`
  - `custom_edges = [0, 121, 132, 256]`
  - `alpha_values = [0.54, 0.66 i, -0.72]`
  - `M = 6`

结论：

- 这一段只是新增了“当前默认主线”的说明，没有改主流程思想。

### 3.2 第 1 段：参数配置

原脚本：

- 配置 `selected_mu_list`、`q_selected`、`M`、`full_mu`。
- 其中输入态默认来自 `alpha = sqrt(mu)`。
- 输出粗粒化使用 `N` 等分 256 个原始 bin。

新脚本：

- 仍然保留 `selected_mu_list`、`q_selected`、`M`、`full_mu`、`shift`。
- 但把输出区间从“等分 `N` 块”改为“固定自定义边界 `custom_edges`”。
- 把输入态从“由 `sqrt(mu)` 自动生成”改为“直接固定一组 `alpha_values`”。

结论：

- 原脚本是“固定输入标签 + 等分 coarse-graining + 对角输入近似”；
- 新脚本是“固定输入标签 + 固定 coarse-graining 边界 + 固定 non-diagonal coherent alphabet”。

### 3.3 第 2 段：输入检查与初始化

原脚本：

- 检查 `selected_mu_list` 是否属于 `full_mu`；
- 检查 `q_selected` 长度是否匹配；
- 初始化 `N`、`p`、`rho`、`shift`。

新脚本：

- 做了同样的检查；
- 额外检查：
  - `alpha_values` 长度是否和输入数一致；
  - `custom_edges` 是否从 `0` 到 `256` 且严格递增；
  - `q_selected` 是否可归一化。
- 初始化 `p_raw`、`p`、`rho`、`rho_diag`、`selected_full_indices`。

结论：

- 这一段的角色相同，都是“为主计算准备统一输入”；
- 新脚本只是把更多与 constrained 主线相关的参数显式检查出来。

### 3.4 第 3 段：构造输入态

原脚本：

- 通过
  - `alpha = sqrt(selected_mu_list(i))`
  - `coeff(n+1) = exp(-|alpha|^2/2) * alpha^n / sqrt(n!)`
  构造相干态；
- 随后只提取 `rho_diag`，在后续 SDP 中只使用对角部分。

新脚本：

- 不再把 `alpha` 绑死为 `sqrt(mu)`；
- 而是直接使用固定的 `alpha_values = [0.54, 0.66 i, -0.72]`；
- 仍然先构造完整 `rho = |alpha><alpha|`；
- 但这一次 **完整的 `rho` 会被 full primal 直接使用**；
- `rho_diag` 只保留给 diagonal primal 对照问题。

结论：

- 这是两份脚本在物理模型上的第一处关键差异；
- 原脚本“构造了完整相干态，但真正送进 SDP 的只有对角部分”；
- 新脚本“真正把完整 non-diagonal trusted input 送进了主问题”。

### 3.5 第 4 段：读取 `Probability.mat` 并做 coarse-graining

原脚本：

- 读取 `Probability.mat`；
- 按 `N = 8` 等宽分块；
- 每块把连续 `256/N` 个原始 bin 累加，得到 `p(x,y)`。

新脚本：

- 同样读取 `Probability.mat`；
- 但 coarse-graining 不再按等宽分块，而是使用固定边界
  - `[0, 121, 132, 256]`
- 因此 3 个输出区间宽度分别是
  - `121`
  - `11`
  - `124`
- 之后还多做了一步 `prob_floor` 正则化。

结论：

- 这一段的数据来源保持不变；
- 真正变化的是“如何从 256 个原始输出合并成最终离散输出”。

### 3.6 第 5 段：生成 `LambdaIndices`

原脚本：

- 用 `ndgrid` 枚举全部 `N^(D+1)` 个策略；
- 得到 `LambdaIndices`。

新脚本：

- 完全保留这套结构；
- 仍然使用 `LambdaIndices` 作为 primal 里的策略索引。

结论：

- 这一段在数学结构上基本不变。

### 3.7 第 6 段：原脚本的主 CVX 问题 vs 新脚本的 diagonal primal

原脚本：

- 只有一个主问题；
- 变量是
  - `M_elements(M, N, num_strategies) nonnegative`
- 本质上是 **Fock 对角测量元** 的 primal SDP。

新脚本：

- 先保留了一份结构相同的 diagonal primal，变量是
  - `M_diag(M, N, num_strategies) nonnegative`
- 它的作用不是主结果，而是作为对照。

结论：

- 新脚本的第 6 段可以视为“把原脚本的主问题保留下来，作为 baseline/对照问题”。

### 3.8 第 7 段：新脚本新增的 full primal

这是 constrained 脚本最关键的新部分，原脚本没有这一段。

新脚本新增：

- 变量
  - `M_full(M, M, num_operator_variables) hermitian semidefinite`
  - `s_lambda(num_strategies) nonnegative`
- 归一化约束
  - `sum_y M_{y,lambda} = s_lambda I`
- 统计约束
  - `Tr(rho_x M_y) = p(x,y)`
- 目标函数
  - 使用完整 `rho_x` 计算 `Tr(rho_x M_sum_for_y)`。

这意味着：

1. 测量元不再被限制为对角；
2. trusted input 的非对角元真正进入了优化问题；
3. 当前正式的 `H_min` 就是从这一段得到的。

结论：

- 这一段是原 route4 到 route4-ex-constrained 的核心升级。

### 3.9 第 8 段：结果输出

原脚本：

- 只输出一个主问题的 `p_guess / H_min`。

新脚本：

- 同时输出：
  - `diagonal_result`
  - `full_result`
- 并把关键配置统一打包到 `result` 结构体中。

结论：

- 新脚本更适合作为实验室复查脚本，因为它把中间配置和最终结果都保留了下来。

### 3.10 本地函数

原脚本：

- 没有拆出本地函数。

新脚本：

- 新增了几个本地函数：
  - `build_truncated_coherent_density(...)`
  - `resolve_probability_path(...)`
  - `build_result_struct(...)`
  - `is_cvx_solved(...)`

这些函数的作用主要是：

1. 提高可读性；
2. 把重复逻辑单独封装；
3. 不改变主问题定义。

---

## 4. 哪些内容保持不变

相对原脚本，新脚本仍然保留了以下核心要素：

1. 仍然直接使用 `Probability.mat`；
2. 仍然从 `full_mu` 中选取 `selected_mu_list`；
3. 仍然保留 `q_selected` 作为目标函数权重；
4. 仍然通过 `LambdaIndices` 枚举 `N^(D+1)` 个策略；
5. 仍然用 CVX 求解 primal 型 SDP；
6. 仍然最终输出 `p_guess` 和 `H_min`。

所以它不是“完全换路线”，而是保留原 route4 结构后的一个 constrained 扩展版。

---

## 5. 哪些内容被真正修改了

真正的模型级修改只有三条：

1. 输入态从“只在 SDP 中使用 `rho_diag`”改为“主问题中使用完整 `rho`”；
2. coarse-graining 从“等宽分块”改为“固定高信息量边界 `[0,121,132,256]`”；
3. 主问题从“diagonal primal”改为“full primal”。

如果要用一句最短的话概括，就是：

- 原脚本是“`Probability.mat + diagonal trusted model + diagonal primal`”；
- 新脚本是“`Probability.mat + non-diagonal trusted coherent inputs + full primal`”。

---

## 6. 如何理解这两份脚本的关系

最合适的理解方式是：

1. 原脚本是 route4 的保守基线；
2. 新脚本不是推翻原脚本，而是在它的骨架上，把
   - trusted input
   - coarse-graining
   - 主问题测量约束
   这三件事替换成 constrained 主线版本；
3. 因此新脚本非常适合导师做“逐段审查”：
   - 先看哪些部分根本没变；
   - 再重点看第 3、4、7 段这三处关键修改。

---

## 7. 当前默认参数下的结果口径

在新脚本的默认参数下：

- `selected_mu_list = [100,120,140]`
- `q_selected = [1,0,0]`
- `custom_edges = [0,121,132,256]`
- `alpha_values = [0.54, 0.66 i, -0.72]`
- `M = 6`

你本机 Matlab 跑出的结果是：

- `Full primal status: Solved`
- `Full primal H_min: 1.227498940472`
- `Diagonal primal status: Infeasible`

这与 Python/MOSEK 主线的

- `H_min ≈ 1.227500864253`

只有约 `1e-6` 量级的差别，可以视为正常的数值误差。

---

## 8. 导师优先该看哪里

如果导师时间紧，最值得优先对照的部分是：

1. 原脚本第 3 段 vs 新脚本第 3 段
   - trusted input 是否仍然只是对角；
2. 原脚本第 4 段 vs 新脚本第 4 段
   - coarse-graining 是否还是等分；
3. 原脚本第 6 段 vs 新脚本第 7 段
   - SDP 主问题是否仍然只允许对角测量元。

这三处基本就概括了 route4 到 route4-ex-constrained 的全部关键变化。
