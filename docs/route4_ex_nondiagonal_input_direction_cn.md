# Route4-ex：非对角输入扩展方向的简要分析

## 1. 结论先说

如果把主线从当前 `route4` 转向一个允许**非对角输入态**的扩展版本 `route4-ex`，那么这是一个合理的研究方向，而且从理论上看，确实比当前 `route4` 更有希望提升认证值。

但必须同时明确三点：

1. 这已经不再是当前代码和当前报告对应的那个 `route4` 模型。
2. 之前关于“Fock 对角 POVM 假设合理”“去掉 POVM 非对角限制不会提升结果”的结论，不能再直接照搬到 `route4-ex`。
3. 若目标从 `H_min >= 2` 降到 `H_min >= 1`，当前原始 `route4` 仍然不乐观；而 `route4-ex` 才是更值得继续投入实验分析的版本。

## 2. 之前 route4 报告里说的“对角”到底指什么

在
[route4_full_primal_matrix_check_cn.md](./route4_full_primal_matrix_check_cn.md)
和
[route4_nondiagonal_povm_relaxation_check_cn.md](./route4_nondiagonal_povm_relaxation_check_cn.md)
里，重点讨论的是：

- 优化变量 \(M_{c,\lambda}\) 在 Fock 基下是否需要**非对角元**；
- 而不是“输入态必须是 Fock 本征态”。

也就是说，之前检查的是：

```text
full-primal 的最优 POVM 会不会主动长出非对角结构？
```

而不是：

```text
输入态能不能选成非对角密度矩阵？
```

这两件事不能混为一谈。

## 3. 当前 route4 的输入态其实是什么

当前 `route4` 的输入态不是“任意相干态”，而是**相位随机化后的相干态**，也就是 Fock 基下的对角混合态：

\[
\rho_\mu = \sum_n p_\mu(n)\,|n\rangle\langle n|,
\qquad
p_\mu(n)=e^{-\mu}\mu^n/n!.
\]

对应代码在：

- [src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L315)

这里的 `build_coherent_diagonals(...)` 明确构造的是 Poisson 光子数分布，也就是只保留 Fock 对角元的输入模型。

因此，当前 `route4` 的真实含义是：

```text
可信输入 = phase-randomized coherent states
测量模型 = phase-insensitive APD
SDP 只读取 Fock 对角统计
```

## 4. 为什么之前“对角 POVM 结论”成立

之前关于“full primal 与 diagonal primal 几乎一致”的结论，依赖的是一个很强的前提：

\[
\rho_x
\]

本身在 Fock 基下是对角的。

这样一来，所有目标函数和统计约束都只通过

\[
\mathrm{Tr}(\rho_x M)
\]

进入，而当 \(\rho_x\) 是对角矩阵时，这个量只依赖 \(M\) 的对角元。

这也是
[src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L1011)
中 `solve_phaseinsensitive_full_primal(...)` 的理论基础。

因此，在**当前 route4 模型**下：

- 允许 \(M_{c,\lambda}\) 是 full matrix 并不会真正增加可见自由度；
- 非对角元对目标函数和统计约束是“不可见”的；
- 所以对角 POVM 假设在这个模型里是合理的。

## 5. 输入态能否是非对角的

答案是：**能。**

例如，固定相位的纯相干态

\[
|\alpha\rangle\langle\alpha|
\]

在 Fock 基下一般就是非对角的，因为它包含

\[
|n\rangle\langle m|,\quad n\neq m
\]

的相干项。

所以从物理上说，“输入态能否非对角”完全没有问题。

真正的问题是：

```text
一旦你把输入态改成非对角，之前 route4 的很多简化就同时失效了。
```

## 6. 一旦做成 route4-ex，哪些旧结论会失效

如果定义 `route4-ex` 为：

- 输入端允许固定相位或更一般的非对角相干态；
- 测量端仍保留 APD / coarse-graining / route4 风格的优化框架；

那么以下结论都不能再直接继承：

1. “full primal 与 diagonal primal 应当等价”
2. “POVM 非对角元在优化里不可见”
3. “对角投影不会改变 route4 的目标函数和统计约束”
4. “最优解本身近似对角，所以放松 POVM 不会变好”

原因很简单：

一旦 \(\rho_x\) 非对角，

\[
\mathrm{Tr}(\rho_x M)
\]

就会真的依赖 \(M\) 的非对角元。

这时：

- 非对角输入态能够探测到更多测量结构；
- trusted inputs 对测量的约束会更强；
- 但同时 SDP 规模和数据需求也会上升。

## 7. 为什么 route4-ex 比当前 route4 更值得继续

如果目标从 `H_min >= 2` 降到 `H_min >= 1`，当前 `route4` 仍然不算乐观。

原因是：

1. 当前正式认证结果明显低于 1；
2. 当前最好超过 1 的数字主要还是 `distribution-only` 上界，不是正式 SDP 认证值；
3. 当前模型故意把输入态压缩成 Fock 对角分布，这会损失输入中的相干信息。

所以路线判断应当是：

- `route4-main`：
  继续保留，作为“现有 APD phase-insensitive 数据在最保守模型下能认证多少”的基线；
- `route4-ex`：
  作为更积极的扩展路线，尝试把输入侧的非对角相干信息重新引回认证模型。

这也是为什么，若主线重新回到 route4，我更建议把主投入放在 `route4-ex`，而不是继续只在原始对角版 route4 上调参。

## 8. route4-ex 的物理含义

`route4-ex` 可以理解为：

```text
保留 route4 “贴近 APD / coarse-grained 实验统计”的优点，
但不再把 trusted input 压缩成 phase-randomized 的 Fock 对角态。
```

更具体地说，它可能采用：

1. 固定光强但带已知相位的 coherent states；
2. 多个相位点组成的输入字母表；
3. 必要时允许多半径输入，而不是只区分平均光子数 `mu`。

这样一来，`route4-ex` 在思想上会更接近：

- `route3/route6` 的“输入更强”方向；
- 但在实验接口上仍可尽量保留 route4 的 APD / coarse-graining 视角。

## 9. route4-ex 第一轮最值得做什么实验

如果要正式启动 `route4-ex`，我建议第一轮不要直接求“大而全”的版本，而是做三个层次的递进验证。

### 9.1 第一层：只替换输入模型，不改优化框架

先把 `rho_diag` 替换成完整的 `rho_x`，看：

- `Tr(\rho_x M)` 是否确实开始对 POVM 非对角元敏感；
- diagonal primal 与 full primal 是否开始出现明显差距。

这一步的目的不是立刻冲高熵，而是确认：

```text
route4-ex 真的已经脱离“非对角元不可见”的旧 regime。
```

### 9.2 第二层：用小规模可解点做 primal / dual / full-primal 对照

建议从很小的参数点开始，例如：

- 2 个输入态；
- 2 或 4 个输出；
- 较小 cutoff；
- MOSEK 小规模验证。

要先回答：

1. 模型是否数值可解；
2. full primal 是否真的优于 diagonal primal；
3. 非对角输入是否确实提高了正式认证值。

### 9.3 第三层：再讨论 `H_min >= 1`

只有在前两层都成立后，才值得进入“参数搜索是否能到 1 bit”的阶段。

否则很可能会重蹈 route6 的问题：

- 表面上 raw 分布变好；
- 但 formal 认证并没有同步上升。

## 10. 对 route4-ex 的当前建议

基于现有证据，我建议把后续路线分成下面两条：

1. `route4-main`
   - 保持现有 phase-insensitive、Fock 对角输入版本；
   - 作为基线、对照和实验数据诊断工具继续保留。

2. `route4-ex`
   - 允许非对角输入态；
   - 重新检查 diagonal/full primal 的关系；
   - 重新判断在 `H_min >= 1` 目标下是否值得继续推进。

一句话总结就是：

```text
输入态当然可以非对角；
但一旦这样做，就已经进入一个新的 route4-ex 模型，
之前关于“对角 POVM 足够”的结论必须重新验证。
```

这也正是当前最值得继续实验分析的方向。
