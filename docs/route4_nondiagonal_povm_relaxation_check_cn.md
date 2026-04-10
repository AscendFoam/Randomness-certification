# Route4 去除对角 POVM 限制是否会提升结果：理论分析与实验验证

## 1. 结论先说

在当前 `route4` 的模型下，去掉“POVM 在 Fock 基下对角”的限制，不会让认证结果变好。更准确地说：

1. 对当前 `route4` 而言，这不是“经验上大概率没帮助”，而是一个由模型结构决定的结论。
2. 只要输入态仍然是当前代码里的 Fock 对角态 `rho_diag`，并且统计约束与目标函数仍然只通过 `Tr(rho_diag M)` 进入 SDP，那么任意非对角 POVM 解都可以做一次“取对角投影”，得到一个：
   - 仍然可行；
   - 统计量完全相同；
   - 目标函数值完全相同
   的对角 POVM 解。
3. 因而，允许一般 Hermitian PSD POVM 元素并不会把最优值推高；full-POVM 最优值与 diagonal-POVM 最优值应当相同。

这次我给出两层证据：

1. 理论证明：说明为什么当前 `route4` 的非对角元在优化里“不可见”。
2. 数值实验：
   - 一个“对角投影不变性”实验，直接验证 route4 真正使用的统计量和目标函数在投影前后完全不变；
   - 一个小规模 `diagonal primal` vs `full primal` 的 MOSEK 对照点，显示两者的 `p_guess` 只差 `2.06e-6`，属于数值容差量级。

因此，**如果想让 route4 的结果真正变好，关键不是放松 POVM 的对角限制，而是必须连同输入模型/实验数据模型一起改变。**

## 2. 当前 route4 的关键结构

### 2.1 输入态本身就是对角的

当前 `route4` 不是用完整的相干态密度矩阵，而是先把输入态替换成 Fock 基下的对角分布：

- 构造位置：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L78-L92)
- 实例准备位置：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L100-L182)

对应代码里：

```python
def build_coherent_diagonals(selected_mu_list, cutoff):
    diagonals = np.zeros((len(selected_mu_list), cutoff), dtype=float)
    ...
    diagonals[idx, :] = np.exp(log_probs)
    return diagonals
```

也就是说，`route4` 输入到 SDP 里的不是完整的 `rho_x`，而是

\[
\rho_x = \mathrm{diag}(r_x(0), r_x(1), \dots, r_x(d-1)).
\]

### 2.2 当前 primal 只依赖对角线统计

现有对角 primal 在
[phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L315-L392)。

它的统计约束是

```python
rho_diag[input_index, :] @ total_elements[:, output] == probabilities[input_index, output]
```

目标函数也是

```python
objective_expr += q_selected[input_index] * (rho_diag[input_index, :] @ primal_sum)
```

这已经说明：当前优化器真正“看到”的，只是 POVM 对角元与 `rho_diag` 的内积。

### 2.3 我新增的 full primal 也只能通过同一批线性泛函读取数据

为了做对照，我补了一个不强加对角 POVM 假设的 full primal：

- 实现位置：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L395-L498)
- 对照入口：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L595-L644)
- CLI 入口：
  [main.py](../src/python/qrng_routes/route4/main.py#L24-L153)

这个版本把变量改成一般 Hermitian PSD 矩阵：

```python
operators = {
    (output, strategy_id): cp.Variable((cutoff, cutoff), hermitian=True)
    ...
}
```

但它的统计约束和目标函数仍然是

\[
\mathrm{Tr}(D_x M), \quad D_x := \mathrm{diag}(r_x),
\]

的形式，而不是依赖 `M` 的非对角元。

## 3. 为什么理论上 full POVM 不会更优

### 3.1 关键投影

对任意矩阵 `M`，定义它在 Fock 基下的对角投影

\[
\Delta(M) := \sum_n |n\rangle \langle n| M |n\rangle \langle n|.
\]

也就是“把所有非对角元清零，只保留对角线”。

### 3.2 route4 里所有可观测量都对这个投影不敏感

因为 `D_x` 本身是对角矩阵，所以

\[
\mathrm{Tr}(D_x M) = \mathrm{Tr}(D_x \Delta(M)).
\]

原因非常直接：`D_x` 与 `M` 相乘后，迹只会取到 `M` 的对角元，`M` 的非对角元根本不会进入这个数。

这件事同时作用在两类量上：

1. 概率约束 `P(c|x)`。
2. primal 目标函数里的 guessing probability 线性项。

### 3.3 可行性也会被保留

full primal 的完备性约束是

\[
\sum_c M_{c,\lambda} = s_\lambda I.
\]

对两边同时做对角投影，有

\[
\sum_c \Delta(M_{c,\lambda}) = \Delta(s_\lambda I) = s_\lambda I.
\]

因此完备性不变。

同时，如果 `M_{c,\lambda} \succeq 0`，那么 `\Delta(M_{c,\lambda})` 仍然是 PSD。因为它只是把一个 PSD 矩阵替换成“同一条对角线对应的对角矩阵”，其对角元非负，故仍是 PSD。

### 3.4 结论

于是，对任意 full primal 可行解 `{M_{c,\lambda}}`，都可以构造出一个 diagonal primal 可行解 `{Δ(M_{c,\lambda})}`，并且：

1. 所有 `P(c|x)` 完全相同；
2. 目标函数完全相同。

所以：

- full primal 的每个可行点，都能投影成一个同值的 diagonal primal 可行点；
- diagonal primal 本来又是 full primal 的子集。

因此两边最优值必须相等。

这意味着：

> 在当前 route4 里，非对角 POVM 自由度不会带来更大的 guessing probability，也不会带来更小的 guessing probability；它们对最优值是“冗余自由度”。

## 4. 数值实验 1：对角投影不变性检查

### 4.1 实验目的

这个实验不直接求解 SDP，而是直接构造一批“带非对角元”的随机 PSD 算符族，并且强制它们满足 route4 需要的 POVM 完备性条件，然后检查：

1. route4 概率型统计量在投影前后是否变化；
2. route4 风格目标函数在投影前后是否变化；
3. 完备性是否保持；
4. 算符是否仍是 PSD。

对应实现：

- 函数位置：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L746-L896)
- 结果文件：
  [route4_diagonal_projection_invariance_check.json](../output/qrng_routes/route4_diagonal_projection_invariance_check.json)

运行命令：

```bash
PYTHONPATH=src/python python -m qrng_routes.route4 \
  --mode diagonal-projection-check \
  --seed 7 \
  --num-trials 6 \
  > output/qrng_routes/route4_diagonal_projection_invariance_check.json
```

### 4.2 实验设计

我选了两类实例：

1. `default_like_three_inputs_four_outputs`
   - `selected_mu = [100, 120, 140]`
   - `q = [0.25, 0.25, 0.5]`
   - `num_outputs = 4`
   - `cutoff = 16`
   - `num_strategies = 256`
2. `low_intensity_two_inputs_two_outputs`
   - `selected_mu = [0, 20]`
   - `q = [0.5, 0.5]`
   - `num_outputs = 2`
   - `cutoff = 24`
   - `num_strategies = 8`

对每个策略块，我先构造标量倍数的对角基底，再给前两个输出加入一对相反的随机对称非对角扰动 `+K` / `-K`，从而同时保证：

1. `sum_c M_{c,\lambda} = s_\lambda I`
2. 每个 `M_{c,\lambda}` 仍然 PSD

然后比较投影前后的：

1. `P(c|x)` 型统计；
2. route4 primal 风格目标函数。

### 4.3 结果

结果文件给出的关键数值是：

| Case | `max_stats_gap` | `max_objective_gap` | `max_completeness_residual` | `min_operator_eigenvalue` |
|---|---:|---:|---:|---:|
| `default_like_three_inputs_four_outputs` | `0.0` | `0.0` | `4.44e-16` | `6.06e-06` |
| `low_intensity_two_inputs_two_outputs` | `0.0` | `0.0` | `2.22e-16` | `1.22e-02` |

解释如下：

1. `max_stats_gap = 0.0`
   说明 route4 约束真正依赖的统计量在做对角投影前后完全不变。
2. `max_objective_gap = 0.0`
   说明 route4 的 guessing-probability 型目标函数在投影前后完全不变。
3. `max_completeness_residual ~ 1e-16`
   说明构造的 full POVM 族确实保持了完备性，数值误差只在浮点舍入量级。
4. `min_operator_eigenvalue > 0`
   说明实验里的 full 矩阵确实是 PSD，而不是“算符已经坏了所以结论才成立”。

这个实验说明了一件非常关键的事：

> 在当前 route4 中，即使你真的给 POVM 元素塞进大量非对角自由度，只要输入态和统计读取方式不变，这些非对角自由度也不会改变 route4 用到的任何优化量。

## 5. 数值实验 2：小规模 full primal vs diagonal primal

### 5.1 实验目的

上一个实验验证的是“route4 线性泛函对非对角元不敏感”。这一节再进一步，直接把小规模实例喂给：

1. 现有 diagonal primal；
2. 新增的 full primal；

然后看最终 `p_guess` 是否一致。

### 5.2 实验配置

对应结果文件：

- [route4_primal_full_compare_mu0_20_outputs2_cutoff60_mosek.json](../output/qrng_routes/route4_primal_full_compare_mu0_20_outputs2_cutoff60_mosek.json)

命令：

```bash
PYTHONPATH=src/python conda run --no-capture-output -n generic python -m qrng_routes.route4 \
  --mode primal-full-compare \
  --solver MOSEK \
  --selected-mu 0 20 \
  --q-values 0.5 0.5 \
  --num-outputs 2 \
  --cutoff 60 \
  --prob-floor 1e-12 \
  --max-hermitian-scalar-count 400000
```

这里选择的是一个能被 MOSEK 稳定求解的小规模点：

- `selected_mu = [0, 20]`
- `q = [0.5, 0.5]`
- `num_outputs = 2`
- `cutoff = 60`

### 5.3 结果

结果文件中的核心数值是：

| Solver model | status | `p_guess` | `H_min` |
|---|---|---:|---:|
| diagonal primal | `optimal` | `1.0000000032932743` | `-4.75e-09` |
| full primal | `optimal` | `0.9999979476803695` | `2.96e-06` |

两者差异：

- `p_guess_abs_gap = 2.0556e-06`
- `H_min_abs_gap = 2.9656e-06`

这个差异量级很重要。它说明：

1. full primal 没有出现任何“明显优于 diagonal primal”的结果；
2. 两者差距只在 `1e-6` 量级，完全符合数值求解容差的预期；
3. `p_guess` 一侧轻微超过 `1`，另一侧轻微低于 `1`，本身就是数值容差的典型表现，而不是物理上真的出现 `p_guess > 1`。

所以这个实验应当被解读为：

> 在一个可直接 full-vs-diagonal 对照的小规模实例上，去掉对角限制没有带来可分辨的提升；两者的最优值在数值精度内一致。

## 6. 两类实验合在一起意味着什么

如果只做实验 2，其实还可能有人质疑：

- “也许只是这个小点碰巧差不多。”

如果只做实验 1，也可能有人质疑：

- “你没有真的把 full primal 解到底。”

把两类证据放在一起，逻辑就完整了：

1. 理论证明已经说明，在当前 route4 里最优值应当严格相同。
2. 实验 1 直接验证了 route4 真正读取的统计量和目标函数对非对角元完全不敏感。
3. 实验 2 则在一个可算的小规模点上，直接把 full primal 和 diagonal primal 解出来，结果只差 `1e-6` 量级。

因此，当前最稳妥的表述是：

> 在现有 route4 模型下，去掉对角 POVM 限制不会让结果更好；若数值上出现极小偏差，应解释为求解器容差，而不是物理或安全模型上的真实增益。

## 7. 为什么这不意味着“非对角 POVM 永远没用”

这里必须把边界讲清楚。

这次结论只针对**当前 route4** 成立。它依赖两个前提：

1. 输入态被替换成了 Fock 对角的 `rho_diag`；
2. 实验数据约束也只通过这些对角态的 `Tr(rho_diag M)` 进入优化。

如果以后把 route4 改成下面任一种形式，结论就不能直接照搬：

1. 不再只用 Fock 对角输入，而是用完整相干态密度矩阵；
2. 引入相位信息或其它能探测非对角元的实验约束；
3. 改成和当前 route5 类似的 IQ 连续变量前端，再做数字 coarse-graining。

一旦数据本身能“看见”非对角元，那么 POVM 的非对角结构才可能真正影响认证值。

换句话说：

> 不是“非对角 POVM 在任何问题里都没用”，而是“在当前 route4 的这套输入模型与数据接口下，非对角 POVM 没有可利用的信息通道”。

## 8. 对 route4 后续工作的直接建议

基于这次分析，我建议 route4 后续不要再把主要精力放在“是否去掉 POVM 对角限制”上，因为这条路不会带来实质增益。

更值得做的是下面三件事：

1. 继续把 route4 当作“现有 APD 数据诊断线”。
2. 如果希望 route4 的结果提升，就去改输入/数据模型，而不是只改 POVM 参数化。
3. 如果目标是冲击更高的 `H_min`，则应优先考虑 route5 这类能够把更多物理信息带入 SDP 的方案。

## 9. 复现入口

### 9.1 新增代码入口

- full primal:
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L395-L498)
- diagonal/full 对照：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L595-L644)
- 小规模批量对照：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L647-L743)
- 对角投影不变性实验：
  [phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L746-L896)
- CLI：
  [main.py](../src/python/qrng_routes/route4/main.py#L24-L153)

### 9.2 结果文件

- 投影不变性实验：
  [route4_diagonal_projection_invariance_check.json](../output/qrng_routes/route4_diagonal_projection_invariance_check.json)
- MOSEK 小规模 full-vs-diagonal 对照：
  [route4_primal_full_compare_mu0_20_outputs2_cutoff60_mosek.json](../output/qrng_routes/route4_primal_full_compare_mu0_20_outputs2_cutoff60_mosek.json)

## 10. 一句话总结

在当前 route4 里，问题不在于“POVM 限制得太死”。真正的问题是：**输入态和实验统计本身已经把优化问题压缩成了只看对角线的模型。** 所以，仅仅去掉 POVM 的对角限制，不会把 route4 的认证结果推高。
