# Route4 全矩阵 primal 最优算符的近似对角性检查

## 1. 目的

这份短报告的目的，是直接检查 route4 的 unrestricted full-primal 最优解中，POVM 变量
\(M_{c,\lambda}\) 是否会显著使用非对角元。

这一步比只比较最优值更直接，因为导师关心的是：

- 若去掉“POVM 必须为 Fock 对角”的假设，最优自变量本身会不会变成明显非对角；
- 如果最优 \(M_{c,\lambda}\) 本身就已经是对角的，那么对角假设就不仅是“最优值上没有损失”，而且在这个可解例子里还与求解器返回的最优结构一致。

## 2. 检查对象与代码位置

本次检查针对的是 route4 的 full-primal 求解器，即
[src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py) 中的 `solve_phaseinsensitive_full_primal(...)`。

关键实现位置如下：

- [src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L395)
  `solve_phaseinsensitive_full_primal(...)`：定义 full-matrix primal，不对 \(M_{c,\lambda}\) 施加对角约束。
- [src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L432)
  `operators[(output, strategy_id)] = cp.Variable((cutoff, cutoff), hermitian=True)`：每个 \(M_{c,\lambda}\) 都是完整的 Hermitian 矩阵变量。
- [src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L451)
  目标函数通过 `trace(rho_x M_{c,\lambda})` 进入。
- [src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L472)
  统计约束也通过 `trace(rho_x M_c)` 进入。

由于 route4 当前输入态在代码中是 Fock 基底下的对角相干态分布，
即 `rho_x = diag(p_n^{(x)})`，目标函数和统计约束都只显式依赖 \(M_{c,\lambda}\) 的对角部分。

## 3. 本次实验参数

为了确保问题可解且规模可控，本次选用一个已经验证可解的小例子：

- 输入强度：`selected_mu_list = [0, 20]`
- 输入概率：`q_selected = [0.5, 0.5]`
- 输出数：`num_outputs = 2`
- 截断维数：`cutoff = 60`
- 求解器：`MOSEK`

这里选这个例子的原因是：

- 它和之前已保存的 primal/full-primal 对比结果完全兼容；
- 它能够在本机上稳定求解；
- 它足以回答“最优 \(M_{c,\lambda}\) 会不会主动长出非对角结构”这个问题。

对应的数值摘要保存于：

- [output/qrng_routes/route4_full_primal_matrix_diagonality_mu0_20_outputs2_cutoff60_mosek.json](../output/qrng_routes/route4_full_primal_matrix_diagonality_mu0_20_outputs2_cutoff60_mosek.json)

## 4. 检查方法

在 full-primal 求解结束后，对每个最优矩阵 \(M_{c,\lambda}\) 做如下分解：

\[
M_{c,\lambda} = \mathrm{diag}(M_{c,\lambda}) + \mathrm{offdiag}(M_{c,\lambda}).
\]

然后计算以下指标：

- `fro_norm = ||M_{c,\lambda}||_F`
- `diag_norm = ||diag(M_{c,\lambda})||_F`
- `offdiag_norm = ||offdiag(M_{c,\lambda})||_F`
- `offdiag_over_fro = offdiag_norm / fro_norm`
- `offdiag_over_diag = offdiag_norm / diag_norm`
- `max_abs_offdiag = max_{i != j} |(M_{c,\lambda})_{ij}|`

若这些量都接近 0，则说明最优矩阵是近似对角的；
若它们在多个矩阵上都明显非零，则说明求解器确实在利用非对角自由度。

## 5. 结果

本次 full-primal 求解得到：

- `status = optimal`
- `p_guess = 0.9999979671997808`
- `H_min = 2.9327137761028894e-06`
- `num_strategies = 8`
- `matrix_count = 16`

更关键的是所有最优矩阵的非对角指标：

- `max_offdiag_over_fro = 0.0`
- `median_offdiag_over_fro = 0.0`
- `max_offdiag_over_diag = 0.0`
- `median_offdiag_over_diag = 0.0`
- `num_nearly_diagonal_1e-6 = 16 / 16`
- `num_nearly_diagonal_1e-4 = 16 / 16`

也就是说，在导出的数值精度下，本次检查到的所有最优 \(M_{c,\lambda}\) 都是严格对角的。

## 6. 代表性矩阵指标

下面列出几个代表性的最优矩阵条目。

### 6.1 大权重条目

- `strategy_id = 0`
- `lambda_tuple = [0, 0, 0]`
- `output = 0`
- `fro_norm = 3.5571978272300484`
- `diag_norm = 3.5571978272300484`
- `offdiag_norm = 0.0`
- `offdiag_over_fro = 0.0`
- `max_abs_offdiag = 0.0`
- `min_eig = 0.24998101872356454`

对应的互补输出条目：

- `strategy_id = 0`
- `lambda_tuple = [0, 0, 0]`
- `output = 1`
- `fro_norm = 0.842567315948898`
- `diag_norm = 0.842567315948898`
- `offdiag_norm = 0.0`
- `offdiag_over_fro = 0.0`
- `max_abs_offdiag = 0.0`
- `min_eig = -1.9017821871076736e-08`

这里的最小特征值出现了约 `1e-8` 的轻微负数，这是标准数值误差量级，不影响“近似 PSD”判断。

### 6.2 极小权重条目

- `strategy_id = 7`
- `lambda_tuple = [1, 1, 1]`
- `output = 0`
- `fro_norm = 1.9467021219501146e-06`
- `diag_norm = 1.9467021219501146e-06`
- `offdiag_norm = 0.0`
- `offdiag_over_fro = 0.0`
- `max_abs_offdiag = 0.0`

互补输出条目：

- `strategy_id = 7`
- `lambda_tuple = [1, 1, 1]`
- `output = 1`
- `fro_norm = 1.003239542332853e-06`
- `diag_norm = 1.003239542332853e-06`
- `offdiag_norm = 0.0`
- `offdiag_over_fro = 0.0`
- `max_abs_offdiag = 0.0`

即使在这种极小范数条目上，也没有观察到任何非对角结构。

## 7. 这为什么支持对角假设

本次结果对“route4 中对角 POVM 假设是否合理”提供了两层支持。

第一层是最优值层面。
此前已经保存的比较结果
[output/qrng_routes/route4_primal_full_compare_mu0_20_outputs2_cutoff60_mosek.json](../output/qrng_routes/route4_primal_full_compare_mu0_20_outputs2_cutoff60_mosek.json)
表明：

- diagonal primal 的 `p_guess = 1.0000000032932743`
- full primal 的 `p_guess = 0.9999979476803695`

二者差距约为 `2e-6`，属于数值容差量级。

第二层是最优自变量层面。
这次直接抽取 full-primal 的最优 \(M_{c,\lambda}\) 后发现，全部 16 个最优矩阵的非对角范数都为 0。
这说明在这个可解例子里，求解器并没有利用非对角自由度来改进解，反而直接返回了对角矩阵解。

因此，对于 route4 当前这套“输入态为 Fock 对角相干态分布、统计约束只由 `Tr(rho_x M_c)` 给出”的建模，
现有数值证据支持如下判断：

- 放开 full-matrix 自由度后，没有观察到更优的认证结果；
- 在至少一个已解完的可行 full-primal 例子中，最优解本身就是对角的；
- 因而“采用对角 POVM 作为 route4 的工作假设”在当前模型下是合理且数值上被支持的。

## 8. 与已有随机投影检验的一致性

这一结论还与已有的“对角投影不变性”检验一致：

- [output/qrng_routes/route4_diagonal_projection_invariance_check.json](../output/qrng_routes/route4_diagonal_projection_invariance_check.json)

该文件对应
[src/python/qrng_routes/route4/phaseinsensitive.py](../src/python/qrng_routes/route4/phaseinsensitive.py#L746)
中的 `run_route4_diagonal_projection_invariance_check(...)`。

它验证的是：对随机生成的非对角 PSD 算符族，只要保持 POVM 完备性，那么对角投影前后，route4 的统计量和 route4 风格的目标函数都不变。

这说明在当前 route4 线性泛函里，非对角部分本来就处于“不可见”状态；
而本次 full-primal 最优矩阵检查进一步说明，即使显式允许这些不可见自由度存在，求解器也可以直接返回对角最优解。

## 9. 结论

本次针对 route4 full-primal 最优矩阵的直接检查表明：

- 在 `selected_mu = [0, 20]`、`num_outputs = 2`、`cutoff = 60`、`MOSEK` 的可解实例中，
  所有最优 \(M_{c,\lambda}\) 都是数值上严格对角的；
- 非对角指标 `offdiag_norm`、`offdiag_over_fro`、`max_abs_offdiag` 在全部 16 个矩阵上均为 0；
- 这与先前“diagonal primal 与 full primal 的最优值几乎重合”的结果完全一致；
- 因此，当前没有数值证据表明 route4 去掉对角限制后会得到更优、且实质依赖非对角元的最优解。

更谨慎地说，这还不是对所有参数情形的严格数学证明；
但作为导师讨论所需的数值证据，它已经相当直接地支持了：在 route4 当前建模里，对角 POVM 假设是合理的。
