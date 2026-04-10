# Route4 Strict Non-Diagonal 阶段报告

## 1. 报告目的

这份报告总结新路线 `route4_strict_nondiagonal` 的当前实现与首轮结果。

这条路线的提出背景是：

- 导师怀疑原 route4 的关键问题不一定在输入态本身，而可能在于：
  - 原 Matlab 文件虽然来自相干态物理背景；
  - 但 SDP 里实际只使用了 Fock 对角输入；
  - 因而希望考察“如果把输入改成严格的非对角 coherent inputs，同时去掉 POVM 的对角限制，会发生什么”。

为了避免再次走到 `route4-ex` 那种“同时引入太多新自由度”的方向，这条新路线刻意采用了**最小改动原则**。

## 2. 这条路线的严格定义

### 2.1 保留不变的部分

`route4_strict_nondiagonal` 保留了原 Matlab / Python route4 主线的以下部分：

1. 输入标签仍然是 `selected_mu_list`
2. 先验分布仍然是 `q_selected`
3. 实验概率数据仍然直接来自
   [`../src/matlab/Probability.mat`](../src/matlab/Probability.mat)
4. coarse-graining 仍然使用原 route4 的等分规则
5. 输出数 `N`、输入窗口、`Probability.mat` 行索引逻辑不变

也就是说，这条路线**不再引入**：

- `max_abs_alpha` 搜索
- `free_monotone_radii`
- 自定义高熵边界搜索
- phase pattern 扫描
- 外加 toy / APD-like 理论后端

### 2.2 唯一核心改动

唯一核心改动是：

1. 原 route4：
   trusted input 为 Fock 对角的 `rho_diag`
2. 新路线：
   trusted input 改为固定 coherent states
   \[
   \rho_x = |\alpha_x\rangle\langle \alpha_x|
   \]
3. 同时 POVM 允许为一般 Hermitian PSD 矩阵，而不再限制为对角变量

### 2.3 本路线的固定映射

为避免 route4-ex 那种“缩放也拿来搜索”的问题，这里采用固定映射：

\[
\alpha_x = \sqrt{\bar n_x}\, e^{i\phi_x},
\qquad
\bar n_x = c_{\mu} \cdot \mu_x
\]

其中：

- `mu_x` 就是 `selected_mu_list` 里的标签
- `c_mu = mean_photons_per_mu_label` 是固定常数，不参与优化
- `phi_x` 是固定相位列表，不参与搜索

当前实现默认：

- `phase_values = [0,0,...,0]`
- `mean_photons_per_mu_label = 1.0`

之所以默认取 `1.0`，是因为这能与**当前 Matlab 代码实际使用的 `rho_diag`** 精确对齐：

- 当 `mu = 100,120,140`
- 且 coherent input 取 `|\alpha|^2 = 100,120,140`
- 它们在 Fock 基下的对角元就与 route4 的 Poisson `rho_diag` 数值一致

这一点在本路线的实例构造结果里已经被直接验证。

如果后续要改用“Matlab 注释里的物理口径”，即：

- `100 -> 平均光子数 1.0`

则只需把

- `mean_photons_per_mu_label = 0.01`

作为固定映射重新运行即可。

## 3. 为什么引入“支撑子空间 full-primal”

如果直接在原 route4 的 Fock 截断空间上做 full-primal，那么规模会非常大。

以当前主线参数

- `selected_mu = [100,120,140]`
- `N = 4`
- `cutoff = 280`

为例：

- 策略数 `num_strategies = 4^(3+1) = 256`
- full-primal 的算子变量数是 `4 × 256 = 1024`
- 如果每个变量都是 `280 x 280` 厄米矩阵，则实标量规模约为
  `1024 × 280 × 280 = 80,281,600`

这在当前环境下并不现实。

因此，本路线采用了一个关键但严格等价的技术处理：

### 3.1 输入态张成的支撑降维

对一组输入态 ket

\[
|\psi_x\rangle = |\alpha_x\rangle
\]

先计算它们张成空间的一组正交基 `B`，然后把所有输入态投影到这个支撑子空间：

\[
\rho_x^{(\mathrm{red})} = B^\dagger \rho_x B
\]

### 3.2 为什么这是严格等价的

因为：

1. 所有输入态都完全位于该支撑子空间内
2. 目标函数和统计约束都只通过 `Tr(rho_x M)` 进入
3. 对于支撑子空间外的部分，输入态没有振幅，因此不会被观测到

所以：

- 在支撑子空间上解出的 full-primal
- 与在原全空间上、但只被这些输入态探测到的 full-primal

在认证值上是等价的。

### 3.3 本路线的一个重要规模优势

对于当前主线点 `[100,120,140]`，虽然 `cutoff = 280`，但三组 coherent states 的张成空间维数只有：

- `support_dimension = 3`

因此降维后：

- 厄米标量数变成 `1024 × 3 × 3 = 9216`

这就从原本不可算的量级，降到了可以求解的量级。

## 4. 代码实现位置

本路线的新代码位于：

- [`../src/python/qrng_routes/route4_strict_nondiagonal/prototype.py`](../src/python/qrng_routes/route4_strict_nondiagonal/prototype.py)
- [`../src/python/qrng_routes/route4_strict_nondiagonal/main.py`](../src/python/qrng_routes/route4_strict_nondiagonal/main.py)
- [`../src/python/qrng_routes/route4_strict_nondiagonal/README.md`](../src/python/qrng_routes/route4_strict_nondiagonal/README.md)

其中最关键的接口有：

1. `prepare_route4_strict_nondiagonal_instance(...)`
   - 复用原 route4 的实验数据接口
   - 构造固定 coherent inputs
   - 构造支撑降维实例

2. `solve_route4_strict_nondiagonal_full_primal(...)`
   - 在支撑子空间上求解 full-primal

3. `compare_route4_strict_nondiagonal_with_reference(...)`
   - 对比：
     - 原 route4 phase-insensitive diagonal primal
     - strict non-diagonal full-primal

这三者合起来，正好回答：

> 在不改 `Probability.mat`、不加 route4-ex 式搜索自由度的前提下，仅把输入改为严格 non-diagonal coherent states，并放开 POVM，结果会怎样？

## 5. 当前后台状态

在本轮检查时，系统中已经没有仍在运行的 `route4_strict_nondiagonal` Python 进程。

也就是说，当前所有已启动的 strict 路线测试都已经结束，结果已保存到：

- [`../output/qrng_routes/route4_strict_nondiagonal_prepare_mu100120140_N4_cutoff280_scale1.json`](../output/qrng_routes/route4_strict_nondiagonal_prepare_mu100120140_N4_cutoff280_scale1.json)
- [`../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale1.json`](../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale1.json)
- [`../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale001.json`](../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale001.json)

## 6. 第一组关键结果：`scale = 1.0`

### 6.1 这是最重要的“隔离实验”

这里取：

- `selected_mu = [100,120,140]`
- `q = [0.25,0.25,0.5]`
- `N = 4`
- `cutoff = 280`
- `phase_values = [0,0,0]`
- `mean_photons_per_mu_label = 1.0`

对应文件：

- [`../output/qrng_routes/route4_strict_nondiagonal_prepare_mu100120140_N4_cutoff280_scale1.json`](../output/qrng_routes/route4_strict_nondiagonal_prepare_mu100120140_N4_cutoff280_scale1.json)
- [`../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale1.json`](../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale1.json)

### 6.2 为什么这组结果特别重要

因为在这组口径下：

- strict coherent input 的 Fock 对角元
- 与原 route4 的 `rho_diag`

几乎完全一致：

- `rho_diag_reference_linf_gap ≈ 6.28e-15`

这意味着：

> 这组测试几乎就是在原 route4 输入模型上，只额外“加回了非对角相干项”，其它对角统计并没有被改坏。

因此这是一个非常干净的隔离实验。

### 6.3 结果

原 route4 reference primal：

- `status = optimal`
- `p_guess ≈ 0.8947763902`
- `H_min ≈ 0.1604009052`

strict non-diagonal full-primal：

- `status = optimal_inaccurate`
- `p_guess ≈ 0.9481785423`
- `H_min ≈ 0.0767693503`

两者差异：

- `p_guess_abs_gap ≈ 0.0534021521`
- `H_min_gap_strict_minus_reference ≈ -0.0836315550`

### 6.4 解释

这说明：

1. 在这组“最贴近原 route4”的 strict non-diagonal 定义下，
   **仅仅把输入改成完整 coherent states 并放开 POVM，并没有让认证值提高；**
2. 相反，当前 SCS 结果显示 strict full-primal 的认证值更低；
3. 也就是说，从目前数值证据看：
   - `non-diagonal input + full POVM`
   - 并不会自动把原 route4 的 `H_min` 做高。

这对路线判断是一个很重要的负结果。

## 7. 第二组关键结果：`scale = 0.01`

### 7.1 这组结果用于检验“Matlab 注释口径”

这里把

- `mean_photons_per_mu_label = 0.01`

也就是把：

- `100 -> 平均光子数 1.0`
- `120 -> 1.2`
- `140 -> 1.4`

对应文件：

- [`../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale001.json`](../output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale001.json)

### 7.2 结果

在这组口径下：

- `support_dimension = 3`
- 但 `rho_diag_reference_linf_gap ≈ 0.3678794412`

这说明 strict coherent input 的对角部分，已经和原 route4 的参考 `rho_diag` 差得非常大。

原 route4 reference primal 仍然是：

- `status = optimal`
- `H_min ≈ 0.1604009052`

但 strict non-diagonal full-primal 变成：

- `status = infeasible`
- `H_min = null`

### 7.3 解释

这说明：

1. 如果把 `Probability.mat` 的 `[100,120,140]` 严格解释成
   `[1.0,1.2,1.4]` 平均光子数的 coherent inputs；
2. 那么当前观测概率表与这组 strict coherent trusted inputs
   在 full-primal 下并不兼容。

因此，这个结果至少说明一件事：

> “Matlab 注释里的物理量解释”与“Matlab 代码实际送进 SDP 的 Poisson 输入模型”之间，当前并不是同一个口径。

## 8. 当前最值得强调的判断

### 8.1 这条新路线已经实现了它最重要的验证目标

`route4_strict_nondiagonal` 现在已经能回答下面这个问题：

> 如果不引入 route4-ex 那种大量额外自由度，只做一个最小改动版 strict non-diagonal route4，那么结果会不会比原 route4 更好？

当前答案是：

- **至少从首轮结果看，不会自动更好。**

### 8.2 `scale = 1.0` 的结果特别有说服力

因为它满足：

- 原 route4 的输入对角统计被精确保留
- 只额外加入非对角 coherent 结构

而在这种最公平的对照下，strict 路线并没有提高 `H_min`，反而使 `H_min` 下降到：

- `0.07677`

这说明：

- 原 route4 的问题并不一定出在“把非对角结构丢掉了”
- 或至少，**把非对角结构加回来并不足以自动提升结果**

### 8.3 `scale = 0.01` 进一步暴露了物理口径问题

如果按 Matlab 注释去理解光强标签，那么 strict full-primal 会直接 infeasible。

这反过来也说明：

- 后续如果导师要坚持“原 route4 物理上本来就是非对角输入”
- 那么必须先把
  - 光强标签
  - 平均光子数
  - trusted coherent input 的绝对标定
  这几件事重新说清楚

否则 strict non-diagonal 模型没有唯一确定的输入态。

## 9. 当前路线的优点与局限

### 9.1 优点

1. 比 `route4-ex` 干净得多
2. 没有额外引入自由缩放搜索
3. 没有引入高熵边界后选
4. 仍然直接使用原 route4 的 `Probability.mat`
5. 可以把“输入非对角性”的影响单独拎出来检验

### 9.2 局限

1. 目前 strict full-primal 的主结果仍是 `SCS`
2. 在 `scale = 1.0` 的主线点上，strict full-primal 状态是
   `optimal_inaccurate`
3. 虽然支撑降维大大降低了问题规模，但 CVXPY 仍有明显 canonicalization 开销
4. 最关键的物理问题仍未完全解决：
   - `selected_mu` 与真实 coherent amplitude 的映射到底是什么？

## 10. 下一步建议

我建议接下来按下面顺序继续，而不是立刻大规模扫参。

### 10.1 先做一个 MOSEK 复核

最值得做的是对当前主线点：

- `[100,120,140]`
- `q = [0.25,0.25,0.5]`
- `N = 4`
- `cutoff = 280`
- `phase = [0,0,0]`
- `scale = 1.0`

做一次 `MOSEK` 复核。

因为如果 `MOSEK` 也确认 strict 路线不优于原 route4，那么这个负结果会更扎实。

### 10.2 增加 residual / feasibility 诊断

建议给 strict full-primal 补一个类似 `route4-ex residual diagnostics` 的工具，检查：

- PSD 最小特征值
- 完备性残差
- 测量匹配残差

这样可以判断 `optimal_inaccurate` 到底离“正式可信”有多远。

### 10.3 再决定是否要引入固定相位

当前默认用了：

- `phase_values = [0,0,0]`

这只是为了 strict 与最小改动。

如果后续实验确实能支持固定相位输入，那么可以再尝试少量**预先指定**的 phase 配置。但这一步应晚于 `scale` 标定问题的澄清。

### 10.4 如果目标是现实主线，暂时仍应保留原 route4

从现在的证据看，更稳妥的主线判断仍然是：

- 原 route4：
  仍然是当前与 `Probability.mat` 最直接匹配的主线
- `route4_strict_nondiagonal`：
  是一个很有价值的验证工具，但目前还没有显示出“能替代原 route4 主线”的证据

## 11. 当前阶段结论

`route4_strict_nondiagonal` 已经完成了它的第一阶段目标：

1. 成功搭建了一个严格、最小改动版的 non-diagonal route4；
2. 在不引入 route4-ex 那套搜索自由度的前提下，直接检验了
   “strict coherent inputs + full POVM” 的效果；
3. 当前首轮结果表明：
   - 在与原 route4 最对齐的 `scale = 1.0` 口径下，
     strict 路线没有提升认证值，反而给出更低的 `H_min`
   - 在 Matlab 注释口径的 `scale = 0.01` 下，
     strict full-primal 甚至变成 infeasible

因此，当前最合理的阶段性判断是：

> `route4_strict_nondiagonal` 已经证明“仅靠把输入改成非对角 coherent states，并放开 POVM”，并不能自动修复 route4；如果后续要继续沿这条线推进，关键不在于继续随便扫参，而在于先把光强标签与 trusted coherent inputs 的物理标定问题彻底讲清楚。
