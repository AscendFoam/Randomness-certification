# Route4-ex 阶段性总结汇报

## 1. 执行摘要

`route4-ex` 是在原始 `route4` 基础上发展出来的一条扩展路线。它保留了 `route4` “直接吃 APD 概率表、贴近现有实验数据”的优点，但不再把 trusted inputs 限制为 Fock 基对角的相位随机化相干态，而是改为允许**非对角的截断相干态**作为可信输入模型。

截至当前阶段，这条路线已经取得两个足够明确的阶段性结论：

1. 如果目标从 `H_min >= 2` 降到 `H_min >= 1`，那么 `route4-ex` 已经成功达到该目标。
2. 当前最稳的正式结果来自
   - 窗口 `[100,120,140]`
   - `3` 输出粗粒化边界 `[0,121,132,256]`
   - 相位图样 `0_pi2_pi`
   - 偏置生成分布 `q=[1,0,0]`
   - 自由单调半径已经沿病态边界进一步精修到
     `r=[0.5379541015625,0.6620458984375,0.7179541015625]`
   - `MOSEK optimal` 给出 `H_min ≈ 1.54395`

对应结果文件：

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)

与此同时，约束残差 / 可行性余量检查表明，这批 `1.53 ~ 1.54` 的高值稳定点并不是明显的脏解，而是位于一条非常窄但数值上仍然干净的稳定前沿附近。

## 2. Route4-ex 相对原始 Route4 的共同点与区别

### 2.1 共同点

`route4-ex` 和原始 `route4` 的共同点主要有三条。

1. 都直接使用实验侧的概率表作为主要数据入口。
   - 当前用的是 [`../src/matlab/Probability.mat`](../src/matlab/Probability.mat)。
2. 都把原始 `256` 维计数直方图先做 coarse-graining，再把离散输出送入 primal/full-primal 认证流程。
3. 都把“偏置输入分布 `q_selected`”作为生成轮的先验权重显式写进 guessing probability 目标函数。

因此，`route4-ex` 不是完全推翻原始 `route4`，而是在它的实验接口和离散化框架上做了更强的 trusted-input 建模。

### 2.2 关键区别

最核心的区别是 trusted input model。

原始 `route4`：

- 可信输入是**相位随机化的相干态**，在 Fock 基下是对角混合态；
- 测量也被建模为 `phase-insensitive`，因此 Fock 对角 POVM 足够；
- 对应说明见 [`../src/python/qrng_routes/route4/README.md`](../src/python/qrng_routes/route4/README.md)。

`route4-ex`：

- 可信输入是**完整截断相干态** `|alpha><alpha|`；
- 在 Fock 基下带有非对角元；
- 因此 `full primal` 真的能“看见”非对角结构，不能再把旧 route4 的“只看对角元就够了”的结论直接搬过来。

这个区别在早期方向说明文档里已经解释过：

- [`./route4_ex_nondiagonal_input_direction_cn.md`](./route4_ex_nondiagonal_input_direction_cn.md)

### 2.3 一句话理解

可以把两条路线的差别压缩成一句话：

- `route4` 是“现有 APD 数据 + 最保守的 phase-insensitive/Fock 对角 trusted model”；
- `route4-ex` 是“仍然使用现有 APD 概率数据，但把输入侧的相干信息重新放回认证模型里”。

## 3. 为什么要设计 Route4-ex

之所以需要 `route4-ex`，原因来自两方面。

### 3.1 原始 Route4 的正式认证值明显受限

原始 `route4` 虽然实验接口最直接，但正式认证值一直偏低。它更适合作为“保守基线”而不是冲击高熵的主线。

见原始路线说明：

- [`../src/python/qrng_routes/route4/README.md`](../src/python/qrng_routes/route4/README.md)

### 3.2 Probability.mat 中的信息可能比原始 Route4 模型允许的更多

当前的 `Probability.mat` 是在固定实验硬件下记录下来的计数分布。原始 `route4` 把 trusted inputs 压缩成了 Fock 对角 Poisson 分布，这会主动丢掉输入相干信息。

`route4-ex` 的设计动机就是：

- 继续使用这份实验数据；
- 继续沿用 APD/coarse-graining 的统计接口；
- 但改用更强的非对角 trusted inputs 去约束测量。

换句话说，`route4-ex` 的核心设计理由不是“换实验”，而是“在尽量不换实验接口的前提下，提升认证模型的信息利用率”。

## 4. 程序结构与运行流程

### 4.1 核心代码文件

当前 `route4-ex` 的主代码在：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

外围搜索脚本包括：

- [`../src/python/qrng_routes/route4_ex/high_output_model_window_search.py`](../src/python/qrng_routes/route4_ex/high_output_model_window_search.py)
- [`../src/python/qrng_routes/route4_ex/high_output_local_refine.py`](../src/python/qrng_routes/route4_ex/high_output_local_refine.py)

基础说明见：

- [`../src/python/qrng_routes/route4_ex/README.md`](../src/python/qrng_routes/route4_ex/README.md)

### 4.2 运行流程

程序的实际流程可以概括为 6 步。

#### 第一步：加载外部概率表

外部数据入口在：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

对应接口：

- `load_external_probability_table(...)`

当前主要使用：

- [`../src/matlab/Probability.mat`](../src/matlab/Probability.mat)

#### 第二步：选择输入窗口

程序不是一次把全部强度都塞进同一个问题，而是先选一个小窗口，例如：

- `[100,120,140]`
- `[80,100,120]`
- `[120,140,160]`

这些窗口定义在：

- [`../src/python/qrng_routes/route4_ex/joint_compat_search.py`](../src/python/qrng_routes/route4_ex/joint_compat_search.py)

#### 第三步：把输入强度映射成 trusted coherent states

这里有两种建模方式。

1. `rigid_intensity_scaled`
   - 半径由 `sqrt(I)` 规则决定；
   - 代码入口：
     - [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)
     - `intensities_to_alpha_values(...)`

2. `free_monotone_radii`
   - 不再强制半径严格按 `sqrt(I)` 缩放；
   - 只要求三个半径按大小递增；
   - 由搜索脚本直接枚举半径组合。

目前最强结果就是来自第二类。

#### 第四步：对 256 维直方图做 coarse-graining

如果不给自定义边界，就按等覆盖方式粗粒化；
如果给出自定义边界，就用：

- `coarse_grain_probability_table_with_edges(...)`

对应代码：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

这里的关键认知是：

- 输出桶边界不是次要细节；
- 它直接决定 formal feasibility 和认证值。

#### 第五步：构造单个 route4-ex 实例

外部表 + trusted inputs + `q_selected` + coarse-graining 共同被装入：

- `prepare_route4_ex_external_instance(...)`

对应代码：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

#### 第六步：求解 diagonal/full primal

最终正式值来自：

- `solve_route4_ex_full_primal(...)`

对应接口：

- `compare_route4_ex_external_diagonal_full(...)`

这一步输出：

- `full_status`
- `full_p_guess`
- `full_H_min`

## 5. 这阶段做过哪些搜索

### 5.1 从 2 输出主线过渡到 3 输出主线

最开始，`route4-ex` 在 `[100,120,140]` 窗口上先是沿着 `2` 输出方向推进，最好做到了接近 `1` 但还不到 `1`。

见早期窗口报告：

- [`./route4_ex_probabilitymat_window_report_cn.md`](./route4_ex_probabilitymat_window_report_cn.md)

后来更深入的诊断表明：

- 高熵 `3/4` 输出边界并不是“天然不行”；
- 真正的瓶颈在 trusted-state 模型过硬。

### 5.2 高输出窗口/模型联合搜索

这一步由：

- [`../src/python/qrng_routes/route4_ex/high_output_model_window_search.py`](../src/python/qrng_routes/route4_ex/high_output_model_window_search.py)

完成。它同时扫描：

- 输入窗口
- 输出数 `3/4`
- 相位图样
- `rigid` 或 `free_monotone_radii`

对应结果文件：

- [`../output/qrng_routes/route4_ex_high_output_model_window_search_q100.json`](../output/qrng_routes/route4_ex_high_output_model_window_search_q100.json)

### 5.3 围绕最强点做局部半径精修

最强主线确定后，又进一步做了局部精修：

- [`../src/python/qrng_routes/route4_ex/high_output_local_refine.py`](../src/python/qrng_routes/route4_ex/high_output_local_refine.py)

对应结果文件：

- [`../output/qrng_routes/route4_ex_high_output_local_refine_fastnear_3out_q100.json`](../output/qrng_routes/route4_ex_high_output_local_refine_fastnear_3out_q100.json)
- [`../output/qrng_routes/route4_ex_high_output_local_refine_3out_q100.json`](../output/qrng_routes/route4_ex_high_output_local_refine_3out_q100.json)

## 6. 当前阶段的结果分析

### 6.1 最稳的正式结果

当前最稳、最适合写进主结论的点是：

- 窗口：`[100,120,140]`
- `3` 输出边界：`[0,121,132,256]`
- 相位：`0_pi2_pi`
- trusted-state 模型：`free_monotone_radii`
- 半径：`[0.5379541015625,0.6620458984375,0.7179541015625]`
- 生成分布：`q=[1,0,0]`

`MOSEK` 复核结果：

- `status = optimal`
- `H_min ≈ 1.54395`

文件：

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)

这意味着：

- `route4-ex` 已经正式达到 `H_min >= 1`。
- 而且当前 `MOSEK` 正式可确认值已经提升到约 `1.54 bit`。

### 6.2 为什么这不是单点偶然

局部精修和后续病态边界定位都表明，这条主线附近存在一段 `status = optimal` 且 `H_min > 1.5` 的稳定前沿，而不是只存在一个孤立点。

例如在：

- [`../output/qrng_routes/route4_ex_high_output_local_refine_fastnear_3out_q100.json`](../output/qrng_routes/route4_ex_high_output_local_refine_fastnear_3out_q100.json)
- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q13over32_to_q7over16_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q13over32_to_q7over16_2pt.json)
- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q209over512_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q209over512_to_q105over256_2pt.json)

中，有：

- `[0.53796875, 0.66203125, 0.71796875]`，`MOSEK optimal`，`H_min ≈ 1.5299`
- `[0.537958984375, 0.662041015625, 0.717958984375]`，`MOSEK optimal`，`H_min ≈ 1.5385`
- `[0.5379541015625, 0.6620458984375, 0.7179541015625]`，`MOSEK optimal`，`H_min ≈ 1.5440`
- `[0.535, 0.665, 0.715]`，`SCS optimal`，`H_min ≈ 1.3005`

这说明：

- `>1 bit` 不是某一个参数点的偶然峰值；
- `[100,120,140] + free radii + 3输出 + q=[1,0,0]` 已经形成了一段高值稳定带。

### 6.3 更高的尖峰应该怎么解读

局部精修里也出现了更高的尖峰：

- `[0.535,0.660,0.715]`，`H_min ≈ 1.5441`
- `[0.535,0.655,0.720]`，`H_min ≈ 1.5194`

但它们当前状态是：

- `optimal_inaccurate`

因此当前最合理的口径是：

- 这些点可以写成“更高候选值”；
- 但不能取代 `MOSEK` 已确认的 `1.54395` 稳定点成为主结论。

### 6.4 病态边界与残差检查带来的更新

围绕上述高值主线，又进一步做了两类诊断：

- 病态边界阈值定位；
- 高值稳定点的约束残差 / 可行性余量检查。

对应文档：

- [`./route4_ex_threshold_and_residual_check_cn.md`](./route4_ex_threshold_and_residual_check_cn.md)

目前已经确认：

- `MOSEK` 的稳定/失稳转折发生在极窄的半径窗口内；
- 最后确认稳定点为  
  `[0.5379541015625, 0.6620458984375, 0.7179541015625]`
- 再向失败侧推进极小一步  
  到 `[0.53795166015625, 0.66204833984375, 0.71795166015625]`  
  就会 `MOSEK` 失败。

同时，对高值稳定点 `[0.5379541015625, 0.6620458984375, 0.7179541015625]` 的残差检查给出：

- `measurement_violation_max ≈ 3.81e-09`
- `completeness_violation_max ≈ 5.46e-11`
- `psd_min_eig_min ≈ -3.53e-09`

这些残差比旧基线点更大，但仍然非常小，没有出现明显失控。因此当前 `1.53 ~ 1.54` 这批 `MOSEK optimal` 点可以被视为数值上足够干净的正式结果，而不是简单的脏解。

### 6.5 其它窗口的意义

大搜索还给出两个补充信息。

1. `[120,140,160]` 的 rigid `3` 输出点有非常强的 `SCS` 信号，甚至到 `≈ 1.3783`；
2. `[80,100,120]` 的 free-radii `3` 输出也出现了 `optimal ≈ 1.2482` 的点。

但这些方向的 `MOSEK` 复核目前都失败了，没有形成正式确认结果。

因此它们现在更适合被写成：

- 候选补充方向；

而不适合抢占主结论。

## 7. Route4-ex 在实验上的优点

### 7.1 仍然贴着现有实验数据走

`route4-ex` 不要求先丢掉现在的 `Probability.mat`，因此实验接口连续性较好。

### 7.2 比原始 Route4 更强，但没有完全脱离 APD 路线

它不是去追一个全新的硬件架构，而是在原有 APD/coarse-graining 接口上，把输入建模增强了。

### 7.3 可以自然解释“生成输入”和“测试输入”的分工

当前最强点依赖：

- `q=[1,0,0]`

这在实验上可以解释为：

- 第一个输入用于生成轮；
- 其余输入主要用于测试/约束轮。

相关说明见：

- [`./route4_ex_biased_q_experimental_mapping_cn.md`](./route4_ex_biased_q_experimental_mapping_cn.md)

## 8. Route4-ex 在实验上的风险与边界

### 8.1 当前正式强结果仍依赖模型放松

最强主线建立在 `free_monotone_radii` 上，而不是严格的 `sqrt(I)` 映射上。

这意味着：

- 实验上需要说明为什么这组半径是合理的 trusted-input 描述；
- 或者至少说明它代表了一个更灵活但仍物理可接受的输入标定模型。

### 8.2 偏置 `q` 不能直接读成整机平均吞吐率

`q=[1,0,0]` 的合理解释是：

- 生成轮只使用第一个输入；
- 其它输入保留给测试轮。

因此最终实验汇报里，应把：

- 生成轮每轮最小熵

和

- 乘上生成轮占比后的总平均速率

区分开。

### 8.3 当前主结果虽已很强，但仍靠近病态边界

当前最强正式点已经由 `MOSEK` 确认达到 `≈1.54 bit`，并且残差检查没有显示明显失控。

但仍需保留一个重要边界说明：

- 这批高值点位于一条非常窄的稳定前沿附近；
- 再向失稳端推进极小一步，`MOSEK` 就会直接失败。

因此，正式汇报里更严谨的说法应是：

- “已找到可由 `MOSEK` 正式确认的高值稳定带，`H_min` 可达约 `1.54 bit`”

而不是：

- “存在宽阔稳定高平台”。

## 9. 下一阶段的改进计划

### 9.1 第一优先级：形成正式阶段总结图景

当前已经足够写阶段报告，建议把结果分成三层：

1. `MOSEK` 正式确认点
2. `SCS optimal` 的高值可行区域
3. `optimal_inaccurate` 的更高候选峰值

### 9.2 第二优先级：继续做高值稳定带的旁证检查

当前更值得做的不是重新大范围扫点，而是：

- 再选取 `1.53` 附近另外 1-2 个稳定点做同类残差检查；
- 证明主结果不是单点偶然。

### 9.3 第三优先级：把病态边界机理进一步解释清楚

虽然现在已经能把 `≈1.54 bit` 作为正式主结果，但如果导师继续追问“为什么再往前一点就失败”，仍值得补做：

- 稳定/失稳边界的更细局部剖面；
- 不同求解器设定下的数值行为对比；
- 更贴近求解误差分析的解释。

### 9.4 第四优先级：围绕已确认主线做更精细的局部搜索

如果仍想继续推高当前结果，最合理的方向是：

- 固定窗口 `[100,120,140]`
- 固定 `3` 输出边界 `[0,121,132,256]`
- 固定相位 `0_pi2_pi`
- 在 `[0.535, 0.655~0.665, 0.715~0.725]` 一带做更细网格

### 9.5 第五优先级：若要加强实验说服力，再做一轮窗口鲁棒性检查

不是为了再找更高值，而是为了证明：

- route4-ex 的可行性不是只依赖一个窗口偶然成立。

## 10. 如果希望继续冲到 2，可以怎么优化

当前要坦诚地说：

- `route4-ex` 已经明确过了 `1 bit`；
- 但离 `2 bit` 仍然有明显距离。

如果仍希望向 `2` 继续推进，最可能的方向不是单纯重复当前扫描，而是更结构性的优化。

### 10.1 增强 trusted-input 字母表

当前还是 3 个输入态的小字母表。若要进一步抬高约束强度，可能需要：

- 更多输入态；
- 更系统的半径/相位设计；
- 甚至更接近 `route5/route6` 的 coherent alphabet 思路。

### 10.2 改进输出离散化

当前主要还是矩形/连续区间式 coarse-graining。若想冲更高熵：

- 可以尝试更有针对性的边界设计；
- 但同时要注意 formal feasibility 不要被破坏。

### 10.3 研究更稳定的高值求解策略

现在许多高值点都变成了：

- `optimal_inaccurate`
- 或 `MOSEK` 直接失败

因此，想冲 2 之前，先把“高值点为什么在更强求解器下不稳”搞清楚，是必要步骤。

### 10.4 从实验侧补更强的数据

如果后续实验能够提供：

- 更丰富的输入态标定；
- 更细致的分箱统计；
- 更明确的生成/测试轮分工数据；

那么 `route4-ex` 还有进一步提高的空间。

## 11. 一句话总结

`route4-ex` 当前已经从“原始 route4 的一个扩展想法”，走到了“具有正式结果支撑的可行主线”：

- 它相对于原始 `route4` 的关键提升，在于把 trusted inputs 从 Fock 对角模型升级成了非对角截断相干态；
- 它保留了原始 `route4` 贴近 APD 概率表的实验接口；
- 当前最稳的正式高值已经由 `MOSEK` 确认达到 `H_min ≈ 1.54395`；
- 因而如果当前阶段目标是 `H_min >= 1`，那么 `route4-ex` 已经实现目标；
- 若后续仍希望继续冲击 `2 bit`，则需要从 trusted-input 字母表、输出离散化和高值求解稳定性三个方向继续升级。
