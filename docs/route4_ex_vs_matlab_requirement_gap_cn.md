# Route4-ex 相对原 Matlab Route4 的一致性与要求差距分析

## 1. 报告目的

这份报告专门回答下面这个更实际的问题：

- 导师信任的是原 Matlab 文件  
  [`../src/matlab/guessprobprimal_phaseinsensitive.m`](../src/matlab/guessprobprimal_phaseinsensitive.m)
- 但当前 `route4-ex` 已经加入了很多新的变量、自由度和搜索步骤

因此需要判断：

1. `route4-ex` 到底是不是原 Matlab route4 的“Python 重写版”；
2. 如果不是，它具体在哪些地方已经偏离了原实验物理口径；
3. 哪些差异只是工程重构，哪些差异已经改变了协议含义；
4. 如果后续主线要回到导师更信任的 route4 口径，哪些内容应保留，哪些内容不应再作为正式实验结论使用。

## 2. 核心结论

结论先行：

1. 当前 `route4-ex` 不能被看作原 Matlab route4 的“等价重写”。
2. 问题不在于 `route4-ex` 的 SDP 代码一定算错了，而在于它已经改变了 trusted input 模型、输入参数化方式、coarse-graining 方式以及求解目标的物理解释。
3. 如果要求“和现有 `Probability.mat` 以及原 Matlab 物理口径严格一致”，那么 `route4-ex` 目前的大部分高值结果都不应直接当作原 route4 的正式实验结论。
4. 更准确的定位是：
   - 原 Matlab / Python `route4`：实验口径更直接、物理假设更少；
   - `route4-ex`：一个探索“非对角 trusted inputs 是否能显著抬高认证值”的结构性扩展原型。
5. 因此，如果导师要求“先回到可信主线”，建议正式主线回到原 `route4`；
   `route4-ex` 只保留为探索性旁线，除非后续实验能够补充它所需要的额外物理标定与数据支撑。

## 3. 原 Matlab 文件的执行流程

原 Matlab 主文件是：

- [`../src/matlab/guessprobprimal_phaseinsensitive.m`](../src/matlab/guessprobprimal_phaseinsensitive.m)

它的逻辑非常直接，流程可以概括为 6 步。

### 3.1 固定实验输入标签

Matlab 文件先固定：

- `selected_mu_list = [100, 120, 140]`
- `q_selected = [1/4, 1/4, 1/2]`
- `M = 280`
- `N = 4`

这里的核心物理标签是 `mu`。

在原 route4 口径下：

- 每个输入只由光强 `mu` 标记；
- 不引入额外相位标签；
- 也不再引入新的 amplitude scale 参数。

### 3.2 由 `mu` 直接构造 phase-insensitive 输入态

Matlab 接着直接构造：

- `rho_diag(i,n) = exp(-mu_i) * mu_i^n / n!`

也就是 Fock 基对角的 Poisson 分布。

这对应的物理对象是：

- 相位随机化相干态；
- 或更准确地说，phase-insensitive 的 Fock 对角输入模型。

它只保留每个输入在光子数基上的对角统计，不使用任何相位信息。

### 3.3 从 `Probability.mat` 中读取实验概率

Matlab 直接读取：

- [`../src/matlab/Probability.mat`](../src/matlab/Probability.mat)

并根据 `selected_mu_list` 选择对应的 3 行数据。

这一步的含义很清楚：

- `Probability.mat` 的每一行本来就对应实验里一个固定光强档位；
- 输入标签和实验数据行是一一对应的。

### 3.4 把 256 个 raw bins 等分为 `N` 个输出

Matlab 用

- `block_size = round(256 / N)`

把 256 个原始输出 bin 均匀并连续地合并成 `N` 个 coarse-grained 输出。

在当前文件里 `N = 4`，因此这里采用的是非常朴素、固定的四等分 coarse-graining。

### 3.5 建立对角 primal 问题

Matlab 只引入一个变量：

- `M_elements(M, N, num_strategies) >= 0`

它代表的是：

- 每个输出 `c`
- 每个策略 `lambda`
- 在每个 Fock 数 `n` 上的对角测量权重

也就是说，Matlab 求解的是一个只允许 Fock 对角 POVM 元的 primal 问题。

### 3.6 输出 `p_guess` 与 `H_min`

最后 Matlab 输出：

- `p_guess`
- `H_min = -log2(p_guess)`

整个流程里，没有额外搜索：

- 没有 phase pattern 搜索；
- 没有 `max_abs_alpha` 搜索；
- 没有自由半径搜索；
- 没有自定义 coarse-graining 边界搜索；
- 没有 toy / APD-like 理论后端。

从实验接口角度看，它非常“硬”：

- 给定哪几档光强；
- 给定哪几行实验概率；
- 给定固定 coarse-graining；
- 直接解一个 phase-insensitive diagonal primal。

## 4. 当前 route4-ex 的执行流程

`route4-ex` 的核心代码在：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

命令行入口在：

- [`../src/python/qrng_routes/route4_ex/main.py`](../src/python/qrng_routes/route4_ex/main.py)

而当前把结果推高的搜索脚本主要包括：

- [`../src/python/qrng_routes/route4_ex/external_scan.py`](../src/python/qrng_routes/route4_ex/external_scan.py)
- [`../src/python/qrng_routes/route4_ex/high_output_model_window_search.py`](../src/python/qrng_routes/route4_ex/high_output_model_window_search.py)
- [`../src/python/qrng_routes/route4_ex/high_output_local_refine.py`](../src/python/qrng_routes/route4_ex/high_output_local_refine.py)
- [`../src/python/qrng_routes/route4_ex/joint_compat_search.py`](../src/python/qrng_routes/route4_ex/joint_compat_search.py)

它的主流程与 Matlab 已经明显不同。

### 4.1 先指定或生成一组 `alpha_values`

在 `route4-ex` 中，trusted inputs 不再是直接由 `mu` 唯一确定，而是先给出一组复振幅：

- `alpha_values = [alpha_1, alpha_2, ..., alpha_D]`

这些 `alpha` 可能来自：

1. 手动直接指定；
2. 由 `intensities -> alpha_values` 映射生成；
3. 更激进地，由自由半径模型 `free_monotone_radii` 构造。

这一步已经不再是“原 route4 输入标签 = 实验光强档位”的简单关系。

### 4.2 用完整相干态 `|alpha><alpha|` 构造 trusted inputs

`route4-ex` 构造的是：

- 完整截断相干态密度矩阵 `rho_matrices`

而不仅仅是对角元。

这意味着：

- 输入态一般带有 Fock 基下的非对角元；
- 输入态之间的相位关系和重叠结构会直接进入模型。

### 4.3 再把外部概率表接进来

如果走 external 模式，`route4-ex` 仍然会读取：

- [`../src/matlab/Probability.mat`](../src/matlab/Probability.mat)

但这里有一个根本区别：

- 概率表的行仍然只按实验光强索引；
- 而 trusted inputs 却已经变成了由 `alpha` 的模长和相位共同决定的对象。

也就是说，`route4-ex` 已经把“实验数据标签”和“trusted-state 标签”拆成了两层。

### 4.4 coarse-graining 不再固定

`route4-ex` 可以：

1. 使用等分 coarse-graining；
2. 使用自定义边界 `custom_edges`；
3. 甚至根据目标行做高熵 contiguous 边界搜索，例如：
   - `3` 输出边界 `[0,121,132,256]`

这与 Matlab 的固定 `N=4` 四等分 coarse-graining 是不同的协议自由度。

### 4.5 求解问题不再局限于 diagonal primal

`route4-ex` 同时支持：

1. diagonal primal
2. full primal

其中 full primal 允许：

- 一般 Hermitian PSD 测量元；
- 而不是只允许 Fock 对角变量。

这已经不是原 Matlab 的原问题了，而是一个更强、更一般的新问题。

### 4.6 后续脚本继续对很多参数做搜索

当前 `route4-ex` 的结果并不是“固定协议下直接跑出来的”，而是通过搜索脚本继续扫描：

- phase pattern
- `max_abs_alpha`
- `free_monotone_radii`
- `num_outputs`
- `custom_edges`
- `q_selected`
- `cutoff`

也就是说，`route4-ex` 当前更像一个“协议与参数搜索平台”，而不是原 Matlab 那种单一固定模型。

## 5. Matlab 与 route4-ex 的关键区别

下面按层次把差异拆开，并判断它们属于哪一类。

### 5.1 输入态模型不同

Matlab：

- trusted inputs 是 phase-insensitive 的 Poisson Fock 对角态；
- 输入由 `mu` 唯一标记。

`route4-ex`：

- trusted inputs 是完整相干态 `|alpha><alpha|`；
- 输入由复振幅 `alpha = r e^{i\phi}` 标记；
- 一般带有非对角元。

判断：

- 这是最核心的协议级改变；
- 不是简单工程重构；
- 它改变了 SDP 中可用的 trusted information。

### 5.2 实验标签与理论输入之间多了一层映射

Matlab：

- `mu = 100` 这一行数据，就对应 `mu = 100` 的输入态。

`route4-ex`：

- `Probability.mat` 的第几行，仍然由实验光强决定；
- 但 trusted input 却先要经过
  `intensity -> alpha` 的映射；
- 当前代码里这层映射还允许通过 `max_abs_alpha` 调整整体尺度。

判断：

- 这不是纯单位变换；
- 因为改变 `alpha` 的绝对大小会改变相干态重叠，从而改变 SDP；
- 如果没有独立的实验标定，这会使“数据行”和“trusted input”的对应关系失去唯一性。

这正是导师最担心的点之一，而且这个担心是合理的。

### 5.3 引入 phase pattern

Matlab：

- 不用相位模式；
- 输入只有光强标签。

`route4-ex`：

- 常用 `0_pi2_pi`、`0_pi3_2pi3` 等相位模式；
- 同一组实验强度行，可以被赋上不同相位。

判断：

- 如果实验侧并没有为 `Probability.mat` 同时记录“这 3 行数据分别对应哪些确定相位的 coherent states”，
  那么这些 phase pattern 只是理论假设；
- 它们不能直接被当作已经由当前实验数据支持的事实。

因此：

- 把 phase pattern 扫出来的更高值，当作原 route4 的正式实验结论，是不合要求的；
- 但它可以作为“如果未来实验可控相位输入，则可能提升认证值”的探索性结果。

### 5.4 求解问题从 diagonal primal 扩展到了 full primal

Matlab：

- 只解 Fock 对角 primal。

`route4-ex`：

- 允许 full primal，一般 Hermitian PSD 测量元均可参与优化。

判断：

- 从数学上说这是合法的新问题；
- 但它不再是原 Matlab 的同一个 primal；
- 它的提升来自更强的 trusted-input 假设和更一般的测量变量，而不只是“同一个 route4 被 Python 重新实现得更好”。

因此：

- full primal 的结果不能直接表述成“原 Matlab route4 结果的改进版”；
- 更准确的说法应该是“route4 的非对角输入扩展模型”。

### 5.5 coarse-graining 方式变化很大

Matlab：

- 固定等分 coarse-graining；
- `N=4` 只是预先设定的输出数。

`route4-ex`：

- 可以改输出数；
- 可以手动边界；
- 还可以从同一批数据里先挑“看起来高熵”的边界。

判断：

- 这不是不允许，但它已经是新的协议设计自由度；
- 如果边界是看完数据再选，容易带来数据依赖的后选风险；
- 严格实验口径下，coarse-graining 边界最好预注册，或者由独立校准集确定。

因此：

- 类似 `[0,121,132,256]` 这种从数据搜索出来的高熵边界，不应直接和 Matlab 的固定 `N=4` 结果并列比较为“同一主线的改进”。

### 5.6 `q_selected` 从固定权重变成了搜索维度

Matlab：

- `q_selected` 固定为 `[1/4,1/4,1/2]`。

`route4-ex`：

- 会扫描 `[1,1,1]`、`[2,1,1]`、`[5,1,1]`、`[1,0,0]` 等不同配置；
- 代码里还会自动归一化。

判断：

- 如果协议本来就允许“生成轮只用某个输入、测试轮用其它输入”，那么偏置 `q` 并不一定不合理；
- 但它肯定已经不是和 Matlab 同一组固定设置的直接比较。

因此：

- 用偏置 `q` 得到的高值不能直接写成“原 Matlab route4 在现有设置下也能做到这么高”；
- 更合理的描述是“在改变生成轮分布解释后，可得到更高的每生成轮熵”。

### 5.7 截断维数 `cutoff` 的物理口径变化

Matlab：

- `M = 280`
- 这是与 `mu = 100,120,140` 这类高光强档位相匹配的。

`route4-ex`：

- 常用 `cutoff = 4, 6, 8, 12`
- 这是因为它把 `alpha` 缩到很小的量级，问题规模才可算。

判断：

- 如果当前 `alpha` 只是一个探索性、归一化后的模型参数，那么小 `cutoff` 可以作为原型计算；
- 但如果要把它解释成与实验 `mu = 100~160` 直接对应的真实相干态，那么这些 `cutoff` 显然不够。

因此：

- 小 `cutoff` 与“实验高光强直接对应”这两种说法不能同时成立。

### 5.8 增加了 `prob_floor`

Matlab：

- 原脚本里没有对 coarse-grained 概率做 `prob_floor` 正则化。

`route4-ex`：

- 默认常用 `prob_floor = 1e-12`。

判断：

- 这是典型的数值稳定化手段；
- 通常影响很小，但它确实会轻微改动实验概率表。

因此：

- 如果做“和 Matlab 一一对照”的 strict run，建议把 `prob_floor` 关闭；
- 如果保留 `prob_floor`，报告中应明确它是数值正则化，而不是原实验数据本身。

### 5.9 route4-ex 还有 toy / APD-like 理论后端

`route4-ex` 不仅能读 `Probability.mat`，还能跑：

- toy coherent-projector 后端
- APD-like displaced-count 后端

判断：

- 这些后端对理解结构现象有帮助；
- 但它们与当前实验数据主线不是同一个问题。

因此：

- 它们不应进入“现有实验数据正式结果”的主叙事；
- 只能作为模型探索材料。

## 6. 哪些改动可以接受，哪些当前不应作为正式实验主线

### 6.1 可以接受的工程性改动

下面这些改动本身没有问题，甚至是应该保留的：

1. 用 Python 重写 `.mat` 读取、求解器调用和结果保存；
2. 增加 CLI、JSON 输出和批处理脚本；
3. 用更稳定的方式构造概率和记录诊断信息；
4. 在 strict 设置下复现实验数据到 SDP 的整条链路。

这些都属于工程增强，不改变原 route4 的物理口径。

### 6.2 当前不应直接作为“原 Matlab route4 正式结论”的部分

如果导师要求的是“保持原实验物理意义”，那么下面这些在当前阶段都不宜直接写成主结果：

1. 用完整非对角 coherent-state 输入替换原 phase-insensitive 对角输入；
2. 在没有独立标定的情况下，把实验强度再映射成可调的 `max_abs_alpha`；
3. 使用 `free_monotone_radii` 自由半径搜索；
4. 在没有相位标记实验数据的情况下扫描 phase pattern；
5. 使用从同一批数据中挑出的自定义高熵 coarse-graining 边界；
6. 把 `q_selected` 作为搜索变量，而不是协议先验；
7. 用 low-cutoff 原型直接解释高强度实验输入；
8. 把 full primal 结果直接写成“原 route4 的改进值”。

这些做法不是“数学上错误”，但它们会使结果不再满足“原 Matlab 主线的实验物理口径”。

## 7. 如果要让 route4-ex 靠近导师要求，需要满足什么条件

如果后续还想保留 `route4-ex` 这条扩展方向，但又希望它更接近导师可接受的实验口径，那么至少需要满足下面几条。

### 7.1 固定输入映射，而不是优化输入映射

不能再把：

- `max_abs_alpha`
- `free_monotone_radii`

当成纯搜索变量。

更合理的做法是：

1. 由实验给出“光强 -> 振幅模长”的标定关系；
2. 或至少给出一个可信区间；
3. 然后只在这个固定映射下求解 SDP，而不是用它去“找最高熵”。

### 7.2 说明相位信息是否真实存在

如果实验侧没有：

- 受控的输入相位；
- 或相位稳定的输入态标记；

那么 phase pattern 就只能作为探索性理论设想。

只有当实验能明确给出：

- 某一行概率数据对应某个固定 `alpha = r e^{i\phi}`

时，full primal 的非对角输入假设才真正有实验支撑。

### 7.3 coarse-graining 边界必须固定

若要避免后选问题，coarse-graining 边界应当：

1. 由协议预先固定；
2. 或由独立训练数据决定；
3. 而不是在同一份认证数据上直接搜索。

### 7.4 使用与物理规模匹配的截断维数

若输入最终仍要解释为高光强 coherent states，就必须使用足够大的 `cutoff`。

否则只能把当前 low-cutoff 结果解释为：

- 结构原型；
- 而不是实验定量结论。

## 8. 一个非常重要的判断

`route4-ex` 不是“坏代码”，也不是“完全无意义”。

更准确的判断应该是：

1. 它是一个明确的新路线原型；
2. 它试图回答的问题是：
   - “如果 trusted inputs 不再局限于 phase-insensitive 对角态，而允许非对角 coherent inputs，那么能否明显提高 formal 认证值？”
3. 这个问题本身是合理的；
4. 但它已经不是原 Matlab route4 在现有实验物理口径下的同一个问题。

因此，真正需要修正的不是“把 route4-ex 全部否定”，而是它的对外口径。

更合适的表述应是：

- 原 route4 主线：以 Matlab / Python `route4` 为准；
- `route4-ex`：作为一条新的非对角 trusted-input 扩展线，用于探索未来若补充实验准备与标定后，是否有潜力把认证值做高。

## 9. 建议的后续安排

### 9.1 如果目标是先给导师一个稳妥、可信的主线

建议主线回到：

- [`../src/matlab/guessprobprimal_phaseinsensitive.m`](../src/matlab/guessprobprimal_phaseinsensitive.m)
- [`../src/python/qrng_routes/route4/phaseinsensitive.py`](../src/python/qrng_routes/route4/phaseinsensitive.py)

并把 Python `route4` 当成 Matlab 的稳定复现与扩展工具。

已有对照材料可参考：

- [`./route4_matlab_vs_python_report_cn.md`](./route4_matlab_vs_python_report_cn.md)

### 9.2 如果还想保留 route4-ex

建议把 `route4-ex` 的定位降为：

- 探索性结构原型；
- 不作为当前实验主结果；
- 报告里明确写清它需要额外实验支撑：
  - 输入振幅标定；
  - 相位控制或相位标签；
  - 固定 coarse-graining 方案；
  - 与输入规模匹配的数值截断。

同时也应保留它当前仓库中的自我定位：

- [`../src/python/qrng_routes/route4_ex/README.md`](../src/python/qrng_routes/route4_ex/README.md)

其中已经明确写到：

- `route4-ex` 是 structural prototype
- 还不是 experimental-data pipeline

这与本报告的判断是一致的。

## 10. 结论

如果标准是“忠实继承原 Matlab route4 的实验物理意义”，那么当前 `route4-ex` 确实有多处不合要求的地方。

最关键的不是代码复杂，而是它已经引入了以下新的协议自由度：

1. 非对角 coherent trusted inputs；
2. 强度到振幅的可调映射；
3. 相位模式；
4. custom coarse-graining 边界；
5. biased `q_selected`；
6. full primal 一般测量变量。

这些自由度本身不一定错误，但它们意味着：

- `route4-ex` 已经不是原 Matlab route4；
- 它更像一条新的扩展路线。

因此，面向导师或实验室时，最稳妥的说法是：

- 原 route4 正式主线仍以 Matlab / Python `route4` 为准；
- `route4-ex` 只作为探索“非对角输入是否能提升认证”的旁线；
- 除非后续实验能补齐它所需的额外物理标定与输入态信息，否则不应把它当前的高值结果直接当成原 route4 的实验结论。
