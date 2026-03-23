# Route 2: Prepare-and-Measure MDI QRNG

## 1. 为什么这条路线重要

如果实验室当前的目标是尽量清楚、尽量可信地把认证随机性推到 `H_min >= 2`，那么在目前几条路线里，`route2` 是最值得优先投入的主线。

原因不是它“最省事”，而是它同时满足三点：

- 安全模型是干净的，没有旧 `mdi_qrng` 里那种把中央测量错误拆成局域结构之后带来的 `C3/C4` 过强约束问题。
- 数值上已经看到了非常清晰的路径：4 输出 baseline 几乎到 `2 bit`，更高输出搜索已经能超过 `2 bit`。
- 实验图景虽然比 `route1` 更有工程难度，但仍然是可以讲清楚、可以分阶段推进、可以逐步落地的。

这份 README 的目标不是只给写代码的人看，而是也能帮助你向不太熟悉这块的老师解释：

- 这条路线到底在实验上做什么；
- 为什么它比旧 MDI 建模更正确；
- 代码里每一层对象分别对应什么物理含义；
- 现在的结果意味着什么，不意味着什么；
- 真正的工程难点在哪。

## 2. 一句话图景

最简洁的说法是：

“可信源准备一组已知量子输入态，把它们送进一个不可信的中央测量黑盒，只根据观测到的输入-输出统计 `P(c|x,y)`，用单设备 SDP 去认证输出中到底有多少真随机性。”

这里最关键的词有三个：

- `可信输入`
- `中央黑盒测量`
- `单设备 SDP`

Route 2 的本质，就是把“安全性难点”集中到中央测量上，同时尽量不对中央测量做额外错误假设。

### 2.1 理论依据与文献对应关系

这一节专门回答一个很实际的问题：

```text
route2 到底是“直接复现某一篇论文”，
还是“在已有文献背景上，针对当前项目问题重新整理出来的一条路线”？
```

最准确的说法是：

- 它不是对某一篇论文的逐字照搬复现；
- 但它也不是凭空构想出来的全新协议；
- 它是基于已有 `prepare-and-measure` / quantum-input / MDI 文献背景，
  再结合当前项目里旧 `mdi_qrng` 的结构问题，
  重新整理出来的一版“正确单设备化”原型。

下面把三部分分开说明。

#### 2.1.1 哪些部分来自已有文献背景

第一部分是“大的物理思想”，这部分有明确文献来源，并不是当前代码自己发明的。

最关键的背景是 quantum-input / prepare-and-measure / MDI 这条线：

- 实验者可以可信地准备一组已知量子输入态；
- 这些输入态被送入一个不完全表征、甚至不可信的测量装置；
- 通过观测到的输入输出统计去约束这个黑盒测量；
- 再对某个任务做认证，例如纠缠见证、随机性认证或相关资源的下界估计。

对本项目最相关的背景材料包括：

- [`ocr/pdfs2mds/PRA.95.042340.md`](../../../ocr/pdfs2mds/PRA.95.042340.md)
  这份文档对应的论文讨论了 quantum-input / MDI 场景下，如何把未知测量与观测统计写成 SDP 问题。它不是 route2 的逐项蓝图，但它提供了一个非常关键的思想基础：
  在可信输入态的前提下，可以不去完整拆解测量装置内部结构，而是直接通过输入输出统计约束整体测量。
- 同一文档中还提到另一条相关思路：
  若输入态足够强，例如达到 tomographically complete 的程度，就可以对测量设备形成足够强的约束。这正是 route2 强调“信息完备输入”的原因。
- 该文档引用的实验背景文献中，也包含 measurement-device-independent QRNG 的早期实验工作，这说明“可信输入态 + 不可信测量 + 随机性认证”本身有成熟的文献脉络。

因此，route2 在思想上继承的是：

- quantum-input 场景；
- prepare-and-measure MDI 思想；
- 用可信输入态约束未知测量；
- 用 SDP 做最坏情形下的安全认证。

这些都不是当前仓库里的临时发明。

#### 2.1.2 哪些部分是从 `SDP.md` 重新整理出的正确单设备版本

第二部分，是从当前项目已有的 [`SDP.md`](../../../docs/SDP.md) 重新整理出来的。

`SDP.md` 里写的优化变量是：

```text
\tilde{M}_{a,b,e}^{A_0B_0}
```

并且配套有：

- 与观测数据一致的约束；
- 对 Alice 输出求和、对 Bob 输出求和得到的局域 no-signalling / 边缘一致性约束；
- 局域 POVM 归一化约束；
- Eve 的概率归一化。

从“数学形式”看，它是一个自洽的 MDI 写法；但从 route2 当前想研究的物理对象看，它有一个关键问题：

- 它默认中央测量结果天然可以拆成 Alice 输出 `a` 和 Bob 输出 `b` 两部分；
- 然后再在这个拆分后的对象上施加局域结构约束。

而我们当前真正想分析的对象，是：

- 一个单一的 central detector；
- 一个整体黑盒 POVM；
- 一个单一离散输出 `c`。

如果继续沿用 `SDP.md` 的 `(a,b)` 分解，再套 `C3/C4` 一类约束，就会把“单设备 central measurement”错误地改写成“带局域边缘结构的联合测量”，这正是旧 `mdi_qrng` 里最容易过强约束的地方。

因此，route2 从 `SDP.md` 做的核心重整是：

1. 保留“可信输入 + 观测统计 + Eve 最优猜测 + SDP 认证”这一主框架；
2. 去掉对 central measurement 的局域拆分；
3. 把 `(a,b)` 合并成单一 central outcome `c`；
4. 把变量改写成单设备形式的 `M_{c,e}`；
5. 把约束改写成：

```text
M_{c,e} >= 0
sum_e Tr[M_{c,e} rho_xy] = P(c|x,y)
sum_c M_{c,e} = p(e) I
sum_e p(e) = 1
```

所以，route2 不是“否定 `SDP.md` 的全部思想”，而是：

- 保留它关于 MDI 随机性认证的核心目标；
- 但把其中不适合当前 central-black-box 物理图景的局域 `(a,b)` 结构拿掉；
- 重新整理成更贴近单设备问题本身的一版 SDP。

也正因为如此，route2 更适合被理解为：

```text
从当前项目已有 MDI-SDP 写法出发，
做出的“正确单设备化”重构版本。
```

#### 2.1.3 哪些部分属于当前原型自己的建模选择

第三部分，是当前代码原型为了“先把正确思路跑通”而做的具体建模选择。

这些选择是合理的、清楚的，但不应被误解为“某篇论文已经规定必须这样做”。

当前原型里最主要的建模选择有：

- 每边使用 4 个 qubit 可信输入态：
  `|0>`, `|1>`, `|+>`, `|+i>`。
  这样做的目的是用一个尽量小、但足够强的局域输入集，生成 16 个 product inputs。
- 采用当前 4 维有效联合空间作为 baseline 工作空间。
  这是为了先做一个干净、可控、容易数值验证的最小模型。
- baseline 中选用 [`fourier_povm_4d()`](./mdi_single_device.py) 作为 4 输出 POVM。
  这是一种方便分析、方便做基线数值实验的测量模型，不代表实验上只能实现这一种 POVM。
- 高输出搜索中使用 [`random_frame_povm()`](./mdi_single_device.py)。
  这一步的作用是回答“在当前单设备 MDI 框架下，超过 2 bit 是否存在清晰数值路径”，它首先是一个数值探索工具，而不是已经完成实验映射的最终测量方案。
- 目标输入的扫描方式、`raw_best_target` 与 `certified_best_target` 的区分，也是当前原型的工程设计。
  这么做是为了避免“看起来最随机的输入”和“真正认证最优的输入”被混为一谈。

因此，当前 route2 应当被准确理解为：

- 文献思想上有根；
- 从 `SDP.md` 的旧结构中做了关键的单设备重整；
- 再用一组明确、可运行、可解释的原型建模选择，把路线先落成了代码。

#### 2.1.4 最简短的结论

如果要用一句最不容易引起误解的话来总结 route2 的出处，可以这样说：

```text
route2 不是对某一篇论文的逐字复现，
而是基于 MDI / quantum-input 文献背景，
结合当前项目 `SDP.md` 与旧 `mdi_qrng` 的问题，
重新整理并实现的一版“正确单设备化”原型。
```

## 3. 实验原理

### 3.1 实验装置在物理上长什么样

可以把它想成三块：

1. Alice 端的可信态制备模块。
2. Bob 端的可信态制备模块。
3. 中央联合探测器。

实验每一轮做的事情是：

1. Alice 从一个有限集合里选一个输入态 `rho_x`。
2. Bob 从一个有限集合里选一个输入态 `rho_y`。
3. 两边把态同时送入中央探测器。
4. 中央探测器输出一个离散结果 `c`。
5. 重复很多轮后，统计得到 `P(c|x,y)`。

在这条路线里，我们不要求知道中央探测器内部到底做了什么，只把它视为一个黑盒测量。

### 3.2 为什么这叫 prepare-and-measure MDI

这里的 `prepare-and-measure` 指的是：

- 我们可信的是输入态的“制备”；
- 不可信的是最后的“测量”。

这里的 `MDI` 指的是：

- 随机性认证不依赖于对中央测量器件的完整标定；
- 只要输入态是可信、已知、可校准的，就可以从统计上反推测量黑盒到底有多“不可被 Eve 预先分解”。

与通常“测量设备无关”的口号相比，更准确的说法是：

- 探测器不需要被完全表征；
- 但输入源必须是可信的；
- 安全分析依赖的是“可信输入 + 不可信测量”的组合结构。

### 3.3 为什么这里不再使用旧的 `C3/C4`

旧 `mdi_qrng` 的关键问题是：

- 它把中央探测器的输出再拆成了某种“局域”结构；
- 然后对这些局域有效 POVM 强加了 `C3/C4` 一类 no-signalling / 边缘一致性约束；
- 这在某些场景下会显著缩小可行域，导致结果虽然看上去“物理直觉上顺”，但实际上过于保守，甚至已经不对应真正的单黑盒中央测量模型。

Route 2 的修正思想是：

- 直接把中央探测器看作一个整体 POVM；
- 不再把它拆成 Alice/Bob 两个子测量；
- 因此也不再套不必要的 `C3/C4`。

这就是为什么我们说 route2 是“更正确的 prepare-and-measure MDI”。

## 4. 本路线的数学模型

### 4.1 观测量是什么

实验真正能观测到的是：

```text
P(c|x,y)
```

这里：

- `x` 是 Alice 选的可信输入态编号；
- `y` 是 Bob 选的可信输入态编号；
- `c` 是中央探测器的离散输出。

### 4.2 为什么需要信息完备输入

如果输入态集合太弱，那么即使你测到了很多统计，中央黑盒测量仍然可能有大量自由度没有被约束到。

这会导致：

- Eve 还能“藏”很多等价实现；
- SDP 只能给出较低的保守认证值；
- 你明明 raw 统计看起来很好，但 certified `H_min` 上不去。

所以 route2 的关键原则是：

- 输入态集合尽量要信息完备，或者至少足够接近信息完备。

当前 baseline 的做法是：

- 每边 4 个 qubit 输入态；
- 共 16 个 product inputs；
- 它们在 4 维联合输入空间上张成完整算符空间。

这正是 route2 能接近 `2 bit` 的核心原因之一。

### 4.3 单设备 SDP 在优化什么

我们用的单设备 SDP 变量是：

```text
M_{c,e}
```

可以把它理解成：

- `c` 是中央测量对实验者报告的输出；
- `e` 是 Eve 私下掌握的“猜测标签”。

优化问题是：

```text
M_{c,e} >= 0
sum_e Tr[M_{c,e} rho_xy] = P(c|x,y)
sum_c M_{c,e} = p(e) I
sum_e p(e) = 1
```

然后最大化：

```text
sum_c Tr[M_{c,c} rho_x*y*]
```

这里的物理意义是：

- 第一行要求每个有效测量算符都必须是正的；
- 第二行要求它们必须重现实验观测到的统计；
- 第三行是 no-signalling / completeness 约束，保证 Eve 的边缘分解是一个合法模型；
- 目标函数则是在问：如果 Eve 拥有最优的隐藏关联，她对某个目标输入的输出最多能猜到多准。

最后：

```text
H_min = -log2(p_guess)
```

这就是认证随机性的定义。

## 5. 当前代码中的物理对象分别是什么

### 5.1 可信输入态

当前 baseline 里，每边使用 4 个 qubit 输入态：

- `|0>`
- `|1>`
- `|+>`
- `|+i>`

代码在 [`mdi_single_device.py`](./mdi_single_device.py) 里的 [`local_ic_qubit_states()`](./mdi_single_device.py) 生成这些态。

它们的优点是：

- 结构简单；
- 线性独立；
- 足以在 qubit 空间里形成一组很干净的可信输入集合；
- 组合成 16 个 product inputs 后，刚好适合做联合输入空间上的单设备 MDI 认证。

### 5.2 联合输入

Alice 和 Bob 的局域输入态做张量积后得到：

```text
rho_xy = rho_x^A ⊗ rho_y^B
```

代码由 [`product_input_states()`](./mdi_single_device.py) 生成。

实验上可以理解成：

- 两边各自准备一个已知态；
- 同时送入中央探测器；
- 中央探测器只看到这个联合输入，不需要知道内部标签。

### 5.3 中央测量

当前代码里有两种中央测量模型。

第一种是 baseline：

- 使用一个 4 输出的 4 维 extremal POVM。

代码里的 [`fourier_povm_4d()`](./mdi_single_device.py) 就是这个基线模型。

第二种是搜索模型：

- 通过随机 frame 构造更高输出数的 rank-1 POVM；
- 用于探索在更高输出下是否可能超过 `2 bit`。

代码里的 [`random_frame_povm()`](./mdi_single_device.py) 就是做这个的。

需要特别强调：

- `random_frame_povm()` 目前是一个“数值探索工具”；
- 它不是一个已经对应到具体光路和具体探测器设计的实验方案；
- 但它能回答一个很重要的问题：
  在这个 prepare-and-measure MDI 框架下，超过 `2 bit` 在原理上是否存在清晰数值路径。

答案是：存在。

## 6. 代码结构

### 6.1 核心文件

- [`mdi_single_device.py`](./mdi_single_device.py)
  核心模型、统计计算、SDP 和结果整理都在这里。
- [`main.py`](./main.py)
  Route 2 的独立命令行入口。
- [`__main__.py`](./__main__.py)
  允许用 `python -m qrng_routes.route2` 直接运行。

### 6.2 核心函数说明

- `local_ic_qubit_states()`
  生成每边 4 个可信输入态。
- `product_input_states(...)`
  生成全部联合输入态和 `(x,y)` 标签。
- `fourier_povm_4d()`
  baseline 4 输出 POVM。
- `random_frame_povm(...)`
  高输出数随机 POVM 搜索器。
- `measurement_probabilities(...)`
  从输入态和 POVM 直接计算 `P(c|x,y)`。
- `certify_target_inputs(...)`
  对一个或多个目标输入做完整 SDP 认证。
- `run_route2(...)`
  baseline 入口。
- `search_route2_high_entropy(...)`
  高输出搜索入口。

## 7. 当前数值结果到底说明了什么

### 7.1 baseline: 已经几乎到 2 bit

当前 baseline 的结果是：

- 4 输出
- 16 个 product inputs
- 认证最优目标输入是 `(1,1)`
- `H_min ≈ 1.998816`

这基本可以理解为：

- 在一个非常干净的最小模型里，route2 已经打到了 `2 bit` 天花板附近。

这件事的意义很大，因为它说明：

- 这条路线不是“也许有希望”；
- 而是已经在一个明确、可解释、可信的模型里，几乎把 `2 bit` 做出来了。

### 7.2 raw 最优和 certified 最优并不总一样

这是这次修代码之后一个非常重要的发现。

如果你只按原始观测统计的最大输出来排，很容易得到一个 `raw_best_target`。
但真正送进 SDP 之后，最优认证目标输入可能是另外一个。

在当前 baseline 里就是这样：

- raw 最优是 `(3,0)`
- 认证最优是 `(1,1)`

这也是为什么代码现在默认会输出：

- `raw_best_target`
- `certified_best_target`
- `target_scan`

对外交流时可以把它解释成：

“看上去最随机的输入，不一定就是在安全模型下真正最随机的输入。”

### 7.3 高输出搜索: 已经明确超过 2 bit

在当前保存下来的高输出 POVM 搜索结果里，我们已经看到例如：

- `8` 输出时，`H_min ≈ 2.5390`
- `12` 输出时，`H_min ≈ 2.9172`
- `16` 输出时，`H_min ≈ 3.2170`

这说明：

- `H_min > 2` 在 route2 框架下不是理论幻想；
- 而且它不是“勉强刚过 2”，而是已经出现了相当明显的超出空间；
- 真正的问题已经从“能不能超过 2”转变成了“实验上怎么把这种高输出中央测量做出来并稳定复现”。

这也是 route2 最有吸引力的地方。

## 8. 对实验室最重要的现实判断

### 8.1 什么是容易推进的

对实验室来说，route2 里相对容易推进的部分是：

- 可信输入态制备；
- 数据记录 `P(c|x,y)`；
- 单设备 SDP 分析；
- 4 输出 baseline 的概念验证。

原因是：

- 输入源的校准是实验上常见问题；
- 16 个 product inputs 的规模不大；
- baseline 中央 POVM 虽然仍是黑盒，但目标结构相对简单。

### 8.2 什么是真正的难点

真正的工程难点几乎都集中在中央探测器上。

如果实验室目标只是：

- “先稳稳做到接近 `2 bit`”

那么 4 输出 baseline 已经很有价值。

但如果目标变成：

- “要稳定显著超过 `2 bit`”

那就必须把中央测量做成更高输出数的有效黑盒 POVM。

这会带来几类困难：

- 如何在实验上设计高输出联合测量结构；
- 如何稳定校准和长时间维持这些输出通道；
- 输出数增加后，统计量和漂移控制难度都会上升；
- 以后如果要加入有限样本分析，数据量需求会更大。

### 8.3 所以 route2 到底“好不好推进”

最实在的回答是：

- 它不是最轻松的一条；
- 但它是目前唯一明显值得为 `H_min >= 2` 投入工程资源的一条。

换句话说：

- 如果实验室的硬目标真的是 `2 bit` 甚至更高；
- 那 route2 值得做，而且应该优先做；
- 只是应当分阶段推进，而不是一开始就追求复杂高输出装置。

## 9. 建议的推进策略

### 阶段 1: 先把 4 输出 baseline 做扎实

阶段目标：

- 重现实验统计；
- 验证 16 个输入的完整数据流；
- 跑通单设备 SDP；
- 稳定做到接近 `2 bit`。

这一阶段的价值是：

- 它给整个方案建立了可信基线；
- 一旦 baseline 都做不稳，再谈高输出就没有意义。

### 阶段 2: 再做高输出中央测量

阶段目标：

- 把中央测量从 4 输出拓展到 8 输出甚至更高；
- 尝试稳定实现 `H_min > 2`。

这一阶段的关键不再是理论是否可能，而是：

- 中央测量的具体实验实现是否足够好。

## 10. 命令行使用

先激活环境：

```powershell
conda activate DLEnv
$env:PYTHONPATH='D:\Codes\Quantum\Randomness-certification\src\python'
```

查看帮助：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route2 --help
```

### 10.1 baseline

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route2 `
  --mode baseline `
  --solver MOSEK
```

把结果写到文件：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route2 `
  --mode baseline `
  --solver MOSEK `
  --output-json output/qrng_routes/route2_baseline_fullscan.json
```

### 10.2 高输出搜索

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route2 `
  --mode high-output-search `
  --num-outputs 8 `
  --num-trials 20 `
  --seed 7 `
  --solver MOSEK `
  --output-json output/qrng_routes/route2_high_output_search.json
```

### 10.3 统一入口

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.main `
  --mode route2 `
  --route2-mode baseline `
  --route2-max-inputs 16 `
  --solver MOSEK `
  --output-json output/qrng_routes/route2_baseline_fullscan.json
```

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.main `
  --mode route2 `
  --route2-mode high-output-search `
  --num-outputs 8 `
  --num-trials 20 `
  --seed 7 `
  --solver MOSEK `
  --output-json output/qrng_routes/route2_high_output_search.json
```

## 11. 输出字段怎么解释

常见字段有：

- `p_guess`
  Eve 在最优攻击下的最大猜中概率。
- `H_min`
  认证最小熵，等于 `-log2(p_guess)`。
- `target_input`
  最终认证最优的目标输入 `(x,y)`。
- `raw_H_min`
  该目标输入在不做 SDP、只看原始观测最大输出概率时得到的原始熵。
- `raw_best_target`
  只按 raw 统计排序时最优的输入。
- `certified_best_target`
  真正经过安全认证后最优的输入。
- `num_inputs_certified`
  本次一共认证了多少个目标输入。
- `target_scan`
  所有目标输入各自的认证结果列表。

高输出搜索还会返回：

- `selected_trial_index`
  被选中做完整认证的随机 POVM trial 编号。
- `selection_strategy`
  当前试验筛选逻辑说明。

## 12. 应该怎样给老师讲这条路线

如果面对不太熟悉量子信息细节的老师，可以用下面这套说法。

### 12.1 先讲物理直觉

“我们信任的是输入，不信任的是测量。也就是说，我们知道自己送进去了什么态，但不假定中央探测器真的按我们想象那样工作。最终只根据输入和输出统计，来判断这个黑盒到底能不能被 Eve 预先分解。”

### 12.2 再讲为什么 route2 比旧方法更合理

“旧代码的问题，是把中央黑盒拆得过细了，附加了本不该有的结构约束，所以有时会过度保守。Route 2 直接把中央探测器当成一个整体 POVM，这样更符合真正的实验图景。”

### 12.3 最后讲结果的意义

“在这套更正确的模型下，4 输出 baseline 已经几乎做到 2 bit，而高输出数的数值搜索已经明显超过 2 bit。这说明 route2 不是抽象上更漂亮，而是它确实更有希望成为实验室真正冲击高认证熵的主线方案。”

## 13. 一段最小 Python 示例

```python
from qrng_routes.route2 import run_route2, search_route2_high_entropy

baseline = run_route2(preferred_solver="MOSEK")
print("baseline H_min =", baseline["H_min"])
print("certified best target =", baseline["certified_best_target"])

search = search_route2_high_entropy(
    num_outputs=8,
    num_trials=20,
    preferred_solver="MOSEK",
    seed=7,
)
print("high-output H_min =", search["H_min"])
print("certified best target =", search["certified_best_target"])
```

## 14. 当前最简短的结论

如果只保留一句话，那么 route2 的结论就是：

它是目前几条路线里唯一一条既在安全模型上站得住、又已经在数值上明确看到 `H_min >= 2` 甚至 `> 2` 路径的主线方案。
