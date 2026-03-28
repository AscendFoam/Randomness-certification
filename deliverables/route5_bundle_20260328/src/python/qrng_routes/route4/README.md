# Route 4: 基于未表征 APD 的 phase-insensitive QRNG 路线说明

## 1. 这条路线能不能算“第 4 条路线”

可以，而且从现在开始更适合明确地把它当作“路线 4”来管理。

这条路线对应的不是之前 route1-route3 那种 steering 或正确单设备 MDI 的主线思路，而是另一类更贴近现有 APD 数据与装置假设的方案：

- 测量端不是完全任意的黑盒，而是假设为 `phase-insensitive`。
- 在这一假设下，POVM 元素在 Fock 基下对角。
- 因此原本的一般 SDP 会退化成只含对角元素的 primal SDP / dual LP。
- 脚本直接吃实验侧给出的 `Probability.mat`，所以它和“现有 APD 数据能支持什么认证”之间的联系比 route2、route3 更直接。

从项目管理角度看，把它单独列为 route4 有三个好处：

1. 它的物理假设和 route2、route3 明显不同。
2. 它的数值问题来源也不同，主要是粗粒化与零概率病理，不是 `C3/C4` 过强约束那一类问题。
3. 它更像“基于现有 APD 平台的专门路线”，而不是 route2/route3 的简单变体。

## 2. 物理图像与安全分析含义

这条路线最接近 [QRNG_with_uncharacterized_APD (1).pdf](/d:/Codes/Quantum/Randomness-certification/docs/QRNG_with_uncharacterized_APD%20(1).pdf) 里的建模。

它的基本图像是：

1. 可信端准备若干个相干态输入，这里用平均光子数 `mu` 来标记。
2. 探测器是一个未完全表征的 APD，但假设它对相位不敏感。
3. 于是测量 POVM 在 Fock 基下可写成对角形式。
4. 实验实际给出的不是 POVM 本身，而是不同输入 `mu` 下的统计分布 `P(y|x)`。
5. 通过 primal/dual 优化，计算 Eve 最优猜测概率 `p_guess`，再得到
   `H_min = -log2(p_guess)`。

和 route2 的关系可以这样理解：

- route2 是“更干净、更标准”的正确 prepare-and-measure 单设备 MDI 主线。
- route4 是“如果实验室现在手里主要是 APD 统计数据，而且愿意接受 phase-insensitive 对角 POVM 假设，那么这条线最多能做到什么”的专门分析路线。

所以 route4 的价值不是取代 route2，而是回答：

“如果尽量贴着现有 APD 数据和硬件条件走，这条路到底能不能冲到高熵？”

## 3. 代码结构

本目录的核心文件是 [phaseinsensitive.py](./phaseinsensitive.py)。

其中最重要的函数有：

- `prepare_phaseinsensitive_instance(...)`
  统一完成数据载入、`256 -> N` 粗粒化、相干态 Fock 对角分布构造、零概率诊断和可选正则化。
- `solve_phaseinsensitive_dual(...)`
  路线 4 的主力求解器。大多数扫描都应该优先用 dual。
- `solve_phaseinsensitive_primal(...)`
  用于小规模点位的交叉验证。它很快会变得非常大，所以不适合大范围扫描。
- `run_route4_dual(...)`
  最常用的单点入口。
- `run_route4_primal(...)`
  单点 primal 入口。
- `compare_route4_primal_dual(...)`
  在小规模参数上同时跑 primal/dual，看两者是否一致。
- `sweep_route4_outputs(...)`
  固定输入集，扫描粗粒化输出数 `N`。
- `search_route4_triplets(...)`
  先用仅基于分布的快速上界筛选不同 `mu` 组合，再决定是否对最好的若干组做正式 dual 认证。

命令行入口在 [main.py](./main.py)。

此外，原来的两个历史脚本仍然保留为兼容包装层：

- [guessprobprimal_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py)
- [guessprobdual_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py)

它们现在不再各自维护一份独立逻辑，而是统一转到 route4 的实现上，避免 primal/dual 再次出现数据预处理不一致。

## 4. 这次修复到底修了什么

### 4.1 修复 1：`256 -> N` 粗粒化必须完整覆盖

原脚本的核心问题之一是：

```python
block_size = round(256 / N)
```

然后再按固定宽度切块。

这会在 `N` 不能整除 `256` 时漏掉或挤压一部分原始 bin。例如 `N=5` 时原脚本只覆盖了 `255` 个原始 bin，所以 row sum 会失真。这样 primal 会因为归一化不一致直接坏掉。

现在改成了完整覆盖的整除分块：

```python
edges[k] = floor(k * 256 / N)
```

这样每个原始 bin 都会被恰好分到一个 coarse bin，所有 coarse bin 连续、无重叠、无遗漏。

这也是为什么修复后：

- `N=5` 的 exact primal / dual 重新变成了正常可解；
- 说明此前 `N=5` 的失败并不是模型本身坏掉，而是数据粗粒化本身就做错了。

### 4.2 修复 2：primal 和 dual 必须吃同一个问题

原脚本还有第二个根本问题：

- primal 直接使用原始 `p(x,y)`；
- dual 却偷偷对 `p(x,y)` 做了 `P_FLOOR = 1e-12` 的正则化。

这意味着两边解的根本不是同一个优化问题。

现在 route4 把这件事统一到了 `prepare_phaseinsensitive_instance(...)`：

- `prob_floor > 0` 时，primal/dual 都用同一个正则化后的统计；
- `prob_floor <= 0` 时，两者都用 exact 统计。

这样：

- small-scale primal/dual 对比才有意义；
- 也能明确区分“exact 问题真的病态”和“regularized 邻近问题可稳定求解”这两件事。

### 4.3 修复 3：把 mixed-zero pathology 变成显式诊断

现在每次运行都会输出：

- `mixed_zero_columns_raw`
- `all_zero_columns_raw`
- `has_mixed_zero_column_pathology`

这比旧脚本更重要，因为 route4 在 `N >= 6` 的真正困难，很多时候正是来自：

- 同一个输出列对某些输入严格为 0；
- 对另一些输入又严格大于 0。

在 phase-insensitive、Fock 对角、且相干态对角元严格正的设定下，这会把 exact primal/dual 推向 infeasible / unbounded 的病态边界。

## 5. 当前关键数值结果

下面列出修复后最有代表性的结果。这里优先给实验判断最重要的点。

### 5.1 修复后的 exact 一致性检查

1. `N=4`，默认输入集 `[100,120,140]`、`q=[0.25,0.25,0.5]`、`prob_floor=1e-12`
   primal 与 dual 一致给出 `p_guess ≈ 0.89437`，`H_min ≈ 0.161 bit`。
2. `N=5`，同一输入集、`prob_floor=0`
   exact primal 与 exact dual 一致给出 `p_guess ≈ 0.8697534`，`H_min ≈ 0.2013 bit`。

这件事很重要，因为它直接说明：

- 原来 `N=5` 的失败确实主要是错误粗粒化导致的；
- 修复后 `N=5` 已经恢复正常。

### 5.2 `N=6` 的 exact 问题为什么还是坏

修复后如果仍然坚持 exact 统计，即 `prob_floor=0`，在默认输入集下：

- `N=6` 的 exact dual 返回 `unbounded`；
- 其原始数据中出现 `mixed_zero_columns_raw = [1]`；
- 同时还存在 `all_zero_columns_raw = [0]`。

这说明修复后 `N=6` 之所以还坏，不再是粗粒化 bug，而是原始实验统计本身在该 coarse-graining 下出现了 exact 零概率病理。

### 5.3 regularized 后的认证值

使用 `prob_floor = 1e-12` 后，默认输入集 `[100,120,140]` 下：

| `N` | certified `H_min` | 备注 |
|---|---:|---|
| 4 | 0.161 | primal/dual 一致 |
| 6 | 0.332 | dual |
| 8 | 0.343 | dual |

这组结果说明：

- 修复后脚本可以稳定给出结果；
- 但认证值依然很低；
- 单纯把 `N` 从 4 提到 8，并没有把 route4 推向高熵区。

### 5.4 更优 `mu` 子集的效果

用分布上界先筛选时，`[120,140,160]` 是比较好的三态子集之一。

但即使在这个更优子集上：

- `N=8` 的 certified 结果也只有 `H_min ≈ 0.394 bit`。

这说明：

- route4 不是“默认三态选得不好”这么简单；
- 其瓶颈更深，来自实验统计本身和模型结构。

### 5.5 `N=16` 时的快速上界判断

对所有 `9 choose 3 = 84` 个三态子集做分布层面的快速筛选后，`N=16` 的最好三态组合是：

- `[120,140,160]`

它的分布上界只有：

- `distribution_only_H_min ≈ 1.333 bit`

同时，单态情形下最好的分布上界也只有：

- `H_min ≈ 1.445 bit`，对应 `mu = 160`

注意这还只是“只看输出分布”的上界，不是正式认证值。正式认证只会更低，不会更高。

因此从 route4 当前数据看，离 `H_min >= 2` 还差得非常远。

## 6. route4 在实验上好不好推进

如果只问“实验上能不能推进”，答案是：

- 能推进。
- 而且它可能是四条路线里最容易直接贴着现有 APD 数据往前走的一条。

原因很现实：

1. 它直接使用已经存在的 `Probability.mat`。
2. 它假设的测量模型与“未完全表征但 phase-insensitive 的 APD”高度贴近。
3. 不需要像 route2 那样立刻去设计新的高输出中央 POVM 结构。

所以如果实验室的目标是：

- 快速评估“现有 APD 平台在这个安全模型下大概能认证多少随机性”，

那 route4 很适合作为诊断路线。

## 7. route4 的工程困难与风险

### 7.1 最大风险不是数值不稳定，而是“数据先天不够强”

修复后的 route4 已经能把代码层面的 bug 和数值病理分开。

现在更明显的事实是：

- 即使只看分布上界，很多参数点也远不到 `2 bit`；
- 正式认证后还会进一步下降。

这意味着 route4 的主风险并不是“再调一下求解器就好了”，而是：

- 当前 APD 统计本身给出的可认证随机性就不高。

### 7.2 exact 零概率病理会持续困扰大 `N`

当 `N` 增大后，更细的 coarse-graining 更容易产生：

- 某些输入下该 bin 恰好为 0；
- 某些输入下同一 bin 又非零。

这会导致 exact primal / dual 在 phase-insensitive 模型下继续出现病态。

从工程角度看，这意味着：

- 如果实验室希望 route4 成为长期主线，就必须认真设计更平滑、更不稀疏的输出统计；
- 否则每次把 `N` 提高，都会更频繁撞上 exact zero 的边界问题。

### 7.3 正则化能让脚本稳定，但不能把路线变成高熵路线

`prob_floor` 的作用是：

- 让 primal/dual 去求一个稳定的邻近问题；
- 方便比较不同参数点；
- 避免 exact zeros 直接把优化问题送进不可行或无界。

但它不能改变一个更根本的事实：

- 如果原始输出分布本身已经不够平，或者认证约束下 Eve 自由度仍然很大，
- 那么 route4 的 `H_min` 就不会突然因为正则化而冲到 2。

## 8. 这条路线对实验室到底意味着什么

如果实验室要一个很明确的判断，可以概括成三句话：

1. route4 值得保留，因为它是“现有 APD 平台在当前安全假设下到底能做到什么”的直接诊断工具。
2. route4 不适合当作当前冲击 `H_min >= 2` 的主线，因为修复后的结果仍然明显偏低。
3. 如果实验室把 `H_min >= 2` 当成硬指标，主投入应继续放在 route2，route4 更适合作为对照线和平台体检线。

## 9. 推荐的使用方式

推荐环境：

```powershell
conda activate DLEnv
$env:PYTHONPATH='D:\Codes\Quantum\Randomness-certification\src\python'
```

### 9.1 单点 dual

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route4 `
  --mode dual-single `
  --num-outputs 6 `
  --selected-mu 100 120 140 `
  --q-values 0.25 0.25 0.5 `
  --prob-floor 1e-12 `
  --solver MOSEK
```

### 9.2 小规模 primal/dual 交叉验证

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route4 `
  --mode primal-dual-compare `
  --num-outputs 5 `
  --selected-mu 100 120 140 `
  --q-values 0.25 0.25 0.5 `
  --prob-floor 0 `
  --solver MOSEK
```

### 9.3 扫描不同 coarse-graining 输出数

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route4 `
  --mode output-sweep `
  --output-values 4 6 8 `
  --selected-mu 100 120 140 `
  --q-values 0.25 0.25 0.5 `
  --prob-floor 1e-12 `
  --solver MOSEK
```

### 9.4 先做三态子集筛选

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route4 `
  --mode subset-search `
  --num-outputs 16 `
  --subset-size 3 `
  --certify-top-k 0 `
  --solver MOSEK
```

### 9.5 使用旧脚本名兼容运行

旧脚本名仍然可用，例如：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' src/python/guessprobdual_phaseinsensitive.py `
  --num-outputs 6 `
  --selected-mu 100 120 140 `
  --q-values 0.25 0.25 0.5 `
  --prob-floor 1e-12 `
  --solver MOSEK
```

## 10. 最重要的结论

如果只保留一句话，那么 route4 的结论就是：

它完全可以作为“第 4 条路线”，而且修复后已经能稳定回答现有 APD 数据在 phase-insensitive 假设下的可认证随机性；但从当前结果看，它更像一条有现实诊断价值的实验评估路线，而不是一条有希望直接冲到 `H_min >= 2` 的高熵主线。
