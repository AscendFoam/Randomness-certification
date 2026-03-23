# 路线 4 评估报告：phase-insensitive APD 随机性认证方案

## 1. 结论先行

这两个脚本 [guessprobprimal_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py) 和 [guessprobdual_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py) 可以作为“第 4 条路线”来看待。

但对实验室最关键的结论不是“能不能算 route4”，而是：

1. 修复后，这条路线已经能稳定回答当前 APD 数据在 phase-insensitive 假设下的认证结果。
2. 这条路线在实验上是可推进的，而且和现有 APD 数据平台贴合度很高。
3. 但从当前数据与认证结果看，它并不适合作为冲击 `H_min >= 2` 的主线。
4. 如果实验室把 `H_min >= 2` 当成硬指标，主投入仍应优先放在 route2；route4 更适合作为平台诊断线、对照线和快速评估线。

## 2. 这条路线的物理含义

route4 对应的是一种比 route2 更贴近“现有 APD 平台”的建模：

- 输入端用可信相干态，当前代码用不同平均光子数 `mu` 作为测试输入。
- 测量端不是完全任意黑盒，而是假设 APD 是 `phase-insensitive`。
- 这意味着测量 POVM 在 Fock 基下是对角的。
- 实验真正给出的不是 POVM 本身，而是不同输入下的统计分布 `P(y|x)`。
- 再通过 primal SDP / dual LP 计算 Eve 的最优猜测概率。

这个模型和 [QRNG_with_uncharacterized_APD (1).pdf](/d:/Codes/Quantum/Randomness-certification/docs/QRNG_with_uncharacterized_APD%20(1).pdf) 中的思路是一致的。

因此 route4 的定位应当是：

- 不是“正确单设备 MDI 高熵主线”；
- 而是“基于未完全表征 APD、在 phase-insensitive 假设下评估现有平台能认证多少随机性”的专门路线。

## 3. 这次修复后的真正变化

这次修复不是简单调 solver，而是把两个真正问题拆开并分别处理了。

### 3.1 原问题 1：`256 -> N` 的 coarse-graining 写错了

旧脚本使用：

```python
block_size = round(256 / N)
```

然后固定宽度切块。

这会导致：

- 当 `N` 不能整除 `256` 时，某些原始 bin 被漏掉或重复压缩；
- `p(x,·)` 的 row sum 不再严格一致；
- primal 会因为精确等式约束直接变得不可行。

修复后改成了完整覆盖分块：

```text
edge_k = floor(k * 256 / N)
```

这样每个原始 bin 都被恰好覆盖一次。

修复后的直接结果是：

- `N=5` 的 exact primal / dual 重新变成一致可解；
- 说明旧脚本在 `N=5` 的失败主要不是物理模型坏了，而是预处理就做错了。

### 3.2 原问题 2：primal 和 dual 原本解的不是同一个问题

旧脚本里：

- primal 用 exact `p(x,y)`；
- dual 却对 `p(x,y)` 加了 `P_FLOOR = 1e-12`。

这意味着二者从根本上不再可比。

修复后统一到 route4 公共预处理模块：

- `prob_floor > 0` 时，primal / dual 一起解 regularized 问题；
- `prob_floor <= 0` 时，primal / dual 一起解 exact 问题。

这使得 route4 现在可以真正做 small-scale primal/dual 一致性检查。

### 3.3 `N >= 6` 为什么还会坏

修复后，`N >= 6` 的问题并没有彻底消失，但现在它的原因已经很清楚了：

- 不是 `N >= 6` 这个数字本身让模型失效；
- 而是更细 coarse-graining 后，实验统计里出现了 `mixed zero columns`。

也就是：

- 某个输出列对某些输入严格为 0；
- 对另一些输入又严格大于 0。

在 phase-insensitive、Fock 对角、且相干态对角元严格正的模型下，这会把 exact primal / dual 推到病态边界：

- primal 倾向于 infeasible；
- dual 倾向于 unbounded。

所以 route4 现在的理解是：

- `N=5` 的旧失败主要是代码 bug；
- `N>=6` 的 exact 失稳主要是原始实验统计在该 coarse-graining 下真的存在结构性零概率病理。

## 4. 修复后得到的关键结果

以下结果都在 `DLEnv` 里重新运行得到。

### 4.1 一致性验证

在默认输入集 `[100, 120, 140]`、`q=[0.25,0.25,0.5]` 下：

| 设置 | 结果 |
|---|---|
| `N=4`, `prob_floor=1e-12` | primal/dual 一致，`H_min ≈ 0.161 bit` |
| `N=5`, `prob_floor=0` | primal/dual 一致，`H_min ≈ 0.201 bit` |

这说明 route4 的修复是有效的，尤其说明旧脚本在 `N=5` 的坏结果不是可信物理结论。

### 4.2 exact 问题的病态点

同样在默认输入集下：

| 设置 | 结果 |
|---|---|
| `N=6`, `prob_floor=0` | exact dual `unbounded` |

这时结果里会明确显示：

- `mixed_zero_columns_raw = [1]`
- `all_zero_columns_raw = [0]`

这就是 exact 问题不稳定的底层原因。

### 4.3 regularized 后的正式认证值

把 `prob_floor = 1e-12` 打开后，默认输入集下得到：

| `N` | certified `H_min` |
|---|---:|
| 4 | 0.161 |
| 6 | 0.332 |
| 8 | 0.343 |

这说明：

- route4 修复后可以稳定跑；
- 但认证值依然明显偏低；
- 单纯提高 coarse-graining 输出数，并没有把路线推向高熵区。

### 4.4 更优输入子集的效果

先用分布层面的快速筛选后，`[120, 140, 160]` 是较优三态子集之一。

但即使在这个更优子集上：

- `N=8` 的 certified `H_min` 也只有 `≈ 0.394 bit`。

这说明路线 4 的瓶颈不只是“默认三态没选好”，而是更深层的数据与模型瓶颈。

### 4.5 大 `N` 下的快速上界判断

对 `N=16` 做全部 `84` 个三态子集的快速分布上界筛选后：

- 最好的三态子集是 `[120, 140, 160]`
- 它的 `distribution-only H_min` 也只有 `≈ 1.333 bit`

而单态情形下最好的分布上界也只有：

- `mu = 160` 时 `H_min ≈ 1.445 bit`

注意这里还只是“只看输出分布”的快速上界，不是正式认证值。正式认证一定不会更高。

因此：

- route4 目前离 `H_min >= 2` 还很远；
- 而且这个差距在正式认证后只会进一步放大。

## 5. route4 在实验上是否可推进

如果问题是“实验上能不能推进”，答案是可以。

而且从硬件连续性上讲，route4 可能是四条路线里最容易贴着现有 APD 平台继续推进的一条。原因有三点：

1. 它直接使用已有实验统计 `Probability.mat`。
2. 它的物理假设正是“未完全表征但 phase-insensitive 的 APD”。
3. 不需要像 route2 那样马上去设计新的高输出中央 POVM 架构。

所以 route4 很适合做下面这类工作：

- 评估现有 APD 平台在当前安全模型下最多能拿到多少认证随机性；
- 比较不同 `mu` 组合和不同 coarse-graining 的收益；
- 判断“继续在现有 APD 路线加工程资源”是否值得。

## 6. route4 为什么不适合做 `H_min >= 2` 主线

### 6.1 当前分布本身就不够平

最关键的信号不是 solver，而是分布上界：

- 即使 `N=16`，最好的三态组合的分布上界也只有 `≈ 1.333 bit`；
- 最好的单态上界也只有 `≈ 1.445 bit`。

如果分布层面的上界都到不了 `2`，正式认证更不可能到 `2`。

### 6.2 认证损失很大

在 route4 中，正式认证值通常明显低于仅看输出分布的直觉值。例如：

- 默认三态、`N=8` 时，distribution-only `H_min ≈ 0.626`；
- 但正式 certified 只有 `≈ 0.343`。

这说明 Eve 在当前模型下仍有很大可利用自由度。

### 6.3 大 `N` 会持续遇到 exact-zero 病理

随着 coarse-graining 更细，mixed-zero columns 更容易出现。

这带来两个现实问题：

1. exact 问题经常会变成 infeasible / unbounded；
2. regularization 虽然能让脚本稳定，但不会把一条低熵路线变成高熵路线。

所以 route4 的难点不是“代码还能再修一修”，而是“当前 APD 统计本身是否支持高熵认证”。

## 7. 与 route2 的关系

如果实验室的目标是“尽快用现有 APD 平台做出一套可解释的认证分析”，route4 很有价值。

但如果目标被明确写成：

```text
H_min >= 2
```

那么 route4 与 route2 的角色应该明确分开：

- route4：平台诊断线、对照线、现有 APD 数据评估线。
- route2：真正应该继续投入的高熵主线。

原因很简单：

- route4 的当前数据趋势明显不支持 `2 bit`；
- route2 已经在正确的单设备 MDI 框架下显示出接近甚至超过 `2 bit` 的数值潜力；
- 所以 route2 面对的是“如何把高输出 POVM 真实做出来”的工程问题；
- route4 面对的则更像是“当前 APD 统计天花板本来就不高”的模型与数据问题。

## 8. 对实验室的建议

建议把 route4 的地位明确成下面这样：

### 8.1 建议保留

建议保留 route4，因为它回答的是一个 route2 不能直接替代的问题：

“现有 APD 平台在 phase-insensitive 假设下到底能认证多少随机性？”

### 8.2 建议定位为诊断/对照路线

建议把 route4 用作：

- APD 平台体检；
- 输入集和 coarse-graining 的灵敏度分析；
- 与 route2 的高熵主线做对照。

### 8.3 不建议作为当前 `H_min >= 2` 主攻路线

如果实验室资源有限，而目标又明确是 `H_min >= 2`，那么不建议把 route4 作为当前主攻对象。

更合适的策略是：

1. route2 继续做高熵主线推进；
2. route4 保留为平台诊断与对照；
3. 当 route2 的硬件方案需要和现有 APD 平台做对比时，再用 route4 提供基准。

## 9. 相关文件

修复后的 route4 代码与说明位于：

- [phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route4/phaseinsensitive.py)
- [main.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route4/main.py)
- [README.md](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route4/README.md)

原脚本兼容入口：

- [guessprobprimal_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py)
- [guessprobdual_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py)

底层病理分析详见：

- [phaseinsensitive_primal_dual_analysis.md](/d:/Codes/Quantum/Randomness-certification/docs/phaseinsensitive_primal_dual_analysis.md)

## 10. 最终一句话判断

route4 可以正式视为第 4 条路线，而且修复后已经能稳定、清楚地评估现有 APD 平台在该模型下的认证能力；但从当前数值和数据上界看，它更像“可推进但低熵”的现实评估路线，而不是“有希望直接达到 `H_min >= 2`”的主线方案。
