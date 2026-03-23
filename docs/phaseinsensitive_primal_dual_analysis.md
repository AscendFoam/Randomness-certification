# `guessprobprimal_phaseinsensitive.py` / `guessprobdual_phaseinsensitive.py` 分析报告

## 1. 结论先行

对 [guessprobprimal_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py) 和 [guessprobdual_phaseinsensitive.py](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py) 的复现实验表明：

- `N >= 6` 之后“得不到有效结果”，底层真正原因不是“相位不敏感 SDP 在数学上从 `N=6` 开始失效”。
- 真正原因是，代码把从 `Probability.mat` 读出的 256-bin 概率数据，用一种会产生不一致约束的方式喂给了 primal / dual。
- 这些不一致主要来自两个问题：
  1. `round(256 / N)` 的粗粒化方式在某些 `N` 下会漏掉尾部 bin，导致每一行 `p(x,\cdot)` 的总和不一致。
  2. 当 `N` 足够大时，粗粒化后的 `p(x,y)` 会出现“某个输入态在某个 bin 上精确为 0，而另一个输入态在同一 bin 上为正”的情况；这与本模型里“相干态在 Fock 对角上处处严格为正”的假设冲突。
- 在 primal 中，这两个问题会把问题直接推成 `infeasible`。
- 在 exact dual 中，第二类问题会把问题推成 `unbounded`。
- dual 脚本通过 `P_FLOOR = 1e-12` 的正则化和重新归一化，把原本不一致的原问题改成了一个邻近的正则化问题，所以它还能给出有限数值。
- 一旦把同样的正则化也加到 primal 上，primal 和 dual 会重新一致，说明坏掉的不是 SDP 本身，而是数据预处理和精确零概率的处理方式。

因此，`N >= 6` 只是当前这组 `selected_mu_list = [100, 120, 140]`、当前 `Probability.mat`、当前 256-bin 到 `N`-bin 的压缩方式共同作用下出现的一个“病态起点”，不是协议本身的自然边界。

## 2. 代码结构和问题位置

### 2.1 Primal 脚本

primal 里的关键逻辑在以下位置：

- 读取并粗粒化实验概率：
  [guessprobprimal_phaseinsensitive.py:78](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L78)
  到
  [guessprobprimal_phaseinsensitive.py:97](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L97)
- 定义 primal 变量 `M_elements`：
  [guessprobprimal_phaseinsensitive.py:120](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L120)
  到
  [guessprobprimal_phaseinsensitive.py:124](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L124)
- 归一化约束：
  [guessprobprimal_phaseinsensitive.py:158](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L158)
  到
  [guessprobprimal_phaseinsensitive.py:168](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L168)
- 统计兼容约束：
  [guessprobprimal_phaseinsensitive.py:170](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L170)
  到
  [guessprobprimal_phaseinsensitive.py:180](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L180)

### 2.2 Dual 脚本

dual 里的关键逻辑在以下位置：

- 读取并粗粒化实验概率：
  [guessprobdual_phaseinsensitive.py:93](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L93)
  到
  [guessprobdual_phaseinsensitive.py:111](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L111)
- 概率正则化：
  [guessprobdual_phaseinsensitive.py:113](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L113)
  到
  [guessprobdual_phaseinsensitive.py:121](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L121)
- dual 变量和目标函数：
  [guessprobdual_phaseinsensitive.py:170](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L170)
  到
  [guessprobdual_phaseinsensitive.py:187](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L187)

## 3. 复现实验结果

我在 `DLEnv` 中直接复现了当前脚本对应的问题，并用同一份 `Probability.mat` 做了不同 `N` 的扫描。

### 3.1 Exact primal / exact dual / regularized primal 的表现

下面是最关键的对比：

| `N` | exact primal | exact dual | regularized primal | 说明 |
|---|---|---|---|---|
| 4 | `optimal`, `p_guess ≈ 0.8943688585` | `optimal`, `p_guess ≈ 0.8943688625` | 不需要 | primal / dual 一致 |
| 5 | `infeasible` | `optimal`, `p_guess ≈ 0.8697533464` | `optimal`, `p_guess ≈ 0.8697533528` | primal 被错误 binning 压坏；dual 仍能跑，但这时它已经不再是 exact primal 的镜像 |
| 6 | `infeasible` | `unbounded` | `optimal`, `p_guess ≈ 0.7961118329` | exact 问题已病态；正则化后 primal / dual 再次一致 |

regularized dual 在同样的 `P_FLOOR = 1e-12` 下给出的值为：

- `N = 5`: `p_guess ≈ 0.8697535288`
- `N = 6`: `p_guess ≈ 0.7961118203`

与 regularized primal 完全一致到数值误差范围内。

这说明：

- 代码里的 SDP / LP 建模本身并没有在 `N=6` 数学性崩坏。
- 真正崩坏的是“原始粗粒化概率数据 + exact zero + exact equality constraints”这一组合。

## 4. 第一个真正原因：`round(256 / N)` 会制造不守恒的 `p(x,y)`

### 4.1 问题出在哪里

当前粗粒化写法是：

```python
block_size = round(256 / N)
for k in range(N):
    idx_start = k * block_size
    idx_end = (k + 1) * block_size
    p[i, k] = np.sum(prob_256[idx_start:idx_end])
```

见：

- [guessprobprimal_phaseinsensitive.py:90](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L90)
  到
  [guessprobprimal_phaseinsensitive.py:97](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L97)
- [guessprobdual_phaseinsensitive.py:104](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L104)
  到
  [guessprobdual_phaseinsensitive.py:111](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L111)

这个写法的问题是：

- 当 `N` 不能整除 256 时，`round(256 / N)` 有时会让 `N * block_size < 256`。
- 这意味着最后若干个原始 bin 会被直接漏掉。

例如：

- `N = 5` 时，`round(256 / 5) = 51`
- `5 * 51 = 255`
- 最后 1 个原始 bin 根本没有被纳入任何新 bin

于是 `p(x,\cdot)` 的总和就不再严格等于 1。

### 4.2 为什么这会让 primal 不可行

primal 的归一化约束要求：

```text
sum_y M_y^{lambda} ∝ I
```

在代码里被写成“对每个策略 `lambda`，`sum_y M_elements[:, y, k]` 必须是常向量”：

- [guessprobprimal_phaseinsensitive.py:158](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L158)
  到
  [guessprobprimal_phaseinsensitive.py:168](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobprimal_phaseinsensitive.py#L168)

把所有策略再求和后，得到：

```text
sum_y M_total[:, y] = c * (1,1,...,1)
```

再结合统计兼容约束：

```text
rho_x · M_total[:, y] = p(x,y)
```

对 `y` 求和可得：

```text
sum_y p(x,y) = rho_x · (c * 1) = c
```

右边与 `x` 无关，所以所有输入态的行和必须完全相同。

但 `N = 5` 时，当前代码粗粒化得到的三行和是：

- `1.0`
- `1.0`
- `0.999999555`

这已经违反了 primal 约束隐含的必要条件，所以 exact primal 直接 `infeasible`。

### 4.3 为什么 dual 在 `N = 5` 还会给出有限值

这是因为当前 dual 脚本并不是在原始 `p` 上直接求 exact dual，它本身就带了预处理逻辑，而且它的约束形式也没有把 primal 的“行和一致性”显式编码成一个单独条件。

因此：

- exact primal 可以因为行和不一致而 `infeasible`
- 但当前写法下的 dual 仍然可能给出一个有限值

一旦把 `p` 用统一的正则化和归一化处理后，regularized primal 和 dual 就会重新一致。

## 5. 第二个真正原因：mixed zero columns 与相干态正支撑冲突

### 5.1 为什么 `N >= 6` 在当前数据上会出问题

随着 `N` 增大，粗粒化后的 bin 变细，尾部 bin 中出现精确零概率的机会越来越高。

在当前这组数据下，从 `N = 6` 开始，已经出现如下模式：

`N = 6` 时某一列概率为：

```text
y = 1: [2.7627e-4, 2.25e-7, 0.0]
```

也就是说：

- 对第 1 个输入态，这个 bin 的概率为正
- 对第 2 个输入态，这个 bin 的概率也为正
- 对第 3 个输入态，这个 bin 的概率恰好为 0

这类列我称为 mixed zero column。

当前数据下：

- `N = 6` 开始就有 mixed zero column
- `N = 7, 8, 9, 10` 也持续存在

### 5.2 为什么这在 primal 里会直接导致不可行

primal 里使用的是相干态的 Fock 对角分布：

```text
rho_diag[x, n] = e^{-mu_x} mu_x^n / n!
```

对当前 `M = 280`、`mu = 100, 120, 140`，这些对角元在所有 `n = 0,...,279` 上都是严格正的，只是有些非常小：

- `mu = 100` 时最小值约 `6.21e-49`
- `mu = 120` 时最小值约 `7.67e-53`
- `mu = 140` 时最小值约 `1.58e-61`

也就是说，`rho_diag[x, n] > 0` 对所有 `x,n` 都成立。

与此同时 primal 变量满足：

- `M_total[:, y] >= 0`
- `rho_diag[x, :] @ M_total[:, y] = p(x,y)`

现在如果某个 `(x,y)` 满足 `p(x,y) = 0`，那么因为：

- `rho_diag[x, n] > 0`
- `M_total[n, y] >= 0`

所以只有一种可能：

```text
M_total[:, y] = 0
```

也就是说，只要某个输入态在某个输出 bin 上是精确零概率，这个输出列对应的 POVM 元素就必须整体为零。

但如果同一个 `y` 对另一个输入态有 `p(x',y) > 0`，那又要求：

```text
rho_diag[x', :] @ M_total[:, y] > 0
```

这与 `M_total[:, y] = 0` 直接矛盾。

因此，mixed zero column 会把 exact primal 必然推成 `infeasible`。

这就是为什么对当前数据来说，`N >= 6` 开始 primal 会坏掉。

## 6. 为什么 exact dual 会在 `N >= 6` 变成 `unbounded`

如果不做 `P_FLOOR` 正则化，exact dual 在当前数据上表现为：

- `N = 5`: `optimal`
- `N = 6`: `unbounded`
- `N = 8`: `unbounded`

这其实和上面的 primal infeasibility 是一致的：

- exact primal 已经被 mixed zero columns 推成不可行
- 对应的 exact dual 就会表现成无界或不可行

当前 dual 脚本之所以还能给出有限结果，是因为它明确做了这一步：

```python
P_FLOOR = 1e-12
p = np.maximum(p, P_FLOOR)
p = p / p.sum(axis=1, keepdims=True)
```

见：

- [guessprobdual_phaseinsensitive.py:113](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L113)
  到
  [guessprobdual_phaseinsensitive.py:121](/d:/Codes/Quantum/Randomness-certification/src/python/guessprobdual_phaseinsensitive.py#L121)

这一步的本质是：

- 把所有精确零概率替换成极小正数
- 再把每一行重新归一化

于是：

- mixed zero column 被消掉
- objective 中所有 `p(x,y)` 都严格为正
- 问题重新变成一个可行、且通常有界的近似问题

因此，`guessprobdual_phaseinsensitive.py` 在 `N >= 6` 返回的不是原始 exact 问题的答案，而是“正则化后邻近问题”的答案。

## 7. 第三个原因：primal 的规模在 `N >= 6` 后迅速失控

即使把上面两个数据一致性问题都修掉，primal 在工程上仍然会很快变得难以使用。

### 7.1 变量和约束的增长规律

在 primal 中：

- 策略数 `num_strategies = N^(D+1)`
- 变量 `M_elements` 的形状是 `(M, N, num_strategies)`

因此变量总数是：

```text
M * N * N^(D+1) = M * N^(D+2)
```

当前脚本里：

- `D = 3`
- `M = 280`

所以变量数是：

```text
280 * N^5
```

对应数值为：

| `N` | primal 变量数 | primal 约束量级 |
|---|---:|---:|
| 4 | 286,720 | 71,436 |
| 5 | 875,000 | 174,390 |
| 6 | 2,177,280 | 361,602 |
| 8 | 9,175,040 | 1,142,808 |

### 7.2 实际求解时间

在同一环境下我测得：

- `N = 4` exact primal: 约 `13.9 s`
- `N = 5` regularized primal: 约 `178 s`
- `N = 6` regularized primal: 约 `438 s`

并且 CVXPY 在 canonicalization 时还给出提示：

```text
The problem has an expression with dimension greater than 2.
Defaulting to the SCIPY backend for canonicalization.
```

这说明当前 primal 实现不仅在数学上对 exact zeros 很敏感，而且在实现层面还承受了：

- 3 维 tensor 变量
- `N^(D+1)` 级别的策略枚举
- 巨大的稀疏线性系统 canonicalization 成本

所以从工程角度看，即使 regularization 之后 primal 重新可行，`N >= 6` 也已经进入“能算但很不划算”的区间。

## 8. 为什么说“真正坏掉的不是 SDP 本身”

最有力的证据是：

- `N = 5` 时，exact primal `infeasible`
- 但同样数据做一个极小的 `P_FLOOR = 1e-12` 正则化后，primal 立刻恢复为 `optimal`
- 并且 regularized primal 和 regularized dual 数值一致

`N = 6` 也是一样：

- exact primal `infeasible`
- exact dual `unbounded`
- regularized primal / dual 都变成 `optimal`
- 两边结果一致到数值误差范围内

所以底层真正结论应当是：

- 坏掉的是“原始概率数据 + 当前粗粒化 + exact zero + exact equality”的组合
- 不是 phase-insensitive primal / dual 数学模型本身突然从 `N = 6` 开始失效

## 9. 对当前代码的建议

虽然你这次只要求分析报告，但从代码维护角度，建议至少记住下面四点。

### 9.1 粗粒化必须保证完整覆盖 256 个原始 bin

不要再使用：

```python
block_size = round(256 / N)
```

更稳妥的方式是用整数边界确保：

- 每个原始 bin 恰好被分到一个新 bin
- 没有遗漏
- 没有重复

例如可以用：

```python
edges = np.linspace(0, 256, N + 1, dtype=int)
for k in range(N):
    p[i, k] = prob_256[edges[k]:edges[k+1]].sum()
```

### 9.2 primal / dual 必须共享完全相同的预处理

当前 dual 做了：

- floor regularization
- 行归一化

而 primal 没做。

这会导致两边求解的根本不是同一个问题。

### 9.3 如果继续保留 phase-insensitive 对角模型，就不能把 exact zero 直接当硬约束

在当前模型里，相干态在 Fock 对角上处处为正，因此：

- 任何 exact zero 都是非常强的结构性信息
- 很容易把可行域压空

如果这些零值更多来自有限采样、有限分辨率或数值舍入，那就更不应该直接作为 exact equality 输入求解器。

### 9.4 真正需要高 `N` 时，应优先求 dual

当前 primal 的规模增长太快。

对于当前脚本结构：

- primal 更适合作为小 `N` 校验工具
- dual 才更适合作为主工作马

不过即便用 dual，也要明确它是在解 regularized problem，而不是原始 exact problem。

## 10. 最终判断

如果把问题压缩成一句话，那么这次分析的最终判断是：

`N >= 6` 之后得不到有效结果，真正原因不是“模型天然不支持这么多输出”，而是“当前代码把 256-bin 概率做了会丢质量、会产生 mixed zeros 的粗粒化，并把这些不一致统计当成 exact equality 送进了 primal / dual；primal 因此变 infeasible，exact dual 因此变 unbounded，dual 脚本现在之所以还能出数，是因为它实际上已经悄悄改成了解一个正则化后的邻近问题”。`

从建模诊断角度看，这是一个“数据预处理和 exact-zero handling”的问题；从工程角度看，还叠加了 primal 规模爆炸的问题。
