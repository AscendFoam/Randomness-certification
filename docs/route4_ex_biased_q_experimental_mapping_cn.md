# Route4-ex 中偏置 `q` 与真实实验轮次的映射说明

## 1. 这份说明要回答什么

在 `route4-ex` 里，我们最近发现把

- `q = [1,1,1]`

改成

- `q = [2,1,1]`
- `q = [5,1,1]`
- `q = [1,0,0]`

以后，formal `H_min` 会显著上升。这里最容易引起误解的一点是：

- 代码里的 `q` 到底是不是实验里“三个输入真正出现的物理概率”；
- 如果不是，它和真实实验轮次之间该怎么对应。

这份短文档只澄清这件事。

## 2. 代码里 `q` 的数学含义

在 `route4-ex` 代码中，`q_selected` 先被检查为非负向量，然后归一化：

- [`src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

对应实现：

- [`src/python/qrng_routes/route4_ex/prototype.py#L72-L86`](../src/python/qrng_routes/route4_ex/prototype.py#L72-L86)

也就是说，用户传入的

- `[2,1,1]`
- `[5,1,1]`
- `[1,0,0]`

在进入 SDP 前都会被解释成一个归一化的先验分布：

\[
q_x \ge 0,\qquad \sum_x q_x = 1.
\]

随后，这个分布直接进入 primal 目标函数。对角版里是

\[
\sum_x q_x \cdot \bigl(\rho_x \text{ 对目标猜测项的贡献}\bigr),
\]

见：

- [`src/python/qrng_routes/route4_ex/prototype.py#L637-L645`](../src/python/qrng_routes/route4_ex/prototype.py#L637-L645)

full primal 里同样如此：

- [`src/python/qrng_routes/route4_ex/prototype.py#L727-L739`](../src/python/qrng_routes/route4_ex/prototype.py#L727-L739)

因此，`q` 的数学含义很明确：

- 它是“生成轮里，输入标签 `x` 被选中的先验概率”；
- SDP 认证的是这个输入分布下的平均 guessing probability；
- 不是“把所有输入强行平均后”的固定值。

## 3. 代码里 `q` 目前没有单独建模“测试轮”

当前 `route4-ex external` 实例构造函数

- [`src/python/qrng_routes/route4_ex/prototype.py#L490-L540`](../src/python/qrng_routes/route4_ex/prototype.py#L490-L540)

会把：

- 外部概率表中的若干行；
- 一组 trusted coherent inputs；
- 一个 `q_selected`

一起打包成单个认证问题。

这里没有再额外区分：

- 生成轮分布 `q_gen`
- 参数估计/测试轮分布 `q_test`

也没有把“测试轮占比”单独乘进最终 bit rate。

所以当前代码输出的 `H_min`，应解释为：

- “如果把这组输入按 `q_selected` 作为生成轮先验来使用，那么每个生成轮的认证最小熵是多少”。

它还不是整个实验协议的最终平均吞吐率。

## 4. 怎么把它映射到真实实验轮次

更贴近实验的理解应该分成两层。

第一层：生成轮内部的输入选择分布

- 这层直接对应代码里的 `q_selected`。
- 例如 `q=[1,0,0]` 的意思不是“实验里永远只制备一个态，因此其他态不存在”，而是：
- “在被拿来产出随机数的那一类轮次里，只把第一个输入当作生成输入。”

第二层：生成轮和测试轮的总轮次划分

- 这层当前代码没有显式建模。
- 更真实的协议通常会是：
- 绝大多数轮次用于生成；
- 一小部分轮次用于校准、参数估计、有限尺寸统计。

如果记：

- `p_gen` = 生成轮占比
- `H_min^(gen)` = 代码里在 `q_selected` 下得到的每生成轮最小熵

那么最终总平均认证输出率更像是

\[
R_{\text{avg}} \approx p_{\text{gen}} \cdot H_{\min}^{(\text{gen})},
\]

再乘上后处理与有限尺寸损耗。

所以：

- `q=[1,0,0]` 可以理解成“生成轮只用第一个输入”；
- 但实验仍然完全可以保留第二、第三输入，只是把它们放到测试/校准轮里。

## 5. 为什么偏置 `q` 会把结果抬高

当前窗口 `[100,120,140]` 的结果显示，三个输入并不对称。

已有结果文件：

- [`../output/qrng_routes/route4_ex_external_probabilitymat_outputs2_qbias_alpha063_cutoff6.json`](../output/qrng_routes/route4_ex_external_probabilitymat_outputs2_qbias_alpha063_cutoff6.json)
- [`../output/qrng_routes/route4_ex_mosek_verify_uniform_100120140_q111.json`](../output/qrng_routes/route4_ex_mosek_verify_uniform_100120140_q111.json)
- [`../output/qrng_routes/route4_ex_mosek_verify_biased_100120140_q100.json`](../output/qrng_routes/route4_ex_mosek_verify_biased_100120140_q100.json)

代表点是：

- `q=[1,1,1]` 时，`H_min ≈ 0.2763`
- `q=[2,1,1]` 时，`H_min ≈ 0.4040`
- `q=[5,1,1]` 时，`H_min ≈ 0.5868`
- `q=[1,0,0]` 时，`H_min ≈ 0.8733`

这说明：

- 第一个输入是当前窗口里的“强生成输入”；
- 第二、第三输入更像是帮助约束装置、但会拉低平均生成熵的辅助输入。

因此偏置 `q` 不是数值技巧，而是对应一个非常具体的实验协议设计：

- 用“最好”的输入去做生成；
- 用其它输入去做测试和约束。

## 6. `q=[1,0,0]` 在实验上是否允许

从当前这套 SDP 写法看，是允许的。

原因是：

- 目标函数只关心生成轮采用哪个输入；
- 约束本身仍然可以把其它输入的观测概率一起放进去；
- 所以“只用第一个输入产出随机数，同时仍然测另外两个输入来做装置约束”在逻辑上是成立的。

真正需要单独注意的是协议口径：

- 如果导师或实验室更习惯“所有输入都按固定比例混入总吞吐率”来报结果，那么 `q=[1,0,0]` 不能直接被读成“整机平均每轮都有 `0.8733 bit`”；
- 更准确的读法应是：“在生成轮只选第一个输入时，生成轮的认证最小熵约为 `0.8733 bit`。”

## 7. 对后续实验设计的直接建议

基于当前结果，最自然的实验口径是：

1. 先把“生成输入”和“测试输入”在实验流程里分开。
2. 生成轮优先只使用当前最强输入。
3. 第二、第三输入保留为参数估计/一致性检查输入，而不是硬塞进平均生成分布。
4. 最终汇报时同时给两类数字：
   - 生成轮最小熵 `H_min^(gen)`
   - 乘上生成轮占比后的平均输出率

## 8. 一句话结论

`route4-ex` 里的偏置 `q`，最合理的实验解释不是“篡改了物理输入分布”，而是“把生成轮输入分布和测试轮输入分工显式区分开了”。因此 `q=[1,0,0]` 更应读作：

- “生成轮只选第一个输入，其他输入留给测试约束”，

而不是

- “实验里永远不再制备其它输入”。
