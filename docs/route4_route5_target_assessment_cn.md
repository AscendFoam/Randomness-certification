# Route4 / Route5 目标可达性与下一步行动评估

## 1. 执行结论

这份短报告回答三个问题：

1. 原始 `route4` 在当前实验数据与当前建模约束下，是否还有现实希望达到 `H_min >= 1`。
2. `route5` 对真实实验而言是否可行，以及它的主要风险是什么。
3. 在“`route4` 至少到 `1 bit`、其它路线至少到 `2 bit`”这个总目标下，后续最合适的行动方案是什么。

当前最简结论是：

- 对**原始 route4** 来说，`H_min >= 1` 没有严格的数学不可能证明，但按现有结果看已经是一个**高风险、低胜率目标**，更准确的说法应是“在当前 `Probability.mat` 与当前 route4 约束下，继续冲到 `1 bit` 已经基本不现实”。
- 对**route5** 来说，`H_min > 2` 在理论原型层面已经成立，而且固定光强主线也已经过 `2 bit`；但它对真实实验的可行性，前提是把它当成一条**新的 IQ / dual-homodyne 实验路线**，而不是当前 APD `Probability.mat` 路线的小改版。
- 因而若按“近期稳妥汇报 + 中期冲高值”的组合目标来排优先级，最合理的分工是：
  - `route4` 负责当前正式实验主线与保守结果汇报；
  - `route5` 负责真正去冲 `2 bit` 以上，并作为下一阶段实验设计主线。

---

## 2. 对另一个 AI 回答的校正

另一个 AI 的总体方向判断**大体正确**，但有三点需要修正。

### 2.1 正确的地方

- 它正确指出：当前应把原始 `route4` 主线重新收回到“实验真实 `mu` + 原始对角 POVM 建模”的框架内，而不是再把主要精力放在 `route4_strict_nondiagonal` 或自由 `alpha` 的 `route4-ex` 路线。
- 它正确指出：`route5` 更像一条“下一代实验路线”，不是当前 `Probability.mat` 数据的小修小补。
- 它正确指出：`route5` 真正需要的是它自己的 IQ / coarse-grained 概率数据，而不是直接复用当前 APD 概率表。

### 2.2 需要修正的地方

- 它把原始 `route4` 的当前最好结果写成了约 `0.555 bit`。这已经被新结果更新。
  现在原始 `route4` 的当前最好正式点是  
  [`route4_local_refine_pair_140_160_N20_edges156_167_177_189_201_mosek.json`](../output/qrng_routes/route4_local_refine_pair_140_160_N20_edges156_167_177_189_201_mosek.json)，
  对应  
  `H_min ≈ 0.5626365641`。
- 它把 contiguous coarse-graining 说成“低优先级可考虑”。这个判断现在也需要更新。
  因为 contiguous / local refine 已经不只是探索性尝试，而是**实打实改写了 route4 当前最好 formal 值**。
- 它建议把三态 `[120,140,160]` 作为 route4 正式主线优先窗口。这对“保守汇报”是可以理解的，但如果目标是**尽量把原始 route4 做高**，则当前数据明确表明最强窗口仍然是两态 `[140,160]`，而不是三态 `[120,140,160]`。

因此，更准确的修正版应当是：

- 如果目标是“写一版更保守、看起来更像原协议骨架的汇报”，三态 `[120,140,160]` 可以保留。
- 如果目标是“尽量把原始 route4 的 formal 值往上推”，两态 `[140,160]` 必须继续作为主优化窗口。

---

## 3. 原始 Route4 是否还有现实希望达到 1 bit

### 3.1 当前最好结果到底是多少

当前和原始 route4 最相关的几条正式结果如下：

| 结果类型 | 参数 | `H_min` | 来源 |
|---|---|---:|---|
| Matlab 兼容主线 | `[140,160]`, `q=[0.5,0.5]`, `N=16` | `0.527280` | [`route4_targeted_scan_pair_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json) |
| Python 等覆盖高输出主线 | `[140,160]`, `q=[0.5,0.5]`, `N=20` | `0.554987` | [`route4_targeted_scan_pair_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_pair_140_160_v1.json) |
| contiguous `N=4` 强点 | `[140,160]`, `q=[0.5,0.5]`, `edges=[0,165,181,196,256]` | `0.531761` | [`route4_contiguous_search_pair_140_160_N4_q0505_mosek.json`](../output/qrng_routes/route4_contiguous_search_pair_140_160_N4_q0505_mosek.json) |
| 当前全局最好 route4 点 | `[140,160]`, `q=[0.5,0.5]`, `N=20` 局部边界精修 | `0.562637` | [`route4_local_refine_pair_140_160_N20_edges156_167_177_189_201_mosek.json`](../output/qrng_routes/route4_local_refine_pair_140_160_N20_edges156_167_177_189_201_mosek.json) |
| 三态较优正式点 | `[120,140,160]`, `q=[1/3,1/3,1/3]`, `N=16` | `0.525002` | [`route4_targeted_scan_triple_120_140_160_v1.json`](../output/qrng_routes/route4_targeted_scan_triple_120_140_160_v1.json) |

所以，原始 route4 当前最好正式值不是 `0.55`，而是约

$$
H_{\min}^{\mathrm{route4,best}} \approx 0.5626365641.
$$

### 3.2 离 1 bit 还差多少

对 `route4` 当前最好点，

$$
H_{\min}\approx 0.5626365641
\quad\Longleftrightarrow\quad
p_{\mathrm{guess}} \approx 0.6770636802.
$$

而要达到

$$
H_{\min} \ge 1,
$$

等价于要求

$$
p_{\mathrm{guess}} \le 0.5.
$$

也就是说，当前 route4 还需要把 guessing probability 从

$$
0.6771
$$

继续压到

$$
0.5,
$$

差距是：

$$
\Delta p_{\mathrm{guess}} \approx 0.1771.
$$

按相对比例看，仍需再压掉约

$$
26.15\%
$$

的当前 guessing probability。

这个差距不小，而且比最近几轮优化带来的正式提升要大得多。

### 3.3 从现有优化轨迹看，这个目标有多难

已有的 route4 正式提升轨迹大致是：

- `N=12` 等覆盖：`H_min ≈ 0.4507`
- `N=16` 等覆盖：`H_min ≈ 0.5273`
- `N=20` 等覆盖：`H_min ≈ 0.5550`
- `N=20` 局部边界精修：`H_min ≈ 0.5626`

从这个轨迹可以看到两件事：

1. route4 还没有完全饱和。
   这点很明确，因为 `N=20` 局部精修确实又把值抬高了。
2. 但提升幅度已经明显进入“慢增区”。
   最近一次从 `0.5550` 到 `0.5626` 的提升只有约 `0.0076 bit`。

而从当前 `0.5626` 到目标 `1.0`，还差

$$
0.4374\ \text{bit}.
$$

这比最近一轮正式提升大了一个数量级以上。

### 3.4 因此，应该怎样表述 route4 到 1 bit 的可能性

最准确的说法不是“严格不可能”，而是：

```text
在当前 Probability.mat、当前原始 route4 建模、以及目前已知最优窗口 [140,160] 下，
继续把 route4 正式结果推进到 1 bit，已经是一个高风险、低胜率目标。
```

如果要再压缩成一句更直接的话，可以写成：

```text
原始 route4 冲到 1 bit 目前基本不现实，但还可以做一轮有边界的收尾优化，
用于把“不现实”从经验判断进一步收紧为阶段性结论。
```

也就是说：

- 我不建议把 `route4 >= 1` 继续当成当前项目的**主计划假设**；
- 但我建议再做一轮**非常有限的、带停止条件的** route4 收尾优化，然后就应当把主要资源切到 `route5`。

---

## 4. Route5 对真实实验而言是否可行

### 4.1 结论先行

`route5` 对真实实验而言是**可行的，但前提是把它视作一条新的 CV / IQ 实验路线**。

更准确地说：

- 如果问题是“`route5` 能不能直接接当前 APD `Probability.mat` 实验”，答案基本是否定的。
- 如果问题是“实验室能否按 route5 的物理需求重新采数，并走同一套 SDP 认证链”，答案是肯定的。

这一区别非常重要。

### 4.2 为什么说它在理论原型上已经足够强

当前 route5 已有两条最关键结果：

- 自由搜索最佳正式点  
  [`route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json`](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)  
  对应  
  `H_min ≈ 2.1163917383`
- 固定光强主线 `[0,80,160]`  
  [`route5_fixed_intensity_080160_scale120.json`](../output/qrng_routes/route5_fixed_intensity_080160_scale120.json)  
  对应  
  `H_min ≈ 2.1010172143`

固定光强这一点尤其重要，因为它说明 route5 并不是只有在“自由 alphabet”时才过 2；在更贴近实验限制的固定光强主线下，它仍然能保持

$$
H_{\min} > 2.
$$

### 4.3 但它不是“当前 APD 实验的小改版”

按 [route5_detailed_technical_report_cn.md](./route5_detailed_technical_report_cn.md) 和代码
[`src/python/qrng_routes/route5/hybrid_iq.py`](../src/python/qrng_routes/route5/hybrid_iq.py) /
[`src/python/qrng_routes/route5/intensity_menu_search.py`](../src/python/qrng_routes/route5/intensity_menu_search.py)，
route5 的协议骨架是：

1. trusted coherent alphabet
2. beam splitter + dual-homodyne / IQ measurement
3. digital coarse-graining
4. SDP 认证

因此 route5 要求的数据是：

- 每个 `(x,y)` 输入对对应的 IQ 原始样本；
- 或至少二维 IQ 直方图；
- 或至少已经固定分箱后的 `P_exp(c|x,y)`。

这与当前 APD 的一维点击概率表 `Probability.mat` 不是同一种数据结构。

所以 route5 的现实定位应当是：

```text
它是下一阶段值得单独设计实验的数据路线，
不是当前 APD 路线的小修小补。
```

---

## 5. Route5 的主要实验风险

这里把 route5 的主要风险按“对实验闭环的影响”来排序。

### 5.1 风险一：当前结果离 2 bit 阈值并不算非常远

固定光强主线 `[0,80,160]` 的结果是

$$
H_{\min}\approx 2.1010172143,
$$

对应

$$
p_{\mathrm{guess}} \approx 0.23309.
$$

而 `2 bit` 阈值对应

$$
p_{\mathrm{guess}} = 0.25.
$$

也就是说，当前固定光强主线虽然已经过 `2 bit`，但它在 guessing probability 上只比阈值多出约

$$
0.0169
$$

的裕量。

这说明：

- route5 过 `2 bit` 是真的；
- 但这个裕量并不算“非常宽”；
- 实验噪声、标定漂移、边界偏移，都有可能把这个裕量吃掉一部分。

### 5.2 风险二：固定光强版仍带有一个需要实验标定闭合的缩放参数

在
[`intensity_menu_search.py`](../src/python/qrng_routes/route5/intensity_menu_search.py)
里，当前固定光强扫描采用的是

$$
\text{radius}
=
\text{max\_radius}\sqrt{\frac{\text{intensity}}{\text{max\_intensity}}}.
$$

代码对应位置可见：

- `intensity_menu_to_radii(...)`
- `scaling_rule = "radius = max_radius * sqrt(intensity / max_intensity_in_menu)"`

这在理论搜索阶段是合理的，因为它帮助我们快速扫描“在实验光强菜单上，哪种整体尺度最有利”。

但对实验闭环来说，这也是 route5 当前最敏感的口径风险之一：

- 如果未来实验版还把 `max_radius` 当成事后搜索参数，那就会出现和早期 `route4-ex` 类似的口径争议；
- 因而实验版 route5 必须把 `alpha / radius` 的映射通过独立标定固定下来，或者直接由实验端给出每个输入的已标定 `alpha`。

因此，route5 不是“不能实验”，而是**必须先把这个缩放从搜索参数变成标定参数**。

### 5.3 风险三：字母表较大，测试轮与标定压力不小

按 [route5_detailed_technical_report_cn.md](./route5_detailed_technical_report_cn.md)，当前强点对应的本地 alphabet 是：

- 1 个真空点
- 2 个非零半径层
- 每层 8 个相位

所以本地共有 `17` 个状态，联合输入共有

$$
17^2 = 289
$$

个输入对。

这会带来：

- 标定工作量大；
- 参数估计数据量大；
- 有限尺寸测试轮开销大。

因此 route5 的优势不是“实验工作量最省”，而是“在工作量上升的前提下，换来了 `H_min > 2` 的可能性”。

### 5.4 风险四：当前代码的实验接入结构是“可接”，但还不是“现成能交实验室直接跑”

从结构上看，route5 的认证层已经与概率层部分分离：

- `certify_target_inputs(...)` 可以直接吃 `input_states + probabilities`
- 认证核心来自 `common.py` 的 `SingleDeviceGuessingProblem`

这说明 route5 **不是只能做理论玩具**。

但当前主流程
[`run_route5(...)`](../src/python/qrng_routes/route5/hybrid_iq.py)
仍然是先内部调用
`dual_homodyne_probabilities(...)`
生成理论概率，再去做 SDP。

因此当前还缺少一个真正面向实验室的、开箱即用的接口，例如：

- 从实验 IQ 样本读入；
- 从二维直方图读入；
- 或从实验版 `P_exp(c|x,y)` 直接读入。

所以 route5 的实验适配状态更准确的说法是：

```text
结构上已具备实验接入逻辑，
但工程上还没有整理成一个直接给实验室用的入口。
```

### 5.5 风险五：当前 formal 结果只认证了少数 target inputs，不是“所有输入都同等地产随机”

route5 当前代码里有一个重要但容易被忽略的点：

- 在 `run_route5(...)` 中，会先按 `raw_h` 排序；
- 然后只对前若干个输入做 formal 认证；
- 具体数量由 `max_inputs_to_certify` 控制。

当前自由主线的 Matlab 理论说明里就写明了：

- `max_inputs_to_certify = 3`

这不是协议错误，但实验表述里必须讲清：

- 哪些输入是生成输入；
- 哪些输入是测试输入；
- 为什么不是所有 `289` 个输入都被同等当作最终生随机输入。

否则容易在对外汇报时把“最佳 target 输入的 formal 熵”误说成“整个 alphabet 平均都这么高”。

### 5.6 风险六：实验实现还会多出相位参考与数字链路稳定性问题

这一点是基于协议结构作出的工程推论。

由于 route5 使用的是：

- 多相位 coherent alphabet；
- dual-homodyne / IQ measurement；

所以实验上还会额外敏感于：

- 相位参考稳定性；
- 本振和信号的相对校准；
- ADC 归一化与去直流处理；
- 分箱边界的长期漂移。

这些问题在当前理论结果里还没有被真正计入。

因此，route5 的“实验可行”不等于“实验实现轻松”；它更像是一条**高潜力但更重工程化**的路线。

---

## 6. 在总目标下的合适行动方案

这里把总目标写清楚：

```text
route4 结果至少达到 1；
其它路线结果至少达到 2。
```

按当前证据，我建议把它转写成如下更现实的执行版本。

### 6.1 对 route4：做“一轮有停止条件的收尾优化”，不要无限追 1 bit

推荐的 route4 策略不是继续大范围盲扫，而是：

1. 固定两态窗口 `[140,160]` 与 `q=[0.5,0.5]`，因为它是当前最强正式窗口。
2. 以当前最好边界  
   `[0,12,25,38,51,64,76,89,102,115,128,140,156,167,177,189,201,217,230,243,256]`  
   为中心，再做一轮更窄的 `N=20` 或 `N=24/28/32` 局部 refine。
3. 同时做 `prob_floor`、`cutoff`、`MOSEK/SCS` 的稳定性复核。
4. 给 route4 输出一份正式表和补充表，不再让结果散落在多个 JSON 里。

但这轮优化必须带停止条件。建议的停止条件可以是：

- 如果再做一轮局部 refine 后，最好 formal 值仍明显低于 `0.65 bit`；
- 或者连续几轮都只带来 `0.01 bit` 量级的边际提升；

那么就应当把“route4 冲 1 bit”正式降级为**长期 stretch goal**，而不是当前阶段主 KPI。

### 6.2 对 route5：把主要资源转向“实验适配”，而不是继续盲搜更高理论点

对 route5，当前最值钱的不是继续自由搜更高 `2.11x` 点，而是把它变成实验室可接的路线。

我建议直接做两件工程事：

1. 增加一个 route5 外部数据接入入口。
   目标是直接读：
   - IQ 原始样本；
   - 二维直方图；
   - 或实验版 `P_exp(c|x,y)`。
2. 把固定光强版的 `max_radius` 搜索口径改成“外部标定输入”口径。
   也就是：
   - 要么让实验端直接给 `alpha_values`；
   - 要么给固定的强度到半径映射表；
   - 但不要在实验闭环里继续把它当作可事后调的搜索参数。

做完这两步之后，route5 才会从“理论设计图”真正变成“实验准备就绪的协议主线”。

### 6.3 对 route4-ex / route4-ex-constrained：暂不作为主攻目标

按当前导师要求，这两条线不应当再作为正式实验主线。

更合适的定位是：

- 作为理论对照；
- 作为说明“为什么非对角 trusted-input 模型会显著抬高 formal 值”的材料；
- 但不再占用主开发资源。

### 6.4 因而最合理的整体分工是

综合来看，我建议把目标拆成：

- `route4`：争取在正式实验主线下把结果做稳、做清楚，并在可能范围内继续向上精修；
- `route5`：承担“至少 2 bit”的主目标；
- `route4-ex` 家族：降级为解释性/探索性参考。

如果必须把判断压缩成一句话，我会建议写成：

```text
route4 继续做一轮有停止条件的收尾优化，但不要再把 1 bit 作为当前主计划假设；
route5 则应正式转入实验适配阶段，因为它才是当前最有现实希望达到 2 bit 以上的路线。
```

---

## 7. 建议的立即下一步

如果只选一件最值得马上做的事，我建议优先级如下：

1. 给 route5 做“外部实验概率 / IQ 数据接入 + 固定标定 alpha/radius”的实验适配入口。
2. 同时给 route4 做一轮很窄的 `N=20`/`N=24` 局部精修与稳定性扫，作为收尾优化。
3. 然后写一份正式分层汇报：
   - route4 当前最好正式值；
   - route4 的保守口径值；
   - route5 固定光强 > 2 bit 的理论可行性；
   - route5 的实验风险与所需数据清单。

如果目标必须强行对应成“哪条路线负责哪个数字”，那么当前最现实的分配就是：

- `route4`：负责尽量逼近，但不要再把 `1 bit` 当作高概率承诺；
- `route5`：负责真正去冲并维持 `2 bit` 以上。
