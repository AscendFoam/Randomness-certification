# 下一位 AI 接手本项目的详细 Prompt（中文）

请把下面整份内容当作“接手说明 + 工作指令 + 当前结论摘要”来使用。你接手的是一个已经持续迭代了较长时间的量子随机数认证项目，仓库名为 `Randomness-certification`。在继续开发之前，请先完整理解当前主线、已被否定的方向、导师最新确认的实验口径，以及哪些结果是“正式可汇报”，哪些只能算“探索性结果”。

---

## 0. 你的角色与目标

你现在接手的是一个围绕多条 QRNG 认证路线展开的研究型代码仓库。你的任务不是从零开始重新设计协议，而是在**尊重已有结果、导师最新要求和实验真实口径**的前提下，继续推进后续开发、验证、报告整理和必要的代码修改。

你的工作重点应当是：

1. 优先服务当前**实验主线**，而不是盲目追求数值更高的探索性结果；
2. 明确区分：
   - 哪些结果是“严格对应实验真实输入”的正式结果；
   - 哪些结果只是“理论探索 / 机制验证 / 上界参考”；
3. 后续若继续扩展 `route4` 体系，必须严格遵守导师最新给出的实验口径；
4. 若后续用户再次切回 `route5` 或更强 CV 路线，可以利用已有代码和文档，但不要混淆 `route4` 与 `route5` 的实验前提。

---

## 1. 仓库当前结构总览

### 1.1 Python 路线代码目录

核心代码位于：

- `src/python/qrng_routes/common.py`
- `src/python/qrng_routes/route1/`
- `src/python/qrng_routes/route2/`
- `src/python/qrng_routes/route3/`
- `src/python/qrng_routes/route4/`
- `src/python/qrng_routes/route4_ex/`
- `src/python/qrng_routes/route4_ex_constrained/`
- `src/python/qrng_routes/route4_strict_nondiagonal/`
- `src/python/qrng_routes/route5/`
- `src/python/qrng_routes/route6/`

### 1.2 Matlab 参考脚本目录

Matlab 文件位于：

- `src/matlab/Probability.mat`
- `src/matlab/guessprobprimal_phaseinsensitive.m`
- `src/matlab/guessprobprimal_phaseinsensitive_original.m`
- `src/matlab/guessprobprimal_route4_ex.m`
- `src/matlab/guessprobprimal_route4_ex_constrained.m`
- `src/matlab/guessprobprimal_route5_hybrid_iq.m`

### 1.3 当前最关键的中文文档

请优先阅读这些文件：

1. `docs/route4_matlab_theory_formulation_cn.md`
2. `docs/route4_ex_vs_matlab_requirement_gap_cn.md`
3. `docs/route4_strict_nondiagonal_stage_report_cn.md`
4. `docs/route4_ex_two_questions_clarification_cn.md`
5. `docs/QRNG_with_uncharacterized_APD.md`
6. `docs/route5_matlab_theory_formulation_cn.md`
7. `docs/route5_detailed_technical_report_cn.md`

如果时间有限，至少先读前 4 个。

---

## 2. 当前最重要的导师确认信息（必须视为硬约束）

下面这些不是推测，而是导师已经明确回复的信息。后续任何模型修改都不能违背它们。

### 2.1 关于 `mu`

实验中的

$$
\mu = 20,40,60,80,100,120,140,160
$$

是**实验上已经标定过的真实平均光子数**，不是仅仅的档位标签。

也就是说：

- `100` 就是 `mu = 100`
- `120` 就是 `mu = 120`
- `140` 就是 `mu = 140`

不能再把这几个数当成“只是标签，真实 `mu` 还要另算”。

### 2.2 关于测试态

测试态由激光光源调制产生，当前**没有做相位随机化或相位平均**。

因此：

- 测试态本身是**固定相位**的；
- 它们不是实验上先做了 phase randomization 的态。

### 2.3 关于原始 `route4` 为什么只用 `rho_x^{diag}`

原始 `route4` / Matlab 主线里之所以只保留 `rho_x` 的对角部分，不是因为实验把态先做了相位平均，而是因为：

1. 原始问题里假设 POVM 在 Fock 基下对角；
2. 于是
   $$
   \mathrm{Tr}(\rho_x M_c)
   $$
   中只有 `rho_x` 的对角项会参与；
3. 所以原始问题直接把优化变量写成向量形式 `m_{c,\ell}(n)`，POVM 的“对角约束”已经内建在变量定义里，不需要额外再加一条矩阵对角约束。

### 2.4 关于去掉 POVM 对角限制后的输入态

如果后续考虑去掉 POVM 的对角限制，那么：

- 测试态**仍然应该保持实验真实输入**；
- 也就是应写成固定相位的相干态，而不是再做相位平均，更不能自由搜索另一组与实验不一致的输入态。

在最简建模口径下，可以先把共同相位参考取为 0，于是

$$
\alpha_x = \sqrt{\mu_x}
$$

作为固定相位测试态的参考表示。

注意：这并不是说相位在原理上完全不重要，而是说在当前已知实验信息下，不应再引入额外的相位搜索自由度。

### 2.5 关于 `Probability.mat`

导师已说明给出的概率分布数据就是每个实验测试态对应的输出分布，形状应类似 9 条不同峰位的概率曲线。

当前可合理采用的工作假设是：

- `Probability.mat` 的 9 行按
  `[0,20,40,60,80,100,120,140,160]`
  排列；
- 每一行是对应 `mu` 下的 `256` 维条件分布；
- 后续 coarse-graining 只是对这 `256` 个原始输出 bin 做数字后处理。

如果后续要做正式实验版报告，仍建议继续确认：

1. 9 行顺序是否确实如此；
2. 每个 `mu` 对应的原始 shot 数 / 总计数；
3. 256 个 bin 的物理含义与归一化方式。

---

## 3. 目前各条路线的状态与定位

### 3.1 `route4`：当前实验正式主线

目录：

- `src/python/qrng_routes/route4/`
- `src/matlab/guessprobprimal_phaseinsensitive_original.m`
- `src/matlab/guessprobprimal_phaseinsensitive.m`

定位：

- 这是当前**最严格对应实验真实输入口径**的主线；
- 使用实验真实 `mu`；
- 使用 `Probability.mat`；
- 假设 APD / 测量是 phase-insensitive，因此 POVM 在 Fock 基下对角；
- 输入态在计算中只保留 `rho_x^{diag}`。

当前可信结果口径：

- Matlab 兼容、较保守的正式最好结果大约在
  $$
  H_{\min} \approx 0.527
  $$
  附近；
- 更激进的 Python 扩展扫描在非 Matlab 兼容 `N=20` 口径下曾到约
  $$
  H_{\min} \approx 0.555,
  $$
  但这不应压过 Matlab 兼容主结论。

当前更稳妥的主线表述是：

- 正式实验口径下，`route4` 当前大约能做到 `0.5 bit` 左右，而不是接近 `2 bit`。

### 3.2 `route4_ex`：探索性旁线，不再是正式实验主线

目录：

- `src/python/qrng_routes/route4_ex/`

定位：

- 它探索的是“如果 trusted input 换成 non-diagonal coherent states，会不会显著抬高 formal 熵”；
- 它保留了 `Probability.mat` 接口，但曾引入了额外的 `alpha` 自由度、相位图样、自由半径搜索和高熵边界搜索。

当前最著名的高值结果：

$$
H_{\min} \approx 1.54395
$$

对应文件：

- `output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`

但注意：

- 这个结果**不能再作为严格实验口径的正式结论**；
- 因为它使用的 trusted input 不是实验真实固定输入，而是另外精修得到的 `alpha`。

因此目前应把 `route4_ex` 明确降级为：

- 探索性结果；
- 机制验证；
- 上界参考；
- 帮助理解“non-diagonal trusted input 可能带来什么增益”。

不要再把 `1.54 bit` 当作实验上 `[100,120,140]` 三个真实输入已经达到的正式结果。

### 3.3 `route4_ex_constrained`：比 `route4_ex` 更接近主线，但仍属探索性

目录：

- `src/python/qrng_routes/route4_ex_constrained/`
- `src/matlab/guessprobprimal_route4_ex_constrained.m`

定位：

- 它是对 `route4_ex` 的“收缩版 / Matlab 风格版”；
- 比 `route4_ex` 更固定、更接近原始 route4 的骨架；
- 但它仍然使用了额外指定的 coherent trusted inputs。

当前典型结果：

$$
H_{\min} \approx 1.22750
$$

对应文件：

- `output/qrng_routes/route4_ex_constrained_baseline_compare.json`

但和 `route4_ex` 一样，当前也不应直接当作实验正式结果。

### 3.4 `route4_strict_nondiagonal`：当前最值得继续推进的 `route4` 衍生主线

目录：

- `src/python/qrng_routes/route4_strict_nondiagonal/`

定位：

- 这是为回应导师要求而新开的路线；
- 核心思想是：**保持实验真实输入不变**，只去掉“POVM 必须对角”的限制；
- 不再引入 `route4_ex` 那种自由 `alpha` 搜索。

请把它理解成：

> 原始 route4 的“真实输入固定版 non-diagonal 扩展”

而不是 `route4_ex` 的继续自由优化版。

当前已做过的代表性尝试：

1. `scale = 1.0`
   - 即按
     $$
     |\alpha_x|^2 = \mu_x
     $$
     构造固定 coherent inputs；
   - `selected_mu = [100,120,140]`
   - `N = 4`
   - `cutoff = 280`
   - `support_dimension = 3`
   - 对比结果文件：
     `output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale1.json`

   当前结果：
   - 原始参考 `route4` primal：`H_min ≈ 0.1604`
   - strict non-diagonal full primal：`H_min ≈ 0.0768`
   - 状态：`optimal_inaccurate`

2. `scale = 0.01`
   - 即尝试用更小平均光子数比例；
   - 对应文件：
     `output/qrng_routes/route4_strict_nondiagonal_compare_mu100120140_N4_cutoff280_scale001.json`
   - 当前 strict full primal 为 `infeasible`

这说明：

- “真实输入固定 + 去掉 POVM 对角限制”这条线是对的；
- 但当前实现和当前参数下，它还**没有**超过原始 `route4`；
- 它仍然是接下来最值得继续推进的 `route4` 衍生方向。

### 3.5 `route5`：理论上最强，但不是当前 `route4` 实验主线

目录：

- `src/python/qrng_routes/route5/`
- `src/matlab/guessprobprimal_route5_hybrid_iq.m`

定位：

- `route5` 是一条不同物理图像的混合路线：
  “generalized coherent alphabet + CV / IQ measurement + digital coarse-graining + single-device SDP”
- 它已经在理论数值层面做到了 `H_min > 2`。

当前最强结果：

1. 自由字母表强点
   $$
   H_{\min} \approx 2.11639
   $$
   文件：
   `output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json`

2. 固定光强 `[0,80,160]` 主线
   $$
   H_{\min} \approx 2.10102
   $$
   文件：
   `output/qrng_routes/route5_fixed_intensity_080160_scale120.json`

但它需要不同于 `Probability.mat` 的实验数据形态，不能直接拿来替代当前 `route4` 实验主线。

因此：

- `route5` 很强；
- 文档也很完整；
- 但当前如果用户/导师继续追问“实验真实输入的 `route4` 该怎么做”，不要拿 `route5` 混为一谈。

### 3.6 `route6`：解析 Gram + IQ 方向，当前暂停

目录：

- `src/python/qrng_routes/route6/`

这是 route3/route5 混合思路的进一步探索，目前不是当前主线。除非用户再次明确切回这条线，否则不要优先投入。

---

## 4. 当前最重要的结论边界

### 4.1 现在可以正式说什么

1. 原始 `route4` 是当前最严格贴实验真实口径的正式主线；
2. 在当前 `Probability.mat` 和 phase-insensitive 假设下，`route4` 正式值大约在 `0.5 bit` 量级；
3. `route4_ex` / `route4_ex_constrained` 已经证明：
   - 如果 trusted input 引入 non-diagonal coherent 结构，
   - formal 熵有可能显著抬高到 `1 bit` 以上，甚至在探索性设置下到 `1.54 bit`；
4. 但这些高值目前不能直接当作实验正式结果；
5. `route4_strict_nondiagonal` 是当前把“真实输入固定”与“去掉 POVM 对角限制”结合起来的最合理下一步；
6. `route5` 在理论原型上已给出 `>2 bit` 的清晰证据，但它依赖不同实验路线。

### 4.2 现在不能再说什么

1. 不能再说 `route4_ex` 的 `1.54 bit` 就是实验 `[100,120,140]` 三个真实输入的正式结果；
2. 不能再自由搜索与实验输入不一致的 `alpha`，然后把结果包装成实验主线结论；
3. 不能再把“测试态是否固定相位”和“计算里为什么只保留对角项”混为一谈；
4. 不能再假设 `100/120/140` 只是档位标签；
5. 不能在当前主线上默认引入 route5 那种不同数据形态而不说明物理前提变化。

---

## 5. 你接手后应该遵守的工作原则

### 5.1 优先级排序

请按下面优先级推进：

1. **最高优先级**：`route4-original` 与 `route4_strict_nondiagonal`
2. **中优先级**：整理实验口径、补充文档、把已有结论写清楚
3. **低优先级**：`route4_ex` / `route4_ex_constrained` 的进一步搜索
4. **更低优先级**：除非用户明确要求，否则不要主动继续 `route6`

### 5.2 研发主线建议

接手后，建议把 `route4` 体系拆成三层：

1. **正式基线层**
   - 原始 `route4`
   - 固定实验真实输入
   - 对角 POVM
   - 输出可继续扫 `N=4/8/16` 和 `q_selected`

2. **真实输入固定的扩展层**
   - `route4_strict_nondiagonal`
   - 保持测试态固定为实验真实的固定相位 coherent states
   - 只放开 POVM 的对角限制
   - 不再自由搜索 `alpha`

3. **探索性原型层**
   - `route4_ex`
   - `route4_ex_constrained`
   - 只作为机制参考，不作为正式实验主线

### 5.3 对 `route4_strict_nondiagonal` 的具体建议

后续若继续开发，请重点做这些事：

1. 保持输入态固定为实验真实输入：
   $$
   \rho_x = |\sqrt{\mu_x}\rangle\langle\sqrt{\mu_x}|
   $$
   其中共同相位先固定为 0，不再引入相位搜索。

2. 重新梳理统计约束：
   - 原始 `route4` 是
     $$
     \sum_\ell\sum_n \rho_x^{\mathrm{diag}}(n)\,m_{c,\ell}(n)=p(c|x)
     $$
   - strict non-diagonal 应改为
     $$
     \sum_\ell \mathrm{Tr}(\rho_x M_{c,\ell}) = p(c|x)
     $$

3. 充分利用支撑子空间降维：
   - 这是 `route4_strict_nondiagonal` 能算下去的关键。

4. 优先在 Matlab 兼容的 `N` 上测试：
   - 例如 `N=4,8,16`
   - 不要一开始就用任意 `N`

5. 对结果的要求要务实：
   - 当前目标不是直接冲 `2 bit`
   - 而是先判断：
     “在真实输入固定的前提下，只放开 POVM，对正式值到底有没有提升”

### 5.4 对 `route4-original` 的具体建议

如果 strict 线短期内没有明显优势，请继续把原始 `route4` 做扎实：

1. 严格保留 Matlab 兼容口径；
2. 继续围绕真实窗口扫：
   - `[140,160]`
   - `[120,140,160]`
3. 优先考虑 Matlab 兼容/易解释的 `N`；
4. 继续整理 primal/dual 一致性、零概率病理、coarse-graining 完整覆盖等问题；
5. 如果需要正式给实验室报告，原始 `route4` 仍是最稳妥的基线。

### 5.5 对 `route4_ex` / `route4_ex_constrained` 的原则

1. 不要删掉这些代码和文档；
2. 但不要再把它们的高值当作正式实验主线；
3. 可以保留它们作为：
   - 上界参考；
   - 机制分析；
   - 为什么 non-diagonal structure 可能重要的证据；
4. 如果用户要给导师解释“为什么当初会出现 `1.54 bit`”，这些文档仍然有用。

---

## 6. 建议的阅读顺序与开发顺序

### 6.1 第一次接手时建议阅读顺序

1. `docs/route4_matlab_theory_formulation_cn.md`
2. `docs/route4_ex_vs_matlab_requirement_gap_cn.md`
3. `docs/route4_strict_nondiagonal_stage_report_cn.md`
4. `docs/route4_ex_two_questions_clarification_cn.md`
5. `docs/QRNG_with_uncharacterized_APD.md`

然后根据用户当前任务，决定是否继续读：

6. `docs/route5_matlab_theory_formulation_cn.md`
7. `docs/route5_detailed_technical_report_cn.md`

### 6.2 第一次接手时建议执行的检查

1. 检查当前设备环境是否具备：
   - Python
   - 所需 solver（SCS / MOSEK）
   - Matlab 或者至少可以读取 `.m` 文件

2. 先不要跑长时间任务，先做以下轻量核对：
   - 确认 `Probability.mat` 可读；
   - 确认 `route4` baseline 命令可跑；
   - 确认 `route4_strict_nondiagonal` 的一个小实例可复现；
   - 确认关键输出文件能读。

3. 如果用户下一步还是围绕 `route4`，就先不要继续 route5 搜索。

---

## 7. 如果你需要继续向用户/导师确认实验信息，优先问什么

当前已经知道很多，但仍建议优先确认下面几项：

1. `Probability.mat` 的 9 行顺序是否确实是
   `[0,20,40,60,80,100,120,140,160]`
2. 每个 `mu` 对应的原始总计数 / shot 数是否可提供
3. `256` 个 raw bins 的物理含义和归一化方式是否固定
4. 固定相位测试态是否可以默认看成“同一相位参考下的实正 \(\alpha=\sqrt{\mu}\)”；如果不行，实验上能否提供相位标定信息

注意：这些确认问题只有在确实推进正式实验版路线时才需要主动追问；如果用户当前只是在整理代码或文档，不必无谓打断。

---

## 8. 你不应该做的事情

1. 不要再把 `route4_ex` 的 `1.54 bit` 当作实验正式主结果；
2. 不要在 `route4` 主线里继续自由搜索另一组 `alpha`，却声称输入仍对应实验 `[100,120,140]`；
3. 不要把“固定相位测试态”误说成“相位平均态”；
4. 不要把原始 route4 中的向量变量又包一层“矩阵对角约束”，因为导师已明确：原问题里对角约束已经内建于变量定义；
5. 不要混淆 `route4` 和 `route5` 的物理前提与实验数据要求。

---

## 9. 目前最推荐的后续工作主线（简版）

如果用户没有新方向，请默认按下面主线推进：

1. 继续把 `route4` 作为正式实验主线维护；
2. 把 `route4_strict_nondiagonal` 作为当前最有意义的扩展方向推进；
3. 仅在“输入态严格固定为实验真实测试态”的前提下，研究去掉 POVM 对角限制有没有帮助；
4. 把 `route4_ex` / `route4_ex_constrained` 保留为探索性旁线，不再作为正式实验结论；
5. `route5` 保留为另一条成熟但不同实验前提的高熵路线，必要时再切回。

---

## 10. 可以直接复制给下一位 AI 的一句话版任务定义

你现在接手 `Randomness-certification` 项目。当前正式实验主线应回到 `route4`：实验真实输入是已标定平均光子数 `mu=20,40,...,160` 的固定相位测试态；原始 `route4` 里只保留 `rho_x^{diag}`，是因为 POVM 对角假设已内建在向量变量中，而不是因为实验做了相位平均。后续最值得推进的扩展不是继续 `route4_ex` 式自由搜索 `alpha`，而是在输入态严格固定为实验真实测试态的前提下，开发和评估 `route4_strict_nondiagonal`，也就是只去掉 POVM 的对角限制，比较其是否能在正式口径下超过原始 `route4`。与此同时，请明确把 `route4_ex` / `route4_ex_constrained` 的高值结果仅视为探索性证据，不再当作正式实验结果；`route5` 虽然理论上已超过 `2 bit`，但属于不同物理数据路线，不应与当前 `route4` 主线混淆。
