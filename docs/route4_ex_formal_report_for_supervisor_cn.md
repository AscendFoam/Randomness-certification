# Route4-ex 正式汇报稿

## 1. 结论先行

`route4-ex` 是在原始 `route4` 基础上构造的非对角 trusted-input 扩展路线。  
它保持了原始 `route4` “继续使用 APD 概率表、继续使用离散 coarse-graining 输出”的实验接口，但不再把 trusted inputs 限制为 Fock 基对角的 phase-insensitive 模型，而是改用完整截断相干态 `|alpha><alpha|` 作为可信输入。

当前阶段，这条路线已经得到如下正式结果：

- 在窗口 `[100,120,140]`
- `3` 输出边界 `[0,121,132,256]`
- 相位模式 `0_pi2_pi`
- 偏置生成分布 `q=[1,0,0]`
- 自由单调半径模型 `free_monotone_radii`

下，`MOSEK` 已正式确认：

- 半径  
  `[0.5379541015625, 0.6620458984375, 0.7179541015625]`
- `status = optimal`
- `H_min ≈ 1.54395`

对应结果文件：

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)

因此，如果当前阶段目标是把 `route4` 主线从原先较低的正式值提升到 `H_min >= 1`，那么 `route4-ex` 不仅已经达到目标，而且正式值已经抬高到约 `1.54 bit`。

---

## 2. Route4-ex 相对原始 Route4 的关系

### 2.1 共同点

`route4-ex` 和原始 `route4` 保持了三件关键事情不变：

1. 仍然直接使用实验侧概率表：
   - [`../src/matlab/Probability.mat`](../src/matlab/Probability.mat)
2. 仍然从 `256` 维计数直方图做 coarse-graining，再进入离散输出认证；
3. 仍然保留偏置生成分布 `q_selected`，以区分生成轮和测试轮。

因此它不是完全换路线，而是在原有 APD / coarse-graining 接口上的模型增强。

### 2.2 关键区别

原始 `route4`：

- 可信输入被建模为 Fock 基对角的相位随机化相干态；
- 因而很多分析天然只“看见”对角部分。

`route4-ex`：

- 可信输入改为完整截断相干态 `|alpha><alpha|`；
- 在 Fock 基下带有非对角元；
- 因而 full-primal 会真正利用输入的相干信息来约束测量。

这就是 `route4-ex` 能比原始 `route4` 给出更强正式认证值的核心原因。

---

## 3. 为什么要做这条扩展路线

动机有两点。

第一，原始 `route4` 的正式认证值始终偏低，更像保守基线，而不适合作为冲击更高熵的主线。

第二，当前 `Probability.mat` 记录的是固定实验硬件下的真实统计分布。  
如果在理论建模中主动把 trusted inputs 压缩成 Fock 对角 Poisson 分布，就会丢掉输入相干结构所能提供的额外约束。

因此，`route4-ex` 的核心目标不是换实验，而是在尽量不改变实验接口的前提下，提升现有数据在认证模型中的利用率。

---

## 4. 当前程序流程

当前 `route4-ex` 的主代码位于：

- [`../src/python/qrng_routes/route4_ex/prototype.py`](../src/python/qrng_routes/route4_ex/prototype.py)

外围搜索与诊断脚本包括：

- [`../src/python/qrng_routes/route4_ex/high_output_model_window_search.py`](../src/python/qrng_routes/route4_ex/high_output_model_window_search.py)
- [`../src/python/qrng_routes/route4_ex/high_output_local_refine.py`](../src/python/qrng_routes/route4_ex/high_output_local_refine.py)
- [`../src/python/qrng_routes/route4_ex/pathology_boundary_scan.py`](../src/python/qrng_routes/route4_ex/pathology_boundary_scan.py)
- [`../src/python/qrng_routes/route4_ex/residual_diagnostics.py`](../src/python/qrng_routes/route4_ex/residual_diagnostics.py)

整体流程可概括为：

1. 从 `Probability.mat` 读取指定输入行；
2. 把 `256` 维原始直方图按给定边界 coarse-grain 成离散输出；
3. 用给定半径和相位构造非对角 trusted coherent states；
4. 建立 full-primal 认证问题；
5. 用 `SCS` 做大范围搜索与局部精修；
6. 用 `MOSEK` 对高值点做正式确认；
7. 对靠近高值尖峰的点进一步做病态边界定位与残差体检。

---

## 5. 结果如何从 `1.2275` 推进到 `1.54`

### 5.1 早期正式点

最早正式确认点是：

- [`../output/qrng_routes/route4_ex_mosek_verify_3out_free_r054_066_072_q100.json`](../output/qrng_routes/route4_ex_mosek_verify_3out_free_r054_066_072_q100.json)

对应：

- 半径 `[0.54,0.66,0.72]`
- `MOSEK optimal`
- `H_min ≈ 1.22750`

这一步已经说明 `route4-ex` 正式过了 `1 bit`。

### 5.2 局部精修

随后对主线附近做了更细的半径搜索：

- [`../output/qrng_routes/route4_ex_high_output_local_refine_fastnear_3out_q100.json`](../output/qrng_routes/route4_ex_high_output_local_refine_fastnear_3out_q100.json)
- [`../output/qrng_routes/route4_ex_high_output_local_refine_3out_q100.json`](../output/qrng_routes/route4_ex_high_output_local_refine_3out_q100.json)

这一阶段表明：

- 主线附近并不是只有一个 `>1` 点；
- 存在一片高值区域；
- 但其中最高的 `1.5` 左右尖峰大多是 `optimal_inaccurate`，还不能直接当正式结论。

### 5.3 病态边界定位

为了解决“高值点到底能不能正式确认”这个问题，又进一步沿

- `[0.54,0.66,0.72] -> [0.535,0.665,0.715]`

做了 `MOSEK-only` 的病态边界剖面扫描。

关键结果包括：

- 稳定点  
  [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q209over512_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q209over512_to_q105over256_2pt.json)  
  半径 `[0.537958984375, 0.662041015625, 0.717958984375]`  
  `MOSEK optimal`，`H_min ≈ 1.53846`

- 当前最高的稳定点  
  [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)  
  半径 `[0.5379541015625, 0.6620458984375, 0.7179541015625]`  
  `MOSEK optimal`，`H_min ≈ 1.54395`

- 失稳点  
  [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q839over2048_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q839over2048_to_q105over256_2pt.json)  
  半径 `[0.53795166015625, 0.66204833984375, 0.71795166015625]`  
  `MOSEK` 失败

这说明：

- 高值正式点不是完全虚假的数值峰；
- 但它们位于一条非常窄的稳定前沿附近；
- 再往失稳端推进极小一步，`MOSEK` 就会直接失败。

---

## 6. 这些 `1.53 ~ 1.54` 的高值点是否数值干净

这是最关键的可信度问题。

为此额外做了约束残差 / 可行性余量检查：

- [`../src/python/qrng_routes/route4_ex/residual_diagnostics.py`](../src/python/qrng_routes/route4_ex/residual_diagnostics.py)

### 6.1 最高稳定点的残差

文件：

- [`../output/qrng_routes/route4_ex_residual_diag_q419over1024.json`](../output/qrng_routes/route4_ex_residual_diag_q419over1024.json)

对应点：

- `[0.5379541015625, 0.6620458984375, 0.7179541015625]`
- `H_min ≈ 1.54395`

主要残差：

- `psd_min_eig_min ≈ -3.53e-09`
- `completeness_violation_max ≈ 5.46e-11`
- `measurement_violation_max ≈ 3.81e-09`
- `strategy_weight_sum ≈ 0.99999999945`

### 6.2 邻近稳定点的残差旁证

文件：

- [`../output/qrng_routes/route4_ex_residual_diag_q209over512.json`](../output/qrng_routes/route4_ex_residual_diag_q209over512.json)

对应点：

- `[0.537958984375, 0.662041015625, 0.717958984375]`
- `H_min ≈ 1.53846`

主要残差：

- `psd_min_eig_min ≈ -1.28e-09`
- `completeness_violation_max ≈ 1.91e-11`
- `measurement_violation_max ≈ 1.38e-09`
- `strategy_weight_sum ≈ 0.99999999985`

### 6.3 对比旧基线点

旧基线点：

- [`../output/qrng_routes/route4_ex_residual_diag_baseline_r054_066_072.json`](../output/qrng_routes/route4_ex_residual_diag_baseline_r054_066_072.json)

对应：

- `[0.54,0.66,0.72]`
- `H_min ≈ 1.22750`

其残差更小，但高值点的残差也仍然只有 `1e-9 ~ 1e-10` 量级，并没有出现明显失控。

因此，当前最合理的判断是：

- 这批 `1.53 ~ 1.54` 的 `MOSEK optimal` 点可以视为数值上足够干净的正式结果；
- 它们不是简单的脏解；
- 但仍应在报告中说明，它们靠近病态边界。

---

## 7. 当前最适合对外汇报的口径

我建议对导师使用下面这组口径。

### 7.1 可以明确写成正式结果

- `route4-ex` 已正式达到并明显超过 `H_min = 1`
- 在当前主线模型下，`MOSEK` 已确认 `H_min` 可达约 `1.54 bit`
- 该结果仍然直接建立在现有 `Probability.mat` 数据接口之上，而不是完全换实验路线

### 7.2 需要同时写清的边界

- 这一高值结果位于一条很窄的稳定前沿上；
- 再向高值失败端推进极小一步，`MOSEK` 就会失败；
- 因此这更像“窄稳定带”，而不是“大平台”。

### 7.3 目前不建议写成主结论的内容

- `optimal_inaccurate` 的 `1.5+` 尖峰
- rigid `[120,140,160]` 那些 `SCS` 很高但 `MOSEK` 未确认的点

这些内容适合作为补充候选，不适合压过当前 `1.54` 的正式主结果。

---

## 8. 实验上的优点与风险

### 8.1 优点

- 仍然复用现有 APD 概率表接口；
- 不需要像 route5 那样立刻切换到新的实验数据形态；
- 在不大改实验接口的前提下，显著提高了正式认证值。

### 8.2 风险

- 当前最强结果依赖 `free_monotone_radii`，而不是 rigid `sqrt(I)` 映射；
- 高值点靠近数值病态边界，后续还需要持续关注求解稳定性；
- `q=[1,0,0]` 应理解为“生成轮偏置”，不能直接当成整机平均吞吐率。

---

## 9. 下一步建议

### 9.1 短期最值得做的事情

1. 把当前 `1.54` 主结果、阈值定位和残差检查统一并入总报告；
2. 再选一两个邻近稳定点做同类残差检查，作为旁证；
3. 给导师汇报时，把“主结果”和“边界条件”一起说清。

### 9.2 如果还想继续推高

后续若还想继续冲高，建议先研究：

- 为什么稳定前沿如此窄；
- 能否通过更稳定的数值表示，把更靠近尖峰的点也转成 `MOSEK optimal`；
- 是否需要在 trusted-input 字母表或离散化边界上做更结构性的升级。

### 9.3 如果长期目标仍是冲 `2 bit`

那就不能只靠继续抠当前这条线，而更可能需要：

- 更丰富的 trusted-input 设计；
- 更强的输出离散化；
- 或直接与 route5/route6 的 coherent alphabet 思路结合。

---

## 10. 一句话总结

`route4-ex` 当前已经从“原始 route4 的扩展想法”，推进成了一条具有正式结果支撑的可行主线：

- 它保留了原始 route4 的实验接口；
- 通过引入非对角 trusted coherent inputs，大幅增强了认证约束；
- 当前 `MOSEK` 已正式确认 `H_min ≈ 1.54395`；
- 因而在“把 route4 主线推进到 `H_min >= 1`”这一阶段目标上，`route4-ex` 已经完成得相当充分。
