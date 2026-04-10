# Route4-ex 病态边界阈值定位与残差检查

本文档记录 `route4-ex` 在主线窗口 `[100,120,140]`、`3` 输出边界 `[0,121,132,256]`、相位模式 `0_pi2_pi`、偏置 `q=[1,0,0]` 下的两项后续诊断：

1. 沿自由半径主线做 `MOSEK` 病态边界定位；
2. 对高值稳定点做约束残差 / 可行性余量检查。

---

## 1. 背景

此前已经确认：

- 基线正式确认点  
  [`../output/qrng_routes/route4_ex_mosek_verify_3out_free_r054_066_072_q100.json`](../output/qrng_routes/route4_ex_mosek_verify_3out_free_r054_066_072_q100.json)  
  半径 `[0.54,0.66,0.72]`，`MOSEK optimal`，`H_min ≈ 1.22750`

- 更高的 `SCS` 候选点接近  
  `[0.535,0.665,0.715]`，但 `MOSEK` 直接失败。

因此需要回答两个问题：

1. `MOSEK` 失败到底发生在什么位置；
2. 如果能在失败前找到更高的 `MOSEK optimal` 点，这些点的残差是否足够干净。

---

## 2. 阈值定位方法

沿下述线段做二分式 `MOSEK-only` 探针：

- 稳定端：`[0.54,0.66,0.72]`
- 高值失败端：`[0.535,0.665,0.715]`

使用脚本：

- [`../src/python/qrng_routes/route4_ex/pathology_boundary_scan.py`](../src/python/qrng_routes/route4_ex/pathology_boundary_scan.py)

本轮实际只使用 `--skip-scs`，即只检查 `MOSEK` 是否还能给出 `optimal`。

---

## 3. 阈值定位结果

### 3.1 已确认的稳定点

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q13over32_to_q7over16_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q13over32_to_q7over16_2pt.json)  
  半径 `[0.53796875, 0.66203125, 0.71796875]`  
  `MOSEK optimal`，`H_min ≈ 1.52993`

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q209over512_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q209over512_to_q105over256_2pt.json)  
  半径 `[0.537958984375, 0.662041015625, 0.717958984375]`  
  `MOSEK optimal`，`H_min ≈ 1.53846`

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q419over1024_to_q105over256_2pt.json)  
  半径 `[0.5379541015625, 0.6620458984375, 0.7179541015625]`  
  `MOSEK optimal`，`H_min ≈ 1.54395`

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q1677over4096_to_q419over1024_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q1677over4096_to_q419over1024_2pt.json)  
  半径 `[0.53795654296875, 0.66204345703125, 0.71795654296875]`  
  `MOSEK optimal`，`H_min ≈ 1.54052`

### 3.2 已确认的失稳点

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q839over2048_to_q105over256_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q839over2048_to_q105over256_2pt.json)  
  半径 `[0.53795166015625, 0.66204833984375, 0.71795166015625]`  
  `MOSEK` 失败

- [`../output/qrng_routes/route4_ex_pathology_boundary_scan_q105over256_to_q53over128_2pt.json`](../output/qrng_routes/route4_ex_pathology_boundary_scan_q105over256_to_q53over128_2pt.json)  
  半径 `[0.53794921875, 0.66205078125, 0.71794921875]`  
  `MOSEK` 失败

### 3.3 当前阈值区间

结合上面的二分结果，可以把当前 `MOSEK` 稳定/失稳转折定位到非常窄的窗口：

- 第一半径 `r1 ∈ [0.53795166, 0.53795410]`
- 第二半径 `r2 ∈ [0.66204590, 0.66204834]`
- 第三半径 `r3 ∈ [0.71795166, 0.71795410]`

更准确地说：

- `[0.5379541015625, 0.6620458984375, 0.7179541015625]` 仍然 `MOSEK optimal`
- `[0.53795166015625, 0.66204833984375, 0.71795166015625]` 已经 `MOSEK` 失败

所以这条高值主线确实存在一条非常窄的“稳定前沿”。

---

## 4. 为什么 `distribution-only H_min` 不是这里的上界

在这些结果文件中，`distribution_only_H_min ≈ 1.53819`，而个别 `MOSEK optimal` 点给出了略高于它的值，例如 `1.54395`。

这不是自动矛盾。

原因是此处 `q=[1,0,0]` 只把第一个输入作为生成输入，但 formal SDP 仍然同时使用另外两个输入行的观测约束。  
因此：

- `distribution-only H_min` 只反映目标行自身输出分布的最大概率；
- formal `H_min` 则利用了其它输入行对 Eve 策略的额外限制；
- 所以 formal 值可以略高于目标行单独的 `distribution-only` 值。

真正需要检查的不是“有没有略高于单行 distribution-only”，而是这些 formal 点的约束残差是否仍然足够小。

---

## 5. 残差检查方法

新增脚本：

- [`../src/python/qrng_routes/route4_ex/residual_diagnostics.py`](../src/python/qrng_routes/route4_ex/residual_diagnostics.py)

该脚本对给定半径点重新构造 full-primal，并在 `MOSEK` 求解后输出：

- PSD 变量最小特征值；
- PSD 约束 violation；
- 完备性约束 `Σ_c M_{c,λ} = s_λ I` 的 violation；
- 观测概率匹配约束的 violation；
- `strategy_weights` 的归一化情况。

---

## 6. 残差检查结果

### 6.1 高值稳定点

文件：

- [`../output/qrng_routes/route4_ex_residual_diag_q419over1024.json`](../output/qrng_routes/route4_ex_residual_diag_q419over1024.json)

对应点：

- 半径 `[0.5379541015625, 0.6620458984375, 0.7179541015625]`
- `MOSEK optimal`
- `H_min ≈ 1.54395`

残差摘要：

- `psd_min_eig_min ≈ -3.53e-09`
- `psd_violation_max = 0`
- `completeness_violation_max ≈ 5.46e-11`
- `completeness_direct_fro_max ≈ 1.54e-10`
- `measurement_violation_max ≈ 3.81e-09`
- `measurement_direct_abs_max ≈ 3.81e-09`
- `strategy_weight_sum ≈ 0.99999999945`

### 6.2 旧基线稳点

文件：

- [`../output/qrng_routes/route4_ex_residual_diag_baseline_r054_066_072.json`](../output/qrng_routes/route4_ex_residual_diag_baseline_r054_066_072.json)

对应点：

- 半径 `[0.54,0.66,0.72]`
- `MOSEK optimal`
- `H_min ≈ 1.22750`

残差摘要：

- `psd_min_eig_min ≈ -8.49e-11`
- `psd_violation_max = 0`
- `completeness_violation_max ≈ 1.53e-12`
- `measurement_violation_max ≈ 9.02e-11`
- `strategy_weight_sum ≈ 0.99999999996`

### 6.3 对比结论

高值稳定点的残差确实比旧基线点更大，但量级仍然只有：

- 测量残差：`~ 1e-9`
- 完备性残差：`~ 1e-10`
- PSD 最小负特征值：`~ 1e-9`

这仍然属于非常小的数值误差，并没有出现明显失控的约束破坏。

因此，从“约束残差是否过大”这个角度看，这批 `1.53 ~ 1.54` 点并不能简单地视为脏解。

---

## 7. 当前判断

### 7.1 可以写进正式报告的主结果

当前完全可以把如下结果写进正式报告主线：

- `route4-ex` 在窗口 `[100,120,140]`、`3` 输出边界 `[0,121,132,256]`、相位 `0_pi2_pi`、偏置 `q=[1,0,0]`、自由单调半径下，
- 已经由 `MOSEK` 给出一条稳定高值前沿，
- 其中可明确引用的高值点为  
  半径 `[0.5379541015625, 0.6620458984375, 0.7179541015625]`，
  `H_min ≈ 1.54395`。

### 7.2 仍需保留的谨慎口径

虽然这些点数值上已经相当干净，但仍应在正式表述里写清两件事：

1. 它们位于一条非常窄的稳定前沿附近；
2. 再往失稳端推进极小一步，`MOSEK` 就会直接失败。

因此，更严谨的说法不是“存在一个宽阔高平台”，而是：

- 在当前模型和数据下，已经找到一段可由 `MOSEK` 正式确认的高值稳定带；
- 其正式值可达约 `1.54 bit`；
- 但这条稳定带非常窄，数值上接近 full-primal 的病态边界。

---

## 8. 下一步建议

后续不宜再单纯追更高数值，而应做下面两件事：

1. 选取 `1.54` 附近两个稳定点，再做一次同类残差检查，确认结果不是单点偶然；
2. 把这轮“阈值定位 + 残差体检”结论合并回 route4-ex 阶段报告，作为主结果的数值可信度说明。
