# 非 Route 2 路线接近 `H_min = 2` 的数值检查

## 结论先行

截至当前代码版本和本轮数值检查，结论很明确：

- `route1` 没有显示出能逼近 `2 bit` 的趋势。
- `route3` 在当前 CV 硬件 + 单设备 MDI 分析框架下，也没有显示出能逼近 `2 bit` 的趋势。
- `route4` 离 `2 bit` 更远。

因此，如果实验室的硬目标仍然是：

```text
H_min >= 2
```

那么主线仍应放在 `route2`。

## 1. Route 1

### 1.1 更接近论文参数的 paper-like sweep

已生成文件：

- [route1_tmsv_paper_like_d5_oA8_oB16.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_tmsv_paper_like_d5_oA8_oB16.json)
- [route1_tmsv_paper_like_d5_oA8_oB16.png](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_tmsv_paper_like_d5_oA8_oB16.png)

参数：

- `source = tmsv`
- `dimension = 5`
- `squeezing_db = -4.0`
- `o_A = 8`
- `o_B = 16`
- `T_q in {2,4,6,8,10}`
- `eta in {0.8, 0.9, 1.0}`

最重要的现象有两个：

1. 最优 `T_q` 稳定落在 `6.0`。
2. `m_B = 6` 与 tomography 已经几乎重合。

代表性数值：

- `eta = 0.8`
  tomography `H_min = 0.2153`
  `m_B = 2` 时 `0.1872`
  `m_B = 4` 时 `0.2126`
  `m_B = 6` 时 `0.2153`
- `eta = 0.9`
  tomography `H_min = 0.4213`
  `m_B = 2` 时 `0.3449`
  `m_B = 4` 时 `0.4153`
  `m_B = 6` 时 `0.4213`
- `eta = 1.0`
  tomography `H_min = 0.8165`
  `m_B = 2` 时 `0.6974`
  `m_B = 4` 时 `0.8136`
  `m_B = 6` 时 `0.8163`

### 1.2 更激进点位检查

额外测试了更激进的 tomography 点位：

- `source = tmsv`
- `dimension = 5`
- `squeezing_db = -6.0`
- `num_alice_bins = 10`
- `eta = 1.0`

结果：

- `H_min = 1.0622`

再把截断增到 `dimension = 6`：

- `H_min = 1.1671`

### 1.3 判断

这说明：

- `route1` 即使往更激进方向推，也没有显示出接近 `2 bit` 的趋势。
- 它更像一条适合做稳健 steering 认证的中等熵路线，而不是冲击 `2 bit` 的主线。

## 2. Route 3

### 2.1 已有相位 sweep 结果

已有文件：

- [route3_phase_sweep.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_phase_sweep.json)

当前 `2 x 2` 输出下：

- `4` 相位：`H_min = 0.5471`
- `5` 相位：`H_min = 0.5891`
- `6` 相位：`H_min = 0.6343`

这说明增加可信相位态确实有帮助，但提升幅度有限。

### 2.2 本轮额外检查：增加输出分箱

#### 四相位，8 输出，`2 x 4` 分箱

参数：

- `mu = 0.02`
- `cutoff = 8`
- `num_phases = 4`
- `num_x_bins = 2`
- `num_p_bins = 4`

结果：

- `raw_H_min = 2.0460`
- `certified H_min = 0.5796`

#### 四相位，9 输出，`3 x 3` 分箱

参数：

- `mu = 0.02`
- `cutoff = 8`
- `num_phases = 4`
- `num_x_bins = 3`
- `num_p_bins = 3`

结果：

- `raw_H_min = 0.5297`
- `certified H_min = 0.3327`

#### 六相位，4 输出，`2 x 2` 分箱

参数：

- `mu = 0.02`
- `cutoff = 10`
- `num_phases = 6`

结果：

- `certified H_min = 0.6472`

#### 六相位，8 输出，`2 x 4` 分箱

参数：

- `mu = 0.02`
- `cutoff = 10`
- `num_phases = 6`
- `num_x_bins = 2`
- `num_p_bins = 4`

结果：

- `raw_H_min = 2.0427`
- `certified H_min = 0.6594`

### 2.3 判断

本轮最重要的观察是：

- `route3` 不是“输出数一增加，认证熵就会逼近 2”。
- 即使 raw 熵看起来超过 `2`，认证后的 `H_min` 仍然只在 `0.58 - 0.66` 左右。

因此当前这条 route3 的主要瓶颈不是“输出数不够”这么简单，而更像是：

- trusted input family 的信息完备性仍然不够强；
- central POVM 的物理结构虽然更一致了，但对 Eve 的自由度约束仍然不足；
- 所以 raw 随机性不能顺利转化成高 certified 随机性。

## 3. Route 4

本轮重新跑了一个输出数 sweep：

- `selected_mu = [100, 120, 140]`
- `q = [0.25, 0.25, 0.5]`
- `prob_floor = 1e-12`

结果：

- `N = 4` 时 `H_min = 0.1611`
- `N = 6` 时 `H_min = 0.3318`
- `N = 8` 时 `H_min = 0.3432`

这和之前 route4 的判断一致：

- 它离 `2 bit` 非常远；
- 更适合作为 APD 平台诊断路线，而不是冲击高熵的主线。

## 4. 总判断

如果只关心“除了 route2 之外，其他路线有没有现实希望接近 `H_min = 2`”，那么当前答案是：

- `route1`: 没有看到接近 `2` 的趋势。
- `route3`: 没有看到接近 `2` 的趋势。
- `route4`: 更没有接近 `2` 的趋势。

从投入产出比看：

- `route1` 适合做稳健 steering 基线。
- `route3` 适合做保留 CV 硬件外形的过渡探索。
- `route4` 适合做 APD 数据诊断。
- 真正值得作为 `H_min >= 2` 主线推进的，仍然是 `route2`。
