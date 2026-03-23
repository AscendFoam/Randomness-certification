# Route 1: Steering QRNG 原型说明

## 1. 这条路线的定位

本目录对应本项目中的路线 1，也就是参考 2022 年 steering QRNG 工作的连续变量随机性认证方案。它的核心特点是：

- 使用高斯纠缠源，例如 TMSV 或 split-SMS。
- Alice 侧做非受信的周期 coarse-grained `q/p` 测量。
- Bob 侧做受信测量，可以是有限设置的 homodyne，也可以用 tomography 作为上界参考。
- 目标不是最激进地追求超高熵，而是在实验上更稳地拿到一段可认证随机性。

这条路线当前最合适的定位，是实验室近期更容易讨论和推进的 steering 随机性验证方案。按照目前数值结果，它更像一条 `0.5 - 1 bit/round` 级别的路线，而不是现阶段最适合冲击 `H_min >= 2` 的路线。

## 2. 物理图像和实验原理

这条路线的物理图像可以概括为：

1. 制备一个双模连续变量纠缠态。
2. Alice 在两个互补象限之间切换，也就是对 `q` 或 `p` 做周期 coarse-graining。
3. Bob 做受信的连续变量测量。
4. 通过观测到的 assemblage 或 Bob 的测量统计，解一个 steering SDP，得到 Eve 对 Alice 输出的最优猜测概率 `p_guess`。
5. 再由 `H_min = -log2(p_guess)` 得到可认证的 min-entropy。

本目录里的实现支持两类高斯源：

- `tmsv`: 双模压缩真空态。
- `split_sms`: 单模压缩态经过 50:50 分束器后的双模态。

为什么要同时支持这两类源：

- 从理论上看，两者都可以用来做 steering QRNG。
- 从实验上看，它们对应不同的实现习惯和器件路径。
- 从目前的原型结果看，在相近资源下，TMSV 明显比 split-SMS 更强，因此当前建议把 TMSV 作为主 baseline。

## 3. 本目录代码结构

本目录的核心文件是 [steering_2022.py](./steering_2022.py)。

主要函数如下：

- `periodic_binning_povms(...)`
  为 Alice 构造周期 coarse-grained quadrature POVM。
- `nonperiodic_binning_povms(...)`
  为 Bob 构造有限范围的受信 homodyne binning POVM。
- `bob_homodyne_angles(...)`
  给出 Bob 的多个 trusted homodyne 角度，目前是在 `q` 到 `p` 之间均匀取点。
- `assemblage_tomography(...)`
  计算 tomography 版本需要的 assemblage。
- `joint_probabilities(...)`
  计算 `P[x,a,y,b]`。
- `guessing_prob_sdp_tomography(...)`
  用 tomography 数据解 steering guessing-probability SDP。
- `guessing_prob_sdp_homodyne(...)`
  用 Bob 的 trusted homodyne 统计解 SDP。
- `run_route1(...)`
  这是当前的主入口，负责生成状态、构造测量、扫描 `T_q` 并返回最佳点。
- `sweep_route1_eta(...)`
  用于在不同总效率 `eta` 下做参数扫描。

本路线依赖的公共函数在 [common.py](../common.py) 中，包括：

- 高斯态构造，例如 `tmsv_density(...)`、`split_sms_density(...)`
- 纯损耗模型 `apply_symmetric_loss(...)`
- quadrature 算符 `quadrature_op(...)`
- SDP 求解接口 `solve_cvxpy_problem(...)`

## 4. 代码和实验的对应关系

代码里的参数可以大致这样理解：

- `dimension`
  截断 Fock 空间维数。它不是实验的真实维数，而是数值求解中的截断。
- `squeezing_db`
  压缩强度，单位 dB。
- `eta`
  双模对称损耗后的总透过率或总效率。
- `num_alice_bins`
  Alice 周期 coarse-graining 后的输出数。
- `num_bob_bins`
  Bob 每个 trusted homodyne 设置的离散 bin 数。
- `num_bob_settings`
  Bob trusted homodyne 的设置数。这个量越大，通常越接近 tomography 上界，但实验复杂度也更高。
- `T_q`
  Alice 周期 coarse-graining 的周期长度。当前代码会在一个候选网格上扫描并选最好值。

Tomography 模式和 homodyne 模式的含义不同：

- `bob_mode="tomography"`
  更接近“理论上界”或“信息足够多时能做到什么”。
- `bob_mode="homodyne"`
  更接近实验真实可做的版本。

因此，在做实验判断时，应优先看 homodyne 结果；tomography 结果主要用来理解上限、判断是否还有继续优化的空间。

## 5. 本轮数值结果

本轮最重要的路线 1 扫描结果保存在：

- `output/qrng_routes/route1_tmsv_sweep.json`
- `output/qrng_routes/route1_tmsv_sweep.png`
- `output/qrng_routes/route1_split_sms_spotcheck.json`

主扫描采用的 TMSV 参数为：

- `dimension = 5`
- `squeezing_db = -4.0`
- `num_alice_bins = 6`
- `num_bob_bins = 8`
- `T_q = 4.0`

主结果如下：

| `eta` | tomography | homodyne `m_B=2` | homodyne `m_B=4` | homodyne `m_B=6` |
|---|---:|---:|---:|---:|
| 0.80 | 0.384 | 0.066 | 0.333 | 0.362 |
| 0.85 | 0.526 | 0.128 | 0.450 | 0.495 |
| 0.90 | 0.705 | 0.205 | 0.603 | 0.668 |
| 0.95 | 0.929 | 0.313 | 0.817 | 0.895 |
| 1.00 | 1.177 | 0.493 | 1.177 | 1.177 |

这组结果最重要的结论有三条：

- 增加 Bob 的 trusted homodyne 设置数是有效的。
- `m_B=6` 时已经非常接近 tomography 上界。
- 在 `eta=0.9` 左右，这条路线已经能比较稳地进入 `0.5 - 1 bit/round` 区间。

split-SMS 的 spot check 结果较弱：

| source | `eta` | tomography | homodyne `m_B=2` |
|---|---:|---:|---:|
| split-SMS | 0.80 | 0.121 | 0.008 |
| split-SMS | 0.90 | 0.295 | 0.048 |
| split-SMS | 1.00 | 1.155 | 0.395 |

因此，当前更建议实验优先考虑 TMSV，而不是 split-SMS。

## 6. 对实验室的意义

路线 1 的优势是：

- 物理图像清楚，和 steering 文献联系紧密。
- 高斯源和 homodyne 检测对实验室来说通常比较熟悉。
- 它不要求中心站实现高度结构化的高输出联合 POVM。
- 当前数值结果已经足够支撑一个“近期可做”的实验讨论。

路线 1 的不足是：

- 它目前更像中等随机性路线，而不是极高熵路线。
- 真实实验只能做 homodyne 版本，不能直接达到 tomography 上界。
- 目前最激进的补充 tomography 检查也只到 `H_min ≈ 1.44`，因此它距离 `H_min >= 2` 还有明显差距。

换句话说，如果实验室的硬指标就是 `H_min >= 2`，路线 1 不应该作为当前主攻路线；但如果实验室接受“先做稳妥版本，再考虑更高熵路线”，那路线 1 是一个很好的近期目标。

## 7. 工程上可行的地方

从工程角度看，这条路线最容易落地的原因主要是：

- 只需要在已知 CV 平台上准备高斯纠缠源和 homodyne 测量。
- Bob 侧的 trusted measurement 可以逐步增加设置数，而不是一步到位做复杂联合测量。
- 参数扫描清晰，实验室容易根据 `eta`、压缩强度、binning 数目去做敏感性评估。

当前最推荐的实验讨论起点是：

- `source = "tmsv"`
- `bob_mode = "homodyne"`
- `dimension = 5`
- `squeezing_db = -4.0`
- `eta` 以 `0.9` 附近为重点
- `num_alice_bins = 6`
- `num_bob_bins = 8`
- `num_bob_settings = 6`

这组参数不是唯一可行点，但它是目前最稳的一组参考起点。

## 8. 工程困难、局限和风险

需要明确注意以下几点：

- 数值中的 `dimension` 是截断维数，不等于真实实验的全空间，因此所有结论都依赖截断近似。
- Bob 的 homodyne 离散化是数值理想化模型，实验上的 bin 边界、电子噪声和漂移会降低结果。
- 总效率 `eta` 对结果非常敏感，尤其在 `0.8` 到 `0.95` 区间。
- 随着 `num_bob_settings` 和 `num_bob_bins` 增大，实验标定和统计采集工作量也会上升。
- 这条路线不太像“靠单纯调参就能突然冲到 2 bit”，因此不建议把它作为当前的高熵主线。

还有一个代码层面的重要说明：

- 这份脚本中的 no-signalling 约束已经修复了旧 notebook 里“只对 `e=0` enforce”的 bug。
- 因此，本目录结果比旧 notebook 更可信，不能简单用旧 notebook 的数值直接比较高低。

## 9. 如何运行

推荐环境：

```powershell
conda activate DLEnv
$env:PYTHONPATH='D:\Codes\Quantum\Randomness-certification\src\python'
```

除了统一入口 `qrng_routes.main` 之外，现在也可以直接从本路线目录单独运行：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route1 --help
```

运行一个路线 1 点：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.main `
  --mode route1 `
  --source tmsv `
  --bob-mode homodyne `
  --dimension 5 `
  --squeezing-db -4.0 `
  --eta 0.9 `
  --alice-bins 6 `
  --bob-bins 8 `
  --bob-settings 6 `
  --solver MOSEK
```

使用路线 1 的独立入口，等价命令是：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route1 `
  --mode single `
  --source tmsv `
  --bob-mode homodyne `
  --dimension 5 `
  --squeezing-db -4.0 `
  --eta 0.9 `
  --alice-bins 6 `
  --bob-bins 8 `
  --bob-settings 6 `
  --solver MOSEK
```

如果只想看上界参考，可以改成：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.main `
  --mode route1 `
  --source tmsv `
  --bob-mode tomography `
  --dimension 5 `
  --squeezing-db -4.0 `
  --eta 0.9 `
  --alice-bins 6 `
  --bob-bins 8 `
  --solver MOSEK
```

如果想直接从路线 1 独立入口做 `eta` 扫描，可以使用：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route1 `
  --mode sweep-eta `
  --source tmsv `
  --bob-mode homodyne `
  --dimension 5 `
  --squeezing-db -4.0 `
  --alice-bins 6 `
  --bob-bins 8 `
  --bob-settings 6 `
  --eta-values 0.8 0.85 0.9 0.95 1.0 `
  --tq-grid 2 4 6 `
  --solver MOSEK
```

如果要编程方式批量扫描，可以直接导入：

```python
import numpy as np
from qrng_routes.route1 import sweep_route1_eta

results = sweep_route1_eta(
    source="tmsv",
    bob_mode="homodyne",
    eta_values=np.array([0.8, 0.85, 0.9, 0.95, 1.0]),
    dimension=5,
    squeezing_db=-4.0,
    num_alice_bins=6,
    num_bob_bins=8,
    tq_grid=np.array([2.0, 4.0, 6.0]),
    num_bob_settings=6,
    preferred_solver="MOSEK",
)
```

## 10. 推荐的下一步

如果实验室把路线 1 作为近期方案，建议按以下顺序推进：

1. 先围绕 TMSV 做一组实验可达参数表，重点是总效率、可实现压缩、Bob 设置数。
2. 用 `m_B=4` 和 `m_B=6` 分别做复杂度与收益对比。
3. 把本路线作为 steering 验证和平台校准方案。
4. 不要把路线 1 当成当前最有希望实现 `H_min >= 2` 的主线。

一句话总结：

路线 1 是当前三条路线里最稳、最容易和实验讨论对接的一条，但它更像“稳健拿到中等随机性”的方案，而不是“最优高熵”方案。
