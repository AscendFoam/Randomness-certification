# Route 3: CV Hardware + Single-Device MDI Analysis

## 1. 这条路线在做什么

`route3` 保留连续变量实验硬件的直觉：

- Alice 和 Bob 发送可信相干态输入。
- 中央测量近似为 beam splitter 后接 `q/p` 双 homodyne 粗粒化。
- 安全分析不再使用旧 `mdi_qrng` 的局域拆分约束。
- 认证端仍然采用单设备 prepare-and-measure MDI SDP。

当前代码和之前版本最大的区别是：

- 现在 `P(c|x,y)` 是从同一套截断相干态、同一个 beam splitter、同一组粗粒化 POVM 直接算出来的。
- 不再使用和实际输入态支撑子空间脱节的 Gaussian surrogate。

也就是说，现在的流程是统一的：

`可信输入态 -> 中央 CV Bell / dual-homodyne POVM -> reduced support -> SDP`

## 2. 文件结构

- [`cv_four_phase.py`](./cv_four_phase.py)
  路线 3 的核心实现。
- [`main.py`](./main.py)
  独立命令行入口。
- [`__main__.py`](./__main__.py)
  允许直接使用 `python -m qrng_routes.route3`。

## 3. 核心函数

- `phase_alphabet(...)`
  构造每边的相位相干态字母表。
- `reduced_joint_inputs(...)`
  将 product inputs 投影到 trusted alphabet 的精确支撑子空间。
- `quadrature_povms_from_bounds(...)`
  根据边界数组构造粗粒化 quadrature POVM。
- `dual_homodyne_povm(...)`
  用 beam splitter 加 `q/p` binning 生成中心 POVM。
- `project_povm_to_basis(...)`
  将中心 POVM 投影到输入支撑子空间。
- `dual_homodyne_probabilities(...)`
  直接从 reduced states 和 reduced POVM 计算 `P(c|x,y)`。
- `run_route3(...)`
  完整运行一次路线 3 的统计生成与单设备 SDP 认证。

## 4. 重要参数

- `mu`
  相干态平均光子数，振幅大小为 `sqrt(mu)`。
- `cutoff`
  Fock 截断。
- `num_phases`
  每边相位态个数。`4` 表示四相位，`6` 表示六相位。
- `num_x_bins`, `num_p_bins`
  中央双 homodyne 在 `q` 和 `p` 两个方向的离散 bin 数。
- `x_bounds`, `p_bounds`
  手动指定 bin 边界。如果不提供，就按默认边界生成。
- `quadrature_range`
  当未手动指定边界时，用于生成有限中心区间。
- `num_quadrature_nodes`
  构造 quadrature POVM 时使用的数值积分节点数。
- `max_inputs_to_certify`
  要做 SDP 认证的目标输入个数。默认 CLI 为 `1`，便于快速扫参数。

## 5. 命令行入口

先激活环境：

```powershell
conda activate DLEnv
$env:PYTHONPATH='D:\Codes\Quantum\Randomness-certification\src\python'
```

查看帮助：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 --help
```

### 5.1 单次运行

四相位、`2 x 2` 输出的最小示例：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 `
  --mode single `
  --mu 0.05 `
  --cutoff 12 `
  --num-phases 4 `
  --max-inputs 1 `
  --solver MOSEK
```

六相位、增大 `q/p` 分箱：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 `
  --mode single `
  --mu 0.05 `
  --cutoff 12 `
  --num-phases 6 `
  --num-x-bins 3 `
  --num-p-bins 3 `
  --quadrature-range 3.5 `
  --max-inputs 2 `
  --solver MOSEK
```

手动指定边界：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 `
  --mode single `
  --mu 0.05 `
  --cutoff 12 `
  --num-phases 4 `
  --num-x-bins 3 `
  --num-p-bins 3 `
  --x-bounds -5  -0.5  0.5  5 `
  --p-bounds -5  -0.5  0.5  5 `
  --solver MOSEK
```

结果写入 JSON：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 `
  --mode single `
  --mu 0.05 `
  --cutoff 12 `
  --num-phases 6 `
  --output-json output/qrng_routes/route3_single_6phase.json `
  --solver MOSEK
```

### 5.2 相位数 sweep

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 `
  --mode phase-sweep `
  --mu 0.05 `
  --cutoff 12 `
  --phase-values 4 5 6 `
  --max-inputs 1 `
  --solver MOSEK `
  --output-json output/qrng_routes/route3_phase_sweep.json
```

### 5.3 `mu` sweep

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.route3 `
  --mode mu-sweep `
  --num-phases 6 `
  --cutoff 12 `
  --mu-values 0.02 0.05 0.10 `
  --max-inputs 1 `
  --solver MOSEK `
  --output-json output/qrng_routes/route3_mu_sweep_6phase.json
```

## 6. 统一入口

也可以从统一入口调用：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.main `
  --mode route3 `
  --route3-mode single `
  --mu 0.05 `
  --cutoff 12 `
  --num-phases 6 `
  --num-x-bins 3 `
  --num-p-bins 3 `
  --quadrature-range 3.5 `
  --max-inputs 2 `
  --solver MOSEK `
  --output-json output/qrng_routes/route3_single_6phase.json
```

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m qrng_routes.main `
  --mode route3 `
  --route3-mode phase-sweep `
  --mu 0.05 `
  --cutoff 12 `
  --phase-values 4 5 6 `
  --num-x-bins 2 `
  --num-p-bins 2 `
  --solver MOSEK `
  --output-json output/qrng_routes/route3_phase_sweep.json
```

## 7. 结果字段解释

单次运行会返回：

- `p_guess`
  认证后的猜测概率。
- `H_min`
  认证后的最小熵。
- `target_input`
  本次真正被 SDP 认证的目标输入 `(x, y)`。
- `raw_H_min`
  对应目标输入只看观测统计时的原始熵。
- `num_inputs`
  总输入数，通常是 `num_phases^2`。
- `num_outputs`
  中央测量离散输出总数，等于 `num_x_bins * num_p_bins`。
- `output_labels`
  每个离散输出对应的 `(x_bin, p_bin)` 标签。
- `local_rank`
  单边 trusted input 字母表在截断空间中的有效秩。
- `joint_dim`
  reduced joint support 的维数。
- `operator_span_rank`
  输入态算符张成空间的秩。
- `operator_space_dim`
  reduced joint space 上全部算符空间维数。
- `x_bounds`, `p_bounds`
  本次真正使用的 bin 边界。
- `num_quadrature_nodes`
  quadrature POVM 数值积分节点数。
- `num_inputs_certified`
  这次实际做了多少个目标输入的 SDP 认证。

## 8. 当前实现相对旧版本的关键修正

- 不再使用与真实输入态支撑空间无关的 Gaussian surrogate。
- 中央统计现在来自真正的截断相干态和 beam-splitter 后 POVM。
- `cutoff` 变化会真实影响状态、支撑子空间和统计。
- 默认仍然使用单设备 MDI SDP，因此不会重新引入旧 `C3/C4` 约束问题。

## 9. 实验与数值上的理解

这条路线的价值主要在于：

- 硬件上比较贴近 CV Bell / dual-homodyne 直觉。
- 可以自然扩展可信输入集，从四相位推进到六相位甚至更多。
- 安全分析是 route2 那套单设备 POVM 逻辑，而不是旧的局域拆分。

但它和 route2 仍有明显区别：

- route2 是最干净的 qubit baseline，更容易冲击或接近 `2 bit`。
- route3 更强调“保留 CV 硬件外形，同时改正安全分析”。
- 如果实验室要优先追求 `H_min >= 2` 的清晰基线，通常还是 route2 更直接。

## 10. 一段最小 Python 调用示例

```python
from qrng_routes.route3 import run_route3

result = run_route3(
    mu=0.05,
    cutoff=12,
    num_phases=6,
    num_x_bins=2,
    num_p_bins=2,
    max_inputs_to_certify=1,
    preferred_solver="MOSEK",
)

print(result["H_min"])
print(result["output_labels"])
```
