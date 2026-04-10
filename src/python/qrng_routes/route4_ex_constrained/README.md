# Route4-ex-constrained

`route4_ex_constrained` 是把 `route4_ex` 收缩后的核心版本。

它只保留一条固定主线：

1. 概率数据直接来自 `src/matlab/Probability.mat`
2. 输入窗口固定为原 `route4` 菜单中的一个子集
3. coarse-graining 边界固定
4. trusted coherent alphas 固定
5. 正式认证仍然用 non-diagonal trusted inputs + full primal

它刻意不再包含：

- 自由半径搜索
- 相位图样扫描
- 输入窗口大规模扫描
- 高熵边界搜索
- toy / APD-like 内建概率后端

默认主线参数是此前已经通过 `MOSEK` 复核、且能保持 `H_min > 1` 的一个简洁基线点：

- `selected_mu = [100, 120, 140]`
- `q = [1, 0, 0]`
- `alpha = [0.54, 0.66 i, -0.72]`
- `custom_edges = [0, 121, 132, 256]`
- `cutoff = 6`

## 典型命令

准备实例摘要：

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_ex_constrained \
  --mode prepare-instance
```

直接运行 formal full primal：

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_ex_constrained \
  --mode full-primal \
  --solver MOSEK
```

比较同一 constrained 实例上的 diagonal/full：

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_ex_constrained \
  --mode compare \
  --solver MOSEK
```

如果希望给导师看一份更贴近 Matlab 单文件阅读顺序的版本，可以直接运行：

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_ex_constrained.matlab_style_reference \
  --solver MOSEK
```

对应代码文件是：

- `src/python/qrng_routes/route4_ex_constrained/matlab_style_reference.py`

这份脚本会按“参数配置 -> 读 Probability.mat -> 构造 rho/rho_diag -> coarse-graining
-> 生成 LambdaIndices -> 求解 diagonal/full primal”的顺序输出结果，方便逐段对照
`guessprobprimal_phaseinsensitive_original.m`。

如果导师更希望直接拿 Matlab 文件运行，则可以使用：

- `src/matlab/guessprobprimal_route4_ex_constrained.m`

这份 `.m` 文件和当前 constrained 主线保持同一套默认参数：

- `selected_mu_list = [100, 120, 140]`
- `q_selected = [1, 0, 0]`
- `custom_edges = [0, 121, 132, 256]`
- `alpha_values = [0.54, 0.66 i, -0.72]`
- `M = 6`

按 `MOSEK` 的正式口径，预期主结果仍应为 `H_min ≈ 1.2275008643`。

如果振幅以 `-` 开头，请像 `-0.72+0j` 这样加引号，避免被 shell 误判为命令行选项。
