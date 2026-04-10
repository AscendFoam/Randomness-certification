# Route4 strict non-diagonal

这条路线是一个“最小改动版”的 route4 扩展。

它的设计原则是：

1. 保留原 route4 / Matlab 的实验接口：
   - `selected_mu_list`
   - `q_selected`
   - `Probability.mat`
   - 等分 coarse-graining
2. 不引入 `route4-ex` 里的自由搜索层：
   - 不扫 `max_abs_alpha`
   - 不扫自由半径
   - 不扫自定义高熵边界
3. 只做一件关键修改：
   - 把 trusted input 从 Fock 对角 Poisson 模型换成固定 coherent states
4. 同时允许 full-primal 的一般 Hermitian PSD POVM。

## 技术定义

给定 `selected_mu_list = [mu_1, ..., mu_D]`，本路线先固定

- `mean_photon_numbers = mean_photons_per_mu_label * selected_mu_list`
- `alpha_x = sqrt(mean_photon_numbers[x]) * exp(i * phase_x)`

然后构造完整截断 coherent states：

- `rho_x = |alpha_x><alpha_x|`

默认不搜索 `alpha`，只使用这组固定映射。

## 为什么要做支撑降维

如果直接在原始 Fock 截断空间上做 full-primal，规模会非常大。
但所有输入态都只落在它们自己的张成子空间中，因此我们可以把 full-primal
严格等价地投影到该支撑子空间上求解。

这会显著降低 full-primal 的维度，而不改变：

- 目标函数
- 统计约束
- 可行性

## 当前提供的模式

- `prepare-instance`
  只构造实例并输出关键信息。
- `full-primal-single`
  直接运行 strict non-diagonal full-primal。
- `compare-reference`
  对比：
  - 原 route4 的 phase-insensitive diagonal primal
  - strict non-diagonal route4 的 full-primal

## 一个典型命令

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_strict_nondiagonal \
  --mode compare-reference \
  --selected-mu 100 120 140 \
  --q-values 0.25 0.25 0.5 \
  --num-outputs 4 \
  --cutoff 280 \
  --solver MOSEK
```

如果后续实验确认 `mu` 标签需要先除以 100 才是真实平均光子数，可以显式指定：

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_strict_nondiagonal \
  --mode compare-reference \
  --selected-mu 100 120 140 \
  --mean-photons-per-mu-label 0.01
```
