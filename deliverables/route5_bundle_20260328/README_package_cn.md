# Route5 交付包说明

这个压缩包用于单独交付 `route5` 的代码、结果和报告，方便在解压后直接查阅和复现实验前的数值流程。

## 目录结构

- `docs/`
  - `route5_detailed_technical_report_cn.md`
  - `cv_bell_integral_vs_trace_probability_check_cn.md`
  - `SDP_solve.tex`
- `src/python/qrng_routes/`
  - `route5` 相关代码
  - `route3` 概率内核
  - `common.py` 与共享求解器
  - 其余 `qrng_routes` 子模块一并保留，以保证包内导入结构与原仓库一致
- `output/qrng_routes/`
  - `route5` 关键结果 JSON
  - `cv_bell_probability_formula_check.json`
- `requirements.txt`
  - 运行本包所需的 Python 依赖
- `ENVIRONMENT_CN.md`
  - 打包时的环境快照与复现建议

## 解压后如何直接运行

建议在压缩包根目录执行：

```bash
cd route5_bundle_20260328
export PYTHONPATH=src/python
python -m qrng_routes.route5 --help
```

如果需要运行固定光强搜索：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.intensity_menu_search --help
```

如果需要运行本地精修队列：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.refine_queue --help
```

如果需要复核积分公式与 `Tr(M_c rho)` 的对比脚本：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.verify_cv_bell_probabilities --help
```

## 依赖说明

最基本依赖见 `requirements.txt`，主要包括：

- `numpy`
- `scipy`
- `cvxpy`
- `tqdm`
- `mosek`

说明：

- 如果只是先把代码跑通，可以使用 `SCS` 作为开源求解器。
- 如果希望更接近包内正式结果，尤其是 `H_min > 2` 的主结果，建议使用带许可证的 `MOSEK` 环境。

## 报告路径说明

本包中的两份报告已经改成包内相对路径：

- `docs/route5_detailed_technical_report_cn.md`
- `docs/cv_bell_integral_vs_trace_probability_check_cn.md`

因此在支持 Markdown 本地链接的阅读器中，点击后可以直接跳到：

- 包内代码文件
- 包内结果文件
- 包内补充文档

## 关于实验数据

当前包中的 `route5` 结果主要是理论模型与数值认证结果，不是实验实测概率数据结果。

后续当实验室测得 `route5` 所需的真实 IQ / coarse-grained 概率数据后，可以沿用本包中的代码与流程，把理论概率替换为实验概率，再进入同一条 SDP 认证流程。
