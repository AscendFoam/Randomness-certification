# Route5 交付包说明

这个交付包用于单独提交 `route5` 的 Python 代码、理论报告、运行说明和关键结果文件，方便导师或实验室在不打开整个仓库的前提下，直接查阅与复核。

## 1. 这份包的定位

这份包的主线是：

- 以 Python 实现为准；
- Matlab 文件保留为参考材料；
- 理论报告、运行说明、关键结果 JSON 与代码一起提供；
- 重点保留 `route5` 当前最重要的两条信息：
  1. 历史 `trace_povm` 主线曾出现 `H_min > 2` 的强候选窗口；
  2. 解析概率后端已暴露出当前 formal 模型与概率层之间的自洽性问题。

因此，这份包既适合阅读，也适合做后续复算与继续开发。

## 2. 目录结构

- `docs/`
  - `route5_matlab_theory_formulation_cn.md`
  - `route5_run_guide_cn.md`
  - `route5_probability_sdp_explanation_cn.md`
  - `route5_analytic_backend_mismatch_diagnosis_cn.md`
  - `route5_detailed_technical_report_cn.md`
  - `cv_bell_integral_vs_trace_probability_check_cn.md`
  - `QRNG_CVBSM.md`
- `src/python/qrng_routes/`
  - `route5/` 主代码
  - `route3/` 中被 `route5` 概率层调用的双 Homodyne 支持代码
  - `common.py`
  - `__init__.py`
- `src/matlab/`
  - `guessprobprimal_route5_hybrid_iq.m`
- `output/qrng_routes/`
  - `route5` 当前关键结果 JSON
- `requirements.txt`
  - Python 依赖列表
- `ENVIRONMENT_CN.md`
  - 环境与复现建议

## 3. 推荐先读什么

如果是导师或同学第一次接触这份包，建议按这个顺序阅读：

1. `docs/route5_matlab_theory_formulation_cn.md`
2. `docs/route5_run_guide_cn.md`
3. `src/python/qrng_routes/route5/README.md`
4. `output/qrng_routes/` 中的关键 JSON

其中：

- 理论报告负责解释模型和结果；
- 运行说明负责告诉你代码怎么跑；
- JSON 文件负责保留当前阶段的重要数值输出。

## 4. 如何直接运行

建议在交付包根目录执行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 --help
```

如果需要更细的测试入口，可以运行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.node_convergence_scan --help
python -m qrng_routes.route5.intensity_menu_search --help
python -m qrng_routes.route5.refine_queue --help
python -m qrng_routes.route5.analytic_backend_diagnostics --help
python -m qrng_routes.route5.fixed_partition_radius_search --help
```

更具体的命令示例见：

- `docs/route5_run_guide_cn.md`

## 5. 当前最关键的结果口径

使用这份包时，请务必同时记住下面三点。

1. 历史 `trace_povm` 主线曾给出 `H_min > 2` 的强候选结果，但对 `num_quadrature_nodes` 明显敏感。
2. `analytic_gaussian_rectangles` 后端目前更适合作为一致性诊断工具，而不是现成正式主结果后端。
3. 如果要把结果带去实验室讨论，不能只摘一个 `H_min` 数字，至少还要一起报告：
   - `cutoff`
   - `num_quadrature_nodes`
   - `probability_engine`
   - 输入字母表参数
   - IQ 分箱参数

## 6. 关于实验数据

当前这份交付包中的 `route5` 结果主要是：

- 理论模型结果
- 数值概率结果
- SDP 认证结果

并不是已经接上真实实验 IQ 数据后的正式实验认证结果。

如果后续实验室要继续推进这条线，建议直接以这份包为基础，把实验版：

$$
P_{\mathrm{exp}}(c\mid x,y)
$$

接入现有代码结构，再沿用同一条 formal 认证流程。
