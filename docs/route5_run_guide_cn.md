# Route5 Python 运行与测试说明

## 1. 文档目的

这份文档专门说明如何在本地运行、测试和复核 `route5` 的 Python 代码。

它和主理论报告
[`route5_matlab_theory_formulation_cn.md`](./route5_matlab_theory_formulation_cn.md)
的分工不同：

- 主理论报告负责解释物理模型、概率构造、SDP 与当前结果；
- 本文档只负责回答“代码怎么跑”“先跑什么”“结果文件看哪里”“哪些命令最适合复核当前主线”。

---

## 2. 代码入口与依赖关系

当前 `route5` 的 Python 主线代码位于：

- [`src/python/qrng_routes/route5/hybrid_iq.py`](../src/python/qrng_routes/route5/hybrid_iq.py)
- [`src/python/qrng_routes/route5/main.py`](../src/python/qrng_routes/route5/main.py)
- [`src/python/qrng_routes/route5/refine_queue.py`](../src/python/qrng_routes/route5/refine_queue.py)
- [`src/python/qrng_routes/route5/node_convergence_scan.py`](../src/python/qrng_routes/route5/node_convergence_scan.py)
- [`src/python/qrng_routes/route5/intensity_menu_search.py`](../src/python/qrng_routes/route5/intensity_menu_search.py)
- [`src/python/qrng_routes/route5/analytic_backend_diagnostics.py`](../src/python/qrng_routes/route5/analytic_backend_diagnostics.py)
- [`src/python/qrng_routes/route5/fixed_partition_radius_search.py`](../src/python/qrng_routes/route5/fixed_partition_radius_search.py)

运行时还依赖：

- [`src/python/qrng_routes/common.py`](../src/python/qrng_routes/common.py)
- [`src/python/qrng_routes/route3/cv_four_phase.py`](../src/python/qrng_routes/route3/cv_four_phase.py)

因此，如果要把 `route5` 单独打包给别人，不能只拷 `route5/` 这个子目录，还必须一并带上：

- `qrng_routes/common.py`
- `qrng_routes/route3/`
- `qrng_routes/__init__.py`

---

## 3. 环境准备

建议至少具备以下 Python 依赖：

- `python >= 3.9`
- `numpy`
- `scipy`
- `cvxpy`
- `tqdm`
- 可选但强烈建议：`mosek`

仓库根目录已有：

- [`requirements.txt`](../requirements.txt)

如果只想先测试代码能否跑通，可以使用 `SCS`。  
如果希望尽量复核当前正式主结果，建议使用 `MOSEK`，并确保本机许可证可用。

---

## 4. 基本运行方式

在仓库根目录执行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 --help
```

若命令正常输出帮助信息，说明包导入结构基本无误。

如果要查看某个子脚本的帮助信息，可分别执行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.node_convergence_scan --help
python -m qrng_routes.route5.intensity_menu_search --help
python -m qrng_routes.route5.refine_queue --help
python -m qrng_routes.route5.analytic_backend_diagnostics --help
python -m qrng_routes.route5.fixed_partition_radius_search --help
```

---

## 5. 最小可运行测试

如果只想确认主程序入口可运行，建议先做一个小规模 `single` 测试。

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 \
  --mode single \
  --cutoff 4 \
  --radius-values 0.0 0.85 1.25 \
  --phase-values 0.0 1.5707963267948966 3.141592653589793 4.71238898038469 \
  --num-x-bins 2 \
  --num-p-bins 2 \
  --num-quadrature-nodes 12 \
  --solver SCS
```

这条命令的目标不是复现正式高分结果，而是快速确认：

1. 字母表构造正常；
2. IQ 概率层正常；
3. SDP 能正常被调用；
4. JSON 风格输出结构是完整的。

---

## 6. 当前主线单点复核

如果想复核目前最重要的历史 `trace_povm` 主线，可以使用：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 \
  --mode single \
  --cutoff 4 \
  --radius-values 0.0 0.85 1.25 \
  --phase-values \
    0.0 \
    0.7853981633974483 \
    1.5707963267948966 \
    2.356194490192345 \
    3.141592653589793 \
    3.9269908169872414 \
    4.71238898038469 \
    5.497787143782138 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-range 1.8 \
  --boundary-gamma 1.0 \
  --num-quadrature-nodes 12 \
  --probability-engine trace_povm \
  --max-inputs 3 \
  --solver MOSEK \
  --output-json output/qrng_routes/route5_single_trace_mainline_recheck.json
```

这条命令对应的核心口径是：

- `cutoff = 4`
- `radii = [0, 0.85, 1.25]`
- `8` 个均匀相位
- `num_x_bins = 6`
- `num_p_bins = 2`
- `quadrature_range = 1.8`
- `boundary_gamma = 1.0`
- `num_quadrature_nodes = 12`
- `probability_engine = trace_povm`

如果环境、求解器和数值设置完全一致，结果应接近历史文件：

- [`output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json`](../output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json)

需要注意：这一类结果对 `num_quadrature_nodes` 有敏感性，不能只看单次高分值而忽略收敛性。

---

## 7. 节点数收敛扫描

如果要严谨复核 `num_quadrature_nodes` 对结果的影响，使用：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.node_convergence_scan \
  --cutoff 4 \
  --radius-values 0.0 0.85 1.25 \
  --phase-values \
    0.0 \
    0.7853981633974483 \
    1.5707963267948966 \
    2.356194490192345 \
    3.141592653589793 \
    3.9269908169872414 \
    4.71238898038469 \
    5.497787143782138 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-range 1.8 \
  --boundary-gamma 1.0 \
  --probability-engine trace_povm \
  --num-nodes-values 12 16 20 \
  --max-inputs 3 \
  --top-raw-k 8 \
  --solver MOSEK \
  --output-json output/qrng_routes/route5_node_convergence_scan_recheck.json \
  --result-dir output/qrng_routes/route5_node_convergence_scan_recheck
```

这个脚本会逐个节点数扫描，并记录：

1. `raw_best_H_min`
2. `raw_top_targets`
3. `status`
4. `p_guess`
5. `H_min`
6. `target_scan`

它是当前判断 `trace_povm` 高分是否稳定的关键测试入口。

---

## 8. 固定光强菜单搜索

如果要复核“固定光强约束下仍可能超过 2 bit”的那条线，使用：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.intensity_menu_search \
  --intensity-values 0 80 160 \
  --max-radius-values 1.2 \
  --cutoff 4 \
  --phase-values \
    0.0 \
    0.7853981633974483 \
    1.5707963267948966 \
    2.356194490192345 \
    3.141592653589793 \
    3.9269908169872414 \
    4.71238898038469 \
    5.497787143782138 \
  --num-radii-values 3 \
  --num-phase-values 8 \
  --num-x-bins-values 6 \
  --num-p-bins-values 2 \
  --quadrature-ranges 1.8 1.85 1.9 1.95 2.0 \
  --gamma-values 1.0 \
  --num-quadrature-nodes 12 \
  --alphabet-top-k 4 \
  --certify-top-k 3 \
  --max-inputs 3 \
  --solver MOSEK \
  --output-json output/qrng_routes/route5_fixed_intensity_recheck.json
```

这条入口适合回答：

1. 在固定强度菜单下还能否保住高 formal 熵；
2. `max_radius` 映射对结果有多大影响；
3. 哪个固定光强字母表最值得后续实验口径跟进。

---

## 9. 解析概率后端复核

如果要检查 `analytic_gaussian_rectangles` 这条解析概率后端，可直接运行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 \
  --mode single \
  --cutoff 4 \
  --radius-values 0.0 0.85 1.25 \
  --phase-values \
    0.0 \
    0.7853981633974483 \
    1.5707963267948966 \
    2.356194490192345 \
    3.141592653589793 \
    3.9269908169872414 \
    4.71238898038469 \
    5.497787143782138 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-range 1.8 \
  --boundary-gamma 1.0 \
  --num-quadrature-nodes 12 \
  --probability-engine analytic_gaussian_rectangles \
  --max-inputs 3 \
  --solver MOSEK \
  --output-json output/qrng_routes/route5_single_analytic_recheck.json
```

当前已知现象是：  
在这组主线参数下，解析后端通常会给出较高的 raw 值，但 formal 可能返回 `infeasible`。

如果要进一步做后端失配诊断，运行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.analytic_backend_diagnostics \
  --cutoffs 4 5 6 8 12 16 \
  --feasibility-cutoffs 4 5 6 \
  --radius-values 0.0 0.85 1.25 \
  --phase-values \
    0.0 \
    0.7853981633974483 \
    1.5707963267948966 \
    2.356194490192345 \
    3.141592653589793 \
    3.9269908169872414 \
    4.71238898038469 \
    5.497787143782138 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-range 1.8 \
  --boundary-gamma 1.0 \
  --num-trace-nodes 400 \
  --feasibility-solver MOSEK \
  --output-json output/qrng_routes/route5_analytic_backend_diagnostics_recheck.json
```

---

## 10. 长时间局部精修

如果要围绕某个高潜力窗口持续精修，可用：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5.refine_queue \
  --cutoff 4 \
  --phase-values \
    0.0 \
    0.7853981633974483 \
    1.5707963267948966 \
    2.356194490192345 \
    3.141592653589793 \
    3.9269908169872414 \
    4.71238898038469 \
    5.497787143782138 \
  --r1-values 0.85 0.9 0.95 1.0 \
  --r2-values 1.15 1.2 1.25 1.3 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-ranges 1.8 1.85 1.9 \
  --gamma-values 1.0 1.05 1.1 \
  --scout-num-quadrature-nodes 12 \
  --cert-num-quadrature-nodes 12 \
  --probability-engine trace_povm \
  --solver MOSEK \
  --max-inputs 3 \
  --candidate-limit 8 \
  --output-json output/qrng_routes/route5_refine_queue_recheck.json \
  --result-dir output/qrng_routes/route5_refine_queue_recheck
```

这个脚本适合长时间跑后台，因为它会边跑边落盘。

---

## 11. 如何理解输出结果

最值得优先查看的字段包括：

- `status`
- `p_guess`
- `H_min`
- `raw_H_min` 或 `raw_best_H_min`
- `target_input`
- `target_alphas`
- `probability_engine`
- `num_quadrature_nodes`
- `local_rank`
- `local_operator_span_rank`

其中：

- `raw_H_min` 只反映分布表面上的平坦度；
- `H_min` 才是 formal SDP 认证结果；
- `status = infeasible` 通常说明当前概率表与 trusted-state formal 模型不兼容；
- `num_quadrature_nodes` 与 `probability_engine` 一定要和结果一起报告，不能只摘一个 `H_min` 数字。

---

## 12. 推荐测试顺序

如果后续接手的人想快速进入状态，建议按下面顺序运行。

1. 先跑 `python -m qrng_routes.route5 --help`
2. 再跑一个小规模 `--mode single --solver SCS`
3. 再跑历史主线单点 `trace_povm` 复核
4. 再跑 `node_convergence_scan`
5. 再跑 `analytic_backend_diagnostics`
6. 最后再决定是否做 `intensity_menu_search` 或 `refine_queue`

这个顺序的好处是：

- 先确保代码能跑；
- 再确认主线结果能复现；
- 最后才做贵的或长时间的搜索。

---

## 13. 当前最重要的使用提醒

在当前项目阶段，`route5` 的代码使用上有三条必须记住。

1. `trace_povm` 的高熵结果不能脱离 `num_quadrature_nodes` 单独汇报。
2. `analytic_gaussian_rectangles` 目前更适合作为一致性诊断工具，而不是现成的正式主结果后端。
3. 如果要交给实验室，除了代码本身，还必须同时附上理论报告和关键结果 JSON，否则很难说清楚结果口径。
