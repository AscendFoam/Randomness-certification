# Route5 打包时环境与复现说明

## 1. 这份包适合在哪种环境下运行

推荐使用独立的 Python / conda 环境，并满足：

- `Python >= 3.9`
- `numpy`
- `scipy`
- `cvxpy`
- `tqdm`
- 可选但强烈建议：`mosek`

交付包根目录保留了：

- `requirements.txt`

可作为最基本的依赖参考。

## 2. 关于求解器

### 2.1 如果只是先测试代码能否跑通

可以优先使用 `SCS`：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 --mode single --solver SCS
```

这适合做：

- 包结构检查
- 导入检查
- 小规模 smoke test
- 初步搜索

### 2.2 如果想尽量接近正式结果

建议使用 `MOSEK`，并确保：

1. `mosek` Python 包已安装；
2. 本机许可证可用；
3. `cvxpy` 可以正常调用 `MOSEK`。

例如：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 --mode single --solver MOSEK
```

## 3. 当前包内结果的复现边界

这份包中的结果文件包含了 `route5` 的当前关键输出，但要注意三点。

1. 历史 `trace_povm` 高熵结果对 `num_quadrature_nodes` 有敏感性。
2. 解析概率后端当前会暴露 formal `infeasible` 的自洽性问题。
3. 因此，不能把任意一次复算的单点值都自动理解成“主结果完全复现”。

更稳妥的复现方式是：

1. 先看 `docs/route5_run_guide_cn.md`
2. 先跑 `--help`
3. 再跑小规模 `single`
4. 再跑主线单点
5. 再跑 `node_convergence_scan`

## 4. 这份包最推荐优先复核的命令

推荐优先做下面三类复核。

1. 主线单点复核  
   目的是确认当前 `trace_povm` 主线是否能在本环境下正常重跑。

2. 节点数收敛扫描  
   目的是确认 `num_quadrature_nodes` 的敏感性。

3. 解析后端诊断  
   目的是确认当前模型自洽性问题在本环境下也存在。

具体命令已写在：

- `docs/route5_run_guide_cn.md`

## 5. 关于 Matlab 文件

包内保留了：

- `src/matlab/guessprobprimal_route5_hybrid_iq.m`

它的定位是：

- 便于导师直接阅读与对照；
- 便于在 Matlab 环境下做单点参考；
- 不替代 Python 主线的搜索、诊断与批量结果管理。

所以如果 Python 与 Matlab 单点输出有差异，建议优先检查：

1. 参数是否完全一致；
2. 概率后端是否一致；
3. 求解器与数值精度设置是否一致。

## 6. 对后续继续开发的建议

如果实验室或后续接手者想继续推进 `route5`，建议以这份包作为新的起点，并优先做：

1. 统一实验版数据接口；
2. 继续检查 `trace_povm` 主线的数值稳健性；
3. 进一步闭合解析概率层与 formal trusted-state 层的一致性。
