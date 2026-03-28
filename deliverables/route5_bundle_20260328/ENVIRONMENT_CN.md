# Route5 打包时环境信息

## 1. 打包时的当前 Shell 环境快照

本交付包打包时，在当前 shell 中直接检测到的 Python 环境为：

- `Python 3.13.5`
- `numpy 2.3.3`
- `scipy 1.16.2`
- `cvxpy 1.7.3`
- 当前这个 shell 里 `mosek` 模块不可直接导入

对应说明：

- 这说明“当前打包 shell”不一定等于当初生成正式 `MOSEK` 结果时使用的环境。
- 包内已经保留了正式结果 JSON，因此阅读报告和检查数值不受影响。

## 2. 对复现实验前数值流程的建议环境

建议使用一个单独的 Python/conda 环境，并满足：

- Python `3.9+`
- `numpy`
- `scipy`
- `cvxpy`
- `tqdm`
- 可选但强烈建议：`mosek`

安装后，在包根目录执行：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 --help
```

## 3. 关于求解器

### 3.1 如果只想先把代码跑通

可以使用 `SCS`：

```bash
export PYTHONPATH=src/python
python -m qrng_routes.route5 --mode single --solver SCS
```

### 3.2 如果想更接近包内正式结果

建议使用：

- 已安装 `mosek` Python 包
- 本机已配置可用许可证

因为包内 `route5` 的主结论，尤其是：

- 自由最优点 `H_min ≈ 2.11639`
- 固定光强 `[0,80,160]` 主线 `H_min ≈ 2.10102`

都以正式 `MOSEK` 结果为准。

## 4. 包内结果的口径

包内结果文件分为三类：

1. `MOSEK` 正式结果  
   主要用于最终结论与汇报。
2. `SCS` 快速结果  
   主要用于搜索过程说明和高潜力窗口定位。
3. 概率模型一致性检查结果  
   用于支撑 `route5` 的概率生成逻辑与 CV Bell 物理模型之间的一致性判断。

## 5. 与实验数据的关系

本包中的 `route5` 输出主要是理论模型和数值认证结果。  
后续当实验室测得 `route5` 所需真实 IQ / coarse-grained 概率数据后，建议在一个带完整依赖的环境中，直接沿用本包的 `route5` 代码做实验版复算。
