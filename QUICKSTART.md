# 快速开始指南

## 5分钟上手 MDI-QRNG SDP 求解器

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

**重要**: MOSEK 需要许可证
- 学术用户免费：https://www.mosek.com/products/academic-licenses/
- 下载许可证文件（`mosek.lic`）
- Windows: 放到 `%USERPROFILE%\mosek\` 或 `C:\Users\你的用户名\mosek\`
- Linux/Mac: 放到 `~/mosek/`

### 第二步：验证安装

```bash
python test_SDP.py
```

如果看到 "🎉 所有测试通过！"，说明安装成功！

### 第三步：运行第一个优化

```bash
cd src
python SDP.py
```

你会看到详细的进度显示和结果：

```
============================================================
MDI-QRNG 半定规划优化
============================================================

物理参数:
  平均光子数 μ = 0.5
  离散化区间数 n = 3
  积分边界 = ±10.0
  目标输入对 (x*, y*) = (0, 0)

[... 进度条显示 ...]

============================================================
求解完成
============================================================
求解状态: optimal
目标函数值 G_MDI: 0.2876543210
求解用时: 2.34 秒

随机性分析:
  最小熵 H_min = 1.798765 bits
  可提取随机比特数 ≈ 1.7988 bits per round

结果已保存:
  数据文件: results/sdp_result_mu0.5_n3_20250127_143022.npz
  元数据文件: results/sdp_result_mu0.5_n3_20250127_143022.json
```

### 第四步：查看保存的结果

结果自动保存在 `results/` 目录：

```python
import numpy as np
import json

# 加载数值数据
data = np.load('results/sdp_result_mu0.5_n3_20250127_143022.npz')
print("最优值:", data['optimal_value'])
print("平均光子数:", data['mu'])

# 加载元数据
with open('results/sdp_result_mu0.5_n3_20250127_143022.json', 'r') as f:
    metadata = json.load(f)
    print("最小熵:", metadata['h_min'], "bits")
    print("求解时间:", metadata['solve_time_seconds'], "秒")
```

## 常见使用场景

### 场景1: 改变平均光子数

编辑 `src/SDP.py` 的主程序部分：

```python
mu_val = 1.0  # 从 0.5 改为 1.0
```

重新运行：
```bash
python SDP.py
```

### 场景2: 增加离散化精度

```python
n_bins_val = 4  # 从 3 改为 4（求解时间会增加）
```

**注意**: n_bins 越大，求解时间和内存需求越高

| n_bins | 变量数 | 约束数 | 内存  | 时间(24线程) |
|--------|--------|--------|-------|--------------|
| 2      | 22     | 42     | ~100MB| ~1秒         |
| 3      | 54     | 106    | ~500MB| ~3秒         |
| 4      | 102    | 202    | ~2GB  | ~15秒        |
| 5      | 170    | 330    | ~8GB  | ~60秒        |

### 场景3: 参数扫描

扫描多个 μ 值：

```bash
python SDP.py scan
```

这会运行 40 次优化（10个μ值 × 4个n_bins），并生成汇总报告。

自定义扫描范围：

```python
# 在 SDP.py 的 __main__ 部分修改
mu_range = np.linspace(0.1, 3.0, 15)  # 15个点
n_bins_range = [2, 3, 4]              # 3个离散化级别
```

### 场景4: 调整线程数

根据你的CPU调整：

```python
num_threads=16  # 改为你的CPU线程数
```

查看你的CPU线程数：
- Windows: 任务管理器 → 性能 → CPU → 逻辑处理器
- Linux: `lscpu | grep "CPU(s)"`
- Mac: `sysctl -n hw.ncpu`

## 结果解释

### 关键指标

1. **G_MDI (最优猜测概率)**
   - 范围: (0, 1]
   - 越小越好（意味着更多随机性）
   - 物理意义: Eve猜对Alice和Bob输入的最大概率

2. **H_min (最小熵)**
   - 范围: [0, ∞) bits
   - 越大越好
   - 计算公式: H_min = -log₂(G_MDI)
   - 物理意义: 可提取的随机比特数

### 典型结果范围

对于 μ = 0.5:
- G_MDI ≈ 0.25 - 0.35
- H_min ≈ 1.5 - 2.0 bits

对于 μ = 1.0:
- G_MDI ≈ 0.20 - 0.30
- H_min ≈ 1.7 - 2.3 bits

对于 μ = 2.0:
- G_MDI ≈ 0.15 - 0.25
- H_min ≈ 2.0 - 2.7 bits

### 求解状态说明

- `optimal`: ✅ 最佳状态，结果可信
- `optimal_inaccurate`: ⚠️  求解成功但精度稍低，通常可接受
- `infeasible`: ❌ 问题无可行解，检查参数设置
- `unbounded`: ❌ 问题无界，检查约束设置

## 故障排除速查

### 问题1: "No license for MOSEK"

**解决**:
1. 访问 https://www.mosek.com/products/academic-licenses/
2. 注册并下载许可证
3. 将 `mosek.lic` 放到正确位置（见上方"第一步"）

### 问题2: 求解时间太长

**解决**:
1. 减小 `n_bins` (从4改为3或2)
2. 增加 `num_threads`
3. 使用更快的CPU

### 问题3: 内存不足

**解决**:
1. 减小 `n_bins`
2. 关闭其他程序
3. 对于32GB内存，推荐 `n_bins ≤ 5`

### 问题4: 结果不合理

**检查**:
1. 查看概率归一化警告
2. 确认 μ > 0
3. 确认 range_val 足够大（推荐10-15）
4. 检查求解状态是否为 "optimal"

## 进阶功能

### 保存和加载POVM算子

如果需要保存最优的测量算子：

```python
result, results_dict = run_single_optimization(
    mu_val=0.5,
    n_bins_val=3,
    range_val=10.0,
    save_results=True
)

# 获取最优 M 矩阵
if results_dict['status'] == 'optimal':
    M_vars = results_dict['M_vars']
    M_Alice = results_dict['M_Alice']
    M_Bob = results_dict['M_Bob']

    # 保存到文件
    np.savez('optimal_POVM.npz',
             M_vars=M_vars,
             M_Alice=M_Alice,
             M_Bob=M_Bob)
```

### 批量处理多个参数

```python
from SDP import parameter_scan
import numpy as np

# 自定义扫描
results = parameter_scan(
    mu_range=np.arange(0.1, 3.1, 0.2),  # 0.1, 0.3, 0.5, ..., 3.0
    n_bins_range=[2, 3],
    range_val=10.0,
    num_threads=24
)

# 找到最优参数
best_result = max(results, key=lambda x: x['h_min'])
print(f"最佳参数: μ={best_result['mu']}, n={best_result['n_bins']}")
print(f"最大熵: {best_result['h_min']:.4f} bits")
```

### 绘制结果图

```python
import matplotlib.pyplot as plt
import json

# 加载扫描结果
with open('results/scan_summary_20250127_143022.json', 'r') as f:
    scan_results = json.load(f)

# 提取数据
mu_vals = [r['mu'] for r in scan_results if r['n_bins'] == 3]
h_min_vals = [r['h_min'] for r in scan_results if r['n_bins'] == 3]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(mu_vals, h_min_vals, 'o-', linewidth=2, markersize=8)
plt.xlabel('Average Photon Number μ', fontsize=14)
plt.ylabel('Min-Entropy H_min (bits)', fontsize=14)
plt.title('Randomness vs Photon Number (n_bins=3)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('h_min_vs_mu.png', dpi=300)
plt.show()
```

## 下一步

- 📖 阅读 [README_SDP.md](README_SDP.md) 了解完整功能
- 📄 查看 [docs/SDP.md](docs/SDP.md) 了解理论背景
- 🔬 查看 [docs/SDP_solve.tex](docs/SDP_solve.tex) 了解数学推导

## 获取帮助

如遇问题：
1. 运行 `python test_SDP.py` 检查基本功能
2. 检查本文档的"故障排除"部分
3. 查看代码中的详细注释

祝使用愉快！🎉
