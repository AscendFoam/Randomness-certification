# Route 5: Generalized Coherent Alphabet + Physically Constrained IQ Search

## 1. 这条路线在做什么

`route5` 对应的是我们刚整理出来的混合主线原型：

```text
高速 CV 前端
+ 数字 coarse-graining
+ route2 单设备 MDI SDP
```

它和 `route3` 的关键区别不在中央测量结构，而在 trusted inputs：

- `route3` 主要是固定振幅、只扫相位的 coherent-state alphabet；
- `route5` 允许每边输入同时扫描振幅和相位，形成广义 coherent alphabet；
- 中央测量仍保持物理受限的 dual-homodyne / IQ 粗粒化结构；
- 认证仍使用 route2 那套单设备 prepare-and-measure MDI SDP。

这条路线的目标不是直接复现 route2 的自由高输出 POVM，而是回答更现实的问题：

```text
如果中央接收机被限制为真实可解释的 IQ / dual-homodyne 结构，
同时 trusted inputs 从 phase-only 升级为更强的 coherent alphabet，
认证值是否会明显比 route3 更高？
```

## 2. 当前实现包含什么

- 广义 coherent alphabet：
  可以直接给任意复振幅 `alpha_k`，也可以用 `radius_values x phase_values` 生成网格。
- 局域 / 联合支撑降维：
  自动把 trusted inputs 投影到精确支撑子空间，并输出 `local_rank`、`operator_span_rank` 等诊断量。
- 物理受限 IQ 分区：
  中央测量固定为 beam splitter + dual-homodyne；
  搜索的不是自由 POVM，而是 axis-aligned 的 `x/p` 分箱边界。
- 单设备 SDP 认证：
  继续调用项目里现有的 single-device guessing-probability SDP。

## 3. 关键文件

- [`hybrid_iq.py`](./hybrid_iq.py)
  route5 的核心模型、alphabet 构造、IQ 分区搜索和 SDP 认证都在这里。
- [`main.py`](./main.py)
  route5 的独立命令行入口。
- [`__main__.py`](./__main__.py)
  允许直接使用 `python -m qrng_routes.route5`。

## 4. 核心接口

- `run_route5(...)`
  给定一个广义 coherent alphabet 和一组 `x/p` 分区，运行单次 route5。
- `search_route5_iq_partitions(...)`
  在物理受限的 axis-aligned IQ 分区家族上做候选搜索，并认证 top-k 候选。
- `search_route5_alphabets(...)`
  先在半径池和相位池上系统生成 alphabet 候选，再对排名靠前的 alphabet 做 IQ 分区搜索和 SDP 认证。

## 5. 命令行示例

先激活环境并设置 `PYTHONPATH`：

```bash
conda activate DLEnv
export PYTHONPATH=src/python
```

### 5.1 单次运行

默认的 `3` 个半径、`4` 个相位、`2 x 2` 分区：

```bash
python -m qrng_routes.route5 \
  --mode single \
  --cutoff 6 \
  --num-x-bins 2 \
  --num-p-bins 2 \
  --solver MOSEK
```

指定更广义的半径-相位网格：

```bash
python -m qrng_routes.route5 \
  --mode single \
  --cutoff 6 \
  --radius-values 0.0 0.4 0.8 1.2 \
  --phase-values 0.0 1.57079632679 3.14159265359 4.71238898038 \
  --num-x-bins 4 \
  --num-p-bins 2 \
  --quadrature-range 3.5 \
  --boundary-gamma 1.2 \
  --solver MOSEK
```

直接给复振幅列表：

```bash
python -m qrng_routes.route5 \
  --mode single \
  --cutoff 6 \
  --alpha-values 0j 0.6+0j -0.6+0j 0.6j -0.6j 1.0+0j -1.0+0j \
  --num-x-bins 2 \
  --num-p-bins 2 \
  --solver MOSEK
```

### 5.2 IQ 分区搜索

在 `2 x 2` 和 `4 x 2` 家族上搜索：

```bash
python -m qrng_routes.route5 \
  --mode partition-search \
  --cutoff 6 \
  --radius-values 0.0 0.6 1.2 \
  --phase-values 0.0 1.57079632679 3.14159265359 4.71238898038 \
  --num-x-bins-values 2 4 \
  --num-p-bins-values 2 \
  --quadrature-ranges 2.0 3.0 4.0 \
  --gamma-values 0.75 1.0 1.5 \
  --certify-top-k 3 \
  --max-inputs 1 \
  --solver MOSEK
```

把结果写到 JSON：

```bash
python -m qrng_routes.route5 \
  --mode partition-search \
  --cutoff 6 \
  --radius-values 0.0 0.6 1.2 \
  --phase-values 0.0 1.57079632679 3.14159265359 4.71238898038 \
  --num-x-bins-values 2 4 \
  --num-p-bins-values 2 4 \
  --quadrature-ranges 2.0 3.0 4.0 \
  --gamma-values 0.75 1.0 1.5 \
  --certify-top-k 3 \
  --max-inputs 1 \
  --solver MOSEK \
  --output-json output/qrng_routes/route5_partition_search.json
```

### 5.3 Trusted Alphabet Search

先从半径池和相位池生成系统候选，再对 top-k alphabet 做后续分区认证：

```bash
python -m qrng_routes.route5 \
  --mode alphabet-search \
  --cutoff 4 \
  --radius-values 0.0 0.4 0.8 1.2 \
  --phase-values 0.0 0.78539816339 1.57079632679 2.35619449019 3.14159265359 3.92699081699 4.71238898038 5.49778714378 \
  --num-radii-values 2 3 4 \
  --num-phase-values 4 8 \
  --num-x-bins-values 2 4 \
  --num-p-bins-values 2 \
  --quadrature-ranges 2.5 3.5 \
  --gamma-values 0.75 1.0 1.5 \
  --alphabet-top-k 2 \
  --certify-top-k 1 \
  --max-inputs 1 \
  --max-local-states 20 \
  --output-json output/qrng_routes/route5_alphabet_search.json
```

如果想允许“不带真空”的 alphabet 候选，可以加：

```bash
--no-require-vacuum
```

### 5.4 本地精修队列

如果已经知道要围绕某个高分窗口长期精修，可以直接用队列脚本：

```bash
python -m qrng_routes.route5.refine_queue \
  --cutoff 4 \
  --phase-values 0.0 0.78539816339 1.57079632679 2.35619449019 3.14159265359 3.92699081699 4.71238898038 5.49778714378 \
  --r1-values 0.85 0.9 0.95 1.0 1.05 \
  --r2-values 1.05 1.1 1.15 1.2 1.25 1.3 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-ranges 1.8 1.85 1.9 1.95 2.0 \
  --gamma-values 1.0 1.05 1.1 1.15 1.2 1.25 \
  --scout-num-quadrature-nodes 20 \
  --cert-num-quadrature-nodes 20 \
  --solver SCS \
  --scs-max-iters 12000 \
  --scs-eps-abs 5e-5 \
  --scs-eps-rel 5e-5 \
  --scs-eps-infeas 1e-6 \
  --max-inputs 3 \
  --candidate-limit 8 \
  --output-json output/qrng_routes/route5_local_refine_queue.json \
  --result-dir output/qrng_routes/route5_local_refine_queue
```

这个脚本会：

- 先对每个半径候选做 raw-only 的 IQ 分区搜索；
- 按 raw 排名选出最有希望的候选；
- 再逐个做正式 SDP 认证；
- 每完成一个候选就把总表和单点结果落盘，方便中断后续跑。

## 6. 结果里最重要的字段

- `local_rank`
  trusted coherent alphabet 的有效局域支撑维数。
- `local_operator_span_rank`
  局域输入算符张成秩；这是判断 alphabet 是否足够强的关键诊断量。
- `operator_span_rank`
  联合输入算符张成秩。
- `raw_best_H_min`
  只看观测统计时最好的原始熵。
- `H_min`
  认证后的最小熵。
- `raw_partition_ranking`
  搜索模式下按 raw 指标排序的候选 IQ 分区摘要。
- `certified_partition_results`
  搜索模式下真正做过 SDP 认证的 top-k 分区结果。
- `raw_alphabet_ranking`
  `alphabet-search` 模式下按结构指标和 raw 分区指标排序的 alphabet 候选摘要。
- `certified_alphabet_results`
  `alphabet-search` 模式下真正跑过分区搜索与 SDP 认证的 top-k alphabet 结果。

## 7. 当前原型的边界

这个 route5 还只是协议-硬件共设计原型，不是最终实验方案。当前边界主要有三点：

1. 只搜索 axis-aligned 的 `x/p` 分区，没有做更一般的二维区域优化。
2. 还没有有限尺寸、测试轮偏置和平均认证速率模块。
3. 如果 trusted alphabet 太大，`num_inputs = num_local_states^2` 会很快把 SDP 压重，所以默认参数保持在中等规模。

## 8. 一句话理解

如果只记一句话，那么 route5 就是：

```text
把 route3 的高速 CV 前端保留下来，
把 route2 的单设备 MDI 认证保留下来，
再把 trusted inputs 从 phase-only 升级成广义 coherent alphabet。
```
