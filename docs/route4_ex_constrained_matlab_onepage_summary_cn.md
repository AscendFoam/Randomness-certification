# Route4-ex-constrained Matlab 版 1页摘要

## 1. 这份脚本是什么

文件：

- `src/matlab/guessprobprimal_route4_ex_constrained.m`

它是一份可直接在 Matlab + CVX 中运行的单文件脚本，用来复现当前 `route4-ex-constrained` 主线的正式计算结果。

它保留了原始 Matlab route4 的整体骨架：

1. 选取 `selected_mu_list`
2. 读取 `Probability.mat`
3. 做输出 coarse-graining
4. 构造 `LambdaIndices`
5. 写 primal SDP
6. 输出 `p_guess` 与 `H_min`

因此它不是完全新路线，而是一个“贴近原 route4 写法的 constrained 扩展版”。

---

## 2. 相对原始 Matlab route4，哪些地方没变

以下几点保持不变：

1. 仍然直接使用 `Probability.mat` 作为实验概率数据入口。
2. 仍然从固定光强菜单 `full_mu = [0,20,40,60,80,100,120,140,160]` 中选输入窗口。
3. 仍然保留 `q_selected` 作为目标函数中的生成轮权重。
4. 仍然使用 `LambdaIndices` 枚举 `N^(D+1)` 个确定性策略。
5. 仍然通过 CVX 求解 primal 型 SDP。

所以从实验室视角看，这份脚本仍然沿用原 route4 的“数据接口 + 优化骨架”。

---

## 3. 相对原始 Matlab route4，哪些地方被改了

真正的修改主要有三处。

### 3.1 trusted input 从“只用对角”改成“用完整相干态”

原始 route4：

- 虽然也写出 `rho = |alpha><alpha|`，但真正送进 SDP 的是 `rho_diag`；
- 因此模型本质上只使用 Fock 对角信息。

constrained 版：

- 仍然构造 `rho = |alpha><alpha|`；
- 但正式主问题 `full primal` 直接使用完整 `rho`；
- 因而 trusted input 的非对角结构真正参与了认证。

### 3.2 coarse-graining 不再等分，而是固定边界

原始 route4：

- 一般把 256 个原始输出按 `N` 等宽分块。

constrained 版：

- 固定使用
  - `custom_edges = [0,121,132,256]`
- 因此 3 个输出区间宽度分别为
  - `121`
  - `11`
  - `124`

这一步仍然只是对 `Probability.mat` 做离散化，但边界选择更贴近当前高值主线。

### 3.3 主结果不再取 diagonal primal，而取 full primal

原始 route4：

- 主问题是 Fock 对角测量元的 primal。

constrained 版：

- 保留 diagonal primal 作为对照；
- 但正式主结果来自 full primal：
  - 测量元是一般 Hermitian PSD 矩阵；
  - 约束为 `sum_y M_{y,lambda} = s_lambda I`；
  - 统计匹配为 `Tr(rho_x M_y) = p(x,y)`。

这正是 constrained 版相对原 route4 最核心的升级。

---

## 4. 当前默认参数

脚本默认参数为：

- `selected_mu_list = [100,120,140]`
- `q_selected = [1,0,0]`
- `custom_edges = [0,121,132,256]`
- `alpha_values = [0.54, 0.66 i, -0.72]`
- `M = 6`

其中：

- `q=[1,0,0]` 的意思不是后两个输入不用；
- 而是目标函数只对第一个输入加权；
- 后两个输入仍然通过统计约束进入 SDP。

---

## 5. 当前结果

你本机 Matlab 实际跑出的结果是：

- `Full primal status: Solved`
- `Full primal H_min: 1.227498940472`
- `Diagonal primal status: Infeasible`

这与 Python/MOSEK 主线的

- `H_min ≈ 1.227500864253`

只差约 `1.9e-6`，属于正常数值误差。

因此当前可以把结论写成：

- `route4-ex-constrained` 在这组固定参数下能够稳定实现 `H_min > 1`；
- 并且这一结果已经在 Matlab 版本中复现。

---

## 6. 导师最值得先看的三处

如果只看最关键的地方，建议优先看：

1. 输入态构造部分
   - 看 trusted input 是否还是只用 `rho_diag`。
2. coarse-graining 部分
   - 看是否从等分改成了固定边界 `[0,121,132,256]`。
3. full primal 部分
   - 看主结果是否改成了一般 Hermitian PSD POVM。

这三处基本概括了 constrained 版相对原始 route4 的全部关键修改。

---

## 7. 建议怎么发给导师

最合适的组合是：

1. 先发这份 1页摘要；
2. 再附上脚本
   - `src/matlab/guessprobprimal_route4_ex_constrained.m`
3. 如果导师需要细看，再补充长一点的逐段对照文档
   - `docs/route4_ex_constrained_matlab_comparison_cn.md`

这样导师可以先快速建立整体判断，再决定要不要往细节里钻。

