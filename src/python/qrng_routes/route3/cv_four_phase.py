"""
连续变量四相位CV Bell测量路由 (Route 3)
==========================================

物理背景
--------
本模块实现了一种基于连续变量(Continuous Variable, CV) Bell测量的量子随机数认证方案。

核心思想：
1. Alice准备四种不同相位的相干态（0°, 90°, 180°, 270°）
2. 两束光通过平衡分束器(50:50 beamsplitter)进行干涉
3. 对干涉后的输出做双Homodyne测量（同时测量X和P quadrature）
4. 利用测量结果的概率分布，通过SDP计算可认证的随机数熵率

关键物理概念：
- 相干态(Coherent State): |α⟩ = |√μ·e^(iφ)⟩，最接近经典光场的量子态
- Quadrature算符: X̂ = (â + â†)/√2, P̂ = (â - â†)/(i√2)，类比位置和动量
- 平衡分束器: 50:50分束器，实现光的干涉和Bell态测量
- Homodyne测量: 测量光场的特定quadrature分量
- POVM (Positive Operator-Valued Measure): 量子测量的数学描述
- SDP (Semi-Definite Programming): 半定规划，用于计算H_min下界

术语约定：
- mu: 平均光子数 (|α|²)
- cutoff: 截断维度（用于数值计算的光子数上限）
- num_phases: 相位数量（默认4，对应四相位协议）
"""

from __future__ import annotations

import math

import numpy as np

from ..common import (
    SingleDeviceGuessingProblem,
    balanced_beamsplitter_unitary,
    coherent_state,
    density_from_ket,
    kron,
    operator_span_rank,
    project_density_to_basis,
    quadrature_hermite_data,
    quadrature_povms_from_node_masks,
    support_basis,
)


def phase_alphabet(
    mu: float,
    cutoff: int,
    num_phases: int = 4,
) -> tuple[list[np.ndarray], list[float]]:
    """
    生成均匀相位的本地相干态字母表
    =================================

    物理原理
    --------
    相干态 |α⟩ = |√μ·e^(iφ)⟩ 的相位 φ 均匀分布在 [0, 2π) 区间时，
    构成一组用于量子协议的输入态集合。

    四相位协议（num_phases=4）：
        φ ∈ {0, π/2, π, 3π/2}
    对应四个态：|+√μ⟩, |+i√μ⟩, |-√μ⟩, |-i√μ⟩

    这些态在X-P相空间形成十字交叉分布，是实现CV Bell不等式违背的关键。

    参数
    ----
    mu : float
        平均光子数，即 |α|²。决定相干态的强度。
        - μ 较小 → 相干态接近真空态，不同相位态重叠大
        - μ 较大 → 不同相位态更易区分，但技术实现更难
        - 典型值：μ ∈ [0.1, 10]

    cutoff : int
        截断维度。由于相干态在Fock基下的展开 |α⟩ = Σ c_n |n⟩
        是无限长的，我们截取前cutoff个光子数态来进行数值计算。
        - cutoff必须足够大以捕捉相干态的主要概率
        - 通常取 cutoff ≈ 2μ + 10

    num_phases : int
        相位数量。默认4对应四相位协议。

    返回
    ----
    states : list[np.ndarray]
        密度矩阵列表，每个元素是对应相干态的密度矩阵 ρ = |α⟩⟨α|
        维度为 cutoff × cutoff

    phases : list[float]
        各个相干态的相位值（弧度制）

    示例
    ----
    >>> states, phases = phase_alphabet(mu=0.5, cutoff=10, num_phases=4)
    >>> # 生成4个相干态，相位分别为 0, π/2, π, 3π/2
    >>> # 密度矩阵是纯态：rank=1

    数学细节
    --------
    对于相位 φ，相干态的复振幅为：
        α = √μ · e^(iφ)

    在Fock基下展开：
        |α⟩ = e^(-|α|²/2) Σ (α^n/√n!) |n⟩

    密度矩阵 ρ = |α⟩⟨α| 是纯态密度矩阵（迹为1，rank=1）
    """
    # 生成均匀分布的相位点：[0, 2π/num_phases, 2·2π/num_phases, ...]
    phases = list(np.linspace(0.0, 2.0 * np.pi, num_phases, endpoint=False))

    # 振幅 α = √μ，模长由平均光子数决定
    amplitude = math.sqrt(mu)

    states = []
    for phase in phases:
        # 生成复振幅 α = √μ · e^(iφ)
        ket = coherent_state(cutoff, amplitude * np.exp(1j * phase))

        # 从态矢量构造密度矩阵 ρ = |ψ⟩⟨ψ|
        states.append(density_from_ket(ket))

    return states, phases


def four_phase_alphabet(mu: float, cutoff: int) -> tuple[list[np.ndarray], list[float]]:
    """
    四相位字母表的便捷封装
    ======================

    为了向后兼容保留的辅助函数，等价于调用 phase_alphabet(mu, cutoff, num_phases=4)

    四相位协议的物理动机
    -------------------
    - 四个相位 {0, π/2, π, 3π/2} 对应四个输入相干态
    - 注意：不同相位的相干态彼此不正交（相干态的非正交性是量子特性之一）
    - 0和π相位对应实振幅 ±√μ
    - π/2和3π/2相位对应虚振幅 ±i√μ
    - 这种对称性有助于后续Bell不等式的违背分析
    - 在X-P相空间中，四个态形成90°对称的十字形分布
    """
    return phase_alphabet(mu, cutoff, num_phases=4)


def reduced_joint_inputs(
    mu: float,
    cutoff: int,
    num_phases: int = 4,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, int, int]:
    """
    生成约化到字母表支持空间的联合输入态
    =====================================

    物理原理
    --------
    目标：构造 Alice-Bob 双模系统的输入态集合。

    问题：完整的双模相干态空间是无限维的，无法直接数值计算。
    解决方案：只考虑"字母表"张成的有限维子空间。

    具体步骤
    --------
    1. 生成 num_phases 个单模相干态 { |α_k⟩ }
    2. 找到这些态张成的正交归一基（支撑空间 support）
    3. 将所有态投影到这个有限维子空间
    4. 构造双模态：|α_i⟩ ⊗ |α_j⟩（张量积）

    参数
    ----
    mu : float
        平均光子数

    cutoff : int
        截断维度

    num_phases : int
        单模态的相位数量

    返回
    ----
    joint_states : list[np.ndarray]
        双模密度矩阵列表，包含 num_phases² 个元素

    labels : list[tuple[int, int]]
        每个双模态对应的 (i, j) 标签，i和j分别是两个模的相位索引

    joint_basis : np.ndarray
        双模空间的正交归一基矩阵

    local_rank : int
        单模支撑空间的维度（通常等于 num_phases）

    joint_dim : int
        双模态矩阵的维度，等于 local_rank²

    数学细节
    --------
    单模基的计算：
        从密度矩阵提取基态：ρ = Σ λ_k |v_k⟩⟨v_k|
        取最大特征值对应的特征向量 |v_max⟩

    支撑空间的定义：
        span{ |α_0⟩, |α_1⟩, ..., |α_{n-1}⟩ } 的正交归一化基

    张量积：
        对于态 |ψ_A⟩ ⊗ |ψ_B⟩，向量表示为 |ψ_A⟩ ⊗ |ψ_B⟩ 的Kronecker积
    """
    # 步骤1：生成单模相干态字母表
    local_states, _ = phase_alphabet(mu, cutoff, num_phases=num_phases)

    # 步骤2：从每个密度矩阵提取基态矢量
    # 密度矩阵 ρ 的特征分解：ρ = Σ λ_k |v_k⟩⟨v_k|
    # 对于纯态密度矩阵，最大的特征值 λ_0 ≈ 1，对应特征向量就是态矢量 |ψ⟩
    local_kets = []
    for rho in local_states:
        values, vectors = np.linalg.eigh(rho)  # 特征值升序排列
        local_kets.append(vectors[:, -1])       # 取最大特征值对应的特征向量

    # 步骤3：计算这些态矢量张成的正交归一基
    local_basis = support_basis(local_kets)

    # 步骤4：将每个单模态投影到支撑空间
    reduced_local_states = [project_density_to_basis(rho, local_basis) for rho in local_states]

    # 步骤5：构造双模联合态（张量积）
    joint_states: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x, rho_a in enumerate(reduced_local_states):
        for y, rho_b in enumerate(reduced_local_states):
            # 张量积：ρ_AB = ρ_A ⊗ ρ_B
            joint_states.append(kron(rho_a, rho_b))
            labels.append((x, y))

    # 双模基：单个模的基的张量积
    joint_basis = kron(local_basis, local_basis)

    return joint_states, labels, joint_basis, local_basis.shape[1], joint_states[0].shape[0]


def default_quadrature_nodes(cutoff: int) -> int:
    """
    计算数值积分的默认节点数
    ========================

    物理原理
    --------
    CV Bell测量涉及对连续变量的高斯积分：
        P(k|l) = ∫_{bin_k} f(x) dx

    为了数值计算这个积分，我们需要用求和近似积分：
        ∫ f(x)dx ≈ Σ w_i · f(x_i)

    其中 {x_i} 是求积节点，{w_i} 是对应的权重。
    这是Gauss-Hermite积分或类似的数值方法。

    节点数选择依据
    -------------
    - cutoff 越大，相干态分布越广，需要更多节点覆盖
    - 经验公式：nodes = max(400, 60 × cutoff)
    - cutoff=10 → 600个节点；cutoff=12 → 720个节点
    - 最少400个节点保证数值精度
    - 这个选择平衡了精度和计算效率
    """
    return max(400, 60 * cutoff)


def default_quadrature_bounds(num_bins: int, finite_range: float = 3.0) -> np.ndarray:
    """
    生成连续变量离散化的边界
    ========================

    物理原理
    --------
    连续变量的测量结果必须离散化才能与离散变量协议对接。
    我们把实数轴分成 num_bins 个区间（bins）。

    边界设置策略
    ------------
    - 在 X̂ = (â + â†)/√2 的约定下，真空态的 quadrature 方差为 1/2
    - ±3.0 range 覆盖了 ±3σ 范围，包含 99.7% 的概率
    - 边界点：-∞, -3, -2, -1, 0, 1, 2, 3, +∞

    特殊情况
    --------
    num_bins=2 时，使用自然划分点 0：
        bins = (-∞, 0), [0, +∞)
    这对应于"正/负"的二值测量结果。

    参数
    ----
    num_bins : int
        区间（箱子）数量。例如 2 → 2个区间，4 → 4个区间

    finite_range : float
        有限边界的范围，默认3.0（以标准差为单位）

    返回
    ----
    bounds : np.ndarray
        长度为 num_bins + 1 的边界数组
        包含 -∞, finite_boundary_points..., +∞
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    # 二值测量的特殊情况：使用0作为划分点
    if num_bins == 2:
        return np.array([-np.inf, 0.0, np.inf], dtype=float)

    # 均匀划分 finite_range 区间，两端设为 ±∞
    bounds = np.linspace(-finite_range, finite_range, num_bins + 1, dtype=float)
    bounds[0] = -np.inf
    bounds[-1] = np.inf

    return bounds


def quadrature_povms_from_bounds(
    cutoff: int,
    theta: float,
    bounds: np.ndarray,
    num_nodes: int | None = None,
) -> list[np.ndarray]:
    """
    从区间边界构造粗粒化Quadrature POVM
    ===================================

    物理原理
    --------
    POVM (Positive Operator-Valued Measure) 是量子测量的数学描述。

    对于连续变量 quadrature 测量，POVM元为：
        F_k = ∫_{bin_k} |x⟩⟨x| dx

    其中 |x⟩ 是 quadrature 本征态（ position-like state）。

    实际计算中使用截断的Fock基， quadrature 态表示为
    有限维矩阵，积分用数值求积近似。

    参数
    ----
    cutoff : int
        Fock空间的截断维度

    theta : float
        quadrature的角度
        - theta=0: 测量 X quadrature (X̂ = (â + â†)/√2)
        - theta=π/2: 测量 P quadrature (P̂ = (â - â†)/(i√2))
        - 任意theta: X(θ) = Xcosθ + Psinθ

    bounds : np.ndarray
        区间边界，例如对于二值测量：[-∞, 0, +∞]

    num_nodes : int | None
        数值积分的节点数，默认使用 default_quadrature_nodes(cutoff)

    返回
    ----
    povms : list[np.ndarray]
        POVM矩阵列表，每个矩阵维度为 cutoff × cutoff
        满足完备性：Σ_k F_k = I

    算法步骤
    --------
    1. 获取Hermite-Gauss积分节点和权重
    2. 对每个区间 bin_k，构造节点掩码（mask）
    3. 使用掩码加权求和构造POVM元
    4. 对每个POVM元进行对称化（确保厄米性）
    """
    edges = np.asarray(bounds, dtype=float)

    # 参数验证
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bounds must be a one-dimensional array with at least two entries.")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("bounds must be nondecreasing.")

    # 获取数值积分节点
    nodes_count = default_quadrature_nodes(cutoff) if num_nodes is None else num_nodes
    nodes, _, _ = quadrature_hermite_data(cutoff, nodes_count)

    # 为每个区间创建掩码
    num_bins = edges.size - 1
    masks = np.zeros((num_bins, nodes.size), dtype=float)
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]

        # 初始化为全选，然后排除边界外的点
        mask = np.ones(nodes.size, dtype=bool)
        if np.isfinite(lower):
            mask &= nodes >= lower
        if np.isfinite(upper):
            if index == num_bins - 1:
                mask &= nodes <= upper
            else:
                mask &= nodes < upper

        masks[index, mask] = 1.0

    # 从掩码构造POVM
    return quadrature_povms_from_node_masks(
        cutoff,
        theta,
        masks,
        num_nodes=nodes_count,
    )


def dual_homodyne_povm(
    cutoff: int,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    num_nodes: int | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, np.ndarray]:
    """
    中心 POVM：平衡分束器 + X/P 粗粒化测量
    ==========================================

    物理原理
    --------
    这是实现 CV Bell 测量的核心测量方案。

    CV Bell 测量的目标：
    - 同时测量两个光的联合变量
    - 使用可实现的线性光学元件（分束器 + Homodyne检测）

    测量结构
    --------
    1. 输入：两束光 (Mode A, Mode B)
    2. 通过50:50平衡分束器进行干涉
    3. 分束器输出：Mode C = (Mode A + Mode B)/√2, Mode D = (Mode A - Mode B)/√2
    4. 分别对Mode C做X quadrature测量，对Mode D做P quadrature测量

    物理直觉
    --------
    - 分束器将 (X_A, P_A) 和 (X_B, P_B) 转化为 (X_+, P_-)
    - X_+ = (X_A + X_B)/√2
    - P_- = (P_A - P_B)/√2
    - 这正是实现CV Bell态测量的标准配置！

    参数
    ----
    cutoff : int
        Fock空间截断维度

    num_x_bins : int
        X quadrature 的离散区间数（默认2 → 正/负二值）

    num_p_bins : int
        P quadrature 的离散区间数

    x_bounds, p_bounds : np.ndarray | None
        自定义边界，默认使用 default_quadrature_bounds

    quadrature_range : float
        有限边界范围（以标准差为单位）

    num_nodes : int | None
        数值积分节点数

    返回
    ----
    povm : list[np.ndarray]
        POVM矩阵列表，共 num_x_bins × num_p_bins 个元素

    labels : list[tuple[int, int]]
        每个POVM元对应的 (x_index, p_index) 标签

    x_edges, p_edges : np.ndarray
        X和P的区间边界（用于分析）

    数学表示
    --------
    完整的POVM元为：
        F_{kl} = B† · (F_k^X ⊗ F_l^P) · B

    其中 B 是分束器算符，F_k^X 和 F_l^P 是单模quadrature POVM。
    """
    # 确定X和P的边界
    x_edges = (
        default_quadrature_bounds(num_x_bins, finite_range=quadrature_range)
        if x_bounds is None
        else np.asarray(x_bounds, dtype=float)
    )
    p_edges = (
        default_quadrature_bounds(num_p_bins, finite_range=quadrature_range)
        if p_bounds is None
        else np.asarray(p_bounds, dtype=float)
    )

    # 构造X和P的POVM
    # theta=0.0 → X quadrature
    # theta=π/2 → P quadrature
    x_povms = quadrature_povms_from_bounds(cutoff, 0.0, x_edges, num_nodes=num_nodes)
    p_povms = quadrature_povms_from_bounds(cutoff, np.pi / 2.0, p_edges, num_nodes=num_nodes)

    # 平衡分束器算符
    beamsplitter = balanced_beamsplitter_unitary(cutoff)

    # 构造双模POVM：张量积 + 基底变换
    povm: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x_index, q_effect in enumerate(x_povms):
        for p_index, p_effect in enumerate(p_povms):
            # 构造联合POVM元：F_k^X ⊗ F_l^P
            output_effect = kron(q_effect, p_effect)

            # 通过分束器基底变换：F' = B† · F · B
            povm.append(beamsplitter.conj().T @ output_effect @ beamsplitter)
            labels.append((x_index, p_index))

    return povm, labels, x_edges, p_edges


def project_povm_to_basis(povm: list[np.ndarray], basis: np.ndarray) -> list[np.ndarray]:
    """
    将POVM投影到特定子空间
    ======================

    物理原理
    --------
    当输入态被限制在某个子空间时，对应的POVM也需要投影到该子空间。

    如果原始POVM {F_k} 作用在全空间，投影后的POVM为：
        F'_k = Π · F_k · Π

    其中 Π 是到目标子空间的投影算符。

    这个步骤确保：
    1. POVM仍然满足完备性 Σ F'_k = Π（而不是I）
    2. 概率计算只在子空间内进行

    参数
    ----
    povm : list[np.ndarray]
        原始POVM矩阵列表

    basis : np.ndarray
        目标子空间的正交归一基（列向量为基向量）

    返回
    ----
    projected : list[np.ndarray]
        投影后的POVM列表

    数学细节
    --------
    对每个POVM元，投影到由 basis 列向量张成的子空间：
        - 在原空间中：F'_k = Π · F_k · Π = basis · (basis† · F_k · basis) · basis†
        - 在约化空间中（代码实际计算）：F'_k = basis† · F_k · basis

    对称化步骤 0.5*(F'_k + F'_k†) 确保结果是厄米矩阵。
    """
    projected: list[np.ndarray] = []
    for effect in povm:
        # 投影到子空间
        reduced = basis.conj().T @ effect @ basis
        # 对称化确保厄米性
        projected.append(0.5 * (reduced + reduced.conj().T))
    return projected


def measurement_probabilities(
    input_states: list[np.ndarray],
    measurement_povm: list[np.ndarray],
) -> np.ndarray:
    """
    计算条件概率 P(output|input)
    ============================

    物理原理
    --------
    给定输入态 ρ_s 和 POVM {F_c}，测量得到结果 c 的概率为：
        P(c|s) = Tr(ρ_s · F_c)

    这就是 Born 规则在POVM形式下的表述。

    参数
    ----
    input_states : list[np.ndarray]
        输入态密度矩阵列表

    measurement_povm : list[np.ndarray]
        POVM矩阵列表

    返回
    ----
    probabilities : np.ndarray
        形状为 (len(input_states), len(measurement_povm)) 的概率矩阵
        probabilities[s, c] = P(c|s)

    数学细节
    --------
    对于纯态 ρ = |ψ⟩⟨ψ|：
        P(c|s) = ⟨ψ| F_c |ψ⟩

    对于混合态：
        P(c|s) = Tr(ρ_s F_c) = Σ_i λ_i ⟨v_i| F_c |v_i⟩

    代码实现
    --------
    使用矩阵乘法 + trace，计算复数结果的实部（物理上应该是实数）
    """
    out = np.zeros((len(input_states), len(measurement_povm)))
    for s, rho in enumerate(input_states):
        for c, povm in enumerate(measurement_povm):
            # P(c|s) = Tr(ρ · F)
            out[s, c] = float(np.real(np.trace(povm @ rho)))
    return out


def dual_homodyne_probabilities(
    joint_states: list[np.ndarray],
    joint_basis: np.ndarray,
    cutoff: int,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    num_nodes: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    """
    计算双Homodyne测量的联合概率分布
    ================================

    物理原理
    --------
    这是 Route 3 的核心函数，计算所有输入态组合下
    CV Bell测量的概率分布。

    物理流程
    --------
    1. Alice 和 Bob 各持有一个模（Mode A 和 Mode B）
    2. 他们各自准备相干态 |α_i⟩ 和 |α_j⟩
    3. 两束光在 Charlie（测量方）处通过分束器
    4. Charlie 测量 X_+ 和 P_-，得到离散结果 (k, l)

    概率计算
    --------
    P((k,l)|(i,j)) = ⟨α_i| ⊗ ⟨α_j| · F_{kl} · |α_i⟩ ⊗ |α_j⟩

    其中 F_{kl} 是 dual_homodyne_povm 构造的POVM元。

    参数
    ----
    joint_states : list[np.ndarray]
        双模输入态列表（张量积形式）

    joint_basis : np.ndarray
        双模空间的基

    cutoff : int
        Fock截断维度

    num_x_bins, num_p_bins : int
        X和P的离散区间数

    x_bounds, p_bounds : np.ndarray | None
        自定义边界

    quadrature_range : float
        有限边界范围

    num_nodes : int | None
        数值积分节点数

    返回
    ----
    probabilities : np.ndarray
        概率矩阵，形状为 (num_inputs, num_outputs)

    output_labels : list[tuple[int, int]]
        每个输出对应的 (x_index, p_index) 标签

    x_edges, p_edges : np.ndarray
        区间边界
    """
    # 构造完整的POVM（作用在全空间）
    full_povm, output_labels, x_edges, p_edges = dual_homodyne_povm(
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        x_bounds=x_bounds,
        p_bounds=p_bounds,
        quadrature_range=quadrature_range,
        num_nodes=num_nodes,
    )

    # 投影POVM到输入态的支撑空间
    reduced_povm = project_povm_to_basis(full_povm, joint_basis)

    # 计算概率分布
    probabilities = measurement_probabilities(joint_states, reduced_povm)

    return probabilities, output_labels, x_edges, p_edges


def run_route3(
    mu: float = 0.5,
    cutoff: int = 12,
    num_phases: int = 4,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    num_quadrature_nodes: int | None = None,
    max_inputs_to_certify: int | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    运行 Route 3：CV四相位 + 单设备SDP认证
    ========================================

    物理原理
    --------
    Route 3 是一种 Measurement-Device-Independent (MDI) QRNG 方案。

    协议流程
    --------
    1. Alice 随机选择相位 i ∈ {0,1,2,3}，准备态 |α_i⟩
    2. Bob 随机选择相位 j ∈ {0,1,2,3}，准备态 |α_j⟩
    3. 两束光发送到 Charlie（测量方）
    4. Charlie 对两束光做 CV Bell 测量（dual homodyne）
    5. Charlie 公布测量结果 (k, l)
    6. Alice 和 Bob 保留各自输入，丢弃不匹配的回合

    随机性认证
    ---------
    利用测量概率分布 P(k,l|i,j)，通过半定规划(SDP)计算
    最坏情况下的条件最小熵 H_min：

        H_min = min_{i} max_{j} [-log₂ P(guess|j,i)]

    其中 guess 是对 Alice 输入的最佳猜测。

    SDP的物理约束
    -------------
    - POVM约束：测量算符半正定、求和为单位算符
    - 态约束：输入态必须物理上可实现
    - 概率约束：与观察到的概率分布一致

    参数
    ----
    mu : float
        平均光子数，默认0.5

    cutoff : int
        Fock截断维度，默认12

    num_phases : int
        相位数量，默认4

    num_x_bins, num_p_bins : int
        X和P的离散区间数，默认各2（总共4个输出）

    x_bounds, p_bounds : np.ndarray | None
        自定义边界

    quadrature_range : float
        有限边界范围（标准差倍数），默认3.0

    num_quadrature_nodes : int | None
        数值积分节点数

    max_inputs_to_certify : int | None
        最多认证的输入数，默认None表示全部

    preferred_solver : str | None
        SDP求解器偏好，如 'mosek', 'scs'

    verbose : bool
        是否打印详细信息

    返回
    ----
    result : dict
        包含以下关键字段：
        - 'H_min': 认证的最小熵（bits）
        - 'raw_H_min': 原始熵估计（未优化）
        - 'target_input': 最佳输入标签
        - 'probabilities': 概率矩阵
        - 各种参数记录

    算法流程
    --------
    1. 生成输入态和概率分布
    2. 对每个输入态计算原始熵
    3. 按熵降序排列输入
    4. 对每个候选输入运行SDP
    5. 返回最佳结果
    """
    # 生成联合输入态
    joint_states, labels, joint_basis, local_rank, joint_dim = reduced_joint_inputs(
        mu,
        cutoff,
        num_phases=num_phases,
    )

    # 计算CV Bell测量概率
    probabilities, output_labels, x_bounds, p_bounds = dual_homodyne_probabilities(
        joint_states,
        joint_basis,
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        x_bounds=x_bounds,
        p_bounds=p_bounds,
        quadrature_range=quadrature_range,
        num_nodes=num_quadrature_nodes,
    )

    # 计算每个输入的原始熵估计
    # H = -log₂ max_c P(c|i)
    # 这是最佳猜测熵的简单上界
    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))

    # 按熵降序排列输入
    candidate_order = list(np.argsort(-raw_h))
    if max_inputs_to_certify is not None:
        candidate_order = candidate_order[:max_inputs_to_certify]

    # 对每个候选输入运行SDP
    best: dict | None = None
    reusable_problem = SingleDeviceGuessingProblem(joint_states, probabilities)

    for target_input in candidate_order:
        current = reusable_problem.solve(
            target_input=target_input,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )
        current["target_input"] = labels[target_input]
        current["raw_H_min"] = float(raw_h[target_input])

        # 保留最佳结果（H_min最大的）
        if best is None or (current["H_min"] or -np.inf) > (best["H_min"] or -np.inf):
            best = current

    assert best is not None

    # 组装完整结果
    best.update(
        {
            "route": "route3_cv_four_phase",
            "mu": mu,
            "cutoff": cutoff,
            "num_phases": num_phases,
            "num_inputs": len(joint_states),
            "num_outputs": probabilities.shape[1],
            "output_labels": output_labels,
            "local_rank": local_rank,
            "joint_dim": joint_dim,
            "operator_span_rank": operator_span_rank(joint_states),
            "operator_space_dim": joint_dim**2,
            "x_bounds": x_bounds.tolist(),
            "p_bounds": p_bounds.tolist(),
            "num_x_bins": num_x_bins,
            "num_p_bins": num_p_bins,
            "quadrature_range": quadrature_range,
            "num_quadrature_nodes": default_quadrature_nodes(cutoff)
            if num_quadrature_nodes is None
            else int(num_quadrature_nodes),
            "num_inputs_certified": len(candidate_order),
        }
    )

    return best
