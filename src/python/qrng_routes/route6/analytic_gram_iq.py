"""
Route 6: Gram矩阵 + 解析高斯IQ测量 QRNG认证
==============================================

物理背景
--------
本模块实现了一种基于 **Gram矩阵精确态表示** 和 **解析高斯概率计算** 的
连续变量量子随机数认证方案。与 Route 5 的核心区别在于：

1. **态表示方式不同**：
   - Route 5: 在 Fock 基下截断，构造有限维密度矩阵（需要 cutoff 参数）
   - Route 6: 利用相干态的 Gram 矩阵（重叠矩阵）精确表示态矢，
     无需 Fock 截断，避免了截断误差

2. **概率计算方式不同**：
   - Route 5: 在 Fock 空间中数值积分计算 quadrature POVM
   - Route 6: 利用相干态高斯分布的解析公式直接计算区间概率，
     使用误差函数(erf)给出精确结果

核心物理概念：
- Gram矩阵: G_ij = ⟨α_i|α_j⟩，相干态之间的重叠（内积）
- 相干态重叠公式: ⟨α|β⟩ = exp(-1/2(|α|²+|β|²) + α*β)
- Gram对角化: 将Gram矩阵的特征向量作为支撑空间的正交基，
  精确表示相干态张成的有限维子空间
- 解析高斯概率: 相干态在X和P方向上的边缘分布是高斯分布，
  在任意区间上的概率可由误差函数精确计算
- IQ测量: 同时测量X（同相分量I）和P（正交分量Q）两个quadrature
- 幂律分箱(power-spaced binning): 使用幂函数生成非均匀分箱边界，
  gamma参数控制分箱密度在原点附近的集中程度

Route 6 的优势：
- 无Fock截断误差：Gram矩阵方法在有限维子空间中精确表示相干态
- 解析概率：使用erf函数替代数值积分，计算速度快且无积分误差
- 适用于小振幅相干态（|α| 较小时，Fock截断需要很大维度才能收敛，
  而Gram方法仅需与态数量相同的维度）

模块结构：
- 数据类：AxisBoundsCandidate, AlphabetCandidateSpec
- 相干态代数：coherent_overlap, gram_state_vectors_from_alphas
- 字母表生成：generalized_coherent_alphabet, generate_*_subsets
- 态构造：exact_joint_inputs_from_alphas
- 分箱构造：power_spaced_bounds, generate_axis_bound_candidates
- 解析概率：analytic_iq_probabilities
- SDP认证：certify_target_inputs
- 高层接口：run_route6, search_route6_*, search_route6_fixed_partition_alphabets
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from scipy.special import erf

from ..common import SingleDeviceGuessingProblem, density_from_ket, kron, operator_span_rank


# ── 全局默认参数 ──────────────────────────────────────────────────────────
# DEFAULT_RADIUS_VALUES: 相空间中相干态的默认半径值
# 0.0 对应真空态，0.6 和 1.2 对应两种不同强度的相干态
# 这些值在 X-P 相空间中形成三个同心圆
DEFAULT_RADIUS_VALUES = [0.0, 0.6, 1.2]

# DEFAULT_PHASE_VALUES: 默认的四相位集合 {0, π/2, π, 3π/2}
# 四个方向均匀分布在相空间中，与 X 和 P 轴对齐
DEFAULT_PHASE_VALUES = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

# DEFAULT_QUADRATURE_RANGES: IQ测量的默认有限边界范围（标准差倍数）
# 2.0覆盖约95%概率，3.0覆盖约99.7%，4.0覆盖约99.99%
DEFAULT_QUADRATURE_RANGES = [2.0, 3.0, 4.0]

# DEFAULT_GAMMA_VALUES: 幂律分箱的默认间距参数
# gamma < 1: 分箱在原点附近更密集（有利于小振幅态）
# gamma = 1: 均匀分箱
# gamma > 1: 分箱在边界附近更密集
DEFAULT_GAMMA_VALUES = [0.75, 1.0, 1.5]


@dataclass(frozen=True)
class AxisBoundsCandidate:
    """
    单个quadrature轴的分箱边界候选配置

    属性
    ----
    num_bins : int
        分箱数量（如 2 表示正/负两个区间）
    finite_range : float
        有限边界的范围（±finite_range），超出部分延伸到 ±∞
    gamma : float
        幂律间距参数。gamma=1 为均匀分箱，gamma<1 原点附近更密集
    bounds : tuple[float, ...]
        完整的边界值列表，长度为 num_bins+1，首尾为 ±inf
    """
    num_bins: int
    finite_range: float
    gamma: float
    bounds: tuple[float, ...]


@dataclass(frozen=True)
class AlphabetCandidateSpec:
    """
    相干态字母表的候选规格

    描述一组用于QRNG协议的可信相干态集合。
    由半径子集和相位子集的笛卡尔积生成。

    属性
    ----
    radius_values : tuple[float, ...]
        选用的半径值（|α|），按升序排列
    phase_values : tuple[float, ...]
        选用的相位值（arg α），按升序排列
    alpha_values : tuple[complex, ...]
        最终的复振幅列表 α = radius × e^(iφ)，已去重
    """
    radius_values: tuple[float, ...]
    phase_values: tuple[float, ...]
    alpha_values: tuple[complex, ...]


def _canonicalize_scalar(value: float) -> float:
    """
    将极小的有限浮点数归零，避免数值噪声

    参数
    ----
    value : float
        待规范化的标量值

    返回
    ----
    result : float
        若 |value| < 1e-12 且有限则返回 0.0，否则原样返回
    """
    if math.isfinite(value) and abs(value) < 1e-12:
        return 0.0
    return float(value)


def _serialize_complex(alpha: complex) -> dict:
    """
    将复数序列化为JSON友好的字典

    参数
    ----
    alpha : complex
        复数（通常为相干态振幅）

    返回
    ----
    result : dict
        包含 real, imag, abs, phase 四个字段的字典
    """
    return {
        "real": float(np.real(alpha)),
        "imag": float(np.imag(alpha)),
        "abs": float(abs(alpha)),
        "phase": float(np.angle(alpha)),
    }


def serialize_complex_list(alphas: list[complex]) -> list[dict]:
    """
    批量序列化复数列表

    参数
    ----
    alphas : list[complex]
        复数列表

    返回
    ----
    result : list[dict]
        序列化后的字典列表
    """
    return [_serialize_complex(alpha) for alpha in alphas]


def _normalize_phase(value: float) -> float:
    """
    将相位归一化到 [0, 2π) 区间

    参数
    ----
    value : float
        原始相位值（弧度）

    返回
    ----
    normalized : float
        归一化后的相位，0 ≤ result < 2π
    """
    return float(value % (2.0 * math.pi))


def _deduplicate_alphas(alphas: list[complex], tol: float = 1e-12) -> list[complex]:
    """
    去除重复的复振幅（保持原始顺序）

    参数
    ----
    alphas : list[complex]
        复振幅列表（可能含重复）
    tol : float
        判定重复的容差（|α_i - α_j| ≤ tol 视为相同）

    返回
    ----
    unique : list[complex]
        去重后的列表
    """
    unique: list[complex] = []
    for alpha in alphas:
        if any(abs(alpha - existing) <= tol for existing in unique):
            continue
        unique.append(alpha)
    return unique


def _unique_sorted_radii(radius_values: list[float], tol: float = 1e-12) -> list[float]:
    """
    对半径值列表去重并排序

    参数
    ----
    radius_values : list[float]
        原始半径值（可能有重复或无序）
    tol : float
        判定重复的容差

    返回
    ----
    unique : list[float]
        升序排列的去重列表
    """
    unique: list[float] = []
    for radius in sorted(float(value) for value in radius_values):
        if any(abs(radius - existing) <= tol for existing in unique):
            continue
        unique.append(radius)
    return unique


def _unique_sorted_phases(phase_values: list[float], tol: float = 1e-12) -> list[float]:
    """
    对相位值去重并排序（先归一化到 [0, 2π)）

    参数
    ----
    phase_values : list[float]
        原始相位值
    tol : float
        判定重复的容差

    返回
    ----
    unique : list[float]
        升序排列的去重列表
    """
    unique: list[float] = []
    for phase in sorted(_normalize_phase(value) for value in phase_values):
        if any(abs(phase - existing) <= tol for existing in unique):
            continue
        unique.append(phase)
    return unique


def coherent_overlap(alpha: complex, beta: complex) -> complex:
    """
    计算两个相干态的内积（重叠） ⟨α|β⟩
    ======================================

    物理原理
    --------
    相干态 |α⟩ 和 |β⟩ 的内积为：

        ⟨α|β⟩ = exp(-1/2(|α|² + |β|²) + α*β)

    这是Route 6的核心数学基础：
    - |⟨α|β⟩| = exp(-1/2|α-β|²)，即两个相干态的距离越远，重叠越小
    - ⟨α|α⟩ = 1（归一性）
    - ⟨α|β⟩ ≠ 0（相干态的非正交性——过完备性）

    参数
    ----
    alpha, beta : complex
        两个相干态的复振幅

    返回
    ----
    overlap : complex
        内积值，模 ≤ 1

    示例
    ----
    >>> coherent_overlap(1.0, 1.0)   # 同一个态
    (1+0j)
    >>> abs(coherent_overlap(0.0, 1.0))  # 真空与|α=1⟩的重叠
    0.6065...  # = e^(-0.5)
    """
    return np.exp(-0.5 * (abs(alpha) ** 2 + abs(beta) ** 2) + np.conjugate(alpha) * beta)


def gram_state_vectors_from_alphas(
    alpha_values: list[complex],
    tol: float = 1e-10,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """
    从相干态振幅构造Gram矩阵并提取支撑空间中的正交态矢量
    =====================================================

    物理原理
    --------
    这是 Route 6 的核心态表示方法。

    给定一组相干态 {|α_k⟩}，定义 Gram 矩阵 G：
        G_ij = ⟨α_i|α_j⟩ = exp(-1/2(|α_i|²+|α_j|²) + α_i*·α_j)

    Gram矩阵的性质：
    - G 是半正定厄米矩阵（G = G†, G ≥ 0）
    - G 的秩 = 相干态张成的子空间维度
    - G 的特征值 λ_k 满足 0 < λ_k ≤ 1

    Gram对角化方法：
    对 G 做特征分解 G = V·Λ·V†，其中 Λ = diag(λ_1,...,λ_r)。
    则在特征向量张成的 r 维子空间中，每个相干态 |α_k⟩ 可以精确表示为：

        |α_k⟩ = Σ_j √λ_j · V_jk* |e_j⟩

    其中 {|e_j⟩} 是特征向量定义的正交基。

    参数
    ----
    alpha_values : list[complex]
        相干态的复振幅列表
    tol : float
        特征值截断阈值，小于 tol 的特征值对应的维度被丢弃

    返回
    ----
    state_vectors : list[np.ndarray]
        每个相干态在支撑空间中的矢量表示，长度为 rank
    gram : np.ndarray
        Gram矩阵，形状为 (n, n)
    rank : int
        支撑空间的维度（Gram矩阵的有效秩）

    数学细节
    --------
    构造矩阵 Ψ = diag(√λ) · V†，则 Ψ 的每一列就是一个态矢量。
    验证：Ψ†·Ψ 的 (i,j) 元素 = Σ_k √λ_k · V_ki · √λ_k · V_kj*
             = Σ_k λ_k · V_ki · V_kj* = G_ij ✓
    """
    if len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")

    gram = np.array(
        [[coherent_overlap(alpha_i, alpha_j) for alpha_j in alpha_values] for alpha_i in alpha_values],
        dtype=complex,
    )
    gram = 0.5 * (gram + gram.conj().T)
    values, vectors = np.linalg.eigh(gram)
    keep = values > tol
    if not np.any(keep):
        raise RuntimeError("The coherent-alphabet Gram matrix is numerically rank deficient.")

    kept_values = values[keep]
    kept_vectors = vectors[:, keep]
    psi = np.diag(np.sqrt(kept_values)) @ kept_vectors.conj().T
    state_vectors = [psi[:, index].copy() for index in range(psi.shape[1])]
    return state_vectors, gram, int(psi.shape[0])


def generalized_coherent_alphabet(
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
) -> list[complex]:
    """
    构造广义相干态字母表
    ====================

    物理原理
    --------
    相干态字母表是 QRNG 协议中可信源可制备的所有量子态集合。
    在相空间（X-P 平面）中，每个相干态 |α⟩ 由复振幅 α = r·e^(iφ) 确定，
    其中 r 为半径（振幅大小），φ 为相位（方向）。

    字母表有两种构造方式：
    1. 直接指定：给出完整的复振幅列表 alpha_values
    2. 网格构造：通过半径和相位的笛卡尔积生成
       α_{k,l} = r_k · e^(i·φ_l)

    当使用网格构造时，字母表中的态在相空间中排列在以原点为圆心的
    同心圆上，每个圆上均匀分布若干个相干态。

    去重机制：由于半径为 0（真空态）的相干态在所有相位上重叠，
    构造后会自动去除重复态。

    参数
    ----
    alpha_values : list[complex] | None
        直接指定的复振幅列表。若提供则忽略 radius_values 和 phase_values。
        不能为空列表。默认为 None（使用网格构造）。
    radius_values : list[float] | None
        半径值列表（|α|）。alpha_values 为 None 时生效。
        默认使用 DEFAULT_RADIUS_VALUES = [0.0, 0.6, 1.2]。
    phase_values : list[float] | None
        相位值列表（arg α，弧度）。alpha_values 为 None 时生效。
        默认使用 DEFAULT_PHASE_VALUES = [0, π/2, π, 3π/2]。

    返回
    ----
    alphas : list[complex]
        去重后的相干态复振幅列表。顺序与输入一致。

    异常
    ----
    ValueError
        当 alpha_values 为空列表，或 radius_values/phase_values 为空时抛出。
    """
    if alpha_values is not None and len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")

    if alpha_values is None:
        radii = DEFAULT_RADIUS_VALUES if radius_values is None else list(radius_values)
        phases = DEFAULT_PHASE_VALUES if phase_values is None else list(phase_values)
        if len(radii) == 0 or len(phases) == 0:
            raise ValueError("radius_values and phase_values must be non-empty.")
        alpha_values = [radius * np.exp(1j * phase) for radius in radii for phase in phases]

    return _deduplicate_alphas([complex(alpha) for alpha in alpha_values])


def _build_alpha_values(radius_values: tuple[float, ...], phase_values: tuple[float, ...]) -> tuple[complex, ...]:
    """
    由半径和相位子集构建去重的复振幅元组
    ====================================

    用途
    ----
    将给定的半径子集和相位子集通过笛卡尔积生成相干态复振幅，
    并自动去重（主要处理真空态在不同相位下的重复问题）。

    这是 generate_alphabet_candidates_from_grid 内部使用的辅助函数，
    将网格搜索中的 (radius_subset, phase_subset) 对转换为具体的
    Alpha 值集合。

    参数
    ----
    radius_values : tuple[float, ...]
        半径值元组，已排序去重
    phase_values : tuple[float, ...]
        相位值元组，已排序去重

    返回
    ----
    alpha_values : tuple[complex, ...]
        去重后的复振幅元组，每个元素为 radius * exp(i·phase)
    """
    alpha_values = [radius * np.exp(1j * phase) for radius in radius_values for phase in phase_values]
    return tuple(_deduplicate_alphas([complex(alpha) for alpha in alpha_values]))


def generate_radius_subsets(
    radius_values: list[float],
    num_radii_values: list[int],
    require_vacuum: bool = True,
) -> list[tuple[float, ...]]:
    """
    生成所有合法的半径子集组合
    ==========================

    用途
    ----
    在字母表搜索中，需要枚举不同半径值的组合方案。本函数根据
    给定的半径候选值和每组所需的半径数量，生成所有合法的子集。

    当 require_vacuum=True 时，真空态（r=0）被视为必选态，
    其余半径值通过组合枚举。这反映了物理约束：源始终能制备
    真空态（不发光），它是量子随机性的重要来源之一。

    生成策略：
    - require_vacuum=True 且存在 r≈0：
      对每个 requested_count，子集 = {0} ∪ C(nonzero_radii, count-1)
    - require_vacuum=False：
      对每个 requested_count，子集 = C(all_radii, count)

    参数
    ----
    radius_values : list[float]
        候选半径值列表（可含重复和无序元素）
    num_radii_values : list[int]
        需要枚举的半径数量列表。例如 [1, 2, 3] 表示分别生成
        含 1、2、3 个半径的子集。
    require_vacuum : bool
        是否要求每个子集都包含真空态（r=0）。默认 True。

    返回
    ----
    subsets : list[tuple[float, ...]]
        所有合法子集的列表，每个子集为升序排列的元组。
        子集间保证不重复（通过 seen 集合去重）。

    异常
    ----
    ValueError
        当 radius_values 为空时抛出。
    """
    unique_radii = _unique_sorted_radii(radius_values)
    if len(unique_radii) == 0:
        raise ValueError("radius_values cannot be empty.")

    zero_radii = [radius for radius in unique_radii if abs(radius) <= 1e-12]
    has_vacuum = len(zero_radii) > 0
    vacuum = zero_radii[0] if has_vacuum else None
    nonzero_radii = [radius for radius in unique_radii if abs(radius) > 1e-12]

    subsets: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for requested_count in sorted(set(int(value) for value in num_radii_values if int(value) > 0)):
        if require_vacuum and has_vacuum:
            if requested_count == 1:
                subset = (float(vacuum),)
                if subset not in seen:
                    seen.add(subset)
                    subsets.append(subset)
                continue
            choose_count = requested_count - 1
            if choose_count > len(nonzero_radii):
                continue
            for combo in combinations(nonzero_radii, choose_count):
                subset = (float(vacuum),) + tuple(float(value) for value in combo)
                if subset in seen:
                    continue
                seen.add(subset)
                subsets.append(subset)
        else:
            if requested_count > len(unique_radii):
                continue
            for combo in combinations(unique_radii, requested_count):
                subset = tuple(float(value) for value in combo)
                if subset in seen:
                    continue
                seen.add(subset)
                subsets.append(subset)
    return subsets


def _phase_subset_from_offset(phases: list[float], count: int, offset: int) -> tuple[float, ...]:
    """
    从均匀相位集合中按等间距偏移选取子集
    ======================================

    用途
    ----
    给定一组均匀分布的相位值，从中选取 count 个等间距分布的相位。
    起始位置由 offset 参数控制。当等间距选取发生冲突（两个索引
    映射到同一位置）时，自动向后寻找最近的未使用位置。

    这保证了从 N 个均匀相位中选取 M 个时，所选相位尽可能均匀
    分布在 [0, 2π) 区间上，使相空间覆盖最大化。

    参数
    ----
    phases : list[float]
        完整的相位值列表（已排序）
    count : int
        需要选取的相位数量
    offset : int
        起始偏移索引

    返回
    ----
    subset : tuple[float, ...]
        选取的相位子集，按原始索引升序排列
    """
    total = len(phases)
    chosen: list[int] = []
    used: set[int] = set()
    for index in range(count):
        raw = int(round(offset + index * total / count)) % total
        while raw in used:
            raw = (raw + 1) % total
        used.add(raw)
        chosen.append(raw)
    return tuple(float(phases[position]) for position in sorted(chosen))


def generate_phase_subsets(
    phase_values: list[float],
    num_phase_values: list[int],
) -> list[tuple[float, ...]]:
    """
    生成所有合法的相位子集组合
    ==========================

    用途
    ----
    在字母表搜索中，需要枚举不同相位值的组合方案。本函数根据
    给定的相位候选值和每组所需的相位数量，生成所有合法的子集。

    对于均匀分布的相位集合，子集选取策略如下：
    1. 若 requested_count == 总数：返回完整集合
    2. 若总数能被 requested_count 整除：按等间距选取，
       枚举所有偏移起始位置（确保均匀覆盖相空间）
    3. 否则：使用 _phase_subset_from_offset 枚举所有
       可能的偏移起始位置

    物理意义：相位决定了相干态在 X-P 平面上的方向。
    选择均匀分布的相位子集可以最大化相空间的覆盖，
    有利于提高认证安全性和随机性产量。

    参数
    ----
    phase_values : list[float]
        候选相位值列表（弧度），可含重复和无序元素
    num_phase_values : list[int]
        需要枚举的相位数量列表。例如 [2, 4] 表示分别
        生成含 2 和 4 个相位的子集。

    返回
    ----
    subsets : list[tuple[float, ...]]
        所有合法子集的列表，每个子集为升序排列的元组。
        子集间保证不重复。

    异常
    ----
    ValueError
        当 phase_values 为空时抛出。
    """
    unique_phases = _unique_sorted_phases(phase_values)
    if len(unique_phases) == 0:
        raise ValueError("phase_values cannot be empty.")

    subsets: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    total = len(unique_phases)
    for requested_count in sorted(set(int(value) for value in num_phase_values if int(value) > 0)):
        if requested_count > total:
            continue
        if requested_count == total:
            subset = tuple(unique_phases)
            if subset not in seen:
                seen.add(subset)
                subsets.append(subset)
            continue
        if total % requested_count == 0:
            step = total // requested_count
            for offset in range(step):
                subset = tuple(float(unique_phases[offset + index * step]) for index in range(requested_count))
                if subset in seen:
                    continue
                seen.add(subset)
                subsets.append(subset)
            continue
        for offset in range(total):
            subset = _phase_subset_from_offset(unique_phases, requested_count, offset)
            if subset in seen:
                continue
            seen.add(subset)
            subsets.append(subset)
    return subsets


def generate_alphabet_candidates_from_grid(
    radius_values: list[float],
    phase_values: list[float],
    num_radii_values: list[int],
    num_phase_values: list[int],
    require_vacuum: bool = True,
    max_local_states: int | None = None,
) -> list[AlphabetCandidateSpec]:
    """
    通过半径×相位网格搜索生成所有字母表候选规格
    =============================================

    用途
    ----
    这是字母表搜索的顶层枚举函数。将半径子集和相位子集的
    笛卡尔积转化为完整的 AlphabetCandidateSpec 列表。

    工作流程：
    1. 调用 generate_radius_subsets 生成所有半径子集
    2. 调用 generate_phase_subsets 生成所有相位子集
    3. 对每个 (radius_subset, phase_subset) 对：
       a. 构建复振幅集合
       b. 检查是否超过 max_local_states 限制
       c. 去重后生成 AlphabetCandidateSpec

    物理意义：不同的字母表对应源的不同制备能力。字母表越大，
    可制备的态越多，但 SDP 求解的计算复杂度也越高（联合空间
    维度随本地态数量平方增长）。

    参数
    ----
    radius_values : list[float]
        候选半径值的完整列表
    phase_values : list[float]
        候选相位值的完整列表
    num_radii_values : list[int]
        需要枚举的半径数量列表
    num_phase_values : list[int]
        需要枚举的相位数量列表
    require_vacuum : bool
        是否要求每个半径子集包含真空态。默认 True。
    max_local_states : int | None
        每个字母表允许的最大本地态数量。超过此限制的候选
        将被跳过。None 表示不限制。默认 None。

    返回
    ----
    candidates : list[AlphabetCandidateSpec]
        所有合法的字母表候选规格列表，保证不重复。
    """
    radius_subsets = generate_radius_subsets(radius_values, num_radii_values, require_vacuum=require_vacuum)
    phase_subsets = generate_phase_subsets(phase_values, num_phase_values)

    candidates: list[AlphabetCandidateSpec] = []
    seen: set[tuple[complex, ...]] = set()
    for radius_subset in radius_subsets:
        for phase_subset in phase_subsets:
            alpha_values = _build_alpha_values(radius_subset, phase_subset)
            if max_local_states is not None and len(alpha_values) > max_local_states:
                continue
            if alpha_values in seen:
                continue
            seen.add(alpha_values)
            candidates.append(
                AlphabetCandidateSpec(
                    radius_values=tuple(float(value) for value in radius_subset),
                    phase_values=tuple(float(value) for value in phase_subset),
                    alpha_values=alpha_values,
                )
            )
    return candidates


def exact_joint_inputs_from_alphas(
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    tol: float = 1e-10,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[complex], int, int, int, np.ndarray]:
    """
    从相干态振幅构造精确的联合输入态（Gram矩阵方法）
    =================================================

    物理原理
    --------
    在单设备半设备无关 QRNG 协议中，认证需要构造双体联合态。
    对于每个输入对 (x, y)，源制备联合态 ρ_xy = |α_x⟩⟨α_x| ⊗ |α_y⟩⟨α_y|，
    其中 |α_x⟩ 和 |α_y⟩ 是本地相干态。

    构造流程：
    1. 通过 generalized_coherent_alphabet 获取本地态的复振幅列表
    2. 通过 gram_state_vectors_from_alphas 在 Gram 支撑空间中
       获取每个相干态的精确矢量表示 |ψ_k⟩
    3. 将每个 |ψ_k⟩ 转换为密度矩阵 ρ_k = |ψ_k⟩⟨ψ_k|
    4. 对所有本地态对 (x, y) 构造联合态 ρ_xy = ρ_x ⊗ ρ_y
       （使用 Kronecker 积 kron）

    这里使用 Gram 矩阵方法而非 Fock 截断，因此态表示是精确的，
    没有截断误差。

    参数
    ----
    alpha_values : list[complex] | None
        直接指定的复振幅列表。若提供则忽略后两个参数。
    radius_values : list[float] | None
        半径候选值（alpha_values 为 None 时生效）
    phase_values : list[float] | None
        相位候选值（alpha_values 为 None 时生效）
    tol : float
        Gram 矩阵特征值截断阈值，默认 1e-10。
        小于此阈值的特征值对应的维度被丢弃。

    返回
    ----
    joint_states : list[np.ndarray]
        联合输入态列表（密度矩阵），长度为 n²（n 为本地态数）。
        每个 ρ_xy 的形状为 (rank², rank²)。
    labels : list[tuple[int, int]]
        每个联合态对应的本地态索引对 (x, y)
    local_alphas : list[complex]
        本地相干态的复振幅列表
    local_rank : int
        本地 Gram 支撑空间的维度
    joint_dim : int
        联合空间的维度 = local_rank²
    local_operator_span : int
        本地态集合张成的算符空间秩（反映测量能力的丰富程度）
    local_gram : np.ndarray
        本地态的 Gram 矩阵，形状 (n, n)
    """
    local_alphas = generalized_coherent_alphabet(
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
    )
    local_kets, local_gram, local_rank = gram_state_vectors_from_alphas(local_alphas, tol=tol)
    local_states = [density_from_ket(ket) for ket in local_kets]

    joint_states: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x, ket_a in enumerate(local_kets):
        for y, ket_b in enumerate(local_kets):
            joint_states.append(density_from_ket(kron(ket_a, ket_b)))
            labels.append((x, y))

    joint_dim = local_rank**2
    local_operator_span = operator_span_rank(local_states)
    return (
        joint_states,
        labels,
        local_alphas,
        int(local_rank),
        int(joint_dim),
        int(local_operator_span),
        local_gram,
    )


def power_spaced_bounds(num_bins: int, finite_range: float, gamma: float = 1.0) -> np.ndarray:
    """
    生成幂律间距的分箱边界
    ======================

    物理原理
    --------
    在 IQ 测量中，连续的 quadrature 测量值需要被离散化为有限个输出。
    分箱边界决定了输出符号的概率分布。

    幂律间距（power-spaced）通过非线性变换控制边界密度：
    - 将 [-1, 1] 均匀分布的归一化坐标通过幂函数变换：
      edge = sign(t) · |t|^gamma · finite_range
    - 首尾两个边界固定为 -∞ 和 +∞，覆盖整个实数轴

    gamma 参数的物理效果：
    - gamma < 1：边界在原点附近更密集，对真空态（原点处的高斯峰）
      的精细分辨更有利
    - gamma = 1：等间距边界（线性分箱）
    - gamma > 1：边界在 ±finite_range 附近更密集，
      对远离原点的尾部区域分辨更精细

    特殊情况 num_bins=2：仅使用单条边界 {−∞, 0, +∞}，
    即正负二分法，与 gamma 和 finite_range 无关。

    参数
    ----
    num_bins : int
        分箱数量。必须 ≥ 2。
    finite_range : float
        有限边界的范围（±finite_range）。边界超出此范围的部分
        延伸到 ±∞。必须 > 0。
    gamma : float
        幂律间距参数。必须 > 0。
        gamma=1 为均匀分箱，gamma<1 原点附近更密集。默认 1.0。

    返回
    ----
    bounds : np.ndarray
        分箱边界数组，长度为 num_bins + 1。
        bounds[0] = -inf, bounds[-1] = +inf。
        极小值（|x| < 1e-12）被归零以避免数值噪声。

    异常
    ----
    ValueError
        当 num_bins < 2、finite_range ≤ 0 或 gamma ≤ 0 时抛出。
    """
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    if finite_range <= 0:
        raise ValueError("finite_range must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    if num_bins == 2:
        return np.array([-np.inf, 0.0, np.inf], dtype=float)

    normalized = np.linspace(-1.0, 1.0, num_bins + 1, dtype=float)
    edges = np.sign(normalized) * (np.abs(normalized) ** gamma) * finite_range
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.array([_canonicalize_scalar(value) for value in edges], dtype=float)


def generate_axis_bound_candidates(
    num_bins_values: list[int],
    quadrature_ranges: list[float],
    gamma_values: list[float],
) -> list[AxisBoundsCandidate]:
    """
    生成单个 quadrature 轴上所有分箱边界候选配置
    ==============================================

    用途
    ----
    对分箱参数的三个维度（分箱数、有限范围、幂律参数）做网格枚举，
    生成所有合法且互不重复的 AxisBoundsCandidate 对象。

    枚举逻辑：
    - num_bins=2：仅生成一个候选（正负二分法，忽略 range 和 gamma）
    - num_bins≥3：对 (range, gamma) 的每个组合生成一个候选

    所有候选通过 seen 集合按 (num_bins, bounds_tuple) 去重，
    避免不同参数组合产生相同的边界配置。

    参数
    ----
    num_bins_values : list[int]
        候选分箱数量列表。不能为空。
        常见值：[2]（仅二分法）、[2, 4, 6]（多粒度搜索）
    quadrature_ranges : list[float]
        候选有限范围列表（标准差倍数）。
        常见值：[2.0, 3.0, 4.0]
    gamma_values : list[float]
        候选幂律间距参数列表。
        gamma<1 原点密集，gamma=1 均匀，gamma>1 边界密集。

    返回
    ----
    candidates : list[AxisBoundsCandidate]
        所有去重后的分箱边界候选列表。

    异常
    ----
    ValueError
        当 num_bins_values 为空时抛出。
    """
    if len(num_bins_values) == 0:
        raise ValueError("num_bins_values cannot be empty.")

    candidates: list[AxisBoundsCandidate] = []
    seen: set[tuple[int, tuple[float, ...]]] = set()
    for num_bins in num_bins_values:
        if num_bins == 2:
            bounds = tuple(power_spaced_bounds(2, 1.0, gamma=1.0).tolist())
            key = (2, bounds)
            if key not in seen:
                seen.add(key)
                candidates.append(AxisBoundsCandidate(num_bins=2, finite_range=0.0, gamma=1.0, bounds=bounds))
            continue

        for finite_range in quadrature_ranges:
            for gamma in gamma_values:
                bounds = tuple(power_spaced_bounds(num_bins, finite_range, gamma=gamma).tolist())
                key = (num_bins, bounds)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    AxisBoundsCandidate(
                        num_bins=int(num_bins),
                        finite_range=float(finite_range),
                        gamma=float(gamma),
                        bounds=bounds,
                    )
                )
    return candidates


def _gaussian_interval_probability(lower: np.ndarray, upper: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    计算标准高斯分布在多个区间上的概率（向量化）
    ==============================================

    数学原理
    --------
    对于标准正态分布 N(0,1)，其在区间 [lower, upper) 上的概率为：

        P(lower ≤ X < upper) = Φ(upper - μ) - Φ(lower - μ)

    其中 Φ(x) = (1/2)(1 + erf(x/√2)) 是标准正态的累积分布函数。

    利用 Φ(x) - Φ(y) = (1/2)(erf(x/√2) - erf(y/√2)) 和
    注意到相干态的 X/P quadrature 方差为 1/2，
    经过变量替换后，区间概率可以写为：

        P = (1/2)(erf(upper - mean) - erf(lower - mean))

    其中 mean 是该 quadrature 分量的均值。

    此函数支持向量化输入，一次性计算所有区间的概率。

    参数
    ----
    lower : np.ndarray
        区间下界数组（长度为 n）
    upper : np.ndarray
        区间上界数组（长度为 n）
    mean : np.ndarray
        高斯分布的均值数组（广播到与 lower/upper 相同形状）

    返回
    ----
    probabilities : np.ndarray
        每个区间上的概率值数组，长度为 n
    """
    return 0.5 * (erf(upper - mean) - erf(lower - mean))


def analytic_iq_probabilities(
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    x_bounds: np.ndarray,
    p_bounds: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    """
    解析计算所有联合输入的 IQ 测量输出概率矩阵
    =============================================

    物理原理
    --------
    IQ 测量（双 homodyne）同时测量相干态的 X（同相）和 P（正交）
    两个正交分量。对于联合输入 (x, y) 对应的态 |α_x⟩ ⊗ |α_y⟩：

    X quadrature 的均值：μ_X = Re(α_x + α_y)
    P quadrature 的均值：μ_P = Im(α_x - α_y)

    相干态的 X 和 P 边缘分布都是方差为 1/2 的高斯分布，
    且 X 和 P 相互独立（联合分布为二维高斯）。

    因此，联合输入 (x, y) 在矩形区域 [x_i, x_{i+1}) × [p_j, p_{j+1})
    上的概率为：

        P(i,j | x,y) = P_X(x_i ≤ X < x_{i+1}) × P_P(p_j ≤ P < p_{j+1})

    其中每个一维概率通过 _gaussian_interval_probability（基于 erf）
    解析计算。最终概率表按行归一化。

    参数
    ----
    labels : list[tuple[int, int]]
        联合输入的本地态索引对列表 (x, y)，长度为 n_inputs
    local_alphas : list[complex]
        本地相干态的复振幅列表
    x_bounds : np.ndarray
        X 轴的分箱边界，长度为 n_x_bins + 1，首尾为 ±inf
    p_bounds : np.ndarray
        P 轴的分箱边界，长度为 n_p_bins + 1，首尾为 ±inf

    返回
    ----
    probabilities : np.ndarray
        归一化概率矩阵，形状 (n_inputs, n_outputs)。
        n_outputs = n_x_bins × n_p_bins。
        probabilities[i, :] 是第 i 个输入的输出概率分布。
    output_labels : list[tuple[int, int]]
        输出标签列表 (i_x, i_p)，按行优先排列
    x_edges : np.ndarray
        实际使用的 X 边界（与输入相同）
    p_edges : np.ndarray
        实际使用的 P 边界（与输入相同）

    异常
    ----
    ValueError
        当边界不是一维数组或长度不足时抛出。
    RuntimeError
        当某个输入的概率表总和为零时抛出。
    """
    x_edges = np.asarray(x_bounds, dtype=float)
    p_edges = np.asarray(p_bounds, dtype=float)
    if x_edges.ndim != 1 or p_edges.ndim != 1:
        raise ValueError("x_bounds and p_bounds must be 1-D arrays.")
    if x_edges.size < 3 or p_edges.size < 3:
        raise ValueError("x_bounds and p_bounds must each define at least two bins.")

    num_x_bins = x_edges.size - 1
    num_p_bins = p_edges.size - 1
    output_labels = [(i, j) for i in range(num_x_bins) for j in range(num_p_bins)]
    probabilities = np.zeros((len(labels), num_x_bins * num_p_bins), dtype=float)

    x_lower = x_edges[:-1]
    x_upper = x_edges[1:]
    p_lower = p_edges[:-1]
    p_upper = p_edges[1:]

    for input_index, (x_index, y_index) in enumerate(labels):
        alpha = local_alphas[x_index]
        beta = local_alphas[y_index]
        mu_x = np.real(alpha + beta)
        mu_p = np.imag(alpha - beta)
        p_x = _gaussian_interval_probability(x_lower, x_upper, mu_x)
        p_p = _gaussian_interval_probability(p_lower, p_upper, mu_p)
        joint = np.outer(p_x, p_p).reshape(-1)
        joint = np.maximum(joint, 0.0)
        total = float(joint.sum())
        if total <= 0.0:
            raise RuntimeError("Analytic IQ probability table contains a zero-sum row.")
        probabilities[input_index, :] = joint / total

    return probabilities, output_labels, x_edges, p_edges


def _target_metadata(
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    target_index: int,
) -> dict:
    """
    生成目标输入的元数据字典
    ========================

    用途
    ----
    将目标输入（需要认证随机性的那个输入）的索引和对应的相干态
    振幅信息打包成字典，用于结果报告中标记哪个输入被选为认证目标。

    参数
    ----
    labels : list[tuple[int, int]]
        联合输入的本地态索引对列表
    local_alphas : list[complex]
        本地相干态复振幅列表
    target_index : int
        目标输入在 labels 中的索引

    返回
    ----
    metadata : dict
        包含以下键的字典：
        - target_index: 目标输入的索引
        - target_input: 本地态索引对 (x, y)
        - target_alphas: 两个本地态的复振幅信息（序列化为字典）
    """
    x_index, y_index = labels[target_index]
    return {
        "target_index": int(target_index),
        "target_input": (int(x_index), int(y_index)),
        "target_alphas": [
            _serialize_complex(local_alphas[x_index]),
            _serialize_complex(local_alphas[y_index]),
        ],
    }


def certify_target_inputs(
    input_states: list[np.ndarray],
    probabilities: np.ndarray,
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    target_indices: list[int] | None = None,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> tuple[dict, list[dict]]:
    """
    对目标输入执行 SDP 认证，求解最小熵下界
    =========================================

    物理原理
    --------
    SDP（半正定规划）认证是 QRNG 协议的核心步骤。对每个目标输入 t，
    求解如下优化问题：

        p_guess* = max Σ_y Tr[F_y · ρ_t]

        约束条件：
        - {F_y} 构成 POVM（正定算符值测量），即 F_y ≥ 0, Σ_y F_y = I
        - 对所有输入 x 和输出 y 满足 Tr[F_y · ρ_x] = P(y|x)
          （测量结果与观察到的概率分布一致）

    最小熵 H_min = -log2(p_guess*) 给出了可提取随机性的量。

    求解器容错策略：先尝试 preferred_solver，若失败则回退到 CLARABEL。
    如果所有求解器都失败，返回 status="solver_failed" 的结果。

    参数
    ----
    input_states : list[np.ndarray]
        所有联合输入态的密度矩阵列表
    probabilities : np.ndarray
        概率矩阵，形状 (n_inputs, n_outputs)
    labels : list[tuple[int, int]]
        联合输入的本地态索引对列表
    local_alphas : list[complex]
        本地相干态复振幅列表
    target_indices : list[int] | None
        需要认证的目标输入索引列表。None 表示认证所有输入。
    preferred_solver : str | None
        优先使用的 SDP 求解器名称。None 表示使用默认求解器。
    solver_options : dict[str, dict] | None
        传递给求解器的额外选项
    verbose : bool
        求解器是否输出详细信息。默认 False。

    返回
    ----
    best : dict
        最优认证结果（H_min 最大的目标），包含：
        - solver: 使用的求解器
        - status: 求解状态
        - p_guess: 猜测概率上界
        - H_min: 最小熵下界（比特）
        - target_index/input/alphas: 目标输入信息
        - raw_H_min/raw_p_guess: 未认证的原始估计
    scan : list[dict]
        所有目标的认证结果列表，格式与 best 相同
    """
    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    indices = list(range(len(input_states))) if target_indices is None else list(target_indices)
    reusable_problem = SingleDeviceGuessingProblem(input_states, probabilities)

    best: dict | None = None
    scan: list[dict] = []
    for target_input in indices:
        solver_attempts = [preferred_solver] if preferred_solver is not None else [None, "CLARABEL"]
        solver_errors: list[str] = []
        current: dict | None = None
        for solver_name in solver_attempts:
            try:
                current = reusable_problem.solve(
                    target_input=target_input,
                    preferred_solver=solver_name,
                    solver_options=solver_options,
                    verbose=verbose,
                )
                break
            except RuntimeError as exc:
                label = "default" if solver_name is None else str(solver_name)
                solver_errors.append(f"{label}: {exc}")

        if current is None:
            current = {
                "solver": preferred_solver if preferred_solver is not None else "default+CLARABEL",
                "status": "solver_failed",
                "p_guess": None,
                "H_min": None,
                "solver_errors": solver_errors,
            }

        current.update(_target_metadata(labels, local_alphas, target_input))
        current.update(
            {
                "raw_H_min": float(raw_h[target_input]),
                "raw_p_guess": float(np.max(probabilities[target_input])),
            }
        )
        entry = dict(current)
        scan.append(entry)
        if best is None or (current["H_min"] or -np.inf) > (best["H_min"] or -np.inf):
            best = dict(entry)

    assert best is not None
    return best, scan


def _candidate_summary(
    candidate_index: int,
    x_candidate: AxisBoundsCandidate,
    p_candidate: AxisBoundsCandidate,
    probabilities: np.ndarray,
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
) -> dict:
    """
    生成分箱候选的摘要信息（含原始熵估计）
    ========================================

    用途
    ----
    对一个具体的 (X 分箱, P 分箱) 候选配置，计算其原始最小熵
    （未经 SDP 认证的理论上界）并汇总所有配置信息。

    原始最小熵的计算：
        raw_H_min = -log2(max_y P(y|x*))
    其中 x* 是使 raw_H_min 最大的输入（即最随机的输入）。

    这个函数用于分箱搜索的粗筛阶段：先快速计算 raw_H_min，
    然后只对 raw_H_min 最高的几个候选执行耗时的 SDP 认证。

    参数
    ----
    candidate_index : int
        候选的序号（用于标识）
    x_candidate : AxisBoundsCandidate
        X 轴的分箱配置
    p_candidate : AxisBoundsCandidate
        P 轴的分箱配置
    probabilities : np.ndarray
        该候选配置下的概率矩阵
    labels : list[tuple[int, int]]
        联合输入标签列表
    local_alphas : list[complex]
        本地相干态复振幅列表

    返回
    ----
    summary : dict
        包含分箱配置参数、输出维度和原始熵信息的字典
    """
    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    raw_best_index = int(np.argmax(raw_h))
    summary = {
        "candidate_index": int(candidate_index),
        "num_x_bins": int(x_candidate.num_bins),
        "num_p_bins": int(p_candidate.num_bins),
        "num_outputs": int(probabilities.shape[1]),
        "x_bounds": list(x_candidate.bounds),
        "p_bounds": list(p_candidate.bounds),
        "x_range": float(x_candidate.finite_range),
        "p_range": float(p_candidate.finite_range),
        "x_gamma": float(x_candidate.gamma),
        "p_gamma": float(p_candidate.gamma),
        "raw_best_H_min": float(raw_h[raw_best_index]),
        "raw_best_target_index": raw_best_index,
    }
    summary.update(
        {
            "raw_best_target": labels[raw_best_index],
            "raw_best_target_alphas": [
                _serialize_complex(local_alphas[labels[raw_best_index][0]]),
                _serialize_complex(local_alphas[labels[raw_best_index][1]]),
            ],
        }
    )
    return summary


def _alphabet_summary(
    candidate_index: int,
    candidate: AlphabetCandidateSpec,
    local_rank: int,
    local_operator_span: int,
    joint_dim: int,
    joint_operator_span: int,
) -> dict:
    """
    生成字母表候选的摘要信息（含空间维度和算符秩）
    ==============================================

    用途
    ----
    对一个字母表候选，汇总其空间维度、Gram 秩、算符空间秩等
    关键参数，用于字母表搜索结果的排序和比较。

    关键指标说明：
    - local_rank: Gram 矩阵的秩，等于相干态张成的子空间维度
    - local_operator_span: 本地态集合张成的算符空间秩，
      反映态在 Hilbert-Schmidt 空间中的线性独立性
    - local_span_ratio: 算符空间秩与总维度的比值，
      比值越高说明测量结果受量子约束越强
    - joint_dim: 联合 Hilbert 空间维度 = local_rank²

    参数
    ----
    candidate_index : int
        字母表候选的序号
    candidate : AlphabetCandidateSpec
        字母表候选规格对象
    local_rank : int
        本地 Gram 支撑空间维度
    local_operator_span : int
        本地算符空间秩
    joint_dim : int
        联合空间维度
    joint_operator_span : int
        联合算符空间秩

    返回
    ----
    summary : dict
        包含字母表参数和空间维度信息的字典
    """
    local_space_dim = int(local_rank**2)
    local_span_ratio = 0.0 if local_space_dim == 0 else float(local_operator_span / local_space_dim)
    return {
        "alphabet_candidate_index": int(candidate_index),
        "radius_values": list(candidate.radius_values),
        "phase_values": list(candidate.phase_values),
        "num_local_states": int(len(candidate.alpha_values)),
        "alpha_values": serialize_complex_list(list(candidate.alpha_values)),
        "local_rank": int(local_rank),
        "local_operator_span_rank": int(local_operator_span),
        "local_operator_space_dim": int(local_space_dim),
        "local_span_ratio": float(local_span_ratio),
        "joint_dim": int(joint_dim),
        "operator_span_rank": int(joint_operator_span),
        "operator_space_dim": int(joint_dim**2),
    }


def _raw_partition_candidates(
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    num_x_bins_values: list[int] | None,
    num_p_bins_values: list[int] | None,
    quadrature_ranges: list[float] | None,
    gamma_values: list[float] | None,
    store_probabilities: bool = True,
) -> tuple[list[dict], list[AxisBoundsCandidate], list[AxisBoundsCandidate]]:
    """
    枚举所有分箱候选并按原始熵排序（粗筛阶段）
    =============================================

    用途
    ----
    对给定的字母表，枚举 X 轴和 P 轴分箱参数的所有组合，
    为每个组合计算解析 IQ 概率和原始最小熵，然后按
    raw_best_H_min 降序排列。

    这是搜索流程中的粗筛步骤：快速评估大量分箱方案，
    只将 raw_H_min 最高的几个候选送入 SDP 认证。

    工作流程：
    1. 分别生成 X 轴和 P 轴的分箱候选列表
    2. 对每个 (X候选, P候选) 对：
       a. 调用 analytic_iq_probabilities 计算概率矩阵
       b. 调用 _candidate_summary 生成摘要（含 raw_H_min）
       c. 可选保留概率矩阵以供后续 SDP 使用
    3. 按 raw_best_H_min 降序排列所有候选

    参数
    ----
    labels : list[tuple[int, int]]
        联合输入的本地态索引对列表
    local_alphas : list[complex]
        本地相干态复振幅列表
    num_x_bins_values : list[int] | None
        X 轴候选分箱数量列表。None 使用 [2]。
    num_p_bins_values : list[int] | None
        P 轴候选分箱数量列表。None 使用 [2]。
    quadrature_ranges : list[float] | None
        候选有限范围列表。None 使用 DEFAULT_QUADRATURE_RANGES。
    gamma_values : list[float] | None
        候选幂律参数列表。None 使用 DEFAULT_GAMMA_VALUES。
    store_probabilities : bool
        是否在结果中保留概率矩阵（用于后续 SDP 认证）。
        设为 False 可节省内存但无法做进一步认证。默认 True。

    返回
    ----
    ranked_candidates : list[dict]
        按 raw_best_H_min 降序排列的候选摘要列表。
        每个元素包含分箱参数和原始熵信息。
    x_candidates : list[AxisBoundsCandidate]
        X 轴分箱候选列表
    p_candidates : list[AxisBoundsCandidate]
        P 轴分箱候选列表

    异常
    ----
    RuntimeError
        当没有生成任何候选时抛出。
    """
    x_candidates = generate_axis_bound_candidates(
        [2] if num_x_bins_values is None else list(num_x_bins_values),
        DEFAULT_QUADRATURE_RANGES if quadrature_ranges is None else list(quadrature_ranges),
        DEFAULT_GAMMA_VALUES if gamma_values is None else list(gamma_values),
    )
    p_candidates = generate_axis_bound_candidates(
        [2] if num_p_bins_values is None else list(num_p_bins_values),
        DEFAULT_QUADRATURE_RANGES if quadrature_ranges is None else list(quadrature_ranges),
        DEFAULT_GAMMA_VALUES if gamma_values is None else list(gamma_values),
    )

    raw_candidates: list[dict] = []
    for candidate_index, (x_candidate, p_candidate) in enumerate(
        (pair for pair in ((x_candidate, p_candidate) for x_candidate in x_candidates for p_candidate in p_candidates))
    ):
        probabilities, _, _, _ = analytic_iq_probabilities(
            labels,
            local_alphas,
            np.array(x_candidate.bounds, dtype=float),
            np.array(p_candidate.bounds, dtype=float),
        )
        summary = _candidate_summary(
            candidate_index,
            x_candidate,
            p_candidate,
            probabilities,
            labels,
            local_alphas,
        )
        if store_probabilities:
            summary["probabilities"] = probabilities
        raw_candidates.append(summary)

    if len(raw_candidates) == 0:
        raise RuntimeError("No IQ partition candidates were generated.")

    ranked_candidates = sorted(raw_candidates, key=lambda item: item["raw_best_H_min"], reverse=True)
    return ranked_candidates, x_candidates, p_candidates


def run_route6(
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    boundary_gamma: float = 1.0,
    max_inputs_to_certify: int | None = 1,
    gram_tol: float = 1e-10,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Route 6 完整单次运行：构造态 + 解析概率 + SDP 认证
    ==================================================

    物理原理
    --------
    这是 Route 6 的主入口函数，完成一次完整的 QRNG 认证流程：

    1. **态构造阶段**：
       调用 exact_joint_inputs_from_alphas，使用 Gram 矩阵方法
       在支撑空间中精确表示所有本地相干态，并构造联合输入态。
       无需 Fock 截断，避免了传统方法的截断误差。

    2. **概率计算阶段**：
       调用 analytic_iq_probabilities，利用相干态高斯分布的
       解析公式（erf 函数）计算每个联合输入在 IQ 测量各输出
       矩形区域上的精确概率。

    3. **目标选择阶段**：
       计算所有输入的原始最小熵 raw_H_min，按降序排列。
       选取 raw_H_min 最大的 max_inputs_to_certify 个输入作为
       SDP 认证目标。

    4. **SDP 认证阶段**：
       调用 certify_target_inputs，对每个目标输入求解半正定规划，
       获得 p_guess 上界和 H_min 下界。

    默认配置使用 2×2 分箱（X 和 P 各 2 个区间，即正负二分法），
    3.0 倍标准差范围和均匀间距（gamma=1.0）。

    参数
    ----
    alpha_values : list[complex] | None
        直接指定的复振幅列表。默认 None（使用网格构造）。
    radius_values : list[float] | None
        半径候选值。默认 None 使用 DEFAULT_RADIUS_VALUES。
    phase_values : list[float] | None
        相位候选值。默认 None 使用 DEFAULT_PHASE_VALUES。
    num_x_bins : int
        X 轴分箱数量。默认 2。
    num_p_bins : int
        P 轴分箱数量。默认 2。
    x_bounds : np.ndarray | None
        自定义 X 轴边界。None 则通过 power_spaced_bounds 生成。
    p_bounds : np.ndarray | None
        自定义 P 轴边界。None 则通过 power_spaced_bounds 生成。
    quadrature_range : float
        有限边界的范围（标准差倍数）。默认 3.0（~99.7% 覆盖）。
    boundary_gamma : float
        幂律间距参数。默认 1.0（均匀）。
    max_inputs_to_certify : int | None
        最多认证多少个目标输入。None 表示全部。默认 1。
    gram_tol : float
        Gram 矩阵特征值截断阈值。默认 1e-10。
    preferred_solver : str | None
        优先 SDP 求解器。默认 None。
    solver_options : dict[str, dict] | None
        求解器额外选项。默认 None。
    verbose : bool
        求解器详细输出。默认 False。

    返回
    ----
    result : dict
        完整的认证结果，包含：
        - route/state_representation/probability_engine: 方法标识
        - H_min/p_guess: 认证的最小熵和猜测概率
        - target_index/input/alphas: 最优目标信息
        - local_alphas/local_rank/joint_dim: 态空间信息
        - x_bounds/p_bounds/num_x_bins/num_p_bins: 分箱信息
        - target_scan: 所有认证目标的详细结果
        以及更多配置和中间结果字段。
    """
    (
        joint_states,
        labels,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
        local_gram,
    ) = exact_joint_inputs_from_alphas(
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
        tol=gram_tol,
    )
    resolved_x_bounds = (
        power_spaced_bounds(num_x_bins, quadrature_range, gamma=boundary_gamma)
        if x_bounds is None
        else np.asarray(x_bounds, dtype=float)
    )
    resolved_p_bounds = (
        power_spaced_bounds(num_p_bins, quadrature_range, gamma=boundary_gamma)
        if p_bounds is None
        else np.asarray(p_bounds, dtype=float)
    )

    probabilities, output_labels, x_bounds_out, p_bounds_out = analytic_iq_probabilities(
        labels,
        local_alphas,
        resolved_x_bounds,
        resolved_p_bounds,
    )

    raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
    candidate_order = list(np.argsort(-raw_h))
    if max_inputs_to_certify is not None:
        candidate_order = candidate_order[:max_inputs_to_certify]

    best, target_scan = certify_target_inputs(
        joint_states,
        probabilities,
        labels,
        local_alphas,
        target_indices=candidate_order,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
    )
    raw_best_index = int(np.argmax(raw_h))
    best.update(
        {
            "route": "route6_cv_gram_analytic_iq",
            "probability_engine": "analytic_gaussian_rectangles",
            "state_representation": "exact_coherent_gram_support",
            "num_local_states": len(local_alphas),
            "num_inputs": len(joint_states),
            "num_outputs": int(probabilities.shape[1]),
            "num_x_bins": int(num_x_bins),
            "num_p_bins": int(num_p_bins),
            "output_labels": output_labels,
            "local_alphas": serialize_complex_list(local_alphas),
            "local_rank": int(local_rank),
            "local_operator_span_rank": int(local_operator_span),
            "local_operator_space_dim": int(local_rank**2),
            "joint_dim": int(joint_dim),
            "operator_span_rank": int(operator_span_rank(joint_states)),
            "operator_space_dim": int(joint_dim**2),
            "x_bounds": x_bounds_out.tolist(),
            "p_bounds": p_bounds_out.tolist(),
            "boundary_gamma": float(boundary_gamma),
            "quadrature_range": float(quadrature_range),
            "gram_tol": float(gram_tol),
            "local_gram_rank": int(np.linalg.matrix_rank(local_gram, tol=gram_tol)),
            "raw_best_target_index": raw_best_index,
            "raw_best_target": labels[raw_best_index],
            "raw_best_target_alphas": [
                _serialize_complex(local_alphas[labels[raw_best_index][0]]),
                _serialize_complex(local_alphas[labels[raw_best_index][1]]),
            ],
            "raw_best_H_min": float(raw_h[raw_best_index]),
            "certified_best_target_index": int(best["target_index"]),
            "certified_best_target": best["target_input"],
            "certified_best_target_alphas": best["target_alphas"],
            "num_inputs_certified": len(target_scan),
            "target_scan": target_scan,
        }
    )
    return best


def search_route6_iq_partitions(
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins_values: list[int] | None = None,
    num_p_bins_values: list[int] | None = None,
    quadrature_ranges: list[float] | None = None,
    gamma_values: list[float] | None = None,
    certify_top_k: int = 3,
    max_inputs_to_certify: int | None = 1,
    gram_tol: float = 1e-10,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """
    固定字母表，搜索最优分箱方案（粗筛 + SDP 精选）
    ================================================

    用途
    ----
    在字母表（相干态集合）固定的前提下，搜索使认证 H_min 最大化的
    IQ 分箱方案。采用"粗筛 + 精选"两阶段策略：

    阶段一（粗筛）：枚举所有 (num_x_bins, num_p_bins, range, gamma)
    的组合，对每个组合快速计算解析概率和 raw_H_min，按 raw_H_min
    降序排列。这一步无需 SDP 求解，速度极快。

    阶段二（精选）：取粗筛排名前 certify_top_k 个候选，
    对每个候选调用 certify_target_inputs 执行 SDP 认证，
    获得严格的 H_min 下界。

    这种两阶段策略避免了为每个分箱方案都执行 SDP，大幅减少
    计算时间，同时确保最优方案被找到。

    参数
    ----
    alpha_values : list[complex] | None
        直接指定的复振幅列表。默认 None。
    radius_values : list[float] | None
        半径候选值。默认 None 使用 DEFAULT_RADIUS_VALUES。
    phase_values : list[float] | None
        相位候选值。默认 None 使用 DEFAULT_PHASE_VALUES。
    num_x_bins_values : list[int] | None
        X 轴候选分箱数量列表。None 默认 [2]。
    num_p_bins_values : list[int] | None
        P 轴候选分箱数量列表。None 默认 [2]。
    quadrature_ranges : list[float] | None
        候选有限范围列表。None 使用 DEFAULT_QUADRATURE_RANGES。
    gamma_values : list[float] | None
        候选幂律参数列表。None 使用 DEFAULT_GAMMA_VALUES。
    certify_top_k : int
        从粗筛结果中选取前 k 个候选进行 SDP 认证。默认 3。
    max_inputs_to_certify : int | None
        每个分箱候选中最多认证的目标输入数。默认 1。
    gram_tol : float
        Gram 矩阵特征值截断阈值。默认 1e-10。
    preferred_solver : str | None
        优先 SDP 求解器。默认 None。
    solver_options : dict[str, dict] | None
        求解器额外选项。默认 None。
    verbose : bool
        求解器详细输出。默认 False。

    返回
    ----
    result : dict
        搜索结果，包含：
        - H_min/p_guess: 最优分箱方案的认证结果
        - raw_partition_ranking: 粗筛排名（前 10 个）
        - certified_partition_results: 精选结果
        - num_partition_candidates: 总候选数
        - certify_top_k: 实际认证数量
        - selection_strategy: 搜索策略描述
        以及态空间、分箱参数等完整信息。
    """
    (
        joint_states,
        labels,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
        local_gram,
    ) = exact_joint_inputs_from_alphas(
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
        tol=gram_tol,
    )

    ranked_candidates, x_candidates, p_candidates = _raw_partition_candidates(
        labels,
        local_alphas,
        num_x_bins_values=num_x_bins_values,
        num_p_bins_values=num_p_bins_values,
        quadrature_ranges=quadrature_ranges,
        gamma_values=gamma_values,
        store_probabilities=True,
    )
    certify_count = min(max(0, certify_top_k), len(ranked_candidates))
    certified_candidates: list[dict] = []
    for candidate in ranked_candidates[:certify_count]:
        probabilities = candidate["probabilities"]
        raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
        target_indices = list(np.argsort(-raw_h))
        if max_inputs_to_certify is not None:
            target_indices = target_indices[:max_inputs_to_certify]

        certified, target_scan = certify_target_inputs(
            joint_states,
            probabilities,
            labels,
            local_alphas,
            target_indices=target_indices,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        certified_entry = {key: value for key, value in candidate.items() if key != "probabilities"}
        certified_entry.update(certified)
        certified_entry["num_inputs_certified"] = len(target_scan)
        certified_entry["target_scan"] = target_scan
        certified_candidates.append(certified_entry)

    if len(certified_candidates) > 0:
        best = dict(
            max(
                certified_candidates,
                key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
            )
        )
    else:
        best = {key: value for key, value in ranked_candidates[0].items() if key != "probabilities"}
        best.update(
            {
                "solver": None,
                "status": "not_certified",
                "p_guess": None,
                "H_min": None,
                "target_index": int(best["raw_best_target_index"]),
                "target_input": best["raw_best_target"],
                "target_alphas": best["raw_best_target_alphas"],
                "num_inputs_certified": 0,
                "target_scan": [],
            }
        )

    best.update(
        {
            "route": "route6_cv_gram_analytic_iq_search",
            "probability_engine": "analytic_gaussian_rectangles",
            "state_representation": "exact_coherent_gram_support",
            "num_local_states": len(local_alphas),
            "num_inputs": len(joint_states),
            "local_alphas": serialize_complex_list(local_alphas),
            "local_rank": int(local_rank),
            "local_operator_span_rank": int(local_operator_span),
            "local_operator_space_dim": int(local_rank**2),
            "joint_dim": int(joint_dim),
            "operator_span_rank": int(operator_span_rank(joint_states)),
            "operator_space_dim": int(joint_dim**2),
            "gram_tol": float(gram_tol),
            "local_gram_rank": int(np.linalg.matrix_rank(local_gram, tol=gram_tol)),
            "num_x_candidates": len(x_candidates),
            "num_p_candidates": len(p_candidates),
            "num_partition_candidates": len(ranked_candidates),
            "certify_top_k": int(certify_count),
            "selection_strategy": "rank partitions by raw-best target entropy, then certify top candidates",
            "raw_partition_ranking": [
                {key: value for key, value in candidate.items() if key != "probabilities"}
                for candidate in ranked_candidates[: min(10, len(ranked_candidates))]
            ],
            "certified_partition_results": certified_candidates,
        }
    )
    return best


def search_route6_alphabets(
    radius_values: list[float],
    phase_values: list[float],
    num_radii_values: list[int],
    num_phase_values: list[int],
    num_x_bins_values: list[int] | None = None,
    num_p_bins_values: list[int] | None = None,
    quadrature_ranges: list[float] | None = None,
    gamma_values: list[float] | None = None,
    require_vacuum: bool = True,
    max_local_states: int | None = None,
    certify_top_k_per_alphabet: int = 1,
    max_inputs_to_certify: int | None = 1,
    gram_tol: float = 1e-10,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """
    联合搜索最优字母表和分箱方案
    =============================

    用途
    ----
    这是 Route 6 最高层级的搜索接口。同时搜索最优的相干态字母表
    （即源可制备的态集合）和最优的 IQ 分箱方案。

    搜索策略采用嵌套三层枚举：
    1. **字母表层**：枚举所有 (radius_subset, phase_subset) 组合，
       每个组合对应一个 AlphabetCandidateSpec
    2. **分箱层**（对每个字母表）：调用 search_route6_iq_partitions
       搜索最优分箱方案（内含粗筛 + 精选两阶段）
    3. **目标层**（对每个分箱方案）：SDP 认证最优目标输入

    最终按认证 H_min 对所有字母表排名，返回最优方案。

    物理意义：不同的字母表代表源不同的制备能力。更大的字母表
    可以提供更多的随机性来源，但也增加了 SDP 的维度和计算复杂度。
    此函数自动在计算复杂度和随机性产量之间寻找最优平衡。

    参数
    ----
    radius_values : list[float]
        候选半径值的完整列表
    phase_values : list[float]
        候选相位值的完整列表
    num_radii_values : list[int]
        需要枚举的半径数量列表
    num_phase_values : list[int]
        需要枚举的相位数量列表
    num_x_bins_values : list[int] | None
        X 轴候选分箱数量列表。None 默认 [2]。
    num_p_bins_values : list[int] | None
        P 轴候选分箱数量列表。None 默认 [2]。
    quadrature_ranges : list[float] | None
        候选有限范围列表。None 使用默认值。
    gamma_values : list[float] | None
        候选幂律参数列表。None 使用默认值。
    require_vacuum : bool
        是否要求字母表包含真空态。默认 True。
    max_local_states : int | None
        每个字母表的最大态数量限制。None 不限制。
    certify_top_k_per_alphabet : int
        每个字母表内最多认证的分箱候选数。默认 1。
    max_inputs_to_certify : int | None
        每个分箱方案内最多认证的目标输入数。默认 1。
    gram_tol : float
        Gram 矩阵特征值截断阈值。默认 1e-10。
    preferred_solver : str | None
        优先 SDP 求解器。默认 None。
    solver_options : dict[str, dict] | None
        求解器额外选项。默认 None。
    verbose : bool
        求解器详细输出。默认 False。

    返回
    ----
    result : dict
        全局搜索结果，包含：
        - H_min/p_guess: 最优方案的认证结果
        - alphabet_ranking: 所有字母表的排名（前 20 个）
        - num_alphabet_candidates: 总字母表候选数
        - selection_strategy: 搜索策略描述
        以及最优方案的完整配置信息。

    异常
    ----
    RuntimeError
        当没有生成任何字母表候选时抛出。
    """
    candidates = generate_alphabet_candidates_from_grid(
        radius_values=radius_values,
        phase_values=phase_values,
        num_radii_values=num_radii_values,
        num_phase_values=num_phase_values,
        require_vacuum=require_vacuum,
        max_local_states=max_local_states,
    )
    if len(candidates) == 0:
        raise RuntimeError("No alphabet candidates were generated.")

    alphabet_results: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        (
            joint_states,
            labels,
            local_alphas,
            local_rank,
            joint_dim,
            local_operator_span,
            local_gram,
        ) = exact_joint_inputs_from_alphas(alpha_values=list(candidate.alpha_values), tol=gram_tol)
        partition_result = search_route6_iq_partitions(
            alpha_values=list(candidate.alpha_values),
            num_x_bins_values=num_x_bins_values,
            num_p_bins_values=num_p_bins_values,
            quadrature_ranges=quadrature_ranges,
            gamma_values=gamma_values,
            certify_top_k=certify_top_k_per_alphabet,
            max_inputs_to_certify=max_inputs_to_certify,
            gram_tol=gram_tol,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        summary = _alphabet_summary(
            candidate_index,
            candidate,
            local_rank,
            local_operator_span,
            joint_dim,
            operator_span_rank(joint_states),
        )
        summary.update(
            {
                "gram_tol": float(gram_tol),
                "local_gram_rank": int(np.linalg.matrix_rank(local_gram, tol=gram_tol)),
                "raw_best_H_min": float(partition_result["raw_partition_ranking"][0]["raw_best_H_min"]),
                "H_min": partition_result["H_min"],
                "p_guess": partition_result["p_guess"],
                "solver": partition_result["solver"],
                "status": partition_result["status"],
                "best_partition": {
                    "candidate_index": partition_result["candidate_index"],
                    "num_x_bins": partition_result["num_x_bins"],
                    "num_p_bins": partition_result["num_p_bins"],
                    "x_bounds": partition_result["x_bounds"],
                    "p_bounds": partition_result["p_bounds"],
                    "x_range": partition_result["x_range"],
                    "p_range": partition_result["p_range"],
                    "x_gamma": partition_result["x_gamma"],
                    "p_gamma": partition_result["p_gamma"],
                },
                "target_index": partition_result["target_index"],
                "target_input": partition_result["target_input"],
                "target_alphas": partition_result["target_alphas"],
            }
        )
        alphabet_results.append(summary)

    best = dict(
        max(
            alphabet_results,
            key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
        )
    )
    best.update(
        {
            "route": "route6_cv_gram_analytic_alphabet_search",
            "probability_engine": "analytic_gaussian_rectangles",
            "state_representation": "exact_coherent_gram_support",
            "num_alphabet_candidates": len(candidates),
            "alphabet_ranking": sorted(
                alphabet_results,
                key=lambda item: (item["H_min"] if item["H_min"] is not None else -np.inf),
                reverse=True,
            )[: min(20, len(alphabet_results))],
        }
    )
    return best


def search_route6_fixed_partition_alphabets(
    radius_values: list[float],
    phase_values: list[float],
    num_radii_values: list[int],
    num_phase_values: list[int],
    num_x_bins: int = 6,
    num_p_bins: int = 2,
    quadrature_range: float = 1.5,
    boundary_gamma: float = 1.5,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    require_vacuum: bool = False,
    max_local_states: int | None = None,
    min_local_states: int | None = None,
    max_inputs_to_certify: int | None = None,
    gram_tol: float = 1e-10,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """
    固定分箱方案，搜索最优字母表
    =============================

    用途
    ----
    与 search_route6_alphabets 不同，本函数使用**固定的分箱方案**
    （一组预定义的 X/P 边界），只枚举字母表候选。这适用于以下场景：

    1. 已通过先验知识或预实验确定了最优分箱参数
    2. 需要在固定测量设置下比较不同态集合的认证效果
    3. 计算资源有限，只需搜索字母表空间

    默认分箱配置：X 轴 6 箱 × P 轴 2 箱，范围 ±1.5，gamma=1.5。
    此配置对中等振幅相干态提供了较好的 X 方向分辨能力。

    对每个字母表候选：
    1. 构造联合输入态
    2. 用固定边界计算解析 IQ 概率
    3. 对最优目标执行 SDP 认证
    4. 额外计算 Gram 矩阵的非对角重叠统计量（max/mean/min）

    非对角重叠统计量的物理意义：
    - max_abs_offdiag_overlap: 最大态间重叠，反映态的可区分性
    - 重叠越小 → 态越可区分 → 通常 H_min 越高

    参数
    ----
    radius_values : list[float]
        候选半径值的完整列表
    phase_values : list[float]
        候选相位值的完整列表
    num_radii_values : list[int]
        需要枚举的半径数量列表
    num_phase_values : list[int]
        需要枚举的相位数量列表
    num_x_bins : int
        X 轴分箱数量。默认 6。
    num_p_bins : int
        P 轴分箱数量。默认 2。
    quadrature_range : float
        有限边界范围。默认 1.5。
    boundary_gamma : float
        幂律间距参数。默认 1.5。
    x_bounds : np.ndarray | None
        自定义 X 边界。None 通过 power_spaced_bounds 生成。
    p_bounds : np.ndarray | None
        自定义 P 边界。None 通过 power_spaced_bounds 生成。
    require_vacuum : bool
        是否要求字母表包含真空态。默认 False（允许纯非真空态字母表）。
    max_local_states : int | None
        字母表最大态数。None 不限制。
    min_local_states : int | None
        字母表最小态数。None 不限制。
    max_inputs_to_certify : int | None
        每个字母表最多认证的目标数。None 认证所有。
    gram_tol : float
        Gram 矩阵截断阈值。默认 1e-10。
    preferred_solver : str | None
        优先 SDP 求解器。默认 None。
    solver_options : dict[str, dict] | None
        求解器额外选项。默认 None。
    verbose : bool
        求解器详细输出。默认 False。

    返回
    ----
    result : dict
        搜索结果，包含：
        - H_min/p_guess: 最优字母表的认证结果
        - num_alphabet_candidates: 总候选数
        - alphabet_ranking: 所有字母表排名（前 20 个）
        - selection_strategy: "fixed partition, rank alphabets directly by formal H_min"
        - 每个 alphabet_ranking 条目包含：
          Gram 矩阵统计、认证结果、分箱配置等完整信息

    异常
    ----
    RuntimeError
        当没有生成任何字母表候选时抛出。
    """
    candidates = generate_alphabet_candidates_from_grid(
        radius_values=radius_values,
        phase_values=phase_values,
        num_radii_values=num_radii_values,
        num_phase_values=num_phase_values,
        require_vacuum=require_vacuum,
        max_local_states=max_local_states,
    )
    if min_local_states is not None:
        candidates = [candidate for candidate in candidates if len(candidate.alpha_values) >= int(min_local_states)]
    if len(candidates) == 0:
        raise RuntimeError("No alphabet candidates were generated for the fixed-partition search.")

    resolved_x_bounds = (
        power_spaced_bounds(num_x_bins, quadrature_range, gamma=boundary_gamma)
        if x_bounds is None
        else np.asarray(x_bounds, dtype=float)
    )
    resolved_p_bounds = (
        power_spaced_bounds(num_p_bins, quadrature_range, gamma=boundary_gamma)
        if p_bounds is None
        else np.asarray(p_bounds, dtype=float)
    )

    alphabet_results: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        (
            joint_states,
            labels,
            local_alphas,
            local_rank,
            joint_dim,
            local_operator_span,
            local_gram,
        ) = exact_joint_inputs_from_alphas(alpha_values=list(candidate.alpha_values), tol=gram_tol)
        probabilities, output_labels, x_bounds_out, p_bounds_out = analytic_iq_probabilities(
            labels,
            local_alphas,
            resolved_x_bounds,
            resolved_p_bounds,
        )
        raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
        target_indices = list(np.argsort(-raw_h))
        if max_inputs_to_certify is not None:
            target_indices = target_indices[:max_inputs_to_certify]

        certified, target_scan = certify_target_inputs(
            joint_states,
            probabilities,
            labels,
            local_alphas,
            target_indices=target_indices,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
        )

        gram_abs = np.abs(local_gram)
        offdiag_mask = ~np.eye(gram_abs.shape[0], dtype=bool)
        offdiag_values = gram_abs[offdiag_mask]
        raw_best_index = int(np.argmax(raw_h))

        summary = _alphabet_summary(
            candidate_index,
            candidate,
            local_rank,
            local_operator_span,
            joint_dim,
            operator_span_rank(joint_states),
        )
        summary.update(
            {
                "route": "route6_cv_gram_fixed_partition_alphabet_search",
                "probability_engine": "analytic_gaussian_rectangles",
                "state_representation": "exact_coherent_gram_support",
                "num_inputs": len(joint_states),
                "num_outputs": int(probabilities.shape[1]),
                "output_labels": output_labels,
                "x_bounds": x_bounds_out.tolist(),
                "p_bounds": p_bounds_out.tolist(),
                "num_x_bins": int(num_x_bins),
                "num_p_bins": int(num_p_bins),
                "quadrature_range": float(quadrature_range),
                "boundary_gamma": float(boundary_gamma),
                "gram_tol": float(gram_tol),
                "local_gram_rank": int(np.linalg.matrix_rank(local_gram, tol=gram_tol)),
                "max_abs_offdiag_overlap": 0.0 if offdiag_values.size == 0 else float(np.max(offdiag_values)),
                "mean_abs_offdiag_overlap": 0.0 if offdiag_values.size == 0 else float(np.mean(offdiag_values)),
                "min_abs_offdiag_overlap": 0.0 if offdiag_values.size == 0 else float(np.min(offdiag_values)),
                "raw_best_H_min": float(raw_h[raw_best_index]),
                "raw_best_target_index": raw_best_index,
                "raw_best_target": labels[raw_best_index],
                "raw_best_target_alphas": [
                    _serialize_complex(local_alphas[labels[raw_best_index][0]]),
                    _serialize_complex(local_alphas[labels[raw_best_index][1]]),
                ],
                "solver": certified["solver"],
                "status": certified["status"],
                "p_guess": certified["p_guess"],
                "H_min": certified["H_min"],
                "target_index": certified["target_index"],
                "target_input": certified["target_input"],
                "target_alphas": certified["target_alphas"],
                "num_inputs_certified": len(target_scan),
                "target_scan": target_scan,
            }
        )
        alphabet_results.append(summary)

    ranking = sorted(
        alphabet_results,
        key=lambda item: (item["H_min"] if item["H_min"] is not None else -np.inf),
        reverse=True,
    )
    best = dict(ranking[0])
    best.update(
        {
            "num_alphabet_candidates": len(candidates),
            "selection_strategy": "fixed partition, rank alphabets directly by formal H_min",
            "alphabet_ranking": ranking[: min(20, len(ranking))],
        }
    )
    return best
