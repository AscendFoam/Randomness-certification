from __future__ import annotations

from itertools import combinations
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from ..common import (
    SingleDeviceGuessingProblem,
    coherent_state,
    density_from_ket,
    kron,
    operator_span_rank,
    project_density_to_basis,
    support_basis,
)
from ..route3.cv_four_phase import dual_homodyne_probabilities


DEFAULT_RADIUS_VALUES = [0.0, 0.6, 1.2]
DEFAULT_PHASE_VALUES = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
DEFAULT_QUADRATURE_RANGES = [2.0, 3.0, 4.0]
DEFAULT_GAMMA_VALUES = [0.75, 1.0, 1.5]
DEFAULT_PROBABILITY_ENGINE = "trace_povm"
SUPPORTED_PROBABILITY_ENGINES = ("trace_povm", "analytic_gaussian_rectangles")


@dataclass(frozen=True)
class AxisBoundsCandidate:
    num_bins: int
    finite_range: float
    gamma: float
    bounds: tuple[float, ...]


@dataclass(frozen=True)
class AlphabetCandidateSpec:
    radius_values: tuple[float, ...]
    phase_values: tuple[float, ...]
    alpha_values: tuple[complex, ...]


def _canonicalize_scalar(value: float) -> float:
    """将微小的数值噪声归一化为精确的零值。
    
    用于清理JSON输出中的数值精度问题。当数值非常接近零时（小于1e-12），
    将其归零以避免输出中出现科学计数法表示的微小数值。
    
    执行逻辑：
    1. 检查数值是否有限且绝对值小于1e-12
    2. 如果是，返回0.0
    3. 否则返回原始值的浮点数表示
    
    参数：
        value: 待处理的标量值
    
    返回：
        归一化后的标量值，微小值被归零
    """
    if math.isfinite(value) and abs(value) < 1e-12:
        return 0.0
    return float(value)


def _serialize_complex(alpha: complex) -> dict:
    """将复数转换为JSON友好的字典格式。
    
    将相干态的复振幅参数转换为包含实部、虚部、模和相位的字典，
    便于JSON序列化和结果展示。
    
    执行逻辑：
    1. 提取复数的实部和虚部
    2. 计算复数的模（绝对值）
    3. 计算复数的相位角
    4. 将所有值封装为字典返回
    
    参数：
        alpha: 复数，通常表示相干态的振幅参数
    
    返回：
        包含real、imag、abs、phase四个字段的字典
    """
    return {
        "real": float(np.real(alpha)),
        "imag": float(np.imag(alpha)),
        "abs": float(abs(alpha)),
        "phase": float(np.angle(alpha)),
    }


def serialize_complex_list(alphas: list[complex]) -> list[dict]:
    """将复数列表转换为JSON友好的字典列表。
    
    批量处理复数列表，每个复数转换为包含实部、虚部、模和相位的字典。
    
    执行逻辑：
    1. 遍历输入的复数列表
    2. 对每个复数调用_serialize_complex进行转换
    3. 返回字典列表
    
    参数：
        alphas: 复数列表，通常表示多个相干态的振幅参数
    
    返回：
        字典列表，每个字典包含real、imag、abs、phase字段
    """
    return [_serialize_complex(alpha) for alpha in alphas]


def _deduplicate_alphas(alphas: list[complex], tol: float = 1e-12) -> list[complex]:
    """去除重复的相干态振幅，保持原始顺序。
    
    在相干态字母表生成过程中，不同的半径和相位组合可能产生相同的复振幅。
    该函数去除这些重复值，同时保持元素的原始出现顺序。
    
    执行逻辑：
    1. 初始化空列表用于存储唯一值
    2. 遍历输入的复数列表
    3. 对每个复数，检查是否与已存在的值在容差范围内相等
    4. 如果不重复，添加到唯一值列表
    5. 返回去重后的列表
    
    参数：
        alphas: 复数列表，可能包含重复值
        tol: 判断重复的容差阈值，默认1e-12
    
    返回：
        去重后的复数列表，保持原始顺序
    """
    unique: list[complex] = []
    for alpha in alphas:
        if any(abs(alpha - existing) <= tol for existing in unique):
            continue
        unique.append(alpha)
    return unique


def _unique_sorted_radii(radius_values: list[float], tol: float = 1e-12) -> list[float]:
    """对半径值去重并排序。
    
    处理相干态字母表的半径参数，去除重复值并按升序排列。
    
    执行逻辑：
    1. 将输入值转换为浮点数并排序
    2. 遍历排序后的值，去除在容差范围内的重复值
    3. 返回去重后的升序列表
    
    参数：
        radius_values: 半径值列表，可能包含重复值
        tol: 判断重复的容差阈值，默认1e-12
    
    返回：
        去重并排序后的半径值列表
    """
    unique: list[float] = []
    for radius in sorted(float(value) for value in radius_values):
        if any(abs(radius - existing) <= tol for existing in unique):
            continue
        unique.append(radius)
    return unique


def _normalize_phase(phase: float) -> float:
    """将相位归一化到[0, 2π)区间。
    
    确保相位值在标准区间内，便于后续的子集枚举和比较。
    
    执行逻辑：
    1. 对相位值取模2π，得到[0, 2π)范围内的值
    2. 如果结果非常接近2π，归零
    3. 返回归一化后的相位
    
    参数：
        phase: 原始相位值（弧度）
    
    返回：
        归一化到[0, 2π)区间的相位值
    """
    two_pi = 2.0 * math.pi
    value = float(phase) % two_pi
    if abs(value - two_pi) < 1e-12:
        value = 0.0
    return value


def _unique_sorted_phases(phase_values: list[float], tol: float = 1e-12) -> list[float]:
    """对相位值去重（模2π）并排序。
    
    处理相干态字母表的相位参数，考虑相位的周期性，去除模2π后的重复值并排序。
    
    执行逻辑：
    1. 对每个相位值进行归一化（模2π到[0, 2π)）
    2. 对归一化后的值排序
    3. 去除在容差范围内的重复值
    4. 返回去重并排序后的相位列表
    
    参数：
        phase_values: 相位值列表（弧度），可能包含重复值
        tol: 判断重复的容差阈值，默认1e-12
    
    返回：
        去重并排序后的相位值列表，所有值在[0, 2π)区间
    """
    unique: list[float] = []
    for phase in sorted(_normalize_phase(value) for value in phase_values):
        if any(abs(phase - existing) <= tol for existing in unique):
            continue
        unique.append(phase)
    return unique


def generalized_coherent_alphabet(
    cutoff: int,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
) -> tuple[list[np.ndarray], list[complex]]:
    """生成广义相干态字母表。
    
    构造一组相干态作为量子随机数生成的可信态集合。相干态由其在相空间中的
    复振幅参数α定义。可以指定具体的振幅值，或者通过半径和相位组合生成。
    
    执行逻辑：
    1. 如果提供了alpha_values，直接使用这些复振幅
    2. 否则，使用半径和相位参数生成振幅：α = radius × exp(i×phase)
    3. 对振幅值去重
    4. 为每个振幅构造相干态密度矩阵
    5. 返回态列表和对应的振幅列表
    
    参数：
        cutoff: Fock空间的截断维度
        alpha_values: 直接指定的复振幅列表，如果提供则忽略radius_values和phase_values
        radius_values: 半径值列表，用于生成振幅
        phase_values: 相位值列表（弧度），用于生成振幅
    
    返回：
        元组(states, alphas)：
        - states: 相干态密度矩阵列表
        - alphas: 对应的复振幅列表
    
    异常：
        ValueError: 当alpha_values为空列表，或radius_values/phase_values为空时抛出
    """
    if alpha_values is not None and len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")

    if alpha_values is None:
        radii = DEFAULT_RADIUS_VALUES if radius_values is None else list(radius_values)
        phases = DEFAULT_PHASE_VALUES if phase_values is None else list(phase_values)
        if len(radii) == 0 or len(phases) == 0:
            raise ValueError("radius_values and phase_values must be non-empty.")
        alpha_values = [radius * np.exp(1j * phase) for radius in radii for phase in phases]

    unique_alphas = _deduplicate_alphas([complex(alpha) for alpha in alpha_values])
    states = [density_from_ket(coherent_state(cutoff, alpha)) for alpha in unique_alphas]
    return states, unique_alphas


def _build_alpha_values(radius_values: tuple[float, ...], phase_values: tuple[float, ...]) -> tuple[complex, ...]:
    """从半径和相位规格构造相干态振幅。
    
    根据给定的半径和相位组合，生成所有可能的复振幅值，并去重。
    
    执行逻辑：
    1. 对每个半径和相位的组合，计算α = radius × exp(i×phase)
    2. 对生成的振幅列表去重
    3. 返回元组形式的振幅列表
    
    参数：
        radius_values: 半径值元组
        phase_values: 相位值元组（弧度）
    
    返回：
        去重后的复振幅元组
    """
    alpha_values = [radius * np.exp(1j * phase) for radius in radius_values for phase in phase_values]
    return tuple(_deduplicate_alphas([complex(alpha) for alpha in alpha_values]))


def generate_radius_subsets(
    radius_values: list[float],
    num_radii_values: list[int],
    require_vacuum: bool = True,
) -> list[tuple[float, ...]]:
    """系统地生成半径子集用于字母表搜索。
    
    从给定的半径池中生成所有可能的半径子集组合，用于系统性地搜索最优的
    相干态字母表配置。可以选择是否必须包含真空态（半径为0）。
    
    执行逻辑：
    1. 对输入半径去重并排序
    2. 分离零半径（真空态）和非零半径
    3. 对每个请求的半径数量：
       - 如果require_vacuum且存在真空态：
         * 数量为1时只返回真空态
         * 数量>1时，从非零半径中选择(数量-1)个与真空态组合
       - 否则：直接从所有半径中选择指定数量
    4. 使用组合数生成所有可能的子集
    5. 去重后返回子集列表
    
    参数：
        radius_values: 半径值池
        num_radii_values: 需要生成的半径数量列表
        require_vacuum: 是否要求子集必须包含真空态（半径≈0）
    
    返回：
        半径子集元组列表，每个元组代表一个半径组合
    
    异常：
        ValueError: 当radius_values为空时抛出
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
    """从偏移量生成近似均匀间隔的相位子集。
    
    给定相位列表，选择指定数量的相位，使其在圆周上近似均匀分布。
    通过偏移参数可以生成不同的均匀分布模式。
    
    执行逻辑：
    1. 计算总相位数
    2. 根据偏移量和间隔计算每个位置的索引
    3. 处理索引冲突（如果索引已被使用，向后移动）
    4. 返回排序后的相位子集
    
    参数：
        phases: 相位列表（已排序）
        count: 需要选择的相位数量
        offset: 起始偏移量
    
    返回：
        选中的相位值元组，按升序排列
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
    """系统地生成相位子集用于字母表搜索。
    
    从给定的相位池中生成所有可能的相位子集组合。优先选择在圆周上均匀分布的
    相位组合，这对于相干态字母表的对称性很重要。
    
    执行逻辑：
    1. 对相位去重（模2π）并排序
    2. 对每个请求的相位数量：
       - 如果数量等于总数，返回完整集合
       - 如果总数能被请求数量整除，生成均匀分布的子集（多个偏移版本）
       - 否则，尝试所有可能的偏移生成近似均匀分布
    3. 去重后返回子集列表
    
    参数：
        phase_values: 相位值池（弧度）
        num_phase_values: 需要生成的相位数量列表
    
    返回：
        相位子集元组列表，每个元组代表一个相位组合
    
    异常：
        ValueError: 当phase_values为空时抛出
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
    """从半径/相位池系统生成相干态字母表候选。
    
    组合所有可能的半径子集和相位子集，生成完整的相干态字母表候选列表。
    每个候选由一组半径值和相位值定义，最终生成对应的复振幅集合。
    
    执行逻辑：
    1. 生成所有半径子集组合
    2. 生成所有相位子集组合
    3. 对每个半径-相位组合：
       - 构建对应的复振幅集合
       - 检查是否超过最大态数限制
       - 去重后添加到候选列表
    4. 返回候选规格列表
    
    参数：
        radius_values: 半径值池
        phase_values: 相位值池（弧度）
        num_radii_values: 半径数量列表
        num_phase_values: 相位数量列表
        require_vacuum: 是否要求包含真空态
        max_local_states: 最大本地态数量限制，None表示无限制
    
    返回：
        AlphabetCandidateSpec对象列表，每个包含半径、相位和振幅信息
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


def reduced_joint_inputs_from_alphas(
    cutoff: int,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, list[complex], int, int, int]:
    """从相干态振幅生成投影到支持空间的联合输入态。
    
    构造双模联合输入态，用于量子随机数生成的协议。首先生成单模相干态，
    然后构造所有可能的张量积组合，最后投影到实际的支持空间以降低维度。
    
    执行逻辑：
    1. 生成广义相干态字母表（单模态）
    2. 从每个密度矩阵提取主特征矢量（纯态近似）
    3. 计算所有态张成空间的正交归一基
    4. 将所有单模态投影到这个基上
    5. 构造所有双模张量积组合：ρ_AB = ρ_A ⊗ ρ_B
    6. 计算联合基和算符张成秩
    7. 返回联合态、标签、基和维度信息
    
    参数：
        cutoff: Fock空间的截断维度
        alpha_values: 直接指定的复振幅列表
        radius_values: 半径值列表（用于生成振幅）
        phase_values: 相位值列表（用于生成振幅）
    
    返回：
        元组(joint_states, labels, joint_basis, local_alphas, local_rank, joint_dim, local_operator_span)：
        - joint_states: 联合输入态密度矩阵列表
        - labels: 每个联合态的标签列表，格式为(x_index, y_index)
        - joint_basis: 联合系统的正交归一基矩阵
        - local_alphas: 本地振幅列表
        - local_rank: 本地基的维度
        - joint_dim: 联合态的维度
        - local_operator_span: 本地算符张成秩
    """
    local_states, local_alphas = generalized_coherent_alphabet(
        cutoff,
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
    )
    local_kets = []
    for rho in local_states:
        values, vectors = np.linalg.eigh(rho)
        local_kets.append(vectors[:, int(np.argmax(values))])

    local_basis = support_basis(local_kets)
    reduced_local_states = [project_density_to_basis(rho, local_basis) for rho in local_states]

    joint_states: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for x, rho_a in enumerate(reduced_local_states):
        for y, rho_b in enumerate(reduced_local_states):
            joint_states.append(kron(rho_a, rho_b))
            labels.append((x, y))

    joint_basis = kron(local_basis, local_basis)
    local_operator_span = operator_span_rank(reduced_local_states)
    return (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_basis.shape[1],
        joint_states[0].shape[0],
        local_operator_span,
    )


def power_spaced_bounds(num_bins: int, finite_range: float, gamma: float = 1.0) -> np.ndarray:
    """生成物理约束的对称轴对齐分箱边界。
    
    构造IQ平面上正交相位测量的分箱边界。边界采用幂次间距分布，
    可以在中心区域提供更精细的分辨率。边界对称分布，两端延伸到无穷远。
    
    执行逻辑：
    1. 参数验证（num_bins≥2, finite_range>0, gamma>0）
    2. 特殊情况：num_bins=2时返回[-∞, 0, +∞]
    3. 一般情况：
       - 在[-1, 1]区间生成均匀分布的归一化坐标
       - 应用幂次变换：edge = sign(x) × |x|^γ × finite_range
       - 第一和最后一个边界设为±∞
       - 对微小值归零
    4. 返回边界数组
    
    参数：
        num_bins: 分箱数量（至少为2）
        finite_range: 有限区域的范围，控制分箱的尺度
        gamma: 幂次参数，γ>1时中心区域更精细，γ<1时边缘区域更精细
    
    返回：
        边界值数组，形状为(num_bins+1,)，首尾为±∞
    
    异常：
        ValueError: 当参数不满足约束条件时抛出
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
    """生成一个正交相位轴的候选边界族。
    
    为IQ分区搜索生成所有可能的边界配置组合。每个配置由分箱数、
    有限范围和幂次参数定义，对应不同的测量分辨率分布。
    
    执行逻辑：
    1. 遍历所有分箱数配置
    2. 对num_bins=2的特殊情况，使用固定配置
    3. 对其他分箱数，遍历所有范围和γ参数组合
    4. 对每个组合生成边界并封装为候选对象
    5. 去重后返回候选列表
    
    参数：
        num_bins_values: 分箱数量列表
        quadrature_ranges: 有限范围值列表
        gamma_values: 幂次参数列表
    
    返回：
        AxisBoundsCandidate对象列表，每个包含分箱数、范围、γ参数和边界
    
    异常：
        ValueError: 当num_bins_values为空时抛出
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
                candidates.append(
                    AxisBoundsCandidate(
                        num_bins=2,
                        finite_range=0.0,
                        gamma=1.0,
                        bounds=bounds,
                    )
                )
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
                        num_bins=num_bins,
                        finite_range=float(finite_range),
                        gamma=float(gamma),
                        bounds=bounds,
                    )
                )
    return candidates


def _resolve_probability_engine(probability_engine: str | None) -> str:
    """规范化 Route 5 的概率计算后端名称。

    Route 5 目前支持两类概率引擎：

    - ``trace_povm``：原有后端，在截断 Fock 空间中构造双 Homodyne POVM，
      再计算 ``Tr(\rho M)``。这是当前正式流程的默认实现。
    - ``analytic_gaussian_rectangles``：新增后端，直接把输入视为理想相干态，
      利用平衡分束器后的高斯输出分布，解析计算每个 X-P 矩形 bin 的概率。

    参数：
        probability_engine: 用户传入的后端名称。``None`` 表示使用默认后端。

    返回：
        规范化后的后端名称字符串。

    异常：
        ValueError: 当后端名称不在支持列表中时抛出。
    """
    engine = DEFAULT_PROBABILITY_ENGINE if probability_engine is None else str(probability_engine)
    if engine not in SUPPORTED_PROBABILITY_ENGINES:
        raise ValueError(
            f"Unsupported probability_engine={engine!r}. "
            f"Supported values are {SUPPORTED_PROBABILITY_ENGINES}."
        )
    return engine


def _gaussian_interval_probability(lower: np.ndarray, upper: np.ndarray, mean: float) -> np.ndarray:
    """解析计算方差为 1/2 的高斯分布在多个区间上的概率。

    对于相干态的 quadrature 测量，在本项目采用的约定下：

    - ``X = (a + a^†) / sqrt(2)``
    - ``P = (a - a^†) / (i sqrt(2))``

    单模相干态在任一 quadrature 上的边缘分布都是均值为 ``mean``、
    方差为 ``1/2`` 的高斯分布，因此区间概率可写成：

    ``P([l, u)) = 1/2 * [erf(u - mean) - erf(l - mean)]``.

    参数：
        lower: 各区间下界数组。
        upper: 各区间上界数组。
        mean: 该 quadrature 分布的均值。

    返回：
        与区间一一对应的概率数组。
    """
    return 0.5 * (erf(upper - mean) - erf(lower - mean))


def analytic_iq_probabilities(
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    x_bounds: np.ndarray,
    p_bounds: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    """用解析高斯积分计算 Route 5 的 IQ 离散概率表。

    这个后端直接使用 Route 5 的相干态字母表参数 ``alpha``，不再在
    Fock 截断空间中构造 quadrature POVM，而是使用 CV Bell / 双
    Homodyne 的理想解析模型：

    1. 输入为 ``|alpha_x> ⊗ |alpha_y>``；
    2. 经过 50:50 平衡分束器后，输出仍是相干态张量积：
       ``|gamma> ⊗ |delta>``，
       其中 ``gamma = (alpha_x + alpha_y)/sqrt(2)``，
       ``delta = (alpha_x - alpha_y)/sqrt(2)``；
    3. 对输出 c 模测量 X，对输出 d 模测量 P；
    4. 对应的一维均值分别是：
       ``mu_x = Re(alpha_x + alpha_y)``，
       ``mu_p = Im(alpha_x - alpha_y)``；
    5. 每个矩形 bin 的二维概率等于两条一维高斯区间概率之积。

    说明：
        该后端保留 Route 5 的输入字母表、分箱方式与 SDP 主流程，
        但概率层对应的是“理想无限维相干态 + 解析 Gaussian 积分”模型。
        它主要用于数值稳定性检查、与理论公式对照，默认不会替代旧后端。

    参数：
        labels: 联合输入标签列表，每个元素为 ``(x_index, y_index)``。
        local_alphas: 本地相干态复振幅列表。
        x_bounds: X 轴分箱边界，长度为 ``num_x_bins + 1``。
        p_bounds: P 轴分箱边界，长度为 ``num_p_bins + 1``。

    返回：
        元组 ``(probabilities, output_labels, x_edges, p_edges)``：
        - probabilities: 概率矩阵，形状为 ``(num_inputs, num_outputs)``；
        - output_labels: 输出标签 ``(x_bin_index, p_bin_index)``；
        - x_edges: 实际使用的 X 边界；
        - p_edges: 实际使用的 P 边界。

    异常：
        ValueError: 当边界数组格式非法时抛出。
        RuntimeError: 当某一行概率和不为正时抛出。
    """
    x_edges = np.asarray(x_bounds, dtype=float)
    p_edges = np.asarray(p_bounds, dtype=float)
    if x_edges.ndim != 1 or p_edges.ndim != 1:
        raise ValueError("x_bounds and p_bounds must be 1-D arrays.")
    if x_edges.size < 3 or p_edges.size < 3:
        raise ValueError("x_bounds and p_bounds must each define at least two bins.")

    num_x_bins = x_edges.size - 1
    num_p_bins = p_edges.size - 1
    output_labels = [(x_index, p_index) for x_index in range(num_x_bins) for p_index in range(num_p_bins)]
    probabilities = np.zeros((len(labels), num_x_bins * num_p_bins), dtype=float)

    x_lower = x_edges[:-1]
    x_upper = x_edges[1:]
    p_lower = p_edges[:-1]
    p_upper = p_edges[1:]

    for input_index, (x_index, y_index) in enumerate(labels):
        alpha = local_alphas[x_index]
        beta = local_alphas[y_index]
        mu_x = float(np.real(alpha + beta))
        mu_p = float(np.imag(alpha - beta))
        p_x = _gaussian_interval_probability(x_lower, x_upper, mu_x)
        p_p = _gaussian_interval_probability(p_lower, p_upper, mu_p)
        joint = np.outer(p_x, p_p).reshape(-1)
        joint = np.maximum(joint, 0.0)
        total = float(joint.sum())
        if total <= 0.0:
            raise RuntimeError("Analytic IQ probability table contains a zero-sum row.")
        probabilities[input_index, :] = joint / total

    return probabilities, output_labels, x_edges, p_edges


def route5_iq_probabilities(
    joint_states: list[np.ndarray],
    labels: list[tuple[int, int]],
    joint_basis: np.ndarray,
    local_alphas: list[complex],
    cutoff: int,
    num_x_bins: int,
    num_p_bins: int,
    x_bounds: np.ndarray,
    p_bounds: np.ndarray,
    quadrature_range: float,
    num_quadrature_nodes: int | None,
    probability_engine: str | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    """统一调度 Route 5 的概率计算后端。

    参数：
        joint_states: 联合输入态列表。旧后端需要它来计算 ``Tr(rho M)``。
        labels: 联合输入标签列表。
        joint_basis: 联合态支撑空间基。旧后端需要先把 POVM 投影到该基。
        local_alphas: 本地相干态复振幅列表。解析后端直接使用它构造概率。
        cutoff: Fock 截断维度。
        num_x_bins: X 轴分箱数。
        num_p_bins: P 轴分箱数。
        x_bounds: X 轴分箱边界。
        p_bounds: P 轴分箱边界。
        quadrature_range: quadrature 有限范围。旧后端在构造 POVM 时使用。
        num_quadrature_nodes: quadrature 数值积分节点数。仅旧后端使用。
        probability_engine: 概率后端名称。

    返回：
        与 ``dual_homodyne_probabilities`` / ``analytic_iq_probabilities`` 一致的
        四元组 ``(probabilities, output_labels, x_edges, p_edges)``。
    """
    engine = _resolve_probability_engine(probability_engine)
    if engine == "trace_povm":
        return dual_homodyne_probabilities(
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
    return analytic_iq_probabilities(
        labels,
        local_alphas,
        x_bounds=np.asarray(x_bounds, dtype=float),
        p_bounds=np.asarray(p_bounds, dtype=float),
    )


def _target_metadata(
    labels: list[tuple[int, int]],
    local_alphas: list[complex],
    target_index: int,
) -> dict:
    """生成目标输入态的元数据字典。
    
    为指定的目标输入态生成包含索引、标签和振幅信息的字典，
    用于结果记录和追踪。
    
    执行逻辑：
    1. 从标签列表获取目标态的双模索引
    2. 从本地振幅列表获取对应的两个振幅
    3. 序列化振幅信息
    4. 返回包含所有信息的字典
    
    参数：
        labels: 联合态标签列表，每个元素是(x_index, y_index)
        local_alphas: 本地振幅列表
        target_index: 目标态在联合态列表中的索引
    
    返回：
        包含target_index、target_input和target_alphas的字典
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
    """认证一个或多个目标输入态并返回最佳认证结果。
    
    对指定的目标输入态进行SDP认证，计算猜测概率和最小熵。
    遍历所有目标态，对每个态求解SDP问题，返回最佳结果和完整扫描记录。
    
    执行逻辑：
    1. 计算所有输入态的原始最小熵（基于最大概率）
    2. 确定要认证的目标态列表
    3. 创建可重用的SDP问题对象
    4. 对每个目标态：
       - 尝试多个求解器（首选和CLARABEL后备）
       - 求解SDP获取猜测概率
       - 计算最小熵H_min = -log2(p_guess)
       - 记录结果和元数据
    5. 选择最小熵最大的结果作为最佳结果
    6. 返回最佳结果和完整扫描列表
    
    参数：
        input_states: 输入态密度矩阵列表
        probabilities: 观测概率矩阵，形状为(num_inputs, num_outputs)
        labels: 输入态标签列表
        local_alphas: 本地振幅列表
        target_indices: 目标态索引列表，None表示认证所有态
        preferred_solver: 首选求解器名称
        solver_options: 求解器选项字典
        verbose: 是否输出详细求解信息
    
    返回：
        元组(best, scan)：
        - best: 最佳认证结果字典，包含H_min、p_guess、target等信息
        - scan: 所有目标态的认证结果列表
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
    """生成IQ分区候选的摘要信息。
    
    为一个IQ分区配置生成包含分箱参数、原始熵和最佳目标态信息的摘要字典。
    
    执行逻辑：
    1. 计算所有输入态的原始最小熵
    2. 找到原始熵最大的目标态索引
    3. 收集分箱参数（分箱数、边界、范围、γ参数）
    4. 收集最佳目标态的标签和振幅信息
    5. 返回完整摘要字典
    
    参数：
        candidate_index: 候选配置的索引
        x_candidate: X轴边界候选
        p_candidate: P轴边界候选
        probabilities: 观测概率矩阵
        labels: 输入态标签列表
        local_alphas: 本地振幅列表
    
    返回：
        包含候选配置和原始熵信息的摘要字典
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
    """生成字母表候选的紧凑结构摘要。
    
    为一个相干态字母表配置生成包含半径、相位、维度和张成秩信息的摘要字典。
    用于评估字母表的结构特性和优化潜力。
    
    执行逻辑：
    1. 计算本地算符空间维度和张成比例
    2. 收集字母表参数（半径、相位、振幅）
    3. 收集维度信息（本地秩、联合维度）
    4. 收集算符张成信息（本地和联合）
    5. 返回完整摘要字典
    
    参数：
        candidate_index: 字母表候选的索引
        candidate: 字母表候选规格对象
        local_rank: 本地基的维度
        local_operator_span: 本地算符张成秩
        joint_dim: 联合态的维度
        joint_operator_span: 联合算符张成秩
    
    返回：
        包含字母表结构和算符空间信息的摘要字典
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
    joint_states: list[np.ndarray],
    labels: list[tuple[int, int]],
    joint_basis: np.ndarray,
    local_alphas: list[complex],
    cutoff: int,
    num_x_bins_values: list[int] | None,
    num_p_bins_values: list[int] | None,
    quadrature_ranges: list[float] | None,
    gamma_values: list[float] | None,
    num_quadrature_nodes: int | None,
    probability_engine: str | None = None,
    store_probabilities: bool = True,
) -> tuple[list[dict], list[AxisBoundsCandidate], list[AxisBoundsCandidate]]:
    """枚举物理约束的IQ分区并按原始熵排序。
    
    生成所有可能的X-P正交相位分区配置，计算每种配置下的观测概率，
    并根据原始最小熵对候选配置排序。这是IQ分区搜索的核心步骤。
    
    执行逻辑：
    1. 生成X轴和P轴的边界候选列表
    2. 对每个X-P边界组合：
       - 计算双零差探测概率分布
       - 生成候选摘要（包含分箱参数和原始熵）
       - 可选地存储概率矩阵
    3. 按原始最佳最小熵降序排序
    4. 返回排序后的候选列表和边界候选列表
    
    参数：
        joint_states: 联合输入态列表
        labels: 输入态标签列表
        joint_basis: 联合基矩阵
        local_alphas: 本地振幅列表
        cutoff: Fock空间截断维度
        num_x_bins_values: X轴分箱数列表
        num_p_bins_values: P轴分箱数列表
        quadrature_ranges: 有限范围列表
        gamma_values: 幂次参数列表
        num_quadrature_nodes: 正交相位求积节点数，仅 ``trace_povm`` 后端使用
        probability_engine: 概率计算后端名称
        store_probabilities: 是否在结果中存储概率矩阵
    
    返回：
        元组(ranked_candidates, x_candidates, p_candidates)：
        - ranked_candidates: 按原始熵排序的候选摘要列表
        - x_candidates: X轴边界候选列表
        - p_candidates: P轴边界候选列表
    
    异常：
        RuntimeError: 当没有生成任何候选时抛出
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
        probabilities, _, _, _ = route5_iq_probabilities(
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            cutoff,
            num_x_bins=x_candidate.num_bins,
            num_p_bins=p_candidate.num_bins,
            x_bounds=np.array(x_candidate.bounds, dtype=float),
            p_bounds=np.array(p_candidate.bounds, dtype=float),
            quadrature_range=max(x_candidate.finite_range, p_candidate.finite_range, 1.0),
            num_quadrature_nodes=num_quadrature_nodes,
            probability_engine=probability_engine,
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


def run_route5(
    cutoff: int = 6,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    x_bounds: np.ndarray | None = None,
    p_bounds: np.ndarray | None = None,
    quadrature_range: float = 3.0,
    boundary_gamma: float = 1.0,
    num_quadrature_nodes: int | None = None,
    probability_engine: str | None = None,
    max_inputs_to_certify: int | None = 1,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """运行广义相干态字母表的Route 5协议。
    
    使用物理约束的IQ分区执行完整的量子随机数生成协议。
    包括：生成相干态字母表、构造联合输入态、计算双零差探测概率、
    认证目标输入态并计算最小熵。
    
    执行逻辑：
    1. 生成广义相干态字母表和联合输入态
    2. 解析或生成X和P轴的分箱边界
    3. 计算双零差探测的概率分布
    4. 计算所有输入态的原始最小熵
    5. 选择原始熵最高的候选态进行认证
    6. 执行SDP认证，计算认证后的最小熵
    7. 收集并返回完整的结果字典
    
    参数：
        cutoff: Fock空间的截断维度，默认6
        alpha_values: 直接指定的复振幅列表
        radius_values: 半径值列表（用于生成振幅）
        phase_values: 相位值列表（用于生成振幅）
        num_x_bins: X轴分箱数，默认2
        num_p_bins: P轴分箱数，默认2
        x_bounds: X轴边界数组，None时自动生成
        p_bounds: P轴边界数组，None时自动生成
        quadrature_range: 正交相位范围，默认3.0
        boundary_gamma: 边界幂次参数，默认1.0
        num_quadrature_nodes: 正交相位求积节点数，仅 ``trace_povm`` 后端使用
        probability_engine: 概率计算后端名称，默认保留原 ``trace_povm`` 方法
        max_inputs_to_certify: 最多认证的输入态数量，None表示认证所有
        preferred_solver: 首选SDP求解器名称
        solver_options: 求解器选项字典
        verbose: 是否输出详细信息
    
    返回：
        完整的结果字典，包含：
        - route: 协议名称
        - cutoff, num_local_states, num_inputs, num_outputs: 维度信息
        - local_alphas: 本地振幅列表
        - x_bounds, p_bounds: 分箱边界
        - raw_best_H_min: 原始最佳最小熵
        - H_min: 认证后的最小熵
        - p_guess: 猜测概率
        - target_scan: 目标态扫描结果
    """
    (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
    ) = reduced_joint_inputs_from_alphas(
        cutoff,
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
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

    resolved_probability_engine = _resolve_probability_engine(probability_engine)
    probabilities, output_labels, x_bounds_out, p_bounds_out = route5_iq_probabilities(
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        x_bounds=resolved_x_bounds,
        p_bounds=resolved_p_bounds,
        quadrature_range=quadrature_range,
        num_quadrature_nodes=num_quadrature_nodes,
        probability_engine=resolved_probability_engine,
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
            "route": "route5_cv_generalized_iq",
            "probability_engine": resolved_probability_engine,
            "state_representation": "truncated_fock_projected_support",
            "cutoff": int(cutoff),
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
            "num_quadrature_nodes": (
                None
                if resolved_probability_engine != "trace_povm" or num_quadrature_nodes is None
                else int(num_quadrature_nodes)
            ),
            "requested_num_quadrature_nodes": None if num_quadrature_nodes is None else int(num_quadrature_nodes),
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


def search_route5_iq_partitions(
    cutoff: int = 6,
    alpha_values: list[complex] | None = None,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins_values: list[int] | None = None,
    num_p_bins_values: list[int] | None = None,
    quadrature_ranges: list[float] | None = None,
    gamma_values: list[float] | None = None,
    num_quadrature_nodes: int | None = None,
    probability_engine: str | None = None,
    certify_top_k: int = 3,
    max_inputs_to_certify: int | None = 1,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """搜索Route 5的物理约束轴对齐IQ分区。
    
    系统地搜索最优的X-P正交相位分区配置。生成多种分箱配置，
    按原始熵排序，然后对top-k候选进行完整的SDP认证。
    
    执行逻辑：
    1. 生成广义相干态字母表和联合输入态
    2. 枚举所有X-P分区候选并按原始熵排序
    3. 对top-k个候选进行SDP认证
    4. 选择认证后最小熵最高的结果
    5. 返回完整的结果字典，包含所有候选的排名和认证结果
    
    参数：
        cutoff: Fock空间的截断维度，默认6
        alpha_values: 直接指定的复振幅列表
        radius_values: 半径值列表
        phase_values: 相位值列表
        num_x_bins_values: X轴分箱数候选列表
        num_p_bins_values: P轴分箱数候选列表
        quadrature_ranges: 有限范围候选列表
        gamma_values: 幂次参数候选列表
        num_quadrature_nodes: 正交相位求积节点数，仅 ``trace_povm`` 后端使用
        probability_engine: 概率计算后端名称
        certify_top_k: 认证top-k个分区候选，默认3
        max_inputs_to_certify: 每个分区最多认证的输入态数量
        preferred_solver: 首选SDP求解器
        solver_options: 求解器选项
        verbose: 是否输出详细信息
    
    返回：
        完整的结果字典，包含：
        - route: 协议名称
        - num_partition_candidates: 分区候选总数
        - raw_partition_ranking: 原始熵排名
        - certified_partition_results: 认证结果列表
        - selected_candidate_index: 选中的候选索引
        - H_min: 最佳认证最小熵
    """
    (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
    ) = reduced_joint_inputs_from_alphas(
        cutoff,
        alpha_values=alpha_values,
        radius_values=radius_values,
        phase_values=phase_values,
    )

    resolved_probability_engine = _resolve_probability_engine(probability_engine)
    ranked_candidates, x_candidates, p_candidates = _raw_partition_candidates(
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        cutoff,
        num_x_bins_values=num_x_bins_values,
        num_p_bins_values=num_p_bins_values,
        quadrature_ranges=quadrature_ranges,
        gamma_values=gamma_values,
        num_quadrature_nodes=num_quadrature_nodes,
        probability_engine=resolved_probability_engine,
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
        certified_entry = {
            key: value
            for key, value in candidate.items()
            if key != "probabilities"
        }
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
        best = {
            key: value
            for key, value in ranked_candidates[0].items()
            if key != "probabilities"
        }
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
            "route": "route5_cv_generalized_iq_search",
            "probability_engine": resolved_probability_engine,
            "state_representation": "truncated_fock_projected_support",
            "cutoff": int(cutoff),
            "num_local_states": len(local_alphas),
            "num_inputs": len(joint_states),
            "local_alphas": serialize_complex_list(local_alphas),
            "local_rank": int(local_rank),
            "local_operator_span_rank": int(local_operator_span),
            "local_operator_space_dim": int(local_rank**2),
            "joint_dim": int(joint_dim),
            "operator_span_rank": int(operator_span_rank(joint_states)),
            "operator_space_dim": int(joint_dim**2),
            "num_x_candidates": len(x_candidates),
            "num_p_candidates": len(p_candidates),
            "num_partition_candidates": len(ranked_candidates),
            "certify_top_k": int(certify_count),
            "selection_strategy": "rank partitions by raw-best target entropy, then certify top candidates",
            "num_quadrature_nodes": (
                None
                if resolved_probability_engine != "trace_povm" or num_quadrature_nodes is None
                else int(num_quadrature_nodes)
            ),
            "requested_num_quadrature_nodes": None if num_quadrature_nodes is None else int(num_quadrature_nodes),
            "raw_partition_ranking": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "probabilities"
                }
                for candidate in ranked_candidates[: min(10, len(ranked_candidates))]
            ],
            "certified_partition_results": certified_candidates,
            "selected_candidate_index": int(best["candidate_index"]),
            "certified_best_target_index": int(best["target_index"]),
            "certified_best_target": best["target_input"],
            "certified_best_target_alphas": best["target_alphas"],
        }
    )
    return best


def search_route5_alphabets(
    cutoff: int = 4,
    radius_values: list[float] | None = None,
    phase_values: list[float] | None = None,
    num_radii_values: list[int] | None = None,
    num_phase_values: list[int] | None = None,
    require_vacuum: bool = True,
    max_local_states: int | None = None,
    num_x_bins_values: list[int] | None = None,
    num_p_bins_values: list[int] | None = None,
    quadrature_ranges: list[float] | None = None,
    gamma_values: list[float] | None = None,
    num_quadrature_nodes: int | None = None,
    probability_engine: str | None = None,
    alphabet_top_k: int = 3,
    certify_top_k: int = 1,
    max_inputs_to_certify: int | None = 1,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """搜索可信字母表，然后认证最佳IQ分区候选。
    
    两阶段优化搜索：首先搜索最优的相干态字母表配置，然后对每个字母表
    搜索最优的IQ分区配置。这是Route 5的最高层接口，提供完整的参数空间搜索。
    
    执行逻辑：
    1. 生成所有字母表候选（半径和相位组合）
    2. 对每个字母表候选：
       - 生成联合输入态
       - 枚举并排序IQ分区候选
       - 记录原始熵和结构信息
    3. 按本地张成比例和原始熵对字母表排序
    4. 对top-k个字母表进行完整认证：
       - 对每个字母表执行IQ分区搜索
       - 认证最佳分区候选
    5. 选择认证后最小熵最高的结果
    6. 返回完整的结果字典
    
    参数：
        cutoff: Fock空间的截断维度，默认4
        radius_values: 半径值池
        phase_values: 相位值池
        num_radii_values: 半径数量候选列表
        num_phase_values: 相位数量候选列表
        require_vacuum: 是否要求包含真空态
        max_local_states: 最大本地态数量限制
        num_x_bins_values: X轴分箱数候选列表
        num_p_bins_values: P轴分箱数候选列表
        quadrature_ranges: 有限范围候选列表
        gamma_values: 幂次参数候选列表
        num_quadrature_nodes: 正交相位求积节点数，仅 ``trace_povm`` 后端使用
        probability_engine: 概率计算后端名称
        alphabet_top_k: 认证top-k个字母表候选，默认3
        certify_top_k: 每个字母表认证top-k个分区候选，默认1
        max_inputs_to_certify: 每个分区最多认证的输入态数量
        preferred_solver: 首选SDP求解器
        solver_options: 求解器选项
        verbose: 是否输出详细信息
    
    返回：
        完整的结果字典，包含：
        - route: 协议名称
        - num_alphabet_candidates: 字母表候选总数
        - raw_alphabet_ranking: 字母表原始排名
        - certified_alphabet_results: 字母表认证结果
        - selected_alphabet_candidate_index: 选中的字母表索引
        - best_certified_H_min: 最佳认证最小熵
        - best_partition_search_result: 最佳分区搜索结果
    
    异常：
        RuntimeError: 当没有生成任何字母表候选时抛出
    """
    radius_pool = DEFAULT_RADIUS_VALUES if radius_values is None else list(radius_values)
    phase_pool = DEFAULT_PHASE_VALUES if phase_values is None else list(phase_values)
    radius_count_values = [2, 3] if num_radii_values is None else list(num_radii_values)
    phase_count_values = [4] if num_phase_values is None else list(num_phase_values)

    resolved_probability_engine = _resolve_probability_engine(probability_engine)
    candidate_specs = generate_alphabet_candidates_from_grid(
        radius_pool,
        phase_pool,
        radius_count_values,
        phase_count_values,
        require_vacuum=require_vacuum,
        max_local_states=max_local_states,
    )
    if len(candidate_specs) == 0:
        raise RuntimeError("No trusted-alphabet candidates were generated.")

    raw_alphabet_results: list[dict] = []
    for alphabet_index, candidate in enumerate(candidate_specs):
        (
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            local_rank,
            joint_dim,
            local_operator_span,
        ) = reduced_joint_inputs_from_alphas(
            cutoff,
            alpha_values=list(candidate.alpha_values),
        )
        ranked_partitions, _, _ = _raw_partition_candidates(
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            cutoff,
            num_x_bins_values=num_x_bins_values,
            num_p_bins_values=num_p_bins_values,
            quadrature_ranges=quadrature_ranges,
            gamma_values=gamma_values,
            num_quadrature_nodes=num_quadrature_nodes,
            probability_engine=resolved_probability_engine,
            store_probabilities=False,
        )
        summary = _alphabet_summary(
            alphabet_index,
            candidate,
            local_rank=local_rank,
            local_operator_span=local_operator_span,
            joint_dim=joint_dim,
            joint_operator_span=operator_span_rank(joint_states),
        )
        summary["raw_best_partition"] = ranked_partitions[0]
        summary["raw_partition_ranking"] = ranked_partitions[: min(5, len(ranked_partitions))]
        raw_alphabet_results.append(summary)

    ranked_alphabets = sorted(
        raw_alphabet_results,
        key=lambda item: (
            item["local_span_ratio"],
            item["raw_best_partition"]["raw_best_H_min"],
            item["local_operator_span_rank"],
            item["operator_span_rank"],
            -item["num_local_states"],
        ),
        reverse=True,
    )

    alphabet_certify_count = min(max(0, alphabet_top_k), len(ranked_alphabets))
    certified_alphabet_results: list[dict] = []
    for alphabet_result in ranked_alphabets[:alphabet_certify_count]:
        partition_result = search_route5_iq_partitions(
            cutoff=cutoff,
            alpha_values=[
                complex(entry["real"], entry["imag"])
                for entry in alphabet_result["alpha_values"]
            ],
            num_x_bins_values=num_x_bins_values,
            num_p_bins_values=num_p_bins_values,
            quadrature_ranges=quadrature_ranges,
            gamma_values=gamma_values,
            num_quadrature_nodes=num_quadrature_nodes,
            probability_engine=resolved_probability_engine,
            certify_top_k=certify_top_k,
            max_inputs_to_certify=max_inputs_to_certify,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        certified_entry = dict(alphabet_result)
        certified_entry["partition_search_result"] = partition_result
        certified_entry["best_certified_H_min"] = partition_result["H_min"]
        certified_entry["best_certified_partition"] = {
            "candidate_index": partition_result["selected_candidate_index"],
            "num_outputs": partition_result["num_outputs"],
            "num_x_bins": partition_result["num_x_bins"],
            "num_p_bins": partition_result["num_p_bins"],
            "x_bounds": partition_result["x_bounds"],
            "p_bounds": partition_result["p_bounds"],
            "raw_best_H_min": partition_result["raw_best_H_min"],
            "certified_H_min": partition_result["H_min"],
            "certified_best_target": partition_result["certified_best_target"],
            "certified_best_target_alphas": partition_result["certified_best_target_alphas"],
        }
        certified_alphabet_results.append(certified_entry)

    if len(certified_alphabet_results) > 0:
        best = dict(
            max(
                certified_alphabet_results,
                key=lambda item: item["best_certified_H_min"] if item["best_certified_H_min"] is not None else -np.inf,
            )
        )
        partition_result = best["partition_search_result"]
    else:
        best = dict(ranked_alphabets[0])
        partition_result = None

    best.update(
        {
            "route": "route5_cv_alphabet_search",
            "probability_engine": resolved_probability_engine,
            "state_representation": "truncated_fock_projected_support",
            "cutoff": int(cutoff),
            "radius_pool": [float(value) for value in radius_pool],
            "phase_pool": [float(value) for value in _unique_sorted_phases(phase_pool)],
            "num_radii_values": [int(value) for value in radius_count_values],
            "num_phase_values": [int(value) for value in phase_count_values],
            "require_vacuum": bool(require_vacuum),
            "max_local_states": None if max_local_states is None else int(max_local_states),
            "num_quadrature_nodes": (
                None
                if resolved_probability_engine != "trace_povm" or num_quadrature_nodes is None
                else int(num_quadrature_nodes)
            ),
            "requested_num_quadrature_nodes": None if num_quadrature_nodes is None else int(num_quadrature_nodes),
            "num_alphabet_candidates": len(candidate_specs),
            "alphabet_top_k": int(alphabet_certify_count),
            "partition_certify_top_k": int(certify_top_k),
            "alphabet_selection_strategy": "rank alphabets by local span ratio, then raw-best IQ entropy",
            "raw_alphabet_ranking": ranked_alphabets[: min(10, len(ranked_alphabets))],
            "certified_alphabet_results": certified_alphabet_results,
            "selected_alphabet_candidate_index": int(best["alphabet_candidate_index"]),
            "selected_partition_candidate_index": (
                int(partition_result["selected_candidate_index"])
                if partition_result is not None
                else int(best["raw_best_partition"]["candidate_index"])
            ),
            "best_certified_H_min": None if partition_result is None else partition_result["H_min"],
            "best_partition_search_result": partition_result,
        }
    )
    return best
