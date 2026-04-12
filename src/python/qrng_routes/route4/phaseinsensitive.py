"""
相位不敏感QRNG路线 (Route 4)
==============================

物理背景
--------
本模块实现了一种基于相位不敏感(phase-insensitive)探测的量子随机数认证方案。
与Route 3（CV Bell测量，需要干涉和相位敏感检测）不同，Route 4仅使用单模探测，
不需要干涉仪，实验实现更为简单。

核心思想：
1. Alice在光脉冲中编码随机相位，但探测器仅响应光子数（相位信息被丢弃）
2. 由于相位随机化，量子态可以用Fock基的对角密度矩阵描述：
   ρ = Σ_n p(n) |n⟩⟨n|
3. 利用不同光强下探测器输出的统计差异，通过SDP计算可认证的随机数熵率

关键物理概念：
- 相位不敏感探测器：仅测量光子数，忽略相位信息（如单光子探测器SPD）
- Poisson分布：相干态在Fock基下的光子数分布为 Poisson(μ)
  p(n) = e^(-μ) · μ^n / n!
- Fock对角态(Phase-averaged coherent state)：
  ρ_μ = (1/2π) ∫ |√μ·e^(iφ)⟩⟨√μ·e^(iφ)| dφ = Σ_n e^(-μ)μ^n/n! |n⟩⟨n|
- 粗粒化(Coarse-graining)：将探测器的原始输出（如256个bin）合并为较少的区间
- 猜测概率(Guessing probability)：对手在已知探测器输出后，最优猜测输入的概率
- 对偶问题(Dual LP)：猜测概率的线性规划上界
- 原始问题(Primal SDP)：猜测概率的半定规划下界

术语约定：
- mu (μ): 平均光子数（整数），对应实验中不同的光强设置
- cutoff: Fock空间截断维度，用于数值计算
- num_outputs: 粗粒化后的输出区间数
- q_selected: 各输入态的先验概率权重
- prob_floor: 概率下限，用于正则化零概率项

模块结构：
- 数据加载与预处理：load_probability_data, build_coherent_diagonals
- 粗粒化：build_equal_cover_edges, coarse_grain_row
- SDP/LP求解：solve_phaseinsensitive_dual, solve_phaseinsensitive_primal, solve_phaseinsensitive_full_primal
- 批量运行与比较：run_route4_*, compare_route4_*, sweep_route4_*, search_route4_triplets
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations, product as iterproduct
from math import comb
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.io import loadmat
from scipy.special import gammaln

from ..common import solve_cvxpy_problem

# ── 全局默认参数 ──────────────────────────────────────────────────────────
# FULL_MU: 实验数据中所有可用的光强设置（平均光子数μ值，整数）
# 典型实验中，μ=0 对应暗计数（无光输入），μ=160 对应最高光强
FULL_MU = [0, 20, 40, 60, 80, 100, 120, 140, 160]

# DEFAULT_SELECTED_MU: 默认选择的光强组合，用于主计算
# 选择 {100, 120, 140} 是因为这些中等偏高的光强在实验中信号较好，
# 且它们之间的探测器响应差异足够大，有利于随机性认证
DEFAULT_SELECTED_MU = [100, 120, 140]

# DEFAULT_Q: 默认的输入先验概率分布
# 对应 μ=100, 120, 140 三个光强的权重分别为 0.25, 0.25, 0.5
# 不均匀的权重可以让认证偏向某个特定光强
DEFAULT_Q = [0.25, 0.25, 0.5]

# DEFAULT_CUTOFF: Fock空间的截断维度
# 对于 μ≈100-160 的光强，Poisson分布的峰值在 √μ ≈ 10-13 附近，
# 截断到280确保累积概率 > 1 - 10^(-50)
DEFAULT_CUTOFF = 280

# DEFAULT_NUM_OUTPUTS: 粗粒化后的默认输出区间数
# 将探测器原始的256个bin合并为6个区间
DEFAULT_NUM_OUTPUTS = 6

# DEFAULT_PROB_FLOOR: 概率正则化的下限值
# 用于处理实验数据中的零概率bin，避免对数运算中的数值问题
# 设为1e-12是经验值，足够小不影响物理结果，但避免数值奇异
DEFAULT_PROB_FLOOR = 1e-12

# DEFAULT_SHIFT: 行偏移量，用于从概率表中选取特定光强对应的数据行
DEFAULT_SHIFT = 0


def _default_probability_path() -> Path:
    """
    获取默认的实验概率数据文件路径
    ==============================

    文件位置约定
    -----------
    概率数据文件存放在项目的 matlab/ 子目录下，文件名为 Probability.mat。
    该文件包含不同光强设置下探测器的原始输出统计。

    路径构造
    --------
    从当前文件 (phaseinsensitive.py) 出发，向上3级到达项目根目录，
    再进入 matlab/ 子目录：
    phaseinsensitive.py → route4/ → qrng_routes/ → src/python/ → 项目根目录

    返回
    ----
    path : Path
        Probability.mat 文件的绝对路径
    """
    return Path(__file__).resolve().parents[3] / "matlab" / "Probability.mat"


def _clean_value(value: Any) -> Any:
    """
    JSON序列化辅助：将NumPy类型转换为Python原生类型
    ===============================================

    处理规则
    --------
    - np.ndarray → list（递归转为嵌套Python列表）
    - np.floating / np.integer → Python float / int（通过 .item()）
    - 其他类型 → 原样返回（str, dict, list, None等）

    参数
    ----
    value : Any
        待转换的值，可能包含NumPy类型

    返回
    ----
    converted : Any
        JSON可序列化的Python原生类型
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def result_to_json(result: Any) -> str:
    """
    将SDP/LP结果字典序列化为格式化的JSON字符串
    ===========================================

    用途
    ----
    将运行结果（包含NumPy数组、浮点数等）转换为可读的JSON格式，
    便于保存到文件或打印输出。

    参数
    ----
    result : Any
        通常是 run_route4_* 系列函数返回的字典

    返回
    ----
    json_str : str
        缩进2空格、支持中文（ensure_ascii=False）的JSON字符串

    示例
    ----
    >>> result = run_route4_dual()
    >>> print(result_to_json(result))  # 输出格式化的JSON
    """
    return json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value)


def build_equal_cover_edges(num_raw_bins: int, num_outputs: int) -> np.ndarray:
    """
    构造等覆盖粗粒化的区间边界
    ==========================

    物理原理
    --------
    探测器的原始输出通常有很精细的分箱（如256个bin），但过多的输出类别
    会导致SDP问题规模爆炸（变量数 ∝ num_outputs^(num_inputs+1)）。
    因此需要将连续的原始bin合并为较少的粗粒化区间。

    粗粒化策略
    ----------
    将 num_raw_bins 个原始bin均匀划分为 num_outputs 个连续区间，
    每个区间包含 floor(num_raw_bins/num_outputs) 或 ceil(num_raw_bins/num_outputs) 个原始bin。
    所有原始bin恰好被覆盖一次，无重叠、无遗漏。

    参数
    ----
    num_raw_bins : int
        原始bin的总数（如256个探测器输出bin）

    num_outputs : int
        粗粒化后的区间数（如2、4、6个输出）

    返回
    ----
    edges : np.ndarray
        长度为 num_outputs + 1 的整数数组，表示每个粗粒化区间的起止索引
        第k个区间覆盖 raw_bins[edges[k] : edges[k+1]]

    算法
    ----
    使用整数除法实现均匀分割：
        edges[k] = (k × num_raw_bins) // num_outputs

    这保证了每个区间宽度差不超过1，且总和恰好等于 num_raw_bins。

    示例
    ----
    >>> build_equal_cover_edges(256, 4)
    array([  0,  64, 128, 192, 256])
    >>> # 4个区间，每个覆盖64个原始bin
    """
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive.")
    if num_outputs > num_raw_bins:
        raise ValueError(
            f"num_outputs={num_outputs} exceeds the available raw bins ({num_raw_bins})."
        )
    edges = np.array([(k * num_raw_bins) // num_outputs for k in range(num_outputs + 1)], dtype=int)
    if edges[0] != 0 or edges[-1] != num_raw_bins:
        raise RuntimeError("Internal error while constructing coarse-graining edges.")
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Coarse-graining edges must be strictly increasing.")
    return edges


def validate_custom_edges(
    custom_edges: list[int] | tuple[int, ...] | np.ndarray,
    num_raw_bins: int,
) -> np.ndarray:
    """
    校验用户给定的连续粗粒化边界
    ============================

    参数
    ----
    custom_edges : list[int] | tuple[int, ...] | np.ndarray
        用户显式指定的边界数组，必须从0开始、以 num_raw_bins 结束，
        且中间边界严格递增。

    num_raw_bins : int
        原始输出bin总数。

    返回
    ----
    edges : np.ndarray
        通过校验后的整型边界数组。
    """
    edges = np.asarray(custom_edges, dtype=int).reshape(-1)
    if edges.size < 2:
        raise ValueError("custom_edges must contain at least two endpoints.")
    if int(edges[0]) != 0 or int(edges[-1]) != num_raw_bins:
        raise ValueError(
            f"custom_edges must start at 0 and end at {num_raw_bins}, got {edges.tolist()}."
        )
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"custom_edges must be strictly increasing, got {edges.tolist()}.")
    return edges


def coarse_grain_row(
    probabilities_256: np.ndarray,
    num_outputs: int | None = None,
    custom_edges: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将单行原始概率分布粗粒化为较少的输出区间
    =========================================

    物理原理
    --------
    给定某个光强设置下探测器的原始概率分布（如256个bin的概率），
    将其合并为 num_outputs 个粗粒化区间的概率和。

    粗粒化保持概率归一化：Σ(粗粒化概率) = Σ(原始概率) = 1。

    参数
    ----
    probabilities_256 : np.ndarray
        原始概率分布（一维数组，长度通常为256）
        probabilities_256[i] = 探测器输出落在第i个bin的概率

    num_outputs : int | None
        粗粒化后的区间数。当 custom_edges 为 None 时必须提供。

    custom_edges : list[int] | tuple[int, ...] | np.ndarray | None
        若提供，则直接使用这组连续边界；此时 num_outputs 会被忽略。

    返回
    ----
    coarse : np.ndarray
        粗粒化后的概率分布，长度为 num_outputs
        coarse[k] = Σ_{i∈bin_k} probabilities_256[i]

    edges : np.ndarray
        区间边界索引数组，由 build_equal_cover_edges 生成

    示例
    ----
    >>> raw = np.ones(256) / 256  # 均匀分布
    >>> coarse, edges = coarse_grain_row(raw, 4)
    >>> # coarse ≈ [0.25, 0.25, 0.25, 0.25]
    """
    raw = np.asarray(probabilities_256, dtype=float).reshape(-1)
    if custom_edges is not None:
        edges = validate_custom_edges(custom_edges, raw.size)
    else:
        if num_outputs is None:
            raise ValueError("num_outputs must be provided when custom_edges is None.")
        edges = build_equal_cover_edges(raw.size, num_outputs)
    num_outputs = int(edges.size - 1)
    coarse = np.array([raw[edges[k] : edges[k + 1]].sum() for k in range(num_outputs)], dtype=float)
    return coarse, edges


def load_probability_data(probability_path: str | Path | None = None) -> np.ndarray:
    """
    加载实验探测器概率数据
    ======================

    物理原理
    --------
    从MATLAB格式的数据文件中读取不同光强设置下探测器的输出统计。
    该数据是Route 4认证方案的实验基础——它记录了每种光强μ下，
    探测器输出落在各个bin中的概率分布。

    数据格式
    --------
    文件为 .mat 格式（MATLAB数据文件），包含一个二维概率矩阵：
    - 行索引：对应不同光强设置（与 FULL_MU 列表对应）
    - 列索引：对应探测器的原始输出bin（通常256个）
    - 矩阵元素：P(输出bin | 光强μ)

    参数
    ----
    probability_path : str | Path | None
        .mat 文件的路径。若为 None，使用默认路径
        （项目根目录下的 matlab/Probability.mat）

    返回
    ----
    table : np.ndarray
        二维概率矩阵，形状为 (num_intensities, num_raw_bins)
        每行对应一个光强设置，每列对应一个探测器输出bin

    异常
    ----
    ValueError : 如果文件为空、数据不是二维数组，或路径无效
    """
    path = Path(probability_path) if probability_path is not None else _default_probability_path()
    mat_data = loadmat(path)
    variable_names = [name for name in mat_data.keys() if not name.startswith("__")]
    if not variable_names:
        raise ValueError(f"No data arrays found in {path}.")
    table = np.asarray(mat_data[variable_names[0]], dtype=float)
    if table.ndim != 2:
        raise ValueError(f"Expected a 2-D probability table in {path}, got shape {table.shape}.")
    return table


def build_coherent_diagonals(selected_mu_list: list[int] | tuple[int, ...], cutoff: int) -> np.ndarray:
    """
    构造Fock对角相干态概率分布
    ==========================

    物理原理
    --------
    对于相位随机化的相干态（即对相位积分后的混合态），
    其密度矩阵在Fock基下是对角的：

        ρ_μ = Σ_n p_μ(n) |n⟩⟨n|

    其中 p_μ(n) 是Poisson分布：
        p_μ(n) = e^(-μ) · μ^n / n!

    这是Route 4的核心物理假设：输入态的相位完全随机化，
    因此密度矩阵退化为Fock对角形式。

    参数
    ----
    selected_mu_list : list[int] | tuple[int, ...]
        选用的平均光子数列表（如 [100, 120, 140]）
        每个μ对应一个不同的输入态

    cutoff : int
        Fock空间截断维度。仅计算 n=0, 1, ..., cutoff-1 的概率。
        截断后的概率不会严格归一（尾部被丢弃），但对于足够大的cutoff
        误差可忽略。

    返回
    ----
    diagonals : np.ndarray
        形状为 (len(selected_mu_list), cutoff) 的二维数组
        diagonals[i, n] = e^(-μ_i) · μ_i^n / n!

    数值稳定性
    ----------
    使用对数空间计算避免大数溢出：
        log p(n) = -μ + n·log(μ) - log(n!)
        其中 log(n!) = gammaln(n+1)
    最后通过 exp() 还原。

    特殊情况
    --------
    μ=0 时：Poisson分布退化为 p(0)=1, p(n>0)=0，即真空态。
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    diagonals = np.zeros((len(selected_mu_list), cutoff), dtype=float)
    photon_numbers = np.arange(cutoff, dtype=float)
    for idx, mu in enumerate(selected_mu_list):
        if mu < 0:
            raise ValueError("Mean photon numbers must be non-negative.")
        if mu == 0:
            diagonals[idx, 0] = 1.0
            continue
        log_probs = -mu + photon_numbers * np.log(mu) - gammaln(photon_numbers + 1.0)
        diagonals[idx, :] = np.exp(log_probs)
    return diagonals


def distribution_only_guessing_probability(probabilities: np.ndarray, q_selected: np.ndarray) -> float:
    """
    仅从输出分布计算的猜测概率上界
    ===============================

    物理原理
    --------
    如果对手仅利用输出分布的统计特征（不构造量子态模型），
    最优猜测策略是：对于每个输出c，选择使 P(c|x) 最大的输入x。

    猜测概率为：
        P_guess = Σ_x q(x) · max_c P(c|x)

    这是不考虑量子约束时最简单的猜测概率估计，
    它给出的是P_guess的上界（因为真实量子约束会进一步限制对手能力）。

    参数
    ----
    probabilities : np.ndarray
        条件概率矩阵，形状为 (num_inputs, num_outputs)
        probabilities[x, c] = P(输出c | 输入x)

    q_selected : np.ndarray
        输入的先验概率分布，长度为 num_inputs
        q_selected[x] = P(选择输入x)

    返回
    ----
    p_guess : float
        猜测概率的上界估计
        对应的最小熵下界为 H_min = -log₂(P_guess)

    数学细节
    --------
    这等价于对手采用"逐行最大值"策略：
    对于每个输入x，对手面对的最可区分输出是 argmax_c P(c|x)。
    加权求和即得到整体猜测概率。
    """
    return float(np.dot(q_selected, probabilities.max(axis=1)))


def prepare_phaseinsensitive_instance(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
    custom_edges: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    准备相位不敏感SDP/LP问题的共享数据实例
    =======================================

    物理原理
    --------
    在运行对偶LP或原始SDP之前，需要将实验数据（探测器统计）和
    理论模型（Poisson对角分布）整合为一个统一的数据结构。
    本函数完成以下工作：
    1. 加载实验概率数据
    2. 按选定的光强提取对应行
    3. 粗粒化为指定数量的输出区间
    4. 可选地正则化零概率项
    5. 构造Poisson对角分布矩阵

    参数
    ----
    num_outputs : int
        粗粒化后的输出区间数，默认6

    selected_mu_list : list[int] | tuple[int, ...]
        选用的平均光子数列表，默认 [100, 120, 140]
        每个值必须出现在 full_mu 列表中

    q_selected : list[float] | tuple[float, ...]
        输入的先验概率分布（归一化前），默认 [0.25, 0.25, 0.5]
        长度必须与 selected_mu_list 相同

    cutoff : int
        Fock空间截断维度，默认280

    prob_floor : float | None
        概率正则化下限。若非None且大于0，将所有零概率替换为该值，
        然后重新归一化。设为None则跳过正则化。

    shift : int
        行偏移量。选取概率表行时，实际行为 full_mu.index(μ) + shift。
        用于处理数据文件中的行排列。

    probability_path : str | Path | None
        概率数据文件路径，None则使用默认路径

    full_mu : list[int] | tuple[int, ...]
        全部可用的光强设置列表

    custom_edges : list[int] | tuple[int, ...] | np.ndarray | None
        若提供，则直接使用这组连续边界做 coarse-graining，并覆盖 num_outputs。

    返回
    ----
    instance : dict[str, Any]
        包含所有预处理数据的字典，主要字段：
        - 'rho_diag': Poisson对角分布矩阵 (num_inputs × cutoff)
        - 'probabilities': 正则化后的概率矩阵 (num_inputs × num_outputs)
        - 'probabilities_raw': 原始概率矩阵（未经正则化）
        - 'q_selected': 归一化后的先验概率
        - 'edges': 粗粒化边界
        - 'block_widths': 各粗粒化区间的宽度
        - 'mixed_zero_columns_raw': 部分输入为零、部分非零的输出列
        - 'all_zero_columns_raw': 所有输入均为零的输出列
        - 'distribution_only_p_guess': 仅分布估计的猜测概率

    正则化的必要性
    --------------
    实验数据中某些bin的概率可能为零（样本不足或物理上极不可能）。
    但SDP/LP中的对数运算要求概率严格大于0。
    prob_floor 将零概率提升到一个小的正值，然后重新归一化，
    确保概率归一性的同时避免数值奇异性。

    异常
    ----
    ValueError : 输入参数不合法（空列表、μ不在可用范围、概率和为零等）
    """
    selected_mu = list(selected_mu_list)
    full_mu_list = list(full_mu)
    if len(selected_mu) == 0:
        raise ValueError("At least one input state is required.")
    if any(mu not in full_mu_list for mu in selected_mu):
        raise ValueError(f"selected_mu_list must be a subset of {full_mu_list}.")

    q = np.asarray(q_selected, dtype=float).reshape(-1)
    if q.size != len(selected_mu):
        raise ValueError("q_selected must have the same length as selected_mu_list.")
    if np.any(q < 0):
        raise ValueError("q_selected must be non-negative.")
    if float(q.sum()) <= 0.0:
        raise ValueError("q_selected must sum to a positive value.")
    q = q / q.sum()

    probability_table = load_probability_data(probability_path)
    if probability_table.shape[0] <= max(full_mu_list.index(mu) + shift for mu in selected_mu):
        raise ValueError("Probability table does not contain the requested shifted rows.")

    num_raw_bins = int(probability_table.shape[1])
    validated_edges = (
        validate_custom_edges(custom_edges, num_raw_bins) if custom_edges is not None else None
    )
    effective_num_outputs = (
        int(validated_edges.size - 1) if validated_edges is not None else int(num_outputs)
    )

    raw_probabilities = np.zeros((len(selected_mu), effective_num_outputs), dtype=float)
    edges: np.ndarray | None = None
    selected_indices = [full_mu_list.index(mu) for mu in selected_mu]
    for row_idx, full_idx in enumerate(selected_indices):
        coarse, row_edges = coarse_grain_row(
            probability_table[full_idx + shift, :],
            num_outputs=effective_num_outputs if validated_edges is None else None,
            custom_edges=validated_edges,
        )
        raw_probabilities[row_idx, :] = coarse
        if edges is None:
            edges = row_edges

    assert edges is not None
    regularized_probabilities = raw_probabilities.copy()
    regularized_entries = 0
    if prob_floor is not None and prob_floor > 0:
        regularized_entries = int((regularized_probabilities == 0.0).sum())
        regularized_probabilities = np.maximum(regularized_probabilities, prob_floor)
        regularized_probabilities = regularized_probabilities / regularized_probabilities.sum(
            axis=1, keepdims=True
        )

    rho_diag = build_coherent_diagonals(selected_mu, cutoff)
    mixed_zero_columns = [
        int(column)
        for column in range(effective_num_outputs)
        if np.any(raw_probabilities[:, column] == 0.0) and np.any(raw_probabilities[:, column] > 0.0)
    ]
    all_zero_columns = [
        int(column)
        for column in range(effective_num_outputs)
        if np.all(raw_probabilities[:, column] == 0.0)
    ]

    return {
        "selected_mu_list": selected_mu,
        "q_selected": q,
        "cutoff": cutoff,
        "num_inputs": len(selected_mu),
        "num_outputs": effective_num_outputs,
        "shift": shift,
        "probability_path": str(
            Path(probability_path) if probability_path is not None else _default_probability_path()
        ),
        "rho_diag": rho_diag,
        "probabilities_raw": raw_probabilities,
        "probabilities": regularized_probabilities,
        "edges": edges,
        "block_widths": np.diff(edges),
        "row_sums_raw": raw_probabilities.sum(axis=1),
        "mixed_zero_columns_raw": mixed_zero_columns,
        "all_zero_columns_raw": all_zero_columns,
        "regularized_entries": regularized_entries,
        "prob_floor": prob_floor,
        "distribution_only_p_guess_raw": distribution_only_guessing_probability(raw_probabilities, q),
        "distribution_only_p_guess": distribution_only_guessing_probability(regularized_probabilities, q),
    }


def estimate_primal_problem_size(
    num_inputs: int,
    num_outputs: int,
    cutoff: int,
) -> dict[str, int]:
    """
    估算对角原始问题的规模
    ======================

    物理背景
    --------
    在Fock对角POVM假设下，原始问题的规模可以精确估算。
    这有助于在运行求解器之前判断问题是否可解。

    估算公式
    --------
    - 策略数(num_strategies): num_outputs^(num_inputs+1)
      每个策略是一个确定性函数 e: {0,...,num_inputs} → {0,...,num_outputs-1}
      对手选择一个策略来最大化猜测概率
    - 变量数(variable_count): cutoff × num_outputs × num_strategies
      每个策略的每个输出对应一个cutoff维非负向量（Fock对角POVM元）
    - 归一化约束(normalization_constraints): cutoff × num_strategies - num_strategies
      每个策略的POVM元之和等于一个非负权重向量（逐分量相等约束）
    - 统计约束(statistics_constraints): num_inputs × num_outputs
      与实验观察到的概率分布一致

    参数
    ----
    num_inputs : int
        输入态数量
    num_outputs : int
        输出区间数量
    cutoff : int
        Fock空间截断维度

    返回
    ----
    size_info : dict[str, int]
        包含策略数、变量数、约束数等信息
    """
    num_strategies = num_outputs ** (num_inputs + 1)
    variable_count = cutoff * num_outputs * num_strategies
    normalization_constraints = cutoff * num_strategies - num_strategies
    statistics_constraints = num_inputs * num_outputs
    return {
        "num_strategies": int(num_strategies),
        "variable_count": int(variable_count),
        "normalization_constraints": int(normalization_constraints),
        "statistics_constraints": int(statistics_constraints),
    }


def estimate_full_primal_problem_size(
    num_inputs: int,
    num_outputs: int,
    cutoff: int,
) -> dict[str, int]:
    """
    估算全矩阵原始问题的规模
    =========================

    物理背景
    --------
    与 estimate_primal_problem_size 不同，这里不假设POVM是Fock对角的。
    每个POVM元是一个 cutoff×cutoff 的半正定厄米矩阵，
    因此问题规模显著增大。

    估算公式
    --------
    - 策略数(num_strategies): num_outputs^(num_inputs+1)
    - 算子变量数(num_operator_variables): num_outputs × num_strategies
    - 厄米标量数(hermitian_scalar_count): num_operator_variables × cutoff²
      每个厄米矩阵有 cutoff² 个实参数（cutoff个实对角元 + cutoff(cutoff-1)/2 个复上三角元×2）
    - PSD约束(psd_constraints): num_operator_variables
      每个POVM元必须是半正定矩阵
    - 归一化约束: 每个策略的所有POVM元之和等于权重×单位矩阵
    - 统计约束: 与实验概率一致

    参数
    ----
    num_inputs : int
        输入态数量
    num_outputs : int
        输出区间数量
    cutoff : int
        Fock空间截断维度

    返回
    ----
    size_info : dict[str, int]
        包含策略数、算子变量数、厄米标量数、约束数等信息

    注意
    ----
    全矩阵问题的规模随cutoff二次增长，远大于对角问题（线性增长）。
    通常仅在cutoff较小时（如 ≤60）才能实际求解。
    """
    num_strategies = num_outputs ** (num_inputs + 1)
    num_operator_variables = num_outputs * num_strategies
    hermitian_scalar_count = num_operator_variables * cutoff * cutoff
    normalization_constraints = num_strategies
    statistics_constraints = num_inputs * num_outputs
    psd_constraints = num_operator_variables
    return {
        "num_strategies": int(num_strategies),
        "num_operator_variables": int(num_operator_variables),
        "hermitian_scalar_count": int(hermitian_scalar_count),
        "normalization_constraints": int(normalization_constraints),
        "statistics_constraints": int(statistics_constraints),
        "psd_constraints": int(psd_constraints),
    }


def _sum_matrices(expressions: list[cp.Expression], dimension: int) -> cp.Expression:
    """
    求和CVXPY矩阵表达式（安全处理空列表）
    ======================================

    用途
    ----
    在全矩阵原始问题中，需要对多个 cutoff×cutoff 的CVXPY矩阵变量求和。
    当列表为空时，返回零矩阵（保持维度一致）。

    参数
    ----
    expressions : list[cp.Expression]
        待求和的CVXPY矩阵表达式列表

    dimension : int
        矩阵维度（用于构造空列表时的零矩阵）

    返回
    ----
    total : cp.Expression
        求和结果，或 dimension×dimension 的零矩阵
    """
    if not expressions:
        return cp.Constant(np.zeros((dimension, dimension), dtype=complex))
    total = expressions[0]
    for expr in expressions[1:]:
        total = total + expr
    return total


def _instance_summary(instance: dict[str, Any]) -> dict[str, Any]:
    """
    提取实例数据的关键摘要信息
    ==========================

    用途
    ----
    从完整的 instance 字典中提取可序列化的摘要信息，
    用于嵌入到SDP/LP求解结果中，便于后续分析。

    提取的字段
    ----------
    - 选择的μ值列表和先验概率
    - 输入/输出数量、截断维度
    - 粗粒化边界和区间宽度
    - 原始概率的行求和（检查归一性）
    - 混合零列和全零列信息
    - 仅分布估计的猜测概率和对应最小熵

    参数
    ----
    instance : dict[str, Any]
        prepare_phaseinsensitive_instance 返回的完整实例字典

    返回
    ----
    summary : dict[str, Any]
        可JSON序列化的摘要字典
    """
    raw_p_guess = float(instance["distribution_only_p_guess_raw"])
    reg_p_guess = float(instance["distribution_only_p_guess"])
    return {
        "selected_mu_list": list(instance["selected_mu_list"]),
        "q_selected": np.asarray(instance["q_selected"], dtype=float).tolist(),
        "num_inputs": int(instance["num_inputs"]),
        "num_outputs": int(instance["num_outputs"]),
        "cutoff": int(instance["cutoff"]),
        "shift": int(instance["shift"]),
        "prob_floor": None if instance["prob_floor"] is None else float(instance["prob_floor"]),
        "regularized_entries": int(instance["regularized_entries"]),
        "probability_path": str(instance["probability_path"]),
        "edges": np.asarray(instance["edges"], dtype=int).tolist(),
        "block_widths": np.asarray(instance["block_widths"], dtype=int).tolist(),
        "row_sums_raw": np.asarray(instance["row_sums_raw"], dtype=float).tolist(),
        "mixed_zero_columns_raw": list(instance["mixed_zero_columns_raw"]),
        "all_zero_columns_raw": list(instance["all_zero_columns_raw"]),
        "has_mixed_zero_column_pathology": bool(instance["mixed_zero_columns_raw"]),
        "distribution_only_p_guess_raw": raw_p_guess,
        "distribution_only_H_min_raw": float(-np.log2(raw_p_guess)) if raw_p_guess > 0 else None,
        "distribution_only_p_guess": reg_p_guess,
        "distribution_only_H_min": float(-np.log2(reg_p_guess)) if reg_p_guess > 0 else None,
    }


def solve_phaseinsensitive_dual(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    求解相位不敏感模型的对偶线性规划
    =================================

    物理原理
    --------
    对偶问题给出猜测概率 P_guess 的上界。
    通过强对偶性（当原始问题可行时），对偶最优值等于原始最优值。

    SDP框架
    -------
    对手的目标是最大化猜测概率：
        P_guess = max_{策略e} Σ_x q(x) · Tr(ρ_x · M_{e(x),e})

    其中策略 e 是一个确定性函数 e: {输入} → {输出}，
    M_{c,e} 是与策略e关联的POVM元，ρ_x 是输入态。

    对偶变量 λ(x,c) 对应于统计约束（概率匹配条件），
    对偶目标是最小化 Σ_{x,c} P(x,c) · λ(x,c)。

    参数
    ----
    instance : dict[str, Any]
        prepare_phaseinsensitive_instance 返回的数据实例

    preferred_solver : str | None
        优先使用的求解器（如 'mosek', 'scs'），None则自动选择

    verbose : bool
        是否打印求解器详细输出

    返回
    ----
    result : dict[str, Any]
        包含以下字段：
        - 'route': 标识字符串
        - 'solver': 实际使用的求解器名称
        - 'status': 求解状态（'optimal', 'infeasible' 等）
        - 'p_guess': 最优猜测概率（对偶值）
        - 'H_min': 认证的最小熵 = -log₂(p_guess)
        - 'num_guess_functions': 策略总数
        以及实例摘要信息

    算法细节
    --------
    1. 枚举所有 num_outputs^num_inputs 个确定性策略
    2. 对每个策略，计算其对猜测概率的系数矩阵
    3. 构造对偶变量和约束
    4. 使用CVXPY求解线性规划
    5. 将最优值转换为最小熵

    数学公式
    --------
    对偶问题：
        min  Σ_{x,c} P(x,c) · λ(x,c)
        s.t. 对所有策略 e 和输出 c：
             Σ_{x: e(x)=c} q(x) · ρ_x(n) ≤ Σ_c λ(x,c) · ρ_x(n), ∀n
    """
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_diag = np.asarray(instance["rho_diag"], dtype=float)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_diag.shape[1]

    num_guess_funcs = num_outputs**num_inputs
    guess_funcs = np.array(list(iterproduct(range(num_outputs), repeat=num_inputs)), dtype=int)

    q_rho = q_selected[:, None] * rho_diag
    coeffs = np.zeros((num_guess_funcs, num_outputs, cutoff), dtype=float)
    for guess_index in range(num_guess_funcs):
        for output in range(num_outputs):
            mask = guess_funcs[guess_index, :] == output
            if np.any(mask):
                coeffs[guess_index, output, :] = q_rho[mask, :].sum(axis=0)

    dual_vars = cp.Variable((num_inputs, num_outputs))
    sigma_matrix = dual_vars.T @ rho_diag
    constraints: list[cp.Constraint] = []
    for guess_index in range(num_guess_funcs):
        expr_matrix = coeffs[guess_index] - sigma_matrix
        constraints.append(cp.sum(cp.max(expr_matrix, axis=0)) <= 0)

    objective = cp.Minimize(cp.sum(cp.multiply(probabilities, dual_vars)))
    problem = cp.Problem(objective, constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_phaseinsensitive_dual",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
        "num_guess_functions": int(num_guess_funcs),
    }
    result.update(_instance_summary(instance))
    return result


def solve_phaseinsensitive_primal(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
) -> dict[str, Any]:
    """
    求解Fock对角假设下的原始问题
    =============================

    物理原理
    --------
    原始问题直接构造对手的最优策略，给出猜测概率的下界。
    在Fock对角假设下，每个POVM元 M_{c,e} 是一个 cutoff 维的非负向量
    （而非 cutoff×cutoff 的半正定矩阵），大大降低了问题复杂度。

    关键假设：POVM是Fock对角的
    --------------------------
    由于输入态 ρ_μ 是Fock对角的，且探测器是相位不敏感的，
    可以证明最优POVM也是Fock对角的。这个假设在
    run_route4_nondiagonal_relaxation_check 中被数值验证。

    参数
    ----
    instance : dict[str, Any]
        prepare_phaseinsensitive_instance 返回的数据实例

    preferred_solver : str | None
        优先使用的求解器

    verbose : bool
        是否打印求解器详细输出

    max_primal_variables : int | None
        变量数上限（安全阀），超过则抛出 ValueError
        默认 None 表示不限制

    返回
    ----
    result : dict[str, Any]
        包含以下字段：
        - 'route': 标识字符串
        - 'solver', 'status': 求解器和状态
        - 'p_guess': 最优猜测概率
        - 'H_min': 认证的最小熵
        以及问题规模信息和实例摘要

    数学公式
    --------
    原始问题：
        max  Σ_{x,e} q(x) · ρ_x^T · m_{e(x),e}
        s.t. m_{c,e} ≥ 0                              （非负性）
             Σ_c m_{c,e}(n) = w_e(n), ∀n              （归一化：逐分量）
             ρ_x^T · (Σ_e m_{c,e}) = P(c|x), ∀x,c    （统计匹配）

    其中 m_{c,e} 是非负向量（Fock对角POVM元），w_e 是权重向量。
    """
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_diag = np.asarray(instance["rho_diag"], dtype=float)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_diag.shape[1]

    size_info = estimate_primal_problem_size(num_inputs, num_outputs, cutoff)
    if max_primal_variables is not None and size_info["variable_count"] > max_primal_variables:
        raise ValueError(
            "The requested primal instance is too large for the configured safeguard: "
            f"{size_info['variable_count']} > {max_primal_variables}."
        )
    if size_info["variable_count"] > 3_000_000:
        warnings.warn(
            "The primal route4 SDP is very large and may take a long time to canonicalize or solve. "
            f"Estimated variables: {size_info['variable_count']}.",
            stacklevel=2,
        )

    lambda_indices = np.array(
        list(iterproduct(range(num_outputs), repeat=num_inputs + 1)),
        dtype=int,
    )
    num_strategies = lambda_indices.shape[0]
    primal_elements = cp.Variable((cutoff, num_outputs, num_strategies), nonneg=True)

    objective_expr = 0
    for input_index in range(num_inputs):
        target_outputs = lambda_indices[:, input_index + 1]
        for output in range(num_outputs):
            strategy_ids = np.where(target_outputs == output)[0]
            if strategy_ids.size == 0:
                continue
            primal_sum = cp.sum(primal_elements[:, output, strategy_ids], axis=1)
            objective_expr += q_selected[input_index] * (rho_diag[input_index, :] @ primal_sum)

    constraints: list[cp.Constraint] = []
    sum_over_outputs = cp.sum(primal_elements, axis=1)
    for strategy_id in range(num_strategies):
        vec = sum_over_outputs[:, strategy_id]
        constraints.append(vec[1:] == vec[:-1])

    total_elements = cp.sum(primal_elements, axis=2)
    for input_index in range(num_inputs):
        for output in range(num_outputs):
            constraints.append(
                rho_diag[input_index, :] @ total_elements[:, output] == probabilities[input_index, output]
            )

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_phaseinsensitive_primal",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
    }
    result.update(size_info)
    result.update(_instance_summary(instance))
    return result


def solve_phaseinsensitive_full_primal(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """
    求解无Fock对角约束的全矩阵原始SDP
    ==================================

    物理原理
    --------
    与 solve_phaseinsensitive_primal 不同，这里不假设POVM是Fock对角的。
    每个POVM元是一个 cutoff×cutoff 的半正定厄米矩阵。

    这个更一般的问题用于验证对角假设的合理性：
    如果全矩阵问题的最优值与对角问题一致，说明对角假设不损失最优性。

    为什么对角假设通常成立
    ----------------------
    由于输入态 ρ_μ 是Fock对角的，只有POVM的对角元素参与目标函数的计算。
    非对角元素可以被任意设置而不影响目标值，但受限于PSD约束。
    对角化操作（将非对角元素置零）保持PSD性质和统计约束，
    因此总是可以找到一组对角POVM达到同等目标值。

    参数
    ----
    instance : dict[str, Any]
        prepare_phaseinsensitive_instance 返回的数据实例

    preferred_solver : str | None
        优先使用的求解器

    verbose : bool
        是否打印求解器详细输出

    max_hermitian_scalar_count : int | None
        厄米标量数上限（安全阀），防止问题规模过大

    返回
    ----
    result : dict[str, Any]
        包含以下字段：
        - 'route': 标识字符串
        - 'solver', 'status': 求解器和状态
        - 'p_guess': 最优猜测概率
        - 'H_min': 认证的最小熵
        以及问题规模信息和实例摘要

    注意
    ----
    全矩阵问题的计算代价远高于对角问题。
    cutoff=60, num_outputs=2, num_inputs=2 时，厄米标量数已达 4×8×3600=115200。
    仅用于验证目的，不建议作为日常计算工具。
    """
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_diag = np.asarray(instance["rho_diag"], dtype=float)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_diag.shape[1]

    size_info = estimate_full_primal_problem_size(num_inputs, num_outputs, cutoff)
    if (
        max_hermitian_scalar_count is not None
        and size_info["hermitian_scalar_count"] > max_hermitian_scalar_count
    ):
        raise ValueError(
            "The requested full primal instance is too large for the configured safeguard: "
            f"{size_info['hermitian_scalar_count']} > {max_hermitian_scalar_count}."
        )
    if size_info["hermitian_scalar_count"] > 400_000:
        warnings.warn(
            "The full-matrix route4 primal SDP is large and may take a long time to canonicalize "
            f"or solve. Estimated Hermitian scalar count: {size_info['hermitian_scalar_count']}.",
            stacklevel=2,
        )

    lambda_indices = np.array(
        list(iterproduct(range(num_outputs), repeat=num_inputs + 1)),
        dtype=int,
    )
    num_strategies = lambda_indices.shape[0]
    identity = np.eye(cutoff, dtype=complex)
    rho_matrices = [np.diag(rho_diag[input_index, :]).astype(complex) for input_index in range(num_inputs)]

    operators = {
        (output, strategy_id): cp.Variable((cutoff, cutoff), hermitian=True)
        for output in range(num_outputs)
        for strategy_id in range(num_strategies)
    }
    strategy_weights = cp.Variable(num_strategies, nonneg=True)

    objective_expr = 0
    for input_index in range(num_inputs):
        target_outputs = lambda_indices[:, input_index + 1]
        rho_matrix = rho_matrices[input_index]
        for output in range(num_outputs):
            strategy_ids = np.where(target_outputs == output)[0]
            if strategy_ids.size == 0:
                continue
            matrix_sum = _sum_matrices(
                [operators[(output, int(strategy_id))] for strategy_id in strategy_ids],
                cutoff,
            )
            objective_expr += q_selected[input_index] * cp.real(cp.trace(rho_matrix @ matrix_sum))

    constraints: list[cp.Constraint] = []
    for output in range(num_outputs):
        for strategy_id in range(num_strategies):
            constraints.append(operators[(output, strategy_id)] >> 0)

    for strategy_id in range(num_strategies):
        strategy_sum = _sum_matrices(
            [operators[(output, strategy_id)] for output in range(num_outputs)],
            cutoff,
        )
        constraints.append(strategy_sum == strategy_weights[strategy_id] * identity)

    total_elements = {
        output: _sum_matrices([operators[(output, strategy_id)] for strategy_id in range(num_strategies)], cutoff)
        for output in range(num_outputs)
    }
    for input_index in range(num_inputs):
        rho_matrix = rho_matrices[input_index]
        for output in range(num_outputs):
            constraints.append(
                cp.real(cp.trace(rho_matrix @ total_elements[output]))
                == probabilities[input_index, output]
            )

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_phaseinsensitive_full_primal",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
    }
    result.update(size_info)
    result.update(_instance_summary(instance))
    return result


def run_route4_dual(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    custom_edges: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Route 4 对偶求解的一站式入口
    ============================

    用途
    ----
    将数据准备和对偶求解合并为一个调用，是最常用的Route 4运行接口。
    适合日常计算和生产环境使用。

    参数
    ----
    num_outputs : int
        粗粒化输出区间数，默认6

    selected_mu_list : list[int] | tuple[int, ...]
        选用的光强列表，默认 [100, 120, 140]

    q_selected : list[float] | tuple[float, ...]
        先验概率分布，默认 [0.25, 0.25, 0.5]

    cutoff : int
        Fock截断维度，默认280

    prob_floor : float | None
        概率正则化下限，默认1e-12

    shift : int
        行偏移量，默认0

    preferred_solver : str | None
        优先求解器

    verbose : bool
        是否打印详细信息

    probability_path : str | Path | None
        概率数据文件路径

    返回
    ----
    result : dict[str, Any]
        solve_phaseinsensitive_dual 的完整结果
    """
    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        custom_edges=custom_edges,
    )
    return solve_phaseinsensitive_dual(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
    )


def run_route4_primal(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    max_primal_variables: int | None = 3_000_000,
    custom_edges: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Route 4 原始求解的一站式入口
    ============================

    用途
    ----
    将数据准备和Fock对角原始求解合并为一个调用。
    与对偶问题配合使用，通过强对偶性验证结果的正确性。

    参数
    ----
    num_outputs : int
        粗粒化输出区间数，默认6

    selected_mu_list : list[int] | tuple[int, ...]
        选用的光强列表，默认 [100, 120, 140]

    q_selected : list[float] | tuple[float, ...]
        先验概率分布，默认 [0.25, 0.25, 0.5]

    cutoff : int
        Fock截断维度，默认280

    prob_floor : float | None
        概率正则化下限

    shift : int
        行偏移量

    preferred_solver : str | None
        优先求解器

    verbose : bool
        是否打印详细信息

    probability_path : str | Path | None
        概率数据文件路径

    max_primal_variables : int | None
        变量数上限，默认3,000,000。超过此值会抛出异常。

    返回
    ----
    result : dict[str, Any]
        solve_phaseinsensitive_primal 的完整结果
    """
    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        custom_edges=custom_edges,
    )
    return solve_phaseinsensitive_primal(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )


def compare_route4_primal_dual(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    max_primal_variables: int | None = 3_000_000,
    custom_edges: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    对比对偶和原始问题的求解结果
    ============================

    用途
    ----
    同时运行对偶LP和原始SDP，比较两者的最优值。
    如果强对偶性成立，两者应该给出相同的 P_guess（在数值精度内）。
    这是对计算结果正确性的重要验证。

    物理意义
    --------
    - 对偶值 ≤ 真实 P_guess ≤ 原始值（弱对偶）
    - 强对偶：对偶值 = 原始值（当Slater条件满足时）
    - 如果两者吻合，说明结果可靠
    - 如果两者差距较大，说明数值精度不足或模型有误

    参数
    ----
    （与 run_route4_dual / run_route4_primal 相同的参数集）

    返回
    ----
    result : dict[str, Any]
        包含 'dual' 和 'primal' 两个子字典，以及实例摘要
    """

    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        custom_edges=custom_edges,
    )
    return {
        "route": "route4_phaseinsensitive_compare",
        "instance": _instance_summary(instance),
        "dual": solve_phaseinsensitive_dual(
            instance,
            preferred_solver=preferred_solver,
            verbose=verbose,
        ),
        "primal": solve_phaseinsensitive_primal(
            instance,
            preferred_solver=preferred_solver,
            verbose=verbose,
            max_primal_variables=max_primal_variables,
        ),
    }


def compare_route4_primal_full(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    max_primal_variables: int | None = 3_000_000,
    max_hermitian_scalar_count: int | None = 400_000,
    custom_edges: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    比较对角原始问题和全矩阵原始问题的结果
    =======================================

    用途
    ----
    验证Fock对角POVM假设的最优性。同时求解两个版本的原始问题，
    并比较猜测概率的差异。

    物理意义
    --------
    - 对角原始：假设最优POVM是Fock对角的（非负向量）
    - 全矩阵原始：不假设POVM结构（cutoff×cutoff 半正定矩阵）
    - 如果 P_guess(对角) = P_guess(全矩阵)，则对角假设不损失最优性
    - 这是因为Fock对角输入态只能"感知"POVM的对角部分

    参数
    ----
    （与前面函数相同的参数集，加上 max_hermitian_scalar_count 安全阀）

    max_hermitian_scalar_count : int | None
        全矩阵问题的厄米标量数上限，默认400,000

    返回
    ----
    result : dict[str, Any]
        包含 'diagonal_primal' 和 'full_primal' 两个子字典，
        以及 p_guess_abs_gap（猜测概率的绝对差）和 H_min_abs_gap（最小熵的绝对差）
    """

    instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        custom_edges=custom_edges,
    )
    diagonal_primal = solve_phaseinsensitive_primal(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )
    full_primal = solve_phaseinsensitive_full_primal(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )
    diagonal_p_guess = diagonal_primal.get("p_guess")
    full_p_guess = full_primal.get("p_guess")
    diagonal_h_min = diagonal_primal.get("H_min")
    full_h_min = full_primal.get("H_min")
    return {
        "route": "route4_phaseinsensitive_primal_full_compare",
        "instance": _instance_summary(instance),
        "diagonal_primal": diagonal_primal,
        "full_primal": full_primal,
        "p_guess_abs_gap": None
        if diagonal_p_guess is None or full_p_guess is None
        else float(abs(diagonal_p_guess - full_p_guess)),
        "H_min_abs_gap": None
        if diagonal_h_min is None or full_h_min is None
        else float(abs(diagonal_h_min - full_h_min)),
    }


def run_route4_nondiagonal_relaxation_check(
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = 3_000_000,
    max_hermitian_scalar_count: int | None = 400_000,
    probability_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    在多个小规模实例上验证对角假设的最优性
    =======================================

    物理原理
    --------
    这是Route 4理论框架的关键验证步骤。
    在Fock对角输入态的假设下，POVM的非对角元素不参与任何可观测量，
    因此对角化POVM不应改变最优猜测概率。

    验证方法
    --------
    在4个精心设计的小规模测试用例上，同时运行对角原始和全矩阵原始：
    - case_a: 2输入2输出，cutoff=40，μ={0,20}（验证低维情况）
    - case_b: 2输入2输出，cutoff=60，μ={0,20}（增大截断）
    - case_c: 2输入3输出，cutoff=60（增加输出维度）
    - case_d: 3输入2输出，cutoff=100（增加输入维度）

    参数
    ----
    preferred_solver : str | None
        优先求解器

    verbose : bool
        是否打印详细信息

    max_primal_variables : int | None
        对角原始的变量数上限

    max_hermitian_scalar_count : int | None
        全矩阵原始的厄米标量数上限

    probability_path : str | Path | None
        概率数据文件路径

    返回
    ----
    result : dict[str, Any]
        包含各测试用例的结果、最大猜测概率差距，以及结论性描述

    预期结论
    --------
    对于所有测试用例，对角原始和全矩阵原始的 P_guess 应在数值精度内一致
    （绝对差 < 1e-6），证实Fock对角假设的最优性。
    """
    cases = [
        {
            "name": "case_a_two_inputs_two_outputs_cutoff40_mu0_20_infeasible",
            "num_outputs": 2,
            "selected_mu_list": [0, 20],
            "q_selected": [0.5, 0.5],
            "cutoff": 40,
            "prob_floor": 1e-12,
        },
        {
            "name": "case_b_two_inputs_two_outputs_cutoff60_mu0_20_optimal",
            "num_outputs": 2,
            "selected_mu_list": [0, 20],
            "q_selected": [0.5, 0.5],
            "cutoff": 60,
            "prob_floor": 1e-12,
        },
        {
            "name": "case_c_two_inputs_three_outputs_cutoff60_mu0_20_optimal",
            "num_outputs": 3,
            "selected_mu_list": [0, 20],
            "q_selected": [0.5, 0.5],
            "cutoff": 60,
            "prob_floor": 1e-12,
        },
        {
            "name": "case_d_three_inputs_two_outputs_cutoff100_mu0_20_40_optimal",
            "num_outputs": 2,
            "selected_mu_list": [0, 20, 40],
            "q_selected": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "cutoff": 100,
            "prob_floor": 1e-12,
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            comparison = compare_route4_primal_full(
                num_outputs=int(case["num_outputs"]),
                selected_mu_list=list(case["selected_mu_list"]),
                q_selected=list(case["q_selected"]),
                cutoff=int(case["cutoff"]),
                prob_floor=float(case["prob_floor"]),
                preferred_solver=preferred_solver,
                verbose=verbose,
                probability_path=probability_path,
                max_primal_variables=max_primal_variables,
                max_hermitian_scalar_count=max_hermitian_scalar_count,
            )
            results.append(
                {
                    "name": case["name"],
                    **comparison,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": case["name"],
                    "status": "error",
                    "error": str(exc),
                    **case,
                }
            )

    solved_cases = [
        item
        for item in results
        if np.isfinite(item.get("diagonal_primal", {}).get("p_guess", np.nan))
        and np.isfinite(item.get("full_primal", {}).get("p_guess", np.nan))
    ]
    max_gap = None
    if solved_cases:
        max_gap = float(max(item["p_guess_abs_gap"] for item in solved_cases if item["p_guess_abs_gap"] is not None))

    return {
        "route": "route4_phaseinsensitive_nondiagonal_relaxation_check",
        "preferred_solver": preferred_solver,
        "max_primal_variables": max_primal_variables,
        "max_hermitian_scalar_count": max_hermitian_scalar_count,
        "cases": results,
        "num_cases": len(results),
        "num_solved_cases": len(solved_cases),
        "max_p_guess_abs_gap": max_gap,
        "conclusion": (
            "For these diagonal-input route4 instances, the unrestricted full-matrix primal "
            "matches the diagonal primal up to numerical tolerance."
        ),
    }


def run_route4_diagonal_projection_invariance_check(
    seed: int = 7,
    num_trials: int = 4,
) -> dict[str, Any]:
    """
    数值验证对角投影保持所有Route 4线性泛函不变
    ============================================

    物理原理
    --------
    这是验证Fock对角假设的另一种方法，不需要运行SDP求解器。

    核心命题：如果输入态是Fock对角的（ρ = Σ p(n)|n⟩⟨n|），那么：
    1. Tr(ρ · M) = Tr(ρ · M_diag) 对任何算符 M 成立
    2. 将任意POVM的非对角元素置零后，所有统计量和目标函数值不变

    验证方法
    --------
    1. 随机生成满足POVM完备性的非对角算符族
    2. 对每个算符做对角投影（保留对角元素，非对角元素置零）
    3. 比较投影前后的：
       - 统计量（Tr(ρ · M)）
       - 目标函数值（Σ q(x) · Tr(ρ_x · M_{e(x),e})）
       - 完备性残差

    参数
    ----
    seed : int
        随机数种子，用于可重复性，默认7

    num_trials : int
        每个测试用例的随机试验次数，默认4

    返回
    ----
    result : dict[str, Any]
        包含各测试用例的结果：
        - max_stats_gap: 投影前后统计量的最大差异（预期 ≈ 0）
        - max_objective_gap: 投影前后目标函数的最大差异（预期 ≈ 0）
        - max_completeness_residual: POVM完备性的最大残差
        - min_operator_eigenvalue: 最小算符特征值（验证PSD性质）
        以及结论性描述

    随机算符的构造
    --------------
    1. 对每个策略，生成随机权重（Dirichlet分布）乘以单位矩阵
    2. 可选地添加随机对称扰动（保持PSD）
    3. 验证完备性（所有输出的POVM元之和等于权重×单位矩阵）
    """
    rng = np.random.default_rng(seed)
    cases = [
        {
            "name": "default_like_three_inputs_four_outputs",
            "selected_mu_list": [100, 120, 140],
            "q_selected": [0.25, 0.25, 0.5],
            "num_outputs": 4,
            "cutoff": 16,
            "prob_floor": 1e-12,
        },
        {
            "name": "low_intensity_two_inputs_two_outputs",
            "selected_mu_list": [0, 20],
            "q_selected": [0.5, 0.5],
            "num_outputs": 2,
            "cutoff": 24,
            "prob_floor": 1e-12,
        },
    ]

    case_results: list[dict[str, Any]] = []
    for case in cases:
        instance = prepare_phaseinsensitive_instance(
            num_outputs=int(case["num_outputs"]),
            selected_mu_list=list(case["selected_mu_list"]),
            q_selected=list(case["q_selected"]),
            cutoff=int(case["cutoff"]),
            prob_floor=float(case["prob_floor"]),
        )
        rho_diag = np.asarray(instance["rho_diag"], dtype=float)
        q_selected = np.asarray(instance["q_selected"], dtype=float)
        num_inputs, cutoff = rho_diag.shape
        num_outputs = int(instance["num_outputs"])
        lambda_indices = np.array(
            list(iterproduct(range(num_outputs), repeat=num_inputs + 1)),
            dtype=int,
        )
        num_strategies = lambda_indices.shape[0]
        rho_matrices = [np.diag(rho_diag[input_index, :]) for input_index in range(num_inputs)]
        identity = np.eye(cutoff)

        max_stats_gap = 0.0
        max_objective_gap = 0.0
        max_completeness_residual = 0.0
        min_operator_eigenvalue = np.inf

        for _ in range(num_trials):
            operators = np.zeros((num_outputs, num_strategies, cutoff, cutoff), dtype=float)
            strategy_scales = rng.uniform(0.6, 1.4, size=num_strategies)
            for strategy_id in range(num_strategies):
                weights = rng.dirichlet(np.ones(num_outputs))
                for output in range(num_outputs):
                    operators[output, strategy_id, :, :] = strategy_scales[strategy_id] * weights[output] * identity

                if num_outputs >= 2:
                    random_matrix = rng.standard_normal((cutoff, cutoff))
                    perturbation = 0.5 * (random_matrix + random_matrix.T)
                    np.fill_diagonal(perturbation, 0.0)
                    perturbation_norm = np.linalg.norm(perturbation, ord=2)
                    if perturbation_norm > 0.0:
                        scale_budget = min(
                            operators[0, strategy_id, 0, 0],
                            operators[1, strategy_id, 0, 0],
                        )
                        epsilon = 0.25 * scale_budget / perturbation_norm
                        operators[0, strategy_id, :, :] += epsilon * perturbation
                        operators[1, strategy_id, :, :] -= epsilon * perturbation

                strategy_sum = operators[:, strategy_id, :, :].sum(axis=0)
                max_completeness_residual = max(
                    max_completeness_residual,
                    float(np.max(np.abs(strategy_sum - strategy_scales[strategy_id] * identity))),
                )
                for output in range(num_outputs):
                    min_operator_eigenvalue = min(
                        min_operator_eigenvalue,
                        float(np.linalg.eigvalsh(operators[output, strategy_id, :, :]).min()),
                    )

            diagonal_operators = np.zeros_like(operators)
            diagonal_indices = np.arange(cutoff)
            diagonal_operators[:, :, diagonal_indices, diagonal_indices] = operators[
                :, :, diagonal_indices, diagonal_indices
            ]

            full_probabilities = np.zeros((num_inputs, num_outputs), dtype=float)
            diagonal_probabilities = np.zeros((num_inputs, num_outputs), dtype=float)
            for input_index in range(num_inputs):
                rho_matrix = rho_matrices[input_index]
                for output in range(num_outputs):
                    total_full = operators[output, :, :, :].sum(axis=0)
                    total_diagonal = diagonal_operators[output, :, :, :].sum(axis=0)
                    full_probabilities[input_index, output] = float(np.trace(rho_matrix @ total_full))
                    diagonal_probabilities[input_index, output] = float(
                        np.trace(rho_matrix @ total_diagonal)
                    )
            max_stats_gap = max(
                max_stats_gap,
                float(np.max(np.abs(full_probabilities - diagonal_probabilities))),
            )

            full_objective = 0.0
            diagonal_objective = 0.0
            for input_index in range(num_inputs):
                rho_matrix = rho_matrices[input_index]
                target_outputs = lambda_indices[:, input_index + 1]
                for output in range(num_outputs):
                    strategy_ids = np.where(target_outputs == output)[0]
                    if strategy_ids.size == 0:
                        continue
                    full_matrix = operators[output, strategy_ids, :, :].sum(axis=0)
                    diagonal_matrix = diagonal_operators[output, strategy_ids, :, :].sum(axis=0)
                    full_objective += q_selected[input_index] * float(np.trace(rho_matrix @ full_matrix))
                    diagonal_objective += q_selected[input_index] * float(
                        np.trace(rho_matrix @ diagonal_matrix)
                    )
            max_objective_gap = max(max_objective_gap, float(abs(full_objective - diagonal_objective)))

        case_results.append(
            {
                "name": case["name"],
                "selected_mu_list": list(case["selected_mu_list"]),
                "q_selected": list(case["q_selected"]),
                "num_outputs": int(case["num_outputs"]),
                "cutoff": int(case["cutoff"]),
                "num_trials": int(num_trials),
                "num_strategies": int(num_strategies),
                "max_stats_gap": max_stats_gap,
                "max_objective_gap": max_objective_gap,
                "max_completeness_residual": max_completeness_residual,
                "min_operator_eigenvalue": min_operator_eigenvalue,
            }
        )

    return {
        "route": "route4_phaseinsensitive_diagonal_projection_invariance_check",
        "seed": int(seed),
        "num_trials": int(num_trials),
        "cases": case_results,
        "conclusion": (
            "For randomly generated non-diagonal PSD operator families that preserve POVM completeness, "
            "all route4 statistics and route4-style objective values are unchanged after diagonal projection."
        ),
    }


def sweep_route4_outputs(
    output_values: list[int],
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    扫描不同粗粒化输出数的Route 4结果
    ==================================

    用途
    ----
    系统地改变粗粒化的输出区间数，观察 H_min 随 num_outputs 的变化趋势。
    这有助于选择最优的粗粒化策略。

    物理直觉
    --------
    - num_outputs 太小（如2）：粗粒化丢失了太多信息，H_min较低
    - num_outputs 适中（如4-8）：在信息保留和问题规模之间取得平衡
    - num_outputs 太大：SDP问题规模指数增长，求解困难，但H_min更高

    参数
    ----
    output_values : list[int]
        要测试的输出区间数列表，如 [2, 3, 4, 6, 8]

    其余参数：与 run_route4_dual 相同

    返回
    ----
    results : list[dict[str, Any]]
        每个输出区间数对应的对偶求解结果列表
    """

    return [
        run_route4_dual(
            num_outputs=num_outputs,
            selected_mu_list=selected_mu_list,
            q_selected=q_selected,
            cutoff=cutoff,
            prob_floor=prob_floor,
            shift=shift,
            preferred_solver=preferred_solver,
            verbose=verbose,
            probability_path=probability_path,
        )
        for num_outputs in output_values
    ]


def search_route4_triplets(
    num_outputs: int,
    subset_size: int = 3,
    certify_top_k: int = 3,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
) -> dict[str, Any]:
    """
    搜索最优的光强子集组合
    ======================

    物理原理
    --------
    不同光强组合产生的探测器统计差异不同，直接影响可认证的随机数量。
    本函数在所有可能的 C(|full_mu|, subset_size) 种组合中搜索最优子集。

    搜索策略：两阶段方法
    --------------------
    阶段1（筛选）：
    - 对所有 C(9,3) = 84 种三元组，仅计算分布级别的猜测概率
      P_guess = Σ q(x) · max_c P(c|x)
    - 这不需要运行SDP，计算量极小

    阶段2（认证）：
    - 按 P_guess 升序排列（P_guess越小 → H_min越高 → 越有希望）
    - 对前 certify_top_k 个候选运行完整的对偶LP认证

    参数
    ----
    num_outputs : int
        粗粒化输出区间数

    subset_size : int
        光强子集大小，默认3（三元组）

    certify_top_k : int
        认证前k个最有希望的候选，默认3

    cutoff : int
        Fock截断维度

    prob_floor : float | None
        概率正则化下限

    shift : int
        行偏移量

    preferred_solver : str | None
        优先求解器

    verbose : bool
        是否打印详细信息

    probability_path : str | Path | None
        概率数据文件路径

    full_mu : list[int] | tuple[int, ...]
        全部可用光强列表

    返回
    ----
    result : dict[str, Any]
        包含：
        - 'num_candidates': 候选子集总数
        - 'best_distribution_only': 分布级别最优的子集
        - 'top_distribution_only': 前k个分布级别最优的子集
        - 'certified': 经过完整SDP认证的结果列表
    """
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    candidates: list[dict[str, Any]] = []
    for subset in combinations(list(full_mu), subset_size):
        q_selected = np.full(subset_size, 1.0 / subset_size)
        instance = prepare_phaseinsensitive_instance(
            num_outputs=num_outputs,
            selected_mu_list=list(subset),
            q_selected=q_selected.tolist(),
            cutoff=cutoff,
            prob_floor=prob_floor,
            shift=shift,
            probability_path=probability_path,
            full_mu=full_mu,
        )
        candidates.append(
            {
                "selected_mu_list": list(subset),
                "q_selected": q_selected.tolist(),
                "distribution_only_p_guess": float(instance["distribution_only_p_guess"]),
                "distribution_only_H_min": float(-np.log2(instance["distribution_only_p_guess"])),
                "distribution_only_p_guess_raw": float(instance["distribution_only_p_guess_raw"]),
                "distribution_only_H_min_raw": float(-np.log2(instance["distribution_only_p_guess_raw"])),
                "mixed_zero_columns_raw": list(instance["mixed_zero_columns_raw"]),
            }
        )

    candidates.sort(key=lambda item: item["distribution_only_p_guess"])
    certified: list[dict[str, Any]] = []
    for item in candidates[: max(certify_top_k, 0)]:
        certified.append(
            run_route4_dual(
                num_outputs=num_outputs,
                selected_mu_list=item["selected_mu_list"],
                q_selected=item["q_selected"],
                cutoff=cutoff,
                prob_floor=prob_floor,
                shift=shift,
                preferred_solver=preferred_solver,
                verbose=verbose,
                probability_path=probability_path,
            )
        )

    return {
        "route": "route4_phaseinsensitive_subset_search",
        "num_outputs": num_outputs,
        "subset_size": subset_size,
        "certify_top_k": certify_top_k,
        "num_candidates": len(candidates),
        "best_distribution_only": candidates[0] if candidates else None,
        "top_distribution_only": candidates[: min(len(candidates), max(certify_top_k, 5))],
        "certified": certified,
    }


def search_route4_contiguous_edges(
    num_outputs: int,
    selected_mu_list: list[int] | tuple[int, ...],
    q_selected: list[float] | tuple[float, ...],
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
    certify_top_k: int = 5,
    record_top_k: int = 10,
    min_bin_width: int = 1,
) -> dict[str, Any]:
    """
    在固定输入窗口上搜索 contiguous coarse-graining
    ================================================

    说明
    ----
    该函数保持原始 route4 的物理主线不变：
    - 输入仍取实验真实的 μ；
    - trusted state 仍只使用 Fock 对角分布；
    - 正式认证仍调用 route4 对偶问题。

    新增的唯一自由度是把 256 个原始 bin 按任意连续边界合并，
    先用 distribution-only 指标筛选，再对前若干候选做 formal 认证。
    """
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive.")
    if min_bin_width <= 0:
        raise ValueError("min_bin_width must be positive.")

    selected_mu = list(selected_mu_list)
    full_mu_list = list(full_mu)
    if len(selected_mu) == 0:
        raise ValueError("At least one input state is required.")
    if any(mu not in full_mu_list for mu in selected_mu):
        raise ValueError(f"selected_mu_list must be a subset of {full_mu_list}.")

    q = np.asarray(q_selected, dtype=float).reshape(-1)
    if q.size != len(selected_mu):
        raise ValueError("q_selected must have the same length as selected_mu_list.")
    if np.any(q < 0):
        raise ValueError("q_selected must be non-negative.")
    if float(q.sum()) <= 0.0:
        raise ValueError("q_selected must sum to a positive value.")
    q = q / q.sum()

    probability_table = load_probability_data(probability_path)
    selected_indices = [full_mu_list.index(mu) + shift for mu in selected_mu]
    if probability_table.shape[0] <= max(selected_indices):
        raise ValueError("Probability table does not contain the requested shifted rows.")
    raw_rows = np.asarray(probability_table[selected_indices, :], dtype=float)
    num_raw_bins = int(raw_rows.shape[1])
    if num_outputs > num_raw_bins:
        raise ValueError(
            f"num_outputs={num_outputs} exceeds the available raw bins ({num_raw_bins})."
        )

    prefix = np.concatenate(
        [np.zeros((raw_rows.shape[0], 1), dtype=float), np.cumsum(raw_rows, axis=1)],
        axis=1,
    )
    total_candidates = int(comb(num_raw_bins - 1, num_outputs - 1))
    keep_top_k = max(int(certify_top_k), int(record_top_k), 1)
    ranked_candidates: list[dict[str, Any]] = []
    evaluated_candidates = 0

    def maybe_keep_candidate(candidate: dict[str, Any]) -> None:
        if len(ranked_candidates) < keep_top_k:
            ranked_candidates.append(candidate)
            return
        worst_index = max(
            range(len(ranked_candidates)),
            key=lambda idx: ranked_candidates[idx]["distribution_only_p_guess"],
        )
        if (
            candidate["distribution_only_p_guess"]
            < ranked_candidates[worst_index]["distribution_only_p_guess"]
        ):
            ranked_candidates[worst_index] = candidate

    boundary_iter = [tuple()] if num_outputs == 1 else combinations(range(1, num_raw_bins), num_outputs - 1)
    for boundaries in boundary_iter:
        edges = np.array((0, *boundaries, num_raw_bins), dtype=int)
        block_widths = np.diff(edges)
        if int(block_widths.min()) < min_bin_width:
            continue
        evaluated_candidates += 1

        coarse_raw = prefix[:, edges[1:]] - prefix[:, edges[:-1]]
        raw_p_guess = float(distribution_only_guessing_probability(coarse_raw, q))

        regularized_probabilities = coarse_raw.copy()
        regularized_entries = 0
        if prob_floor is not None and prob_floor > 0:
            regularized_entries = int((regularized_probabilities == 0.0).sum())
            regularized_probabilities = np.maximum(regularized_probabilities, prob_floor)
            regularized_probabilities = regularized_probabilities / regularized_probabilities.sum(
                axis=1, keepdims=True
            )
        p_guess = float(distribution_only_guessing_probability(regularized_probabilities, q))

        maybe_keep_candidate(
            {
                "edges": edges.tolist(),
                "block_widths": block_widths.tolist(),
                "distribution_only_p_guess_raw": raw_p_guess,
                "distribution_only_H_min_raw": float(-np.log2(raw_p_guess)),
                "distribution_only_p_guess": p_guess,
                "distribution_only_H_min": float(-np.log2(p_guess)),
                "regularized_entries": regularized_entries,
                "mixed_zero_columns_raw": [
                    int(column)
                    for column in range(num_outputs)
                    if np.any(coarse_raw[:, column] == 0.0)
                    and np.any(coarse_raw[:, column] > 0.0)
                ],
                "all_zero_columns_raw": [
                    int(column)
                    for column in range(num_outputs)
                    if np.all(coarse_raw[:, column] == 0.0)
                ],
            }
        )

    ranked_candidates.sort(key=lambda item: item["distribution_only_p_guess"])

    certified: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked_candidates[: max(certify_top_k, 0)], start=1):
        result = run_route4_dual(
            num_outputs=num_outputs,
            selected_mu_list=selected_mu,
            q_selected=q.tolist(),
            cutoff=cutoff,
            prob_floor=prob_floor,
            shift=shift,
            preferred_solver=preferred_solver,
            verbose=verbose,
            probability_path=probability_path,
            custom_edges=candidate["edges"],
        )
        result["distribution_screening_rank"] = rank
        certified.append(result)

    solved_certified = [
        item
        for item in certified
        if item.get("status") in ("optimal", "optimal_inaccurate") and item.get("H_min") is not None
    ]
    best_certified = max(solved_certified, key=lambda item: float(item["H_min"]), default=None)

    return {
        "route": "route4_phaseinsensitive_contiguous_search",
        "selected_mu_list": selected_mu,
        "q_selected": q.tolist(),
        "num_outputs": int(num_outputs),
        "cutoff": int(cutoff),
        "prob_floor": None if prob_floor is None else float(prob_floor),
        "shift": int(shift),
        "probability_path": str(
            Path(probability_path) if probability_path is not None else _default_probability_path()
        ),
        "num_raw_bins": num_raw_bins,
        "total_candidates": total_candidates,
        "evaluated_candidates": evaluated_candidates,
        "min_bin_width": int(min_bin_width),
        "certify_top_k": int(certify_top_k),
        "record_top_k": int(record_top_k),
        "best_distribution_only": ranked_candidates[0] if ranked_candidates else None,
        "top_distribution_only": ranked_candidates,
        "certified": certified,
        "best_certified": best_certified,
    }
