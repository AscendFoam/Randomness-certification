"""Route4-ex 的核心建模与求解接口。

本文件承载 route4-ex 的绝大部分核心逻辑，职责包括：

1. 构造非对角 trusted coherent inputs；
2. 构造 toy / APD-like / external-table 三类概率实例；
3. 对原始概率表做 coarse-graining；
4. 建立 diagonal primal 与 full primal 两类认证问题；
5. 提供 run/compare 风格的统一封装，供 CLI 与搜索脚本直接调用。

其它脚本如 `main.py`、`external_scan.py`、`high_output_local_refine.py`、
`pathology_boundary_scan.py` 等，基本都把这里的函数当成底层能力层。
因此本文件既是 route4-ex 的“核心算法库”，也是实验分析脚本共享的
统一数据与求解接口。
"""

from __future__ import annotations

import json
import math
import warnings
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.io import loadmat
from scipy.linalg import expm
from scipy.special import gammaln

from ..common import coherent_state, create, density_from_ket, destroy, solve_cvxpy_problem
from ..route4.phaseinsensitive import (
    build_equal_cover_edges,
    distribution_only_guessing_probability,
    estimate_full_primal_problem_size,
    estimate_primal_problem_size,
)

DEFAULT_ALPHA_VALUES = [0.6 + 0.0j, 0.0 + 0.6j, -0.6 + 0.0j]
DEFAULT_Q = [1.0, 1.0, 1.0]
DEFAULT_CUTOFF = 12
DEFAULT_PROBE_ALPHA = 0.4 + 0.4j
DEFAULT_DISPLACEMENT_ALPHA = 0.35 + 0.35j
DEFAULT_PROB_FLOOR = 1e-12
DEFAULT_NUM_OUTPUTS = 4
DEFAULT_RAW_NUM_BINS = 16
DEFAULT_DETECTION_EFFICIENCY = 0.6
DEFAULT_DARK_COUNT_MEAN = 0.02


def _clean_value(value: Any) -> Any:
    """把 NumPy 类型递归转成适合 JSON 序列化的基础类型。

    功能：
        供 `json.dumps(..., default=_clean_value)` 调用，把 NumPy 数组与标量
        转成 Python 原生 list / int / float。

    参数：
        value：待清洗的任意对象。

    返回：
        适合 JSON 序列化的对象；若对象本身无需处理，则原样返回。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def result_to_json(result: Any) -> str:
    """把结果对象格式化为中文报告友好的 JSON 字符串。

    功能：
        统一 route4-ex 各命令行入口的输出格式。

    参数：
        result：任意可被 `_clean_value` 递归处理的结果对象。

    返回：
        带缩进、保留中文、适合直接打印或写文件的 JSON 字符串。
    """
    return json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value)


def _serialize_complex(alpha: complex) -> dict[str, float]:
    """把单个复振幅转成便于展示的字典。

    功能：
        将相干态振幅 `alpha` 分解为实部、虚部、模长与相位，
        方便结果文件与报告直接引用。

    参数：
        alpha：单个复数振幅。

    返回：
        包含 `real`、`imag`、`abs`、`phase` 的字典。
    """
    return {
        "real": float(np.real(alpha)),
        "imag": float(np.imag(alpha)),
        "abs": float(abs(alpha)),
        "phase": float(np.angle(alpha)),
    }


def _serialize_complex_list(alpha_values: list[complex]) -> list[dict[str, float]]:
    """批量序列化一组复振幅。"""
    return [_serialize_complex(alpha) for alpha in alpha_values]


def _sum_matrices(expressions: list[cp.Expression], dimension: int) -> cp.Expression:
    """安全地求一组 CVXPY 矩阵表达式之和。

    功能：
        统一处理“列表为空”和“列表非空”两种情况，避免在构造 SDP 目标或约束时
        手工判断。

    参数：
        expressions：待求和的矩阵表达式列表。
        dimension：矩阵维数；当列表为空时用来创建零矩阵常量。

    返回：
        CVXPY 矩阵表达式，表示所有输入表达式之和。
    """
    if not expressions:
        return cp.Constant(np.zeros((dimension, dimension), dtype=complex))
    total = expressions[0]
    for expr in expressions[1:]:
        total = total + expr
    return total


def _validate_alpha_and_q(
    alpha_values: list[complex],
    q_selected: list[float],
) -> tuple[list[complex], np.ndarray]:
    """校验输入振幅列表和生成轮权重，并把权重归一化。

    功能：
        route4-ex 里许多接口都接受 `alpha_values` 与 `q_selected`。
        该函数负责在最底层统一做合法性检查，避免上层重复写校验逻辑。

    参数：
        alpha_values：trusted input 对应的相干态振幅列表。
        q_selected：每个输入在生成轮目标函数中的权重。

    返回：
        二元组 `(alpha_values, q)`：
        - `alpha_values`：转成 `complex` 后的振幅列表；
        - `q`：归一化后的 NumPy 浮点数组。

    异常：
        当输入为空、长度不匹配、存在负权重或总权重非正时抛出 `ValueError`。
    """
    alpha_values = [complex(alpha) for alpha in alpha_values]
    if len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")

    q = np.asarray(q_selected, dtype=float).reshape(-1)
    if q.size != len(alpha_values):
        raise ValueError("q_selected must have the same length as alpha_values.")
    if np.any(q < 0):
        raise ValueError("q_selected must be non-negative.")
    if float(q.sum()) <= 0.0:
        raise ValueError("q_selected must sum to a positive value.")
    q = q / q.sum()
    return alpha_values, q


def build_coherent_density_matrices(alpha_values: list[complex], cutoff: int) -> np.ndarray:
    """构造一组截断相干态的密度矩阵。

    功能：
        对每个复振幅 `alpha` 生成 Fock 截断维数为 `cutoff` 的相干态 ket，
        再转成密度矩阵 `|alpha><alpha|`。

    参数：
        alpha_values：相干态振幅列表。
        cutoff：Fock 截断维数。

    返回：
        形状为 `(num_inputs, cutoff, cutoff)` 的复矩阵数组。

    说明：
        这是 route4-ex 与原始 route4 的重要区别之一：
        这里构造的是完整非对角的 trusted coherent states，而不是只取 Fock
        基对角部分。
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    if len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")

    density_matrices = []
    for alpha in alpha_values:
        ket = coherent_state(cutoff, complex(alpha))
        density_matrices.append(density_from_ket(ket))
    return np.asarray(density_matrices, dtype=complex)


def build_binary_coherent_projector_povm(cutoff: int, probe_alpha: complex) -> np.ndarray:
    """构造 toy 二元 coherent-projector POVM。

    功能：
        用投影算符 `|probe><probe|` 及其补算符 `I-|probe><probe|` 形成
        一个两输出 toy POVM，主要用于快速结构验证。

    参数：
        cutoff：Fock 截断维数。
        probe_alpha：探针相干态振幅。

    返回：
        形状为 `(2, cutoff, cutoff)` 的 POVM 元数组。
    """
    probe_ket = coherent_state(cutoff, probe_alpha)
    projector = density_from_ket(probe_ket)
    identity = np.eye(cutoff, dtype=complex)
    povm = np.stack([projector, identity - projector], axis=0)
    return povm


def build_displacement_unitary(cutoff: int, displacement_alpha: complex) -> np.ndarray:
    """构造有限维位移算符 `D(alpha)` 的矩阵表示。

    功能：
        根据 `D(alpha)=exp(alpha a^† - alpha^* a)` 在截断 Fock 空间中生成
        位移酉算符，用于把对角 Fock 计数 POVM 平移到 IQ / displaced-count
        场景。

    参数：
        cutoff：Fock 截断维数。
        displacement_alpha：位移振幅。

    返回：
        `cutoff x cutoff` 的复矩阵。
    """
    a = destroy(cutoff)
    adag = create(cutoff)
    generator = displacement_alpha * adag - np.conjugate(displacement_alpha) * a
    return expm(generator)


def build_displaced_fock_povm(cutoff: int, displacement_alpha: complex) -> np.ndarray:
    """构造位移后的 Fock 投影测量。

    功能：
        先生成标准 Fock 基投影 `|n><n|`，再用位移酉算符做共轭变换，
        得到 displaced-number measurement。

    参数：
        cutoff：Fock 截断维数。
        displacement_alpha：位移振幅。

    返回：
        形状为 `(cutoff, cutoff, cutoff)` 的 POVM 数组，其中第一个轴是输出 n。
    """
    displacement = build_displacement_unitary(cutoff, displacement_alpha)
    projectors: list[np.ndarray] = []
    for photon_number in range(cutoff):
        basis = np.zeros(cutoff, dtype=complex)
        basis[photon_number] = 1.0
        projector = density_from_ket(basis)
        projectors.append(displacement.conj().T @ projector @ displacement)
    return np.asarray(projectors, dtype=complex)


def build_apd_count_povm(
    cutoff: int,
    raw_num_bins: int,
    detection_efficiency: float,
    dark_count_mean: float,
) -> np.ndarray:
    """构造 APD 风格计数 POVM 的对角近似版本。

    功能：
        在 Fock 基下，把探测效率与暗计数纳入计数模型，生成
        `raw_num_bins` 个原始输出 bin 的对角 POVM。

    逻辑：
        - 对每个输入光子数 `n`；
        - 枚举实际透过并被探测到的光子数；
        - 与暗计数的 Poisson 分布卷积；
        - 最后把超过 `raw_num_bins-1` 的尾部概率并入最后一个溢出 bin。

    参数：
        cutoff：Fock 截断维数。
        raw_num_bins：原始计数直方图的输出 bin 数。
        detection_efficiency：探测效率，取值范围 `[0,1]`。
        dark_count_mean：单轮平均暗计数。

    返回：
        形状为 `(raw_num_bins, cutoff, cutoff)` 的对角 POVM 数组。
    """
    if raw_num_bins < 2:
        raise ValueError("raw_num_bins must be at least 2.")
    if not (0.0 <= detection_efficiency <= 1.0):
        raise ValueError("detection_efficiency must lie in [0, 1].")
    if dark_count_mean < 0.0:
        raise ValueError("dark_count_mean must be non-negative.")

    photon_numbers = np.arange(cutoff, dtype=int)
    exact_count_cap = raw_num_bins - 1
    diagonal_probabilities = np.zeros((raw_num_bins, cutoff), dtype=float)

    for n in photon_numbers:
        exact_probabilities = np.zeros(exact_count_cap, dtype=float)
        for detected_count in range(exact_count_cap):
            total = 0.0
            upper = min(n, detected_count)
            for transmitted_photons in range(upper + 1):
                binomial_weight = (
                    math.comb(int(n), int(transmitted_photons))
                    * (detection_efficiency ** transmitted_photons)
                    * ((1.0 - detection_efficiency) ** (n - transmitted_photons))
                )
                dark_count = detected_count - transmitted_photons
                poisson_log = -dark_count_mean
                if dark_count > 0:
                    poisson_log += dark_count * math.log(dark_count_mean) - gammaln(dark_count + 1.0)
                poisson_weight = math.exp(poisson_log) if dark_count_mean > 0.0 or dark_count == 0 else 0.0
                total += binomial_weight * poisson_weight
            exact_probabilities[detected_count] = total

        tail_probability = max(0.0, 1.0 - float(exact_probabilities.sum()))
        diagonal_probabilities[:exact_count_cap, n] = exact_probabilities
        diagonal_probabilities[exact_count_cap, n] = tail_probability

    povm = np.zeros((raw_num_bins, cutoff, cutoff), dtype=complex)
    for output in range(raw_num_bins):
        povm[output] = np.diag(diagonal_probabilities[output, :])
    return povm


def build_displaced_apd_povm(
    cutoff: int,
    displacement_alpha: complex,
    raw_num_bins: int,
    detection_efficiency: float,
    dark_count_mean: float,
) -> np.ndarray:
    """构造带位移的 APD-like POVM。

    功能：
        先构造对角 APD 计数 POVM，再通过位移酉算符做共轭变换，
        近似描述 displaced-count / IQ 风格的测量前端。

    参数：
        cutoff：Fock 截断维数。
        displacement_alpha：位移振幅。
        raw_num_bins：原始计数 bin 数。
        detection_efficiency：探测效率。
        dark_count_mean：暗计数均值。

    返回：
        位移后的 POVM 元数组。
    """
    diagonal_povm = build_apd_count_povm(
        cutoff=cutoff,
        raw_num_bins=raw_num_bins,
        detection_efficiency=detection_efficiency,
        dark_count_mean=dark_count_mean,
    )
    displacement = build_displacement_unitary(cutoff, displacement_alpha)
    return np.asarray(
        [displacement.conj().T @ element @ displacement for element in diagonal_povm],
        dtype=complex,
    )


def probabilities_from_povm(rho_matrices: np.ndarray, povm: np.ndarray) -> np.ndarray:
    """计算 `Tr(rho_x M_c)` 概率表并做逐行归一化。

    功能：
        给定一组输入态密度矩阵与 POVM 元，生成二维概率表 `P(c|x)`。

    参数：
        rho_matrices：输入态密度矩阵数组，形状 `(num_inputs, d, d)`。
        povm：POVM 元数组，形状 `(num_outputs, d, d)`。

    返回：
        形状 `(num_inputs, num_outputs)` 的概率表。

    说明：
        由于有限维截断和数值误差，函数会先把极小的负值裁到 0，
        再逐行归一化。
    """
    num_inputs = rho_matrices.shape[0]
    num_outputs = povm.shape[0]
    probabilities = np.zeros((num_inputs, num_outputs), dtype=float)
    for input_index in range(num_inputs):
        for output in range(num_outputs):
            value = np.real_if_close(np.trace(rho_matrices[input_index] @ povm[output]))
            probabilities[input_index, output] = float(np.real(value))
    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("Generated POVM probabilities contain a zero-sum row.")
    return probabilities / row_sums


def coarse_grain_probability_table(
    raw_probabilities: np.ndarray,
    num_outputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把原始概率直方图按等覆盖规则压缩成较少输出。

    功能：
        根据 `build_equal_cover_edges(...)` 生成等覆盖边界，再把相邻 raw bins
        累加成 coarse-grained 输出。

    参数：
        raw_probabilities：二维原始概率表，列对应 raw bins。
        num_outputs：希望压缩成的输出数。

    返回：
        二元组 `(coarse_probabilities, edges)`：
        - `coarse_probabilities`：压缩后的概率表；
        - `edges`：使用的边界数组。
    """
    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 2:
        raise ValueError("raw_probabilities must be a 2-D array.")
    num_raw_bins = raw.shape[1]
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive.")
    if num_outputs > num_raw_bins:
        raise ValueError("num_outputs cannot exceed the number of raw bins.")
    edges = build_equal_cover_edges(num_raw_bins, num_outputs)
    coarse = np.zeros((raw.shape[0], num_outputs), dtype=float)
    for output in range(num_outputs):
        coarse[:, output] = raw[:, edges[output] : edges[output + 1]].sum(axis=1)
    return coarse, edges


def validate_coarse_grain_edges(
    edges: list[int] | np.ndarray,
    *,
    num_raw_bins: int,
) -> np.ndarray:
    """校验自定义 coarse-graining 边界是否合法。

    参数：
        edges：边界端点列表。
        num_raw_bins：原始 bin 总数。

    返回：
        规范化后的整型边界数组。

    异常：
        当边界未从 0 开始、未以 `num_raw_bins` 结束，或不是严格递增时，
        抛出 `ValueError`。
    """
    resolved = np.asarray(edges, dtype=int).reshape(-1)
    if resolved.size < 2:
        raise ValueError("custom coarse-grain edges must contain at least two endpoints.")
    if int(resolved[0]) != 0 or int(resolved[-1]) != int(num_raw_bins):
        raise ValueError("custom coarse-grain edges must start at 0 and end at num_raw_bins.")
    if np.any(np.diff(resolved) <= 0):
        raise ValueError("custom coarse-grain edges must be strictly increasing.")
    return resolved


def coarse_grain_probability_table_with_edges(
    raw_probabilities: np.ndarray,
    edges: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按用户指定边界压缩原始概率表。

    功能：
        与 `coarse_grain_probability_table` 类似，但边界不再自动生成，
        而是由调用者显式给出。

    参数：
        raw_probabilities：二维原始概率表。
        edges：自定义边界。

    返回：
        二元组 `(coarse_probabilities, resolved_edges)`。
    """
    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 2:
        raise ValueError("raw_probabilities must be a 2-D array.")
    resolved_edges = validate_coarse_grain_edges(edges, num_raw_bins=int(raw.shape[1]))
    num_outputs = int(len(resolved_edges) - 1)
    coarse = np.zeros((raw.shape[0], num_outputs), dtype=float)
    for output in range(num_outputs):
        coarse[:, output] = raw[:, resolved_edges[output] : resolved_edges[output + 1]].sum(axis=1)
    return coarse, resolved_edges


def load_external_probability_table(
    probability_path: str | Path,
    variable_name: str | None = None,
) -> np.ndarray:
    """加载外部概率表文件。

    功能：
        支持 `.npy`、`.npz`、`.json`、`.mat` 四类格式，用于把实验或预处理
        后的概率表接入 route4-ex。

    参数：
        probability_path：文件路径。
        variable_name：当输入是 `.npz` 或 `.mat` 时，指定要读取的变量名。

    返回：
        二维浮点概率表。

    异常：
        当格式不支持、变量不存在或载入后不是二维数组时抛出 `ValueError`。
    """
    path = Path(probability_path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        table = np.asarray(np.load(path), dtype=float)
    elif suffix == ".npz":
        archive = np.load(path)
        if variable_name is not None:
            table = np.asarray(archive[variable_name], dtype=float)
        else:
            keys = list(archive.keys())
            if len(keys) == 0:
                raise ValueError(f"No arrays found in {path}.")
            table = np.asarray(archive[keys[0]], dtype=float)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        table = np.asarray(data, dtype=float)
    elif suffix == ".mat":
        mat_data = loadmat(path)
        if variable_name is not None:
            if variable_name not in mat_data:
                raise ValueError(f"Variable {variable_name!r} not found in {path}.")
            table = np.asarray(mat_data[variable_name], dtype=float)
        else:
            variable_names = [name for name in mat_data.keys() if not name.startswith("__")]
            if not variable_names:
                raise ValueError(f"No data arrays found in {path}.")
            table = np.asarray(mat_data[variable_names[0]], dtype=float)
    else:
        raise ValueError(f"Unsupported external probability format: {path.suffix}")

    if table.ndim != 2:
        raise ValueError(f"Expected a 2-D probability table, got shape {table.shape}.")
    return table


def _input_offdiagonal_metrics(rho_matrices: np.ndarray) -> dict[str, Any]:
    """计算 trusted input 的非对角强度指标。

    功能：
        对每个输入态统计：
        - 整体 Frobenius 范数；
        - 非对角部分 Frobenius 范数；
        - 非对角部分占比；
        - 最大非对角元素绝对值。

    参数：
        rho_matrices：输入态密度矩阵数组。

    返回：
        适合写入结果 JSON 的诊断字典。
    """
    metrics: list[dict[str, float]] = []
    ratios: list[float] = []
    max_entries: list[float] = []
    for index, rho in enumerate(rho_matrices):
        diagonal = np.diag(np.diag(rho))
        offdiag = rho - diagonal
        fro_norm = float(np.linalg.norm(rho, ord="fro"))
        offdiag_norm = float(np.linalg.norm(offdiag, ord="fro"))
        ratio = 0.0 if fro_norm == 0.0 else offdiag_norm / fro_norm
        max_offdiag = 0.0 if rho.size == 0 else float(np.max(np.abs(offdiag)))
        ratios.append(ratio)
        max_entries.append(max_offdiag)
        metrics.append(
            {
                "input_index": int(index),
                "fro_norm": fro_norm,
                "offdiag_fro_norm": offdiag_norm,
                "offdiag_over_fro": ratio,
                "max_abs_offdiag": max_offdiag,
            }
        )
    return {
        "per_input": metrics,
        "max_offdiag_over_fro": 0.0 if not ratios else float(max(ratios)),
        "mean_offdiag_over_fro": 0.0 if not ratios else float(np.mean(ratios)),
        "max_abs_offdiag": 0.0 if not max_entries else float(max(max_entries)),
    }


def _build_instance_from_probabilities(
    *,
    alpha_values: list[complex],
    q_selected: list[float],
    cutoff: int,
    raw_probabilities: np.ndarray,
    prob_floor: float | None,
    input_model: str,
    probability_model: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把输入振幅与概率表整理成统一的 route4-ex 实例字典。

    功能：
        这是所有 `prepare_*_instance` 的公共后端。它负责：
        - 校验 `alpha_values` 与 `q_selected`；
        - 构造 non-diagonal coherent-state density matrices；
        - 记录 Fock 对角部分，供 diagonal primal 使用；
        - 归一化并可选正则化概率表；
        - 计算 distribution-only 指标；
        - 记录输入非对角诊断信息。

    参数：
        alpha_values：trusted coherent input 振幅列表。
        q_selected：生成轮目标函数权重。
        cutoff：Fock 截断维数。
        raw_probabilities：尚未做 prob-floor 正则化的概率表。
        prob_floor：最小概率地板；为 `None` 或非正时表示不做正则化。
        input_model：输入态模型名称，写入元数据。
        probability_model：概率来源名称，写入元数据。
        extra_metadata：额外附加到实例字典中的字段。

    返回：
        一个统一实例字典，后续所有 primal/full-primal 求解器都基于它工作。
    """
    alpha_values, q = _validate_alpha_and_q(alpha_values, q_selected)
    rho_matrices = build_coherent_density_matrices(alpha_values, cutoff)
    rho_diag = np.real_if_close(np.diagonal(rho_matrices, axis1=1, axis2=2)).astype(float)

    raw_probabilities = np.asarray(raw_probabilities, dtype=float)
    if raw_probabilities.shape[0] != len(alpha_values):
        raise ValueError("raw_probabilities row count must match the number of alpha_values.")
    raw_row_sums = raw_probabilities.sum(axis=1, keepdims=True)
    if np.any(raw_row_sums <= 0.0):
        raise ValueError("raw_probabilities contains a zero-sum row.")
    normalized_raw_probabilities = raw_probabilities / raw_row_sums

    probabilities = normalized_raw_probabilities.copy()
    regularized_entries = 0
    if prob_floor is not None and prob_floor > 0:
        regularized_entries = int((probabilities == 0.0).sum())
        probabilities = np.maximum(probabilities, prob_floor)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    instance = {
        "route": "route4_ex_nondiagonal_inputs",
        "input_model": input_model,
        "probability_model": probability_model,
        "alpha_values": alpha_values,
        "q_selected": q,
        "cutoff": int(cutoff),
        "num_inputs": int(len(alpha_values)),
        "num_outputs": int(probabilities.shape[1]),
        "prob_floor": prob_floor,
        "regularized_entries": regularized_entries,
        "rho_matrices": rho_matrices,
        "rho_diag": rho_diag,
        "probabilities_raw": normalized_raw_probabilities,
        "probabilities": probabilities,
        "raw_row_sums_before_normalization": raw_row_sums.reshape(-1),
        "distribution_only_p_guess_raw": distribution_only_guessing_probability(normalized_raw_probabilities, q),
        "distribution_only_p_guess": distribution_only_guessing_probability(probabilities, q),
        "input_offdiagonal_metrics": _input_offdiagonal_metrics(rho_matrices),
    }
    if extra_metadata is not None:
        instance.update(extra_metadata)
    return instance


def prepare_route4_ex_toy_instance(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probe_alpha: complex = DEFAULT_PROBE_ALPHA,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
) -> dict[str, Any]:
    """构造 toy coherent-projector 场景下的实例。

    功能：
        用二元 coherent-projector POVM 生成一张小型理论概率表，
        主要用于验证“非对角 trusted inputs 会让 full-primal 与 diagonal-primal
        分叉”这一结构现象。

    参数：
        alpha_values：输入振幅列表。
        q_selected：生成轮权重。
        cutoff：截断维数。
        probe_alpha：toy 探针相干态。
        prob_floor：概率正则化地板。

    返回：
        可直接喂给 route4-ex 求解器的统一实例字典。
    """
    alpha_values = [complex(alpha) for alpha in alpha_values]
    toy_povm = build_binary_coherent_projector_povm(cutoff, probe_alpha)
    rho_matrices = build_coherent_density_matrices(alpha_values, cutoff)
    raw_probabilities = probabilities_from_povm(rho_matrices, toy_povm)
    return _build_instance_from_probabilities(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        raw_probabilities=raw_probabilities,
        prob_floor=prob_floor,
        input_model="exact_truncated_coherent_states",
        probability_model="binary_coherent_projector_povm",
        extra_metadata={
            "probe_alpha": complex(probe_alpha),
            "raw_num_bins": int(raw_probabilities.shape[1]),
            "toy_povm": toy_povm,
        },
    )


def prepare_route4_ex_apdlike_instance(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    displacement_alpha: complex = DEFAULT_DISPLACEMENT_ALPHA,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    raw_num_bins: int = DEFAULT_RAW_NUM_BINS,
    detection_efficiency: float = DEFAULT_DETECTION_EFFICIENCY,
    dark_count_mean: float = DEFAULT_DARK_COUNT_MEAN,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
) -> dict[str, Any]:
    """构造 APD-like 位移计数模型下的理论实例。

    功能：
        使用 displaced APD POVM 生成 raw histogram，再按 `num_outputs`
        做 coarse-graining，得到更贴近 APD / IQ 前端的理论概率表。

    参数：
        alpha_values：输入振幅列表。
        q_selected：生成轮权重。
        cutoff：截断维数。
        displacement_alpha：位移振幅。
        num_outputs：最终 coarse-grained 输出数。
        raw_num_bins：原始计数直方图 bin 数。
        detection_efficiency：探测效率。
        dark_count_mean：暗计数均值。
        prob_floor：概率正则化地板。

    返回：
        统一实例字典，并在元数据里带上原始 histogram 与边界。
    """
    alpha_values = [complex(alpha) for alpha in alpha_values]
    rho_matrices = build_coherent_density_matrices(alpha_values, cutoff)
    raw_povm = build_displaced_apd_povm(
        cutoff=cutoff,
        displacement_alpha=displacement_alpha,
        raw_num_bins=raw_num_bins,
        detection_efficiency=detection_efficiency,
        dark_count_mean=dark_count_mean,
    )
    raw_histogram = probabilities_from_povm(rho_matrices, raw_povm)
    coarse_probabilities, edges = coarse_grain_probability_table(raw_histogram, num_outputs)
    return _build_instance_from_probabilities(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        raw_probabilities=coarse_probabilities,
        prob_floor=prob_floor,
        input_model="exact_truncated_coherent_states",
        probability_model="displaced_apd_count_histogram_coarse_grained",
        extra_metadata={
            "displacement_alpha": complex(displacement_alpha),
            "raw_num_bins": int(raw_histogram.shape[1]),
            "detection_efficiency": float(detection_efficiency),
            "dark_count_mean": float(dark_count_mean),
            "coarse_grain_edges": np.asarray(edges, dtype=int),
            "raw_histogram_probabilities": raw_histogram,
        },
    )


def intensity_to_alpha(
    intensity: float,
    *,
    max_intensity: float,
    max_abs_alpha: float,
    phase: float = 0.0,
) -> complex:
    """把实验强度值映射成相干态振幅。

    功能：
        使用 `|alpha| ∝ sqrt(intensity)` 的常见映射，把实验光强转换为
        coherent-state 振幅，并附上给定相位。

    参数：
        intensity：当前输入对应的光强。
        max_intensity：参考最大光强。
        max_abs_alpha：最大光强对应的振幅模长。
        phase：相位。

    返回：
        对应的复数振幅 `alpha`。
    """
    if max_intensity <= 0.0:
        raise ValueError("max_intensity must be positive.")
    if intensity < 0.0:
        raise ValueError("intensity must be non-negative.")
    radius = max_abs_alpha * math.sqrt(float(intensity) / float(max_intensity))
    return complex(radius * np.exp(1j * phase))


def intensities_to_alpha_values(
    intensities: list[float],
    *,
    max_abs_alpha: float,
    phases: list[float] | None = None,
    max_intensity: float | None = None,
) -> list[complex]:
    """把一组实验强度批量映射成 coherent-state 振幅列表。

    参数：
        intensities：强度列表。
        max_abs_alpha：最大强度对应的最大振幅模长。
        phases：每个输入对应的相位列表；若为 `None` 则默认全 0。
        max_intensity：显式指定参考最大强度；若省略则取 `max(intensities)`。

    返回：
        复振幅列表，顺序与输入强度一一对应。
    """
    if len(intensities) == 0:
        raise ValueError("intensities cannot be empty.")
    if phases is None:
        phases = [0.0] * len(intensities)
    if len(phases) != len(intensities):
        raise ValueError("phases must have the same length as intensities.")
    resolved_max_intensity = float(max(intensities)) if max_intensity is None else float(max_intensity)
    return [
        intensity_to_alpha(
            intensity=float(intensity),
            max_intensity=resolved_max_intensity,
            max_abs_alpha=max_abs_alpha,
            phase=float(phase),
        )
        for intensity, phase in zip(intensities, phases)
    ]


def prepare_route4_ex_external_instance(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path | None = None,
    num_outputs: int | None = DEFAULT_NUM_OUTPUTS,
    row_indices: list[int] | None = None,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    variable_name: str | None = None,
    already_coarse: bool = False,
    custom_edges: list[int] | None = None,
) -> dict[str, Any]:
    """从外部概率表构造 route4-ex 实例。

    功能：
        这是当前实验数据接入 route4-ex 的主入口。它负责：
        - 加载外部概率表；
        - 选取指定输入行；
        - 根据需要直接使用 coarse 表，或先按边界做 coarse-graining；
        - 与给定的 trusted coherent inputs 结合，生成统一实例字典。

    参数：
        alpha_values：trusted input 振幅列表。
        q_selected：生成轮权重。
        cutoff：截断维数。
        probability_path：外部概率表路径。
        num_outputs：目标输出数；若 `already_coarse=True` 则可省略。
        row_indices：从外部表中选取哪些输入行。
        prob_floor：概率正则化地板。
        variable_name：`.mat/.npz` 变量名。
        already_coarse：外部表是否已经是 coarse-grained 输出。
        custom_edges：自定义 coarse-graining 边界。

    返回：
        统一实例字典。

    说明：
        这是实验口径最关键的接口之一，当前主要服务于 `Probability.mat`
        这类外部概率数据。
    """
    if probability_path is None:
        raise ValueError("probability_path is required for external route4-ex instances.")

    alpha_values = [complex(alpha) for alpha in alpha_values]
    full_table = load_external_probability_table(probability_path, variable_name=variable_name)
    if row_indices is None:
        row_indices = list(range(len(alpha_values)))
    if len(row_indices) != len(alpha_values):
        raise ValueError("row_indices must have the same length as alpha_values.")
    if max(row_indices) >= full_table.shape[0] or min(row_indices) < 0:
        raise ValueError("row_indices are out of range for the external probability table.")

    selected_rows = np.asarray(full_table[row_indices, :], dtype=float)
    if already_coarse:
        if custom_edges is not None:
            raise ValueError("custom_edges cannot be used when external_table_already_coarse=True.")
        if num_outputs is not None and selected_rows.shape[1] != int(num_outputs):
            raise ValueError("External coarse table column count must match num_outputs.")
        coarse_probabilities = selected_rows
        edges = None
    else:
        if custom_edges is not None:
            coarse_probabilities, edges = coarse_grain_probability_table_with_edges(selected_rows, custom_edges)
            if num_outputs is not None and int(num_outputs) != int(len(edges) - 1):
                raise ValueError("num_outputs must match len(custom_edges) - 1 when custom_edges is provided.")
        else:
            if num_outputs is None:
                raise ValueError("num_outputs must be provided when external data is not already coarse-grained.")
            coarse_probabilities, edges = coarse_grain_probability_table(selected_rows, int(num_outputs))

    return _build_instance_from_probabilities(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        raw_probabilities=coarse_probabilities,
        prob_floor=prob_floor,
        input_model="exact_truncated_coherent_states",
        probability_model="external_probability_table",
        extra_metadata={
            "external_probability_path": str(Path(probability_path)),
            "external_variable_name": variable_name,
            "external_row_indices": list(row_indices),
            "external_table_shape": list(full_table.shape),
            "external_table_already_coarse": bool(already_coarse),
            "raw_num_bins": int(selected_rows.shape[1]),
            "coarse_grain_edges": None if edges is None else np.asarray(edges, dtype=int),
            "external_selected_rows_raw": selected_rows,
        },
    )


def _instance_summary(instance: dict[str, Any]) -> dict[str, Any]:
    """把完整实例字典压缩成结果文件友好的摘要。

    功能：
        求解器返回结果时通常不需要把所有中间矩阵都写进 JSON。
        该函数只保留实验解释、输入态、概率表、distribution-only 指标等
        高价值字段，形成可写文件的摘要。

    参数：
        instance：由 `prepare_*_instance` 生成的完整实例字典。

    返回：
        去除了大块矩阵对象后的轻量摘要字典。
    """
    raw_p_guess = float(instance["distribution_only_p_guess_raw"])
    reg_p_guess = float(instance["distribution_only_p_guess"])
    return {
        "input_model": str(instance["input_model"]),
        "probability_model": str(instance["probability_model"]),
        "alpha_values": _serialize_complex_list(list(instance["alpha_values"])),
        "q_selected": np.asarray(instance["q_selected"], dtype=float).tolist(),
        "num_inputs": int(instance["num_inputs"]),
        "num_outputs": int(instance["num_outputs"]),
        "cutoff": int(instance["cutoff"]),
        "prob_floor": None if instance["prob_floor"] is None else float(instance["prob_floor"]),
        "regularized_entries": int(instance["regularized_entries"]),
        "raw_num_bins": int(instance["raw_num_bins"]) if "raw_num_bins" in instance else None,
        "raw_row_sums_before_normalization": np.asarray(
            instance["raw_row_sums_before_normalization"], dtype=float
        ).tolist(),
        "distribution_only_p_guess_raw": raw_p_guess,
        "distribution_only_H_min_raw": float(-np.log2(raw_p_guess)) if raw_p_guess > 0 else None,
        "distribution_only_p_guess": reg_p_guess,
        "distribution_only_H_min": float(-np.log2(reg_p_guess)) if reg_p_guess > 0 else None,
        "input_offdiagonal_metrics": instance["input_offdiagonal_metrics"],
        "probabilities_raw": np.asarray(instance["probabilities_raw"], dtype=float).tolist(),
        "probabilities": np.asarray(instance["probabilities"], dtype=float).tolist(),
        "probe_alpha": None
        if "probe_alpha" not in instance
        else _serialize_complex(complex(instance["probe_alpha"])),
        "displacement_alpha": None
        if "displacement_alpha" not in instance
        else _serialize_complex(complex(instance["displacement_alpha"])),
        "detection_efficiency": None
        if "detection_efficiency" not in instance
        else float(instance["detection_efficiency"]),
        "dark_count_mean": None
        if "dark_count_mean" not in instance
        else float(instance["dark_count_mean"]),
        "coarse_grain_edges": None
        if "coarse_grain_edges" not in instance or instance["coarse_grain_edges"] is None
        else np.asarray(instance["coarse_grain_edges"], dtype=int).tolist(),
        "external_probability_path": str(instance["external_probability_path"])
        if "external_probability_path" in instance
        else None,
        "external_variable_name": instance.get("external_variable_name"),
        "external_row_indices": list(instance["external_row_indices"])
        if "external_row_indices" in instance
        else None,
        "external_table_shape": list(instance["external_table_shape"])
        if "external_table_shape" in instance
        else None,
        "external_table_already_coarse": bool(instance["external_table_already_coarse"])
        if "external_table_already_coarse" in instance
        else None,
    }


def solve_route4_ex_diagonal_primal(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
) -> dict[str, Any]:
    """求解仅允许 Fock 对角测量元的 diagonal primal。

    功能：
        在 trusted inputs 已固定的前提下，把测量元限制为 Fock 基对角变量，
        求出给定 `q_selected` 下的最大 guessing probability。

    逻辑：
        - 只保留每个 POVM 元在 Fock 对角上的自由度；
        - 用策略索引 `lambda` 枚举确定性后处理；
        - 强制每个策略下所有输出的对角向量满足完备性约束；
        - 用观测概率表约束总测量元；
        - 最大化生成轮权重下的 guessing probability。

    参数：
        instance：统一实例字典，必须包含 `rho_diag` 与 `probabilities`。
        preferred_solver：首选求解器。
        solver_options：传给求解器的附加参数。
        verbose：是否输出求解器详细日志。
        max_primal_variables：变量规模保护阈值。

    返回：
        包含状态、求解器、`p_guess`、`H_min`、规模估计和实例摘要的结果字典。
    """
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_diag = np.asarray(instance["rho_diag"], dtype=float)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_diag.shape[1]

    size_info = estimate_primal_problem_size(num_inputs, num_outputs, cutoff)
    if max_primal_variables is not None and size_info["variable_count"] > max_primal_variables:
        raise ValueError(
            "The requested route4-ex diagonal primal instance is too large for the configured safeguard: "
            f"{size_info['variable_count']} > {max_primal_variables}."
        )
    if size_info["variable_count"] > 3_000_000:
        warnings.warn(
            "The route4-ex diagonal primal is very large and may take a long time to canonicalize or solve. "
            f"Estimated variables: {size_info['variable_count']}.",
            stacklevel=2,
        )

    lambda_indices = np.array(list(iterproduct(range(num_outputs), repeat=num_inputs + 1)), dtype=int)
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
        solver_options=solver_options,
        verbose=verbose,
    )

    if status in ("optimal", "optimal_inaccurate") and problem.value is not None:
        value = float(np.real_if_close(problem.value))
    else:
        value = None
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_ex_diagonal_primal",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
        "measurement_constraint": "Fock_diagonal_only",
    }
    result.update(size_info)
    result.update(_instance_summary(instance))
    return result


def solve_route4_ex_full_primal(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """求解允许一般 Hermitian PSD 测量元的 full primal。

    功能：
        这是 route4-ex 的核心认证问题。与 diagonal primal 相比，
        full primal 会完整使用 trusted coherent inputs 的非对角结构。

    逻辑：
        - 为每个输出 `c` 和每个策略 `lambda` 引入 Hermitian PSD 变量；
        - 每个策略下要求 `sum_c M_{c,lambda} = s_lambda I`；
        - 对每个输入态和输出，强制 `Tr(rho_x M_c)` 与观测概率匹配；
        - 按 `q_selected` 加权最大化 guessing probability。

    参数：
        instance：统一实例字典，必须包含完整 `rho_matrices`。
        preferred_solver：首选求解器。
        solver_options：求解器参数。
        verbose：是否输出详细日志。
        max_hermitian_scalar_count：Hermitian 变量规模保护阈值。

    返回：
        包含 `status`、`p_guess`、`H_min`、规模估计和实例摘要的结果字典。

    说明：
        该函数是当前 route4-ex 正式结果的核心求解入口。
    """
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_matrices = np.asarray(instance["rho_matrices"], dtype=complex)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_matrices.shape[1]

    size_info = estimate_full_primal_problem_size(num_inputs, num_outputs, cutoff)
    if (
        max_hermitian_scalar_count is not None
        and size_info["hermitian_scalar_count"] > max_hermitian_scalar_count
    ):
        raise ValueError(
            "The requested route4-ex full primal instance is too large for the configured safeguard: "
            f"{size_info['hermitian_scalar_count']} > {max_hermitian_scalar_count}."
        )
    if size_info["hermitian_scalar_count"] > 400_000:
        warnings.warn(
            "The route4-ex full primal is large and may take a long time to canonicalize or solve. "
            f"Estimated Hermitian scalar count: {size_info['hermitian_scalar_count']}.",
            stacklevel=2,
        )

    lambda_indices = np.array(list(iterproduct(range(num_outputs), repeat=num_inputs + 1)), dtype=int)
    num_strategies = lambda_indices.shape[0]
    identity = np.eye(cutoff, dtype=complex)

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
        solver_options=solver_options,
        verbose=verbose,
    )

    if status in ("optimal", "optimal_inaccurate") and problem.value is not None:
        value = float(np.real_if_close(problem.value))
    else:
        value = None
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_ex_full_primal",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
        "measurement_constraint": "general_Hermitian_PSD",
    }
    result.update(size_info)
    result.update(_instance_summary(instance))
    return result


def run_route4_ex_toy_diagonal_primal(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probe_alpha: complex = DEFAULT_PROBE_ALPHA,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
) -> dict[str, Any]:
    """一键运行 toy 实例上的 diagonal primal。

    功能：
        先构造 toy 实例，再调用 `solve_route4_ex_diagonal_primal(...)`。

    参数：
        与 `prepare_route4_ex_toy_instance(...)` 和
        `solve_route4_ex_diagonal_primal(...)` 对应。

    返回：
        diagonal primal 的求解结果字典。
    """
    instance = prepare_route4_ex_toy_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        probe_alpha=probe_alpha,
        prob_floor=prob_floor,
    )
    return solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )


def run_route4_ex_toy_full_primal(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probe_alpha: complex = DEFAULT_PROBE_ALPHA,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """一键运行 toy 实例上的 full primal。"""
    instance = prepare_route4_ex_toy_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        probe_alpha=probe_alpha,
        prob_floor=prob_floor,
    )
    return solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )


def compare_route4_ex_toy_diagonal_full(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probe_alpha: complex = DEFAULT_PROBE_ALPHA,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """比较 toy 实例下 diagonal primal 与 full primal 的差异。

    功能：
        用同一 toy 概率实例分别求解 diagonal 与 full 模型，并输出两者的
        `p_guess` / `H_min` 差异。

    参数：
        与 toy 实例构造和两类求解器接口一致。

    返回：
        包含实例摘要、两类求解结果和 gap 指标的对照字典。
    """
    instance = prepare_route4_ex_toy_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        probe_alpha=probe_alpha,
        prob_floor=prob_floor,
    )
    diagonal_primal = solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )
    full_primal = solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )

    diagonal_p_guess = diagonal_primal.get("p_guess")
    full_p_guess = full_primal.get("p_guess")
    diagonal_h_min = diagonal_primal.get("H_min")
    full_h_min = full_primal.get("H_min")
    return {
        "route": "route4_ex_toy_diagonal_full_compare",
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


def run_route4_ex_apdlike_diagonal_primal(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    displacement_alpha: complex = DEFAULT_DISPLACEMENT_ALPHA,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    raw_num_bins: int = DEFAULT_RAW_NUM_BINS,
    detection_efficiency: float = DEFAULT_DETECTION_EFFICIENCY,
    dark_count_mean: float = DEFAULT_DARK_COUNT_MEAN,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
) -> dict[str, Any]:
    """一键运行 APD-like 理论实例上的 diagonal primal。"""
    instance = prepare_route4_ex_apdlike_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        displacement_alpha=displacement_alpha,
        num_outputs=num_outputs,
        raw_num_bins=raw_num_bins,
        detection_efficiency=detection_efficiency,
        dark_count_mean=dark_count_mean,
        prob_floor=prob_floor,
    )
    return solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )


def run_route4_ex_apdlike_full_primal(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    displacement_alpha: complex = DEFAULT_DISPLACEMENT_ALPHA,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    raw_num_bins: int = DEFAULT_RAW_NUM_BINS,
    detection_efficiency: float = DEFAULT_DETECTION_EFFICIENCY,
    dark_count_mean: float = DEFAULT_DARK_COUNT_MEAN,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """一键运行 APD-like 理论实例上的 full primal。"""
    instance = prepare_route4_ex_apdlike_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        displacement_alpha=displacement_alpha,
        num_outputs=num_outputs,
        raw_num_bins=raw_num_bins,
        detection_efficiency=detection_efficiency,
        dark_count_mean=dark_count_mean,
        prob_floor=prob_floor,
    )
    return solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )


def compare_route4_ex_apdlike_diagonal_full(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    displacement_alpha: complex = DEFAULT_DISPLACEMENT_ALPHA,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    raw_num_bins: int = DEFAULT_RAW_NUM_BINS,
    detection_efficiency: float = DEFAULT_DETECTION_EFFICIENCY,
    dark_count_mean: float = DEFAULT_DARK_COUNT_MEAN,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """比较 APD-like 理论实例下 diagonal 与 full 模型的差异。"""
    instance = prepare_route4_ex_apdlike_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        displacement_alpha=displacement_alpha,
        num_outputs=num_outputs,
        raw_num_bins=raw_num_bins,
        detection_efficiency=detection_efficiency,
        dark_count_mean=dark_count_mean,
        prob_floor=prob_floor,
    )
    diagonal_primal = solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )
    full_primal = solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )
    diagonal_p_guess = diagonal_primal.get("p_guess")
    full_p_guess = full_primal.get("p_guess")
    diagonal_h_min = diagonal_primal.get("H_min")
    full_h_min = full_primal.get("H_min")
    return {
        "route": "route4_ex_apdlike_diagonal_full_compare",
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


def run_route4_ex_external_diagonal_primal(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path | None = None,
    num_outputs: int | None = DEFAULT_NUM_OUTPUTS,
    row_indices: list[int] | None = None,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    variable_name: str | None = None,
    already_coarse: bool = False,
    custom_edges: list[int] | None = None,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
) -> dict[str, Any]:
    """一键运行外部概率表实例上的 diagonal primal。

    功能：
        这是把实验概率数据接入 diagonal baseline 的便捷入口。

    参数：
        与 `prepare_route4_ex_external_instance(...)` 及
        `solve_route4_ex_diagonal_primal(...)` 对应。

    返回：
        diagonal primal 的求解结果。
    """
    instance = prepare_route4_ex_external_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        probability_path=probability_path,
        num_outputs=num_outputs,
        row_indices=row_indices,
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=already_coarse,
        custom_edges=custom_edges,
    )
    return solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )


def run_route4_ex_external_full_primal(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path | None = None,
    num_outputs: int | None = DEFAULT_NUM_OUTPUTS,
    row_indices: list[int] | None = None,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    variable_name: str | None = None,
    already_coarse: bool = False,
    custom_edges: list[int] | None = None,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """一键运行外部概率表实例上的 full primal。

    功能：
        这是 route4-ex 处理真实/外部概率数据时最常用的正式求解入口。

    参数：
        与 `prepare_route4_ex_external_instance(...)` 及
        `solve_route4_ex_full_primal(...)` 对应。

    返回：
        full primal 求解结果。
    """
    instance = prepare_route4_ex_external_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        probability_path=probability_path,
        num_outputs=num_outputs,
        row_indices=row_indices,
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=already_coarse,
        custom_edges=custom_edges,
    )
    return solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )


def compare_route4_ex_external_diagonal_full(
    alpha_values: list[complex] = DEFAULT_ALPHA_VALUES,
    q_selected: list[float] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path | None = None,
    num_outputs: int | None = DEFAULT_NUM_OUTPUTS,
    row_indices: list[int] | None = None,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    variable_name: str | None = None,
    already_coarse: bool = False,
    custom_edges: list[int] | None = None,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = None,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """比较外部概率表实例下 diagonal 与 full 模型的认证差异。

    功能：
        对同一外部概率表实例同时求解 diagonal primal 与 full primal，
        量化“非对角 trusted inputs 是否真正带来更强约束”。

    参数：
        与 `prepare_route4_ex_external_instance(...)` 以及两类求解器一致。

    返回：
        包含实例摘要、两类求解结果以及 gap 指标的对照字典。

    说明：
        搜索脚本与诊断脚本中大量调用的就是这个接口。
    """
    instance = prepare_route4_ex_external_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=cutoff,
        probability_path=probability_path,
        num_outputs=num_outputs,
        row_indices=row_indices,
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=already_coarse,
        custom_edges=custom_edges,
    )
    diagonal_primal = solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )
    full_primal = solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )
    diagonal_p_guess = diagonal_primal.get("p_guess")
    full_p_guess = full_primal.get("p_guess")
    diagonal_h_min = diagonal_primal.get("H_min")
    full_h_min = full_primal.get("H_min")
    return {
        "route": "route4_ex_external_diagonal_full_compare",
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
