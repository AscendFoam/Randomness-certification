"""Route4 strict non-diagonal 扩展的核心实现。

这条路线的目标是做一个“最小改动版”的 route4 扩展：

1. 保留原 route4 / Matlab 主线的实验接口：
   - `selected_mu_list`
   - `q_selected`
   - `Probability.mat`
   - 等分 coarse-graining
2. 不做 `max_abs_alpha`、自由半径、phase-pattern 搜索等额外优化；
3. 只把 trusted input 从 Fock 对角 Poisson 模型替换为
   固定参数下的完整截断 coherent states；
4. 同时允许一般 Hermitian PSD POVM 元，以检查“输入非对角性”是否会
   真正改变认证值。

为了避免 full-primal 在原始 Fock 截断空间中规模爆炸，本文件对 full-primal
只在输入态张成的支撑子空间上求解。对当前问题，这个降维是严格等价的：
输入态全部位于该支撑子空间中，因此目标函数和统计约束只依赖 POVM 在该
子空间上的压缩；任意子空间解都可以平凡延拓回全空间而不改变观测统计。
"""

from __future__ import annotations

import cmath
import json
import math
import warnings
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np

from ..common import density_from_ket, project_density_to_basis, solve_cvxpy_problem, support_basis
from ..route4.phaseinsensitive import (
    DEFAULT_CUTOFF,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_PROB_FLOOR,
    DEFAULT_Q,
    DEFAULT_SELECTED_MU,
    DEFAULT_SHIFT,
    FULL_MU,
    estimate_full_primal_problem_size,
    prepare_phaseinsensitive_instance,
    solve_phaseinsensitive_primal,
)

DEFAULT_MEAN_PHOTONS_PER_MU_LABEL = 1.0


def _clean_value(value: Any) -> Any:
    """把 NumPy 类型递归转成适合 JSON 序列化的基础类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return _serialize_complex(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def result_to_json(result: Any) -> str:
    """把结果对象格式化为 JSON 字符串。"""
    return json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value)


def _serialize_complex(alpha: complex) -> dict[str, float]:
    """把单个复振幅转成便于展示的字典。"""
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
    """安全地求一组 CVXPY 矩阵表达式之和。"""
    if not expressions:
        return cp.Constant(np.zeros((dimension, dimension), dtype=complex))
    total = expressions[0]
    for expr in expressions[1:]:
        total = total + expr
    return total


def _input_offdiagonal_metrics(rho_matrices: np.ndarray) -> dict[str, Any]:
    """计算输入态非对角强度的简单诊断指标。"""
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


def mu_labels_to_mean_photon_numbers(
    selected_mu_list: list[int] | tuple[int, ...],
    mean_photons_per_mu_label: float = DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
) -> list[float]:
    """把 route4 的 `mu` 标签转换为用于 coherent state 的平均光子数。"""
    if mean_photons_per_mu_label <= 0.0:
        raise ValueError("mean_photons_per_mu_label must be positive.")
    mean_photons = [float(mu) * float(mean_photons_per_mu_label) for mu in selected_mu_list]
    if any(value < 0.0 for value in mean_photons):
        raise ValueError("selected_mu_list must be non-negative.")
    return mean_photons


def selected_mu_to_alpha_values(
    selected_mu_list: list[int] | tuple[int, ...],
    *,
    phase_values: list[float] | tuple[float, ...] | None = None,
    mean_photons_per_mu_label: float = DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
) -> list[complex]:
    """把 route4 的光强标签映射成固定 coherent amplitudes。

    这是 strict non-diagonal route4 的关键定义：
    - 不再搜索 `alpha`
    - 直接由 `selected_mu_list` 和固定相位 `phase_values`
      唯一定义 trusted coherent inputs
    """
    mean_photons = mu_labels_to_mean_photon_numbers(
        selected_mu_list,
        mean_photons_per_mu_label=mean_photons_per_mu_label,
    )
    if phase_values is None:
        phase_values = [0.0] * len(mean_photons)
    if len(phase_values) != len(mean_photons):
        raise ValueError("phase_values must have the same length as selected_mu_list.")
    return [
        complex(math.sqrt(mean_photon) * np.exp(1j * float(phase)))
        for mean_photon, phase in zip(mean_photons, phase_values)
    ]


def build_coherent_density_matrices(alpha_values: list[complex], cutoff: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """构造完整截断 coherent states 及其密度矩阵。"""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    if len(alpha_values) == 0:
        raise ValueError("alpha_values cannot be empty.")
    kets: list[np.ndarray] = []
    density_matrices: list[np.ndarray] = []
    for alpha in alpha_values:
        ket = _stable_coherent_state(cutoff, complex(alpha))
        kets.append(ket)
        density_matrices.append(density_from_ket(ket))
    return np.asarray(density_matrices, dtype=complex), kets


def _stable_coherent_state(dimension: int, alpha: complex) -> np.ndarray:
    """稳定构造高截断维下的相干态。

    这里不用 `common.coherent_state(...)` 的 `alpha^n / sqrt(n!)` 直接公式，
    因为在 `dimension=280` 这类 route4 典型参数下，`factorial(n)` 很容易溢出。
    改用递推：

    c_0 = exp(-|alpha|^2 / 2)
    c_n = c_{n-1} * alpha / sqrt(n)
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    coeffs = np.zeros(dimension, dtype=complex)
    coeffs[0] = cmath.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, dimension):
        coeffs[n] = coeffs[n - 1] * alpha / math.sqrt(float(n))

    norm = np.linalg.norm(coeffs)
    if norm > 0.0:
        coeffs /= norm
    return coeffs


def prepare_route4_strict_nondiagonal_instance(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
    phase_values: list[float] | tuple[float, ...] | None = None,
    mean_photons_per_mu_label: float = DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
    support_tol: float = 1e-9,
) -> dict[str, Any]:
    """构造 strict non-diagonal route4 的统一实例。

    逻辑：
    1. 完整复用原 route4 的概率表加载与 coarse-graining；
    2. 把同一组 `selected_mu_list` 映射成固定 coherent amplitudes；
    3. 构造完整截断 coherent-state 密度矩阵；
    4. 提取输入态支撑子空间，对 full-primal 做严格等价的降维表示。
    """
    reference_instance = prepare_phaseinsensitive_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        full_mu=full_mu,
    )

    selected_mu = list(reference_instance["selected_mu_list"])
    phase_values_list = [0.0] * len(selected_mu) if phase_values is None else [float(v) for v in phase_values]
    alpha_values = selected_mu_to_alpha_values(
        selected_mu,
        phase_values=phase_values_list,
        mean_photons_per_mu_label=mean_photons_per_mu_label,
    )
    rho_full, kets = build_coherent_density_matrices(alpha_values, cutoff)
    rho_diag_from_full = np.real_if_close(np.diagonal(rho_full, axis1=1, axis2=2)).astype(float)

    basis = support_basis(kets, tol=support_tol)
    if basis.shape[1] == 0:
        raise RuntimeError("Support basis unexpectedly has rank zero.")
    rho_reduced = np.asarray(
        [project_density_to_basis(rho, basis) for rho in rho_full],
        dtype=complex,
    )
    rho_reduced = np.asarray([0.5 * (rho + rho.conj().T) for rho in rho_reduced], dtype=complex)

    gram_matrix = np.asarray(
        [[np.vdot(ket_i, ket_j) for ket_j in kets] for ket_i in kets],
        dtype=complex,
    )
    reference_rho_diag = np.asarray(reference_instance["rho_diag"], dtype=float)
    diag_gap = float(np.max(np.abs(rho_diag_from_full - reference_rho_diag)))
    mean_photon_numbers = mu_labels_to_mean_photon_numbers(
        selected_mu,
        mean_photons_per_mu_label=mean_photons_per_mu_label,
    )

    instance = dict(reference_instance)
    instance.update(
        {
            "route": "route4_strict_nondiagonal",
            "alpha_values": [complex(alpha) for alpha in alpha_values],
            "phase_values": phase_values_list,
            "mean_photon_numbers": mean_photon_numbers,
            "mean_photons_per_mu_label": float(mean_photons_per_mu_label),
            "rho_matrices_full": rho_full,
            "rho_diag_from_full": rho_diag_from_full,
            "rho_reduced": rho_reduced,
            "support_basis": basis,
            "support_dimension": int(basis.shape[1]),
            "support_tol": float(support_tol),
            "gram_matrix": gram_matrix,
            "rho_diag_reference": reference_rho_diag,
            "rho_diag_reference_linf_gap": diag_gap,
            "input_offdiagonal_metrics_full": _input_offdiagonal_metrics(rho_full),
            "reference_phaseinsensitive_instance": reference_instance,
        }
    )
    return instance


def _instance_summary(instance: dict[str, Any]) -> dict[str, Any]:
    """把完整实例字典压缩成结果文件友好的摘要。"""
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
        "distribution_only_p_guess_raw": raw_p_guess,
        "distribution_only_H_min_raw": float(-np.log2(raw_p_guess)) if raw_p_guess > 0 else None,
        "distribution_only_p_guess": reg_p_guess,
        "distribution_only_H_min": float(-np.log2(reg_p_guess)) if reg_p_guess > 0 else None,
        "alpha_values": _serialize_complex_list(list(instance["alpha_values"])),
        "phase_values": [float(value) for value in instance["phase_values"]],
        "mean_photon_numbers": [float(value) for value in instance["mean_photon_numbers"]],
        "mean_photons_per_mu_label": float(instance["mean_photons_per_mu_label"]),
        "support_dimension": int(instance["support_dimension"]),
        "support_tol": float(instance["support_tol"]),
        "rho_diag_reference_linf_gap": float(instance["rho_diag_reference_linf_gap"]),
        "input_offdiagonal_metrics_full": instance["input_offdiagonal_metrics_full"],
        "gram_matrix_abs": np.abs(np.asarray(instance["gram_matrix"], dtype=complex)).tolist(),
        "probabilities_raw": np.asarray(instance["probabilities_raw"], dtype=float).tolist(),
        "probabilities": np.asarray(instance["probabilities"], dtype=float).tolist(),
    }


def solve_route4_strict_nondiagonal_full_primal(
    instance: dict[str, Any],
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """求解 strict non-diagonal route4 的支撑降维 full-primal。"""
    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_reduced = np.asarray(instance["rho_reduced"], dtype=complex)
    q_selected = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    support_dimension = rho_reduced.shape[1]

    size_info = estimate_full_primal_problem_size(num_inputs, num_outputs, support_dimension)
    if (
        max_hermitian_scalar_count is not None
        and size_info["hermitian_scalar_count"] > max_hermitian_scalar_count
    ):
        raise ValueError(
            "The requested strict non-diagonal full primal instance is too large for the configured safeguard: "
            f"{size_info['hermitian_scalar_count']} > {max_hermitian_scalar_count}."
        )
    if size_info["hermitian_scalar_count"] > 400_000:
        warnings.warn(
            "The strict non-diagonal route4 full primal is large and may take a long time to canonicalize "
            f"or solve. Estimated Hermitian scalar count: {size_info['hermitian_scalar_count']}.",
            stacklevel=2,
        )

    lambda_indices = np.array(
        list(iterproduct(range(num_outputs), repeat=num_inputs + 1)),
        dtype=int,
    )
    num_strategies = lambda_indices.shape[0]
    identity = np.eye(support_dimension, dtype=complex)

    operators = {
        (output, strategy_id): cp.Variable((support_dimension, support_dimension), hermitian=True)
        for output in range(num_outputs)
        for strategy_id in range(num_strategies)
    }
    strategy_weights = cp.Variable(num_strategies, nonneg=True)

    objective_expr = 0
    for input_index in range(num_inputs):
        target_outputs = lambda_indices[:, input_index + 1]
        rho_matrix = rho_reduced[input_index]
        for output in range(num_outputs):
            strategy_ids = np.where(target_outputs == output)[0]
            if strategy_ids.size == 0:
                continue
            matrix_sum = _sum_matrices(
                [operators[(output, int(strategy_id))] for strategy_id in strategy_ids],
                support_dimension,
            )
            objective_expr += q_selected[input_index] * cp.real(cp.trace(rho_matrix @ matrix_sum))

    constraints: list[cp.Constraint] = []
    for output in range(num_outputs):
        for strategy_id in range(num_strategies):
            constraints.append(operators[(output, strategy_id)] >> 0)

    for strategy_id in range(num_strategies):
        strategy_sum = _sum_matrices(
            [operators[(output, strategy_id)] for output in range(num_outputs)],
            support_dimension,
        )
        constraints.append(strategy_sum == strategy_weights[strategy_id] * identity)

    total_elements = {
        output: _sum_matrices(
            [operators[(output, strategy_id)] for strategy_id in range(num_strategies)],
            support_dimension,
        )
        for output in range(num_outputs)
    }
    for input_index in range(num_inputs):
        rho_matrix = rho_reduced[input_index]
        for output in range(num_outputs):
            constraints.append(
                cp.real(cp.trace(rho_matrix @ total_elements[output])) == probabilities[input_index, output]
            )

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    solver_name, status = solve_cvxpy_problem(
        problem,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
    )

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None
    if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
        h_min = float(-np.log2(value))

    result = {
        "route": "route4_strict_nondiagonal_full_primal",
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
        "measurement_constraint": "general_Hermitian_PSD",
        "input_model": "fixed_truncated_coherent_states_on_support_subspace",
    }
    result.update(size_info)
    result.update(_instance_summary(instance))
    return result


def run_route4_strict_nondiagonal_full_primal(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
    phase_values: list[float] | tuple[float, ...] | None = None,
    mean_photons_per_mu_label: float = DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
    support_tol: float = 1e-9,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """一键运行 strict non-diagonal route4 的 full-primal。"""
    instance = prepare_route4_strict_nondiagonal_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        full_mu=full_mu,
        phase_values=phase_values,
        mean_photons_per_mu_label=mean_photons_per_mu_label,
        support_tol=support_tol,
    )
    return solve_route4_strict_nondiagonal_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )


def compare_route4_strict_nondiagonal_with_reference(
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = DEFAULT_SHIFT,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    probability_path: str | Path | None = None,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
    phase_values: list[float] | tuple[float, ...] | None = None,
    mean_photons_per_mu_label: float = DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
    support_tol: float = 1e-9,
    max_primal_variables: int | None = 3_000_000,
    max_hermitian_scalar_count: int | None = None,
) -> dict[str, Any]:
    """比较原 route4 对角 primal 与 strict non-diagonal full-primal。

    这个 compare 是当前最重要的接口，因为它只改变 trusted input 模型，
    并允许一般 POVM；与此同时，实验侧 `selected_mu/q/N/Probability.mat`
    全部保持与原 route4 主线一致。
    """
    instance = prepare_route4_strict_nondiagonal_instance(
        num_outputs=num_outputs,
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        cutoff=cutoff,
        prob_floor=prob_floor,
        shift=shift,
        probability_path=probability_path,
        full_mu=full_mu,
        phase_values=phase_values,
        mean_photons_per_mu_label=mean_photons_per_mu_label,
        support_tol=support_tol,
    )

    reference_primal = solve_phaseinsensitive_primal(
        instance["reference_phaseinsensitive_instance"],
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )
    strict_full = solve_route4_strict_nondiagonal_full_primal(
        instance,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )

    reference_p_guess = reference_primal.get("p_guess")
    strict_p_guess = strict_full.get("p_guess")
    p_guess_abs_gap = (
        None
        if (
            reference_p_guess is None
            or strict_p_guess is None
            or not np.isfinite(reference_p_guess)
            or not np.isfinite(strict_p_guess)
        )
        else float(abs(strict_p_guess - reference_p_guess))
    )
    reference_h = reference_primal.get("H_min")
    strict_h = strict_full.get("H_min")
    h_min_gap = (
        None
        if (
            reference_h is None
            or strict_h is None
            or not np.isfinite(reference_h)
            or not np.isfinite(strict_h)
        )
        else float(strict_h - reference_h)
    )

    return {
        "route": "route4_strict_nondiagonal_compare",
        "instance": _instance_summary(instance),
        "reference_phaseinsensitive_primal": reference_primal,
        "strict_nondiagonal_full_primal": strict_full,
        "p_guess_abs_gap": p_guess_abs_gap,
        "H_min_gap_strict_minus_reference": h_min_gap,
    }
