"""Route4-ex-constrained 的核心封装。

这条路线不是新的大规模搜索框架，而是对现有 `route4_ex` 做“收缩”后的核心版：

1. 只保留外部概率表 `Probability.mat` 的接入方式；
2. 只保留固定输入窗口 + 固定 coarse-graining + 固定 trusted alphas；
3. 仍然使用 non-diagonal coherent trusted inputs 与 full primal 认证；
4. 不再包含自由半径扫描、相位图样搜索、窗口批量搜索等扩展层。

因此，本文件的职责是把此前已经验证过的 `route4_ex` 主线，
整理成一个更接近“单一核心模型”的可复现接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..route4_ex.prototype import (
    compare_route4_ex_external_diagonal_full,
    prepare_route4_ex_external_instance,
    result_to_json,
    run_route4_ex_external_full_primal,
)

FULL_MU = [0, 20, 40, 60, 80, 100, 120, 140, 160]
DEFAULT_SELECTED_MU = [100, 120, 140]
DEFAULT_Q = [1.0, 0.0, 0.0]
DEFAULT_CUTOFF = 6
DEFAULT_CUSTOM_EDGES = [0, 121, 132, 256]
DEFAULT_PHASES = [0.0, float(np.pi / 2.0), float(np.pi)]
DEFAULT_RADII = [0.54, 0.66, 0.72]
DEFAULT_PROB_FLOOR = 1e-12
DEFAULT_VARIABLE_NAME = "Probability"


def _default_probability_path() -> Path:
    """返回 `Probability.mat` 的默认路径。"""
    return Path(__file__).resolve().parents[3] / "matlab" / "Probability.mat"


DEFAULT_PROBABILITY_PATH = _default_probability_path()


def radii_and_phases_to_alpha_values(
    radii: list[float] | tuple[float, ...],
    phases: list[float] | tuple[float, ...],
) -> list[complex]:
    """把固定半径与相位转换成 trusted coherent 振幅列表。

    参数：
        radii：每个输入态对应的振幅模长。
        phases：每个输入态对应的相位。

    返回：
        复数振幅列表 `alpha_values`。
    """
    if len(radii) == 0:
        raise ValueError("radii cannot be empty.")
    if len(radii) != len(phases):
        raise ValueError("radii and phases must have the same length.")
    resolved: list[complex] = []
    for radius, phase in zip(radii, phases):
        if float(radius) < 0.0:
            raise ValueError("radii must be non-negative.")
        resolved.append(complex(float(radius) * np.exp(1j * float(phase))))
    return resolved


DEFAULT_ALPHA_VALUES = radii_and_phases_to_alpha_values(DEFAULT_RADII, DEFAULT_PHASES)


def selected_mu_to_row_indices(
    selected_mu_list: list[int] | tuple[int, ...],
    *,
    full_mu: list[int] | tuple[int, ...] = FULL_MU,
    shift: int = 0,
) -> list[int]:
    """把选定光强标签映射到外部概率表的行索引。

    参数：
        selected_mu_list：当前 constrained 实例选用的光强标签列表。
        full_mu：概率表完整光强菜单，默认使用原 route4 的九个标签。
        shift：对最终索引施加的整体偏移，保留原 route4 的兼容口径。

    返回：
        与 `selected_mu_list` 一一对应的零基行索引列表。
    """
    if len(selected_mu_list) == 0:
        raise ValueError("selected_mu_list cannot be empty.")
    full_mu_list = list(full_mu)
    row_indices: list[int] = []
    for mu in selected_mu_list:
        if int(mu) not in full_mu_list:
            raise ValueError(f"Selected mu {mu} is not contained in full_mu={full_mu_list}.")
        row_indices.append(full_mu_list.index(int(mu)) + int(shift))
    return row_indices


def _serialize_complex(alpha: complex) -> dict[str, float]:
    """把单个复振幅转成便于结果文件阅读的字典。"""
    return {
        "real": float(np.real(alpha)),
        "imag": float(np.imag(alpha)),
        "abs": float(abs(alpha)),
        "phase": float(np.angle(alpha)),
    }


def summarize_route4_ex_constrained_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """提取 constrained 实例的轻量摘要。

    参数：
        instance：由 `prepare_route4_ex_constrained_instance(...)` 返回的完整实例。

    返回：
        适合直接输出到 JSON 的摘要字典。
    """
    raw_p_guess = float(instance["distribution_only_p_guess_raw"])
    reg_p_guess = float(instance["distribution_only_p_guess"])
    return {
        "route": "route4_ex_constrained_instance",
        "selected_mu_list": list(instance["selected_mu_list"]),
        "alpha_values": [_serialize_complex(alpha) for alpha in instance["alpha_values"]],
        "q_selected": np.asarray(instance["q_selected"], dtype=float).tolist(),
        "num_inputs": int(instance["num_inputs"]),
        "num_outputs": int(instance["num_outputs"]),
        "cutoff": int(instance["cutoff"]),
        "probability_path": str(instance["external_probability_path"]),
        "variable_name": instance.get("external_variable_name"),
        "external_row_indices": list(instance["external_row_indices"]),
        "coarse_grain_edges": np.asarray(instance["coarse_grain_edges"], dtype=int).tolist(),
        "prob_floor": None if instance["prob_floor"] is None else float(instance["prob_floor"]),
        "distribution_only_p_guess_raw": raw_p_guess,
        "distribution_only_H_min_raw": float(-np.log2(raw_p_guess)) if raw_p_guess > 0 else None,
        "distribution_only_p_guess": reg_p_guess,
        "distribution_only_H_min": float(-np.log2(reg_p_guess)) if reg_p_guess > 0 else None,
        "probabilities": np.asarray(instance["probabilities"], dtype=float).tolist(),
        "input_offdiagonal_metrics": instance["input_offdiagonal_metrics"],
        "constrained_profile": "fixed_window_fixed_edges_fixed_alphas",
    }


def prepare_route4_ex_constrained_instance(
    *,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    alpha_values: list[complex] | tuple[complex, ...] = DEFAULT_ALPHA_VALUES,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path = DEFAULT_PROBABILITY_PATH,
    variable_name: str | None = DEFAULT_VARIABLE_NAME,
    custom_edges: list[int] | tuple[int, ...] = DEFAULT_CUSTOM_EDGES,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = 0,
) -> dict[str, Any]:
    """构造 route4-ex-constrained 的单个固定实例。

    逻辑：
        1. 用 `selected_mu_list` 决定从 `Probability.mat` 里取哪些输入行；
        2. 使用固定 `alpha_values` 作为 trusted coherent inputs；
        3. 使用固定 `custom_edges` 对 256 维原始直方图做 coarse-graining；
        4. 返回可直接送入 full primal 或 diagonal/full compare 的统一实例。

    参数：
        selected_mu_list：从原 route4 光强菜单中选取的输入窗口。
        q_selected：生成轮权重。
        alpha_values：固定 trusted coherent 振幅列表。
        cutoff：trusted input 所用的 Fock 截断维数。
        probability_path：外部概率表路径，默认指向 `Probability.mat`。
        variable_name：`.mat` 文件中的变量名。
        custom_edges：固定 coarse-graining 边界。
        prob_floor：概率正则化地板。
        shift：行索引整体偏移。

    返回：
        route4-ex 的统一实例字典，并额外打上 constrained 标记字段。
    """
    resolved_mu = [int(mu) for mu in selected_mu_list]
    resolved_alphas = [complex(alpha) for alpha in alpha_values]
    if len(resolved_mu) != len(resolved_alphas):
        raise ValueError("selected_mu_list and alpha_values must have the same length.")

    row_indices = selected_mu_to_row_indices(resolved_mu, shift=shift)
    instance = prepare_route4_ex_external_instance(
        alpha_values=resolved_alphas,
        q_selected=list(q_selected),
        cutoff=int(cutoff),
        probability_path=probability_path,
        num_outputs=len(custom_edges) - 1,
        row_indices=row_indices,
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=False,
        custom_edges=list(custom_edges),
    )
    instance["route"] = "route4_ex_constrained"
    instance["selected_mu_list"] = resolved_mu
    instance["constrained_profile"] = "fixed_window_fixed_edges_fixed_alphas"
    return instance


def run_route4_ex_constrained_full_primal(
    *,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    alpha_values: list[complex] | tuple[complex, ...] = DEFAULT_ALPHA_VALUES,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path = DEFAULT_PROBABILITY_PATH,
    variable_name: str | None = DEFAULT_VARIABLE_NAME,
    custom_edges: list[int] | tuple[int, ...] = DEFAULT_CUSTOM_EDGES,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = 0,
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_hermitian_scalar_count: int | None = 50_000,
) -> dict[str, Any]:
    """运行 constrained 主线上的 full primal。

    返回：
        在 `route4_ex` full primal 结果基础上，附加 constrained 配置标签后的结果字典。
    """
    instance = prepare_route4_ex_constrained_instance(
        selected_mu_list=selected_mu_list,
        q_selected=q_selected,
        alpha_values=alpha_values,
        cutoff=cutoff,
        probability_path=probability_path,
        variable_name=variable_name,
        custom_edges=custom_edges,
        prob_floor=prob_floor,
        shift=shift,
    )
    result = run_route4_ex_external_full_primal(
        alpha_values=[complex(alpha) for alpha in alpha_values],
        q_selected=list(q_selected),
        cutoff=int(cutoff),
        probability_path=probability_path,
        num_outputs=len(custom_edges) - 1,
        row_indices=selected_mu_to_row_indices(selected_mu_list, shift=shift),
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=False,
        custom_edges=list(custom_edges),
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )
    result["route"] = "route4_ex_constrained_full_primal"
    result["selected_mu_list"] = [int(mu) for mu in selected_mu_list]
    result["constrained_profile"] = "fixed_window_fixed_edges_fixed_alphas"
    result["instance_summary"] = summarize_route4_ex_constrained_instance(instance)
    return result


def compare_route4_ex_constrained_diagonal_full(
    *,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    alpha_values: list[complex] | tuple[complex, ...] = DEFAULT_ALPHA_VALUES,
    cutoff: int = DEFAULT_CUTOFF,
    probability_path: str | Path = DEFAULT_PROBABILITY_PATH,
    variable_name: str | None = DEFAULT_VARIABLE_NAME,
    custom_edges: list[int] | tuple[int, ...] = DEFAULT_CUSTOM_EDGES,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = 0,
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = 100_000,
    max_hermitian_scalar_count: int | None = 50_000,
) -> dict[str, Any]:
    """比较 constrained 实例上的 diagonal primal 与 full primal。

    这相当于给 route4-ex 的核心版提供一个最小诊断接口：
    同一张概率表、同一组 fixed alphas、同一组 fixed edges 下，
    看 full primal 是否仍然能保持显著高于 diagonal 基线的认证值。
    """
    result = compare_route4_ex_external_diagonal_full(
        alpha_values=[complex(alpha) for alpha in alpha_values],
        q_selected=list(q_selected),
        cutoff=int(cutoff),
        probability_path=probability_path,
        num_outputs=len(custom_edges) - 1,
        row_indices=selected_mu_to_row_indices(selected_mu_list, shift=shift),
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=False,
        custom_edges=list(custom_edges),
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )
    result["route"] = "route4_ex_constrained_diagonal_full_compare"
    result["selected_mu_list"] = [int(mu) for mu in selected_mu_list]
    result["constrained_profile"] = "fixed_window_fixed_edges_fixed_alphas"
    return result

