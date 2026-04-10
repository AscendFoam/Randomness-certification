"""Route4-ex-constrained 的 Matlab 风格单文件参考脚本。

这个文件的目标不是替代 `prototype.py`，而是给导师提供一个更容易对照
原 Matlab 脚本 `guessprobprimal_phaseinsensitive_original.m` 的阅读入口。

写法原则：

1. 尽量沿用 Matlab 脚本的段落顺序和变量命名；
2. 把核心执行流程写成一条从上到下的“单文件主线”；
3. 只在最底层调用已验证过的 Python 辅助函数，避免重写求解器逻辑；
4. 明确标出这条 constrained 主线相对原 Matlab 的关键改动：
   - 概率表仍然来自 `Probability.mat`；
   - 输入窗口、`q_selected`、粗粒化边界都是固定的；
   - 但 trusted input 不再只保留 Fock 对角，而是完整截断相干态；
   - 正式结果取自 full primal，而非对角 primal。
"""

from __future__ import annotations

import argparse
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import numpy as np

from ..route4_ex.prototype import (
    build_coherent_density_matrices,
    coarse_grain_probability_table_with_edges,
    load_external_probability_table,
    result_to_json,
    solve_route4_ex_diagonal_primal,
    solve_route4_ex_full_primal,
)
from .prototype import (
    DEFAULT_CUTOFF,
    DEFAULT_CUSTOM_EDGES,
    DEFAULT_PHASES,
    DEFAULT_PROBABILITY_PATH,
    DEFAULT_PROB_FLOOR,
    DEFAULT_Q,
    DEFAULT_RADII,
    DEFAULT_SELECTED_MU,
    DEFAULT_VARIABLE_NAME,
    prepare_route4_ex_constrained_instance,
    radii_and_phases_to_alpha_values,
    selected_mu_to_row_indices,
)


def _serialize_complex(alpha: complex) -> dict[str, float]:
    """把复振幅转成便于 JSON 阅读的字典。"""
    return {
        "real": float(np.real(alpha)),
        "imag": float(np.imag(alpha)),
        "abs": float(abs(alpha)),
        "phase": float(np.angle(alpha)),
    }


def _parse_int_list(text: str) -> list[int]:
    """解析逗号分隔的整数列表。"""
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    """解析逗号分隔的浮点列表。"""
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def run_route4_ex_constrained_matlab_style_compare(
    *,
    selected_mu_list: list[int] | tuple[int, ...] = DEFAULT_SELECTED_MU,
    q_selected: list[float] | tuple[float, ...] = DEFAULT_Q,
    cutoff: int = DEFAULT_CUTOFF,
    custom_edges: list[int] | tuple[int, ...] = DEFAULT_CUSTOM_EDGES,
    radii: list[float] | tuple[float, ...] = DEFAULT_RADII,
    phases: list[float] | tuple[float, ...] = DEFAULT_PHASES,
    probability_path: str | Path = DEFAULT_PROBABILITY_PATH,
    variable_name: str | None = DEFAULT_VARIABLE_NAME,
    prob_floor: float | None = DEFAULT_PROB_FLOOR,
    shift: int = 0,
    preferred_solver: str | None = None,
    verbose: bool = False,
    max_primal_variables: int | None = 100_000,
    max_hermitian_scalar_count: int | None = 50_000,
) -> dict[str, Any]:
    """按 Matlab 单文件风格组织 constrained 主线的比较计算。

    逻辑：
        1. 配置 `selected_mu_list / q_selected / M / N / custom_edges`；
        2. 从 `Probability.mat` 读出对应输入行并 coarse-grain，得到 `p`；
        3. 用固定 `radii + phases` 构造相干态振幅 `alpha_values`；
        4. 生成完整 trusted density matrices `rho` 及其 `rho_diag`；
        5. 构造与原 Matlab 同样的策略索引 `LambdaIndices` 供对照；
        6. 调用已验证的 diagonal primal 与 full primal 求解器；
        7. 返回一份同时包含“Matlab 风格中间量”和正式结果的 JSON 字典。

    返回：
        既适合给导师阅读，也适合直接写入结果文件的字典。
    """

    # ===================== 1. 基本参数配置 =====================
    selected_mu_list = [int(mu) for mu in selected_mu_list]
    q_selected_array = np.asarray(q_selected, dtype=float).reshape(-1)
    if q_selected_array.size != len(selected_mu_list):
        raise ValueError("q_selected must have the same length as selected_mu_list.")
    if np.any(q_selected_array < 0.0) or float(q_selected_array.sum()) <= 0.0:
        raise ValueError("q_selected must be non-negative and sum to a positive value.")
    q_selected_array = q_selected_array / q_selected_array.sum()

    M = int(cutoff)
    D = int(len(selected_mu_list))
    custom_edges_array = np.asarray(custom_edges, dtype=int)
    N = int(len(custom_edges_array) - 1)

    if D == 0:
        raise ValueError("selected_mu_list cannot be empty.")
    if N <= 0:
        raise ValueError("custom_edges must define at least one output bin.")

    # ===================== 2. 初始化变量与输入态参数 =====================
    row_indices_zero_based = selected_mu_to_row_indices(selected_mu_list, shift=shift)
    row_indices_one_based = [index + 1 for index in row_indices_zero_based]
    alpha_values = radii_and_phases_to_alpha_values(radii, phases)
    if len(alpha_values) != D:
        raise ValueError("radii/phases length must match selected_mu_list length.")

    # ===================== 3. 构建输入态 rho 和 rho_diag =====================
    rho = build_coherent_density_matrices(alpha_values, M)
    rho_diag = np.real_if_close(np.diagonal(rho, axis1=1, axis2=2)).astype(float)

    # ===================== 4. 读取 Probability.mat 并做 coarse-graining =====================
    probability_table = load_external_probability_table(probability_path, variable_name=variable_name)
    probability_rows_256 = np.asarray(probability_table[row_indices_zero_based, :], dtype=float)
    p, resolved_edges = coarse_grain_probability_table_with_edges(probability_rows_256, custom_edges_array)

    # ===================== 5. 生成 LambdaIndices =====================
    lambda_indices = np.array(list(iterproduct(range(N), repeat=D + 1)), dtype=int)
    num_strategies = int(lambda_indices.shape[0])

    # ===================== 6. 构造统一实例并调用 formal SDP =====================
    instance = prepare_route4_ex_constrained_instance(
        selected_mu_list=selected_mu_list,
        q_selected=q_selected_array.tolist(),
        alpha_values=alpha_values,
        cutoff=M,
        probability_path=probability_path,
        variable_name=variable_name,
        custom_edges=resolved_edges.tolist(),
        prob_floor=prob_floor,
        shift=shift,
    )

    diagonal_result = solve_route4_ex_diagonal_primal(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_primal_variables=max_primal_variables,
    )
    full_result = solve_route4_ex_full_primal(
        instance,
        preferred_solver=preferred_solver,
        verbose=verbose,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )

    # ===================== 7. 输出结果 =====================
    p_guess_abs_gap = None
    h_min_abs_gap = None
    if diagonal_result.get("p_guess") is not None and full_result.get("p_guess") is not None:
        p_guess_abs_gap = abs(float(full_result["p_guess"]) - float(diagonal_result["p_guess"]))
    if diagonal_result.get("H_min") is not None and full_result.get("H_min") is not None:
        h_min_abs_gap = abs(float(full_result["H_min"]) - float(diagonal_result["H_min"]))

    return {
        "route": "route4_ex_constrained_matlab_style_compare",
        "matlab_style_trace": {
            "selected_mu_list": list(selected_mu_list),
            "selected_full_indices_zero_based": list(row_indices_zero_based),
            "selected_full_indices_one_based": list(row_indices_one_based),
            "q_selected": q_selected_array.tolist(),
            "M": M,
            "D": D,
            "N": N,
            "shift": int(shift),
            "custom_edges": resolved_edges.tolist(),
            "block_widths": np.diff(resolved_edges).astype(int).tolist(),
            "radii": [float(radius) for radius in radii],
            "phases": [float(phase) for phase in phases],
            "alpha_values": [_serialize_complex(alpha) for alpha in alpha_values],
            "probability_path": str(Path(probability_path)),
            "variable_name": variable_name,
            "probability_table_shape": list(probability_table.shape),
            "probability_rows_256": probability_rows_256.tolist(),
            "p_matrix": p.tolist(),
            "rho_diag": rho_diag.tolist(),
            "lambda_indices_shape": list(lambda_indices.shape),
            "num_strategies": num_strategies,
        },
        "constrained_delta_vs_original_matlab": {
            "probability_data_source": "same_Probability_mat",
            "selected_mu_list": "same_menu_subset",
            "q_selected": "same_role_as_generation_weights",
            "coarse_graining": "custom_edges_not_equal_blocks",
            "trusted_input_model": "full_truncated_coherent_state_not_only_diagonal_part",
            "formal_solver_target": "general_Hermitian_PSD_full_primal",
        },
        "diagonal_primal": diagonal_result,
        "full_primal": full_result,
        "p_guess_abs_gap": p_guess_abs_gap,
        "H_min_abs_gap": h_min_abs_gap,
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="Matlab-style single-file reference runner for route4-ex-constrained."
    )
    parser.add_argument("--selected-mu", nargs="+", type=int, default=list(DEFAULT_SELECTED_MU))
    parser.add_argument("--q-values", nargs="+", type=float, default=list(DEFAULT_Q))
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--custom-edges", type=_parse_int_list, default=list(DEFAULT_CUSTOM_EDGES))
    parser.add_argument("--radii", type=_parse_float_list, default=list(DEFAULT_RADII))
    parser.add_argument("--phases", type=_parse_float_list, default=[float(value) for value in DEFAULT_PHASES])
    parser.add_argument("--probability-path", type=str, default=str(DEFAULT_PROBABILITY_PATH))
    parser.add_argument("--variable-name", type=str, default=DEFAULT_VARIABLE_NAME)
    parser.add_argument("--prob-floor", type=float, default=DEFAULT_PROB_FLOOR)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-primal-variables", type=int, default=100_000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=50_000)
    args = parser.parse_args()

    prob_floor = None if args.prob_floor <= 0 else args.prob_floor
    result = run_route4_ex_constrained_matlab_style_compare(
        selected_mu_list=list(args.selected_mu),
        q_selected=list(args.q_values),
        cutoff=args.cutoff,
        custom_edges=list(args.custom_edges),
        radii=list(args.radii),
        phases=list(args.phases),
        probability_path=args.probability_path,
        variable_name=args.variable_name,
        prob_floor=prob_floor,
        shift=args.shift,
        preferred_solver=args.solver,
        verbose=args.verbose,
        max_primal_variables=args.max_primal_variables,
        max_hermitian_scalar_count=args.max_hermitian_scalar_count,
    )
    print(result_to_json(result))


if __name__ == "__main__":
    main()

