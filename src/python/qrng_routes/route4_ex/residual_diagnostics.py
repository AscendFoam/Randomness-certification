"""Route4-ex 的约束残差与可行性余量诊断脚本。

本脚本用于回答一个比“值多高”更细的问题：

- 某个 `MOSEK optimal` 高值点到底是不是数值上足够干净？

它会对指定半径点重新构造 full-primal 问题，并在求解后统计：

1. PSD 变量的最小特征值；
2. PSD 约束 violation；
3. 完备性约束 `sum_c M_{c,lambda} = s_lambda I` 的残差；
4. 观测概率匹配约束的残差；
5. strategy weights 的归一化情况。

脚本同时支持失败点诊断：

- 若求解失败，也会把实例信息与错误原因写入输出 JSON，
  便于后续定位问题，而不是直接中断。
"""

from __future__ import annotations

import argparse
import json
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np

from qrng_routes.common import solve_cvxpy_problem

from .prototype import _sum_matrices, prepare_route4_ex_external_instance


def _parse_float_triplet(text: str) -> list[float]:
    """解析长度为 3 的半径列表。"""
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("Expected exactly three comma-separated floats.")
    return values


def _parse_int_list(text: str) -> list[int]:
    """解析逗号分隔的整数边界或行号列表。"""
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def _clean(value: Any) -> Any:
    """把 NumPy 类型转成 JSON 兼容的基础类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _fro_norm(matrix: np.ndarray) -> float:
    """计算矩阵的 Frobenius 范数。"""
    return float(np.linalg.norm(matrix, ord="fro"))


def _real_eig_min(matrix: np.ndarray) -> float:
    """计算 Hermitian 化后矩阵的最小实特征值。"""
    eigvals = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    return float(np.min(np.real_if_close(eigvals)))


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    """把诊断结果写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_clean))


def main() -> None:
    """执行单点残差与可行性余量诊断。

    逻辑：
        1. 从指定半径、窗口行号和边界构造 external instance；
        2. 显式重建 full-primal 变量与约束；
        3. 用指定求解器求解；
        4. 若成功，则统计各类约束 violation 与矩阵余量；
        5. 若失败，则仍输出错误信息和实例摘要。

    输出：
        JSON 结果中最关键的字段是 `residual_summary`，它集中给出：
        - PSD 最小特征值余量
        - 完备性残差
        - 测量匹配残差
        - strategy weights 的和与最小值
    """
    parser = argparse.ArgumentParser(description="Diagnose residuals and feasibility margins for a route4-ex full-primal point.")
    parser.add_argument("--probability-path", type=str, required=True)
    parser.add_argument("--variable-name", type=str, default="Probability")
    parser.add_argument("--row-indices", type=str, default="5,6,7")
    parser.add_argument("--edges", type=str, default="0,121,132,256")
    parser.add_argument("--q-config-json", type=str, default="[1,0,0]")
    parser.add_argument("--radii", type=str, required=True)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--solver", type=str, default="MOSEK")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    row_indices = _parse_int_list(args.row_indices)
    edges = _parse_int_list(args.edges)
    q_selected = json.loads(args.q_config_json)
    radii = _parse_float_triplet(args.radii)
    alpha_values = [
        radii[0] * np.exp(1j * 0.0),
        radii[1] * np.exp(1j * (np.pi / 2.0)),
        radii[2] * np.exp(1j * np.pi),
    ]

    instance = prepare_route4_ex_external_instance(
        alpha_values=alpha_values,
        q_selected=q_selected,
        cutoff=int(args.cutoff),
        probability_path=args.probability_path,
        num_outputs=len(edges) - 1,
        row_indices=row_indices,
        prob_floor=float(args.prob_floor),
        variable_name=args.variable_name,
        already_coarse=False,
        custom_edges=edges,
    )

    probabilities = np.asarray(instance["probabilities"], dtype=float)
    rho_matrices = np.asarray(instance["rho_matrices"], dtype=complex)
    q_vector = np.asarray(instance["q_selected"], dtype=float)
    num_inputs, num_outputs = probabilities.shape
    cutoff = rho_matrices.shape[1]

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
            objective_expr += q_vector[input_index] * cp.real(cp.trace(rho_matrix @ matrix_sum))

    psd_constraints: list[cp.Constraint] = []
    completeness_constraints: list[cp.Constraint] = []
    measurement_constraints: list[cp.Constraint] = []

    for output in range(num_outputs):
        for strategy_id in range(num_strategies):
            constraint = operators[(output, strategy_id)] >> 0
            psd_constraints.append(constraint)

    for strategy_id in range(num_strategies):
        strategy_sum = _sum_matrices(
            [operators[(output, strategy_id)] for output in range(num_outputs)],
            cutoff,
        )
        completeness_constraints.append(strategy_sum == strategy_weights[strategy_id] * identity)

    total_elements = {
        output: _sum_matrices([operators[(output, strategy_id)] for strategy_id in range(num_strategies)], cutoff)
        for output in range(num_outputs)
    }
    for input_index in range(num_inputs):
        rho_matrix = rho_matrices[input_index]
        for output in range(num_outputs):
            measurement_constraints.append(
                cp.real(cp.trace(rho_matrix @ total_elements[output]))
                == probabilities[input_index, output]
            )

    constraints = psd_constraints + completeness_constraints + measurement_constraints
    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    output_path = Path(args.output_path)
    try:
        solver_name, status = solve_cvxpy_problem(
            problem,
            preferred_solver=args.solver,
            solver_options=None,
            verbose=bool(args.verbose),
            warm_start=False,
        )
    except Exception as exc:  # noqa: BLE001
        failure_result = {
            "radii": radii,
            "alpha_values": [
                {"real": float(np.real(alpha)), "imag": float(np.imag(alpha)), "abs": float(abs(alpha))}
                for alpha in alpha_values
            ],
            "row_indices": row_indices,
            "edges": edges,
            "solver": str(args.solver),
            "status": None,
            "p_guess": None,
            "H_min": None,
            "distribution_only_p_guess": float(instance["distribution_only_p_guess"]),
            "distribution_only_H_min": float(-np.log2(float(instance["distribution_only_p_guess"]))),
            "regularized_entries": int(instance["regularized_entries"]),
            "input_offdiagonal_metrics": instance["input_offdiagonal_metrics"],
            "error": type(exc).__name__,
            "error_message": str(exc),
            "residual_summary": None,
            "measurement_direct_abs_matrix": None,
        }
        _write_result(output_path, failure_result)
        print(
            json.dumps(
                {
                    "error": failure_result["error"],
                    "error_message": failure_result["error_message"],
                },
                indent=2,
                ensure_ascii=False,
                default=_clean,
            )
        )
        return

    value = None if problem.value is None else float(np.real_if_close(problem.value))
    h_min = None if value is None or value <= 0 else float(-np.log2(value))

    op_values = {
        key: np.asarray(var.value, dtype=complex)
        for key, var in operators.items()
        if var.value is not None
    }
    weight_values = None if strategy_weights.value is None else np.asarray(strategy_weights.value, dtype=float)

    psd_min_eigs = [_real_eig_min(mat) for mat in op_values.values()]
    psd_viols = [float(np.max(np.atleast_1d(c.violation()))) for c in psd_constraints]
    completeness_viols = [float(np.max(np.atleast_1d(c.violation()))) for c in completeness_constraints]
    measurement_viols = [float(np.max(np.atleast_1d(c.violation()))) for c in measurement_constraints]

    completeness_direct = []
    if weight_values is not None:
        for strategy_id in range(num_strategies):
            strategy_sum_value = sum(op_values[(output, strategy_id)] for output in range(num_outputs))
            residual = strategy_sum_value - weight_values[strategy_id] * identity
            completeness_direct.append(_fro_norm(residual))

    measurement_direct = []
    for input_index in range(num_inputs):
        rho_matrix = rho_matrices[input_index]
        row = []
        for output in range(num_outputs):
            total_value = sum(op_values[(output, strategy_id)] for strategy_id in range(num_strategies))
            lhs = float(np.real_if_close(np.trace(rho_matrix @ total_value)))
            row.append(abs(lhs - probabilities[input_index, output]))
        measurement_direct.append(row)

    result = {
        "radii": radii,
        "alpha_values": [
            {"real": float(np.real(alpha)), "imag": float(np.imag(alpha)), "abs": float(abs(alpha))}
            for alpha in alpha_values
        ],
        "row_indices": row_indices,
        "edges": edges,
        "solver": solver_name,
        "status": status,
        "p_guess": value,
        "H_min": h_min,
        "distribution_only_p_guess": float(instance["distribution_only_p_guess"]),
        "distribution_only_H_min": float(-np.log2(float(instance["distribution_only_p_guess"]))),
        "regularized_entries": int(instance["regularized_entries"]),
        "input_offdiagonal_metrics": instance["input_offdiagonal_metrics"],
        "residual_summary": {
            "psd_min_eig_min": min(psd_min_eigs) if psd_min_eigs else None,
            "psd_min_eig_max_neg_part": max(max(0.0, -x) for x in psd_min_eigs) if psd_min_eigs else None,
            "psd_violation_max": max(psd_viols) if psd_viols else None,
            "psd_violation_mean": float(np.mean(psd_viols)) if psd_viols else None,
            "completeness_violation_max": max(completeness_viols) if completeness_viols else None,
            "completeness_violation_mean": float(np.mean(completeness_viols)) if completeness_viols else None,
            "completeness_direct_fro_max": max(completeness_direct) if completeness_direct else None,
            "measurement_violation_max": max(measurement_viols) if measurement_viols else None,
            "measurement_violation_mean": float(np.mean(measurement_viols)) if measurement_viols else None,
            "measurement_direct_abs_max": max(max(row) for row in measurement_direct) if measurement_direct else None,
            "measurement_direct_abs_mean": float(np.mean(np.asarray(measurement_direct))) if measurement_direct else None,
            "strategy_weight_sum": None if weight_values is None else float(weight_values.sum()),
            "strategy_weight_min": None if weight_values is None else float(np.min(weight_values)),
        },
        "measurement_direct_abs_matrix": measurement_direct,
    }

    _write_result(output_path, result)
    print(json.dumps(result["residual_summary"], indent=2, ensure_ascii=False, default=_clean))


if __name__ == "__main__":
    main()
