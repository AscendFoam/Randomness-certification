"""
Route 5 解析概率后端失配诊断脚本
=================================

本脚本用于系统诊断 Route 5 在切换到
`analytic_gaussian_rectangles` 概率后端后，为什么 formal SDP
会出现 infeasible。

诊断分成三层：

1. 尾概率检查：
   计算不同 cutoff 下相干态在截断之外丢掉的 Poisson 概率质量，
   粗略判断“cutoff 是否明显过小”。

2. 线性空间兼容性检查：
   对每个输出 bin 的概率列向量 `p_c(s)`，检查它是否落在当前
   trusted-state 模型张成的线性像空间内。
   如果这一步都失败，则说明连“存在某个线性算符 E_c 使
   Tr(E_c rho_s)=p_c(s)”都做不到，formal infeasible 就是必然的。

3. 普通 POVM 可行性检查：
   不考虑 guessing SDP 的二级分解，只检查是否存在一组普通 POVM
   元 `E_c >= 0, sum_c E_c = I` 满足所有统计约束。
   如果连普通 POVM 都 infeasible，则说明问题出在更基础的
   “概率表与 trusted states 不自洽”，而不是 guessing SDP 太强。

脚本还会比较：

- `trace_povm` 后端概率
- `analytic_gaussian_rectangles` 后端概率

以观察它们在不同 cutoff 下的偏差是否收敛。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np

from .hybrid_iq import (
    DEFAULT_PHASE_VALUES,
    power_spaced_bounds,
    reduced_joint_inputs_from_alphas,
    route5_iq_probabilities,
)


def _clean_value(value: Any) -> Any:
    """将 NumPy / CVXPY 结果转换为 JSON 友好的原生类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str | None, payload: Any) -> None:
    """将诊断结果写入 JSON 文件。"""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _poisson_tail_probability(mu: float, cutoff: int) -> float:
    """计算 Poisson(mu) 在 n >= cutoff 区间上的尾概率。"""
    kept = sum(math.exp(-mu) * mu**n / math.factorial(n) for n in range(cutoff))
    return float(1.0 - kept)


def _linear_span_residuals(joint_states: list[np.ndarray], probabilities: np.ndarray) -> dict[str, float]:
    """检查概率列是否落在当前 trusted states 张成的线性像空间中。

    设 `S` 是把每个输入态 `rho_s` 向量化后按行堆叠得到的矩阵，则
    任意可实现的单个测量效应 `E_c` 都必须满足：

        p_c = S vec(E_c^T).

    因此 `p_c` 若不在 `S` 的列空间内，就不可能由当前模型中的任何算符
    产生。这里用最小二乘残差来量化这一失配。
    """
    state_matrix = np.stack([rho.reshape(-1).conj() for rho in joint_states], axis=0)
    residuals: list[float] = []
    relative_residuals: list[float] = []
    for column in range(probabilities.shape[1]):
        solution, *_ = np.linalg.lstsq(state_matrix, probabilities[:, column], rcond=None)
        fitted = state_matrix @ solution
        residual = float(np.linalg.norm(fitted - probabilities[:, column]))
        rel = residual / max(float(np.linalg.norm(probabilities[:, column])), 1e-15)
        residuals.append(residual)
        relative_residuals.append(rel)
    return {
        "state_matrix_rank": int(np.linalg.matrix_rank(state_matrix)),
        "max_abs_fit_residual": float(max(residuals)),
        "mean_abs_fit_residual": float(np.mean(residuals)),
        "max_rel_fit_residual": float(max(relative_residuals)),
        "mean_rel_fit_residual": float(np.mean(relative_residuals)),
    }


def _plain_povm_feasibility(
    joint_states: list[np.ndarray],
    probabilities: np.ndarray,
    solver: str,
) -> dict[str, Any]:
    """检查是否存在普通 POVM `E_c` 复现给定统计。"""
    dimension = joint_states[0].shape[0]
    num_outputs = probabilities.shape[1]
    effects = [cp.Variable((dimension, dimension), hermitian=True) for _ in range(num_outputs)]
    constraints: list[cp.Constraint] = [effect >> 0 for effect in effects]
    constraints.append(sum(effects) == np.eye(dimension))
    for state_index, rho in enumerate(joint_states):
        for output_index in range(num_outputs):
            constraints.append(cp.real(cp.trace(effects[output_index] @ rho)) == probabilities[state_index, output_index])
    problem = cp.Problem(cp.Minimize(0), constraints)

    start = perf_counter()
    try:
        problem.solve(solver=getattr(cp, solver), verbose=False)
        error = None
    except Exception as exc:  # pragma: no cover - diagnostic path
        error = str(exc)
    elapsed = perf_counter() - start
    return {
        "solver": solver,
        "status": None if error is not None else problem.status,
        "elapsed_seconds": float(elapsed),
        "error": error,
    }


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", type=int, nargs="+", required=True)
    parser.add_argument("--feasibility-cutoffs", type=int, nargs="*", default=[])
    parser.add_argument("--radius-values", type=float, nargs="+", required=True)
    parser.add_argument("--phase-values", type=float, nargs="+", default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--num-x-bins", type=int, default=6)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--quadrature-range", type=float, default=1.8)
    parser.add_argument("--boundary-gamma", type=float, default=1.0)
    parser.add_argument("--num-trace-nodes", type=int, default=400)
    parser.add_argument("--feasibility-solver", choices=["MOSEK", "SCS"], default="MOSEK")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """执行解析后端失配诊断。"""
    args = _parse_args()
    x_bounds = power_spaced_bounds(args.num_x_bins, args.quadrature_range, gamma=args.boundary_gamma)
    p_bounds = power_spaced_bounds(args.num_p_bins, args.quadrature_range, gamma=args.boundary_gamma)

    aggregate: dict[str, Any] = {
        "route": "route5_analytic_backend_diagnostics",
        "radius_values": [float(value) for value in args.radius_values],
        "phase_values": [float(value) for value in args.phase_values],
        "num_x_bins": int(args.num_x_bins),
        "num_p_bins": int(args.num_p_bins),
        "quadrature_range": float(args.quadrature_range),
        "boundary_gamma": float(args.boundary_gamma),
        "num_trace_nodes": int(args.num_trace_nodes),
        "cutoff_diagnostics": [],
    }

    max_alpha = max(abs(value) for value in args.radius_values)
    aggregate["tail_summary"] = [
        {
            "alpha_abs": float(alpha_abs),
            "mu": float(alpha_abs**2),
            "cutoff": int(cutoff),
            "poisson_tail_probability": _poisson_tail_probability(alpha_abs**2, int(cutoff)),
        }
        for alpha_abs in sorted(set(abs(value) for value in args.radius_values if abs(value) > 0.0))
        for cutoff in sorted(set(int(value) for value in args.cutoffs))
    ]
    aggregate["max_alpha_abs"] = float(max_alpha)

    for cutoff in sorted(set(int(value) for value in args.cutoffs)):
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
            radius_values=list(args.radius_values),
            phase_values=list(args.phase_values),
        )
        trace_probabilities = route5_iq_probabilities(
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            cutoff,
            args.num_x_bins,
            args.num_p_bins,
            x_bounds,
            p_bounds,
            args.quadrature_range,
            args.num_trace_nodes,
            probability_engine="trace_povm",
        )[0]
        analytic_probabilities = route5_iq_probabilities(
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            cutoff,
            args.num_x_bins,
            args.num_p_bins,
            x_bounds,
            p_bounds,
            args.quadrature_range,
            args.num_trace_nodes,
            probability_engine="analytic_gaussian_rectangles",
        )[0]

        diff = np.abs(trace_probabilities - analytic_probabilities)
        tv = 0.5 * diff.sum(axis=1)
        entry: dict[str, Any] = {
            "cutoff": int(cutoff),
            "num_local_states": int(len(local_alphas)),
            "num_inputs": int(len(joint_states)),
            "num_outputs": int(analytic_probabilities.shape[1]),
            "local_rank": int(local_rank),
            "joint_dim": int(joint_dim),
            "local_operator_span_rank": int(local_operator_span),
            "trace_vs_analytic": {
                "max_abs_error": float(diff.max()),
                "mean_abs_error": float(diff.mean()),
                "max_tv_distance": float(tv.max()),
                "mean_tv_distance": float(tv.mean()),
            },
            "analytic_linear_span": _linear_span_residuals(joint_states, analytic_probabilities),
        }
        if cutoff in {int(value) for value in args.feasibility_cutoffs}:
            entry["analytic_plain_povm_feasibility"] = _plain_povm_feasibility(
                joint_states,
                analytic_probabilities,
                solver=args.feasibility_solver,
            )
        aggregate["cutoff_diagnostics"].append(entry)

    _write_json(args.output_json, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
