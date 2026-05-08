"""
Route 5 quadrature 节点数收敛扫描
=================================

这个脚本用于严谨检查 `num_quadrature_nodes` 对 Route 5 结果的影响。

扫描策略：
1. 固定同一组 trusted alphabet、同一组 IQ 分箱、同一组 SDP 配置；
2. 只改变 `num_quadrature_nodes`；
3. 对每个节点数同时记录：
   - 概率层诊断（raw_best_H_min、row sums、top raw targets）
   - formal SDP 结果（status、p_guess、H_min、best target、target_scan）
4. 每个节点完成后立即写入单点 JSON 和聚合 JSON，便于断点续跑。

这份脚本的目标不是继续追更高熵，而是判断：

- 之前的高分结果是否对节点数过于敏感；
- 概率积分是否已经数值收敛；
- formal H_min 在节点数增加后是稳定、漂移还是坍塌。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .hybrid_iq import (
    DEFAULT_PROBABILITY_ENGINE,
    DEFAULT_PHASE_VALUES,
    SUPPORTED_PROBABILITY_ENGINES,
    certify_target_inputs,
    power_spaced_bounds,
    reduced_joint_inputs_from_alphas,
    route5_iq_probabilities,
)


def _clean_value(value: Any) -> Any:
    """将 NumPy 数值清理为 JSON 可序列化的原生类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _build_solver_options(args: argparse.Namespace) -> dict[str, dict] | None:
    """根据命令行参数构造 CVXPY 求解器选项。"""
    scs_options = {
        key: value
        for key, value in {
            "max_iters": args.scs_max_iters,
            "eps_abs": args.scs_eps_abs,
            "eps_rel": args.scs_eps_rel,
            "eps_infeas": args.scs_eps_infeas,
        }.items()
        if value is not None
    }

    mosek_params: dict[str, Any] = {}
    if args.mosek_num_threads is not None:
        mosek_params["MSK_IPAR_NUM_THREADS"] = int(args.mosek_num_threads)
    if args.mosek_solve_form is not None:
        solve_forms = {
            "dual": "MSK_SOLVE_DUAL",
            "primal": "MSK_SOLVE_PRIMAL",
            "free": "MSK_SOLVE_FREE",
        }
        mosek_params["MSK_IPAR_INTPNT_SOLVE_FORM"] = solve_forms[args.mosek_solve_form]

    mosek_options = {
        key: value
        for key, value in {
            "mosek_params": mosek_params if mosek_params else None,
            "eps": args.mosek_eps,
            "accept_unknown": True if args.mosek_accept_unknown else None,
            "save_file": args.mosek_save_file,
        }.items()
        if value is not None
    }

    solver_options: dict[str, dict] = {}
    if scs_options:
        solver_options["SCS"] = scs_options
    if mosek_options:
        solver_options["MOSEK"] = mosek_options
    return solver_options or None


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--radius-values", type=float, nargs="+", required=True)
    parser.add_argument("--phase-values", type=float, nargs="+", default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--num-x-bins", type=int, default=6)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--quadrature-range", type=float, default=1.8)
    parser.add_argument("--boundary-gamma", type=float, default=1.0)
    parser.add_argument(
        "--probability-engine",
        choices=list(SUPPORTED_PROBABILITY_ENGINES),
        default=DEFAULT_PROBABILITY_ENGINE,
    )
    parser.add_argument("--num-nodes-values", type=int, nargs="+", required=True)
    parser.add_argument("--max-inputs", type=int, default=3)
    parser.add_argument("--top-raw-k", type=int, default=8)
    parser.add_argument("--solver", type=str, default="MOSEK")
    parser.add_argument("--mosek-num-threads", type=int)
    parser.add_argument("--mosek-solve-form", choices=["dual", "primal", "free"])
    parser.add_argument("--mosek-eps", type=float)
    parser.add_argument("--mosek-accept-unknown", action="store_true")
    parser.add_argument("--mosek-save-file", type=str)
    parser.add_argument("--scs-max-iters", type=int)
    parser.add_argument("--scs-eps-abs", type=float)
    parser.add_argument("--scs-eps-rel", type=float)
    parser.add_argument("--scs-eps-infeas", type=float)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """执行节点数收敛扫描。"""
    args = _parse_args()
    aggregate_path = Path(args.output_json)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    solver_options = _build_solver_options(args)

    x_bounds = power_spaced_bounds(args.num_x_bins, args.quadrature_range, gamma=args.boundary_gamma)
    p_bounds = power_spaced_bounds(args.num_p_bins, args.quadrature_range, gamma=args.boundary_gamma)

    (
        joint_states,
        labels,
        joint_basis,
        local_alphas,
        local_rank,
        joint_dim,
        local_operator_span,
    ) = reduced_joint_inputs_from_alphas(
        args.cutoff,
        radius_values=list(args.radius_values),
        phase_values=list(args.phase_values),
    )

    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
    else:
        aggregate = {
            "route": "route5_num_quadrature_nodes_convergence_scan",
            "cutoff": int(args.cutoff),
            "radius_values": list(args.radius_values),
            "phase_values": list(args.phase_values),
            "num_local_states": len(local_alphas),
            "num_inputs": len(joint_states),
            "num_x_bins": int(args.num_x_bins),
            "num_p_bins": int(args.num_p_bins),
            "num_outputs": int(args.num_x_bins * args.num_p_bins),
            "quadrature_range": float(args.quadrature_range),
            "boundary_gamma": float(args.boundary_gamma),
            "probability_engine": str(args.probability_engine),
            "x_bounds": x_bounds.tolist(),
            "p_bounds": p_bounds.tolist(),
            "max_inputs": int(args.max_inputs),
            "top_raw_k": int(args.top_raw_k),
            "local_rank": int(local_rank),
            "local_operator_span_rank": int(local_operator_span),
            "local_operator_space_dim": int(local_rank**2),
            "joint_dim": int(joint_dim),
            "operator_span_rank": int(np.linalg.matrix_rank(np.stack([state.reshape(-1) for state in joint_states]))),
            "operator_space_dim": int(joint_dim**2),
            "results": [],
        }

    done_by_nodes = {
        int(entry["num_quadrature_nodes"]): dict(entry)
        for entry in aggregate.get("results", [])
    }

    for num_nodes in sorted(set(int(value) for value in args.num_nodes_values if int(value) > 0)):
        if num_nodes in done_by_nodes:
            continue

        probability_start = perf_counter()
        probabilities, output_labels, x_bounds_out, p_bounds_out = route5_iq_probabilities(
            joint_states,
            labels,
            joint_basis,
            local_alphas,
            args.cutoff,
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=args.quadrature_range,
            num_quadrature_nodes=num_nodes,
            probability_engine=args.probability_engine,
        )
        probability_elapsed = perf_counter() - probability_start

        raw_h = -np.log2(np.maximum(probabilities.max(axis=1), 1e-15))
        raw_order = list(np.argsort(-raw_h))
        candidate_order = raw_order[: args.max_inputs]
        top_raw = raw_order[: args.top_raw_k]

        formal_start = perf_counter()
        best, target_scan = certify_target_inputs(
            joint_states,
            probabilities,
            labels,
            local_alphas,
            target_indices=candidate_order,
            preferred_solver=args.solver,
            solver_options=solver_options,
            verbose=False,
        )
        formal_elapsed = perf_counter() - formal_start

        entry = {
            "num_quadrature_nodes": int(num_nodes),
            "effective_num_quadrature_nodes": (
                int(num_nodes) if args.probability_engine == "trace_povm" else None
            ),
            "probability_elapsed_seconds": float(probability_elapsed),
            "formal_elapsed_seconds": float(formal_elapsed),
            "raw_best_H_min": float(raw_h[raw_order[0]]),
            "raw_best_target_index": int(raw_order[0]),
            "raw_best_target": labels[raw_order[0]],
            "raw_top_targets": [
                {
                    "target_index": int(index),
                    "target_input": labels[index],
                    "raw_H_min": float(raw_h[index]),
                }
                for index in top_raw
            ],
            "row_sum_min": float(probabilities.sum(axis=1).min()),
            "row_sum_max": float(probabilities.sum(axis=1).max()),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "best_target_index": best["target_index"],
            "best_target_input": best["target_input"],
            "status": best["status"],
            "solver": best["solver"],
            "p_guess": best["p_guess"],
            "H_min": best["H_min"],
            "num_inputs_certified": len(target_scan),
            "target_scan": target_scan,
            "output_labels": output_labels,
            "x_bounds": x_bounds_out.tolist(),
            "p_bounds": p_bounds_out.tolist(),
        }

        output_json = result_dir / f"nodes_{num_nodes}.json"
        _write_json(output_json, entry)
        entry["output_json"] = str(output_json)
        done_by_nodes[num_nodes] = entry
        aggregate["results"] = [done_by_nodes[key] for key in sorted(done_by_nodes)]
        aggregate["best_formal_result"] = max(
            aggregate["results"],
            key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
        )
        _write_json(aggregate_path, aggregate)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
