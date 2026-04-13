"""
Route 5 固定分区半径 formal 搜索
================================

这个脚本服务于 Route 5 的“局部精修”阶段：

1. 分区族已经基本固定，例如当前已知较强的
   `num_x_bins = 6`, `num_p_bins = 2`,
   `quadrature_range = 1.8`, `boundary_gamma = 1.0`。
2. 已知 raw 指标与 formal 指标存在明显错位，因此不再先按
   `raw_best_H_min` 筛选少数候选，而是直接对给定的半径窗口做 formal 认证。
3. 每个半径候选都会调用 `run_route5(...)` 完整求解，并落盘保存单点结果。

典型用途：

```bash
python -m qrng_routes.route5.fixed_partition_radius_search \
  --cutoff 4 \
  --phase-values 0.0 0.78539816339 1.57079632679 2.35619449019 \
                 3.14159265359 3.92699081699 4.71238898038 5.49778714378 \
  --r1-values 0.85 0.875 0.9 \
  --r2-values 1.20 1.225 1.25 1.275 1.30 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-range 1.8 \
  --boundary-gamma 1.0 \
  --num-quadrature-nodes 12 \
  --max-inputs 3 \
  --solver MOSEK \
  --output-json output/qrng_routes/route5_fixed_partition_radius_search.json \
  --result-dir output/qrng_routes/route5_fixed_partition_radius_search
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_iq import DEFAULT_PHASE_VALUES, run_route5


def _clean_value(value: Any) -> Any:
    """将 NumPy 数值转换为 JSON 友好的原生类型。"""
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


def _unique_sorted(values: list[float], tol: float = 1e-12) -> list[float]:
    """对浮点列表去重并升序排序。"""
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if any(abs(value - existing) <= tol for existing in unique):
            continue
        unique.append(value)
    return unique


def _candidate_key(radii: list[float]) -> str:
    """为半径三元组生成稳定的文件名键。"""
    return "r" + "_".join(f"{value:.4f}" for value in radii)


def _build_radius_candidates(r1_values: list[float], r2_values: list[float]) -> list[list[float]]:
    """生成所有合法的 `{0, r1, r2}` 半径候选。"""
    first = [value for value in _unique_sorted(r1_values) if value > 0.0]
    second = [value for value in _unique_sorted(r2_values) if value > 0.0]
    out: list[list[float]] = []
    for r1 in first:
        for r2 in second:
            if r2 <= r1 + 1e-12:
                continue
            out.append([0.0, float(r1), float(r2)])
    return out


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


def _result_summary(result: dict, radii: list[float], output_json: str) -> dict:
    """提取单个 formal 认证结果的摘要。"""
    return {
        "candidate_key": _candidate_key(radii),
        "radii": list(radii),
        "output_json": output_json,
        "H_min": result["H_min"],
        "p_guess": result["p_guess"],
        "status": result["status"],
        "solver": result["solver"],
        "target_input": result["target_input"],
        "raw_best_H_min": result["raw_best_H_min"],
        "raw_best_target": result["raw_best_target"],
        "num_inputs_certified": result["num_inputs_certified"],
        "num_x_bins": result["num_x_bins"],
        "num_p_bins": result["num_p_bins"],
        "quadrature_range": result["quadrature_range"],
        "boundary_gamma": result["boundary_gamma"],
        "x_bounds": result["x_bounds"],
        "p_bounds": result["p_bounds"],
        "num_local_states": result["num_local_states"],
        "local_rank": result["local_rank"],
        "operator_span_rank": result["operator_span_rank"],
    }


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--phase-values", type=float, nargs="+", default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--r1-values", type=float, nargs="+", required=True)
    parser.add_argument("--r2-values", type=float, nargs="+", required=True)
    parser.add_argument("--num-x-bins", type=int, default=6)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--quadrature-range", type=float, default=1.8)
    parser.add_argument("--boundary-gamma", type=float, default=1.0)
    parser.add_argument("--num-quadrature-nodes", type=int, default=12)
    parser.add_argument("--max-inputs", type=int, default=3)
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
    """固定分区下逐点执行 Route 5 formal 搜索。"""
    args = _parse_args()
    aggregate_path = Path(args.output_json)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    solver_options = _build_solver_options(args)

    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
    else:
        aggregate = {
            "route": "route5_fixed_partition_radius_search",
            "cutoff": int(args.cutoff),
            "phase_values": list(args.phase_values),
            "num_x_bins": int(args.num_x_bins),
            "num_p_bins": int(args.num_p_bins),
            "quadrature_range": float(args.quadrature_range),
            "boundary_gamma": float(args.boundary_gamma),
            "num_quadrature_nodes": int(args.num_quadrature_nodes),
            "max_inputs": int(args.max_inputs),
            "solver": args.solver,
            "results": [],
        }

    done_by_key = {
        entry["candidate_key"]: dict(entry)
        for entry in aggregate.get("results", [])
    }

    radius_candidates = _build_radius_candidates(args.r1_values, args.r2_values)
    if not radius_candidates:
        raise RuntimeError("No valid radius candidates were generated.")

    for radii in radius_candidates:
        candidate_key = _candidate_key(radii)
        if candidate_key in done_by_key:
            continue

        result = run_route5(
            cutoff=args.cutoff,
            radius_values=list(radii),
            phase_values=list(args.phase_values),
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            quadrature_range=args.quadrature_range,
            boundary_gamma=args.boundary_gamma,
            num_quadrature_nodes=args.num_quadrature_nodes,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
            verbose=False,
        )

        output_json = result_dir / f"{candidate_key}.json"
        _write_json(output_json, result)
        done_by_key[candidate_key] = _result_summary(result, radii, str(output_json))
        aggregate["results"] = sorted(
            done_by_key.values(),
            key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
            reverse=True,
        )
        aggregate["best_result"] = None if not aggregate["results"] else aggregate["results"][0]
        _write_json(aggregate_path, aggregate)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
