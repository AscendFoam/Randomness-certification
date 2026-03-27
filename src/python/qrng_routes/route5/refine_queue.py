from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_iq import DEFAULT_PHASE_VALUES, run_route5, search_route5_iq_partitions


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _candidate_key(radii: list[float]) -> str:
    return "r" + "_".join(f"{value:.4f}" for value in radii)


def _unique_sorted(values: list[float], tol: float = 1e-12) -> list[float]:
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if any(abs(value - existing) <= tol for existing in unique):
            continue
        unique.append(value)
    return unique


def _build_radius_candidates(r1_values: list[float], r2_values: list[float]) -> list[list[float]]:
    first = [value for value in _unique_sorted(r1_values) if value > 0.0]
    second = [value for value in _unique_sorted(r2_values) if value > 0.0]
    out: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for r1 in first:
        for r2 in second:
            if r2 <= r1 + 1e-12:
                continue
            candidate = (0.0, float(r1), float(r2))
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(list(candidate))
    return out


def _build_solver_options(args: argparse.Namespace) -> dict[str, dict] | None:
    """Build CVXPY solver options for SCS / MOSEK from CLI flags."""
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
            "mosek_params": mosek_params if len(mosek_params) > 0 else None,
            "eps": args.mosek_eps,
            "accept_unknown": True if args.mosek_accept_unknown else None,
            "save_file": args.mosek_save_file,
        }.items()
        if value is not None
    }

    solver_options: dict[str, dict] = {}
    if len(scs_options) > 0:
        solver_options["SCS"] = scs_options
    if len(mosek_options) > 0:
        solver_options["MOSEK"] = mosek_options
    return None if len(solver_options) == 0 else solver_options


def _scout_summary(result: dict, radii: list[float]) -> dict:
    return {
        "candidate_key": _candidate_key(radii),
        "radii": list(radii),
        "raw_best_H_min": result["raw_best_H_min"],
        "raw_best_target": result["raw_best_target"],
        "raw_best_target_index": result["raw_best_target_index"],
        "num_x_bins": result["num_x_bins"],
        "num_p_bins": result["num_p_bins"],
        "x_bounds": result["x_bounds"],
        "p_bounds": result["p_bounds"],
        "x_range": result["x_range"],
        "p_range": result["p_range"],
        "x_gamma": result["x_gamma"],
        "p_gamma": result["p_gamma"],
        "num_outputs": result["num_outputs"],
        "local_operator_span_rank": result["local_operator_span_rank"],
        "local_operator_space_dim": result["local_operator_space_dim"],
        "num_partition_candidates": result["num_partition_candidates"],
        "selection_strategy": result["selection_strategy"],
    }


def _certified_summary(result: dict, radii: list[float], scout_rank: int, output_json: str) -> dict:
    return {
        "candidate_key": _candidate_key(radii),
        "radii": list(radii),
        "scout_rank": int(scout_rank),
        "output_json": output_json,
        "H_min": result["H_min"],
        "raw_best_H_min": result["raw_best_H_min"],
        "target_input": result["target_input"],
        "raw_best_target": result["raw_best_target"],
        "num_x_bins": result["num_x_bins"],
        "num_p_bins": result["num_p_bins"],
        "x_bounds": result["x_bounds"],
        "p_bounds": result["p_bounds"],
        "quadrature_range": result["quadrature_range"],
        "boundary_gamma": result["boundary_gamma"],
        "num_quadrature_nodes": result["num_quadrature_nodes"],
        "num_inputs_certified": result["num_inputs_certified"],
        "solver": result["solver"],
        "status": result["status"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-running local refinement queue for route5.")
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--phase-values", nargs="+", type=float, default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--r1-values", nargs="+", type=float, required=True)
    parser.add_argument("--r2-values", nargs="+", type=float, required=True)
    parser.add_argument("--num-x-bins", type=int, default=6)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--quadrature-ranges", nargs="+", type=float, required=True)
    parser.add_argument("--gamma-values", nargs="+", type=float, required=True)
    parser.add_argument("--scout-num-quadrature-nodes", type=int, default=20)
    parser.add_argument("--cert-num-quadrature-nodes", type=int, default=20)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--scs-max-iters", type=int, default=None)
    parser.add_argument("--scs-eps-abs", type=float, default=None)
    parser.add_argument("--scs-eps-rel", type=float, default=None)
    parser.add_argument("--scs-eps-infeas", type=float, default=None)
    parser.add_argument("--mosek-num-threads", type=int, default=None)
    parser.add_argument("--mosek-eps", type=float, default=None)
    parser.add_argument("--mosek-accept-unknown", action="store_true")
    parser.add_argument("--mosek-save-file", type=str, default=None)
    parser.add_argument("--mosek-solve-form", choices=["dual", "primal", "free"], default=None)
    parser.add_argument("--max-inputs", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate_path = Path(args.output_json)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    solver_options = _build_solver_options(args)

    aggregate: dict[str, Any]
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
    else:
        aggregate = {
            "route": "route5_local_refine_queue",
            "cutoff": int(args.cutoff),
            "phase_values": list(args.phase_values),
            "num_x_bins": int(args.num_x_bins),
            "num_p_bins": int(args.num_p_bins),
            "quadrature_ranges": list(args.quadrature_ranges),
            "gamma_values": list(args.gamma_values),
            "scout_num_quadrature_nodes": int(args.scout_num_quadrature_nodes),
            "cert_num_quadrature_nodes": int(args.cert_num_quadrature_nodes),
            "max_inputs": int(args.max_inputs),
            "candidate_limit": None if args.candidate_limit <= 0 else int(args.candidate_limit),
            "scout_results": [],
            "certified_results": [],
        }

    scout_by_key = {
        entry["candidate_key"]: dict(entry)
        for entry in aggregate.get("scout_results", [])
    }
    certified_by_key = {
        entry["candidate_key"]: dict(entry)
        for entry in aggregate.get("certified_results", [])
    }

    radius_candidates = _build_radius_candidates(args.r1_values, args.r2_values)
    if len(radius_candidates) == 0:
        raise RuntimeError("No valid radius candidates were generated.")

    for radii in radius_candidates:
        candidate_key = _candidate_key(radii)
        if candidate_key in scout_by_key:
            continue
        scout = search_route5_iq_partitions(
            cutoff=args.cutoff,
            radius_values=list(radii),
            phase_values=list(args.phase_values),
            num_x_bins_values=[args.num_x_bins],
            num_p_bins_values=[args.num_p_bins],
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            num_quadrature_nodes=args.scout_num_quadrature_nodes,
            certify_top_k=0,
            max_inputs_to_certify=args.max_inputs,
        )
        scout_by_key[candidate_key] = _scout_summary(scout, radii)
        aggregate["scout_results"] = sorted(
            scout_by_key.values(),
            key=lambda item: item["raw_best_H_min"],
            reverse=True,
        )
        _write_json(aggregate_path, aggregate)

    ranked_scouts = sorted(
        scout_by_key.values(),
        key=lambda item: item["raw_best_H_min"],
        reverse=True,
    )
    if args.candidate_limit > 0:
        ranked_scouts = ranked_scouts[: args.candidate_limit]

    for scout_rank, scout in enumerate(ranked_scouts, start=1):
        candidate_key = scout["candidate_key"]
        if candidate_key in certified_by_key:
            continue
        radii = scout["radii"]
        quadrature_range = max(float(scout["x_range"]), float(scout["p_range"]), 1.0)
        result = run_route5(
            cutoff=args.cutoff,
            radius_values=list(radii),
            phase_values=list(args.phase_values),
            num_x_bins=int(scout["num_x_bins"]),
            num_p_bins=int(scout["num_p_bins"]),
            x_bounds=np.array(scout["x_bounds"], dtype=float),
            p_bounds=np.array(scout["p_bounds"], dtype=float),
            quadrature_range=quadrature_range,
            boundary_gamma=float(scout["x_gamma"]),
            num_quadrature_nodes=args.cert_num_quadrature_nodes,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
        )
        output_json = result_dir / f"{candidate_key}.json"
        _write_json(output_json, result)
        certified_by_key[candidate_key] = _certified_summary(
            result,
            radii=list(radii),
            scout_rank=scout_rank,
            output_json=str(output_json),
        )
        aggregate["certified_results"] = sorted(
            certified_by_key.values(),
            key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
            reverse=True,
        )
        aggregate["best_certified_result"] = (
            None if len(aggregate["certified_results"]) == 0 else aggregate["certified_results"][0]
        )
        _write_json(aggregate_path, aggregate)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
