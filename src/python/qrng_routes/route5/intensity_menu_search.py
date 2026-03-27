from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_iq import DEFAULT_GAMMA_VALUES, search_route5_alphabets


DEFAULT_8PHASE_VALUES = [
    0.0,
    0.25 * math.pi,
    0.5 * math.pi,
    0.75 * math.pi,
    math.pi,
    1.25 * math.pi,
    1.5 * math.pi,
    1.75 * math.pi,
]
DEFAULT_QUADRATURE_RANGES = [1.8, 1.85, 1.9, 1.95, 2.0]


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str | Path | None, payload: Any) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _unique_sorted(values: list[float], tol: float = 1e-12) -> list[float]:
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if any(abs(value - existing) <= tol for existing in unique):
            continue
        unique.append(value)
    return unique


def _build_solver_options(args: argparse.Namespace) -> dict[str, dict] | None:
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


def intensity_menu_to_radii(
    intensity_values: list[float],
    max_radius: float,
    require_vacuum: bool = True,
) -> tuple[list[float], list[dict[str, float]]]:
    """Map a fixed intensity menu to route5 radius units while preserving intensity ratios."""
    unique_intensities = _unique_sorted(intensity_values)
    nonnegative = [value for value in unique_intensities if value >= 0.0]
    if len(nonnegative) == 0:
        raise ValueError("At least one non-negative intensity is required.")

    positive = [value for value in nonnegative if value > 0.0]
    if len(positive) == 0 and not require_vacuum:
        raise ValueError("At least one positive intensity is required when vacuum is not forced.")

    max_intensity = max(positive) if len(positive) > 0 else 1.0
    mapping: list[dict[str, float]] = []
    radii: list[float] = []

    if require_vacuum:
        radii.append(0.0)
        mapping.append({"intensity": 0.0, "radius": 0.0})

    for intensity in positive:
        radius = float(max_radius) * math.sqrt(float(intensity) / max_intensity)
        radii.append(radius)
        mapping.append({"intensity": float(intensity), "radius": float(radius)})

    return _unique_sorted(radii), mapping


def _scale_summary(result: dict[str, Any], max_radius: float, radius_mapping: list[dict[str, float]]) -> dict[str, Any]:
    partition_result = result.get("best_partition_search_result")
    return {
        "max_radius": float(max_radius),
        "radius_mapping": radius_mapping,
        "radius_pool": result["radius_pool"],
        "num_alphabet_candidates": result["num_alphabet_candidates"],
        "selected_alphabet_candidate_index": result["selected_alphabet_candidate_index"],
        "selected_partition_candidate_index": result["selected_partition_candidate_index"],
        "best_certified_H_min": result["best_certified_H_min"],
        "selected_radius_values": result["radius_values"],
        "selected_phase_values": result["phase_values"],
        "selected_num_local_states": result["num_local_states"],
        "selected_alpha_values": result["alpha_values"],
        "partition_summary": None
        if partition_result is None
        else {
            "num_outputs": partition_result["num_outputs"],
            "num_x_bins": partition_result["num_x_bins"],
            "num_p_bins": partition_result["num_p_bins"],
            "quadrature_range": partition_result["quadrature_range"],
            "boundary_gamma": partition_result["boundary_gamma"],
            "H_min": partition_result["H_min"],
            "raw_best_H_min": partition_result["raw_best_H_min"],
            "target_input": partition_result["target_input"],
            "certified_best_target_alphas": partition_result["certified_best_target_alphas"],
            "solver": partition_result["solver"],
            "status": partition_result["status"],
        },
        "result": result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route5 search constrained to a fixed experimental intensity menu."
    )
    parser.add_argument("--intensity-values", nargs="+", type=float, required=True)
    parser.add_argument("--max-radius-values", nargs="+", type=float, required=True)
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--phase-values", nargs="+", type=float, default=DEFAULT_8PHASE_VALUES)
    parser.add_argument("--num-radii-values", nargs="+", type=int, default=[3])
    parser.add_argument("--num-phase-values", nargs="+", type=int, default=[8])
    parser.add_argument("--num-x-bins-values", nargs="+", type=int, default=[6])
    parser.add_argument("--num-p-bins-values", nargs="+", type=int, default=[2])
    parser.add_argument("--quadrature-ranges", nargs="+", type=float, default=DEFAULT_QUADRATURE_RANGES)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--num-quadrature-nodes", type=int, default=12)
    parser.add_argument("--alphabet-top-k", type=int, default=4)
    parser.add_argument("--certify-top-k", type=int, default=3)
    parser.add_argument("--max-inputs", type=int, default=3)
    parser.add_argument("--solver", type=str, default="MOSEK")
    parser.add_argument("--scs-max-iters", type=int, default=None)
    parser.add_argument("--scs-eps-abs", type=float, default=None)
    parser.add_argument("--scs-eps-rel", type=float, default=None)
    parser.add_argument("--scs-eps-infeas", type=float, default=None)
    parser.add_argument("--mosek-num-threads", type=int, default=None)
    parser.add_argument("--mosek-eps", type=float, default=None)
    parser.add_argument("--mosek-accept-unknown", action="store_true")
    parser.add_argument("--mosek-save-file", type=str, default=None)
    parser.add_argument("--mosek-solve-form", choices=["dual", "primal", "free"], default=None)
    parser.add_argument("--no-require-vacuum", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    solver_options = _build_solver_options(args)
    require_vacuum = not args.no_require_vacuum

    aggregate: dict[str, Any] = {
        "route": "route5_fixed_intensity_menu_search",
        "intensity_values": _unique_sorted(list(args.intensity_values)),
        "max_radius_values": _unique_sorted(list(args.max_radius_values)),
        "scaling_rule": "radius = max_radius * sqrt(intensity / max_intensity_in_menu)",
        "cutoff": int(args.cutoff),
        "phase_values": [float(value) for value in args.phase_values],
        "num_radii_values": [int(value) for value in args.num_radii_values],
        "num_phase_values": [int(value) for value in args.num_phase_values],
        "num_x_bins_values": [int(value) for value in args.num_x_bins_values],
        "num_p_bins_values": [int(value) for value in args.num_p_bins_values],
        "quadrature_ranges": [float(value) for value in args.quadrature_ranges],
        "gamma_values": [float(value) for value in args.gamma_values],
        "num_quadrature_nodes": int(args.num_quadrature_nodes),
        "alphabet_top_k": int(args.alphabet_top_k),
        "certify_top_k": int(args.certify_top_k),
        "max_inputs": int(args.max_inputs),
        "require_vacuum": bool(require_vacuum),
        "scale_results": [],
    }

    for max_radius in aggregate["max_radius_values"]:
        radius_pool, mapping = intensity_menu_to_radii(
            intensity_values=aggregate["intensity_values"],
            max_radius=max_radius,
            require_vacuum=require_vacuum,
        )
        result = search_route5_alphabets(
            cutoff=args.cutoff,
            radius_values=radius_pool,
            phase_values=list(args.phase_values),
            num_radii_values=list(args.num_radii_values),
            num_phase_values=list(args.num_phase_values),
            require_vacuum=require_vacuum,
            num_x_bins_values=list(args.num_x_bins_values),
            num_p_bins_values=list(args.num_p_bins_values),
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            num_quadrature_nodes=args.num_quadrature_nodes,
            alphabet_top_k=args.alphabet_top_k,
            certify_top_k=args.certify_top_k,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
        )
        aggregate["scale_results"].append(_scale_summary(result, max_radius=max_radius, radius_mapping=mapping))
        _write_json(args.output_json, aggregate)

    if len(aggregate["scale_results"]) > 0:
        aggregate["best_overall"] = max(
            aggregate["scale_results"],
            key=lambda item: item["best_certified_H_min"]
            if item["best_certified_H_min"] is not None
            else -np.inf,
        )
    else:
        aggregate["best_overall"] = None

    _write_json(args.output_json, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
