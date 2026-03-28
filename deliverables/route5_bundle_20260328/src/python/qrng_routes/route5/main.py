from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_iq import (
    DEFAULT_GAMMA_VALUES,
    DEFAULT_PHASE_VALUES,
    DEFAULT_QUADRATURE_RANGES,
    DEFAULT_RADIUS_VALUES,
    run_route5,
    search_route5_alphabets,
    search_route5_iq_partitions,
)


def _clean_value(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str | None, payload: Any) -> None:
    """Persist a JSON payload when requested."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _parse_bounds(values: list[float] | None) -> np.ndarray | None:
    """Convert optional boundary values into a numpy array."""
    if values is None:
        return None
    return np.array(values, dtype=float)


def _parse_complex_values(values: list[str] | None) -> list[complex] | None:
    """Parse Python-style complex literals from CLI strings."""
    if values is None:
        return None
    return [complex(value.replace("i", "j")) for value in values]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone runner for QRNG route 5.")
    parser.add_argument("--mode", choices=["single", "partition-search", "alphabet-search"], default="single")
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--alpha-values", nargs="+", type=str, default=None)
    parser.add_argument("--radius-values", nargs="+", type=float, default=DEFAULT_RADIUS_VALUES)
    parser.add_argument("--phase-values", nargs="+", type=float, default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--num-radii-values", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--num-phase-values", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--num-x-bins", type=int, default=2)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--num-x-bins-values", nargs="+", type=int, default=[2])
    parser.add_argument("--num-p-bins-values", nargs="+", type=int, default=[2])
    parser.add_argument("--x-bounds", nargs="+", type=float, default=None)
    parser.add_argument("--p-bounds", nargs="+", type=float, default=None)
    parser.add_argument("--quadrature-range", type=float, default=3.0)
    parser.add_argument("--quadrature-ranges", nargs="+", type=float, default=DEFAULT_QUADRATURE_RANGES)
    parser.add_argument("--boundary-gamma", type=float, default=1.0)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--num-quadrature-nodes", type=int, default=None)
    parser.add_argument("--scs-max-iters", type=int, default=None)
    parser.add_argument("--scs-eps-abs", type=float, default=None)
    parser.add_argument("--scs-eps-rel", type=float, default=None)
    parser.add_argument("--scs-eps-infeas", type=float, default=None)
    parser.add_argument("--mosek-num-threads", type=int, default=None)
    parser.add_argument("--mosek-eps", type=float, default=None)
    parser.add_argument("--mosek-accept-unknown", action="store_true")
    parser.add_argument("--mosek-save-file", type=str, default=None)
    parser.add_argument("--mosek-solve-form", choices=["dual", "primal", "free"], default=None)
    parser.add_argument("--max-inputs", type=int, default=1)
    parser.add_argument("--certify-top-k", type=int, default=3)
    parser.add_argument("--alphabet-top-k", type=int, default=3)
    parser.add_argument("--max-local-states", type=int, default=0)
    parser.add_argument("--no-require-vacuum", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    alpha_values = _parse_complex_values(args.alpha_values)
    x_bounds = _parse_bounds(args.x_bounds)
    p_bounds = _parse_bounds(args.p_bounds)
    max_local_states = None if args.max_local_states <= 0 else int(args.max_local_states)
    solver_options = _build_solver_options(args)

    if args.mode == "single":
        result = run_route5(
            cutoff=args.cutoff,
            alpha_values=alpha_values,
            radius_values=None if alpha_values is not None else list(args.radius_values),
            phase_values=None if alpha_values is not None else list(args.phase_values),
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=args.quadrature_range,
            boundary_gamma=args.boundary_gamma,
            num_quadrature_nodes=args.num_quadrature_nodes,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
            verbose=args.verbose,
        )
    elif args.mode == "partition-search":
        result = search_route5_iq_partitions(
            cutoff=args.cutoff,
            alpha_values=alpha_values,
            radius_values=None if alpha_values is not None else list(args.radius_values),
            phase_values=None if alpha_values is not None else list(args.phase_values),
            num_x_bins_values=list(args.num_x_bins_values),
            num_p_bins_values=list(args.num_p_bins_values),
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            num_quadrature_nodes=args.num_quadrature_nodes,
            certify_top_k=args.certify_top_k,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
            verbose=args.verbose,
        )
    else:
        result = search_route5_alphabets(
            cutoff=args.cutoff,
            radius_values=list(args.radius_values),
            phase_values=list(args.phase_values),
            num_radii_values=list(args.num_radii_values),
            num_phase_values=list(args.num_phase_values),
            require_vacuum=not args.no_require_vacuum,
            max_local_states=max_local_states,
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
            verbose=args.verbose,
        )

    _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
