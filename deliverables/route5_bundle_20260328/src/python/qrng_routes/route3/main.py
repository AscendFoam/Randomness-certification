from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cv_four_phase import run_route3


def _clean_value(value: Any) -> Any:
    """Convert numpy scalars to JSON-friendly values."""
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


def sweep_route3_phases(
    phase_values: list[int],
    mu: float,
    cutoff: int,
    max_inputs_to_certify: int | None,
    num_x_bins: int,
    num_p_bins: int,
    x_bounds: np.ndarray | None,
    p_bounds: np.ndarray | None,
    quadrature_range: float,
    num_quadrature_nodes: int | None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Sweep route 3 over phase counts."""
    return [
        run_route3(
            mu=mu,
            cutoff=cutoff,
            num_phases=num_phases,
            num_x_bins=num_x_bins,
            num_p_bins=num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=quadrature_range,
            num_quadrature_nodes=num_quadrature_nodes,
            max_inputs_to_certify=max_inputs_to_certify,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )
        for num_phases in phase_values
    ]


def sweep_route3_mu(
    mu_values: list[float],
    cutoff: int,
    num_phases: int,
    max_inputs_to_certify: int | None,
    num_x_bins: int,
    num_p_bins: int,
    x_bounds: np.ndarray | None,
    p_bounds: np.ndarray | None,
    quadrature_range: float,
    num_quadrature_nodes: int | None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Sweep route 3 over amplitudes."""
    return [
        run_route3(
            mu=mu,
            cutoff=cutoff,
            num_phases=num_phases,
            num_x_bins=num_x_bins,
            num_p_bins=num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=quadrature_range,
            num_quadrature_nodes=num_quadrature_nodes,
            max_inputs_to_certify=max_inputs_to_certify,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )
        for mu in mu_values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone runner for QRNG route 3.")
    parser.add_argument("--mode", choices=["single", "phase-sweep", "mu-sweep"], default="single")
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--mu-values", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--cutoff", type=int, default=12)
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--phase-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--max-inputs", type=int, default=1)
    parser.add_argument("--num-x-bins", type=int, default=2)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--x-bounds", nargs="+", type=float, default=None)
    parser.add_argument("--p-bounds", nargs="+", type=float, default=None)
    parser.add_argument("--quadrature-range", type=float, default=3.0)
    parser.add_argument("--num-quadrature-nodes", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    x_bounds = _parse_bounds(args.x_bounds)
    p_bounds = _parse_bounds(args.p_bounds)

    if args.mode == "single":
        result = run_route3(
            mu=args.mu,
            cutoff=args.cutoff,
            num_phases=args.num_phases,
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=args.quadrature_range,
            num_quadrature_nodes=args.num_quadrature_nodes,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            verbose=args.verbose,
        )
    elif args.mode == "phase-sweep":
        result = sweep_route3_phases(
            phase_values=args.phase_values,
            mu=args.mu,
            cutoff=args.cutoff,
            max_inputs_to_certify=args.max_inputs,
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=args.quadrature_range,
            num_quadrature_nodes=args.num_quadrature_nodes,
            preferred_solver=args.solver,
            verbose=args.verbose,
        )
    else:
        result = sweep_route3_mu(
            mu_values=args.mu_values,
            cutoff=args.cutoff,
            num_phases=args.num_phases,
            max_inputs_to_certify=args.max_inputs,
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            quadrature_range=args.quadrature_range,
            num_quadrature_nodes=args.num_quadrature_nodes,
            preferred_solver=args.solver,
            verbose=args.verbose,
        )

    _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
