from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .analytic_gram_iq import (
    DEFAULT_GAMMA_VALUES,
    DEFAULT_PHASE_VALUES,
    DEFAULT_QUADRATURE_RANGES,
    DEFAULT_RADIUS_VALUES,
    run_route6,
    search_route6_alphabets,
    search_route6_fixed_partition_alphabets,
    search_route6_iq_partitions,
)


def _phase_list(values: list[float]) -> list[float]:
    return [float(value) for value in values]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route 6: Gram-represented coherent alphabet with analytic IQ bins.")
    parser.add_argument(
        "--mode",
        choices=["single", "partition-search", "alphabet-search", "fixed-partition-alphabet-search"],
        default="single",
    )
    parser.add_argument("--radius-values", nargs="+", type=float, default=DEFAULT_RADIUS_VALUES)
    parser.add_argument("--phase-values", nargs="+", type=float, default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--num-x-bins", type=int, default=2)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--num-x-bins-values", nargs="+", type=int, default=[2])
    parser.add_argument("--num-p-bins-values", nargs="+", type=int, default=[2])
    parser.add_argument("--quadrature-range", type=float, default=3.0)
    parser.add_argument("--quadrature-ranges", nargs="+", type=float, default=DEFAULT_QUADRATURE_RANGES)
    parser.add_argument("--boundary-gamma", type=float, default=1.0)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--max-inputs-to-certify", type=int, default=1)
    parser.add_argument("--certify-top-k", type=int, default=3)
    parser.add_argument("--num-radii-values", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--num-phase-values", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--certify-top-k-per-alphabet", type=int, default=1)
    parser.add_argument("--max-local-states", type=int, default=None)
    parser.add_argument("--min-local-states", type=int, default=None)
    parser.add_argument("--require-vacuum", action="store_true")
    parser.add_argument("--gram-tol", type=float, default=1e-10)
    parser.add_argument("--preferred-solver", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    phase_values = _phase_list(list(args.phase_values))
    max_inputs_to_certify = None if args.max_inputs_to_certify <= 0 else args.max_inputs_to_certify
    if args.mode == "single":
        result = run_route6(
            radius_values=list(args.radius_values),
            phase_values=phase_values,
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            quadrature_range=args.quadrature_range,
            boundary_gamma=args.boundary_gamma,
            max_inputs_to_certify=max_inputs_to_certify,
            gram_tol=args.gram_tol,
            preferred_solver=args.preferred_solver,
        )
    elif args.mode == "partition-search":
        result = search_route6_iq_partitions(
            radius_values=list(args.radius_values),
            phase_values=phase_values,
            num_x_bins_values=list(args.num_x_bins_values),
            num_p_bins_values=list(args.num_p_bins_values),
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            certify_top_k=args.certify_top_k,
            max_inputs_to_certify=max_inputs_to_certify,
            gram_tol=args.gram_tol,
            preferred_solver=args.preferred_solver,
        )
    elif args.mode == "alphabet-search":
        result = search_route6_alphabets(
            radius_values=list(args.radius_values),
            phase_values=phase_values,
            num_radii_values=list(args.num_radii_values),
            num_phase_values=list(args.num_phase_values),
            num_x_bins_values=list(args.num_x_bins_values),
            num_p_bins_values=list(args.num_p_bins_values),
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            require_vacuum=bool(args.require_vacuum),
            max_local_states=args.max_local_states,
            certify_top_k_per_alphabet=args.certify_top_k_per_alphabet,
            max_inputs_to_certify=max_inputs_to_certify,
            gram_tol=args.gram_tol,
            preferred_solver=args.preferred_solver,
        )
    else:
        result = search_route6_fixed_partition_alphabets(
            radius_values=list(args.radius_values),
            phase_values=phase_values,
            num_radii_values=list(args.num_radii_values),
            num_phase_values=list(args.num_phase_values),
            num_x_bins=args.num_x_bins,
            num_p_bins=args.num_p_bins,
            quadrature_range=args.quadrature_range,
            boundary_gamma=args.boundary_gamma,
            require_vacuum=bool(args.require_vacuum),
            max_local_states=args.max_local_states,
            min_local_states=args.min_local_states,
            max_inputs_to_certify=max_inputs_to_certify,
            gram_tol=args.gram_tol,
            preferred_solver=args.preferred_solver,
        )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
