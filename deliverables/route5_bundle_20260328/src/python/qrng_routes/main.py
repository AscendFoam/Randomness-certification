from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .route1 import run_route1, sweep_route1_eta
from .route1.main import paper_like_route1_sweep
from .route2 import run_route2, search_route2_high_entropy
from .route3 import run_route3
from .route4 import (
    compare_route4_primal_dual,
    run_route4_dual,
    run_route4_primal,
    search_route4_triplets,
    sweep_route4_outputs,
)


def compare_all(preferred_solver: str | None = None, verbose: bool = False) -> list[dict]:
    """Run one modest prototype for each route."""
    results = [
        run_route1(
            source="tmsv",
            bob_mode="tomography",
            dimension=4,
            squeezing_db=-4.0,
            eta=0.90,
            num_alice_bins=4,
            num_bob_bins=6,
            num_bob_settings=2,
            preferred_solver=preferred_solver,
            verbose=verbose,
        ),
        run_route1(
            source="tmsv",
            bob_mode="homodyne",
            dimension=4,
            squeezing_db=-4.0,
            eta=0.90,
            num_alice_bins=4,
            num_bob_bins=6,
            num_bob_settings=2,
            preferred_solver=preferred_solver,
            verbose=verbose,
        ),
        run_route1(
            source="split_sms",
            bob_mode="homodyne",
            dimension=4,
            squeezing_db=-8.0,
            eta=1.00,
            num_alice_bins=4,
            num_bob_bins=6,
            num_bob_settings=2,
            preferred_solver=preferred_solver,
            verbose=verbose,
        ),
        run_route2(preferred_solver=preferred_solver, verbose=verbose),
        run_route3(mu=0.05, cutoff=12, num_phases=4, preferred_solver=preferred_solver, verbose=verbose),
        run_route3(mu=0.05, cutoff=12, num_phases=6, preferred_solver=preferred_solver, verbose=verbose),
        run_route4_dual(
            num_outputs=6,
            selected_mu_list=[100, 120, 140],
            q_selected=[0.25, 0.25, 0.5],
            preferred_solver=preferred_solver,
            verbose=verbose,
        ),
    ]
    return results


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


def _clean_value(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-friendly values."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype comparison for four QRNG routes.")
    parser.add_argument(
        "--mode",
        choices=["compare", "route1", "route2", "route3", "route4"],
        default="compare",
    )
    parser.add_argument("--route1-mode", choices=["single", "sweep-eta", "paper-sweep"], default="single")
    parser.add_argument("--route2-mode", choices=["baseline", "high-output-search"], default="baseline")
    parser.add_argument("--route3-mode", choices=["single", "phase-sweep", "mu-sweep"], default="single")
    parser.add_argument(
        "--route4-mode",
        choices=["dual-single", "primal-single", "primal-dual-compare", "output-sweep", "subset-search"],
        default="dual-single",
    )
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--source", choices=["tmsv", "split_sms"], default="tmsv")
    parser.add_argument("--bob-mode", choices=["tomography", "homodyne"], default="homodyne")
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--eta-values", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.95, 1.0])
    parser.add_argument("--squeezing-db", type=float, default=-4.0)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--mu-values", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--cutoff", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--alice-bins", type=int, default=4)
    parser.add_argument("--bob-bins", type=int, default=6)
    parser.add_argument("--max-inputs", type=int, default=1)
    parser.add_argument("--route2-max-inputs", type=int, default=None)
    parser.add_argument("--bob-settings", type=int, default=2)
    parser.add_argument("--paper-bob-settings", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--skip-tomography", action="store_true")
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--phase-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--tq-grid", nargs="+", type=float, default=None)
    parser.add_argument("--num-quadrature-nodes", type=int, default=None)
    parser.add_argument("--num-x-bins", type=int, default=2)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--x-bounds", nargs="+", type=float, default=None)
    parser.add_argument("--p-bounds", nargs="+", type=float, default=None)
    parser.add_argument("--quadrature-range", type=float, default=3.0)
    parser.add_argument("--num-outputs", type=int, default=8)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selected-mu", nargs="+", type=int, default=[100, 120, 140])
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.25, 0.25, 0.5])
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--output-values", nargs="+", type=int, default=[4, 6, 8, 12, 16])
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--certify-top-k", type=int, default=3)
    parser.add_argument("--full-mu", nargs="+", type=int, default=[0, 20, 40, 60, 80, 100, 120, 140, 160])
    parser.add_argument("--max-primal-variables", type=int, default=3_000_000)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    prob_floor = None if args.prob_floor <= 0 else args.prob_floor
    x_bounds = _parse_bounds(args.x_bounds)
    p_bounds = _parse_bounds(args.p_bounds)

    if args.mode == "compare":
        result = compare_all(preferred_solver=args.solver, verbose=args.verbose)
    elif args.mode == "route1":
        tq_grid = None if args.tq_grid is None else np.array(args.tq_grid, dtype=float)
        if args.route1_mode == "single":
            result = run_route1(
                source=args.source,
                bob_mode=args.bob_mode,
                dimension=args.dimension,
                eta=args.eta,
                squeezing_db=args.squeezing_db,
                num_alice_bins=args.alice_bins,
                num_bob_bins=args.bob_bins,
                num_bob_settings=args.bob_settings,
                tq_grid=tq_grid,
                num_quadrature_nodes=args.num_quadrature_nodes,
                preferred_solver=args.solver,
                verbose=args.verbose,
            )
        elif args.route1_mode == "sweep-eta":
            result = sweep_route1_eta(
                source=args.source,
                bob_mode=args.bob_mode,
                eta_values=np.array(args.eta_values, dtype=float),
                dimension=args.dimension,
                squeezing_db=args.squeezing_db,
                num_alice_bins=args.alice_bins,
                num_bob_bins=args.bob_bins,
                tq_grid=np.array([2.0, 4.0, 6.0]) if tq_grid is None else tq_grid,
                num_bob_settings=args.bob_settings,
                num_quadrature_nodes=args.num_quadrature_nodes,
                preferred_solver=args.solver,
                verbose=args.verbose,
            )
        else:
            result = paper_like_route1_sweep(
                source=args.source,
                dimension=args.dimension,
                squeezing_db=args.squeezing_db,
                num_alice_bins=args.alice_bins,
                num_bob_bins=args.bob_bins,
                eta_values=np.array(args.eta_values, dtype=float),
                tq_grid=tq_grid,
                bob_settings_values=list(args.paper_bob_settings),
                include_tomography=not args.skip_tomography,
                num_quadrature_nodes=args.num_quadrature_nodes,
                preferred_solver=args.solver,
                verbose=args.verbose,
            )
    elif args.mode == "route2":
        if args.route2_mode == "baseline":
            result = run_route2(
                max_inputs_to_certify=args.route2_max_inputs,
                preferred_solver=args.solver,
                verbose=args.verbose,
            )
        else:
            result = search_route2_high_entropy(
                num_outputs=args.num_outputs,
                num_trials=args.num_trials,
                preferred_solver=args.solver,
                seed=args.seed,
                verbose=args.verbose,
            )
    elif args.mode == "route3":
        if args.route3_mode == "single":
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
        elif args.route3_mode == "phase-sweep":
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
    else:
        if args.route4_mode == "dual-single":
            result = run_route4_dual(
                num_outputs=args.num_outputs,
                selected_mu_list=args.selected_mu,
                q_selected=args.q_values,
                cutoff=args.cutoff,
                prob_floor=prob_floor,
                shift=args.shift,
                preferred_solver=args.solver,
                verbose=args.verbose,
            )
        elif args.route4_mode == "primal-single":
            result = run_route4_primal(
                num_outputs=args.num_outputs,
                selected_mu_list=args.selected_mu,
                q_selected=args.q_values,
                cutoff=args.cutoff,
                prob_floor=prob_floor,
                shift=args.shift,
                preferred_solver=args.solver,
                verbose=args.verbose,
                max_primal_variables=args.max_primal_variables,
            )
        elif args.route4_mode == "primal-dual-compare":
            result = compare_route4_primal_dual(
                num_outputs=args.num_outputs,
                selected_mu_list=args.selected_mu,
                q_selected=args.q_values,
                cutoff=args.cutoff,
                prob_floor=prob_floor,
                shift=args.shift,
                preferred_solver=args.solver,
                verbose=args.verbose,
                max_primal_variables=args.max_primal_variables,
            )
        elif args.route4_mode == "output-sweep":
            result = sweep_route4_outputs(
                output_values=args.output_values,
                selected_mu_list=args.selected_mu,
                q_selected=args.q_values,
                cutoff=args.cutoff,
                prob_floor=prob_floor,
                shift=args.shift,
                preferred_solver=args.solver,
                verbose=args.verbose,
            )
        else:
            result = search_route4_triplets(
                num_outputs=args.num_outputs,
                subset_size=args.subset_size,
                certify_top_k=args.certify_top_k,
                cutoff=args.cutoff,
                prob_floor=prob_floor,
                shift=args.shift,
                preferred_solver=args.solver,
                verbose=args.verbose,
                full_mu=args.full_mu,
            )

    _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
