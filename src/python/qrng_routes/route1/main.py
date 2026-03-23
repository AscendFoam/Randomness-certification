from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .steering_2022 import default_quadrature_nodes, run_route1, sweep_route1_eta


DEFAULT_PAPER_ETA_VALUES = np.array([0.80, 0.85, 0.90, 0.95, 1.00], dtype=float)
DEFAULT_PAPER_TQ_GRID = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
DEFAULT_PAPER_BOB_SETTINGS = [2, 4, 6]


def _clean_value(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str | None, payload: Any) -> None:
    """Persist a JSON payload when a path is provided."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _plot_eta_sweep(curves: dict[str, list[dict]], title: str, output_path: str | None) -> None:
    """Plot certified min-entropy versus eta for a small set of route-1 curves."""
    if output_path is None:
        return

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, series in curves.items():
        eta_values = [float(entry["eta"]) for entry in series]
        entropy_values = [float(entry["H_min"]) for entry in series]
        ax.plot(eta_values, entropy_values, marker="o", linewidth=2.0, label=label)

    ax.set_xlabel("Transmission efficiency eta")
    ax.set_ylabel("Certified H_min (bits)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def paper_like_route1_sweep(
    source: str,
    dimension: int,
    squeezing_db: float,
    num_alice_bins: int,
    num_bob_bins: int,
    eta_values: np.ndarray | None,
    tq_grid: np.ndarray | None,
    bob_settings_values: list[int],
    include_tomography: bool,
    num_quadrature_nodes: int | None = None,
    preferred_solver: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a paper-like sweep for route 1 and collect curves for plotting."""
    eta_grid = DEFAULT_PAPER_ETA_VALUES if eta_values is None else np.asarray(eta_values, dtype=float)
    tq_scan = DEFAULT_PAPER_TQ_GRID if tq_grid is None else np.asarray(tq_grid, dtype=float)

    curves: dict[str, list[dict]] = {}
    if include_tomography:
        curves["Tomography"] = sweep_route1_eta(
            source=source,
            bob_mode="tomography",
            eta_values=eta_grid,
            dimension=dimension,
            squeezing_db=squeezing_db,
            num_alice_bins=num_alice_bins,
            num_bob_bins=num_bob_bins,
            tq_grid=tq_scan,
            num_bob_settings=2,
            num_quadrature_nodes=num_quadrature_nodes,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )

    for num_bob_settings in bob_settings_values:
        label = f"mB={num_bob_settings}"
        curves[label] = sweep_route1_eta(
            source=source,
            bob_mode="homodyne",
            eta_values=eta_grid,
            dimension=dimension,
            squeezing_db=squeezing_db,
            num_alice_bins=num_alice_bins,
            num_bob_bins=num_bob_bins,
            tq_grid=tq_scan,
            num_bob_settings=num_bob_settings,
            num_quadrature_nodes=num_quadrature_nodes,
            preferred_solver=preferred_solver,
            verbose=verbose,
        )

    return {
        "route": "route1_paper_like_sweep",
        "source": source,
        "dimension": dimension,
        "squeezing_db": squeezing_db,
        "num_alice_bins": num_alice_bins,
        "num_bob_bins": num_bob_bins,
        "eta_values": eta_grid.tolist(),
        "tq_grid": tq_scan.tolist(),
        "bob_settings_values": list(bob_settings_values),
        "include_tomography": bool(include_tomography),
        "num_quadrature_nodes": default_quadrature_nodes(dimension)
        if num_quadrature_nodes is None
        else int(num_quadrature_nodes),
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone runner for QRNG route 1.")
    parser.add_argument("--mode", choices=["single", "sweep-eta", "paper-sweep"], default="single")
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--source", choices=["tmsv", "split_sms"], default="tmsv")
    parser.add_argument("--bob-mode", choices=["tomography", "homodyne"], default="homodyne")
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--eta-values", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.95, 1.0])
    parser.add_argument("--squeezing-db", type=float, default=-4.0)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--alice-bins", type=int, default=4)
    parser.add_argument("--bob-bins", type=int, default=6)
    parser.add_argument("--bob-settings", type=int, default=2)
    parser.add_argument("--paper-bob-settings", nargs="+", type=int, default=DEFAULT_PAPER_BOB_SETTINGS)
    parser.add_argument("--skip-tomography", action="store_true")
    parser.add_argument("--num-quadrature-nodes", type=int, default=None)
    parser.add_argument("--tq-grid", nargs="+", type=float, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--plot-title", type=str, default=None)
    args = parser.parse_args()

    tq_grid = None if args.tq_grid is None else np.array(args.tq_grid, dtype=float)

    if args.mode == "single":
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
    elif args.mode == "sweep-eta":
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
        title = args.plot_title or (
            f"Route 1 Paper-like Sweep ({args.source}, d={args.dimension}, "
            f"oA={args.alice_bins}, oB={args.bob_bins})"
        )
        _plot_eta_sweep(result["curves"], title, args.output_plot)

    _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
