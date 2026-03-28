from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .route3.cv_four_phase import dual_homodyne_probabilities, phase_alphabet, reduced_joint_inputs
from .route5.hybrid_iq import reduced_joint_inputs_from_alphas
from .route5.intensity_menu_search import intensity_menu_to_radii


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


def _interval_probabilities(bounds: np.ndarray, mean: float) -> np.ndarray:
    """Gaussian interval masses for N(mean, 1/2)."""
    edges = np.asarray(bounds, dtype=float)
    values: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        erf_lower = -1.0 if np.isneginf(lower) else math.erf(float(lower - mean))
        erf_upper = 1.0 if np.isposinf(upper) else math.erf(float(upper - mean))
        values.append(0.5 * (erf_upper - erf_lower))
    return np.asarray(values, dtype=float)


def ideal_dual_homodyne_bin_probabilities(
    alpha_1: complex,
    alpha_2: complex,
    x_bounds: np.ndarray,
    p_bounds: np.ndarray,
) -> np.ndarray:
    """Closed-form ideal probabilities for the current route3/route5 dual-homodyne convention.

    Under the balanced-beamsplitter convention used in this repository, the discretized
    route3/route5 measurement corresponds to:

    - x-axis mean: Re(alpha_1 + alpha_2)
    - p-axis mean: Im(alpha_1 - alpha_2)

    with both quadratures having variance 1/2. For axis-aligned bins, the joint
    probability is the product of two 1-D Gaussian interval masses.
    """

    mean_x = float(np.real(alpha_1 + alpha_2))
    mean_p = float(np.imag(alpha_1 - alpha_2))
    px = _interval_probabilities(np.asarray(x_bounds, dtype=float), mean_x)
    pp = _interval_probabilities(np.asarray(p_bounds, dtype=float), mean_p)
    return np.kron(px, pp)


def _error_summary(errors: list[float]) -> dict[str, float]:
    if len(errors) == 0:
        return {"max_abs_error": 0.0, "mean_abs_error": 0.0, "p95_abs_error": 0.0}
    sorted_errors = sorted(float(item) for item in errors)
    p95_index = min(len(sorted_errors) - 1, int(0.95 * len(sorted_errors)))
    return {
        "max_abs_error": float(sorted_errors[-1]),
        "mean_abs_error": float(sum(sorted_errors) / len(sorted_errors)),
        "p95_abs_error": float(sorted_errors[p95_index]),
    }


def verify_route3_tex_case(
    mu: float = 0.05,
    cutoff: int = 12,
    num_x_bins: int = 2,
    num_p_bins: int = 2,
    quadrature_range: float = 3.0,
    num_nodes: int = 400,
) -> dict[str, Any]:
    """Compare route3 trace probabilities to the tex-style analytic formula in the real-input case."""

    joint_states, labels, joint_basis, local_rank, joint_dim = reduced_joint_inputs(
        mu,
        cutoff,
        num_phases=2,
    )
    probabilities, output_labels, x_bounds, p_bounds = dual_homodyne_probabilities(
        joint_states,
        joint_basis,
        cutoff,
        num_x_bins=num_x_bins,
        num_p_bins=num_p_bins,
        quadrature_range=quadrature_range,
        num_nodes=num_nodes,
    )
    _, phases = phase_alphabet(mu, cutoff, num_phases=2)
    local_alphas = [math.sqrt(mu) * np.exp(1j * phase) for phase in phases]

    per_input = []
    errors: list[float] = []
    for input_index, (x, y) in enumerate(labels):
        analytic = ideal_dual_homodyne_bin_probabilities(
            local_alphas[x],
            local_alphas[y],
            x_bounds,
            p_bounds,
        )
        current = probabilities[input_index]
        abs_error = float(np.max(np.abs(analytic - current)))
        errors.append(abs_error)
        per_input.append(
            {
                "input_index": int(input_index),
                "label": [int(x), int(y)],
                "alpha_1": {
                    "real": float(np.real(local_alphas[x])),
                    "imag": float(np.imag(local_alphas[x])),
                },
                "alpha_2": {
                    "real": float(np.real(local_alphas[y])),
                    "imag": float(np.imag(local_alphas[y])),
                },
                "max_abs_error": abs_error,
            }
        )

    return {
        "case": "route3_tex_compatible",
        "mu": float(mu),
        "cutoff": int(cutoff),
        "num_x_bins": int(num_x_bins),
        "num_p_bins": int(num_p_bins),
        "quadrature_range": float(quadrature_range),
        "num_nodes": int(num_nodes),
        "x_bounds": np.asarray(x_bounds, dtype=float).tolist(),
        "p_bounds": np.asarray(p_bounds, dtype=float).tolist(),
        "output_labels": [list(label) for label in output_labels],
        "note": (
            "This matches SDP_solve.tex after accounting for the repository's current "
            "balanced-beamsplitter / p-axis sign convention."
        ),
        "summary": _error_summary(errors),
        "per_input": per_input,
    }


def verify_route5_fixed_intensity_case(
    intensity_values: list[float] | None = None,
    max_radius: float = 1.2,
    cutoff_values: list[int] | None = None,
    phase_values: list[float] | None = None,
    num_x_bins: int = 6,
    num_p_bins: int = 2,
    quadrature_range: float = 1.8,
    num_nodes: int = 1200,
) -> dict[str, Any]:
    """Compare route5 trace probabilities to the ideal analytic formula under fixed-intensity constraints."""

    intensity_values = [0.0, 80.0, 160.0] if intensity_values is None else list(intensity_values)
    cutoff_values = [4, 8, 12] if cutoff_values is None else list(cutoff_values)
    phase_values = (
        [
            0.0,
            0.25 * math.pi,
            0.5 * math.pi,
            0.75 * math.pi,
            math.pi,
            1.25 * math.pi,
            1.5 * math.pi,
            1.75 * math.pi,
        ]
        if phase_values is None
        else list(phase_values)
    )

    radii, mapping = intensity_menu_to_radii(
        intensity_values=intensity_values,
        max_radius=max_radius,
        require_vacuum=True,
    )
    x_bounds = np.array(
        [-np.inf, -1.2, -0.6, 0.0, 0.6, 1.2, np.inf],
        dtype=float,
    )
    p_bounds = np.array([-np.inf, 0.0, np.inf], dtype=float)

    cutoff_results: list[dict[str, Any]] = []
    for cutoff in cutoff_values:
        joint_states, labels, joint_basis, local_alphas, local_rank, joint_dim, local_span = (
            reduced_joint_inputs_from_alphas(
                cutoff=cutoff,
                radius_values=radii,
                phase_values=phase_values,
            )
        )
        probabilities, output_labels, x_bounds_out, p_bounds_out = dual_homodyne_probabilities(
            joint_states,
            joint_basis,
            cutoff,
            num_x_bins=num_x_bins,
            num_p_bins=num_p_bins,
            x_bounds=x_bounds,
            p_bounds=p_bounds,
            num_nodes=num_nodes,
        )

        per_input = []
        errors: list[float] = []
        for input_index, (x, y) in enumerate(labels):
            analytic = ideal_dual_homodyne_bin_probabilities(
                local_alphas[x],
                local_alphas[y],
                x_bounds_out,
                p_bounds_out,
            )
            current = probabilities[input_index]
            abs_error = float(np.max(np.abs(analytic - current)))
            errors.append(abs_error)
            per_input.append(
                {
                    "input_index": int(input_index),
                    "label": [int(x), int(y)],
                    "max_abs_error": abs_error,
                }
            )

        cutoff_results.append(
            {
                "cutoff": int(cutoff),
                "num_nodes": int(num_nodes),
                "num_inputs": int(len(labels)),
                "num_outputs": int(probabilities.shape[1]),
                "local_rank": int(local_rank),
                "joint_dim": int(joint_dim),
                "local_operator_span_rank": int(local_span),
                "summary": _error_summary(errors),
                "top_error_inputs": sorted(per_input, key=lambda item: item["max_abs_error"], reverse=True)[:10],
            }
        )

    return {
        "case": "route5_fixed_intensity_080160",
        "intensity_values": [float(value) for value in intensity_values],
        "max_radius": float(max_radius),
        "scaling_rule": "radius = max_radius * sqrt(intensity / max_intensity_in_menu)",
        "mapped_radii": [float(value) for value in radii],
        "radius_mapping": mapping,
        "phase_values": [float(value) for value in phase_values],
        "num_x_bins": int(num_x_bins),
        "num_p_bins": int(num_p_bins),
        "quadrature_range": float(quadrature_range),
        "x_bounds": x_bounds.tolist(),
        "p_bounds": p_bounds.tolist(),
        "cutoff_results": cutoff_results,
        "note": (
            "Unlike the tex-compatible route3 test, route5 comparisons should be interpreted "
            "as a convergence diagnostic because the current formal route5 pipeline uses a low "
            "Fock cutoff, while the analytic formula corresponds to the ideal infinite-dimensional model."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SDP_solve-style integral probabilities against the current trace-based route3/route5 implementation."
    )
    parser.add_argument(
        "--mode",
        choices=["route3", "route5", "both"],
        default="both",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload: dict[str, Any] = {
        "route3": None,
        "route5": None,
    }
    if args.mode in ("route3", "both"):
        payload["route3"] = verify_route3_tex_case()
    if args.mode in ("route5", "both"):
        payload["route5"] = verify_route5_fixed_intensity_case()

    _write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
