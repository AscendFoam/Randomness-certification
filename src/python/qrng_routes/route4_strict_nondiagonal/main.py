"""Route4 strict non-diagonal 的命令行入口。"""

from __future__ import annotations

import argparse

import numpy as np

from .prototype import (
    DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
    compare_route4_strict_nondiagonal_with_reference,
    prepare_route4_strict_nondiagonal_instance,
    result_to_json,
    run_route4_strict_nondiagonal_full_primal,
)
from ..route4.phaseinsensitive import (
    DEFAULT_CUTOFF,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_PROB_FLOOR,
    DEFAULT_Q,
    DEFAULT_SELECTED_MU,
)


def _parse_float_list(text: str) -> list[float]:
    """把逗号分隔的浮点列表解析成 Python 列表。"""
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    """解析命令行参数并分发到 strict non-diagonal route4 的不同模式。"""
    parser = argparse.ArgumentParser(description="Strict non-diagonal extension of route4.")
    parser.add_argument(
        "--mode",
        choices=["prepare-instance", "full-primal-single", "compare-reference"],
        default="compare-reference",
    )
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--selected-mu", nargs="+", type=int, default=list(DEFAULT_SELECTED_MU))
    parser.add_argument("--q-values", nargs="+", type=float, default=list(DEFAULT_Q))
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--num-outputs", type=int, default=DEFAULT_NUM_OUTPUTS)
    parser.add_argument("--prob-floor", type=float, default=DEFAULT_PROB_FLOOR)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--probability-path", type=str, default=None)
    parser.add_argument("--phase-values", type=str, default=None, help="Comma-separated phase list in radians.")
    parser.add_argument(
        "--mean-photons-per-mu-label",
        type=float,
        default=DEFAULT_MEAN_PHOTONS_PER_MU_LABEL,
        help="Fixed conversion factor from route4 mu labels to coherent-state mean photon numbers.",
    )
    parser.add_argument("--support-tol", type=float, default=1e-9)
    parser.add_argument("--max-primal-variables", type=int, default=3_000_000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=400_000)
    args = parser.parse_args()

    prob_floor = None if args.prob_floor <= 0 else args.prob_floor
    phase_values = None if args.phase_values is None else _parse_float_list(args.phase_values)

    if args.mode == "prepare-instance":
        instance = prepare_route4_strict_nondiagonal_instance(
            num_outputs=args.num_outputs,
            selected_mu_list=list(args.selected_mu),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            probability_path=args.probability_path,
            phase_values=phase_values,
            mean_photons_per_mu_label=args.mean_photons_per_mu_label,
            support_tol=args.support_tol,
        )
        result = {
            "route": "route4_strict_nondiagonal_prepare_instance",
            "instance": {
                "selected_mu_list": list(instance["selected_mu_list"]),
                "q_selected": np.asarray(instance["q_selected"], dtype=float).tolist(),
                "cutoff": int(instance["cutoff"]),
                "num_outputs": int(instance["num_outputs"]),
                "edges": np.asarray(instance["edges"], dtype=int).tolist(),
                "alpha_values": [
                    {
                        "real": float(np.real(alpha)),
                        "imag": float(np.imag(alpha)),
                        "abs": float(abs(alpha)),
                        "phase": float(np.angle(alpha)),
                    }
                    for alpha in instance["alpha_values"]
                ],
                "phase_values": [float(value) for value in instance["phase_values"]],
                "mean_photon_numbers": [float(value) for value in instance["mean_photon_numbers"]],
                "support_dimension": int(instance["support_dimension"]),
                "rho_diag_reference_linf_gap": float(instance["rho_diag_reference_linf_gap"]),
                "distribution_only_p_guess": float(instance["distribution_only_p_guess"]),
            },
        }
    elif args.mode == "full-primal-single":
        result = run_route4_strict_nondiagonal_full_primal(
            num_outputs=args.num_outputs,
            selected_mu_list=list(args.selected_mu),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            probability_path=args.probability_path,
            phase_values=phase_values,
            mean_photons_per_mu_label=args.mean_photons_per_mu_label,
            support_tol=args.support_tol,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    else:
        result = compare_route4_strict_nondiagonal_with_reference(
            num_outputs=args.num_outputs,
            selected_mu_list=list(args.selected_mu),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            probability_path=args.probability_path,
            phase_values=phase_values,
            mean_photons_per_mu_label=args.mean_photons_per_mu_label,
            support_tol=args.support_tol,
            max_primal_variables=args.max_primal_variables,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    print(result_to_json(result))


if __name__ == "__main__":
    main()
