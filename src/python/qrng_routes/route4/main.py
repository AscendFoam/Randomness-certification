from __future__ import annotations

import argparse

from .phaseinsensitive import (
    DEFAULT_CUTOFF,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_PROB_FLOOR,
    DEFAULT_Q,
    DEFAULT_SELECTED_MU,
    FULL_MU,
    compare_route4_primal_full,
    compare_route4_primal_dual,
    result_to_json,
    run_route4_diagonal_projection_invariance_check,
    run_route4_dual,
    run_route4_nondiagonal_relaxation_check,
    run_route4_primal,
    search_route4_contiguous_edges,
    solve_phaseinsensitive_full_primal,
    search_route4_triplets,
    sweep_route4_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone runner for QRNG route 4.")
    parser.add_argument(
        "--mode",
        choices=[
            "dual-single",
            "primal-single",
            "full-primal-single",
            "primal-dual-compare",
            "primal-full-compare",
            "output-sweep",
            "subset-search",
            "contiguous-search",
            "nondiagonal-check",
            "diagonal-projection-check",
        ],
        default="dual-single",
    )
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--selected-mu", nargs="+", type=int, default=list(DEFAULT_SELECTED_MU))
    parser.add_argument("--q-values", nargs="+", type=float, default=list(DEFAULT_Q))
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--num-outputs", type=int, default=DEFAULT_NUM_OUTPUTS)
    parser.add_argument("--output-values", nargs="+", type=int, default=[4, 6, 8, 12, 16])
    parser.add_argument("--prob-floor", type=float, default=DEFAULT_PROB_FLOOR)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--certify-top-k", type=int, default=3)
    parser.add_argument("--full-mu", nargs="+", type=int, default=list(FULL_MU))
    parser.add_argument("--max-primal-variables", type=int, default=3_000_000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--custom-edges", nargs="+", type=int, default=None)
    parser.add_argument("--record-top-k", type=int, default=10)
    parser.add_argument("--min-bin-width", type=int, default=1)
    args = parser.parse_args()

    prob_floor = None if args.prob_floor <= 0 else args.prob_floor

    if args.mode == "dual-single":
        result = run_route4_dual(
            num_outputs=args.num_outputs,
            selected_mu_list=args.selected_mu,
            q_selected=args.q_values,
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            custom_edges=args.custom_edges,
        )
    elif args.mode == "primal-single":
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
            custom_edges=args.custom_edges,
        )
    elif args.mode == "full-primal-single":
        from .phaseinsensitive import prepare_phaseinsensitive_instance

        instance = prepare_phaseinsensitive_instance(
            num_outputs=args.num_outputs,
            selected_mu_list=args.selected_mu,
            q_selected=args.q_values,
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            custom_edges=args.custom_edges,
        )
        result = solve_phaseinsensitive_full_primal(
            instance,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    elif args.mode == "primal-dual-compare":
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
            custom_edges=args.custom_edges,
        )
    elif args.mode == "primal-full-compare":
        result = compare_route4_primal_full(
            num_outputs=args.num_outputs,
            selected_mu_list=args.selected_mu,
            q_selected=args.q_values,
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
            custom_edges=args.custom_edges,
        )
    elif args.mode == "output-sweep":
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
    elif args.mode == "subset-search":
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
    elif args.mode == "contiguous-search":
        result = search_route4_contiguous_edges(
            num_outputs=args.num_outputs,
            selected_mu_list=args.selected_mu,
            q_selected=args.q_values,
            cutoff=args.cutoff,
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            certify_top_k=args.certify_top_k,
            record_top_k=args.record_top_k,
            min_bin_width=args.min_bin_width,
        )
    elif args.mode == "diagonal-projection-check":
        result = run_route4_diagonal_projection_invariance_check(
            seed=args.seed,
            num_trials=args.num_trials,
        )
    else:
        result = run_route4_nondiagonal_relaxation_check(
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    print(result_to_json(result))


if __name__ == "__main__":
    main()
