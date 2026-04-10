"""Route4-ex-constrained 的命令行入口。

该 CLI 只暴露一条收缩后的核心主线：

1. 从 `Probability.mat` 中读取固定输入窗口；
2. 使用固定 trusted alphas；
3. 使用固定 coarse-graining 边界；
4. 运行 full primal，或做 diagonal/full 对照。
"""

from __future__ import annotations

import argparse

from .prototype import (
    DEFAULT_ALPHA_VALUES,
    DEFAULT_CUTOFF,
    DEFAULT_CUSTOM_EDGES,
    DEFAULT_PROBABILITY_PATH,
    DEFAULT_PROB_FLOOR,
    DEFAULT_Q,
    DEFAULT_SELECTED_MU,
    DEFAULT_VARIABLE_NAME,
    compare_route4_ex_constrained_diagonal_full,
    prepare_route4_ex_constrained_instance,
    result_to_json,
    run_route4_ex_constrained_full_primal,
    summarize_route4_ex_constrained_instance,
)


def _parse_complex(text: str) -> complex:
    """解析命令行里的复振幅字符串。"""
    value = text.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    value = value.replace("i", "j")
    return complex(value)


def _parse_int_list(text: str) -> list[int]:
    """把逗号分隔的整数边界字符串转成列表。"""
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    """解析参数并运行 route4-ex-constrained。"""
    parser = argparse.ArgumentParser(description="Core constrained runner for route4-ex.")
    parser.add_argument(
        "--mode",
        choices=["prepare-instance", "full-primal", "compare"],
        default="compare",
    )
    parser.add_argument("--selected-mu", nargs="+", type=int, default=list(DEFAULT_SELECTED_MU))
    parser.add_argument("--q-values", nargs="+", type=float, default=list(DEFAULT_Q))
    parser.add_argument("--alpha-values", nargs="+", type=_parse_complex, default=list(DEFAULT_ALPHA_VALUES))
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--custom-edges", type=_parse_int_list, default=list(DEFAULT_CUSTOM_EDGES))
    parser.add_argument("--probability-path", type=str, default=str(DEFAULT_PROBABILITY_PATH))
    parser.add_argument("--variable-name", type=str, default=DEFAULT_VARIABLE_NAME)
    parser.add_argument("--prob-floor", type=float, default=DEFAULT_PROB_FLOOR)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-primal-variables", type=int, default=100_000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=50_000)
    args = parser.parse_args()

    prob_floor = None if args.prob_floor <= 0 else args.prob_floor
    if args.mode == "prepare-instance":
        instance = prepare_route4_ex_constrained_instance(
            selected_mu_list=list(args.selected_mu),
            q_selected=list(args.q_values),
            alpha_values=list(args.alpha_values),
            cutoff=args.cutoff,
            probability_path=args.probability_path,
            variable_name=args.variable_name,
            custom_edges=list(args.custom_edges),
            prob_floor=prob_floor,
            shift=args.shift,
        )
        result = summarize_route4_ex_constrained_instance(instance)
    elif args.mode == "full-primal":
        result = run_route4_ex_constrained_full_primal(
            selected_mu_list=list(args.selected_mu),
            q_selected=list(args.q_values),
            alpha_values=list(args.alpha_values),
            cutoff=args.cutoff,
            probability_path=args.probability_path,
            variable_name=args.variable_name,
            custom_edges=list(args.custom_edges),
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    else:
        result = compare_route4_ex_constrained_diagonal_full(
            selected_mu_list=list(args.selected_mu),
            q_selected=list(args.q_values),
            alpha_values=list(args.alpha_values),
            cutoff=args.cutoff,
            probability_path=args.probability_path,
            variable_name=args.variable_name,
            custom_edges=list(args.custom_edges),
            prob_floor=prob_floor,
            shift=args.shift,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    print(result_to_json(result))


if __name__ == "__main__":
    main()

