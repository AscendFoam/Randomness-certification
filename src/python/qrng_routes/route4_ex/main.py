"""Route4-ex 的命令行入口。

本文件负责把 `prototype.py` 中的核心接口暴露成统一 CLI，便于：

1. 快速运行 toy / APD-like / external 三类实例；
2. 在 diagonal primal、full primal、compare 三种模式之间切换；
3. 从命令行传入相干态振幅、外部概率表、coarse-graining 边界以及求解器设置。

搜索脚本通常直接调用 `prototype.py`，而人工调试、快速单点验证和实验室复现时，
更适合直接使用本文件提供的包入口。
"""

from __future__ import annotations

import argparse

from .prototype import (
    DEFAULT_ALPHA_VALUES,
    DEFAULT_CUTOFF,
    DEFAULT_DARK_COUNT_MEAN,
    DEFAULT_DETECTION_EFFICIENCY,
    DEFAULT_DISPLACEMENT_ALPHA,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_RAW_NUM_BINS,
    DEFAULT_PROBE_ALPHA,
    DEFAULT_Q,
    compare_route4_ex_apdlike_diagonal_full,
    compare_route4_ex_external_diagonal_full,
    compare_route4_ex_toy_diagonal_full,
    result_to_json,
    run_route4_ex_apdlike_diagonal_primal,
    run_route4_ex_apdlike_full_primal,
    run_route4_ex_external_diagonal_primal,
    run_route4_ex_external_full_primal,
    run_route4_ex_toy_diagonal_primal,
    run_route4_ex_toy_full_primal,
)


def _parse_complex(text: str) -> complex:
    """解析命令行中的复数振幅字符串。

    功能：
        支持把如 `0.6+0j`、`0+0.6i`、`(0.4+0.4j)` 这类字符串转成 `complex`。

    参数：
        text：命令行输入的复数字符串。

    返回：
        对应的 Python 复数。
    """
    value = text.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    value = value.replace("i", "j")
    return complex(value)


def _parse_int_list(text: str) -> list[int]:
    """把逗号分隔的整数列表解析成 Python 列表。

    参数：
        text：形如 `0,121,132,256` 的字符串。

    返回：
        整数列表。
    """
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    """解析命令行参数并分发到 route4-ex 的不同运行模式。

    功能：
        这是 `python -m qrng_routes.route4_ex` 的真实入口。它根据 `--mode`
        选择：
        - toy 场景；
        - APD-like 理论场景；
        - external probability table 场景；
        并分别调用 diagonal primal、full primal 或 compare 封装函数。

    逻辑：
        1. 解析公共参数与模式参数；
        2. 统一处理 `prob_floor`；
        3. 按 `mode` 分支调用对应 `run_*` / `compare_*` 接口；
        4. 用 `result_to_json(...)` 把结果打印到标准输出。

    参数：
        无显式 Python 参数；所有输入都来自命令行。

    返回：
        无。函数通过 `print(...)` 输出 JSON 结果。
    """
    parser = argparse.ArgumentParser(description="Minimal prototype runner for route4-ex.")
    parser.add_argument(
        "--mode",
        choices=[
            "toy-diagonal-primal",
            "toy-full-primal",
            "toy-compare",
            "apd-diagonal-primal",
            "apd-full-primal",
            "apd-compare",
            "external-diagonal-primal",
            "external-full-primal",
            "external-compare",
        ],
        default="apd-compare",
    )
    parser.add_argument(
        "--alpha-values",
        nargs="+",
        type=_parse_complex,
        default=list(DEFAULT_ALPHA_VALUES),
        help="Trusted input coherent amplitudes, e.g. 0.6+0j 0+0.6j -0.6+0j",
    )
    parser.add_argument("--q-values", nargs="+", type=float, default=list(DEFAULT_Q))
    parser.add_argument("--probe-alpha", type=_parse_complex, default=DEFAULT_PROBE_ALPHA)
    parser.add_argument("--displacement-alpha", type=_parse_complex, default=DEFAULT_DISPLACEMENT_ALPHA)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--num-outputs", type=int, default=DEFAULT_NUM_OUTPUTS)
    parser.add_argument("--raw-num-bins", type=int, default=DEFAULT_RAW_NUM_BINS)
    parser.add_argument("--detection-efficiency", type=float, default=DEFAULT_DETECTION_EFFICIENCY)
    parser.add_argument("--dark-count-mean", type=float, default=DEFAULT_DARK_COUNT_MEAN)
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--external-probability-path", type=str, default=None)
    parser.add_argument("--external-variable-name", type=str, default=None)
    parser.add_argument("--external-row-indices", nargs="+", type=int, default=None)
    parser.add_argument("--external-table-already-coarse", action="store_true")
    parser.add_argument("--custom-edges", type=_parse_int_list, default=None)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-primal-variables", type=int, default=1_000_000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=100_000)
    args = parser.parse_args()

    prob_floor = None if args.prob_floor <= 0 else args.prob_floor
    if args.mode == "toy-diagonal-primal":
        result = run_route4_ex_toy_diagonal_primal(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            probe_alpha=args.probe_alpha,
            prob_floor=prob_floor,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
        )
    elif args.mode == "toy-full-primal":
        result = run_route4_ex_toy_full_primal(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            probe_alpha=args.probe_alpha,
            prob_floor=prob_floor,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    elif args.mode == "apd-diagonal-primal":
        result = run_route4_ex_apdlike_diagonal_primal(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            displacement_alpha=args.displacement_alpha,
            num_outputs=args.num_outputs,
            raw_num_bins=args.raw_num_bins,
            detection_efficiency=args.detection_efficiency,
            dark_count_mean=args.dark_count_mean,
            prob_floor=prob_floor,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
        )
    elif args.mode == "apd-full-primal":
        result = run_route4_ex_apdlike_full_primal(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            displacement_alpha=args.displacement_alpha,
            num_outputs=args.num_outputs,
            raw_num_bins=args.raw_num_bins,
            detection_efficiency=args.detection_efficiency,
            dark_count_mean=args.dark_count_mean,
            prob_floor=prob_floor,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    elif args.mode == "apd-compare":
        result = compare_route4_ex_apdlike_diagonal_full(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            displacement_alpha=args.displacement_alpha,
            num_outputs=args.num_outputs,
            raw_num_bins=args.raw_num_bins,
            detection_efficiency=args.detection_efficiency,
            dark_count_mean=args.dark_count_mean,
            prob_floor=prob_floor,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    elif args.mode == "external-diagonal-primal":
        result = run_route4_ex_external_diagonal_primal(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            probability_path=args.external_probability_path,
            num_outputs=args.num_outputs,
            row_indices=args.external_row_indices,
            prob_floor=prob_floor,
            variable_name=args.external_variable_name,
            already_coarse=bool(args.external_table_already_coarse),
            custom_edges=args.custom_edges,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
        )
    elif args.mode == "external-full-primal":
        result = run_route4_ex_external_full_primal(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            probability_path=args.external_probability_path,
            num_outputs=args.num_outputs,
            row_indices=args.external_row_indices,
            prob_floor=prob_floor,
            variable_name=args.external_variable_name,
            already_coarse=bool(args.external_table_already_coarse),
            custom_edges=args.custom_edges,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    else:
        result = compare_route4_ex_external_diagonal_full(
            alpha_values=list(args.alpha_values),
            q_selected=list(args.q_values),
            cutoff=args.cutoff,
            probability_path=args.external_probability_path,
            num_outputs=args.num_outputs,
            row_indices=args.external_row_indices,
            prob_floor=prob_floor,
            variable_name=args.external_variable_name,
            already_coarse=bool(args.external_table_already_coarse),
            custom_edges=args.custom_edges,
            preferred_solver=args.solver,
            verbose=args.verbose,
            max_primal_variables=args.max_primal_variables,
            max_hermitian_scalar_count=args.max_hermitian_scalar_count,
        )
    print(result_to_json(result))


if __name__ == "__main__":
    main()
