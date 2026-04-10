"""Route4-ex 包级导出入口。

本文件不承载核心算法，而是把 `prototype.py` 中最常用的常量、实例构造函数、
求解器接口与 compare 封装统一导出，方便外部脚本或交互式环境直接：

- `from qrng_routes.route4_ex import ...`

使用 route4-ex 的主要能力。
"""

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
    coarse_grain_probability_table_with_edges,
    intensity_to_alpha,
    intensities_to_alpha_values,
    prepare_route4_ex_apdlike_instance,
    prepare_route4_ex_external_instance,
    prepare_route4_ex_toy_instance,
    result_to_json,
    run_route4_ex_apdlike_diagonal_primal,
    run_route4_ex_apdlike_full_primal,
    run_route4_ex_external_diagonal_primal,
    run_route4_ex_external_full_primal,
    run_route4_ex_toy_diagonal_primal,
    run_route4_ex_toy_full_primal,
    solve_route4_ex_diagonal_primal,
    solve_route4_ex_full_primal,
)

__all__ = [
    "DEFAULT_ALPHA_VALUES",
    "DEFAULT_CUTOFF",
    "DEFAULT_DARK_COUNT_MEAN",
    "DEFAULT_DETECTION_EFFICIENCY",
    "DEFAULT_DISPLACEMENT_ALPHA",
    "DEFAULT_NUM_OUTPUTS",
    "DEFAULT_RAW_NUM_BINS",
    "DEFAULT_PROBE_ALPHA",
    "DEFAULT_Q",
    "compare_route4_ex_apdlike_diagonal_full",
    "compare_route4_ex_external_diagonal_full",
    "compare_route4_ex_toy_diagonal_full",
    "coarse_grain_probability_table_with_edges",
    "intensity_to_alpha",
    "intensities_to_alpha_values",
    "prepare_route4_ex_apdlike_instance",
    "prepare_route4_ex_external_instance",
    "prepare_route4_ex_toy_instance",
    "result_to_json",
    "run_route4_ex_apdlike_diagonal_primal",
    "run_route4_ex_apdlike_full_primal",
    "run_route4_ex_external_diagonal_primal",
    "run_route4_ex_external_full_primal",
    "run_route4_ex_toy_diagonal_primal",
    "run_route4_ex_toy_full_primal",
    "solve_route4_ex_diagonal_primal",
    "solve_route4_ex_full_primal",
]
