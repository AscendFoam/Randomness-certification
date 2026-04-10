"""Route4-ex 高输出主线的局部半径精修脚本。

本脚本用于围绕某个已经较强的窗口 / 相位 / 边界组合，只在三输入半径上做
局部网格精修。它不负责做全局搜索，而是用于：

1. 放大观察高值区域；
2. 判断高值是否是孤立尖峰；
3. 为后续 `MOSEK` 复核与病态边界扫描提供候选点。

输出文件会记录所有扫描 case，并实时维护当前可行点中的 `best` 与 `top20`。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .joint_compat_search import PHASE_PATTERNS, WINDOW_LIBRARY
from .prototype import compare_route4_ex_external_diagonal_full


def _parse_float_list(text: str) -> list[float]:
    """解析逗号分隔的浮点列表。"""
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    """解析逗号分隔的整数列表。"""
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_q(text: str) -> list[float]:
    """解析单组 `q_selected` 的 JSON 列表。"""
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("q-config must be a JSON list.")
    return [float(x) for x in payload]


def _build_alpha_values(radii: tuple[float, ...], phases: list[float]) -> list[complex]:
    """把局部精修得到的半径与固定相位组合成振幅列表。"""
    if len(radii) != len(phases):
        raise ValueError("radii and phases must have the same length.")
    return [float(radius) * np.exp(1j * float(phase)) for radius, phase in zip(radii, phases)]


def main() -> None:
    """执行局部半径精修。

    逻辑：
        1. 固定窗口、边界、相位模式和 `q_selected`；
        2. 对 `r1-grid × r2-grid × r3-grid` 做笛卡尔积扫描；
        3. 自动跳过不满足 `r1 < r2 < r3` 的非单调组合；
        4. 对每个 case 调用 external diagonal/full compare；
        5. 实时写出 `best`、`top20` 和全部 `results`。

    适用阶段：
        当大范围搜索已经定位到一个较强窗口后，用本脚本细看局部形貌。
    """
    parser = argparse.ArgumentParser(description="Local radius refinement for route4-ex high-output candidates.")
    parser.add_argument("--probability-path", type=str, required=True)
    parser.add_argument("--variable-name", type=str, default="Probability")
    parser.add_argument("--window", type=str, choices=sorted(WINDOW_LIBRARY.keys()), required=True)
    parser.add_argument("--num-outputs", type=int, required=True)
    parser.add_argument("--edges", type=str, required=True, help="Comma-separated edges, e.g. 0,121,132,256")
    parser.add_argument("--phase-pattern", type=str, choices=sorted(PHASE_PATTERNS.keys()), required=True)
    parser.add_argument("--q-config-json", type=str, default="[1,0,0]")
    parser.add_argument("--r1-grid", type=str, required=True)
    parser.add_argument("--r2-grid", type=str, required=True)
    parser.add_argument("--r3-grid", type=str, required=True)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--max-primal-variables", type=int, default=200000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=50000)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    window = WINDOW_LIBRARY[args.window]
    row_indices = list(window["row_indices"])
    intensities = [float(x) for x in window["intensities"]]
    phases = PHASE_PATTERNS[args.phase_pattern]
    q_values = _parse_q(args.q_config_json)
    edges = _parse_int_list(args.edges)
    r1_grid = _parse_float_list(args.r1_grid)
    r2_grid = _parse_float_list(args.r2_grid)
    r3_grid = _parse_float_list(args.r3_grid)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    case_id = 0
    for radii in itertools.product(r1_grid, r2_grid, r3_grid):
        if not (radii[0] < radii[1] < radii[2]):
            continue
        case_id += 1
        alpha_values = _build_alpha_values(radii, phases)
        record: dict[str, Any] = {
            "case_id": case_id,
            "window": args.window,
            "row_indices": row_indices,
            "intensities": intensities,
            "num_outputs": int(args.num_outputs),
            "edges": edges,
            "phase_pattern": args.phase_pattern,
            "phases": [float(x) for x in phases],
            "model_family": "free_monotone_radii",
            "model_parameters": {"radii": [float(x) for x in radii]},
            "q_values": q_values,
        }
        try:
            result = compare_route4_ex_external_diagonal_full(
                alpha_values=alpha_values,
                q_selected=q_values,
                cutoff=int(args.cutoff),
                probability_path=args.probability_path,
                num_outputs=int(args.num_outputs),
                row_indices=row_indices,
                prob_floor=float(args.prob_floor),
                variable_name=args.variable_name,
                already_coarse=False,
                custom_edges=edges,
                preferred_solver=args.solver,
                verbose=False,
                max_primal_variables=int(args.max_primal_variables),
                max_hermitian_scalar_count=int(args.max_hermitian_scalar_count),
            )
            record.update(
                {
                    "alpha_values": result["instance"]["alpha_values"],
                    "distribution_only_H_min": result["instance"]["distribution_only_H_min"],
                    "full_status": result["full_primal"].get("status"),
                    "full_H_min": result["full_primal"].get("H_min"),
                    "full_p_guess": result["full_primal"].get("p_guess"),
                }
            )
        except Exception as exc:
            record["error"] = type(exc).__name__
            record["error_message"] = str(exc)
        results.append(record)
        feasible = [row for row in results if row.get("full_H_min") is not None]
        feasible.sort(key=lambda row: row["full_H_min"], reverse=True)
        output_path.write_text(
            json.dumps(
                {
                    "best": feasible[0] if feasible else None,
                    "top20": feasible[:20],
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "radii": [float(x) for x in radii],
                    "full_status": record.get("full_status"),
                    "full_H_min": record.get("full_H_min"),
                    "error": record.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
