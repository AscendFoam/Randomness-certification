"""Route4-ex 的高输出模型 / 窗口联合搜索脚本。

本脚本用于回答一个更结构性的问题：

- 当输出数增加到 3/4、边界选成高熵 contiguous bins 时，
  改 trusted-input 模型或改输入窗口，formal 结果会不会明显变强？

它会在多个窗口、多个输出数、多个相位模式以及两类输入模型之间扫描：

1. `rigid_intensity_scaled`
2. `free_monotone_radii`

输出文件适合做路线比较与高输出可行性诊断。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .joint_compat_search import PHASE_PATTERNS, WINDOW_LIBRARY, _top_contiguous_edges
from .prototype import compare_route4_ex_external_diagonal_full, intensities_to_alpha_values, load_external_probability_table


def _parse_float_list(text: str) -> list[float]:
    """解析逗号分隔的浮点列表。"""
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_q(text: str) -> list[float]:
    """解析单组 `q_selected` 的 JSON 列表。"""
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("q-config must be a JSON list.")
    return [float(x) for x in payload]


def _build_free_alpha_values(radii: tuple[float, ...], phases: list[float]) -> list[complex]:
    """把自由半径模型的半径与相位组合成振幅列表。"""
    if len(radii) != len(phases):
        raise ValueError("radii and phases must have the same length.")
    return [float(radius) * np.exp(1j * float(phase)) for radius, phase in zip(radii, phases)]


def main() -> None:
    """执行窗口 / 模型 / 高输出联合搜索。

    逻辑：
        1. 读入外部概率表；
        2. 对每个窗口提取目标输入行；
        3. 为每个输出数自动生成高熵 contiguous edge 候选；
        4. 在相位模式、刚性强度缩放模型和自由半径模型之间扫描；
        5. 对每个 case 记录 formal 结果并实时更新输出 JSON。

    输出解释：
        - `best`：目前已找到的最佳 formal case；
        - `results`：所有 case 的完整记录；
        - 每条记录里 `model_family` 与 `model_parameters` 表示 trusted-input 的建模方式。
    """
    parser = argparse.ArgumentParser(
        description="Search whether high-entropy 3/4-output bins become feasible after changing trusted-state model or input window."
    )
    parser.add_argument("--probability-path", type=str, required=True)
    parser.add_argument("--variable-name", type=str, default="Probability")
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=sorted(WINDOW_LIBRARY.keys()),
        default=["100_120_140", "80_100_120", "100_140_160", "120_140_160"],
    )
    parser.add_argument("--num-outputs", nargs="+", type=int, default=[3, 4])
    parser.add_argument(
        "--phase-patterns",
        nargs="+",
        choices=sorted(PHASE_PATTERNS.keys()),
        default=["0_pi2_pi", "0_pi3_2pi3"],
    )
    parser.add_argument("--rigid-alpha-values", type=str, default="0.57,0.60,0.63,0.66,0.69")
    parser.add_argument("--free-radii-grid", type=str, default="0.48,0.54,0.60,0.66,0.72")
    parser.add_argument("--q-config-json", type=str, default="[1,0,0]")
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--max-primal-variables", type=int, default=200000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=50000)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    probability_table = load_external_probability_table(args.probability_path, variable_name=args.variable_name)
    q_values = _parse_q(args.q_config_json)
    rigid_alpha_values = _parse_float_list(args.rigid_alpha_values)
    free_radii_grid = _parse_float_list(args.free_radii_grid)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    case_id = 0

    for window_name in args.windows:
        window = WINDOW_LIBRARY[window_name]
        row_indices = list(window["row_indices"])
        intensities = [float(x) for x in window["intensities"]]
        target_row = np.asarray(probability_table[row_indices[0], :], dtype=float)
        target_row = target_row / target_row.sum()

        for num_outputs in args.num_outputs:
            edge_candidate = _top_contiguous_edges(
                target_row,
                num_outputs=int(num_outputs),
                top_k=1,
            )[0]
            edges = list(edge_candidate["edges"])

            for phase_pattern_name in args.phase_patterns:
                phases = PHASE_PATTERNS[phase_pattern_name]

                for max_abs_alpha in rigid_alpha_values:
                    alpha_values = intensities_to_alpha_values(
                        intensities,
                        max_abs_alpha=float(max_abs_alpha),
                        phases=phases,
                        max_intensity=max(intensities),
                    )
                    case_id += 1
                    record: dict[str, Any] = {
                        "case_id": case_id,
                        "window": window_name,
                        "row_indices": row_indices,
                        "intensities": intensities,
                        "num_outputs": int(num_outputs),
                        "edges": edges,
                        "phase_pattern": phase_pattern_name,
                        "phases": [float(x) for x in phases],
                        "model_family": "rigid_intensity_scaled",
                        "model_parameters": {"max_abs_alpha": float(max_abs_alpha)},
                        "q_values": q_values,
                    }
                    try:
                        result = compare_route4_ex_external_diagonal_full(
                            alpha_values=alpha_values,
                            q_selected=q_values,
                            cutoff=int(args.cutoff),
                            probability_path=args.probability_path,
                            num_outputs=int(num_outputs),
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
                                "window": window_name,
                                "num_outputs": int(num_outputs),
                                "phase_pattern": phase_pattern_name,
                                "model_family": record["model_family"],
                                "model_parameters": record["model_parameters"],
                                "edges": edges,
                                "full_status": record.get("full_status"),
                                "full_H_min": record.get("full_H_min"),
                                "error": record.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

                for radii in itertools.combinations(free_radii_grid, len(intensities)):
                    alpha_values = _build_free_alpha_values(radii, phases)
                    case_id += 1
                    record = {
                        "case_id": case_id,
                        "window": window_name,
                        "row_indices": row_indices,
                        "intensities": intensities,
                        "num_outputs": int(num_outputs),
                        "edges": edges,
                        "phase_pattern": phase_pattern_name,
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
                            num_outputs=int(num_outputs),
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
                                "window": window_name,
                                "num_outputs": int(num_outputs),
                                "phase_pattern": phase_pattern_name,
                                "model_family": record["model_family"],
                                "model_parameters": record["model_parameters"],
                                "edges": edges,
                                "full_status": record.get("full_status"),
                                "full_H_min": record.get("full_H_min"),
                                "error": record.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

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


if __name__ == "__main__":
    main()
