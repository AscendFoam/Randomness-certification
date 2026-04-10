"""Route4-ex 的联合兼容性搜索脚本。

本脚本用于同时搜索：

1. 输入窗口；
2. coarse-graining 边界；
3. 相位模式；
4. 强度到振幅的映射尺度；
5. 生成轮权重 `q_selected`。

它的目标不是只看某一维参数，而是回答：

- 某个“看起来高熵”的输出边界，是否能与一组 trusted inputs 和测试约束
  在 formal SDP 中真正兼容。

因此它比简单的单参数扫描更适合做“为什么高熵边界会 formal infeasible”
这类诊断问题。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .prototype import compare_route4_ex_external_diagonal_full, intensities_to_alpha_values, load_external_probability_table


PHASE_PATTERNS: dict[str, list[float]] = {
    "0_pi2_pi": [0.0, math.pi / 2.0, math.pi],
    "0_pi3_2pi3": [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0],
    "0_pi4_pi2": [0.0, math.pi / 4.0, math.pi / 2.0],
    "0_pi4_3pi4": [0.0, math.pi / 4.0, 3.0 * math.pi / 4.0],
}


WINDOW_LIBRARY: dict[str, dict[str, list[int] | list[float]]] = {
    "100_120_140": {"row_indices": [5, 6, 7], "intensities": [100.0, 120.0, 140.0]},
    "80_100_120": {"row_indices": [4, 5, 6], "intensities": [80.0, 100.0, 120.0]},
    "100_140_160": {"row_indices": [5, 7, 8], "intensities": [100.0, 140.0, 160.0]},
    "120_140_160": {"row_indices": [6, 7, 8], "intensities": [120.0, 140.0, 160.0]},
    "60_100_140": {"row_indices": [3, 5, 7], "intensities": [60.0, 100.0, 140.0]},
}


def _parse_float_list(text: str) -> list[float]:
    """解析逗号分隔的浮点列表。"""
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_nested_float_lists(text: str) -> list[list[float]]:
    """解析多组 `q_selected` 的 JSON 字符串。"""
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list.")
    rows: list[list[float]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("Each q configuration must be a JSON list.")
        rows.append([float(x) for x in row])
    return rows


def _distribution_only_hmin(row: np.ndarray, edges: list[int]) -> float:
    """计算单行概率在给定边界下的 distribution-only `H_min`。"""
    coarse = np.asarray([row[edges[i] : edges[i + 1]].sum() for i in range(len(edges) - 1)], dtype=float)
    return float(-math.log2(float(np.max(coarse))))


def _top_contiguous_edges(
    row: np.ndarray,
    *,
    num_outputs: int,
    top_k: int,
    local_window: int = 20,
) -> list[dict[str, Any]]:
    """为目标输入行挑选若干 contiguous 高熵边界候选。

    功能：
        在 2/3/4 输出情形下，从单行目标概率分布出发搜索高 `distribution-only`
        熵的连续边界划分，作为 formal 联合兼容性检查的候选输入。

    参数：
        row：目标输入行的原始概率分布。
        num_outputs：输出数，只支持 2/3/4。
        top_k：保留多少个候选边界。
        local_window：4 输出时围绕分位点的局部搜索宽度。

    返回：
        候选列表。每个候选包含边界、目标行的 distribution-only 熵，以及
        对应 coarse 概率。
    """
    row = np.asarray(row, dtype=float).reshape(-1)
    row = row / row.sum()
    num_bins = int(row.size)
    candidates: list[tuple[float, list[int], list[float]]] = []

    if num_outputs == 2:
        for a in range(1, num_bins):
            edges = [0, a, num_bins]
            probs = [float(row[:a].sum()), float(row[a:].sum())]
            h = float(-math.log2(max(probs)))
            candidates.append((h, edges, probs))
    elif num_outputs == 3:
        prefix = np.cumsum(row)
        for a in range(1, num_bins - 1):
            left = float(prefix[a - 1])
            for b in range(a + 1, num_bins):
                probs = [left, float(prefix[b - 1] - prefix[a - 1]), float(1.0 - prefix[b - 1])]
                h = float(-math.log2(max(probs)))
                candidates.append((h, [0, a, b, num_bins], probs))
    elif num_outputs == 4:
        cdf = np.cumsum(row)
        q1 = int(np.searchsorted(cdf, 0.25))
        q2 = int(np.searchsorted(cdf, 0.50))
        q3 = int(np.searchsorted(cdf, 0.75))
        for a in range(max(1, q1 - local_window), min(num_bins - 2, q1 + local_window) + 1):
            for b in range(max(a + 1, q2 - local_window), min(num_bins - 1, q2 + local_window) + 1):
                for c in range(max(b + 1, q3 - local_window), min(num_bins, q3 + local_window) + 1):
                    edges = [0, a, b, c, num_bins]
                    probs = [float(row[edges[i] : edges[i + 1]].sum()) for i in range(4)]
                    h = float(-math.log2(max(probs)))
                    candidates.append((h, edges, probs))
    else:
        raise ValueError("Only num_outputs in {2,3,4} are supported in this first-round joint search.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for h, edges, probs in candidates:
        key = tuple(edges)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(
            {
                "edges": edges,
                "distribution_only_H_min_target_row": h,
                "target_row_probs": probs,
            }
        )
        if len(dedup) >= top_k:
            break
    return dedup


def main() -> None:
    """执行联合兼容性搜索。

    逻辑：
        1. 按窗口读取外部概率表的候选输入行；
        2. 为每个输出数生成若干高熵边界候选，并加入等分边界作为对照；
        3. 对每个边界，再扫描相位、振幅尺度和 `q_selected`；
        4. 逐 case 调用 external diagonal/full compare；
        5. 持续把当前最优结果写入输出 JSON。

    输出解释：
        该脚本的输出最适合回答“某条高熵边界为什么 formal 上不去”这类问题，
        因为它会同时记录：
        - 边界来自哪一条目标行；
        - 目标行的 distribution-only 熵；
        - formal full-primal 的最终状态和结果。
    """
    parser = argparse.ArgumentParser(description="Joint compatibility search for route4-ex windows.")
    parser.add_argument("--probability-path", type=str, required=True)
    parser.add_argument("--variable-name", type=str, default="Probability")
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=sorted(WINDOW_LIBRARY.keys()),
        default=["100_120_140", "80_100_120", "100_140_160"],
    )
    parser.add_argument("--phase-patterns", nargs="+", choices=sorted(PHASE_PATTERNS.keys()), default=["0_pi2_pi", "0_pi3_2pi3"])
    parser.add_argument("--alpha-values", type=str, default="0.60,0.63,0.66")
    parser.add_argument("--q-configs-json", type=str, default="[[1,1,1],[2,1,1],[5,1,1],[1,0,0]]")
    parser.add_argument("--num-outputs", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--top-edges-per-output", type=int, default=3)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--max-primal-variables", type=int, default=200000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=50000)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    full_table = load_external_probability_table(args.probability_path, variable_name=args.variable_name)
    alpha_values_grid = _parse_float_list(args.alpha_values)
    q_configs = _parse_nested_float_lists(args.q_configs_json)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    case_id = 0
    for window_name in args.windows:
        window = WINDOW_LIBRARY[window_name]
        row_indices = list(window["row_indices"])
        intensities = [float(x) for x in window["intensities"]]
        selected_rows = np.asarray(full_table[row_indices, :], dtype=float)
        target_row = selected_rows[0].astype(float)
        target_row = target_row / target_row.sum()

        for num_outputs in args.num_outputs:
            edge_candidates = _top_contiguous_edges(
                target_row,
                num_outputs=int(num_outputs),
                top_k=int(args.top_edges_per_output),
            )
            equal_edges = np.linspace(0, len(target_row), int(num_outputs) + 1, dtype=int).tolist()
            edge_candidates = [{"edges": equal_edges, "distribution_only_H_min_target_row": _distribution_only_hmin(target_row, equal_edges), "target_row_probs": None}] + edge_candidates
            dedup_edges: list[dict[str, Any]] = []
            seen_edges: set[tuple[int, ...]] = set()
            for candidate in edge_candidates:
                key = tuple(candidate["edges"])
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                dedup_edges.append(candidate)

            for edge_candidate in dedup_edges:
                for phase_name in args.phase_patterns:
                    phases = PHASE_PATTERNS[phase_name]
                    for max_abs_alpha in alpha_values_grid:
                        alpha_values = intensities_to_alpha_values(
                            intensities,
                            max_abs_alpha=float(max_abs_alpha),
                            phases=phases,
                            max_intensity=max(intensities),
                        )
                        for q_values in q_configs:
                            case_id += 1
                            record: dict[str, Any] = {
                                "case_id": case_id,
                                "window": window_name,
                                "row_indices": row_indices,
                                "intensities": intensities,
                                "num_outputs": int(num_outputs),
                                "edges": list(edge_candidate["edges"]),
                                "edge_source_target_row_H_min": float(edge_candidate["distribution_only_H_min_target_row"]),
                                "phase_pattern": phase_name,
                                "phases": [float(x) for x in phases],
                                "max_abs_alpha": float(max_abs_alpha),
                                "cutoff": int(args.cutoff),
                                "q_values": [float(x) for x in q_values],
                                "alpha_values": [{"real": float(z.real), "imag": float(z.imag), "abs": float(abs(z)), "phase": float(math.atan2(z.imag, z.real))} for z in alpha_values],
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
                                    custom_edges=list(edge_candidate["edges"]),
                                    preferred_solver=args.solver,
                                    verbose=False,
                                    max_primal_variables=int(args.max_primal_variables),
                                    max_hermitian_scalar_count=int(args.max_hermitian_scalar_count),
                                )
                                record.update(
                                    {
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
                                        "phase_pattern": phase_name,
                                        "max_abs_alpha": float(max_abs_alpha),
                                        "q_values": [float(x) for x in q_values],
                                        "edges": list(edge_candidate["edges"]),
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
    payload = {
        "best": feasible[0] if feasible else None,
        "top20": feasible[:20],
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nFINAL")
    print(json.dumps(payload["best"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
