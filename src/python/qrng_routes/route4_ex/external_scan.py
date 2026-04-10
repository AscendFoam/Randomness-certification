"""Route4-ex 的定向外部概率表扫描脚本。

本脚本面向已经有外部概率表、并且希望围绕少量参数做快速扫描的场景。
它主要支持两类任务：

1. 固定 `q`，扫描 `max_abs_alpha`；
2. 固定 `max_abs_alpha`，扫描多组 `q`。

典型输入：

- `Probability.mat`
- 指定三输入窗口的 `row_indices`
- 一组实验强度 `intensities`
- 相位模式、输出数、截断维数与求解器设置

典型输出：

- 一个 JSON 文件，记录每个 case 的参数、distribution-only 指标、
  diagonal/full primal 结果以及当前 best case。

适用阶段：

- 小范围窗口可行性检查；
- 围绕已知较强窗口做参数精修；
- 比较不同 `q_selected` 或 `max_abs_alpha` 的影响。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .prototype import compare_route4_ex_external_diagonal_full, intensities_to_alpha_values


PHASE_PATTERNS: dict[str, list[float]] = {
    "0_pi2_pi": [0.0, math.pi / 2.0, math.pi],
    "0_pi3_2pi3": [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0],
    "0_pi4_pi2": [0.0, math.pi / 4.0, math.pi / 2.0],
    "0_pi4_3pi4": [0.0, math.pi / 4.0, 3.0 * math.pi / 4.0],
}


def _parse_float_list(text: str) -> list[float]:
    """把逗号分隔的浮点串解析为列表。"""
    values = [item.strip() for item in text.split(",") if item.strip()]
    return [float(item) for item in values]


def _parse_nested_float_lists(text: str) -> list[list[float]]:
    """解析多组 `q` 配置的 JSON 字符串。

    参数：
        text：形如 `[[1,1,1],[1,0,0]]` 的 JSON 字符串。

    返回：
        二维浮点列表，每一行是一组 `q_selected`。
    """
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("q-configs-json must decode to a list.")
    rows: list[list[float]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("Each q configuration must be a list of floats.")
        rows.append([float(x) for x in row])
    return rows


def _serialize_case(case: dict) -> dict:
    """把 case 记录转成纯 JSON 兼容对象。"""
    return json.loads(json.dumps(case, ensure_ascii=False))


def _run_case(
    *,
    probability_path: str,
    variable_name: str | None,
    row_indices: list[int],
    intensities: list[float],
    phases: list[float],
    max_abs_alpha: float,
    cutoff: int,
    num_outputs: int,
    q_values: list[float],
    prob_floor: float,
    solver: str | None,
    max_primal_variables: int,
    max_hermitian_scalar_count: int,
) -> dict:
    """执行单个 external scan case。

    功能：
        给定强度、相位、`max_abs_alpha` 和 `q_values`，先把强度映射成
        trusted coherent-state 振幅，再调用
        `compare_route4_ex_external_diagonal_full(...)` 获取认证结果。

    参数：
        probability_path：外部概率表路径。
        variable_name：`.mat/.npz` 中的变量名。
        row_indices：外部概率表中选取的输入行。
        intensities：实验强度列表。
        phases：每个输入的相位。
        max_abs_alpha：最大强度对应的振幅模长。
        cutoff：截断维数。
        num_outputs：coarse-graining 输出数。
        q_values：生成轮权重。
        prob_floor：概率正则化地板。
        solver：求解器名称。
        max_primal_variables：diagonal primal 规模保护阈值。
        max_hermitian_scalar_count：full primal 规模保护阈值。

    返回：
        一个轻量结果字典，包含 distribution-only、diagonal、full 三类指标。
    """
    alpha_values = intensities_to_alpha_values(
        intensities,
        max_abs_alpha=max_abs_alpha,
        phases=phases,
        max_intensity=max(intensities),
    )
    result = compare_route4_ex_external_diagonal_full(
        alpha_values=alpha_values,
        q_selected=q_values,
        cutoff=cutoff,
        probability_path=probability_path,
        num_outputs=num_outputs,
        row_indices=row_indices,
        prob_floor=prob_floor,
        variable_name=variable_name,
        already_coarse=False,
        preferred_solver=solver,
        verbose=False,
        max_primal_variables=max_primal_variables,
        max_hermitian_scalar_count=max_hermitian_scalar_count,
    )
    return {
        "distribution_only_H_min": result["instance"]["distribution_only_H_min"],
        "distribution_only_p_guess": result["instance"]["distribution_only_p_guess"],
        "diagonal_status": result["diagonal_primal"].get("status"),
        "diagonal_H_min": result["diagonal_primal"].get("H_min"),
        "diagonal_p_guess": result["diagonal_primal"].get("p_guess"),
        "full_status": result["full_primal"].get("status"),
        "full_H_min": result["full_primal"].get("H_min"),
        "full_p_guess": result["full_primal"].get("p_guess"),
        "p_guess_abs_gap": result.get("p_guess_abs_gap"),
        "H_min_abs_gap": result.get("H_min_abs_gap"),
        "alpha_values": result["instance"]["alpha_values"],
        "input_offdiagonal_metrics": result["instance"].get("input_offdiagonal_metrics"),
    }


def main() -> None:
    """解析命令行并执行 external scan。

    逻辑：
        - `alpha-grid` 模式：扫描多个 `max_abs_alpha`
        - `q-grid` 模式：固定 `max_abs_alpha`，扫描多组 `q`
        每跑完一个 case 就把当前结果写入输出 JSON，避免长时间扫描中途丢失进度。

    输出：
        输出文件包含：
        - 扫描模式与公共参数
        - `best`：当前可行结果里最优的一条
        - `cases`：所有 case 的完整记录
    """
    parser = argparse.ArgumentParser(description="Focused external Probability.mat scans for route4-ex.")
    parser.add_argument(
        "--scan-mode",
        choices=["alpha-grid", "q-grid"],
        required=True,
    )
    parser.add_argument("--probability-path", type=str, required=True)
    parser.add_argument("--variable-name", type=str, default="Probability")
    parser.add_argument("--row-indices", type=str, required=True, help="Comma-separated, e.g. 5,6,7")
    parser.add_argument("--intensities", type=str, required=True, help="Comma-separated, e.g. 100,120,140")
    parser.add_argument("--phase-pattern", choices=sorted(PHASE_PATTERNS.keys()), default="0_pi2_pi")
    parser.add_argument("--phases", type=str, default=None, help="Optional comma-separated explicit phase list.")
    parser.add_argument("--max-abs-alpha", type=float, default=None)
    parser.add_argument("--max-abs-alpha-values", type=str, default=None, help="Comma-separated grid.")
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--num-outputs", type=int, default=2)
    parser.add_argument("--q-values", type=str, default="1,1,1")
    parser.add_argument("--q-configs-json", type=str, default=None, help='JSON string, e.g. [[1,1,1],[1,2,1]]')
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--max-primal-variables", type=int, default=200000)
    parser.add_argument("--max-hermitian-scalar-count", type=int, default=50000)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    row_indices = [int(x) for x in _parse_float_list(args.row_indices)]
    intensities = _parse_float_list(args.intensities)
    phases = _parse_float_list(args.phases) if args.phases else PHASE_PATTERNS[args.phase_pattern]
    q_values = _parse_float_list(args.q_values)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []

    if args.scan_mode == "alpha-grid":
        if args.max_abs_alpha_values is None:
            raise ValueError("--max-abs-alpha-values is required for alpha-grid.")
        alpha_values = _parse_float_list(args.max_abs_alpha_values)
        for case_id, max_abs_alpha in enumerate(alpha_values, start=1):
            record = {
                "case_id": case_id,
                "scan_mode": "alpha-grid",
                "phase_pattern": args.phase_pattern,
                "phases": [float(x) for x in phases],
                "max_abs_alpha": float(max_abs_alpha),
                "cutoff": int(args.cutoff),
                "num_outputs": int(args.num_outputs),
                "q_values": [float(x) for x in q_values],
                "row_indices": row_indices,
                "intensities": intensities,
            }
            try:
                record.update(
                    _run_case(
                        probability_path=args.probability_path,
                        variable_name=args.variable_name,
                        row_indices=row_indices,
                        intensities=intensities,
                        phases=phases,
                        max_abs_alpha=float(max_abs_alpha),
                        cutoff=int(args.cutoff),
                        num_outputs=int(args.num_outputs),
                        q_values=q_values,
                        prob_floor=float(args.prob_floor),
                        solver=args.solver,
                        max_primal_variables=int(args.max_primal_variables),
                        max_hermitian_scalar_count=int(args.max_hermitian_scalar_count),
                    )
                )
            except Exception as exc:
                record["error"] = type(exc).__name__
                record["error_message"] = str(exc)
            cases.append(record)
            output_path.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False))
            print(json.dumps({k: record.get(k) for k in ["case_id", "max_abs_alpha", "full_status", "full_H_min", "error"]}, ensure_ascii=False), flush=True)
    else:
        if args.q_configs_json is None:
            raise ValueError("--q-configs-json is required for q-grid.")
        if args.max_abs_alpha is None:
            raise ValueError("--max-abs-alpha is required for q-grid.")
        q_configs = _parse_nested_float_lists(args.q_configs_json)
        for case_id, q_cfg in enumerate(q_configs, start=1):
            record = {
                "case_id": case_id,
                "scan_mode": "q-grid",
                "phase_pattern": args.phase_pattern,
                "phases": [float(x) for x in phases],
                "max_abs_alpha": float(args.max_abs_alpha),
                "cutoff": int(args.cutoff),
                "num_outputs": int(args.num_outputs),
                "q_values": [float(x) for x in q_cfg],
                "row_indices": row_indices,
                "intensities": intensities,
            }
            try:
                record.update(
                    _run_case(
                        probability_path=args.probability_path,
                        variable_name=args.variable_name,
                        row_indices=row_indices,
                        intensities=intensities,
                        phases=phases,
                        max_abs_alpha=float(args.max_abs_alpha),
                        cutoff=int(args.cutoff),
                        num_outputs=int(args.num_outputs),
                        q_values=q_cfg,
                        prob_floor=float(args.prob_floor),
                        solver=args.solver,
                        max_primal_variables=int(args.max_primal_variables),
                        max_hermitian_scalar_count=int(args.max_hermitian_scalar_count),
                    )
                )
            except Exception as exc:
                record["error"] = type(exc).__name__
                record["error_message"] = str(exc)
            cases.append(record)
            output_path.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False))
            print(json.dumps({k: record.get(k) for k in ["case_id", "q_values", "full_status", "full_H_min", "error"]}, ensure_ascii=False), flush=True)

    feasible = [row for row in cases if row.get("full_H_min") is not None]
    feasible.sort(key=lambda row: row["full_H_min"], reverse=True)
    payload = {
        "scan_mode": args.scan_mode,
        "probability_path": args.probability_path,
        "variable_name": args.variable_name,
        "row_indices": row_indices,
        "intensities": intensities,
        "phase_pattern": args.phase_pattern,
        "phases": [float(x) for x in phases],
        "cutoff": int(args.cutoff),
        "num_outputs": int(args.num_outputs),
        "best": _serialize_case(feasible[0]) if feasible else None,
        "cases": cases,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nFINAL")
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
