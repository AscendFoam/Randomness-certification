"""Route4-ex 病态边界定位脚本。

本脚本专门用于研究一个经验现象：

- 某些半径点在 `SCS` 或粗复核下看起来很强，
  但 `MOSEK` 再向高值方向推进极小一步就会失败。

因此它的目标不是大范围搜最优值，而是沿一条给定的半径线段做逐点扫描，
定位：

1. `MOSEK` 还能稳定给出 `optimal` 的区域；
2. 稳定区与失败区之间的转折位置；
3. 该转折附近的 distribution-only 熵、输入非对角强度和求解状态。

脚本已考虑两类健壮性问题：

- 即使 `skip-scs` 打开，也能正常落盘；
- 即使某个求解器失败，也会把错误信息写入输出 JSON，而不是直接中断。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .prototype import prepare_route4_ex_external_instance, run_route4_ex_external_full_primal


def _parse_float_triplet(text: str) -> np.ndarray:
    """解析长度为 3 的半径三元组字符串。"""
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("Expected exactly three comma-separated floats.")
    return np.asarray(values, dtype=float)


def _serialize_status(result: dict[str, Any] | None, error: Exception | None) -> dict[str, Any]:
    """把单个求解器结果或错误统一序列化成 JSON 友好的结构。

    功能：
        统一处理三种情况：
        - 求解成功；
        - 求解失败但捕获了异常；
        - 某个求解器被显式跳过。

    参数：
        result：成功时的求解结果字典。
        error：失败时捕获到的异常。

    返回：
        包含状态、`H_min`、`p_guess` 和错误信息的字典。
    """
    if result is None and error is None:
        return {
            "status": None,
            "solver": None,
            "H_min": None,
            "p_guess": None,
            "error": None,
            "error_message": None,
        }
    if error is not None:
        return {
            "status": None,
            "solver": None,
            "H_min": None,
            "p_guess": None,
            "error": type(error).__name__,
            "error_message": str(error),
        }
    return {
        "status": result.get("status"),
        "solver": result.get("solver"),
        "H_min": result.get("H_min"),
        "p_guess": result.get("p_guess"),
        "error": None,
        "error_message": None,
    }


def _distribution_only_hmin(instance: dict[str, Any]) -> float | None:
    """从实例字典中稳健提取 distribution-only `H_min`。"""
    if "distribution_only_H_min" in instance and instance["distribution_only_H_min"] is not None:
        return float(instance["distribution_only_H_min"])
    p_guess = instance.get("distribution_only_p_guess")
    if p_guess is None:
        return None
    p_guess = float(p_guess)
    if p_guess <= 0.0:
        return None
    return float(-math.log2(p_guess))


def main() -> None:
    """沿一条半径路径扫描 `MOSEK` 稳定/失稳阈值。

    逻辑：
        1. 解析起点与终点半径；
        2. 在线段上均匀取 `num_points` 个样本；
        3. 为每个样本构造 external instance；
        4. 选择性运行 SCS 与/或 MOSEK 的 full-primal；
        5. 每完成一个点就立即落盘，避免长时间运行中途丢进度。

    输出：
        输出 JSON 的 `rows` 中，每一行都记录：
        - 路径位置 `t`
        - 对应半径
        - 概率表与输入非对角指标
        - distribution-only 熵
        - SCS/MOSEK 的状态、结果或错误
    """
    parser = argparse.ArgumentParser(description="Scan a parameter line to diagnose the MOSEK instability threshold.")
    parser.add_argument("--probability-path", type=str, required=True)
    parser.add_argument("--variable-name", type=str, default="Probability")
    parser.add_argument("--row-indices", type=str, default="5,6,7")
    parser.add_argument("--edges", type=str, default="0,121,132,256")
    parser.add_argument("--q-config-json", type=str, default="[1,0,0]")
    parser.add_argument("--start-radii", type=str, required=True)
    parser.add_argument("--end-radii", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--prob-floor", type=float, default=1e-12)
    parser.add_argument("--skip-scs", action="store_true")
    parser.add_argument("--skip-mosek", action="store_true")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    row_indices = [int(part.strip()) for part in args.row_indices.split(",") if part.strip()]
    edges = [int(part.strip()) for part in args.edges.split(",") if part.strip()]
    q_values = json.loads(args.q_config_json)
    start_radii = _parse_float_triplet(args.start_radii)
    end_radii = _parse_float_triplet(args.end_radii)

    if args.num_points < 2:
        raise ValueError("num_points must be at least 2.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, t in enumerate(np.linspace(0.0, 1.0, args.num_points)):
        radii = (1.0 - t) * start_radii + t * end_radii
        alpha_values = [
            radii[0] * np.exp(1j * 0.0),
            radii[1] * np.exp(1j * (np.pi / 2.0)),
            radii[2] * np.exp(1j * np.pi),
        ]

        instance = prepare_route4_ex_external_instance(
            alpha_values=alpha_values,
            q_selected=q_values,
            cutoff=int(args.cutoff),
            probability_path=args.probability_path,
            num_outputs=len(edges) - 1,
            row_indices=row_indices,
            prob_floor=float(args.prob_floor),
            variable_name=args.variable_name,
            already_coarse=False,
            custom_edges=edges,
        )

        scs_result = None
        scs_error = None
        mosek_result = None
        mosek_error = None

        if not args.skip_scs:
            try:
                scs_result = run_route4_ex_external_full_primal(
                    alpha_values=alpha_values,
                    q_selected=q_values,
                    cutoff=int(args.cutoff),
                    probability_path=args.probability_path,
                    num_outputs=len(edges) - 1,
                    row_indices=row_indices,
                    prob_floor=float(args.prob_floor),
                    variable_name=args.variable_name,
                    already_coarse=False,
                    custom_edges=edges,
                    preferred_solver="SCS",
                    verbose=False,
                    max_hermitian_scalar_count=50000,
                )
            except Exception as exc:  # noqa: BLE001
                scs_error = exc

        if not args.skip_mosek:
            try:
                mosek_result = run_route4_ex_external_full_primal(
                    alpha_values=alpha_values,
                    q_selected=q_values,
                    cutoff=int(args.cutoff),
                    probability_path=args.probability_path,
                    num_outputs=len(edges) - 1,
                    row_indices=row_indices,
                    prob_floor=float(args.prob_floor),
                    variable_name=args.variable_name,
                    already_coarse=False,
                    custom_edges=edges,
                    preferred_solver="MOSEK",
                    verbose=False,
                    max_hermitian_scalar_count=50000,
                )
            except Exception as exc:  # noqa: BLE001
                mosek_error = exc

        row = {
            "index": int(idx),
            "t": float(t),
            "radii": [float(x) for x in radii],
            "probabilities": np.asarray(instance["probabilities"], dtype=float).tolist(),
            "input_offdiagonal_metrics": instance["input_offdiagonal_metrics"],
            "distribution_only_H_min": _distribution_only_hmin(instance),
            "scs": _serialize_status(scs_result, scs_error),
            "mosek": _serialize_status(mosek_result, mosek_error),
        }
        rows.append(row)
        output_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
        print(
            json.dumps(
                {
                    "index": row["index"],
                    "t": row["t"],
                    "radii": row["radii"],
                    "scs_status": row["scs"]["status"],
                    "scs_H_min": row["scs"]["H_min"],
                    "mosek_status": row["mosek"]["status"],
                    "mosek_H_min": row["mosek"]["H_min"],
                    "mosek_error": row["mosek"]["error"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    output_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
