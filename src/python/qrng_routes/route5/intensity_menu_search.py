"""
Route 5 固定光强菜单搜索 (Intensity Menu Search)
=================================================

物理背景
--------
本模块是 Route 5 (Hybrid IQ 测量) QRNG 方案的一个命令行入口，
用于在**固定的实验光强菜单**下搜索最优的字母表和粗粒化配置。

Route 5 的核心方案：
1. Alice 和 Bob 各自从一组可信相干态 {|α_k⟩} 中选择并准备一个态
2. 两束光通过平衡分束器干涉后，Charlie 做 IQ 双 Homodyne 测量
3. 测量输出在 X-P 相平面上做粗粒化分箱
4. 通过 SDP 计算可认证的最小熵 H_min

本模块的特殊之处：
- 实验上可用的光强是离散的（"菜单"），如 {0, 20, 40, ..., 160} 光子数
- 需要将实验光强映射到 Route 5 的半径参数：radius = max_radius × √(I / I_max)
- 对多个 max_radius 值进行扫描，找到最优的缩放参数

搜索流程：
1. 将实验光强菜单转换为半径参数池（保留光强间的比例关系）
2. 对每个 max_radius 值，调用 search_route5_alphabets 搜索最优字母表
3. 字母表搜索包括：半径子集选择 × 相位子集选择 × IQ 分箱配置
4. 返回所有 max_radius 下的最优结果

命令行接口：
    python -m qrng_routes.route5.intensity_menu_search \
        --intensity-values 0 20 40 60 80 100 120 140 160 \
        --max-radius-values 0.6 0.8 1.0 1.2 1.5 \
        --output-json results/route5_menu_search.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_iq import DEFAULT_GAMMA_VALUES, search_route5_alphabets


# ── 默认参数 ──────────────────────────────────────────────────────────────
# DEFAULT_8PHASE_VALUES: 8个均匀分布的相位值（间隔 π/4）
# 比标准4相位更精细，允许搜索更多相位组合
DEFAULT_8PHASE_VALUES = [
    0.0,
    0.25 * math.pi,
    0.5 * math.pi,
    0.75 * math.pi,
    math.pi,
    1.25 * math.pi,
    1.5 * math.pi,
    1.75 * math.pi,
]

# DEFAULT_QUADRATURE_RANGES: 默认的 IQ 测量范围（标准差倍数）
# 从 1.8 到 2.0 的细粒度扫描，寻找最优的测量窗口大小
DEFAULT_QUADRATURE_RANGES = [1.8, 1.85, 1.9, 1.95, 2.0]


def _clean_value(value: Any) -> Any:
    """
    JSON序列化辅助：将NumPy类型转换为Python原生类型

    参数
    ----
    value : Any
        待转换的值

    返回
    ----
    converted : Any
        np.ndarray → list, np.floating/np.integer → float/int, 其他原样返回
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str | Path | None, payload: Any) -> None:
    """
    将结果写入JSON文件（自动创建父目录）

    参数
    ----
    path : str | Path | None
        输出文件路径。若为 None 则不写入（静默跳过）。

    payload : Any
        待序列化的数据（通常是大字典）
    """
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _unique_sorted(values: list[float], tol: float = 1e-12) -> list[float]:
    """
    去重并排序浮点数列表

    参数
    ----
    values : list[float]
        输入值列表（可能有重复或无序）

    tol : float
        容差，差值 ≤ tol 视为重复，默认 1e-12

    返回
    ----
    unique : list[float]
        去重后按升序排列的列表
    """
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if any(abs(value - existing) <= tol for existing in unique):
            continue
        unique.append(value)
    return unique


def _build_solver_options(args: argparse.Namespace) -> dict[str, dict] | None:
    """
    从命令行参数构建CVXPY求解器选项字典

    支持的求解器配置：
    - SCS: max_iters, eps_abs, eps_rel, eps_infeas
    - MOSEK: num_threads, solve_form, eps, accept_unknown, save_file

    参数
    ----
    args : argparse.Namespace
        解析后的命令行参数

    返回
    ----
    solver_options : dict[str, dict] | None
        求解器名称到选项字典的映射，无选项时返回 None
    """
    scs_options = {
        key: value
        for key, value in {
            "max_iters": args.scs_max_iters,
            "eps_abs": args.scs_eps_abs,
            "eps_rel": args.scs_eps_rel,
            "eps_infeas": args.scs_eps_infeas,
        }.items()
        if value is not None
    }

    mosek_params: dict[str, Any] = {}
    if args.mosek_num_threads is not None:
        mosek_params["MSK_IPAR_NUM_THREADS"] = int(args.mosek_num_threads)
    if args.mosek_solve_form is not None:
        solve_forms = {
            "dual": "MSK_SOLVE_DUAL",
            "primal": "MSK_SOLVE_PRIMAL",
            "free": "MSK_SOLVE_FREE",
        }
        mosek_params["MSK_IPAR_INTPNT_SOLVE_FORM"] = solve_forms[args.mosek_solve_form]

    mosek_options = {
        key: value
        for key, value in {
            "mosek_params": mosek_params if len(mosek_params) > 0 else None,
            "eps": args.mosek_eps,
            "accept_unknown": True if args.mosek_accept_unknown else None,
            "save_file": args.mosek_save_file,
        }.items()
        if value is not None
    }

    solver_options: dict[str, dict] = {}
    if len(scs_options) > 0:
        solver_options["SCS"] = scs_options
    if len(mosek_options) > 0:
        solver_options["MOSEK"] = mosek_options
    return None if len(solver_options) == 0 else solver_options


def intensity_menu_to_radii(
    intensity_values: list[float],
    max_radius: float,
    require_vacuum: bool = True,
) -> tuple[list[float], list[dict[str, float]]]:
    """
    将实验光强菜单映射为 Route 5 的半径参数
    =========================================

    物理原理
    --------
    实验中可用的光强是离散的（如调制器的固定档位），但 Route 5 的
    相干态参数 α = radius × e^(iφ) 使用的是相空间中的半径。

    相干态的平均光子数 μ = |α|² = radius²，因此：
        radius = √μ = √(intensity)

    映射策略：保持光强间的相对比例
    ------------------------------
    为了让搜索空间覆盖不同的总强度范围，引入 max_radius 参数：
        radius = max_radius × √(intensity / max_intensity)

    这样：
    - 最大光强对应 radius = max_radius
    - 零光强对应 radius = 0（真空态）
    - 其他光强按 √比例缩放，保持相空间中的相对位置

    参数
    ----
    intensity_values : list[float]
        实验可用的光强列表（平均光子数，≥ 0）

    max_radius : float
        最大光强对应的半径值。控制整体缩放。
        - max_radius 较小：所有态靠近真空，重叠大，区分度低
        - max_radius 较大：态在相空间中分散，但数值截断误差可能增大

    require_vacuum : bool
        是否强制包含真空态（radius=0），默认 True。
        真空态是重要的参考态，与任何非真空态的组合都能产生有意义的统计差异。

    返回
    ----
    radii : list[float]
        去重排序后的半径列表

    mapping : list[dict[str, float]]
        每个光强到半径的映射记录，格式为 {"intensity": ..., "radius": ...}

    示例
    ----
    >>> intensity_menu_to_radii([0, 40, 80, 160], max_radius=1.2)
    # radii = [0.0, 0.6, 0.8485..., 1.2]
    # (因为 √(40/160) = 0.5, √(80/160) ≈ 0.707, √(160/160) = 1.0)
    """
    unique_intensities = _unique_sorted(intensity_values)
    nonnegative = [value for value in unique_intensities if value >= 0.0]
    if len(nonnegative) == 0:
        raise ValueError("At least one non-negative intensity is required.")

    positive = [value for value in nonnegative if value > 0.0]
    if len(positive) == 0 and not require_vacuum:
        raise ValueError("At least one positive intensity is required when vacuum is not forced.")

    max_intensity = max(positive) if len(positive) > 0 else 1.0
    mapping: list[dict[str, float]] = []
    radii: list[float] = []

    if require_vacuum:
        radii.append(0.0)
        mapping.append({"intensity": 0.0, "radius": 0.0})

    for intensity in positive:
        radius = float(max_radius) * math.sqrt(float(intensity) / max_intensity)
        radii.append(radius)
        mapping.append({"intensity": float(intensity), "radius": float(radius)})

    return _unique_sorted(radii), mapping


def _scale_summary(result: dict[str, Any], max_radius: float, radius_mapping: list[dict[str, float]]) -> dict[str, Any]:
    """
    提取单个 max_radius 下搜索结果的摘要信息

    参数
    ----
    result : dict[str, Any]
        search_route5_alphabets 的完整返回值

    max_radius : float
        当前使用的最大半径值

    radius_mapping : list[dict[str, float]]
        光强到半径的映射表

    返回
    ----
    summary : dict[str, Any]
        包含缩放参数、最优字母表、分箱配置和认证结果的摘要
    """
    partition_result = result.get("best_partition_search_result")
    return {
        "max_radius": float(max_radius),
        "radius_mapping": radius_mapping,
        "radius_pool": result["radius_pool"],
        "num_alphabet_candidates": result["num_alphabet_candidates"],
        "selected_alphabet_candidate_index": result["selected_alphabet_candidate_index"],
        "selected_partition_candidate_index": result["selected_partition_candidate_index"],
        "best_certified_H_min": result["best_certified_H_min"],
        "selected_radius_values": result["radius_values"],
        "selected_phase_values": result["phase_values"],
        "selected_num_local_states": result["num_local_states"],
        "selected_alpha_values": result["alpha_values"],
        "partition_summary": None
        if partition_result is None
        else {
            "num_outputs": partition_result["num_outputs"],
            "num_x_bins": partition_result["num_x_bins"],
            "num_p_bins": partition_result["num_p_bins"],
            "quadrature_range": partition_result["quadrature_range"],
            "boundary_gamma": partition_result["boundary_gamma"],
            "H_min": partition_result["H_min"],
            "raw_best_H_min": partition_result["raw_best_H_min"],
            "target_input": partition_result["target_input"],
            "certified_best_target_alphas": partition_result["certified_best_target_alphas"],
            "solver": partition_result["solver"],
            "status": partition_result["status"],
        },
        "result": result,
    }


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    参数分组
    --------
    核心参数：
    - --intensity-values: 实验可用的光强列表（必需）
    - --max-radius-values: 要测试的最大半径值列表（必需）
    - --cutoff: Fock截断维度，默认4（Route 5 通常用较小的 cutoff）

    字母表搜索参数：
    - --phase-values: 相位候选池，默认8个均匀相位
    - --num-radii-values: 要尝试的半径子集大小，默认 [3]
    - --num-phase-values: 要尝试的相位子集大小，默认 [8]

    分箱配置参数：
    - --num-x-bins-values: X方向的分箱数候选，默认 [6]
    - --num-p-bins-values: P方向的分箱数候选，默认 [2]
    - --quadrature-ranges: 测量范围候选，默认 [1.8, 1.85, ..., 2.0]
    - --gamma-values: 幂律分箱间距参数，来自 hybrid_iq 模块

    搜索控制参数：
    - --alphabet-top-k: 字母表搜索中保留的前k个候选，默认4
    - --certify-top-k: SDP认证的前k个分箱配置，默认3
    - --max-inputs: 每次SDP最多认证的输入态数，默认3

    求解器参数：
    - --solver: 优先求解器，默认 MOSEK
    - --scs-*: SCS 求解器参数
    - --mosek-*: MOSEK 求解器参数

    输出参数：
    - --output-json: 输出JSON文件路径（必需）
    - --no-require-vacuum: 不强制包含真空态
    """
    parser = argparse.ArgumentParser(
        description="Route5 search constrained to a fixed experimental intensity menu."
    )
    parser.add_argument("--intensity-values", nargs="+", type=float, required=True)
    parser.add_argument("--max-radius-values", nargs="+", type=float, required=True)
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--phase-values", nargs="+", type=float, default=DEFAULT_8PHASE_VALUES)
    parser.add_argument("--num-radii-values", nargs="+", type=int, default=[3])
    parser.add_argument("--num-phase-values", nargs="+", type=int, default=[8])
    parser.add_argument("--num-x-bins-values", nargs="+", type=int, default=[6])
    parser.add_argument("--num-p-bins-values", nargs="+", type=int, default=[2])
    parser.add_argument("--quadrature-ranges", nargs="+", type=float, default=DEFAULT_QUADRATURE_RANGES)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--num-quadrature-nodes", type=int, default=12)
    parser.add_argument("--alphabet-top-k", type=int, default=4)
    parser.add_argument("--certify-top-k", type=int, default=3)
    parser.add_argument("--max-inputs", type=int, default=3)
    parser.add_argument("--solver", type=str, default="MOSEK")
    parser.add_argument("--scs-max-iters", type=int, default=None)
    parser.add_argument("--scs-eps-abs", type=float, default=None)
    parser.add_argument("--scs-eps-rel", type=float, default=None)
    parser.add_argument("--scs-eps-infeas", type=float, default=None)
    parser.add_argument("--mosek-num-threads", type=int, default=None)
    parser.add_argument("--mosek-eps", type=float, default=None)
    parser.add_argument("--mosek-accept-unknown", action="store_true")
    parser.add_argument("--mosek-save-file", type=str, default=None)
    parser.add_argument("--mosek-solve-form", choices=["dual", "primal", "free"], default=None)
    parser.add_argument("--no-require-vacuum", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """
    主入口函数：执行固定光强菜单下的Route 5搜索
    ============================================

    执行流程
    --------
    1. 解析命令行参数，构建求解器选项
    2. 对每个 max_radius 值：
       a. 调用 intensity_menu_to_radii 将光强菜单转换为半径池
       b. 调用 search_route5_alphabets 在该半径池上搜索最优配置
       c. 记录并保存中间结果（支持断点续传）
    3. 从所有 max_radius 结果中选出全局最优
    4. 保存并打印最终聚合结果

    断点续传支持
    -----------
    每完成一个 max_radius 的搜索就立即写入 JSON 文件。
    如果程序中途中断，已保存的结果不会丢失。
    """
    args = _parse_args()
    solver_options = _build_solver_options(args)
    require_vacuum = not args.no_require_vacuum

    aggregate: dict[str, Any] = {
        "route": "route5_fixed_intensity_menu_search",
        "intensity_values": _unique_sorted(list(args.intensity_values)),
        "max_radius_values": _unique_sorted(list(args.max_radius_values)),
        "scaling_rule": "radius = max_radius * sqrt(intensity / max_intensity_in_menu)",
        "cutoff": int(args.cutoff),
        "phase_values": [float(value) for value in args.phase_values],
        "num_radii_values": [int(value) for value in args.num_radii_values],
        "num_phase_values": [int(value) for value in args.num_phase_values],
        "num_x_bins_values": [int(value) for value in args.num_x_bins_values],
        "num_p_bins_values": [int(value) for value in args.num_p_bins_values],
        "quadrature_ranges": [float(value) for value in args.quadrature_ranges],
        "gamma_values": [float(value) for value in args.gamma_values],
        "num_quadrature_nodes": int(args.num_quadrature_nodes),
        "alphabet_top_k": int(args.alphabet_top_k),
        "certify_top_k": int(args.certify_top_k),
        "max_inputs": int(args.max_inputs),
        "require_vacuum": bool(require_vacuum),
        "scale_results": [],
    }

    for max_radius in aggregate["max_radius_values"]:
        radius_pool, mapping = intensity_menu_to_radii(
            intensity_values=aggregate["intensity_values"],
            max_radius=max_radius,
            require_vacuum=require_vacuum,
        )
        result = search_route5_alphabets(
            cutoff=args.cutoff,
            radius_values=radius_pool,
            phase_values=list(args.phase_values),
            num_radii_values=list(args.num_radii_values),
            num_phase_values=list(args.num_phase_values),
            require_vacuum=require_vacuum,
            num_x_bins_values=list(args.num_x_bins_values),
            num_p_bins_values=list(args.num_p_bins_values),
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            num_quadrature_nodes=args.num_quadrature_nodes,
            alphabet_top_k=args.alphabet_top_k,
            certify_top_k=args.certify_top_k,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
        )
        aggregate["scale_results"].append(_scale_summary(result, max_radius=max_radius, radius_mapping=mapping))
        _write_json(args.output_json, aggregate)

    if len(aggregate["scale_results"]) > 0:
        aggregate["best_overall"] = max(
            aggregate["scale_results"],
            key=lambda item: item["best_certified_H_min"]
            if item["best_certified_H_min"] is not None
            else -np.inf,
        )
    else:
        aggregate["best_overall"] = None

    _write_json(args.output_json, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
