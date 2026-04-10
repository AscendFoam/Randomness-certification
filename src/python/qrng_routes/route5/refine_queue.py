"""
Route 5 本地精炼队列 (Refine Queue)
====================================

物理背景
--------
本模块是 Route 5 QRNG 方案的批量精炼命令行入口。
它对一组预定义的半径候选组合（三元组 {0, r1, r2}）进行系统搜索，
包括快速侦察(scout)和精细认证(certification)两个阶段。

Route 5 方案回顾：
- 使用相干态字母表 {|α_k⟩} 作为可信输入态
- 通过平衡分束器 + IQ 双 Homodyne 测量获得 X-P 相平面上的离散输出
- 用 SDP 计算最优猜测概率，从而认证最小熵 H_min

本模块的搜索策略：两阶段分离
--------------------------
阶段1 — 侦察 (Scout)：
  对每个半径候选 {0, r1, r2}，使用 search_route5_iq_partitions 做快速搜索：
  - 遍历多种粗粒化分箱配置（num_x_bins × num_p_bins × quadrature_range × gamma）
  - 仅计算原始熵估计（不做 SDP），极快
  - 记录每种配置的 raw_best_H_min

阶段2 — 认证 (Certification)：
  按侦察阶段的 raw_best_H_min 降序排列候选
  对排名靠前的候选，使用 run_route5 做完整的 SDP 认证：
  - 使用侦察阶段发现的最优分箱边界
  - 调用 MOSEK/SCS 求解器计算精确的 H_min
  - 每个候选的完整结果保存到独立文件

与 intensity_menu_search.py 的区别：
- intensity_menu_search: 从实验光强菜单出发，扫描 max_radius
- refine_queue: 从预定义的 (r1, r2) 网格出发，逐个精炼

断点续传：
  聚合结果在每完成一个候选后立即写入磁盘。
  程序重启时自动跳过已完成的候选。

命令行接口：
    python -m qrng_routes.route5.refine_queue \
        --r1-values 0.3 0.4 0.5 0.6 \
        --r2-values 0.8 0.9 1.0 1.1 1.2 \
        --quadrature-ranges 2.0 3.0 4.0 \
        --gamma-values 0.75 1.0 1.5 \
        --output-json results/route5_refine.json \
        --result-dir results/route5_refine/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_iq import DEFAULT_PHASE_VALUES, run_route5, search_route5_iq_partitions


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


def _write_json(path: Path, payload: Any) -> None:
    """
    将结果写入JSON文件（自动创建父目录）

    参数
    ----
    path : Path
        输出文件路径

    payload : Any
        待序列化的数据
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def _candidate_key(radii: list[float]) -> str:
    """
    为半径候选生成唯一标识字符串

    格式为 "r{r1:.4f}_{r2:.4f}"（真空态 r=0 不在key中，因为所有候选都包含它）。

    参数
    ----
    radii : list[float]
        半径列表，通常为 [0.0, r1, r2]

    返回
    ----
    key : str
        唯一标识字符串，用于去重和索引
    """
    return "r" + "_".join(f"{value:.4f}" for value in radii)


def _unique_sorted(values: list[float], tol: float = 1e-12) -> list[float]:
    """
    去重并排序浮点数列表

    参数
    ----
    values : list[float]
        输入值列表

    tol : float
        容差，差值 ≤ tol 视为重复

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


def _build_radius_candidates(r1_values: list[float], r2_values: list[float]) -> list[list[float]]:
    """
    从两个半径值列表构造所有合法的三元组候选
    ==========================================

    物理原理
    --------
    Route 5 的字母表通常包含真空态 (r=0) 和两个不同半径的相干态。
    因此每个候选是三元组 {0, r1, r2}，其中要求 r1 < r2。

    为什么要 r1 < r2？
    - 两个不同半径的相干态在相空间中形成不同大小的圆
    - 较小的圆靠近原点（接近真空），较大的圆远离原点
    - 这种差异使 SDP 能够区分不同输入，产生可认证的随机性

    参数
    ----
    r1_values : list[float]
        第一个非零半径的候选值（较小半径）

    r2_values : list[float]
        第二个非零半径的候选值（较大半径）

    返回
    ----
    candidates : list[list[float]]
        所有合法的三元组 [0.0, r1, r2]，其中 0 < r1 < r2
        去重并过滤掉无效组合

    示例
    ----
    >>> _build_radius_candidates([0.3, 0.5], [0.8, 1.0, 0.4])
    # 返回 [[0.0, 0.3, 0.8], [0.0, 0.3, 1.0], [0.0, 0.5, 0.8], [0.0, 0.5, 1.0]]
    # 注意 (0.5, 0.4) 被过滤因为 0.4 < 0.5
    """
    first = [value for value in _unique_sorted(r1_values) if value > 0.0]
    second = [value for value in _unique_sorted(r2_values) if value > 0.0]
    out: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for r1 in first:
        for r2 in second:
            if r2 <= r1 + 1e-12:
                continue
            candidate = (0.0, float(r1), float(r2))
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(list(candidate))
    return out


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


def _scout_summary(result: dict, radii: list[float]) -> dict:
    """
    提取侦察阶段的摘要信息

    参数
    ----
    result : dict
        search_route5_iq_partitions 的返回值

    radii : list[float]
        当前的半径候选 [0, r1, r2]

    返回
    ----
    summary : dict
        包含候选标识、原始熵估计、最优分箱配置等关键信息
    """
    return {
        "candidate_key": _candidate_key(radii),
        "radii": list(radii),
        "raw_best_H_min": result["raw_best_H_min"],
        "raw_best_target": result["raw_best_target"],
        "raw_best_target_index": result["raw_best_target_index"],
        "num_x_bins": result["num_x_bins"],
        "num_p_bins": result["num_p_bins"],
        "x_bounds": result["x_bounds"],
        "p_bounds": result["p_bounds"],
        "x_range": result["x_range"],
        "p_range": result["p_range"],
        "x_gamma": result["x_gamma"],
        "p_gamma": result["p_gamma"],
        "num_outputs": result["num_outputs"],
        "local_operator_span_rank": result["local_operator_span_rank"],
        "local_operator_space_dim": result["local_operator_space_dim"],
        "num_partition_candidates": result["num_partition_candidates"],
        "selection_strategy": result["selection_strategy"],
    }


def _certified_summary(result: dict, radii: list[float], scout_rank: int, output_json: str) -> dict:
    """
    提取认证阶段的摘要信息

    参数
    ----
    result : dict
        run_route5 的返回值

    radii : list[float]
        当前的半径候选

    scout_rank : int
        在侦察排名中的位次（1-based）

    output_json : str
        完整结果的保存路径

    返回
    ----
    summary : dict
        包含 SDP 认证的 H_min、求解器状态、分箱配置等信息
    """
    return {
        "candidate_key": _candidate_key(radii),
        "radii": list(radii),
        "scout_rank": int(scout_rank),
        "output_json": output_json,
        "H_min": result["H_min"],
        "raw_best_H_min": result["raw_best_H_min"],
        "target_input": result["target_input"],
        "raw_best_target": result["raw_best_target"],
        "num_x_bins": result["num_x_bins"],
        "num_p_bins": result["num_p_bins"],
        "x_bounds": result["x_bounds"],
        "p_bounds": result["p_bounds"],
        "quadrature_range": result["quadrature_range"],
        "boundary_gamma": result["boundary_gamma"],
        "num_quadrature_nodes": result["num_quadrature_nodes"],
        "num_inputs_certified": result["num_inputs_certified"],
        "solver": result["solver"],
        "status": result["status"],
    }


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    参数分组
    --------
    核心参数：
    - --cutoff: Fock截断维度，默认4
    - --phase-values: 相位候选池，默认4相位 [0, π/2, π, 3π/2]

    半径候选参数：
    - --r1-values: 较小半径的候选值列表（必需）
    - --r2-values: 较大半径的候选值列表（必需）
      所有 (r1, r2) 满足 r1 < r2 的组合构成候选池

    分箱配置参数：
    - --num-x-bins: X方向分箱数，默认6
    - --num-p-bins: P方向分箱数，默认2
    - --quadrature-ranges: 测量范围候选（必需）
    - --gamma-values: 幂律分箱间距参数（必需）

    精度控制参数：
    - --scout-num-quadrature-nodes: 侦察阶段的积分节点数，默认20
    - --cert-num-quadrature-nodes: 认证阶段的积分节点数，默认20

    搜索控制参数：
    - --max-inputs: 每次SDP最多认证的输入态数，默认3
    - --candidate-limit: 最多认证的候选数（0=全部），默认0

    求解器参数：
    - --solver: 优先求解器，默认 SCS（侦察阶段建议用 SCS，认证阶段可切 MOSEK）
    - --scs-*: SCS 求解器参数
    - --mosek-*: MOSEK 求解器参数

    输出参数：
    - --output-json: 聚合结果JSON路径（必需）
    - --result-dir: 完整结果的保存目录（必需）
    """
    parser = argparse.ArgumentParser(description="Long-running local refinement queue for route5.")
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--phase-values", nargs="+", type=float, default=DEFAULT_PHASE_VALUES)
    parser.add_argument("--r1-values", nargs="+", type=float, required=True)
    parser.add_argument("--r2-values", nargs="+", type=float, required=True)
    parser.add_argument("--num-x-bins", type=int, default=6)
    parser.add_argument("--num-p-bins", type=int, default=2)
    parser.add_argument("--quadrature-ranges", nargs="+", type=float, required=True)
    parser.add_argument("--gamma-values", nargs="+", type=float, required=True)
    parser.add_argument("--scout-num-quadrature-nodes", type=int, default=20)
    parser.add_argument("--cert-num-quadrature-nodes", type=int, default=20)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--scs-max-iters", type=int, default=None)
    parser.add_argument("--scs-eps-abs", type=float, default=None)
    parser.add_argument("--scs-eps-rel", type=float, default=None)
    parser.add_argument("--scs-eps-infeas", type=float, default=None)
    parser.add_argument("--mosek-num-threads", type=int, default=None)
    parser.add_argument("--mosek-eps", type=float, default=None)
    parser.add_argument("--mosek-accept-unknown", action="store_true")
    parser.add_argument("--mosek-save-file", type=str, default=None)
    parser.add_argument("--mosek-solve-form", choices=["dual", "primal", "free"], default=None)
    parser.add_argument("--max-inputs", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """
    主入口函数：执行两阶段搜索和认证
    ================================

    执行流程
    --------
    阶段1 — 侦察 (Scout)：
    1. 生成所有合法的半径候选三元组 {0, r1, r2}
    2. 对每个候选调用 search_route5_iq_partitions 做快速搜索
       - 遍历 quadrature_ranges × gamma_values 的组合
       - 仅计算原始熵（不做 SDP），速度极快
    3. 按 raw_best_H_min 降序排列，记录侦察结果
    4. 每完成一个候选就保存到磁盘（断点续传）

    阶段2 — 认证 (Certification)：
    1. 按侦察排名遍历候选（可选截断到 candidate_limit）
    2. 对每个候选调用 run_route5 做完整 SDP 认证
       - 使用侦察阶段发现的最优分箱边界 (x_bounds, p_bounds)
       - quadrature_range 取 x_range 和 p_range 的较大值（确保覆盖）
    3. 每个候选的完整结果保存到 {result_dir}/{candidate_key}.json
    4. 摘要信息写入聚合文件

    断点续传机制
    -----------
    - 如果 output-json 文件已存在，加载已有结果
    - 通过 candidate_key 匹配，跳过已完成的侦察和认证
    - 这允许安全地中断和重启长时间运行的任务

    最终输出
    --------
    - aggregate["scout_results"]: 所有侦察结果（按 raw_best_H_min 降序）
    - aggregate["certified_results"]: 所有认证结果（按 H_min 降序）
    - aggregate["best_certified_result"]: 全局最优的认证结果
    """
    args = _parse_args()
    aggregate_path = Path(args.output_json)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    solver_options = _build_solver_options(args)

    aggregate: dict[str, Any]
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
    else:
        aggregate = {
            "route": "route5_local_refine_queue",
            "cutoff": int(args.cutoff),
            "phase_values": list(args.phase_values),
            "num_x_bins": int(args.num_x_bins),
            "num_p_bins": int(args.num_p_bins),
            "quadrature_ranges": list(args.quadrature_ranges),
            "gamma_values": list(args.gamma_values),
            "scout_num_quadrature_nodes": int(args.scout_num_quadrature_nodes),
            "cert_num_quadrature_nodes": int(args.cert_num_quadrature_nodes),
            "max_inputs": int(args.max_inputs),
            "candidate_limit": None if args.candidate_limit <= 0 else int(args.candidate_limit),
            "scout_results": [],
            "certified_results": [],
        }

    scout_by_key = {
        entry["candidate_key"]: dict(entry)
        for entry in aggregate.get("scout_results", [])
    }
    certified_by_key = {
        entry["candidate_key"]: dict(entry)
        for entry in aggregate.get("certified_results", [])
    }

    radius_candidates = _build_radius_candidates(args.r1_values, args.r2_values)
    if len(radius_candidates) == 0:
        raise RuntimeError("No valid radius candidates were generated.")

    for radii in radius_candidates:
        candidate_key = _candidate_key(radii)
        if candidate_key in scout_by_key:
            continue
        scout = search_route5_iq_partitions(
            cutoff=args.cutoff,
            radius_values=list(radii),
            phase_values=list(args.phase_values),
            num_x_bins_values=[args.num_x_bins],
            num_p_bins_values=[args.num_p_bins],
            quadrature_ranges=list(args.quadrature_ranges),
            gamma_values=list(args.gamma_values),
            num_quadrature_nodes=args.scout_num_quadrature_nodes,
            certify_top_k=0,
            max_inputs_to_certify=args.max_inputs,
        )
        scout_by_key[candidate_key] = _scout_summary(scout, radii)
        aggregate["scout_results"] = sorted(
            scout_by_key.values(),
            key=lambda item: item["raw_best_H_min"],
            reverse=True,
        )
        _write_json(aggregate_path, aggregate)

    ranked_scouts = sorted(
        scout_by_key.values(),
        key=lambda item: item["raw_best_H_min"],
        reverse=True,
    )
    if args.candidate_limit > 0:
        ranked_scouts = ranked_scouts[: args.candidate_limit]

    for scout_rank, scout in enumerate(ranked_scouts, start=1):
        candidate_key = scout["candidate_key"]
        if candidate_key in certified_by_key:
            continue
        radii = scout["radii"]
        quadrature_range = max(float(scout["x_range"]), float(scout["p_range"]), 1.0)
        result = run_route5(
            cutoff=args.cutoff,
            radius_values=list(radii),
            phase_values=list(args.phase_values),
            num_x_bins=int(scout["num_x_bins"]),
            num_p_bins=int(scout["num_p_bins"]),
            x_bounds=np.array(scout["x_bounds"], dtype=float),
            p_bounds=np.array(scout["p_bounds"], dtype=float),
            quadrature_range=quadrature_range,
            boundary_gamma=float(scout["x_gamma"]),
            num_quadrature_nodes=args.cert_num_quadrature_nodes,
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            solver_options=solver_options,
        )
        output_json = result_dir / f"{candidate_key}.json"
        _write_json(output_json, result)
        certified_by_key[candidate_key] = _certified_summary(
            result,
            radii=list(radii),
            scout_rank=scout_rank,
            output_json=str(output_json),
        )
        aggregate["certified_results"] = sorted(
            certified_by_key.values(),
            key=lambda item: item["H_min"] if item["H_min"] is not None else -np.inf,
            reverse=True,
        )
        aggregate["best_certified_result"] = (
            None if len(aggregate["certified_results"]) == 0 else aggregate["certified_results"][0]
        )
        _write_json(aggregate_path, aggregate)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
