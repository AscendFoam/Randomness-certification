"""
MDI-QRNG SDP求解器入口。

用法:
    python -m mdi_qrng.main                    # 默认参数
    python -m mdi_qrng.main --mu 1.0 --n 4     # 指定参数
    python -m mdi_qrng.main --sweep             # mu参数扫描
"""

import argparse
import time
import numpy as np

from .states import build_density_matrices
from .probability import compute_probabilities, validate_probabilities
from .sdp_solver import solve_mdi_qrng_sdp


def compute_hmin_for_mu(
    mu: float,
    n: int,
    boundary: float | None = None,
    x_star: int = 0,
    y_star: int = 0,
    solver: str | None = None,
    verbose: bool = True,
) -> dict:
    """对给定的mu和离散化参数n，计算最小熵H_min。

    Args:
        mu: 平均光子数。
        n: 每个正交分量的bin数量（n_a = n_b = n）。
        boundary: 离散化边界绝对值（None则自适应计算）。
        x_star: 用于认证的Alice输入（0或1）。
        y_star: 用于认证的Bob输入（0或1）。
        solver: 指定求解器（None则自动选择）。
        verbose: 是否打印详细输出。

    Returns:
        包含 p_guess, H_min, status 等信息的字典。
    """
    bnd_str = f"{boundary}" if boundary is not None else "auto"
    print(f"\n{'='*60}")
    print(f"mu = {mu:.4f}, n = {n}, boundary = {bnd_str}")
    print(f"认证输入: x* = {x_star}, y* = {y_star}")
    print(f"{'='*60}")

    # 1. 构建密度矩阵
    rho = build_density_matrices(mu)

    # 2. 计算条件概率（自适应boundary + 概率正则化）
    prob = compute_probabilities(mu, n, n, boundary=boundary)
    validate_probabilities(prob)

    # 打印概率摘要
    for x in range(2):
        for y in range(2):
            p_sum = prob[x, y].sum()
            p_max = prob[x, y].max()
            p_min = prob[x, y].min()
            print(f"  P(·|x={x},y={y}): sum={p_sum:.10f}, "
                  f"max={p_max:.6f}, min={p_min:.2e}")

    # 3. 求解SDP
    result = solve_mdi_qrng_sdp(
        prob, rho, n, n,
        x_star=x_star, y_star=y_star,
        solver=solver, verbose=verbose,
    )

    # 4. 输出结果
    print(f"\n{'='*60}")
    print(f"  求解状态: {result['status']}")
    if result["p_guess"] is not None:
        print(f"  最大猜测概率 p_guess = {result['p_guess']:.8f}")
    if result["H_min"] is not None:
        print(f"  最小熵 H_min = {result['H_min']:.6f} bits")
    print(f"  求解耗时: {result['solve_time']:.2f} 秒")
    print(f"{'='*60}")

    result["mu"] = mu
    result["n"] = n
    return result


def sweep_mu(
    mu_values: np.ndarray,
    n: int,
    boundary: float | None = None,
    x_star: int = 0,
    y_star: int = 0,
    solver: str | None = None,
) -> list[dict]:
    """对多个mu值进行扫描，计算H_min。

    Args:
        mu_values: 要扫描的mu值数组。
        n: bin数量。
        boundary: 离散化边界。
        x_star, y_star: 认证输入。
        solver: 求解器。

    Returns:
        结果列表。
    """
    results = []
    total = len(mu_values)

    print(f"\n开始mu参数扫描: {total}个点, n={n}")
    print(f"mu范围: [{mu_values[0]:.3f}, {mu_values[-1]:.3f}]")

    t_start = time.time()

    for i, mu in enumerate(mu_values):
        print(f"\n[{i+1}/{total}] mu = {mu:.4f}")
        result = compute_hmin_for_mu(
            mu, n, boundary=boundary,
            x_star=x_star, y_star=y_star,
            solver=solver, verbose=False,
        )
        results.append(result)

    t_total = time.time() - t_start
    print(f"\n扫描完成，总耗时: {t_total:.1f} 秒")

    # 汇总表格
    print(f"\n{'='*70}")
    print(f"{'mu':>8} | {'p_guess':>12} | {'H_min (bits)':>12} | {'状态':>20}")
    print(f"{'-'*70}")
    for r in results:
        mu_str = f"{r['mu']:.4f}"
        pg_str = f"{r['p_guess']:.8f}" if r["p_guess"] is not None else "N/A"
        hm_str = f"{r['H_min']:.6f}" if r["H_min"] is not None else "N/A"
        print(f"{mu_str:>8} | {pg_str:>12} | {hm_str:>12} | {r['status']:>20}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MDI-QRNG SDP求解器")
    parser.add_argument("--mu", type=float, default=1.0, help="平均光子数")
    parser.add_argument("--n", type=int, default=4, help="每个正交分量的bin数量")
    parser.add_argument("--boundary", type=float, default=None, help="离散化边界（None则自适应）")
    parser.add_argument("--x-star", type=int, default=0, help="Alice认证输入 (0或1)")
    parser.add_argument("--y-star", type=int, default=0, help="Bob认证输入 (0或1)")
    parser.add_argument("--solver", type=str, default=None, help="指定求解器")
    parser.add_argument("--sweep", action="store_true", help="进行mu参数扫描")
    parser.add_argument("--mu-min", type=float, default=0.1, help="扫描起始mu")
    parser.add_argument("--mu-max", type=float, default=5.0, help="扫描终止mu")
    parser.add_argument("--mu-steps", type=int, default=10, help="扫描点数")
    parser.add_argument("--quiet", action="store_true", help="减少输出")

    args = parser.parse_args()

    if args.sweep:
        mu_values = np.linspace(args.mu_min, args.mu_max, args.mu_steps)
        sweep_mu(
            mu_values, args.n,
            boundary=args.boundary,
            x_star=args.x_star, y_star=args.y_star,
            solver=args.solver,
        )
    else:
        compute_hmin_for_mu(
            args.mu, args.n,
            boundary=args.boundary,
            x_star=args.x_star, y_star=args.y_star,
            solver=args.solver,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
