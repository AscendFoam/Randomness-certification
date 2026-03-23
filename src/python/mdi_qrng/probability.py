"""
条件概率计算模块。

基于连续变量Bell测量 (X+ = X1+X2, P- = P1-P2) 的离散化输出，
使用误差函数(erf)公式计算条件概率 P((k,l)|s1,s2)。

参考: SDP_solve.tex 公式 (1)
"""

import numpy as np
from scipy.special import erf


def make_bin_boundaries(n: int, boundary: float = 10.0) -> np.ndarray:
    """创建均匀分布的离散化bin边界。

    生成 n+1 个边界点: [-inf, -boundary, ..., +boundary, +inf]
    内部有 n-1 个等距分割点。

    Args:
        n: bin数量。
        boundary: 最外侧有限边界的绝对值。

    Returns:
        长度为 n+1 的数组。
    """
    inner = np.linspace(-boundary, boundary, n - 1)
    bounds = np.concatenate([[-np.inf], inner, [np.inf]])
    return bounds


def adaptive_boundary(mu: float, n_sigma: float = 5.0,
                      mu1: float | None = None,
                      mu2: float | None = None) -> tuple[float, float]:
    """根据mu自适应计算bin边界。

    X+ 的均值范围为 [-2*sqrt(mu), +2*sqrt(mu)]，标准差=1。
    P- 的均值为 0，标准差=1。
    边界需覆盖所有高斯中心 ± n_sigma 个标准差。

    Args:
        mu: 平均光子数（mu1=mu2=mu时使用）。
        n_sigma: 边界覆盖的标准差倍数。
        mu1, mu2: 各模式光子数。

    Returns:
        (c_boundary, d_boundary): X+和P-方向的边界绝对值。
    """
    if mu1 is None:
        mu1 = mu
    if mu2 is None:
        mu2 = mu
    max_x_center = np.sqrt(mu1) + np.sqrt(mu2)  # X+均值最大值
    c_boundary = max_x_center + n_sigma  # X+ 方向
    d_boundary = n_sigma                 # P- 方向（均值=0）
    return c_boundary, d_boundary


def compute_probabilities(
    mu: float,
    n_a: int,
    n_b: int,
    c_bounds: np.ndarray | None = None,
    d_bounds: np.ndarray | None = None,
    boundary: float | None = None,
    mu1: float | None = None,
    mu2: float | None = None,
    prob_floor: float = 1e-12,
) -> np.ndarray:
    """计算所有输入态和输出结果的条件概率。

    公式:
      P((k,l)|s1,s2) = (1/4) *
        [erf(c_k/sqrt(2) - s1*sqrt(mu1) - s2*sqrt(mu2))
         - erf(c_{k-1}/sqrt(2) - s1*sqrt(mu1) - s2*sqrt(mu2))]
        * [erf(d_l/sqrt(2)) - erf(d_{l-1}/sqrt(2))]

    Args:
        mu: 平均光子数（当 mu1/mu2 未指定时使用）。
        n_a: X+ 方向的bin数量（Alice结果）。
        n_b: P- 方向的bin数量（Bob结果）。
        c_bounds: X+ 的bin边界 (长度 n_a+1)。若为None则自动生成。
        d_bounds: P- 的bin边界 (长度 n_b+1)。若为None则自动生成。
        boundary: 自动生成边界时使用的绝对值（None则自适应计算）。
        mu1, mu2: 各模式的光子数。默认 mu1=mu2=mu。
        prob_floor: 概率下限，防止精确0值导致SDP不可行。

    Returns:
        4D数组 prob[x][y][a][b]，形状 (2, 2, n_a, n_b)。
        x,y ∈ {0,1} 对应 s1,s2 ∈ {+1,-1}。
    """
    if mu1 is None:
        mu1 = mu
    if mu2 is None:
        mu2 = mu
    if c_bounds is None:
        if boundary is not None:
            c_bnd = boundary
        else:
            c_bnd, _ = adaptive_boundary(mu, mu1=mu1, mu2=mu2)
        c_bounds = make_bin_boundaries(n_a, c_bnd)
    if d_bounds is None:
        if boundary is not None:
            d_bnd = boundary
        else:
            _, d_bnd = adaptive_boundary(mu, mu1=mu1, mu2=mu2)
        d_bounds = make_bin_boundaries(n_b, d_bnd)

    sqrt_mu1 = np.sqrt(mu1)
    sqrt_mu2 = np.sqrt(mu2)
    sqrt2 = np.sqrt(2.0)

    # 预计算 P- 方向的差分（与输入态无关）
    # d_bounds / sqrt(2)
    d_scaled = d_bounds / sqrt2
    erf_d = erf(d_scaled)
    delta_erf_d = np.diff(erf_d)  # 长度 n_b

    # 4个输入态: (s1, s2)
    signs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

    prob = np.zeros((2, 2, n_a, n_b))

    for s1, s2 in signs:
        x = (1 - s1) // 2  # s1=+1 -> x=0, s1=-1 -> x=1
        y = (1 - s2) // 2

        # X+ 均值的偏移量
        mu_shift = s1 * sqrt_mu1 + s2 * sqrt_mu2

        # c_bounds / sqrt(2) - mu_shift
        c_arg = c_bounds / sqrt2 - mu_shift
        erf_c = erf(c_arg)
        delta_erf_c = np.diff(erf_c)  # 长度 n_a

        # P((k,l)|s1,s2) = (1/4) * delta_erf_c[k] * delta_erf_d[l]
        prob[x, y, :, :] = 0.25 * np.outer(delta_erf_c, delta_erf_d)

    # 概率正则化：添加下限防止精确0值
    if prob_floor > 0:
        prob = np.maximum(prob, prob_floor)
        for x in range(2):
            for y in range(2):
                prob[x, y] /= prob[x, y].sum()

    return prob


def validate_probabilities(prob: np.ndarray) -> bool:
    """验证概率数组的合法性。

    检查:
      1. 所有概率 >= 0
      2. 对每个输入态，概率之和 ≈ 1

    Args:
        prob: 形状 (2, 2, n_a, n_b) 的概率数组。

    Returns:
        True 如果所有检查通过。

    Raises:
        ValueError: 如果检查失败。
    """
    if np.any(prob < -1e-15):
        raise ValueError(f"存在负概率: min = {prob.min()}")

    for x in range(2):
        for y in range(2):
            total = prob[x, y].sum()
            if abs(total - 1.0) > 1e-10:
                raise ValueError(
                    f"输入态 (x={x}, y={y}) 的概率之和为 {total}，应为1.0"
                )

    return True
