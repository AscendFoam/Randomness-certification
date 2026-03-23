"""
量子态构建模块。

在4维有效Hilbert空间 {|00>, |01>, |10>, |11>} 中构建
MDI-QRNG的双模相干态矢量和密度矩阵。

物理假设:
  - 两个模式使用相同的平均光子数 mu1 = mu2 = mu
  - 相位选择 phi ∈ {0, π}，对应符号 s ∈ {+1, -1}
  - 4个输入态由 sigma = (s1, s2) 索引
"""

import numpy as np


def compute_delta(mu: float) -> float:
    """计算内积 delta = <alpha|-alpha> = exp(-2*mu)。

    Args:
        mu: 平均光子数 (mu >= 0)。

    Returns:
        delta: 实正标量，范围 (0, 1]。
    """
    return np.exp(-2.0 * mu)


def build_state_vectors(mu: float) -> dict[tuple[int, int], np.ndarray]:
    """构建4个态矢量 |psi_{(s1,s2)}> 在4维基底下的表示。

    基底顺序: {|00>, |01>, |10>, |11>}
    其中 |0> = |alpha>, |1> 是与 |alpha> 正交的归一化矢量，
    |-alpha> = delta|0> + sqrt(1-delta^2)|1>。

    Args:
        mu: 平均光子数。

    Returns:
        字典 {(s1, s2): 4维numpy数组}:
          (+1,+1): [1, 0, 0, 0]
          (+1,-1): [delta, sqrt(1-delta^2), 0, 0]
          (-1,+1): [delta, 0, sqrt(1-delta^2), 0]
          (-1,-1): [delta^2, delta*sqrt(1-delta^2),
                     delta*sqrt(1-delta^2), 1-delta^2]
    """
    delta = compute_delta(mu)
    s = np.sqrt(1.0 - delta**2)

    states = {
        (+1, +1): np.array([1.0, 0.0, 0.0, 0.0]),
        (+1, -1): np.array([delta, s, 0.0, 0.0]),
        (-1, +1): np.array([delta, 0.0, s, 0.0]),
        (-1, -1): np.array([delta**2, delta * s, delta * s, 1.0 - delta**2]),
    }
    return states


def build_density_matrices(mu: float) -> dict[tuple[int, int], np.ndarray]:
    """构建4个密度矩阵 rho_{(s1,s2)} = |psi><psi|。

    Args:
        mu: 平均光子数。

    Returns:
        字典 {(s1, s2): 4x4 numpy数组 (实对称半正定，迹为1)}。
    """
    vectors = build_state_vectors(mu)
    rho = {}
    for key, psi in vectors.items():
        rho[key] = np.outer(psi, psi)
    return rho


# ---------- 索引映射工具 ----------

def sigma_index_to_signs(x: int, y: int) -> tuple[int, int]:
    """将 (x, y) 索引映射到相位符号 (s1, s2)。

    x=0 -> s1=+1, x=1 -> s1=-1
    y=0 -> s2=+1, y=1 -> s2=-1
    """
    return (1 - 2 * x, 1 - 2 * y)


def signs_to_sigma_index(s1: int, s2: int) -> tuple[int, int]:
    """将相位符号 (s1, s2) 映射到 (x, y) 索引。"""
    return ((1 - s1) // 2, (1 - s2) // 2)
