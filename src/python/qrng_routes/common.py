from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable

import cvxpy as cp
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermite, roots_hermite


def destroy(dimension: int) -> np.ndarray:
    """湮灭算符（在Fock基下的矩阵表示）。
    
    在截断的Fock空间中构造湮灭算符的矩阵表示。湮灭算符a的作用是将光子数态|n⟩映射到√n|n-1⟩。
    
    执行逻辑：
    1. 创建一个dimension×dimension的零矩阵
    2. 遍历每个Fock态|n⟩（n从1到dimension-1）
    3. 设置矩阵元素：a[n-1, n] = √n，表示|n⟩ → √n|n-1⟩的跃迁
    4. 返回复数类型的矩阵
    
    参数：
        dimension: Fock空间的维度（截断到|0⟩, |1⟩, ..., |dimension-1⟩）
    
    返回：
        湮灭算符的矩阵表示，形状为(dimension, dimension)
    """
    op = np.zeros((dimension, dimension), dtype=complex)
    for n in range(1, dimension):
        op[n - 1, n] = math.sqrt(n)
    return op


def create(dimension: int) -> np.ndarray:
    """产生算符（在Fock基下的矩阵表示）。
    
    在截断的Fock空间中构造产生算符的矩阵表示。产生算符a†的作用是将光子数态|n⟩映射到√(n+1)|n+1⟩。
    产生算符是湮灭算符的厄米共轭。
    
    执行逻辑：
    1. 调用destroy函数获取湮灭算符
    2. 对湮灭算符取共轭转置得到产生算符
    3. 返回结果矩阵
    
    参数：
        dimension: Fock空间的维度（截断到|0⟩, |1⟩, ..., |dimension-1⟩）
    
    返回：
        产生算符的矩阵表示，形状为(dimension, dimension)
    """
    return destroy(dimension).conj().T


def kron(*operators: np.ndarray) -> np.ndarray:
    """多个算符的Kronecker积（张量积）。
    
    计算任意数量算符的张量积，用于构造多模量子系统的算符。
    例如，kron(A, B, C) = A ⊗ B ⊗ C。
    
    执行逻辑：
    1. 初始化一个1×1的单位矩阵作为起始值
    2. 依次对每个输入算符执行Kronecker积运算
    3. 每次迭代将当前结果与下一个算符做张量积
    4. 返回最终的张量积结果
    
    参数：
        *operators: 任意数量的算符矩阵，每个都是numpy数组
    
    返回：
        所有算符的张量积结果，复数类型矩阵
    """
    out = np.array([[1.0 + 0.0j]])
    for op in operators:
        out = np.kron(out, op)
    return out


def partial_trace(rho: np.ndarray, dims: Iterable[int], keep: Iterable[int]) -> np.ndarray:
    """对部分子系统求偏迹。
    
    对复合量子系统的密度矩阵进行偏迹运算，保留指定的子系统，对其余子系统求迹。
    这是量子信息中常用的操作，用于从复合系统中提取子系统的状态。
    
    执行逻辑：
    1. 将密度矩阵重塑为张量形式：ρ_{i1...in, j1...jn}，其中n是子系统数
    2. 确定需要求迹的子系统索引（不在keep列表中的）
    3. 通过转置操作重新排列张量索引，将保留和求迹的索引分组
    4. 重塑为矩阵形式后，使用爱因斯坦求和约定对迹子系统求迹
    5. 返回约化后的密度矩阵
    
    参数：
        rho: 复合系统的密度矩阵，形状为(prod(dims), prod(dims))
        dims: 各子系统的维度列表，如[2, 3]表示第一个子系统维度为2，第二个为3
        keep: 需要保留的子系统索引列表（从0开始计数）
    
    返回：
        约化后的密度矩阵，形状为(prod(dims[keep]), prod(dims[keep]))
    """
    dims = list(dims)
    keep = list(keep)
    trace = [i for i in range(len(dims)) if i not in keep]
    rho_t = rho.reshape(*(dims + dims))
    order = keep + trace + [len(dims) + i for i in keep] + [len(dims) + i for i in trace]
    rho_perm = rho_t.transpose(order)
    dim_keep = int(np.prod([dims[i] for i in keep]))
    dim_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
    rho_perm = rho_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    return np.einsum("ikjk->ij", rho_perm)


def coherent_state(dimension: int, alpha: complex) -> np.ndarray:
    """截断的相干态（在Fock基下的态矢量表示）。
    
    在有限维Fock空间中构造相干态|α⟩的近似表示。相干态是湮灭算符的本征态：a|α⟩ = α|α⟩，
    在量子光学中对应于经典相干光场的量子态。
    
    执行逻辑：
    1. 计算相干态的Fock展开系数：c_n = exp(-|α|²/2) × α^n / √(n!)
    2. 在截断空间中计算前dimension个Fock态的系数
    3. 对系数进行归一化处理，补偿截断带来的误差
    4. 返回归一化后的态矢量
    
    参数：
        dimension: Fock空间的截断维度
        alpha: 相干态的复振幅参数，α = |α|exp(iφ)
    
    返回：
        相干态的态矢量，形状为(dimension,)，已归一化
    """
    coeffs = np.zeros(dimension, dtype=complex)
    prefactor = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(dimension):
        coeffs[n] = prefactor * alpha**n / math.sqrt(math.factorial(n))
    norm = np.linalg.norm(coeffs)
    if norm > 0:
        coeffs /= norm
    return coeffs


def density_from_ket(ket: np.ndarray) -> np.ndarray:
    """从纯态矢量构造密度矩阵。
    
    将纯态|ψ⟩转换为密度矩阵形式ρ = |ψ⟩⟨ψ|。密度矩阵是描述量子态的一般形式，
    既可以表示纯态也可以表示混合态。
    
    执行逻辑：
    1. 输入态矢量|ψ⟩
    2. 计算外积：ρ = |ψ⟩⟨ψ| = ψ ⊗ ψ†
    3. 返回密度矩阵
    
    参数：
        ket: 纯态的态矢量，形状为(d,)或(d, 1)
    
    返回：
        密度矩阵，形状为(d, d)，其中d是态矢量的维度
    """
    return np.outer(ket, ket.conj())


def single_mode_squeezed_vacuum(dimension: int, squeezing_db: float) -> np.ndarray:
    """单模压缩真空态（在Fock基下的态矢量表示）。
    
    构造单模压缩真空态，这是通过压缩算符作用于真空态得到的非经典光场态。
    压缩态具有低于真空噪声水平的某个正交分量的量子噪声，在精密测量中有重要应用。
    
    执行逻辑：
    1. 将压缩参数从分贝(dB)转换为压缩因子r：r = -0.5 × ln(10^(dB/10))
    2. 构造压缩算符的生成元：G = r/2 × (a² - a†²)，其中a和a†分别是湮灭和产生算符
    3. 通过矩阵指数计算压缩算符：S(r) = exp(G)
    4. 将压缩算符作用于真空态|0⟩得到压缩真空态
    5. 归一化后返回态矢量
    
    参数：
        dimension: Fock空间的截断维度
        squeezing_db: 压缩程度，以分贝(dB)为单位，正值表示压缩
    
    返回：
        单模压缩真空态的态矢量，形状为(dimension,)
    """
    r = -0.5 * np.log(10 ** (squeezing_db / 10.0))
    a = destroy(dimension)
    adag = a.conj().T
    generator = 0.5 * r * (a @ a - adag @ adag)
    squeeze = expm(generator)
    vacuum = np.zeros(dimension, dtype=complex)
    vacuum[0] = 1.0
    ket = squeeze @ vacuum
    ket /= np.linalg.norm(ket)
    return ket


def balanced_beamsplitter_unitary(dimension: int) -> np.ndarray:
    """平衡分束器的幺正算符（双模系统）。
    
    构造50:50平衡分束器的幺正演化算符。分束器是线性光学中的基本元件，
    可以将两个输入模式线性混合。平衡分束器将每个输入模式平均分配到两个输出模式。
    
    执行逻辑：
    1. 构造两个模式的湮灭算符a和b
    2. 构造分束器算符的生成元：G = π/4 × (a†b - ab†)
    3. 通过矩阵指数计算幺正算符：U = exp(G)
    4. 返回双模系统的幺正矩阵
    
    参数：
        dimension: 每个模式的Fock空间截断维度
    
    返回：
        平衡分束器的幺正算符矩阵，形状为(dimension², dimension²)
    """
    a = destroy(dimension)
    b = destroy(dimension)
    generator = (np.pi / 4.0) * (kron(create(dimension), b) - kron(a, create(dimension)))
    return expm(generator)


def tmsv_density(dimension: int, squeezing_db: float) -> np.ndarray:
    """双模压缩真空态（在截断Fock空间中的密度矩阵）。
    
    构造双模压缩真空态(TMSV)的密度矩阵。TMSV态是两模之间的最大纠缠态，
    在量子通信、量子密钥分发和量子隐形传态中有重要应用。
    其形式为|TMSV⟩ = √(1-λ²) Σ_n λ^n |n⟩⊗|n⟩，其中λ = tanh(r)。
    
    执行逻辑：
    1. 将压缩参数从分贝(dB)转换为压缩因子r
    2. 计算参数λ = tanh(r)，λ ∈ [0, 1)
    3. 构造态矢量：在双模Fock基|n⟩⊗|m⟩中，只有n=m的对角项非零
    4. 系数c_n = √(1-λ²) × λ^n，对应|n,n⟩分量
    5. 归一化后转换为密度矩阵返回
    
    参数：
        dimension: 每个模的Fock空间截断维度
        squeezing_db: 压缩程度，以分贝(dB)为单位
    
    返回：
        双模压缩真空态的密度矩阵，形状为(dimension², dimension²)
    """
    r = -0.5 * np.log(10 ** (squeezing_db / 10.0))
    lam = np.tanh(r)
    ket = np.zeros(dimension * dimension, dtype=complex)
    prefactor = math.sqrt(1.0 - lam**2)
    for n in range(dimension):
        ket[n * dimension + n] = prefactor * lam**n
    ket /= np.linalg.norm(ket)
    return density_from_ket(ket)


def split_sms_density(dimension: int, squeezing_db: float) -> np.ndarray:
    """单模压缩真空态经过平衡分束器后的密度矩阵。
    
    模拟单模压缩真空态通过50:50平衡分束器的过程：一个输入端口输入单模压缩真空态，
    另一个输入端口输入真空态。分束器将压缩态分配到两个输出模式，产生纠缠态。
    
    执行逻辑：
    1. 构造单模压缩真空态|SMS⟩
    2. 构造真空态|0⟩作为第二个输入
    3. 构造双模输入态：|in⟩ = |SMS⟩⊗|0⟩
    4. 应用平衡分束器幺正变换：|out⟩ = U_BS |in⟩
    5. 归一化并转换为密度矩阵返回
    
    参数：
        dimension: 每个模的Fock空间截断维度
        squeezing_db: 初始单模压缩态的压缩程度，以分贝(dB)为单位
    
    返回：
        分束后双模系统的密度矩阵，形状为(dimension², dimension²)
    """
    sms = single_mode_squeezed_vacuum(dimension, squeezing_db)
    vacuum = np.zeros(dimension, dtype=complex)
    vacuum[0] = 1.0
    ket_in = np.kron(sms, vacuum)
    unitary = balanced_beamsplitter_unitary(dimension)
    ket_out = unitary @ ket_in
    ket_out /= np.linalg.norm(ket_out)
    return density_from_ket(ket_out)


def loss_kraus_1mode(dimension: int, eta: float) -> list[np.ndarray]:
    """纯损耗通道的Kraus算符（单模）。
    
    构造描述光子损耗过程的Kraus算符集合。纯损耗通道模拟光子在传输过程中的丢失，
    是量子光学和量子通信中的重要噪声模型。损耗通道保持玻色子交换关系，
    其Kraus表示为：ρ → Σ_k K_k ρ K_k†，其中K_k是第k个Kraus算符。
    
    执行逻辑：
    1. 遍历所有可能的损耗事件k（从0到dimension-1）
    2. 对每个k，构造Kraus算符K_k，其矩阵元素为：
       K_k[n-k, n] = √[C(n,k) × (1-η)^k × η^(n-k)]
       其中C(n,k)是组合数，η是传输效率
    3. K_k表示损失k个光子的过程：|n⟩ → √[C(n,k)(1-η)^k η^(n-k)] |n-k⟩
    4. 返回所有Kraus算符的列表
    
    参数：
        dimension: Fock空间的截断维度
        eta: 传输效率（透过率），取值范围[0, 1]，η=1表示无损耗
    
    返回：
        Kraus算符列表，包含dimension个算符，每个形状为(dimension, dimension)
    """
    kraus = []
    for k in range(dimension):
        op = np.zeros((dimension, dimension), dtype=complex)
        for n in range(k, dimension):
            coeff = math.sqrt(math.comb(n, k) * (1.0 - eta) ** k * eta ** (n - k))
            op[n - k, n] = coeff
        kraus.append(op)
    return kraus


def apply_symmetric_loss(rho_ab: np.ndarray, dimension: int, eta: float) -> np.ndarray:
    """对双模系统施加对称损耗通道。
    
    对双模量子系统的两个模式施加相同的纯损耗通道。损耗通道通过Kraus算符表示，
    对每个模式独立作用。这是模拟实际量子通信中损耗的常用方法。
    
    执行逻辑：
    1. 为每个模式生成损耗通道的Kraus算符集合
    2. 构造双模Kraus算符：K_{ka,kb} = K_{ka} ⊗ K_{kb}
    3. 对所有Kraus算符求和：ρ' = Σ_{ka,kb} K_{ka,kb} ρ K_{ka,kb}†
    4. 返回经过损耗后的密度矩阵
    
    参数：
        rho_ab: 双模系统的初始密度矩阵，形状为(dimension², dimension²)
        dimension: 每个模的Fock空间截断维度
        eta: 传输效率（透过率），两个模式使用相同的损耗参数
    
    返回：
        经过对称损耗后的密度矩阵，形状与输入相同
    """
    out = np.zeros_like(rho_ab, dtype=complex)
    kraus_a = loss_kraus_1mode(dimension, eta)
    kraus_b = loss_kraus_1mode(dimension, eta)
    for ka in kraus_a:
        for kb in kraus_b:
            op = kron(ka, kb)
            out += op @ rho_ab @ op.conj().T
    return out


def quadrature_op(dimension: int, theta: float) -> np.ndarray:
    """正交相位算符x_θ。
    
    构造广义正交相位算符，它是湮灭算符和产生算符的线性组合。
    正交相位算符对应于光场的可测量物理量，θ=0对应位置算符x，θ=π/2对应动量算符p。
    在量子光学中，这些算符的本征值对应于平衡零差探测的测量结果。
    
    执行逻辑：
    1. 构造湮灭算符a和产生算符a†
    2. 计算正交相位算符：x_θ = (e^(-iθ)a + e^(iθ)a†) / √2
    3. 返回算符矩阵
    
    参数：
        dimension: Fock空间的截断维度
        theta: 正交相位角度（弧度），θ=0为x算符，θ=π/2为p算符
    
    返回：
        正交相位算符的矩阵表示，形状为(dimension, dimension)
    """
    a = destroy(dimension)
    adag = a.conj().T
    return (np.exp(-1j * theta) * a + np.exp(1j * theta) * adag) / np.sqrt(2.0)


@lru_cache(maxsize=None)
def quadrature_hermite_data(dimension: int, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """高斯-厄米特求积节点和归一化多项式因子。
    
    计算用于正交相位积分的高斯-厄米特数值积分数据。这些数据用于构造正交相位测量的
    POVM元素。使用缓存机制避免重复计算，提高效率。
    
    高斯-厄米特求积用于计算形如∫f(x)e^(-x²)dx的积分，在量子光学中用于计算
    正交相位算符期望值和构造测量算符。
    
    执行逻辑：
    1. 参数验证：确保维度和节点数为正数
    2. 计算高斯-厄米特求积的节点和权重：roots_hermite(num_nodes)
    3. 计算归一化的厄米特多项式在节点处的值：
       H_n(x) / √(2^n n! √π)，这是Fock态在正交相位基下的波函数
    4. 返回节点、权重和多项式值
    
    参数：
        dimension: Fock空间的截断维度
        num_nodes: 高斯-厄米特求积的节点数，越多越精确
    
    返回：
        元组(nodes, weights, values)：
        - nodes: 求积节点，形状为(num_nodes,)
        - weights: 求积权重，形状为(num_nodes,)
        - values: 归一化厄米特多项式值，形状为(dimension, num_nodes)
    
    异常：
        ValueError: 当dimension或num_nodes非正时抛出
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    nodes, weights = roots_hermite(num_nodes)
    values = np.zeros((dimension, num_nodes), dtype=float)
    prefactor = np.pi ** (-0.25)
    for n in range(dimension):
        norm = prefactor / math.sqrt((2.0**n) * math.factorial(n))
        values[n, :] = norm * eval_hermite(n, nodes)
    return nodes, weights, values


def complete_povm_via_whitening(
    povm: list[np.ndarray],
    min_eigenvalue: float = 1e-12,
) -> list[np.ndarray]:
    """通过白化变换数值强制POVM完备性。
    
    在数值积分后，POVM元素的求和可能不完全等于单位矩阵（由于数值误差）。
    该函数通过白化变换修正POVM元素，使其满足完备性关系：Σ_i M_i = I。
    
    白化变换的原理：如果ΣM_i = S，则修正后的POVM元素为M'_i = S^(-1/2) M_i S^(-1/2)，
    这样ΣM'_i = S^(-1/2) (ΣM_i) S^(-1/2) = S^(-1/2) S S^(-1/2) = I。
    
    执行逻辑：
    1. 计算所有POVM元素的和S = ΣM_i
    2. 对S进行厄米化处理，确保其为厄米矩阵
    3. 对S进行谱分解：S = U Λ U†
    4. 将小特征值截断到最小值，避免数值不稳定
    5. 计算白化矩阵：S^(-1/2) = U Λ^(-1/2) U†
    6. 对每个POVM元素应用白化变换：M'_i = S^(-1/2) M_i S^(-1/2)
    7. 对结果进行厄米化处理，返回修正后的POVM
    
    参数：
        povm: POVM元素列表，每个元素是形状为(d, d)的矩阵
        min_eigenvalue: 特征值的最小阈值，用于避免数值不稳定
    
    返回：
        修正后的POVM元素列表，满足完备性关系
    """
    total = sum(povm)
    total = 0.5 * (total + total.conj().T)
    values, basis = np.linalg.eigh(total)
    clipped = np.maximum(values, min_eigenvalue)
    inv_sqrt = basis @ np.diag(1.0 / np.sqrt(clipped)) @ basis.conj().T
    corrected = [inv_sqrt @ element @ inv_sqrt.conj().T for element in povm]
    return [0.5 * (element + element.conj().T) for element in corrected]


def quadrature_povms_from_node_masks(
    dimension: int,
    theta: float,
    node_masks: np.ndarray,
    num_nodes: int = 400,
    enforce_completeness: bool = True,
) -> list[np.ndarray]:
    """从节点掩码构造粗粒化正交相位POVM。
    
    构造正交相位测量的POVM（正算符值测量）元素。通过将正交相位空间离散化为多个区间，
    每个区间对应一个POVM元素。使用高斯-厄米特求积进行数值积分。
    
    POVM元素对应于正交相位x_θ在特定区间的投影测量，用于量子随机数生成和
    量子态层析等应用。
    
    执行逻辑：
    1. 验证输入掩码的形状：(num_bins, num_nodes)
    2. 获取高斯-厄米特求积数据（节点、权重、多项式值）
    3. 对每个测量区间（bin），构造POVM元素：
       - 使用掩码选择该区间对应的节点
       - 计算加权波函数值：ψ_n(x) × √w(x) × √mask(x)
       - POVM元素 = Σ_x |ψ(x)⟩⟨ψ(x)|，其中|ψ(x)⟩是波函数矢量
    4. 应用相位旋转，将x正交相位旋转到x_θ：M_θ = e^(-iθN) M e^(iθN)
    5. 可选地强制POVM完备性
    6. 返回POVM元素列表
    
    参数：
        dimension: Fock空间的截断维度
        theta: 正交相位角度（弧度）
        node_masks: 节点掩码矩阵，形状为(num_bins, num_nodes)，每个掩码定义一个测量区间
        num_nodes: 高斯-厄米特求积的节点数，默认400
        enforce_completeness: 是否强制POVM完备性，默认True
    
    返回：
        POVM元素列表，包含num_bins个元素，每个形状为(dimension, dimension)
    
    异常：
        ValueError: 当node_masks形状不正确或与num_nodes不匹配时抛出
    """
    masks = np.asarray(node_masks, dtype=float)
    if masks.ndim != 2:
        raise ValueError("node_masks must have shape (num_bins, num_nodes).")

    _, weights, values = quadrature_hermite_data(dimension, num_nodes)
    if masks.shape[1] != weights.size:
        raise ValueError(
            f"node_masks has {masks.shape[1]} columns but num_nodes={num_nodes} provides {weights.size} nodes."
        )

    weighted_values = values * np.sqrt(weights)[None, :]
    base_elements: list[np.ndarray] = []
    for mask in masks:
        masked_values = weighted_values * np.sqrt(mask)[None, :]
        base_elements.append(masked_values @ masked_values.T)

    number_indices = np.arange(dimension, dtype=float)
    phase = np.exp(-1j * theta * number_indices)
    rotated = [
        (phase[:, None] * element) * phase.conj()[None, :]
        for element in base_elements
    ]

    if enforce_completeness:
        return complete_povm_via_whitening(rotated)
    return [0.5 * (element + element.conj().T) for element in rotated]


def operator_span_rank(states: list[np.ndarray], tol: float = 1e-9) -> int:
    """计算算符张成空间的秩。
    
    给定一组算符（或密度矩阵），计算它们张成的线性空间的维度。
    这在量子信息中用于判断一组量子态是否线性独立，或者用于确定
    需要多少个基算符来表示这组算符。
    
    执行逻辑：
    1. 将每个算符展平为矢量形式
    2. 将所有展平后的矢量堆叠成矩阵
    3. 对矩阵进行奇异值分解(SVD)
    4. 统计大于给定阈值的奇异值个数，即为秩
    5. 返回秩的值
    
    参数：
        states: 算符列表，每个算符是形状为(d, d)的矩阵
        tol: 奇异值的阈值，小于此值视为零
    
    返回：
        算符张成空间的秩（整数）
    """
    matrix = np.stack([state.reshape(-1) for state in states], axis=0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def support_basis(vectors: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    """计算给定态矢量张成空间的正交归一基。
    
    给定一组量子态矢量，计算它们张成的子空间的一组正交归一基。
    这在降维处理和子空间投影中很有用，例如在处理有限维截断时。
    
    执行逻辑：
    1. 将所有态矢量作为列向量堆叠成矩阵
    2. 对矩阵进行奇异值分解(SVD)：M = U Σ V†
    3. 统计大于阈值的奇异值个数，确定子空间维度（秩）
    4. 取U矩阵的前rank列作为正交归一基
    5. 返回基矩阵，每列是一个基矢量
    
    参数：
        vectors: 态矢量列表，每个矢量形状为(d,)或(d, 1)
        tol: 奇异值的阈值，小于此值视为零
    
    返回：
        正交归一基矩阵，形状为(d, rank)，每列是一个基矢量
    """
    stacked = np.column_stack(vectors)
    u, singular_values, _ = np.linalg.svd(stacked, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    return u[:, :rank]


def project_density_to_basis(rho: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """将密度矩阵投影到约化的正交归一基。
    
    将高维空间中的密度矩阵投影到低维子空间。这在处理有限维截断时很有用，
    可以减少计算复杂度，同时保持量子态在子空间中的信息。
    
    投影公式：ρ_reduced = B† ρ B，其中B是基矩阵。
    
    执行逻辑：
    1. 输入高维密度矩阵ρ和基矩阵B
    2. 计算 ρ_reduced = B† @ ρ @ B
    3. 返回约化后的密度矩阵
    
    参数：
        rho: 原始密度矩阵，形状为(d, d)
        basis: 正交归一基矩阵，形状为(d, r)，其中r是新子空间的维度
    
    返回：
        投影后的密度矩阵，形状为(r, r)
    """
    return basis.conj().T @ rho @ basis


def choose_solvers(
    preferred: str | None = None,
    solver_options: dict[str, dict] | None = None,
) -> list[tuple[object, dict]]:
    """选择CVXPY求解器并返回有序列表。
    
    根据用户偏好和已安装的求解器，返回一个有序的求解器列表及其配置选项。
    求解器按优先级排序：首选求解器 > MOSEK > SCS > CVXOPT。
    SCS作为默认后备求解器，因为它总是可用且稳定。
    
    执行逻辑：
    1. 定义内部函数merge_solver_options用于合并默认和用户选项
    2. 定义内部函数default_options_for_solver返回求解器默认配置
       - MOSEK默认使用对偶形式求解
       - 其他求解器使用空配置
    3. 如果用户指定了首选求解器，只返回该求解器（合并默认和用户选项）
    4. 否则，按优先级构造求解器列表：
       a. 如果MOSEK已安装，添加到列表（高优先级）
       b. 添加SCS求解器（默认后备），配置迭代次数和收敛阈值
       c. 如果CVXOPT已安装，添加到列表（低优先级）
    5. 返回求解器和选项的元组列表
    
    参数：
        preferred: 首选求解器名称（如"MOSEK"、"SCS"），None表示自动选择
        solver_options: 求解器选项字典，键为求解器名称，值为选项字典
    
    返回：
        求解器和选项的元组列表，如[(cp.MOSEK, {...}), (cp.SCS, {...})]
    """
    def merge_solver_options(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        merged = dict(defaults)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def default_options_for_solver(solver_name: object) -> dict[str, Any]:
        if str(solver_name).upper() == "MOSEK":
            return {"mosek_params": {"MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL"}}
        return {}

    solver_options = {} if solver_options is None else dict(solver_options)
    if preferred is not None:
        preferred_key = str(preferred)
        return [
            (
                preferred,
                merge_solver_options(
                    default_options_for_solver(preferred),
                    dict(solver_options.get(preferred_key, {})),
                ),
            )
        ]
    solvers: list[tuple[object, dict]] = []
    installed = set(cp.installed_solvers())
    if "MOSEK" in installed:
        solvers.append(
            (
                cp.MOSEK,
                merge_solver_options(
                    default_options_for_solver(cp.MOSEK),
                    dict(solver_options.get(str(cp.MOSEK), {})),
                ),
            )
        )
    scs_options = {
        "max_iters": 20000,
        "eps_abs": 1e-5,
        "eps_rel": 1e-5,
        "eps_infeas": 1e-7,
    }
    scs_options.update(solver_options.get(str(cp.SCS), {}))
    solvers.append(
        (
            cp.SCS,
            scs_options,
        )
    )
    if "CVXOPT" in installed:
        solvers.append((cp.CVXOPT, dict(solver_options.get(str(cp.CVXOPT), {}))))
    return solvers


def solve_cvxpy_problem(
    problem: cp.Problem,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
    warm_start: bool = False,
) -> tuple[str, str]:
    """使用回退链求解CVXPY问题。
    
    尝试使用多个求解器按顺序求解凸优化问题。如果第一个求解器失败，
    自动尝试下一个求解器，直到成功或所有求解器都失败。
    这提供了鲁棒性，确保问题能够被求解。
    
    执行逻辑：
    1. 初始化错误列表用于记录失败信息
    2. 调用choose_solvers获取求解器列表
    3. 按顺序尝试每个求解器：
       a. 尝试求解问题，传递求解器、选项、详细输出和热启动参数
       b. 如果成功，返回求解器名称和问题状态
       c. 如果失败，记录错误信息并尝试下一个求解器
    4. 如果所有求解器都失败，抛出RuntimeError并包含所有错误信息
    
    参数：
        problem: CVXPY问题对象
        preferred_solver: 首选求解器名称，None表示自动选择
        solver_options: 求解器选项字典
        verbose: 是否输出详细求解信息
        warm_start: 是否使用热启动（利用之前的结果加速求解）
    
    返回：
        元组(solver_name, status)：
        - solver_name: 成功求解的求解器名称
        - status: 问题状态（如"optimal"、"optimal_inaccurate"）
    
    异常：
        RuntimeError: 当所有求解器都失败时抛出
    """
    errors: list[str] = []
    for solver, options in choose_solvers(preferred_solver, solver_options=solver_options):
        try:
            problem.solve(solver=solver, verbose=verbose, warm_start=warm_start, **options)
            return str(solver), problem.status
        except (cp.error.SolverError, Exception) as exc:
            errors.append(f"{solver}: {exc}")
    raise RuntimeError("All solvers failed: " + " | ".join(errors))


class SingleDeviceGuessingProblem:
    """可重用的单设备制备-测量MDI猜测概率SDP问题。
    
    实现设备无关量子随机数生成中的猜测概率计算。该类构造一个半正定规划(SDP)问题，
    用于计算在给定输入态和测量概率分布的情况下，攻击者猜测测量结果的最大概率。
    
    这是量子随机数生成中量化随机性的核心工具，通过SDP计算最小熵H_min = -log2(p_guess)。
    
    问题建模：
    - 变量：POVM元素M_{c,e}，其中c是真实结果，e是猜测结果
    - 约束：
      1. M_{c,e} ≥ 0（半正定）
      2. Σ_e Tr(M_{c,e} ρ_s) = P(s,c)（匹配观测概率）
      3. Σ_c M_{c,e} = p_e I（POVM完备性）
      4. Σ_e p_e = 1（概率归一化）
    - 目标：最大化猜测概率 Σ_c Tr(M_{c,c} ρ_target)
    
    执行逻辑（__init__）：
    1. 保存输入态和概率分布
    2. 定义SDP变量：算符M_{c,e}和概率p_e
    3. 构造约束条件（半正定、概率匹配、完备性、归一化）
    4. 定义目标函数（最大化猜测概率）
    5. 创建CVXPY问题对象
    
    执行逻辑（solve）：
    1. 设置目标输入态参数
    2. 调用求解器求解SDP问题
    3. 提取猜测概率和计算最小熵
    4. 返回求解结果字典
    """

    def __init__(
        self,
        input_states: list[np.ndarray],
        probabilities: np.ndarray,
    ) -> None:
        self.input_states = input_states
        self.num_inputs = len(input_states)
        self.num_outputs = probabilities.shape[1]
        self.dimension = input_states[0].shape[0]

        identity = np.eye(self.dimension)
        operators = {
            (c, e): cp.Variable((self.dimension, self.dimension), hermitian=True)
            for c in range(self.num_outputs)
            for e in range(self.num_outputs)
        }
        p_e = cp.Variable(self.num_outputs, nonneg=True)
        rho_star = cp.Parameter((self.dimension, self.dimension), complex=True)

        constraints: list[cp.Constraint] = []
        for c in range(self.num_outputs):
            for e in range(self.num_outputs):
                constraints.append(operators[(c, e)] >> 0)

        for s in range(self.num_inputs):
            rho_s = input_states[s]
            for c in range(self.num_outputs):
                constraints.append(
                    cp.sum(
                        [cp.real(cp.trace(operators[(c, e)] @ rho_s)) for e in range(self.num_outputs)]
                    )
                    == probabilities[s, c]
                )

        for e in range(self.num_outputs):
            constraints.append(
                sum(operators[(c, e)] for c in range(self.num_outputs)) == p_e[e] * identity
            )

        constraints.append(cp.sum(p_e) == 1)

        objective = cp.Maximize(
            cp.sum(
                [
                    cp.real(cp.trace(operators[(c, c)] @ rho_star))
                    for c in range(self.num_outputs)
                ]
            )
        )

        self.problem = cp.Problem(objective, constraints)
        self.rho_star = rho_star

    def solve(
        self,
        target_input: int,
        preferred_solver: str | None = None,
        solver_options: dict[str, dict] | None = None,
        verbose: bool = False,
    ) -> dict:
        """求解猜测概率SDP问题。
        
        针对指定的目标输入态，求解猜测概率并计算最小熵。
        
        执行逻辑：
        1. 设置目标输入态参数rho_star
        2. 调用solve_cvxpy_problem求解SDP
        3. 提取目标函数值（猜测概率）
        4. 如果求解成功且概率为正，计算最小熵H_min = -log2(p_guess)
        5. 返回包含求解器、状态、猜测概率和最小熵的字典
        
        参数：
            target_input: 目标输入态的索引
            preferred_solver: 首选求解器名称
            solver_options: 求解器选项字典
            verbose: 是否输出详细求解信息
        
        返回：
            结果字典，包含：
            - solver: 使用的求解器名称
            - status: 问题状态
            - p_guess: 猜测概率（成功时）或None
            - H_min: 最小熵（比特，成功时）或None
        """
        self.rho_star.value = self.input_states[target_input]
        solver_name, status = solve_cvxpy_problem(
            self.problem,
            preferred_solver=preferred_solver,
            solver_options=solver_options,
            verbose=verbose,
            warm_start=True,
        )

        value = self.problem.value
        h_min = None
        if value is not None and value > 0 and status in ("optimal", "optimal_inaccurate"):
            h_min = float(-np.log2(value))

        return {
            "solver": solver_name,
            "status": status,
            "p_guess": None if value is None else float(np.real_if_close(value)),
            "H_min": h_min,
        }


def guessing_prob_single_device(
    input_states: list[np.ndarray],
    probabilities: np.ndarray,
    target_input: int,
    preferred_solver: str | None = None,
    solver_options: dict[str, dict] | None = None,
    verbose: bool = False,
) -> dict:
    """单设备制备-测量MDI猜测概率SDP求解。
    
    便捷函数，用于计算设备无关量子随机数生成中的猜测概率。
    创建SingleDeviceGuessingProblem实例并求解，适用于一次性计算场景。
    
    对于需要多次求解不同目标输入态的场景，建议直接使用SingleDeviceGuessingProblem类，
    可以复用SDP问题结构，提高效率。
    
    执行逻辑：
    1. 创建SingleDeviceGuessingProblem实例
    2. 调用solve方法求解指定目标输入态
    3. 返回求解结果
    
    参数：
        input_states: 输入态列表，每个元素是密度矩阵
        probabilities: 观测概率矩阵，形状为(num_inputs, num_outputs)
        target_input: 目标输入态的索引
        preferred_solver: 首选求解器名称，None表示自动选择
        solver_options: 求解器选项字典
        verbose: 是否输出详细求解信息
    
    返回：
        结果字典，包含：
        - solver: 使用的求解器名称
        - status: 问题状态
        - p_guess: 猜测概率
        - H_min: 最小熵（比特）
    """
    reusable_problem = SingleDeviceGuessingProblem(input_states, probabilities)
    return reusable_problem.solve(
        target_input=target_input,
        preferred_solver=preferred_solver,
        solver_options=solver_options,
        verbose=verbose,
    )
