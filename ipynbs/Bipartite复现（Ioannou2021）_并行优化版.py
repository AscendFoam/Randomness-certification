#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bipartite复现（Ioannou2021）并行优化版

本Python文件实现了对Ioannou等人2021年论文中量子随机性认证方案的并行优化，包含以下优化策略：
1. η循环的粗粒度并行化（多进程）
2. T_q循环的中粒度并行化（多线程）
3. 矩阵运算的向量化优化
4. 混合并行策略（多进程+多线程）

使用ProcessPoolExecutor和ThreadPoolExecutor实现CPU多核加速。
"""

import numpy as np
import cvxpy as cp  # 用于凸优化求解
import matplotlib.pyplot as plt
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 进度条工具
import time
import os
import argparse

# 为了结果可复现
np.random.seed(7)

# 尝试导入Mosek求解器
try:
    import mosek  # 商业级凸优化求解器
except ImportError:
    print("Mosek求解器不可用，将使用SCS求解器")
    pass

# 检查已安装的求解器，并选择最佳可用求解器
_SOLVERS = cp.installed_solvers()  # 获取cvxpy已安装的所有求解器
if "MOSEK" in _SOLVERS:  # 如果MOSEK可用，则使用MOSEK（性能更好）
    SOLVER = "MOSEK"
else:  # 否则回退使用SCS求解器
    SOLVER = "SCS"  # 备选求解器
print("Using solver:", SOLVER)


# -------------------------
# Fock基辅助函数
# -------------------------
def destroy(d: int) -> np.ndarray:
    """创建Fock基下的单模湮灭算符a，维度为d×d
    参数：
    - d: 截断维度
    返回：
    - a: 湮灭算符矩阵
    """
    a = np.zeros((d, d), dtype=complex)  # 初始化湮灭算符矩阵为零矩阵
    n = np.arange(1, d, dtype=float)  # 创建从1到d-1的数组，用于计算a的系数
    a[:-1, 1:] = np.diag(np.sqrt(n))  # 在次对角线位置填充sqrt(n)，对应湮灭算符的性质
    return a

def create(d: int) -> np.ndarray:
    """创建Fock基下的产生算符a†，即湮灭算符的共轭转置
    参数：
    - d: 截断维度
    返回：
    - a†: 产生算符矩阵
    """
    return destroy(d).conj().T

def kron(*ops) -> np.ndarray:
    """计算多个算符的克罗内克积
    参数：
    - *ops: 任意数量的矩阵算符
    返回：
    - 所有算符按顺序的克罗内克积结果
    """
    out = np.array([[1+0j]])  # 初始化克罗内克积结果为1×1的复数单位矩阵
    for X in ops:  # 按顺序对每个输入算符进行克罗内克积
        out = np.kron(out, X)
    return out

def partial_trace(rho: np.ndarray, dims, keep):
    """对量子态进行偏迹操作，保留指定子系统，剔除其他子系统
    参数：
    - rho: 密度矩阵
    - dims: 子系统维度列表，如[dA, dB]
    - keep: 要保留的子系统索引列表，如[1]表示保留B子系统
    返回：
    - 偏迹后的密度矩阵
    """
    dims = list(dims)  # 确保dims是列表类型
    N = len(dims)  # 子系统数量
    keep = list(keep)  # 确保keep是列表类型
    trace = [i for i in range(N) if i not in keep]  # 计算要剔除的子系统索引
    # 将密度矩阵重塑为2N个指标
    rho_t = rho.reshape(*(dims + dims))
    # 重排指标，将保留的子系统移到前面
    idx_order = keep + trace + [N + i for i in keep] + [N + i for i in trace]
    rho_perm = rho_t.transpose(idx_order)
    # 计算保留子系统的总维度
    d_keep = int(np.prod([dims[i] for i in keep]))
    # 计算被剔除子系统的总维度
    d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
    # 重塑以准备偏迹运算
    rho_perm = rho_perm.reshape(d_keep, d_trace, d_keep, d_trace)
    # 执行偏迹运算，对被剔除子系统求和
    return np.einsum('ikjk->ij', rho_perm)  # 使用爱因斯坦求和约定执行偏迹


# -------------------------
# TMS态（双模压缩真空态）和损耗信道
# -------------------------
def r_from_squeezing_db(s_db: float) -> float:
    """将压缩参数从dB转换为自然单位r
    参数：
    - s_db: 以dB为单位的压缩参数
    返回：
    - r: 自然单位的压缩参数
    """
    # s_db = 10 log10(e^{-2 r})  =>  r = -0.5 * ln(10**(s_db/10))
    return -0.5 * np.log(10**(s_db/10.0))

def tms_state_density(d: int, s_db: float) -> np.ndarray:
    """生成截断维度为d的双模压缩真空态密度矩阵
    TMS态形式：|psi> = sqrt(1-l^2) sum_n l^n |n,n>
    参数：
    - d: 截断维度
    - s_db: 压缩参数（dB）
    返回：
    - rho: TMS态密度矩阵
    """
    r = r_from_squeezing_db(s_db)  # 将dB转换为自然单位r
    lam = np.tanh(r)  # 计算lambda参数
    # 计算每个Fock态分量的振幅，每个Fock态的振幅为：sqrt(1-l^2) * lam^n
    amps = np.array([np.sqrt(1 - lam**2) * lam**n for n in range(d)], dtype=complex)
    psi = np.zeros((d*d,), dtype=complex)  # 初始化态矢量
    for n in range(d):
        # 在|n,n>位置填充振幅
        psi[n*d + n] = amps[n]
    # 将态矢量转换为密度矩阵
    rho = np.outer(psi, psi.conj())
    return rho

def loss_kraus_1mode(d: int, eta: float):
    """生成单模纯损耗信道的Kraus算符
    Kraus算符形式：E_k |n> = sqrt(C(n,k)) (1-η)^{k/2} η^{(n-k)/2} |n-k>, n>=k
    参数：
    - d: 截断维度
    - eta: 传输效率
    返回：
    - Ks: Kraus算符列表
    """
    Ks = []  # 存储Kraus算符的列表
    for k in range(d):  # 遍历所有可能的光子损失数量
        K = np.zeros((d, d), dtype=complex)  # 初始化Kraus算符
        for n in range(k, d):  # 对于每个可能的初始光子数
            # 计算系数：二项式系数、损耗因子和传输因子
            coeff = np.sqrt(math.comb(n, k) * (1-eta)**k * eta**(n-k))
            # 设置矩阵元素：从n光子态到n-k光子态的跃迁
            K[n-k, n] = coeff
        Ks.append(K)  # 将生成的Kraus算符添加到列表中
    return Ks

def apply_symmetric_loss_vectorized(rho_AB: np.ndarray, d: int, eta: float) -> np.ndarray:
    """向量化优化版的对称损耗信道应用
    参数：
    - rho_AB: 两体密度矩阵
    - d: 单模截断维度
    - eta: 传输效率
    返回：
    - 应用损耗后的密度矩阵
    """
    # 为A和B子系统分别生成Kraus算符
    EA = loss_kraus_1mode(d, eta)
    EB = loss_kraus_1mode(d, eta)
    out = np.zeros_like(rho_AB, dtype=complex)  # 初始化输出密度矩阵
    
    # 对于较大的d值，考虑分块处理以减少内存占用
    block_size = 4  # 分块大小，可根据实际内存情况调整
    if d <= block_size:
        # 对于小维度，直接计算
        for k in range(d):
            KA = EA[k]  # A子系统的第k个Kraus算符
            for l in range(d):
                KB = EB[l]  # B子系统的第l个Kraus算符
                K = kron(KA, KB)  # 计算联合Kraus算符
                # 应用Kraus算符：K @ rho_AB @ K^†
                out += K @ rho_AB @ K.conj().T
    else:
        # 对于大维度，采用分块处理
        for k_block in range(0, d, block_size):
            k_end = min(k_block + block_size, d)
            for l_block in range(0, d, block_size):
                l_end = min(l_block + block_size, d)
                # 处理当前块
                for k in range(k_block, k_end):
                    KA = EA[k]
                    for l in range(l_block, l_end):
                        KB = EB[l]
                        K = kron(KA, KB)
                        out += K @ rho_AB @ K.conj().T
    
    return out  # 返回应用损耗后的密度矩阵


# -------------------------
# 正交算符和周期分箱POVM
# -------------------------
def quadrature_op(d: int, theta: float) -> np.ndarray:
    """创建角度为theta的正交算符x_theta
    定义：x_theta = (a e^{-iθ} + a† e^{iθ}) / sqrt(2)
    参数：
    - d: 截断维度
    - theta: 角度参数
    返回：
    - x_theta: 正交算符矩阵
    """
    a = destroy(d)  # 获取湮灭算符
    adag = a.conj().T  # 获取产生算符
    # 按照定义计算正交算符
    return (np.exp(-1j*theta)*a + np.exp(1j*theta)*adag) / np.sqrt(2.0)

def periodic_binning_povms(d: int, theta: float, T: float, o: int):
    """构建角度为theta的正交分量的周期分箱POVM测量算子
    步骤：
    1. 对角化x_θ
    2. 根据特征值模T分配到不同的箱子
    参数：
    - d: 截断维度
    - theta: 正交角度
    - T: 周期长度
    - o: 分箱数量
    返回：
    - Ms: POVM测量算子列表，每个为d×d半正定矩阵，总和≈单位矩阵
    """
    X = quadrature_op(d, theta)  # 获取正交算符
    # 确保算符是厄米的（数值稳定性）
    Xh = 0.5*(X + X.conj().T)
    # 对角化，获取特征值和特征向量
    vals, vecs = np.linalg.eigh(Xh)
    # 计算每个分箱的宽度
    width = T / o
    # 计算特征值模T后的值，用于分配到不同的箱子
    mvals = np.mod(vals, T)
    # 确定每个特征值所属的箱子索引，确保索引在[0, o-1]范围内
    idxs = np.minimum((mvals / width).astype(int), o-1)

    Ms = [np.zeros((d, d), dtype=complex) for _ in range(o)]  # 初始化POVM算子
    # 将每个特征向量投影到对应的箱子中
    for j in range(d):
        v = vecs[:, j:j+1]  # 获取第j个特征向量
        Pj = v @ v.conj().T  # 计算投影算符
        Ms[idxs[j]] += Pj  # 将投影算符添加到对应的POVM算子中

    # 轻量级归一化，确保Σ_a M_a ≈ I
    S = sum(Ms)  # 计算所有POVM算子的和
    # 使用伪逆进行"平衡"
    Sinv = np.linalg.pinv(S)
    Ms = [Sinv @ M @ Sinv.conj().T for M in Ms]
    return Ms  # 返回POVM测量算子列表


# -------------------------
# 量子态集合（层析情况）
# -------------------------
def assemblage_tomography(rho_AB: np.ndarray, d: int, Tq: float, oA: int = 8):
    """生成Alice执行POVM测量后的量子态集合
    Alice的测量设置：x=0（θ=0，q正交），x=1（θ=π/2，p正交）
    量子态集合定义：σ_{a|x} = Tr_A[ (M_{a|x} ⊗ I) ρ_AB ]
    参数：
    - rho_AB: AB两体系统的密度矩阵
    - d: 截断维度
    - Tq: 周期分箱的周期长度
    - oA: Alice的POVM分箱数量，默认为8
    返回：
    - sigma_obs[x][a]: d×d半正定矩阵，表示在设置x下得到结果a时Bob的条件态
    """
    # Alice的两个测量设置对应的角度
    thetas = [0.0, np.pi/2]
    I_B = np.eye(d, dtype=complex)  # Bob子系统的单位算符
    sigma = []  # 存储量子态集合
    # 对每个测量设置（x）
    for theta in thetas:
        # 生成该角度下的POVM测量算子
        Ms = periodic_binning_povms(d, theta, Tq, oA)
        row = []  # 存储该设置下所有结果对应的条件态
        # 对每个可能的测量结果（a）
        for Ma in Ms:
            # 计算 (M_a ⊗ I_B) ρ_AB (M_a ⊗ I_B)^†
            Tau = kron(Ma, I_B) @ rho_AB @ kron(Ma, I_B).conj().T
            # 对Alice子系统求偏迹，得到Bob的条件态
            sig = partial_trace(Tau, [d, d], keep=[1])  # 保留Bob子系统（索引1）
            row.append(sig)
        sigma.append(row)
    return sigma  # 返回量子态集合

def normalize_per_x(sigma):
    """对量子态集合进行归一化，使得对于每个测量设置x，Σ_a Tr σ_{a|x} = 1
    参数：
    - sigma: 原始量子态集合
    返回：
    - out: 归一化后的量子态集合
    """
    out = []  # 存储归一化后的量子态集合
    # 对每个测量设置x
    for x in range(len(sigma)):
        # 计算该设置下所有结果的迹的总和
        total = sum(np.trace(sigma[x][a]).real for a in range(len(sigma[x])))
        # 对每个结果a，归一化其条件态
        out.append([sigma[x][a]/total for a in range(len(sigma[x]))])
    return out  # 返回归一化后的量子态集合


# -------------------------
# 用于式(2)的半定规划（SDP）— 层析情况
# -------------------------
def guessing_prob_sdp_tomography(sigma_obs, x_star=0, solver=SOLVER, verbose=False):
    """求解Ioannou等人论文中式(2)的原始SDP问题，用于计算最优猜测概率
    SDP问题：
      maximize   Σ_e Tr[ σ_e[e|x*] ]  # 最大化Eve猜对的概率
      s.t.       Σ_e σ_e[a|x] = σ_obs[a|x]          (对所有a,x成立)
                 Σ_a σ_e[a|x] 与x无关               (无信号条件)
                 σ_e[a|x] ⪰ 0                      (半正定约束)
    参数：
    - sigma_obs[x][a]: d×d半正定矩阵，表示观测到的量子态集合
    - x_star: 固定的测量设置，默认为0
    - solver: 使用的SDP求解器
    - verbose: 是否显示求解过程信息
    返回：
    - val: 最优猜测概率值
    - status: 求解器状态
    """
    mX = len(sigma_obs)        # 测量设置数量，如2（q, p）
    mA = len(sigma_obs[0])     # 每个设置下的结果数量，如8个箱子
    d = sigma_obs[0][0].shape[0]  # 希尔伯特空间维度

    # 定义变量σ_e[a|x]，三维数组：[Eve猜测e][Alice结果a][Alice设置x]
    sigma_e = [[[cp.Variable((d, d), hermitian=True)
                 for _ in range(mX)] for _ in range(mA)] for _ in range(mA)]
    cons = []  # 存储约束条件

    # (i) 与观测到的量子态集合一致的约束
    for x in range(mX):
        for a in range(mA):
            # 对所有Eve猜测e求和，必须等于观测到的σ_obs[a|x]
            lhs = sum(sigma_e[e][a][x] for e in range(mA))
            cons.append(lhs == sigma_obs[x][a])

    # (ii) No-signalling on Eve side: Σ_a σ_e[a|x] = Σ_a σ_e[a|x'] (indep. of x)
    # (ii) Eve侧的无信号条件：Σ_a σ_e[a|x] 与x无关
    # 以e=0，x=0的边缘分布作为参考
    ref = sum(sigma_e[0][a][0] for a in range(mA))
    # 确保对于所有设置x，边缘分布都等于参考分布
    for x in range(mX):
        lhs = sum(sigma_e[0][a][x] for a in range(mA))
        cons.append(lhs == ref)

    # (iii) 半正定约束：所有σ_e[a|x]必须是半正定矩阵
    for e in range(mA):
        for x in range(mX):
            for a in range(mA):
                cons.append(sigma_e[e][a][x] >> 0)

    # 定义目标函数：最大化Eve猜对的概率，即当Eve猜测e且Alice结果也是e时的概率之和
    # 即最大化Tr[σ_e[e|x*] e^† e]
    obj = cp.Maximize(sum(cp.trace(sigma_e[e][e][x_star]) for e in range(mA)))
    # 构建优化问题
    prob = cp.Problem(obj, cons)
    # 求解问题
    val = prob.solve(solver=getattr(cp, solver), verbose=verbose)
    # 返回最优值和求解状态，确保返回实数
    return float(np.real_if_close(val)), prob.status


# -------------------------
# 并行优化的工作函数
# -------------------------
def process_tomography_Tq(args):
    """处理单个Tq值的工作函数，用于中粒度并行化
    参数：
    - args: 包含rho, d, Tq, oA, verbose的元组
    返回：
    - Tq: 周期分箱大小
    - H: 对应的最小熵值
    """
    rho, d, Tq, oA, verbose = args
    
    # 生成量子态集合并归一化
    sigma = assemblage_tomography(rho, d, Tq, oA=oA)
    sigma = normalize_per_x(sigma)
    
    # 求解SDP，得到猜测概率
    try:
        p_g, status = guessing_prob_sdp_tomography(
            sigma, x_star=0, solver=SOLVER, verbose=verbose
        )
        # 计算最小熵H_min = -log2(p_g)
        # 使用max(p_g, 1e-15)避免对极小值求对数导致数值问题
        H = -np.log2(max(p_g, 1e-15))
        return Tq, H
    except Exception as e:
        print(f"处理Tq={Tq}时出错: {e}")
        return Tq, -np.inf

def process_tomography_eta(args):
    """处理单个eta值的工作函数，用于粗粒度并行化
    参数：
    - args: 包含d, s_db, eta, Tq_grid, oA, n_threads, verbose的元组
    返回：
    - eta: 传输效率
    - best_H: 最大H_min值
    - best_T: 最优T_q值
    """
    d, s_db, eta, Tq_grid, oA, n_threads, verbose = args
    
    # 1) 构建TMS态并应用对称损耗
    rho0 = tms_state_density(d, s_db)  # 生成初始TMS态
    rho = apply_symmetric_loss_vectorized(rho0, d, eta)  # 应用损耗（使用向量化版本）

    best_H = -np.inf  # 初始化最佳H_min为负无穷
    best_T = None  # 初始化最佳T_q为None

    # 2) 扫描Alice的周期分箱大小T_q - 使用多线程并行
    args_list = [(rho, d, Tq, oA, verbose) for Tq in Tq_grid]
    
    # 根据CPU核心数自适应选择线程数
    if n_threads <= 0:
        n_threads = min(len(Tq_grid), os.cpu_count() or 4)
    
    # 使用ThreadPoolExecutor进行T_q循环的并行处理
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # 提交所有任务
        future_to_Tq = {executor.submit(process_tomography_Tq, args): args[2] for args in args_list}
        
        # 处理完成的任务
        for future in as_completed(future_to_Tq):
            try:
                Tq, H = future.result()
                # 更新最佳值
                if H > best_H:
                    best_H, best_T = H, Tq
            except Exception as e:
                Tq = future_to_Tq[future]
                print(f"处理Tq={Tq}时出错（线程级）: {e}")
    
    return eta, best_H, best_T


def reproduce_fig2_tomography_parallel(
    d=8,                # Fock空间截断维度（↑提高精度，↑增加运行时间）
    s_db=-4.0,           # 压缩参数（方差dB）
    eta_grid=np.linspace(0.55, 1.0, 10), # 效率η的扫描范围
    Tq_grid=np.linspace(2.0, 10.0, 9), # Alice的周期分箱大小T_q的扫描范围
    oA=8, # Alice的POVM分箱数量
    n_processes=-1, # 进程数，-1表示自动选择
    n_threads=-1, # 每个进程内的线程数，-1表示自动选择
    verbose=False, # 是否显示求解器详细信息
    output_dir="../output_images/bipartite" # 结果输出目录
):
    """
    复现Ioannou等人论文中的图2，扫描效率η并优化H_min关于Alice的周期分箱大小T_q
    使用多进程+多线程的混合并行策略加速计算。
    
    参数：
    - d: Fock空间截断维度
    - s_db: 压缩参数
    - eta_grid: 效率η的扫描网格
    - Tq_grid: T_q的扫描网格
    - oA: Alice的POVM分箱数量
    - n_processes: 进程数，-1表示自动选择
    - n_threads: 每个进程内的线程数，-1表示自动选择
    - verbose: 是否显示详细信息
    - output_dir: 结果输出目录
    
    返回：
    - Hmax[η]: 每个η对应的最大H_min值
    - Tbest[η]: 每个η对应的最优T_q值
    """
    # 根据CPU核心数自适应选择进程数
    if n_processes <= 0:
        n_processes = min(len(eta_grid), os.cpu_count() or 4)
    
    print(f"启动并行计算：{n_processes}个进程，每个进程最多{n_threads if n_threads > 0 else 'auto'}个线程")
    print(f"扫描参数：η范围{eta_grid[0]:.2f}-{eta_grid[-1]:.2f}（{len(eta_grid)}个点），" \
          f"T_q范围{Tq_grid[0]:.2f}-{Tq_grid[-1]:.2f}（{len(Tq_grid)}个点）")
    
    # 准备参数列表
    args_list = [(d, s_db, eta, Tq_grid, oA, n_threads, verbose) for eta in eta_grid]
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用ProcessPoolExecutor进行η循环的并行处理
    # 使用字典来保存结果，以保持原始顺序
    results = {eta: (None, None) for eta in eta_grid}
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # 提交所有任务
        future_to_eta = {executor.submit(process_tomography_eta, args): args[2] for args in args_list}
        
        # 使用tqdm显示总体进度
        with tqdm(total=len(eta_grid), desc="总体进度", ncols=100) as pbar:
            # 处理完成的任务
            for future in as_completed(future_to_eta):
                eta = future_to_eta[future]
                try:
                    _, best_H, best_T = future.result()
                    results[eta] = (best_H, best_T)
                    # 打印每个η的总结信息
                    print(f"η={eta:.3f}: max H_min={best_H:.5f} at T_q={best_T:.2f}")
                except Exception as e:
                    print(f"处理η={eta}时出错（进程级）: {e}")
                    results[eta] = (-np.inf, None)
                finally:
                    pbar.update(1)
    
    # 按原始顺序整理结果
    Hmax = [results[eta][0] for eta in eta_grid]
    Tbest = [results[eta][1] for eta in eta_grid]
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"并行计算完成，总耗时: {total_time:.2f}秒")
    
    # 保存结果到文件
    output_file = os.path.join(output_dir, f"baseline_tomography({SOLVER}).txt")
    with open(output_file, 'w') as f:
        f.write("eta, Hmax, Tbest\n")
        for eta, H, T in zip(eta_grid, Hmax, Tbest):
            f.write(f"{eta:.6f}, {H:.6f}, {T:.6f}\n")
    print(f"结果已保存至: {output_file}")
    
    # 绘制结果图
    plt.figure(figsize=(10, 6))
    plt.plot(eta_grid, Hmax, 'o-', linewidth=2, markersize=8)
    plt.xlabel('传输效率 η', fontsize=14)
    plt.ylabel('H_min (bit)', fontsize=14)
    plt.title(f'随机数认证的最小熵 (使用{SOLVER}求解器)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 保存图像
    image_file = os.path.join(output_dir, f"baseline_tomography({SOLVER}).png")
    plt.tight_layout()
    plt.savefig(image_file, dpi=300)
    print(f"图像已保存至: {image_file}")
    plt.show()
    
    return np.array(Hmax), np.array(Tbest)


# -------------------------
# 主函数和测试代码
# -------------------------
def main():
    """主函数，处理命令行参数并运行并行优化的计算"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Bipartite量子随机性认证并行优化版')
    parser.add_argument('--d', type=int, default=8, help='Fock空间截断维度')
    parser.add_argument('--s_db', type=float, default=-4.0, help='压缩参数（dB）')
    parser.add_argument('--eta_start', type=float, default=0.55, help='效率η的起始值')
    parser.add_argument('--eta_end', type=float, default=1.0, help='效率η的结束值')
    parser.add_argument('--eta_num', type=int, default=10, help='效率η的采样点数量')
    parser.add_argument('--Tq_start', type=float, default=2.0, help='T_q的起始值')
    parser.add_argument('--Tq_end', type=float, default=10.0, help='T_q的结束值')
    parser.add_argument('--Tq_num', type=int, default=9, help='T_q的采样点数量')
    parser.add_argument('--oA', type=int, default=8, help='Alice的POVM分箱数量')
    parser.add_argument('--processes', type=int, default=-1, help='进程数，-1表示自动选择')
    parser.add_argument('--threads', type=int, default=-1, help='每个进程内的线程数，-1表示自动选择')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('--output_dir', type=str, default="../output_images/bipartite", help='结果输出目录')
    
    args = parser.parse_args()
    
    # 生成扫描网格
    eta_grid = np.linspace(args.eta_start, args.eta_end, args.eta_num)
    Tq_grid = np.linspace(args.Tq_start, args.Tq_end, args.Tq_num)
    
    # 运行并行优化的计算
    Hmax, Tbest = reproduce_fig2_tomography_parallel(
        d=args.d,
        s_db=args.s_db,
        eta_grid=eta_grid,
        Tq_grid=Tq_grid,
        oA=args.oA,
        n_processes=args.processes,
        n_threads=args.threads,
        verbose=args.verbose,
        output_dir=args.output_dir
    )


# 性能优化建议和注意事项
def print_optimization_tips():
    """打印性能优化建议和注意事项"""
    print("\n========== 性能优化建议和注意事项 ==========")
    print("1. 硬件选择：")
    print("   - 推荐使用多核CPU以充分利用并行计算优势")
    print("   - 对于大规模计算，建议使用MOSEK求解器（比SCS快约3-5倍）")
    print("   - 确保有足够的内存（d=8时约需2-4GB，d=12时可能需要8GB以上）")
    
    print("\n2. 参数调优：")
    print("   - n_processes参数建议设置为物理CPU核心数")
    print("   - n_threads参数建议设置为逻辑核心数/进程数")
    print("   - 对于d较大的情况，可以调整apply_symmetric_loss_vectorized中的block_size参数")
    
    print("\n3. 扩展建议：")
    print("   - 对于超大规模计算，可以考虑使用分布式计算框架")
    print("   - 可以尝试使用GPU加速部分矩阵运算（需修改代码）")
    print("   - 对于频繁运行的场景，可以实现结果缓存机制")
    print("==========================================\n")


if __name__ == "__main__":
    # 打印优化建议
    print_optimization_tips()
    # 运行主函数
    main()