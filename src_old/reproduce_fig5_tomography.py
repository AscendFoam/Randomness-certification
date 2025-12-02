# -*- coding: utf-8 -*-
"""
论文Fig5复现：Fock基截断光子数m与最小熵H_min的关系曲线

程序特性：
    - 支持三种并行计算模式，用户可根据任务类型选择最优方案
    - 自动保存结果到output.txt，包含参数配置和详细结果
    - 绘制H_min与m_cutoff的关系曲线，不同压缩度绘制不同曲线
    - 僵尸进程清理机制，防止后台占用过多资源

并行模式选择（见main函数中的parallel_mode参数）：
    1. 'multi_task'（多任务并行）- 默认推荐
       - 适用场景：多个计算点（例如m=5~25，多个s_db值）
       - 工作原理：使用多进程并行计算不同参数组合，每个SDP单线程
       - 优点：多个任务同时计算，总时间显著缩短
       - 注意：内存消耗较大，m_cutoff>24或s_db过多易内存溢出

    2. 'single_sdp'（单SDP并行）
       - 适用场景：单个或少量大规模SDP问题（例如m=24，o_A=12）
       - 工作原理：串行计算任务，但每个SDP使用MOSEK多线程优化
       - 优点：内存消耗小，适合内存受限或大规模SDP
       - 注意：需要MOSEK求解器支持，多个任务时总时间较长

    3. 'none'（完全串行）
       - 适用场景：极度内存受限或调试
       - 工作原理：完全串行执行，无任何并行
       - 优点：内存消耗最小
       - 缺点：计算时间最长

参数设置注意事项（见main函数）：
    1. 截断数范围：m_lb~m_ub
       - m_cutoff>24会大幅增大内存消耗（multi_task模式）
       - 推荐使用single_sdp模式处理大m值

    2. 压缩度列表：s_db_list
       - s_db<-12后对结果提升不明显，不建议<-18
       - multi_task模式下s_db过多易内存溢出

    3. Alice测量参数：o_A
       - 分箱数增大显著提升结果，但计算时间相应增加
       - 建议8~12之间，过大易报错

    4. T_q参数：Alice q分量测量周期
       - 论文默认T_q=3不太稳定，建议使用T_q=4

    5. 其余参数影响：
       - SDP约束精度、高斯积分范围和采样点数量等
       - 这些都会影响结果，因此与论文结果可能有一定误差

    6. 常见报错原因：
       - 内存不足、误差较大导致SDP约束失败、求解器求解失败
       - 一般可通过减小参数或切换并行模式避免

依赖库：
    - numpy, scipy: 数值计算
    - cvxpy: SDP求解器框架
    - mosek (可选): 高性能SDP求解器（推荐，支持单SDP并行）
    - scs (可选): 开源SDP求解器
    - matplotlib: 绘图
    - psutil: 进程管理

部分代码和注释由AI添加
"""

import numpy as np
import math
from numpy.polynomial.hermite import hermval
import cvxpy as cp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import time
import signal
import sys
import atexit
import psutil


# ==================== 零、进程管理工具函数 ====================

# 全局变量：存储活动的executor和worker进程，用于信号处理
_active_executor = None
_worker_processes = []

def _kill_all_child_processes():
    """强制终止所有子进程（包括孙进程）"""
    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        if children:
            print(f"正在强制终止 {len(children)} 个子进程...")

            # 先尝试温和终止
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # 等待最多1秒
            _, alive = psutil.wait_procs(children, timeout=1)

            # 强制杀死仍在运行的进程
            for proc in alive:
                try:
                    print(f"  强制终止进程 {proc.pid}")
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            print("所有子进程已终止")
    except Exception as e:
        print(f"终止子进程时出错: {e}")

def _kill_worker_processes():
    """强制终止所有worker进程"""
    global _worker_processes

    # 首先使用psutil方法终止所有子进程（更可靠）
    _kill_all_child_processes()

    # 然后尝试使用multiprocessing的方法（作为备份）
    if _worker_processes:
        for proc in _worker_processes:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass

        time.sleep(0.3)

        for proc in _worker_processes:
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass

        _worker_processes.clear()

def _cleanup_processes():
    """清理函数：在程序退出时确保所有子进程被终止"""
    global _active_executor
    if _active_executor is not None:
        print("\n正在清理后台进程...")
        try:
            # 先强制终止worker进程
            _kill_worker_processes()
            # 再关闭executor
            _active_executor.shutdown(wait=False, cancel_futures=True)
            print("后台进程已清理")
        except Exception as e:
            print(f"清理进程时出错: {e}")

# 注册退出时的清理函数
atexit.register(_cleanup_processes)


# ==================== 一、基础工具函数（参数转换与态生成） ====================

def db_to_squeeze_param(s_db):
    """
    函数1：将压缩度（dB）转换为TMS态的压缩参数ṡ

    输入：
        s_db: 压缩度（dB，如-4、-6、-8，负数值越小压缩越强）

    输出：
        s_hat: TMS态的压缩参数ṡ（实数，用于后续态构建）

    功能逻辑：
        1. 根据论文Appendix B的定义，压缩度dB与ṡ的关系为：s_db = 10 * log10(e^(-2ṡ))
        2. 反解公式得：ṡ = -0.5 * ln(10^(s_db / 10))
        3. 验证合理性：s_db=-8时ṡ大于-4时，确保压缩越强ṡ越大，符合TMS态物理意义
    """
    # 步骤1: 根据论文公式 s_db = 10 * log10(e^(-2ṡ))
    # 步骤2: 反解得 ṡ = -0.5 * ln(10^(s_db / 10))
    s_hat = -0.5 * np.log(10 ** (s_db / 10))

    # 步骤3: 验证合理性（可选，仅用于开发阶段验证）
    # 压缩越强（s_db越小，如-8），ṡ应该越大
    # 例如：s_db=-8 应产生比 s_db=-4 更大的 s_hat

    return s_hat


def generate_truncated_tms(s_hat, m_cutoff):
    """
    函数2：生成截断后的TMS态密度矩阵

    输入：
        s_hat: 压缩参数ṡ（来自函数1）
        m_cutoff: Fock基截断光子数（如5、10，光子数范围0~m_cutoff）

    输出：
        tms_rho: 截断后的TMS态密度矩阵（维度为(m_cutoff+1)×(m_cutoff+1)，纯态，外积形式）

    功能逻辑：
        1. 依据论文公式B2，计算归一化因子：N = sqrt( (1 - tanh²(ṡ)) / (1 - tanh^(2*(m_cutoff+1))(ṡ)) )
        2. 构建态矢量：对n从0到m_cutoff，振幅为 N * tanh^n(ṡ)，对应两模纠缠态|nn>
        3. 转换为密度矩阵：tms_rho = |ψ><ψ|（ψ为上述态矢量），确保迹为1（tr(tms_rho)=1）
    """
    # 步骤1: 依据论文公式B2，计算归一化因子
    # N = sqrt( (1 - tanh²(ṡ)) / (1 - tanh^(2*(m_cutoff+1))(ṡ)) )
    tanh_s_hat = np.tanh(s_hat)
    tanh_squared = tanh_s_hat ** 2
    tanh_power_2m_plus_2 = tanh_s_hat ** (2 * (m_cutoff + 1))

    # 计算归一化因子N
    N = np.sqrt((1 - tanh_squared) / (1 - tanh_power_2m_plus_2))

    # 步骤2: 构建态矢量
    # 对n从0到m_cutoff，振幅为 N * tanh^n(ṡ)
    # 两模纠缠态|nn>展平为一维向量（维度为(m_cutoff+1)²）
    # 但由于是|nn>纠缠态，密度矩阵有特殊结构

    # 创建态矢量（仅包含|nn>成分）
    psi = np.zeros(m_cutoff + 1, dtype=complex)
    for n in range(m_cutoff + 1):
        psi[n] = N * (tanh_s_hat ** n)

    # 步骤3: 转换为密度矩阵 tms_rho = |ψ><ψ|
    # 对于两模系统，|nn>态的密度矩阵维度为 (m_cutoff+1)² × (m_cutoff+1)²
    # 但我们使用简化表示：先构建完整的两模态矢量

    # 完整两模态矢量（维度: (m_cutoff+1)²）
    dim = m_cutoff + 1
    psi_full = np.zeros(dim * dim, dtype=complex)

    # |nn> 态：只有对角元素非零
    for n in range(dim):
        idx = n * dim + n  # |n>_A ⊗ |n>_B 在展平向量中的索引
        psi_full[idx] = psi[n]

    # 构建密度矩阵 ρ = |ψ><ψ|
    tms_rho = np.outer(psi_full, np.conj(psi_full))

    # 验证：确保迹为1（tr(tms_rho) = 1）
    trace_value = np.trace(tms_rho)
    assert np.abs(trace_value - 1.0) < 1e-10, f"TMS态迹不为1：trace = {trace_value}"

    return tms_rho


# ==================== 二、核心计算函数（测量算符与SDP求解） ====================

def _compute_fock_normalization(n):
    """
    辅助函数：计算Fock态波函数的归一化系数，避免数值溢出

    归一化系数：1 / (π^(1/4) * sqrt(2^n * n!))
    使用对数运算：log(norm) = -0.25*log(π) - 0.5*(n*log(2) + log(n!))
    """
    if n == 0:
        return 1.0 / (np.pi ** 0.25)

    # 使用对数运算避免溢出
    # log(norm_factor) = -0.25*log(π) - 0.5*(n*log(2) + log(n!))
    log_norm = -0.25 * np.log(np.pi) - 0.5 * (n * np.log(2) + math.lgamma(n + 1))
    norm_factor = np.exp(log_norm)

    return norm_factor


def build_alice_povm(o_A, T_q, m_cutoff):
    """
    函数3：构建Alice的POVM测量算符

    输入：
        o_A: Alice测量结果数（固定为8）
        T_q: Alice q分量测量的周期（固定为3）
        m_cutoff: 截断光子数（与TMS态维度匹配）

    输出：
        alice_povm: Alice的POVM字典（key为测量输入x='q'/'p'，value为POVM元素列表，
                    每个元素是(m_cutoff+1)×(m_cutoff+1)的半正定矩阵）

    功能逻辑：
        1. 计算q分量参数：s_q = T_q / o_A（bin宽度），确保Σ_a M_a|q = 恒等矩阵I
        2. 计算p分量参数：依据论文互无偏条件 T_p = 2π / s_q，s_p = T_p / o_A（p分量bin宽度）
        3. 构建q分量POVM（x='q'）：
           - 对每个结果a（0~o_A-1），定义周期掩码函数 f_a(z, T_q)：
             当 z mod T_q ∈ [a*s_q, (a+1)*s_q) 时为1，否则为0
           - 数值积分计算POVM元素：M_a|q = ∫ f_a(z, T_q) * |z>_q <z| dz
             （用高斯求积离散z轴，确保积分精度，矩阵维度匹配TMS态）
        4. 构建p分量POVM（x='p'）：同q分量逻辑，替换T_q为T_p、s_q为s_p
        5. 输出POVM字典，确保每个x的POVM元素满足"半正定"和"求和为I"
    """
    # 步骤1: 计算q分量参数
    # s_q = T_q / o_A（bin宽度）
    s_q = T_q / o_A

    # 步骤2: 计算p分量参数
    # 依据论文互无偏条件：T_p = 2π / s_q
    T_p = 2 * np.pi / s_q
    # s_p = T_p / o_A（p分量bin宽度）
    s_p = T_p / o_A

    # 维度设置
    dim = m_cutoff + 1

    # 初始化POVM字典
    alice_povm = {}

    # 步骤3: 构建q分量POVM（x='q'）
    povm_q = []

    # 对每个测量结果 a（0 到 o_A-1）
    for a in range(o_A):
        # 定义周期掩码函数 f_a(z, T_q)
        # 当 z mod T_q ∈ [a*s_q, (a+1)*s_q) 时为1，否则为0

        # 数值积分计算 M_a|q = ∫ f_a(z, T_q) * |z>_q <z| dz
        # 使用高斯求积离散z轴

        # 设置积分范围和采样点数（确保高精度）
        # 需要覆盖多个周期以确保积分收敛
        num_periods = 10  # 覆盖10个周期
        z_min = -num_periods * T_q / 2
        z_max = num_periods * T_q / 2
        num_points = 500  # 积分采样点数

        z_values = np.linspace(z_min, z_max, num_points)
        dz = (z_max - z_min) / num_points

        # 初始化POVM元素矩阵
        M_a_q = np.zeros((dim, dim), dtype=complex)

        # 对每个z点进行积分
        for z in z_values:
            # 计算掩码函数值 f_a(z, T_q)
            z_mod = z % T_q  # z对T_q取模

            # 判断是否在 [a*s_q, (a+1)*s_q) 区间内
            if a * s_q <= z_mod < (a + 1) * s_q:
                f_a_z = 1.0
            else:
                f_a_z = 0.0

            # 构建位置本征态 |z>_q 在Fock基下的表示
            # |z>_q = Σ_n ψ_n(z) |n>，其中 ψ_n(z) 是Fock态在位置表象下的波函数
            # ψ_n(z) = (1/π^(1/4)) * (1/sqrt(2^n * n!)) * exp(-z^2/2) * H_n(z)
            # H_n(z) 是第n阶Hermite多项式

            # 计算位置本征态在Fock基下的系数
            psi_z = np.zeros(dim, dtype=complex)

            for n in range(dim):
                # 计算归一化系数（使用辅助函数避免数值溢出）
                norm_factor = _compute_fock_normalization(n)
                # 计算Hermite多项式 H_n(z)
                hermite_coeff = np.zeros(n + 1)
                hermite_coeff[n] = 1
                H_n_z = hermval(z, hermite_coeff)

                # 计算波函数值
                psi_z[n] = norm_factor * np.exp(-z ** 2 / 2) * H_n_z

            # 计算 |z><z| 并乘以掩码函数和积分微元
            # ket_z = psi_z.reshape(-1, 1)
            # bra_z = np.conj(psi_z).reshape(1, -1)
            M_a_q += f_a_z * np.outer(psi_z, np.conj(psi_z)) * dz

        povm_q.append(M_a_q)

    # 归一化q分量POVM，确保 Σ_a M_a|q = I
    sum_povm_q = sum(povm_q)
    trace_sum = np.trace(sum_povm_q) / dim  # 平均迹
    povm_q = [M / trace_sum for M in povm_q]

    alice_povm['q'] = povm_q

    # 步骤4: 构建p分量POVM（x='p'）
    # 同q分量逻辑，替换T_q为T_p、s_q为s_p
    povm_p = []

    for a in range(o_A):
        # 设置积分范围和采样点数
        num_periods = 10
        z_min = -num_periods * T_p / 2
        z_max = num_periods * T_p / 2
        num_points = 500

        z_values = np.linspace(z_min, z_max, num_points)
        dz = (z_max - z_min) / num_points

        # 初始化POVM元素矩阵
        M_a_p = np.zeros((dim, dim), dtype=complex)

        # 对每个z点进行积分
        for z in z_values:
            # 计算掩码函数值 f_a(z, T_p)
            z_mod = z % T_p

            if a * s_p <= z_mod < (a + 1) * s_p:
                f_a_z = 1.0
            else:
                f_a_z = 0.0

            # 构建动量本征态 |z>_p 在Fock基下的表示
            # |z>_p = Σ_n φ_n(z) |n>，其中 φ_n(z) 是Fock态在动量表象下的波函数
            # φ_n(z) = (1/π^(1/4)) * (1/sqrt(2^n * n!)) * exp(-z^2/2) * H_n(z) * i^n
            # （动量表象相比位置表象有相位因子 i^n）

            psi_z = np.zeros(dim, dtype=complex)

            for n in range(dim):
                # 计算归一化系数（使用辅助函数避免数值溢出）
                norm_factor = _compute_fock_normalization(n)
                hermite_coeff = np.zeros(n + 1)
                hermite_coeff[n] = 1
                H_n_z = hermval(z, hermite_coeff)

                # 动量表象的相位因子
                phase_factor = (1j) ** n

                psi_z[n] = norm_factor * np.exp(-z ** 2 / 2) * H_n_z * phase_factor

            # 计算 |z><z| 并乘以掩码函数和积分微元
            M_a_p += f_a_z * np.outer(psi_z, np.conj(psi_z)) * dz

        povm_p.append(M_a_p)

    # 归一化p分量POVM，确保 Σ_a M_a|p = I
    sum_povm_p = sum(povm_p)
    trace_sum = np.trace(sum_povm_p) / dim
    povm_p = [M / trace_sum for M in povm_p]

    alice_povm['p'] = povm_p

    # 步骤5: 验证POVM合法性
    # 验证q分量：求和应为恒等矩阵
    sum_q = sum(alice_povm['q'])
    identity = np.eye(dim)
    assert np.allclose(sum_q, identity, atol=1e-2), "q分量POVM求和不为恒等矩阵"

    # 验证p分量：求和应为恒等矩阵
    sum_p = sum(alice_povm['p'])
    assert np.allclose(sum_p, identity, atol=1e-2), "p分量POVM求和不为恒等矩阵"

    # 验证半正定性（所有本征值非负）
    for a in range(o_A):
        eigenvalues_q = np.linalg.eigvalsh(alice_povm['q'][a])
        eigenvalues_p = np.linalg.eigvalsh(alice_povm['p'][a])
        assert np.all(eigenvalues_q >= -1e-10), f"q分量POVM元素{a}不是半正定"
        assert np.all(eigenvalues_p >= -1e-10), f"p分量POVM元素{a}不是半正定"

    return alice_povm


def compute_observable_assemblage(tms_rho, alice_povm):
    """
    函数4：计算观测汇编σ̂_a|x

    输入：
        tms_rho: 截断TMS态密度矩阵（来自函数2）
        alice_povm: Alice的POVM字典（来自函数3）

    输出：
        assemblage: 观测汇编σ̂_a|x（字典，key为(x,a)，如('q',0)，
                    value为Bob的未归一化条件态密度矩阵，维度与TMS态一致）

    功能逻辑：
        1. 遍历Alice的两个输入x（'q'/'p'）：
           - 对每个结果a（0~o_A-1），提取POVM元素M_a|x
           - 计算联合测量后的态：rho_joint = tms_rho @ (M_a|x ⊗ I)
             （I为Bob模的恒等矩阵，因TMS是两模，Alice作用于第一模，Bob为第二模）
           - 对Alice的模求迹，得到Bob的未归一化条件态：sigma_a_x = tr_A(rho_joint)
             （迹操作剔除Alice模自由度）
        2. 将所有(x,a)对应的sigma_a_x存入字典，形成观测汇编（论文公式中的σ̂_a|x^obs）
    """
    # 初始化观测汇编字典
    assemblage = {}

    # 获取维度信息
    # tms_rho是两模系统，维度为 (dim × dim)²，其中 dim = m_cutoff + 1
    # tms_rho.shape[0] = dim²，所以 dim = sqrt(tms_rho.shape[0])
    total_dim_squared = tms_rho.shape[0]  # 这是 (dim × dim) 的平方
    dim = int(np.sqrt(total_dim_squared))  # 单模维度（m_cutoff + 1）

    # 将密度矩阵重塑为四维张量形式：[dim_A, dim_B, dim_A, dim_B]
    # 方便进行张量运算
    rho_tensor = tms_rho.reshape(dim, dim, dim, dim)

    # 步骤1: 遍历Alice的两个输入x（'q'/'p'）
    for x in ['q', 'p']:
        # 获取当前输入x对应的POVM元素列表
        povm_list = alice_povm[x]
        o_A = len(povm_list)  # 测量结果数

        # 对每个结果a（0~o_A-1）
        for a in range(o_A):
            # 提取POVM元素 M_a|x
            M_a_x = povm_list[a]

            # 计算联合测量后的态：rho_joint = (M_a|x ⊗ I) @ tms_rho @ (M_a|x† ⊗ I)
            # 但由于POVM的定义，实际上是：
            # rho_after = Tr_A[(M_a|x ⊗ I_B) @ rho_AB]

            # 这里使用张量运算来计算对Alice模的偏迹
            # sigma_a_x = Tr_A[(M_a|x ⊗ I_B) @ rho_AB]

            # 方法：对于两模态 rho_AB，其在计算基下的矩阵元为
            # <i,j|rho_AB|k,l> = rho_tensor[i,j,k,l]
            #
            # 计算 (M_a|x ⊗ I_B) @ rho_AB：
            # 先将 rho_tensor 与 M_a|x 在 Alice 模上进行矩阵乘法

            # 将 M_a_x 应用到 Alice 的模上
            # temp[i,j,k,l] = Σ_m M_a_x[i,m] * rho_tensor[m,j,k,l]
            temp = np.einsum('im,mjkl->ijkl', M_a_x, rho_tensor)

            # 对 Alice 的模求偏迹：Tr_A = Σ_i <i| ... |i>
            # sigma_a_x[j,l] = Σ_i temp[i,j,i,l]
            sigma_a_x = np.einsum('ijil->jl', temp)

            # 存入观测汇编字典
            assemblage[(x, a)] = sigma_a_x

    # 步骤2: 验证观测汇编的合法性（可选）
    # 对于每个x，所有a的sigma求和应该等于Bob的约化密度矩阵
    for x in ['q', 'p']:
        o_A = len(alice_povm[x])
        sum_sigma = sum(assemblage[(x, a)] for a in range(o_A))

        # 计算Bob的约化密度矩阵（对Alice模求偏迹）
        rho_B = np.einsum('ijik->jk', rho_tensor)

        # 验证是否接近（允许数值误差）
        assert np.allclose(sum_sigma, rho_B, atol=1e-6), f"输入{x}的汇编求和不等于Bob的约化态"

    return assemblage


def solve_sdp_for_guessing_prob(assemblage, x_star='q', mosek_threads=None):
    """
    函数5：通过SDP优化求解Eve的最大猜测概率

    输入：
        assemblage: 观测汇编σ̂_a|x（来自函数4）
        x_star: 提取随机性的Alice输入（固定为'q'，论文默认从q分量提取）
        mosek_threads: MOSEK求解器使用的线程数（None表示自动，0表示使用所有可用线程）

    输出：
        p_g: Eve的最大猜测概率（实数，范围(0,1]，概率越小随机性越强）

    功能逻辑：
        1. 依据论文公式(2a)-(2d)，构建SDP优化问题：
           - 目标函数：最大化 tr(Σ_e σ̂_{a=e|x_star}^e)（Eve猜对Alice结果a的概率）
           - 约束1：对所有x、a，Σ_e σ̂_a|x^e = assemblage[(x,a)]（匹配观测汇编）
           - 约束2：对所有e、x≠x'，Σ_a σ̂_a|x^e = Σ_a σ̂_a|x'^e（无信号约束，Alice输入不传递给Eve/Bob）
           - 约束3：对所有x、a、e，σ̂_a|x^e ≥ 0（半正定，符合量子态定义）
        2. 调用SDP求解器（如CVXPY+MOSEK/SeDuMi），设置数值精度（如1e-6）
        3. 输出最优目标函数值，即p_g
    """
    # 获取测量设置信息
    # 提取所有的输入x和输出a
    inputs = ['q', 'p']  # Alice的两个测量输入

    # 获取每个输入对应的输出数量
    o_A = len([key for key in assemblage.keys() if key[0] == x_star])

    # 获取Bob态的维度
    sample_key = list(assemblage.keys())[0]
    dim = assemblage[sample_key].shape[0]

    # 步骤1: 构建SDP优化问题

    # 定义优化变量：σ̂_a|x^e
    # 这是一个四维结构：[input_x][output_a][eve_guess_e] -> 密度矩阵(dim × dim)
    # 为了方便，我们使用字典存储CVXPY变量
    sigma_vars = {}

    for x in inputs:
        for a in range(o_A):
            for e in range(o_A):
                # 创建半正定矩阵变量
                sigma_vars[(x, a, e)] = cp.Variable((dim, dim), hermitian=True)

    # 定义约束列表
    constraints = []

    # 约束1：对所有x、a，Σ_e σ̂_a|x^e = assemblage[(x,a)]（匹配观测汇编）
    for x in inputs:
        for a in range(o_A):
            # 计算 Σ_e σ̂_a|x^e
            sum_over_e = sum(sigma_vars[(x, a, e)] for e in range(o_A))
            # 约束等于观测汇编
            constraints.append(sum_over_e == assemblage[(x, a)])

    # 约束2：对所有e、x≠x'，Σ_a σ̂_a|x^e = Σ_a σ̂_a|x'^e（无信号约束）
    # 这确保Alice的输入选择不会影响Bob和Eve的联合态
    for e in range(o_A):
        # 计算第一个输入的边缘态
        sum_x1 = sum(sigma_vars[(inputs[0], a, e)] for a in range(o_A))
        # 计算第二个输入的边缘态
        sum_x2 = sum(sigma_vars[(inputs[1], a, e)] for a in range(o_A))
        # 约束它们相等
        constraints.append(sum_x1 == sum_x2)

    # 约束3：对所有x、a、e，σ̂_a|x^e ≥ 0（半正定）
    for x in inputs:
        for a in range(o_A):
            for e in range(o_A):
                constraints.append(sigma_vars[(x, a, e)] >> 0)  # >> 0 表示半正定

    # 定义目标函数：最大化 tr(Σ_e σ̂_{a=e|x_star}^e)
    # 即Eve猜对的概率（当Eve的猜测e等于Alice的结果a时）
    objective_terms = []
    for e in range(o_A):
        # 当 a = e 时的项
        objective_terms.append(cp.trace(sigma_vars[(x_star, e, e)]))

    objective = cp.Maximize(cp.sum(objective_terms))

    # 步骤2: 构建并求解SDP问题
    problem = cp.Problem(objective, constraints)

    # 求解（使用MOSEK求解器以获得高精度，如果不可用则使用默认求解器）
    try:
        # 配置MOSEK参数
        mosek_params = {
            'verbose': False,
            'eps': 1e-6
        }

        # 如果指定了线程数，设置MOSEK的线程参数
        if mosek_threads is not None:
            # MOSEK参数：MSK_IPAR_NUM_THREADS 控制线程数
            # 0 表示使用所有可用线程
            mosek_params['MSK_IPAR_NUM_THREADS'] = mosek_threads if mosek_threads > 0 else 0

        problem.solve(solver=cp.MOSEK, **mosek_params)
    except:
        try:
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-6)
        except:
            problem.solve(verbose=False)

    # 步骤3: 提取最优值
    if problem.status in ['optimal', 'optimal_inaccurate']:
        p_g = problem.value
    else:
        raise ValueError(f"SDP求解失败，状态：{problem.status}")

    # 确保p_g是实数（去除可能的微小虚部）
    if np.iscomplexobj(p_g):
        p_g = np.real(p_g)

    # 确保p_g在合理范围内
    p_g = np.clip(p_g, 0.0, 1.0)

    return float(p_g)


def calculate_hmin(p_g):
    """
    函数6：从猜测概率计算最小熵

    输入：
        p_g: 猜测概率（来自函数5）

    输出：
        H_min: 最小熵（实数，单位bit，H_min = -log2(p_g)）

    功能逻辑：
        1. 依据论文"剩余哈希引理"，最小熵为猜测概率的负二进制对数
        2. 处理边界：若p_g=1则H_min=0（无随机性），若p_g=0.5则H_min=1（最大单bit随机性）
    """
    # 步骤1: 依据论文"剩余哈希引理"，最小熵为猜测概率的负二进制对数
    # H_min = -log2(p_g)

    # 步骤2: 处理边界情况
    if p_g >= 1.0:
        # 若p_g=1则H_min=0（无随机性）
        H_min = 0.0
    elif p_g <= 0.0:
        # 理论上不应该出现，但为了数值稳定性处理
        H_min = float('inf')
    else:
        # 正常情况：H_min = -log2(p_g)
        H_min = -np.log2(p_g)

    # 验证典型值：若p_g=0.5则H_min=1（最大单bit随机性）
    # 这里只是注释说明，实际计算由上述公式完成

    return H_min


# ==================== 三、整体模拟流程函数 ====================

def _compute_single_point(s_db, m, o_A, T_q, mosek_threads=None):
    """
    并行计算辅助函数：计算单个(s_db, m_cutoff)组合的H_min

    输入：
        s_db: 压缩度（dB）
        m: 截断光子数
        o_A: Alice测量结果数
        T_q: Alice q分量测量的周期
        mosek_threads: MOSEK求解器使用的线程数（用于单SDP并行优化模式）

    输出：
        result_dict: 包含s_db, m_cutoff, h_min的字典
    """
    try:
        # 调用 db_to_squeeze_param(s_db) 得到 ṡ
        s_hat = db_to_squeeze_param(s_db)

        # 调用 generate_truncated_tms(ṡ, m) 生成TMS态
        tms_rho = generate_truncated_tms(s_hat, m)

        # 调用 build_alice_povm(o_A, T_q, m) 生成Alice的POVM
        alice_povm = build_alice_povm(o_A=o_A, T_q=T_q, m_cutoff=m)

        # 调用 compute_observable_assemblage(tms_rho, alice_povm) 得到汇编
        assemblage = compute_observable_assemblage(tms_rho, alice_povm)

        # 调用 solve_sdp_for_guessing_prob(assemblage) 得到 p_g
        p_g = solve_sdp_for_guessing_prob(assemblage, x_star='q', mosek_threads=mosek_threads)

        # 调用 calculate_hmin(p_g) 得到 H_min
        h_min = calculate_hmin(p_g)

        # 返回结果
        result_entry = {
            's_db': s_db,
            'm_cutoff': m,
            'h_min': h_min,
            'p_g': p_g
        }

        print(f"  ✓ 完成: s_db={s_db} dB, m={m}, o_A={o_A}, T_q={T_q}, H_min={h_min:.6f} bits")

        return result_entry

    except Exception as e:
        print(f"  ✗ 错误: s_db={s_db} dB, m={m}, o_A={o_A}, T_q={T_q}, 错误信息: {str(e)}")
        return None


def run_hmin_vs_m_simulation(s_db_list, m_cutoff_list, o_A=8, T_q=3,
                              parallel_mode='multi_task', max_workers=None):
    """
    函数7：运行完整的H_min vs m模拟流程（支持两种并行计算模式）

    输入：
        s_db_list: 压缩度列表（固定为[-8, -6, -4] dB，对应论文3条曲线）
        m_cutoff_list: 截断数列表（固定为1~25，步长1，覆盖论文Fig5的x轴范围）
        o_A: Alice测量结果数（默认为8）
        T_q: Alice q分量测量的周期（默认为3）
        parallel_mode: 并行模式选择（默认'multi_task'）
            - 'multi_task': 多任务并行 - 使用多进程并行计算不同参数组合（适合多个计算点）
            - 'single_sdp': 单SDP并行 - 串行计算任务，但每个SDP使用MOSEK多线程优化（适合单个或少量大规模SDP）
            - 'none': 完全串行 - 不使用任何并行（最节省内存，但最慢）
        max_workers: 最大并行线程数（默认None，自动检测CPU核心数）
            - 'multi_task'模式：控制多进程池大小
            - 'single_sdp'模式：控制MOSEK求解器线程数

    输出：
        simulation_results: 模拟结果列表（每个元素为字典，含键 s_db（压缩度）、m_cutoff（截断数）、h_min（最小熵））
        run_stats: 运行统计信息字典

    功能逻辑：
        1. 生成所有(s_db, m_cutoff)组合的任务列表
        2. 根据parallel_mode选择计算方式：
           - 'multi_task': 使用ProcessPoolExecutor并行计算多个任务，每个SDP单线程
           - 'single_sdp': 串行计算任务，每个SDP使用max_workers个线程
           - 'none': 完全串行执行
        3. 返回结果列表和统计信息
    """
    # 步骤1: 生成所有任务组合
    tasks = [(s_db, m) for s_db in s_db_list for m in m_cutoff_list]
    total_tasks = len(tasks)

    # 自动检测CPU核心数
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        max_workers = cpu_count

    print(f"\n{'='*80}")
    print(f"任务配置：")
    print(f"  压缩度列表: {s_db_list} dB")
    print(f"  截断数范围: {min(m_cutoff_list)} ~ {max(m_cutoff_list)}")
    print(f"  Alice测量结果数 o_A: {o_A}")
    print(f"  Alice q分量周期 T_q: {T_q}")
    print(f"  总任务数: {total_tasks}")
    print(f"  CPU核心数: {cpu_count}")
    print(f"  并行模式: {parallel_mode}")

    if parallel_mode == 'multi_task':
        print(f"  └─ 多任务并行计算，工作进程数: {max_workers}")
        print(f"  └─ 每个SDP使用单线程")
    elif parallel_mode == 'single_sdp':
        print(f"  └─ 串行计算任务")
        print(f"  └─ 每个SDP使用 {max_workers} 个线程（MOSEK并行优化）")
    else:  # 'none'
        print(f"  └─ 完全串行模式（无并行）")
    print(f"{'='*80}\n")

    # 记录开始时间
    start_time = time.time()

    # 步骤2: 执行计算（根据并行模式选择）
    simulation_results = []
    global _active_executor, _worker_processes

    # 定义信号处理函数
    def signal_handler(_signum, _frame):
        print("\n\n检测到中断信号 (Control+C)，正在清理进程...")
        global _active_executor

        # 先强制终止所有worker进程
        _kill_worker_processes()

        # 再关闭executor
        if _active_executor is not None:
            print("正在关闭进程池...")
            _active_executor.shutdown(wait=False, cancel_futures=True)
            print("进程池已关闭")

        print("已中断程序")
        sys.exit(1)

    # 注册信号处理器
    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    try:
        if parallel_mode == 'multi_task':
            # 模式1: 多任务并行计算
            print("使用多任务并行计算模式...\n")
            print("提示: 按 Control+C 可以安全中断程序并清理所有后台进程\n")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 设置全局executor引用
                _active_executor = executor

                # 获取executor的worker进程列表
                if hasattr(executor, '_processes'):
                    _worker_processes = list(executor._processes.values())

                # 提交所有任务（mosek_threads=None表示每个SDP单线程）
                future_to_task = {
                    executor.submit(_compute_single_point, s_db, m, o_A, T_q, mosek_threads=None): (s_db, m)
                    for s_db, m in tasks
                }

                # 收集结果（按完成顺序）
                completed = 0
                for future in as_completed(future_to_task):
                    s_db, m = future_to_task[future]
                    try:
                        result = future.result()
                        if result is not None:
                            simulation_results.append(result)
                        completed += 1
                        print(f"  进度: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)")
                    except Exception as e:
                        print(f"  ✗ 任务失败: s_db={s_db} dB, m={m}, 错误: {str(e)}")

                # 清理全局引用
                _active_executor = None
                _worker_processes.clear()

        elif parallel_mode == 'single_sdp':
            # 模式2: 单SDP并行优化
            print(f"使用单SDP并行优化模式（每个SDP使用{max_workers}线程）...\n")

            for idx, (s_db, m) in enumerate(tasks, 1):
                print(f"\n处理任务 {idx}/{total_tasks}: s_db={s_db} dB, m={m}")
                # mosek_threads设置为max_workers，启用MOSEK内部并行
                result = _compute_single_point(s_db, m, o_A, T_q, mosek_threads=max_workers)
                if result is not None:
                    simulation_results.append(result)

        else:  # parallel_mode == 'none'
            # 模式3: 完全串行计算
            print("使用完全串行计算模式...\n")

            for idx, (s_db, m) in enumerate(tasks, 1):
                print(f"\n处理任务 {idx}/{total_tasks}: s_db={s_db} dB, m={m}")
                # mosek_threads=1表示单线程
                result = _compute_single_point(s_db, m, o_A, T_q, mosek_threads=1)
                if result is not None:
                    simulation_results.append(result)

    except KeyboardInterrupt:
        print("\n\n检测到键盘中断，正在清理...")
        _kill_worker_processes()
        if _active_executor is not None:
            _active_executor.shutdown(wait=False, cancel_futures=True)
            _active_executor = None
        raise

    finally:
        # 恢复原始信号处理器
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        # 确保清理引用
        _active_executor = None
        _worker_processes.clear()

    # 步骤3: 按(s_db, m_cutoff)排序结果
    simulation_results.sort(key=lambda x: (x['s_db'], x['m_cutoff']))

    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time

    # 格式化时间输出
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 步骤4: 返回结果列表和统计信息
    print(f"\n{'='*80}")
    print(f"模拟完成！")
    print(f"  成功计算: {len(simulation_results)}/{total_tasks} 个数据点")
    if len(simulation_results) < total_tasks:
        print(f"  失败任务: {total_tasks - len(simulation_results)} 个")
    print(f"  总运行时长: {hours}小时 {minutes}分钟 {seconds:.2f}秒 (共 {total_time:.2f}秒)")
    if len(simulation_results) > 0:
        avg_time_per_task = total_time / len(simulation_results)
        print(f"  平均每个任务用时: {avg_time_per_task:.2f}秒")
    else:
        avg_time_per_task = 0.0
    print(f"{'='*80}\n")

    # 准备统计信息字典
    run_stats = {
        'successful_points': len(simulation_results),
        'total_tasks': total_tasks,
        'failed_tasks': total_tasks - len(simulation_results),
        'total_time': total_time,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'avg_time_per_task': avg_time_per_task
    }

    return simulation_results, run_stats


# ==================== 四、绘图函数（匹配论文Fig5风格） ====================

def plot_fig5(simulation_results, m_lb=5, m_ub=25):
    """
    函数8：绘制Fig5风格的图像（支持任意s_db_list）

    输入：
        simulation_results: 模拟结果列表（来自函数7）
        m_lb: 截断数下限，用于设置x轴范围（默认5）
        m_ub: 截断数上限，用于设置x轴范围（默认25）

    输出：
        Fig5风格图像（可保存为PNG/SVG，或直接显示）

    功能逻辑：
        1. 数据分组：自动检测所有唯一的s_db值，按s_db分组提取m_cutoff（x轴）和h_min（y轴）
        2. 曲线绘制：
           - 自动为每个s_db分配不同的颜色和标记（从预定义的颜色池和标记池中选择）
           - 支持任意数量的s_db值（不限于3个）
           - 标注曲线标签（如"S=-14 dB"、"S=-12 dB"、"S=-10 dB"等）
        3. 坐标轴设置：
           - x轴：标签"Fock-basis cutoff m"，范围根据m_lb和m_ub动态调整
           - y轴：标签"Bound on the min-entropy H_min"，范围0.8~1.6（匹配论文Fig5的y轴范围）
        4. 图例与格式：
           - 图例放在图像右下角
           - 颜色从15种高对比度颜色池中自动选择
           - 标记从12种易区分标记池中自动选择
        5. 保存图像：分辨率300 dpi，确保文字清晰，曲线平滑（离散点连接无多余插值）
    """
    # 步骤1: 数据分组
    # 自动检测所有唯一的s_db值并分组
    data_groups = {}

    for result in simulation_results:
        s_db = result['s_db']
        if s_db not in data_groups:
            data_groups[s_db] = {'m_cutoff': [], 'h_min': []}

        data_groups[s_db]['m_cutoff'].append(result['m_cutoff'])
        data_groups[s_db]['h_min'].append(result['h_min'])

    # 对每组数据按m_cutoff排序（确保曲线连续）
    for s_db in data_groups:
        sorted_indices = np.argsort(data_groups[s_db]['m_cutoff'])
        data_groups[s_db]['m_cutoff'] = np.array(data_groups[s_db]['m_cutoff'])[sorted_indices]
        data_groups[s_db]['h_min'] = np.array(data_groups[s_db]['h_min'])[sorted_indices]

    # 步骤2: 曲线绘制
    # 创建图像
    plt.figure(figsize=(10, 6))

    # 自动生成颜色和标记样式（支持任意s_db_list）
    # 定义颜色池（使用区分度高的颜色）
    color_palette = [
        'blue', 'red', 'black', 'green', 'purple',
        'orange', 'brown', 'pink', 'gray', 'olive',
        'cyan', 'magenta', 'navy', 'darkred', 'darkgreen'
    ]

    # 定义标记池（使用常见的易区分标记）
    marker_palette = [
        'o',  # 圆圈
        's',  # 方块
        '^',  # 上三角
        'v',  # 下三角
        'D',  # 菱形
        'p',  # 五边形
        '*',  # 星形
        'h',  # 六边形
        '+',  # 加号
        'x',  # 叉号
        '<',  # 左三角
        '>',  # 右三角
    ]

    # 获取所有s_db值并排序（从小到大）
    s_db_values = sorted(data_groups.keys())

    # 为每个s_db分配颜色和标记
    style_map = {}
    for idx, s_db in enumerate(s_db_values):
        style_map[s_db] = {
            'color': color_palette[idx % len(color_palette)],
            'marker': marker_palette[idx % len(marker_palette)],
            'label': f'S={s_db} dB'
        }

    # 按照压缩度从小到大的顺序绘制
    for s_db in s_db_values:
        style = style_map[s_db]
        plt.plot(
            data_groups[s_db]['m_cutoff'],
            data_groups[s_db]['h_min'],
            color=style['color'],
            marker=style['marker'],
            linestyle='-',
            linewidth=2,
            markersize=6,
            label=style['label'],
            markerfacecolor='white',  # 空心标记，提高可读性
            markeredgewidth=2
        )

    # 步骤3: 坐标轴设置
    # x轴：标签"Fock-basis cutoff m"，范围1~25，步长5
    plt.xlabel('Fock-basis cutoff m', fontsize=14, fontweight='bold')
    plt.xlim(m_lb-1, m_ub+1)
    plt.xticks(np.arange(m_lb, m_ub+1, 5), fontsize=12)

    # y轴：标签"Bound on the min-entropy H_min"，范围0.8~2.4
    plt.ylabel('Bound on the min-entropy $H_{\\mathrm{min}}$', fontsize=14, fontweight='bold')
    plt.ylim(0.4, 2.4)
    plt.yticks(np.arange(0.4, 2.4, 0.2), fontsize=12)

    # 添加网格线（可选，提高可读性）
    plt.grid(True, linestyle='--', alpha=0.3)

    # 步骤4: 图例与格式
    # 图例放在图像右侧或下方，无多余边框
    plt.legend(
        loc='lower right',
        fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=0.9
    )

    # 设置图像边距
    plt.tight_layout()

    # 步骤5: 保存图像
    # 分辨率300 dpi，确保文字清晰
    output_filename = 'fig5_reproduction.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存为: {output_filename}")

    # 显示图像
    plt.show()

    return output_filename


# ==================== 五、结果保存函数 ====================

def save_results_to_file(simulation_results, config_params, output_filename='output.txt'):
    """
    函数9：将模拟结果和配置参数保存到文件

    输入：
        simulation_results: 模拟结果列表（来自函数7）
        config_params: 配置参数字典，包含所有运行参数
        output_filename: 输出文件名（默认为'output.txt'）

    输出：
        无（直接写入文件）

    功能逻辑：
        1. 获取当前时间戳作为运行标识
        2. 将参数配置和运行结果以追加模式写入output.txt
        3. 使用时间戳和分隔线区分不同运行记录
    """
    from datetime import datetime

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 以追加模式打开文件
    with open(output_filename, 'a', encoding='utf-8') as f:
        # 写入分隔线和时间戳
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"运行时间: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        # 写入配置参数
        f.write("【配置参数】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  压缩度列表 (s_db_list):         {config_params['s_db_list']} dB\n")
        f.write(f"  截断数下限 (m_lb):              {config_params['m_lb']}\n")
        f.write(f"  截断数上限 (m_ub):              {config_params['m_ub']}\n")
        f.write(f"  Alice测量结果数 (o_A):          {config_params['o_A']}\n")
        f.write(f"  Alice q分量周期 (T_q):          {config_params['T_q']}\n")
        f.write(f"  并行模式 (parallel_mode):       {config_params['parallel_mode']}\n")
        if config_params['parallel_mode'] == 'multi_task':
            f.write(f"    └─ 多任务并行，工作进程数:     {config_params['max_workers']}\n")
        elif config_params['parallel_mode'] == 'single_sdp':
            f.write(f"    └─ 单SDP并行，线程数:         {config_params['max_workers']}\n")
        else:
            f.write(f"    └─ 完全串行模式\n")
        f.write(f"  总计算点数:                     {config_params['total_points']}\n")
        f.write("-" * 80 + "\n\n")

        # 写入运行统计信息
        f.write("【运行统计】\n")
        f.write("-" * 80 + "\n")
        if 'run_stats' in config_params:
            stats = config_params['run_stats']
            f.write(f"  成功计算点数:                   {stats['successful_points']}/{stats['total_tasks']}\n")
            if stats['successful_points'] < stats['total_tasks']:
                f.write(f"  失败任务数:                     {stats['failed_tasks']}\n")
            f.write(f"  总运行时长:                     {stats['hours']}小时 {stats['minutes']}分钟 {stats['seconds']:.2f}秒\n")
            f.write(f"  总运行时长(秒):                 {stats['total_time']:.2f}秒\n")
            if stats['successful_points'] > 0:
                f.write(f"  平均每任务用时:                 {stats['avg_time_per_task']:.2f}秒\n")
        f.write("-" * 80 + "\n\n")

        # 写入详细结果
        f.write("【计算结果详情】\n")
        f.write("-" * 80 + "\n")

        if len(simulation_results) > 0:
            # 按压缩度分组显示
            grouped_by_sdb = {}
            for result in simulation_results:
                s_db = result['s_db']
                if s_db not in grouped_by_sdb:
                    grouped_by_sdb[s_db] = []
                grouped_by_sdb[s_db].append(result)

            # 按压缩度排序输出
            for s_db in sorted(grouped_by_sdb.keys()):
                f.write(f"\n压缩度 S = {s_db} dB:\n")
                f.write(f"{'m_cutoff':<12}{'H_min (bits)':<18}{'p_g (猜测概率)':<20}\n")
                f.write("-" * 50 + "\n")

                # 按截断数排序
                results_sorted = sorted(grouped_by_sdb[s_db], key=lambda x: x['m_cutoff'])
                for result in results_sorted:
                    f.write(f"{result['m_cutoff']:<12}{result['h_min']:<18.10f}{result['p_g']:<20.10f}\n")
        else:
            f.write("  无成功计算的数据点\n")

        f.write("\n" + "=" * 80 + "\n\n")

    print(f"\n结果已保存到: {output_filename}")


# ==================== 主函数 ====================

def main():
    """
    主函数：复现论文Fig5

    使用示例：
        python reproduce_fig5.py
    """
    print("=" * 80)
    print("论文Fig5复现：Fock基截断光子数m与最小熵H_min的关系曲线")
    print("=" * 80)

    # ========== 可配置参数区域 ==========
    # 截断数范围
    m_lb = 24  # 单模截断数下限（m_cutoff，论文默认值5）
    m_ub = 24  # 单模截断数上限（m_cutoff，论文默认值25）

    # 压缩度列表（dB）
    # s_db_list = [-8, -6, -4]  # 论文中的三条曲线
    s_db_list = [-14]  # 自定义压缩度

    # Alice测量参数
    o_A = 11   # Alice测量结果数（论文默认为8）
    T_q = 4   # Alice q分量测量的周期（论文默认为3）

    # 并行计算模式设置
    # 'multi_task': 多任务并行 - 适合多个计算点（默认推荐）
    # 'single_sdp': 单SDP并行 - 适合单个或少量大规模SDP问题
    # 'none': 完全串行 - 最节省内存
    parallel_mode = 'single_sdp'
    max_workers = None  # None表示自动使用CPU最大线程数
    # ====================================

    m_cutoff_list = list(range(m_lb, m_ub+1))  # 截断数列表：m_lb~m_ub

    print(f"\n模拟参数：")
    print(f"  压缩度列表: {s_db_list} dB")
    print(f"  截断数范围: {min(m_cutoff_list)} ~ {max(m_cutoff_list)}")
    print(f"  Alice测量结果数 o_A: {o_A}")
    print(f"  Alice q分量周期 T_q: {T_q}")
    print(f"  并行模式: {parallel_mode}")
    print(f"  总计算点数: {len(s_db_list) * len(m_cutoff_list)}")

    # 运行模拟
    print("\n开始运行模拟...")
    simulation_results, run_stats = run_hmin_vs_m_simulation(
        s_db_list,
        m_cutoff_list,
        o_A=o_A,
        T_q=T_q,
        parallel_mode=parallel_mode,
        max_workers=max_workers
    )

    # 绘制图像，即H_min与m_cutoff的关系曲线
    print("\n开始绘制Fig5...")
    output_file = plot_fig5(simulation_results, m_lb=m_lb, m_ub=m_ub)

    # 准备配置参数字典
    config_params = {
        's_db_list': s_db_list,
        'm_lb': m_lb,
        'm_ub': m_ub,
        'o_A': o_A,
        'T_q': T_q,
        'parallel_mode': parallel_mode,
        'max_workers': max_workers if max_workers else multiprocessing.cpu_count(),
        'total_points': len(s_db_list) * len(m_cutoff_list),
        'run_stats': run_stats
    }

    # 保存结果到output.txt
    save_results_to_file(simulation_results, config_params, output_filename='output.txt')

    print("\n" + "=" * 80)
    print(f"复现完成！")
    print(f"  图像已保存为: {output_file}")
    print(f"  结果已保存到: output.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
