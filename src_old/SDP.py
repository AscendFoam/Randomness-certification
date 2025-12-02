import numpy as np
import cvxpy as cp
from scipy.special import erf
from itertools import product
from tqdm import tqdm
import time
import json
import os
from datetime import datetime

class PhysicsEngine:
    """
    根据 LaTeX 文件内容实现物理模型：
    1. 生成离散化的概率分布 P((k,l) | s1, s2)
    2. 生成输入相干态的向量表示
    """
    def __init__(self, mu, n_bins, range_val):
        self.mu = mu
        self.alpha = np.sqrt(mu)
        self.n_bins = n_bins
        self.limit = range_val  # 对应 LaTeX 中的 |c_1|, |c_{n-1}| 等
        
        # 离散化边界
        # 这里的边界设置：c_0 -> -inf, c_n -> +inf, 中间均匀分布
        # 为了计算方便，通常截断范围外的概率极小，可以用有限边界近似或严格处理首尾
        boundaries = np.linspace(-range_val, range_val, n_bins - 1)
        self.c = np.concatenate(([-np.inf], boundaries, [np.inf]))
        self.d = np.concatenate(([-np.inf], boundaries, [np.inf]))
        
        # 计算内积 delta = exp(-2 mu)
        self.delta = np.exp(-2 * mu)
        
    def get_conditional_prob(self, s1, s2, k, l):
        """
        对应 LaTeX 公式 (1): P((k,l) | s1, s2)
        k, l 索引从 0 到 n_bins-1 (对应 LaTeX 的 1 到 n)
        """
        # 注意：s1, s2 取值为 +1 或 -1
        # Python 索引 k 对应 LaTeX 的 I_{+(k+1)}, 边界为 c[k] 到 c[k+1]
        
        # X+ 部分积分
        term1_upper = (self.c[k+1] / np.sqrt(2)) - s1 * self.alpha - s2 * self.alpha
        term1_lower = (self.c[k] / np.sqrt(2)) - s1 * self.alpha - s2 * self.alpha
        term1 = 0.5 * (erf(term1_upper) - erf(term1_lower))
        
        # P- 部分积分
        term2_upper = self.d[l+1] / np.sqrt(2)
        term2_lower = self.d[l] / np.sqrt(2)
        term2 = 0.5 * (erf(term2_upper) - erf(term2_lower))
        
        return term1 * term2

    def get_input_state_vector(self, s):
        """
        获取单模态的向量表示 (在 {|0>, |1>} 基下)
        s = +1 -> |alpha>  -> [1, 0]^T
        s = -1 -> |-alpha> -> [delta, sqrt(1-delta^2)]^T
        """
        if s == 1:
            return np.array([[1], [0]])
        else: # s == -1
            val = np.sqrt(1 - self.delta**2)
            return np.array([[self.delta], [val]])

    def get_joint_rho(self, s1, s2):
        """
        生成两模联合密度矩阵 rho = |psi><psi|
        |psi> = |phi_s1> (x) |phi_s2>
        """
        v1 = self.get_input_state_vector(s1)
        v2 = self.get_input_state_vector(s2)
        psi_joint = np.kron(v1, v2) # 张量积，生成 4x1 向量
        return psi_joint @ psi_joint.T # 生成 4x4 密度矩阵

    def generate_data(self, verbose=True):
        """
        生成所有需要的输入数据：
        - probs[e_idx, x_idx, y_idx]: P(e|x,y)
        - rhos[x_idx, y_idx]: 密度矩阵
        """
        if verbose:
            print("=" * 60)
            print("开始生成物理数据")
            print("=" * 60)

        # 映射: x,y index 0 -> s=+1, index 1 -> s=-1
        signs = [1, -1]

        # 1. 生成概率张量 P_obs[e][x][y]
        # e 的总数为 n_bins * n_bins
        # e = k * n_bins + l
        n_e = self.n_bins * self.n_bins
        P_obs = np.zeros((n_e, 2, 2))

        if verbose:
            print(f"\n步骤 1/3: 计算条件概率 P(e|x,y)")
            print(f"  - 离散化区间数: {self.n_bins}")
            print(f"  - 总观测结果数 e: {n_e}")
            print(f"  - 输入态数量: 4 (2x2)")

        total_probs = len(signs) * len(signs) * self.n_bins * self.n_bins
        with tqdm(total=total_probs, desc="  计算概率", disable=not verbose) as pbar:
            for x_idx, s1 in enumerate(signs):
                for y_idx, s2 in enumerate(signs):
                    for k in range(self.n_bins):
                        for l in range(self.n_bins):
                            e_idx = k * self.n_bins + l
                            P_obs[e_idx, x_idx, y_idx] = self.get_conditional_prob(s1, s2, k, l)
                            pbar.update(1)

        # 验证概率归一化
        if verbose:
            print(f"\n  概率归一化检查:")
        for x_idx in range(2):
            for y_idx in range(2):
                prob_sum = np.sum(P_obs[:, x_idx, y_idx])
                if verbose:
                    print(f"    P(e|x={x_idx},y={y_idx}) 求和 = {prob_sum:.6f}")
                if abs(prob_sum - 1.0) > 1e-4:
                    print(f"    警告: 概率和偏离1的误差较大: {abs(prob_sum - 1.0):.2e}")

        # 2. 生成量子态 Rho[x][y]
        if verbose:
            print(f"\n步骤 2/3: 构建输入量子态密度矩阵")

        Rho_states = [[None for _ in range(2)] for _ in range(2)]
        for x_idx, s1 in enumerate(signs):
            for y_idx, s2 in enumerate(signs):
                Rho_states[x_idx][y_idx] = self.get_joint_rho(s1, s2)

                # 验证密度矩阵性质
                rho = Rho_states[x_idx][y_idx]
                trace_val = np.trace(rho)
                eigenvals = np.linalg.eigvalsh(rho)
                min_eig = np.min(eigenvals)

                if verbose:
                    print(f"  ρ(x={x_idx},y={y_idx}): Tr={trace_val:.6f}, 最小特征值={min_eig:.6f}")

                if abs(trace_val - 1.0) > 1e-6:
                    print(f"    警告: 迹偏离1: {abs(trace_val - 1.0):.2e}")
                if min_eig < -1e-10:
                    print(f"    警告: 存在负特征值: {min_eig:.2e}")

        # 3. 计算边际概率 p(e) (假设输入 x, y 均匀分布)
        # 约束条件 5, 6, 7 需要 p(e)
        if verbose:
            print(f"\n步骤 3/3: 计算边际概率 p(e)")

        p_e = np.sum(P_obs, axis=(1, 2)) * 0.25

        if verbose:
            p_e_sum = np.sum(p_e)
            print(f"  Σ p(e) = {p_e_sum:.6f}")
            if abs(p_e_sum - 1.0) > 1e-6:
                print(f"  警告: p(e)求和偏离1: {abs(p_e_sum - 1.0):.2e}")
            print(f"  p(e) 范围: [{np.min(p_e):.6f}, {np.max(p_e):.6f}]")
            print("\n数据生成完成！")
            print("=" * 60)

        return P_obs, Rho_states, p_e


def partial_trace(M, dim, keep):
    """
    计算偏迹
    M: (dim*dim, dim*dim) 矩阵
    dim: 子系统维度 (这里是 2)
    keep: 保留的子系统 (0 为 Alice, 1 为 Bob)
    """
    # cvxpy 的 partial trace 实现比较复杂，通常建议重塑维度后求和
    # 这里使用 kron 结构的特性。
    # 对于 4x4 矩阵 (2x2 ⊗ 2x2)
    
    # 这是一个针对 cvxpy 变量的 helper
    # 为了通用性，我们手动实现 2-qubit 的 partial trace
    # M 索引为 |ij><kl| -> (2*i+j, 2*k+l)
    
    # 如果 keep=0 (Trace out B): 返回 dim x dim
    # result_{ik} = sum_j M_{ij, kj}
    
    # 如果 keep=1 (Trace out A): 返回 dim x dim
    # result_{jl} = sum_i M_{ij, il}
    
    # 使用 cvxpy 原子操作
    reshaped = cp.reshape(M, (dim, dim, dim, dim)) # (A_row, B_row, A_col, B_col)
    
    if keep == 0: # Keep A, Trace B
        # Sum over index 1 and 3 where index 1 == index 3
        # einsum equivalent: 'ikjl -> il' if indices were (A_r, B_r, A_c, B_c) and we want A_r, A_c sum over B_r=B_c
        # 但 cvxpy 不支持复杂 einsum。
        # 等价于 sum_{k} reshaped[:, k, :, k]
        return cp.sum([reshaped[:, k, :, k] for k in range(dim)], axis=0)
    
    else: # Keep B, Trace A
        # Sum over index 0 and 2
        return cp.sum([reshaped[k, :, k, :] for k in range(dim)], axis=0)


def solve_mdi_sdp(P_obs, Rho_states, p_e, target_indices, num_threads=24, verbose=True):
    """
    构建并求解 SDP
    P_obs: 观测概率 [e, x, y]
    Rho_states: 输入量子态 [x][y] (4x4 numpy arrays)
    p_e: Eve 结果的概率 [e]
    target_indices: (x*, y*) 需要最大化的目标对
    num_threads: MOSEK使用的线程数 (默认24，适配i7-12800HX)
    verbose: 是否显示详细信息
    """

    if verbose:
        print("\n" + "=" * 60)
        print("开始构建半定规划 (SDP) 问题")
        print("=" * 60)

    # 记录开始时间（无论是否verbose都需要）
    start_time = time.time()

    # 维度定义
    dim_A = 2 # 2D subspace defined in TeX
    dim_B = 2
    dim_total = dim_A * dim_B # 4
    n_e = len(p_e)
    n_x = 2
    n_y = 2

    if verbose:
        print(f"\n问题规模:")
        print(f"  - Alice/Bob 子空间维度: {dim_A} x {dim_B}")
        print(f"  - 联合空间维度: {dim_total}")
        print(f"  - Eve 测量结果数 (e): {n_e}")
        print(f"  - 输入设置数 (x, y): {n_x} x {n_y}")
        print(f"  - 目标输入对 (x*, y*): {target_indices}")
    
    # 对应 SDP 中的变量 M_{a,b,e}
    # 这里 a, b 通常对应 Alice 和 Bob 的推断结果。
    # 在许多 MDI-QRNG 协议中，a, b 直接对应输入设定 x, y (source certification)
    # 或者 a, b 是虚拟测量结果。
    # 根据约束 1: Tr(Sum M * psi) = p(a,b|...)
    # 假设 a, b 的取值范围也是 {0, 1} (对应 s=+1, -1)
    n_a = 2
    n_b = 2
    
    # 定义变量：是一个列表/字典，包含 n_a * n_b * n_e 个 4x4 PSD 矩阵
    # 为了方便索引，使用 3D list
    n_a = 2
    n_b = 2

    if verbose:
        print(f"\n步骤 1/6: 定义优化变量")
        print(f"  创建 M_{{a,b,e}} 变量: {n_a} x {n_b} x {n_e} = {n_a*n_b*n_e} 个 {dim_total}x{dim_total} 厄米矩阵")

    M_vars = [[[cp.Variable((dim_total, dim_total), hermitian=True)
                for _ in range(n_e)]
               for _ in range(n_b)]
              for _ in range(n_a)]

    constraints = []

    # -------------------------------------------------------
    # 约束 2: PSD (已经在定义 Variable 时隐含 hermitian=True，需显式加 >> 0)
    # -------------------------------------------------------
    if verbose:
        print(f"\n步骤 2/6: 添加半正定 (PSD) 约束")

    with tqdm(total=n_a*n_b*n_e, desc="  PSD约束", disable=not verbose) as pbar:
        for a in range(n_a):
            for b in range(n_b):
                for e in range(n_e):
                    constraints.append(M_vars[a][b][e] >> 0)
                    pbar.update(1)

    if verbose:
        print(f"  已添加 {n_a*n_b*n_e} 个PSD约束")

    # -------------------------------------------------------
    # 约束 1: 与观测数据一致
    # Tr( sum_e M_{a,b,e} (psi_x (x) psi_y) ) = p(a,b | x, y)
    # 注意：输入数据 P_obs 只有 P(e|x,y)。
    # 这里需要用户明确 p(a,b | x,y) 是什么。
    # 在标准 QRNG 中，如果是 trusted source，通常意味着我们假设 a=x, b=y 时概率为 1 (完美制备)，或者 a,b 就是 x,y 本身。
    # **假设**：这里的 SDP 是为了验证生成随机数 a,b 的概率。
    # 在 MDI 场景下，通常 a, b 是 Alice/Bob 本地的随机数。
    # 如果是 Source Independent，我们需要满足 P(e|x,y) = sum_{a,b} p(a,b|x,y) ...
    # 但是约束 1 写的是等号。
    #
    # **关键推断**：根据 LaTeX 描述的是 "Input states"，这通常是一个 Prepared-and-Measure 场景。
    # 这里的 a, b 在约束中出现，极有可能是为了模拟一个 "虚拟协议"。
    # 我们假设观测约束实际上是匹配 Eve 的结果 e。
    # 即： sum_{a,b} Tr( M_{a,b,e} rho_{x,y} ) = P(e | x, y)
    # 这是一个更标准的 MDI 约束。
    #
    # 如果严格按照用户公式： "Tr( sum_e M_{a,b,e} rho_{x,y} ) = p(a,b|x,y)"
    # 这要求我们知道 p(a,b|x,y)。在理想制备下，p(a,b|x,y) = delta_{a,x} * delta_{b,y}。
    # 我们按此假设实现（这通常用于计算猜对 x,y 的最大概率）。

    if verbose:
        print(f"\n步骤 3/6: 添加观测数据一致性约束")

    total_constraints_1 = n_x * n_y * n_a * n_b
    with tqdm(total=total_constraints_1, desc="  数据一致性", disable=not verbose) as pbar:
        for x in range(n_x):
            for y in range(n_y):
                for a in range(n_a):
                    for b in range(n_b):
                        # 构建 sum_e M
                        sum_M_e = cp.sum([M_vars[a][b][e] for e in range(n_e)])

                        # 根据SDP_doubao.md的约束1：
                        # Tr(sum_e M_{a,b,e} · ρ_σ) = P((a,b)|σ)
                        # 这里(a,b)对应离散化测量结果，σ对应输入态(x,y)
                        # P_obs[e, x, y]中，e = a * n_bins + b
                        e_idx = a * (n_b) + b
                        target_prob = P_obs[e_idx, x, y]

                        # 添加约束（取实部以确保是实数约束）
                        constraints.append(cp.real(cp.trace(sum_M_e @ Rho_states[x][y])) == target_prob)
                        pbar.update(1)

    if verbose:
        print(f"  已添加 {total_constraints_1} 个数据一致性约束")

    # -------------------------------------------------------
    # 无信号约束 & 归一化 (Constraints 3, 4, 5, 6)
    # -------------------------------------------------------

    if verbose:
        print(f"\n步骤 4/6: 构建无信号约束")
        print(f"  预计算求和项...")

    # 预计算一些求和以减少开销
    # Sum over a: M_{*, b, e}
    Sum_a_M = [[cp.sum([M_vars[a][b][e] for a in range(n_a)])
                for e in range(n_e)]
               for b in range(n_b)]

    # Sum over b: M_{a, *, e}
    Sum_b_M = [[cp.sum([M_vars[a][b][e] for b in range(n_b)])
                for e in range(n_e)]
               for a in range(n_a)]

    # 约束 3: sum_a M_{a,b,e} = I_A (x) M_{b,e}^B (No-signaling from Alice)
    # 这意味着 Sum_a_M[b][e] 的 Trace_A 必须是定义明确的 Operator，且 Trace_B 部分要是 Identity?
    # 不，公式是 Sum_a M = I_A (x) M_bob. 这意味着 Sum_a M 对 A 这一端是 Identity (除了归一化系数)。
    # 更准确的实现：PartialTrace_A(Sum_a M) 必须成比例？
    # 实际上，直接约束 Sum_a M 必须具有 tensor product 结构 I (x) B。
    # 充要条件是：PartialTrace_A(Sum_a M) (x) I_A / dA == Sum_a M  (不完全对)
    # 简单方法：约束 Sum_a M 的所有 block 对角块一致，非对角块为0？
    # 最通用的 SDP 方式：引入约化算子变量 M_Bob[b][e] 和 M_Alice[a][e]

    dim_I_A = np.eye(dim_A)
    dim_I_B = np.eye(dim_B)

    # 引入辅助变量 M_Bob[b][e] (dim_B x dim_B) 和 M_Alice[a][e] (dim_A x dim_A)
    if verbose:
        print(f"  创建辅助变量 M_Alice 和 M_Bob...")

    M_Bob = [[cp.Variable((dim_B, dim_B), hermitian=True) for _ in range(n_e)] for _ in range(n_b)]
    M_Alice = [[cp.Variable((dim_A, dim_A), hermitian=True) for _ in range(n_e)] for _ in range(n_a)]

    total_nosignal = n_e * (n_a + n_b)
    with tqdm(total=total_nosignal, desc="  无信号约束", disable=not verbose) as pbar:
        for e in range(n_e):
            # 约束 3: sum_a M_{a,b,e} = I_A (x) M_{b,e}^B
            for b in range(n_b):
                term_bob = cp.kron(dim_I_A, M_Bob[b][e])
                constraints.append(Sum_a_M[b][e] == term_bob)
                pbar.update(1)

            # 约束 4: sum_b M_{a,b,e} = M_{a,e}^A (x) I_B
            for a in range(n_a):
                term_alice = cp.kron(M_Alice[a][e], dim_I_B)
                constraints.append(Sum_b_M[a][e] == term_alice)
                pbar.update(1)

    if verbose:
        print(f"  已添加 {total_nosignal} 个无信号约束")

    # -------------------------------------------------------
    # 约束 5 & 6: 局部 POVM 归一化
    # sum_b M_{b,e}^B = p(e) I_B
    # sum_a M_{a,e}^A = p(e) I_A
    # -------------------------------------------------------
    if verbose:
        print(f"\n步骤 5/6: 添加局部POVM归一化约束")

    with tqdm(total=n_e*2, desc="  POVM归一化", disable=not verbose) as pbar:
        for e in range(n_e):
            # 对 Bob 归一化
            sum_mb = cp.sum([M_Bob[b][e] for b in range(n_b)])
            constraints.append(sum_mb == p_e[e] * dim_I_B)
            pbar.update(1)

            # 对 Alice 归一化
            sum_ma = cp.sum([M_Alice[a][e] for a in range(n_a)])
            constraints.append(sum_ma == p_e[e] * dim_I_A)
            pbar.update(1)

    if verbose:
        print(f"  已添加 {n_e*2} 个POVM归一化约束")

    # 约束 7: sum p(e) = 1 (这是数据属性，不是变量约束，但在代码里可以忽略或作为检查)
    if verbose:
        print(f"\n  验证: Σ p(e) = {np.sum(p_e):.8f}")
    
    # -------------------------------------------------------
    # 目标函数
    # Max Tr( sum_e M_{x*, y*, e} (psi_x* (x) psi_y*) )
    # -------------------------------------------------------
    if verbose:
        print(f"\n步骤 6/6: 构建目标函数")

    x_star, y_star = target_indices
    # 这里的 a, b 应该对应目标输入 x*, y*
    # 因为我们要猜测的是 x*, y*

    obj_expr = 0
    for e in range(n_e):
        # 目标是猜测 x^*, y^*，所以我们看 M 中对应 a=x^*, b=y^* 的项
        # 注意：虽然理论上结果是实数，但需要显式取实部以满足CVXPY要求
        obj_expr += cp.real(cp.trace(M_vars[x_star][y_star][e] @ Rho_states[x_star][y_star]))

    objective = cp.Maximize(obj_expr)

    # 计算构建时间（无论是否verbose都需要）
    build_time = time.time() - start_time

    if verbose:
        print(f"\nSDP 问题构建完成！")
        print(f"  总变量数: {n_a*n_b*n_e + n_a*n_e + n_b*n_e}")
        print(f"  总约束数: {len(constraints)}")
        print(f"  构建用时: {build_time:.2f} 秒")
        print("\n" + "=" * 60)
        print("开始求解 SDP")
        print("=" * 60)

    # 求解
    prob = cp.Problem(objective, constraints)

    # 配置 MOSEK 求解器参数
    mosek_params = {
        'MSK_IPAR_NUM_THREADS': num_threads,  # 使用多线程
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,  # 相对间隙容差
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,    # 原问题可行性容差
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,    # 对偶问题可行性容差
    }

    # 推荐使用 MOSEK，如果不可用，CVXPY 会尝试其他求解器
    solve_start = time.time()
    try:
        if verbose:
            print(f"\n使用 MOSEK 求解器 (线程数: {num_threads})")
            print(f"求解器参数: 相对间隙容差={mosek_params['MSK_DPAR_INTPNT_CO_TOL_REL_GAP']}")
            print("-" * 60)

        result = prob.solve(solver=cp.MOSEK, verbose=verbose, mosek_params=mosek_params)

        if verbose:
            print("-" * 60)

    except Exception as e:
        print(f"MOSEK 求解失败: {e}")
        print("尝试使用默认求解器...")
        result = prob.solve(verbose=verbose)

    solve_time = time.time() - solve_start

    if verbose:
        print("\n" + "=" * 60)
        print("求解完成")
        print("=" * 60)
        print(f"求解状态: {prob.status}")
        print(f"目标函数值 G_MDI: {result:.10f}")
        print(f"求解用时: {solve_time:.2f} 秒")
        print(f"总用时: {time.time() - start_time:.2f} 秒")

        # 计算随机性
        if result is not None and result > 0:
            h_min = -np.log2(result)
            print(f"\n随机性分析:")
            print(f"  最小熵 H_min = -log2(G_MDI) = {h_min:.6f} bits")
            print(f"  可提取随机比特数 ≈ {h_min:.4f} bits per round")

    # 保存结果到字典
    results_dict = {
        'status': prob.status,
        'optimal_value': result,
        'solve_time': solve_time,
        'build_time': build_time,
        'total_time': time.time() - start_time,
        'num_variables': n_a*n_b*n_e + n_a*n_e + n_b*n_e,
        'num_constraints': len(constraints),
        'target_indices': target_indices,
        'mosek_params': mosek_params
    }

    # 如果求解成功，保存最优的 M 矩阵
    if prob.status == 'optimal':
        results_dict['M_vars'] = [[[M_vars[a][b][e].value for e in range(n_e)]
                                    for b in range(n_b)]
                                   for a in range(n_a)]
        results_dict['M_Alice'] = [[M_Alice[a][e].value for e in range(n_e)] for a in range(n_a)]
        results_dict['M_Bob'] = [[M_Bob[b][e].value for e in range(n_e)] for b in range(n_b)]

    return result, results_dict

# =======================================================
# 主程序入口示例
# =======================================================
def run_single_optimization(mu_val, n_bins_val, range_val, target_idx=(0, 0),
                           num_threads=24, verbose=True, save_results=True):
    """
    运行单次优化

    参数:
        mu_val: 平均光子数
        n_bins_val: 离散化区间数
        range_val: 积分边界
        target_idx: 目标输入对 (x*, y*)
        num_threads: MOSEK线程数
        verbose: 是否显示详细信息
        save_results: 是否保存结果到文件

    返回:
        result: 优化结果值
        results_dict: 详细结果字典
    """
    if verbose:
        print("\n" + "=" * 60)
        print("MDI-QRNG 半定规划优化")
        print("=" * 60)
        print(f"\n物理参数:")
        print(f"  平均光子数 μ = {mu_val}")
        print(f"  离散化区间数 n = {n_bins_val}")
        print(f"  积分边界 = ±{range_val}")
        print(f"  目标输入对 (x*, y*) = {target_idx}")

    # 1. 初始化物理引擎并生成数据
    engine = PhysicsEngine(mu=mu_val, n_bins=n_bins_val, range_val=range_val)
    P_obs, Rho_states, p_e = engine.generate_data(verbose=verbose)

    # 2. 求解 SDP
    result, results_dict = solve_mdi_sdp(P_obs, Rho_states, p_e, target_idx,
                                         num_threads=num_threads, verbose=verbose)

    # 3. 保存结果
    if save_results and results_dict['status'] == 'optimal':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sdp_result_mu{mu_val}_n{n_bins_val}_{timestamp}.npz"

        # 创建 results 目录
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)

        # 保存 numpy 数据
        np.savez(filepath,
                 mu=mu_val,
                 n_bins=n_bins_val,
                 range_val=range_val,
                 target_indices=target_idx,
                 optimal_value=result,
                 P_obs=P_obs,
                 p_e=p_e,
                 solve_time=results_dict['solve_time'],
                 build_time=results_dict['build_time'],
                 total_time=results_dict['total_time'],
                 num_variables=results_dict['num_variables'],
                 num_constraints=results_dict['num_constraints'])

        # 保存 JSON 元数据
        json_filename = filename.replace('.npz', '.json')
        json_filepath = os.path.join("results", json_filename)

        metadata = {
            'mu': float(mu_val),
            'n_bins': int(n_bins_val),
            'range_val': float(range_val),
            'target_indices': target_idx,
            'optimal_value': float(result) if result is not None else None,
            'h_min': float(-np.log2(result)) if result is not None and result > 0 else None,
            'status': results_dict['status'],
            'solve_time_seconds': float(results_dict['solve_time']),
            'build_time_seconds': float(results_dict['build_time']),
            'total_time_seconds': float(results_dict['total_time']),
            'num_variables': int(results_dict['num_variables']),
            'num_constraints': int(results_dict['num_constraints']),
            'num_threads': int(num_threads),
            'timestamp': timestamp
        }

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n结果已保存:")
            print(f"  数据文件: {filepath}")
            print(f"  元数据文件: {json_filepath}")

    return result, results_dict


def parameter_scan(mu_range, n_bins_range, range_val=10.0, target_idx=(0, 0),
                  num_threads=24):
    """
    参数扫描：对不同的 μ 和 n_bins 进行优化

    参数:
        mu_range: μ 值列表或范围
        n_bins_range: n_bins 值列表
        range_val: 积分边界
        target_idx: 目标输入对
        num_threads: MOSEK线程数

    返回:
        scan_results: 扫描结果列表
    """
    print("\n" + "=" * 60)
    print("参数扫描模式")
    print("=" * 60)
    print(f"\nμ 范围: {mu_range}")
    print(f"n_bins 范围: {n_bins_range}")
    print(f"总任务数: {len(mu_range) * len(n_bins_range)}")

    scan_results = []
    total_tasks = len(mu_range) * len(n_bins_range)

    with tqdm(total=total_tasks, desc="总体进度") as pbar:
        for mu_val in mu_range:
            for n_bins_val in n_bins_range:
                pbar.set_description(f"μ={mu_val:.2f}, n={n_bins_val}")

                result, results_dict = run_single_optimization(
                    mu_val=mu_val,
                    n_bins_val=n_bins_val,
                    range_val=range_val,
                    target_idx=target_idx,
                    num_threads=num_threads,
                    verbose=False,  # 扫描时关闭详细输出
                    save_results=True
                )

                scan_results.append({
                    'mu': mu_val,
                    'n_bins': n_bins_val,
                    'optimal_value': result,
                    'h_min': -np.log2(result) if result is not None and result > 0 else None,
                    'status': results_dict['status'],
                    'solve_time': results_dict['solve_time']
                })

                pbar.update(1)

    # 保存扫描结果汇总
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    summary_file = os.path.join("results", f"scan_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(scan_results, f, indent=2, ensure_ascii=False)

    print(f"\n扫描完成！结果汇总已保存到: {summary_file}")

    # 打印结果表格
    print("\n" + "=" * 80)
    print("扫描结果汇总")
    print("=" * 80)
    print(f"{'μ':>8} {'n_bins':>8} {'G_MDI':>15} {'H_min':>15} {'求解时间(s)':>15} {'状态':>10}")
    print("-" * 80)

    for res in scan_results:
        print(f"{res['mu']:>8.3f} {res['n_bins']:>8} "
              f"{res['optimal_value']:>15.8f} "
              f"{res['h_min']:>15.6f} " if res['h_min'] is not None else f"{'N/A':>15} "
              f"{res['solve_time']:>15.2f} "
              f"{res['status']:>10}")

    return scan_results


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        # 参数扫描模式
        mu_range = np.linspace(0.1, 5.0, 10)  # μ 从 0.1 到 5.0，10个点
        n_bins_range = [2, 3, 4, 5]  # 不同的离散化区间数

        scan_results = parameter_scan(
            mu_range=mu_range,
            n_bins_range=n_bins_range,
            range_val=10.0,
            target_idx=(0, 0),
            num_threads=24
        )
    else:
        # 单次运行模式（默认）
        mu_val = 0.5        # 平均光子数
        n_bins_val = 3      # 离散化区间数
        range_val = 10.0    # 积分边界
        target_idx = (0, 0) # 目标输入对

        result, results_dict = run_single_optimization(
            mu_val=mu_val,
            n_bins_val=n_bins_val,
            range_val=range_val,
            target_idx=target_idx,
            num_threads=24,  # i7-12800HX 的 24 线程
            verbose=True,
            save_results=True
        )

        print("\n" + "=" * 60)
        print("优化完成！")
        print("=" * 60)
        if result is not None:
            print(f"最优猜测概率 G_MDI = {result:.10f}")
            if result > 0:
                print(f"最小熵 H_min = {-np.log2(result):.6f} bits")

        print("\n提示: 使用 'python SDP.py scan' 进行参数扫描")