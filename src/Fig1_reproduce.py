import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- 1. 基础量子态与工具函数 ---

def get_tetrahedron_states():
    """
    生成Bloch球上正四面体顶点的四个量子态 (作为输入态 \\psi_x)
    """
    # 顶点坐标
    v0 = np.array([0, 0, 1])
    v1 = np.array([2*np.sqrt(2)/3, 0, -1/3])
    v2 = np.array([-np.sqrt(2)/3, np.sqrt(2/3), -1/3])
    v3 = np.array([-np.sqrt(2)/3, -np.sqrt(2/3), -1/3])
    
    vectors = [v0, v1, v2, v3]
    paulis = [np.array([[0, 1], [1, 0]], dtype=complex),       # X
              np.array([[0, -1j], [1j, 0]], dtype=complex),    # Y
              np.array([[1, 0], [0, -1]], dtype=complex)]      # Z
    eye = np.eye(2, dtype=complex)
    
    states = []
    for v in vectors:
        rho = 0.5 * (eye + v[0]*paulis[0] + v[1]*paulis[1] + v[2]*paulis[2])
        states.append(rho)
    return states

def get_bell_projectors():
    """
    返回4个贝尔态投影算子。
    顺序对应输出 a, b = 0,1,2,3
    """
    # 贝尔基: |Phi+>, |Phi->, |Psi+>, |Psi->
    # 注意：为了和标准BSM对应，我们需要明确基的定义
    # |00> + |11>, |00> - |11>, |01> + |10>, |01> - |10> (忽略归一化系数，但在投影算子中加回)
    
    kets = [
        np.array([1, 0, 0, 1]) / np.sqrt(2), # Phi+
        np.array([1, 0, 0, -1]) / np.sqrt(2),# Phi-
        np.array([0, 1, 1, 0]) / np.sqrt(2), # Psi+
        np.array([0, 1, -1, 0]) / np.sqrt(2) # Psi-
    ]
    
    projectors = [np.outer(k, k.conj()) for k in kets]
    return projectors

def robust_partial_trace(op_full):
    """
    对复合系统 A0(2) x A(2) x B(2) x B0(2) 进行偏迹运算。
    迹掉 A(index 1) 和 B(index 2)，保留 A0(index 0) 和 B0(index 3)。
    输入: 16x16 矩阵
    输出: 4x4 矩阵 (对应 A0 x B0)
    """
    # 重塑为张量: (dim_A0, dim_A, dim_B, dim_B0) x (dim_A0, dim_A, dim_B, dim_B0)
    # 形状: (2, 2, 2, 2, 2, 2, 2, 2)
    reshaped = op_full.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    
    # Einsum 索引定义:
    # 输入下标: i(A0_r), j(A_r), k(B_r), l(B0_r), m(A0_c), n(A_c), o(B_c), p(B0_c)
    # 迹掉 A (j=n) 和 B (k=o) -> 对 j, k 求和
    # 输出下标: i(A0_r), l(B0_r), m(A0_c), p(B0_c) -> 重组为 (A0_r, B0_r) x (A0_c, B0_c)
    # 注意：最终需要reshape成 (4, 4)，即 (i*2+l, m*2+p)
    # 正确写法: 'ijklmjkp -> ilmp' (j重复表示迹掉A, k重复表示迹掉B)

    reduced_tensor = np.einsum('ijklmjkp->ilmp', reshaped)
    
    # 重塑回矩阵 4x4
    return reduced_tensor.reshape(4, 4)

def get_observed_measurements(w):
    """
    计算给定 w 下的观测有效POVM集合 {M_tilde_{a,b}}。
    """
    # 1. 构建 Werner 态 (共享部分 A, B)
    # rho = w |Phi+><Phi+| + (1-w) I/4
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    P_phi_plus = np.outer(phi_plus, phi_plus.conj())
    rho_AB = w * P_phi_plus + (1 - w) * (np.eye(4) / 4)
    
    # 2. 构建全系统算子并计算偏迹
    # 测量算子 M_a 作用在 A0 A 上，M_b 作用在 B B0 上
    # 总测量: P_a^{A0 A} \otimes P_b^{B B0}
    # 这是一个 16x16 矩阵 (2^4)
    # 输入态结构: I^{A0} \otimes rho^{AB} \otimes I^{B0}
    
    bell_projs = get_bell_projectors()
    M_obs = {}
    
    I2 = np.eye(2)
    #Full_State = I_A0 (x) rho_AB (x) I_B0
    # 注意kron顺序: I(2) x rho(4) x I(2) -> 16x16
    state_term = np.kron(np.kron(I2, rho_AB), I2)
    
    total_trace_check = 0.0
    
    for a in range(4):
        for b in range(4):
            # 构建测量算子 P_a (x) P_b
            # P_a 是 4x4 (A0, A), P_b 是 4x4 (B, B0)
            # 它们作用在不同的空间。我们需要把它们组合成 16x16 的大矩阵
            # 顺序: A0(0), A(1), B(2), B0(3)
            # P_a 作用在 0,1; P_b 作用在 2,3.
            # 因此直接 kron(P_a, P_b) 即可符合 0,1,2,3 的顺序
            Meas_Big = np.kron(bell_projs[a], bell_projs[b])
            
            # 待迹算子: M * State
            Op_to_trace = Meas_Big @ state_term
            
            # 计算有效POVM: Tr_{AB}(Op)
            M_tilde = robust_partial_trace(Op_to_trace)
            
            M_obs[(a, b)] = M_tilde
            total_trace_check += np.trace(M_tilde)

    # 验证: 所有 POVM 元素之和的迹应该是 4 (I_4x4 的迹)
    # 因为 Sum_{a,b} M_{a,b} = Tr_{AB}(Sum P_a x P_b ... ) = Tr_{AB}(I x I ...) = I_{A0 B0}
    if abs(total_trace_check - 4.0) > 1e-5:
        print(f"Warning: Trace check failed! Sum = {total_trace_check}")
            
    return M_obs

# --- 2. SDP 求解器 ---

def solve_randomness_sdp(w):
    """
    求解方程 (37) 的SDP，计算最小熵 H_min
    """
    M_obs = get_observed_measurements(w)
    
    # 变量定义
    # M[a,b,e]: Alice得a, Bob得b, Eve猜e. 16*16 = 256个变量
    # 维度 4x4 (A0 B0)
    # 为减少开销，我们将Eve的猜测 e 分解为 (ea, eb) 对应 Alice 和 Bob 的猜测
    # e_idx = 4 * ea + eb
    
    # 定义变量列表 (使用字典方便索引)
    M_vars = {}
    for a in range(4):
        for b in range(4):
            for e in range(16):
                M_vars[(a, b, e)] = cp.Variable((4, 4), hermitian=True)
                
    # 辅助变量用于 No-Signaling (NS) 约束
    # R[b,e] 对应 Bob 端 marginal (sum over a) -> dim 2x2 (作用在 B0)
    # L[a,e] 对应 Alice 端 marginal (sum over b) -> dim 2x2 (作用在 A0)
    # 加上系数 p(e) 用于归一化
    
    R_vars = {} # 对应论文 M_{b,e}^{B0}
    L_vars = {} # 对应论文 M_{a,e}^{A0}
    p_e = cp.Variable(16, nonneg=True) # p(e)
    
    for e in range(16):
        for b in range(4):
            R_vars[(b, e)] = cp.Variable((2, 2), hermitian=True)
        for a in range(4):
            L_vars[(a, e)] = cp.Variable((2, 2), hermitian=True)
            
    constraints = []
    
    # PSD 约束
    for key in M_vars: constraints.append(M_vars[key] >> 0)
    for key in R_vars: constraints.append(R_vars[key] >> 0)
    for key in L_vars: constraints.append(L_vars[key] >> 0)
    
    # 1. 观测一致性约束: Sum_e M_{a,b,e} == M_{a,b}^{obs}
    for a in range(4):
        for b in range(4):
            constraints.append(
                cp.sum([M_vars[(a, b, e)] for e in range(16)]) == M_obs[(a, b)]
            )
            
    # 2. No-Signaling 约束
    I2 = np.eye(2)
    
    for e in range(16):
        # Alice marginal (Sum over b) -> L_a,e (x) I
        for a in range(4):
            sum_over_b = cp.sum([M_vars[(a, b, e)] for b in range(4)])
            constraints.append(sum_over_b == cp.kron(L_vars[(a, e)], I2))
            
        # Bob marginal (Sum over a) -> I (x) R_b,e
        for b in range(4):
            sum_over_a = cp.sum([M_vars[(a, b, e)] for a in range(4)])
            constraints.append(sum_over_a == cp.kron(I2, R_vars[(b, e)]))
            
    # 3. 局部归一化约束
    for e in range(16):
        # Sum_a L_{a,e} == p(e) * I
        constraints.append(cp.sum([L_vars[(a, e)] for a in range(4)]) == p_e[e] * I2)
        # Sum_b R_{b,e} == p(e) * I
        constraints.append(cp.sum([R_vars[(b, e)] for b in range(4)]) == p_e[e] * I2)
        
    # 4. 概率归一化
    constraints.append(cp.sum(p_e) == 1)
    
    # 目标函数: 最大化猜对概率
    # Inputs: x*, y*. 选择 x*=0, y*=0 (由于对称性，任意对皆可)
    inputs = get_tetrahedron_states()
    rho_in = np.kron(inputs[0], inputs[0]) # psi_0 (x) psi_0
    
    obj_expr = 0
    for a in range(4):
        for b in range(4):
            # Eve 猜对意味着 e 对应的索引等于 (a,b)
            e_correct = 4 * a + b
            obj_expr += cp.real(cp.trace(M_vars[(a, b, e_correct)] @ rho_in))
            
    prob = cp.Problem(cp.Maximize(obj_expr), constraints)
    
    try:
        # 使用 MOSEK 求解，精度更高
        prob.solve(solver=cp.MOSEK)
    except:
        prob.solve(solver=cp.SCS, eps=1e-4)
        
    guessing_prob = prob.value
    
    # 处理数值误差 (如果概率极小或负)
    if guessing_prob is None or guessing_prob <= 1e-9:
        return 0.0 # 异常情况，防止log报错
    
    # H_min = -log2(P_guess)
    return -np.log2(guessing_prob)

# --- 3. 主程序 ---

def main():
    print("Reproducing Figure 1 (Corrected)...")
    
    # 根据论文图表，w 从 0 到 1
    w_list = np.linspace(0, 1, 15)
    h_min_list = []
    
    print(f"{'w':<10} | {'H_min (bits)':<15}")
    print("-" * 30)
    
    for w in w_list:
        h_val = solve_randomness_sdp(w)
        h_min_list.append(h_val)
        print(f"{w:<10.2f} | {h_val:<15.4f}")
        
    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(w_list, h_min_list, 'b-', linewidth=2, label='Tetrahedron Inputs')
    plt.xlabel('Noise parameter $w$', fontsize=12)
    plt.ylabel('Min-entropy $H_{\\min}$ (bits)', fontsize=12)
    plt.title('Randomness vs Noise (MDI Scenario)', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 4.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()