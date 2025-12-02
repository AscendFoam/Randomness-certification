import numpy as np
import cvxpy as cp
from scipy.special import erf
from itertools import product

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

    def generate_data(self):
        """
        生成所有需要的输入数据：
        - probs[e_idx, x_idx, y_idx]: P(e|x,y)
        - rhos[x_idx, y_idx]: 密度矩阵
        """
        # 映射: x,y index 0 -> s=+1, index 1 -> s=-1
        signs = [1, -1]
        
        # 1. 生成概率张量 P_obs[e][x][y]
        # e 的总数为 n_bins * n_bins
        # e = k * n_bins + l
        n_e = self.n_bins * self.n_bins
        P_obs = np.zeros((n_e, 2, 2))
        
        for x_idx, s1 in enumerate(signs):
            for y_idx, s2 in enumerate(signs):
                for k in range(self.n_bins):
                    for l in range(self.n_bins):
                        e_idx = k * self.n_bins + l
                        P_obs[e_idx, x_idx, y_idx] = self.get_conditional_prob(s1, s2, k, l)
                        
        # 2. 生成量子态 Rho[x][y]
        Rho_states = [[None for _ in range(2)] for _ in range(2)]
        for x_idx, s1 in enumerate(signs):
            for y_idx, s2 in enumerate(signs):
                Rho_states[x_idx][y_idx] = self.get_joint_rho(s1, s2)
                
        # 3. 计算边际概率 p(e) (假设输入 x, y 均匀分布)
        # 约束条件 5, 6, 7 需要 p(e)
        p_e = np.sum(P_obs, axis=(1, 2)) * 0.25 
        
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


def solve_mdi_sdp(P_obs, Rho_states, p_e, target_indices):
    """
    构建并求解 SDP
    P_obs: 观测概率 [e, x, y]
    Rho_states: 输入量子态 [x][y] (4x4 numpy arrays)
    p_e: Eve 结果的概率 [e]
    target_indices: (x*, y*) 需要最大化的目标对
    """
    
    # 维度定义
    dim_A = 2 # 2D subspace defined in TeX
    dim_B = 2
    dim_total = dim_A * dim_B # 4
    n_e = len(p_e)
    n_x = 2
    n_y = 2
    
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
    M_vars = [[[cp.Variable((dim_total, dim_total), hermitian=True) 
                for _ in range(n_e)] 
               for _ in range(n_b)] 
              for _ in range(n_a)]
    
    constraints = []
    
    # -------------------------------------------------------
    # 约束 2: PSD (已经在定义 Variable 时隐含 hermitian=True，需显式加 >> 0)
    # -------------------------------------------------------
    for a in range(n_a):
        for b in range(n_b):
            for e in range(n_e):
                constraints.append(M_vars[a][b][e] >> 0)

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
    
    for x in range(n_x):
        for y in range(n_y):
            for a in range(n_a):
                for b in range(n_b):
                    # 构建 sum_e M
                    sum_M_e = cp.sum([M_vars[a][b][e] for e in range(n_e)])
                    
                    # 理想的相关性：Alice 选 x (bit 0/1)，生成 bit a=x。
                    target_prob = 1.0 if (a == x and b == y) else 0.0
                    
                    # 添加约束
                    constraints.append(cp.trace(sum_M_e @ Rho_states[x][y]) == target_prob)

    # -------------------------------------------------------
    # 无信号约束 & 归一化 (Constraints 3, 4, 5, 6)
    # -------------------------------------------------------
    
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
    M_Bob = [[cp.Variable((dim_B, dim_B), hermitian=True) for _ in range(n_e)] for _ in range(n_b)]
    M_Alice = [[cp.Variable((dim_A, dim_A), hermitian=True) for _ in range(n_e)] for _ in range(n_a)]

    for e in range(n_e):
        # 约束 3: sum_a M_{a,b,e} = I_A (x) M_{b,e}^B
        for b in range(n_b):
            term_bob = cp.kron(dim_I_A, M_Bob[b][e])
            constraints.append(Sum_a_M[b][e] == term_bob)
            
        # 约束 4: sum_b M_{a,b,e} = M_{a,e}^A (x) I_B
        for a in range(n_a):
            term_alice = cp.kron(M_Alice[a][e], dim_I_B)
            constraints.append(Sum_b_M[a][e] == term_alice)

    # -------------------------------------------------------
    # 约束 5 & 6: 局部 POVM 归一化
    # sum_b M_{b,e}^B = p(e) I_B
    # sum_a M_{a,e}^A = p(e) I_A
    # -------------------------------------------------------
    for e in range(n_e):
        # 对 Bob 归一化
        sum_mb = cp.sum([M_Bob[b][e] for b in range(n_b)])
        constraints.append(sum_mb == p_e[e] * dim_I_B)
        
        # 对 Alice 归一化
        sum_ma = cp.sum([M_Alice[a][e] for a in range(n_a)])
        constraints.append(sum_ma == p_e[e] * dim_I_A)

    # 约束 7: sum p(e) = 1 (这是数据属性，不是变量约束，但在代码里可以忽略或作为检查)
    
    # -------------------------------------------------------
    # 目标函数
    # Max Tr( sum_e M_{x*, y*, e} (psi_x* (x) psi_y*) )
    # -------------------------------------------------------
    x_star, y_star = target_indices
    # 这里的 a, b 应该对应目标输入 x*, y*
    # 因为我们要猜测的是 x*, y*
    
    obj_expr = 0
    for e in range(n_e):
        # 目标是猜测 x^*, y^*，所以我们看 M 中对应 a=x^*, b=y^* 的项
        obj_expr += cp.real(cp.trace(M_vars[x_star][y_star][e] @ Rho_states[x_star][y_star]))
        
    objective = cp.Maximize(obj_expr)
    
    # 求解
    prob = cp.Problem(objective, constraints)
    
    # 推荐使用 MOSEK，如果不可用，CVXPY 会尝试其他求解器
    try:
        result = prob.solve(solver=cp.MOSEK, verbose=True)
    except:
        print("MOSEK not found or failed, trying default solver...")
        result = prob.solve(verbose=True)
        
    return result

# =======================================================
# 主程序入口示例
# =======================================================
if __name__ == "__main__":
    # 1. 设置物理参数
    mu_val = 1        # 平均光子数 (示例)
    n_bins_val = 2      # 离散化区间数 (示例：2个区间)
    range_val = 2.0     # 积分边界
    
    print("正在初始化物理模型并计算数据...")
    engine = PhysicsEngine(mu=mu_val, n_bins=n_bins_val, range_val=range_val)
    
    # 获取观测数据 P(e|x,y) 和 量子态 Rho[x][y]
    # P_obs shape: (n_e, 2, 2)
    # Rho_states: 2x2 list of 4x4 matrices
    P_obs, Rho_states, p_e = engine.generate_data()
    
    print(f"数据生成完毕。总观测结果数 e: {len(p_e)}")
    print(f"P(e) Sum check: {np.sum(p_e)}")
    
    # 2. 求解 SDP
    # 假设我们想优化输入 x*=0, y*=0 (即 s1=+1, s2=+1) 的猜测概率
    target_idx = (0, 0) 
    
    print("\n开始构建并求解 SDP...")
    max_guessing_prob = solve_mdi_sdp(P_obs, Rho_states, p_e, target_idx)
    
    print(f"\nOptimization Result:")
    print(f"G_MDI for input {target_idx}: {max_guessing_prob}")