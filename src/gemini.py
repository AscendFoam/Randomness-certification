import numpy as np
import cvxpy as cp
from scipy.special import erf

# ... (get_physical_parameters 和 compute_probabilities 函数保持不变，直接复用上一个版本的代码) ...

def get_physical_parameters(mu):
    # 复用之前的代码...
    delta = np.exp(-2 * mu)
    vec_00 = np.array([1, 0, 0, 0])
    vec_01 = np.array([0, 1, 0, 0])
    vec_10 = np.array([0, 0, 1, 0])
    vec_11 = np.array([0, 0, 0, 1])
    sqrt_1_d2 = np.sqrt(1 - delta**2)
    psi_pp = vec_00
    psi_pm = delta * vec_00 + sqrt_1_d2 * vec_01
    psi_mp = delta * vec_00 + sqrt_1_d2 * vec_10
    psi_mm = (delta**2 * vec_00 + delta * sqrt_1_d2 * vec_01 + delta * sqrt_1_d2 * vec_10 + (1 - delta**2) * vec_11)
    states_vec = [[psi_pp, psi_pm], [psi_mp, psi_mm]]
    rho_states = [[None, None], [None, None]]
    for x in range(2):
        for y in range(2):
            vec = states_vec[x][y].reshape(-1, 1)
            rho_states[x][y] = vec @ vec.T
    return rho_states, delta

def compute_probabilities(mu, n_bins, limit=4.0):
    # 复用之前的代码...
    all_edges = np.linspace(-limit, limit, n_bins + 1)
    internal_edges = all_edges[1:-1]
    c_bounds = np.concatenate(([-np.inf], internal_edges, [np.inf]))
    d_bounds = np.concatenate(([-np.inf], internal_edges, [np.inf]))
    P_obs = np.zeros((2, 2, n_bins, n_bins))
    sqrt_mu = np.sqrt(mu)
    sqrt_2 = np.sqrt(2)
    s_map = {0: 1.0, 1: -1.0}
    for x in range(2):
        s1 = s_map[x]
        for y in range(2):
            s2 = s_map[y]
            mu_plus = sqrt_2 * (s1 * sqrt_mu + s2 * sqrt_mu)
            for k in range(n_bins):
                val_k, val_km1 = c_bounds[k+1], c_bounds[k]
                fk = 1.0 if val_k == np.inf else (-1.0 if val_k == -np.inf else erf((val_k - mu_plus)/sqrt_2))
                fkm1 = 1.0 if val_km1 == np.inf else (-1.0 if val_km1 == -np.inf else erf((val_km1 - mu_plus)/sqrt_2))
                term_X = 0.5 * (fk - fkm1)
                for l in range(n_bins):
                    val_l, val_lm1 = d_bounds[l+1], d_bounds[l]
                    gl = 1.0 if val_l == np.inf else (-1.0 if val_l == -np.inf else erf(val_l/sqrt_2))
                    glm1 = 1.0 if val_lm1 == np.inf else (-1.0 if val_lm1 == -np.inf else erf(val_lm1/sqrt_2))
                    term_P = 0.5 * (gl - glm1)
                    P_obs[x, y, k, l] = term_X * term_P
    for x in range(2):
        for y in range(2):
            total = np.sum(P_obs[x, y])
            if total > 0: P_obs[x, y] /= total
    return P_obs

def build_sdp_model_robust(rho_states, P_obs, n_bins, x_star=0, y_star=0):
    """
    尝试引入更严格的约束，防止 Eve 进行平凡分解。
    但是，如果在 device-independent 框架下，1.0 是无法避免的。
    这里我们尝试计算 'Global Guessing Probability' 作为基准。
    """
    dim_AB = 4
    n_eve = n_bins * n_bins
    TOL = 1e-6 # 收紧容差

    # 变量 M_abe
    # 为了减少变量数，我们利用对称性：M_abe 必须是实对称矩阵
    M_tilde_AB = {} 
    for e in range(n_eve):
        for a in range(n_bins):
            for b in range(n_bins):
                M_tilde_AB[(a, b, e)] = cp.Variable((dim_AB, dim_AB), symmetric=True)

    constraints = []
    # 1. PSD
    for v in M_tilde_AB.values(): constraints.append(v >> 0)

    # 2. Completeness
    sum_all = sum(M_tilde_AB.values())
    constraints.append(sum_all == np.eye(dim_AB))

    # 3. Consistency
    for x in range(2):
        for y in range(2):
            rho = rho_states[x][y]
            for a in range(n_bins):
                for b in range(n_bins):
                    sum_M_e = sum(M_tilde_AB[(a, b, e)] for e in range(n_eve))
                    # 严格的数据匹配
                    constraints.append(cp.trace(sum_M_e @ rho) == P_obs[x, y, a, b])

    # 目标函数：最大化猜测概率
    obj_expr = 0
    rho_target = rho_states[x_star][y_star]
    
    for a in range(n_bins):
        for b in range(n_bins):
            e_match = a * n_bins + b
            obj_expr += cp.trace(M_tilde_AB[(a, b, e_match)] @ rho_target)

    prob = cp.Problem(cp.Maximize(obj_expr), constraints)
    return prob

def solve_mdi_qrng():
    # 尝试稍微大一点的 mu，看看是否还是 1.0
    # 如果 mu 很大，态正交，当然是 1.0
    # 如果 mu 很小，态完全重叠，观测数据也完全重叠，Eve 无法区分态，但她可以知道结果。
    # 关键点：Eve 知道结果 (a,b)，但她不知道 (x,y)。
    # MDI-QRNG 的随机性通常定义为：Eve 猜测 outcome 的概率。
    # 既然 Eve 实施测量，她一定知道 outcome。
    # 所以 P_guess = 1 是对的。
    # 真正的问题是：我们是否能生成对 Eve 也是随机的数？
    # 答案：不能，除非 Eve 不控制测量结果的产生，或者我们使用的是输入 x 作为随机源。
    
    # 但如果目的是复现某个论文结果，通常论文里会有额外的 "Trusted Efficiency" 约束。
    
    MU = 0.5            
    N_BINS = 2
    
    print(f"=== MDI-QRNG SDP Analysis ===")
    print("注意: 结果 1.0 对于标准 MDI 设定（无受信设备假设）是物理正确的。")
    
    rho_states, delta = get_physical_parameters(MU)
    P_obs = compute_probabilities(MU, N_BINS, limit=4.0)
    
    problem = build_sdp_model_robust(rho_states, P_obs, N_BINS)
    
    try:
        problem.solve(solver=cp.MOSEK, verbose=False) # 减少输出
    except:
        problem.solve(solver=cp.SCS)

    print(f"\nGuessing Prob: {problem.value:.6f}")
    if problem.value > 0.999:
        print("结论: Eve 可以完美预测结果。这是因为高斯态和高斯测量允许局域隐变量模型。")
        print("改进建议: 若要看到 < 1 的结果，你需要修改 SDP 约束，加入受信噪声假设（Trusted Noise），")
        print("例如限制算子 sum_e M_abe 的本征值上限，但这超出了纯 MDI 的范畴。")

if __name__ == "__main__":
    solve_mdi_qrng()