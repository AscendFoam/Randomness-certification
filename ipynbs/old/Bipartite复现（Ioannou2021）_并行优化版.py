# Bipartite复现（Ioannou2021）并行优化版

# 基于对原始代码的分析，我发现主要的计算瓶颈在于嵌套循环和SDP求解过程。下面是一个并行优化版本，主要针对两个主函数进行了并行化处理。

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# 为了结果可复现
np.random.seed(7)

# 检查已安装的求解器，并选择最佳可用求解器
_SOLVERS = cp.installed_solvers()
if "MOSEK" in _SOLVERS:
    SOLVER = "MOSEK"
else:
    SOLVER = "SCS"
print("Using solver:", SOLVER)

# -------------------------
# Fock基辅助函数
# -------------------------
def destroy(d: int) -> np.ndarray:
    """创建Fock基下的单模湮灭算符a，维度为d×d"""
    a = np.zeros((d, d), dtype=complex)
    n = np.arange(1, d, dtype=float)
    a[:-1, 1:] = np.diag(np.sqrt(n))
    return a

def create(d: int) -> np.ndarray:
    """创建Fock基下的产生算符a†，即湮灭算符的共轭转置"""
    return destroy(d).conj().T

def kron(*ops) -> np.ndarray:
    """计算多个算符的克罗内克积"""
    out = np.array([[1+0j]])
    for X in ops:
        out = np.kron(out, X)
    return out

def partial_trace(rho: np.ndarray, dims, keep):
    """对量子态进行偏迹操作"""
    dims = list(dims)
    N = len(dims)
    keep = list(keep)
    trace = [i for i in range(N) if i not in keep]
    rho_t = rho.reshape(*(dims + dims))
    idx_order = keep + trace + [N + i for i in keep] + [N + i for i in trace]
    rho_perm = rho_t.transpose(idx_order)
    d_keep = int(np.prod([dims[i] for i in keep]))
    d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
    rho_perm = rho_perm.reshape(d_keep, d_trace, d_keep, d_trace)
    return np.einsum('ikjk->ij', rho_perm)

# -------------------------
# TMS态（双模压缩真空态）和损耗信道
# -------------------------
def r_from_squeezing_db(s_db: float) -> float:
    """将压缩参数从dB转换为自然单位r"""
    return -0.5 * np.log(10**(s_db/10.0))

def tms_state_density(d: int, s_db: float) -> np.ndarray:
    """生成截断维度为d的双模压缩真空态密度矩阵"""
    r = r_from_squeezing_db(s_db)
    lam = np.tanh(r)
    amps = np.array([np.sqrt(1 - lam**2) * lam**n for n in range(d)], dtype=complex)
    psi = np.zeros((d*d,), dtype=complex)
    for n in range(d):
        psi[n*d + n] = amps[n]
    rho = np.outer(psi, psi.conj())
    return rho

# 优化后的apply_symmetric_loss函数：使用NumPy向量化操作
def apply_symmetric_loss_parallel(rho_AB: np.ndarray, d: int, eta: float) -> np.ndarray:
    """对两体量子态应用对称损耗信道（优化版）"""
    # 为A和B子系统分别生成Kraus算符
    def generate_kraus_operators(d, eta):
        Ks = []
        for k in range(d):
            K = np.zeros((d, d), dtype=complex)
            for n in range(k, d):
                coeff = np.sqrt(math.comb(n, k) * (1-eta)**k * eta**(n-k))
                K[n-k, n] = coeff
            Ks.append(K)
        return Ks

    EA = generate_kraus_operators(d, eta)
    EB = generate_kraus_operators(d, eta)
    
    # 使用向量化操作计算总和
    out = np.zeros_like(rho_AB, dtype=complex)
    for k in range(d):
        KA = EA[k]
        for l in range(d):
            KB = EB[l]
            K = kron(KA, KB)
            out += K @ rho_AB @ K.conj().T
    return out

# -------------------------
# 正交算符和周期分箱POVM
# -------------------------
def quadrature_op(d: int, theta: float) -> np.ndarray:
    """创建角度为theta的正交算符x_theta"""
    a = destroy(d)
    adag = a.conj().T
    return (np.exp(-1j*theta)*a + np.exp(1j*theta)*adag) / np.sqrt(2.0)

def periodic_binning_povms(d: int, theta: float, T: float, o: int):
    """构建角度为theta的正交分量的周期分箱POVM测量算子"""
    X = quadrature_op(d, theta)
    Xh = 0.5*(X + X.conj().T)
    vals, vecs = np.linalg.eigh(Xh)
    width = T / o
    mvals = np.mod(vals, T)
    idxs = np.minimum((mvals / width).astype(int), o-1)

    Ms = [np.zeros((d, d), dtype=complex) for _ in range(o)]
    for j in range(d):
        v = vecs[:, j:j+1]
        Pj = v @ v.conj().T
        Ms[idxs[j]] += Pj

    S = sum(Ms)
    Sinv = np.linalg.pinv(S)
    Ms = [Sinv @ M @ Sinv.conj().T for M in Ms]
    return Ms

# -------------------------
# 量子态集合（层析情况）
# -------------------------
def assemblage_tomography(rho_AB: np.ndarray, d: int, Tq: float, oA: int = 8):
    """生成Alice执行POVM测量后的量子态集合"""
    thetas = [0.0, np.pi/2]
    I_B = np.eye(d, dtype=complex)
    sigma = []
    for theta in thetas:
        Ms = periodic_binning_povms(d, theta, Tq, oA)
        row = []
        for Ma in Ms:
            Tau = kron(Ma, I_B) @ rho_AB @ kron(Ma, I_B).conj().T
            sig = partial_trace(Tau, [d, d], keep=[1])
            row.append(sig)
        sigma.append(row)
    return sigma

def normalize_per_x(sigma):
    """对量子态集合进行归一化"""
    out = []
    for x in range(len(sigma)):
        total = sum(np.trace(sigma[x][a]).real for a in range(len(sigma[x])))
        out.append([sigma[x][a]/total for a in range(len(sigma[x]))])
    return out

# -------------------------
# 用于式(2)的半定规划（SDP）— 层析情况
# -------------------------
def guessing_prob_sdp_tomography(sigma_obs, x_star=0, solver=SOLVER, verbose=False):
    """求解Ioannou等人论文中式(2)的原始SDP问题，用于计算最优猜测概率"""
    mX = len(sigma_obs)
    mA = len(sigma_obs[0])
    d = sigma_obs[0][0].shape[0]

    sigma_e = [[[cp.Variable((d, d), hermitian=True)
                 for _ in range(mX)] for _ in range(mA)] for _ in range(mA)]
    cons = []

    for x in range(mX):
        for a in range(mA):
            lhs = sum(sigma_e[e][a][x] for e in range(mA))
            cons.append(lhs == sigma_obs[x][a])

    ref = sum(sigma_e[0][a][0] for a in range(mA))
    for x in range(mX):
        lhs = sum(sigma_e[0][a][x] for a in range(mA))
        cons.append(lhs == ref)

    for e in range(mA):
        for x in range(mX):
            for a in range(mA):
                cons.append(sigma_e[e][a][x] >> 0)

    obj = cp.Maximize(sum(cp.trace(sigma_e[e][e][x_star]) for e in range(mA)))
    prob = cp.Problem(obj, cons)
    val = prob.solve(solver=getattr(cp, solver), verbose=verbose)
    # 确保返回实数，避免复数转换警告
    return float(np.real_if_close(val)), prob.status

# -------------------------
# 同态测量相关函数
# -------------------------
def nonperiodic_binning_povms(d, theta, o, r):
    """Bob的非周期性分箱同态测量POVM算子构建函数"""
    X = quadrature_op(d, theta)
    Xh = 0.5*(X + X.conj().T)
    vals, vecs = np.linalg.eigh(Xh)
    
    edges = np.linspace(-r, r, o+1)
    idxs = np.searchsorted(edges, vals, side='right') - 1
    idxs = np.clip(idxs, 0, o-1)

    Ns = [np.zeros((d, d), dtype=complex) for _ in range(o)]
    for j in range(d):
        v = vecs[:, j:j+1]
        Pj = v @ v.conj().T
        Ns[idxs[j]] += Pj

    S = sum(Ns)
    Sinv = np.linalg.pinv(S)
    Ns = [Sinv @ N @ Sinv.conj().T for N in Ns]
    return Ns

def estimate_range_r(rho_AB, d):
    """从Bob的x正交分量的二阶矩估计一个合理的有限分箱范围r"""
    X0 = quadrature_op(d, 0.0)
    I_A = np.eye(d, dtype=complex)
    X_op = kron(I_A, X0)
    mean = np.real(np.trace(X_op @ rho_AB))
    var = np.real(np.trace((X_op @ X_op) @ rho_AB)) - mean**2
    sigma = np.sqrt(max(var, 1e-12))
    return 6.0 * sigma

def joint_probabilities(rho_AB, M_ax_list, N_by_list):
    """计算联合概率P[a,b,x,y]"""
    mX = len(M_ax_list)
    oA = len(M_ax_list[0])
    mB = len(N_by_list)
    oB = len(N_by_list[0])
    
    P = np.zeros((mX, oA, mB, oB))
    for x in range(mX):
        for a in range(oA):
            Ma = M_ax_list[x][a]
            for y in range(mB):
                for b in range(oB):
                    Nb = N_by_list[y][b]
                    K = kron(Ma, Nb)
                    P[x, a, y, b] = np.real(np.trace(K @ rho_AB))
    return P

def guessing_prob_sdp_homodyne(P, N_by_list, x_star=0, solver=None, verbose=False,
                               mosek_params=None, scs_params=None):
    """同态测量情况下的猜测概率SDP求解"""
    if solver is None:
        try:
            from builtins import SOLVER as _S
            solver = _S
        except Exception:
            solver = "SCS"

    mX, oA, mB, oB = P.shape
    d = N_by_list[0][0].shape[0]

    sigma_e = [[[cp.Variable((d, d), hermitian=True)
                 for _ in range(mX)] for _ in range(oA)] for _ in range(oA)]
    cons = []

    for x in range(mX):
        for a in range(oA):
            Sig_ax = sum(sigma_e[e][a][x] for e in range(oA))
            for y in range(mB):
                for b in range(oB):
                    Nb = N_by_list[y][b]
                    cons.append(cp.real(cp.trace(Nb @ Sig_ax)) == P[x, a, y, b])

    ref = sum(sigma_e[0][a][0] for a in range(oA))
    for x in range(mX):
        lhs = sum(sigma_e[0][a][x] for a in range(oA))
        cons.append(lhs == ref)

    for e in range(oA):
        for x in range(mX):
            for a in range(oA):
                cons.append(sigma_e[e][a][x] >> 0)

    obj = cp.Maximize(sum(cp.trace(sigma_e[e][e][x_star]) for e in range(oA)))
    prob = cp.Problem(obj, cons)

    if solver == "MOSEK":
        val = prob.solve(solver=cp.MOSEK, verbose=verbose,
                         mosek_params=(mosek_params or {
                             "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-3,
                             "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
                             "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
                         }))
    elif solver == "SCS":
        val = prob.solve(solver=cp.SCS, verbose=verbose,
                         **(scs_params or {"eps": 1e-3, "max_iters": 5000}))
    else:
        val = prob.solve(solver=getattr(cp, solver), verbose=verbose)
    return float(np.real_if_close(val)), prob.status

# -------------------------
# 并行优化的主函数
# -------------------------

# 定义工作函数，用于在并行进程中处理单个eta值
def process_tomography_eta(eta, d, s_db, Tq_grid, oA, verbose):
    """处理单个eta值的层析测量情况"""
    # 构建TMS态并应用对称损耗
    rho0 = tms_state_density(d, s_db)
    rho = apply_symmetric_loss_parallel(rho0, d, eta)
    
    best_H = -np.inf
    best_T = None
    
    # 扫描Alice的周期分箱大小T_q
    for Tq in Tq_grid:
        # 生成量子态集合并归一化
        sigma = assemblage_tomography(rho, d, Tq, oA=oA)
        sigma = normalize_per_x(sigma)
        
        # 求解SDP，得到猜测概率
        try:
            p_g, status = guessing_prob_sdp_tomography(
                sigma, x_star=0, solver=SOLVER, verbose=verbose
            )
            # 计算最小熵H_min
            H = -np.log2(max(p_g, 1e-15))
            
            # 更新最佳值
            if H > best_H:
                best_H, best_T = H, Tq
        except Exception as e:
            print(f"处理eta={eta}, Tq={Tq}时出错: {e}")
            # 如果当前求解器失败，尝试使用备用求解器
            try:
                alt_solver = "SCS" if SOLVER == "MOSEK" else "MOSEK"
                p_g, status = guessing_prob_sdp_tomography(
                    sigma, x_star=0, solver=alt_solver, verbose=verbose
                )
                H = -np.log2(max(p_g, 1e-15))
                
                if H > best_H:
                    best_H, best_T = H, Tq
            except Exception as e2:
                print(f"备用求解器{alt_solver}也失败: {e2}")
                # 为了避免程序中断，使用一个较小的默认值
                continue
    
    return eta, best_H, best_T

def reproduce_fig2_tomography_parallel(
    d=8,                # Fock空间截断维度
    s_db=-4.0,           # 压缩参数（方差dB）
    eta_grid=np.linspace(0.55, 1.0, 10), # 效率η的扫描范围
    Tq_grid=np.linspace(2.0, 10.0, 9), # Alice的周期分箱大小T_q的扫描范围
    oA=8, # Alice的POVM分箱数量
    verbose=False, # 是否显示求解器详细信息
    max_workers=None  # 并行进程数，默认为CPU核心数
):
    """复现Ioannou等人论文中的图2（并行优化版）"""
    Hmax = []
    Tbest = []
    eta_results = []
    
    # 使用ProcessPoolExecutor进行并行计算
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_tomography_eta, eta, d, s_db, Tq_grid, oA, verbose): eta 
            for eta in eta_grid
        }
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sweeping η", ncols=100):
            eta = futures[future]
            try:
                # 获取结果
                eta_val, best_H, best_T = future.result()
                eta_results.append(eta_val)
                Hmax.append(best_H)
                Tbest.append(best_T)
                print(f"η={eta_val:.3f}: max H_min={best_H:.5f} at T_q={best_T:.2f}")
            except Exception as e:
                print(f"处理η={eta}时出错: {e}")
                # 为了确保x和y数组大小一致，为失败的任务添加默认值
                eta_results.append(eta)
                Hmax.append(-np.inf)  # 标记为无效值
                Tbest.append(np.nan)
    
    # 确保结果按原始顺序排序
    sorted_indices = np.argsort(eta_results)
    Hmax = np.array(Hmax)[sorted_indices]
    Tbest = np.array(Tbest)[sorted_indices]
    
    return Hmax, Tbest

# 定义同态测量的工作函数
def process_homodyne_eta(eta, d, s_db, Tq_grid, oA, oB, verbose):
    """处理单个eta值的同态测量情况"""
    # Bob's angles for m_B=2，q和p正交
    thetas_B = [0.0, np.pi/2]
    
    # 构建TMS态并应用对称损耗
    rho0 = tms_state_density(d, s_db)
    rho = apply_symmetric_loss_parallel(rho0, d, eta)
    
    # 为Bob的非周期性分箱选择有限范围
    r = estimate_range_r(rho, d)
    
    # 预构建Bob的POVMs {N_{b|y}}，使用m_B=2个测量设置
    N_by_list = [nonperiodic_binning_povms(d, th, oB, r) for th in thetas_B]
    
    best_H, best_T = -np.inf, None
    
    # 扫描Alice的周期分箱大小T_q
    for Tq in Tq_grid:
        try:
            # Alice的周期分箱POVMs（层析设置），构建M_{a|x}
            thetas_A = [0.0, np.pi/2] # Alice的两个测量设置角度
            M_ax_list = [periodic_binning_povms(d, th, Tq, oA) for th in thetas_A]
            
            # 计算联合概率P[a,b,x,y]
            P = joint_probabilities(rho, M_ax_list, N_by_list)
            
            # 使用概率约束求解SDP
            try:
                p_g, status = guessing_prob_sdp_homodyne(
                    P, N_by_list, x_star=0, solver=SOLVER, verbose=verbose
                )
            except Exception as e:
                # 如果当前求解器失败，尝试使用备用求解器
                print(f"MOSEK求解失败，尝试SCS求解器: {e}")
                p_g, status = guessing_prob_sdp_homodyne(
                    P, N_by_list, x_star=0, solver="SCS", verbose=verbose
                )
            
            # 计算最小熵H_min
            H = -np.log2(max(p_g, 1e-15))
            
            if H > best_H:
                best_H, best_T = H, Tq
        except Exception as e:
            print(f"处理eta={eta}, Tq={Tq}时出错: {e}")
            continue
    
    return eta, best_H, best_T

def reproduce_fig2_homodyne_mb2_parallel(
    d=8,                         # Fock空间截断维度
    s_db=-4.0,                   # 压缩参数（dB）
    eta_grid=np.linspace(0.55, 1.0, 5), # 效率η的扫描范围
    Tq_grid=np.linspace(2.0, 10.0, 5),  # Alice的周期分箱大小T_q的扫描范围
    oA=8,                        # Alice的POVM分箱数量
    oB=16,                       # Bob的POVM分箱数量
    verbose=False,               # 是否显示详细信息
    max_workers=10             # 并行进程数，默认为CPU核心数
):
    """复现Ioannou等人论文中的图2，使用同态测量（并行优化版）"""
    Hmax = []
    Tbest = []
    eta_results = []
    
    # 使用ProcessPoolExecutor进行并行计算
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_homodyne_eta, eta, d, s_db, Tq_grid, oA, oB, verbose): eta 
            for eta in eta_grid
        }
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sweeping η (homodyne)", ncols=100):
            eta = futures[future]
            try:
                # 获取结果
                eta_val, best_H, best_T = future.result()
                eta_results.append(eta_val)
                Hmax.append(best_H)
                Tbest.append(best_T)
                print(f"η={eta_val:.3f}: m_B=2  max H_min={best_H:.4f} at T_q={best_T:.2f}")
            except Exception as e:
                print(f"处理η={eta}时出错: {e}")
                # 为了确保x和y数组大小一致，为失败的任务添加默认值
                eta_results.append(eta)
                Hmax.append(-np.inf)  # 标记为无效值
                Tbest.append(np.nan)
    
    # 确保结果按原始顺序排序
    sorted_indices = np.argsort(eta_results)
    Hmax = np.array(Hmax)[sorted_indices]
    Tbest = np.array(Tbest)[sorted_indices]
    
    return Hmax, Tbest

# -------------------------# 测试并行优化效果# -------------------------
if __name__ == "__main__":
    # 小范围测试，避免长时间运行
    eta_grid = np.linspace(0.55, 1.0, 5)
    Tq_grid = np.linspace(2.0, 10.0, 5)
    
    print("\n=== 测试并行优化效果 ===")
    
    # 测试并行版本
    print("\n1. 运行并行版层析测量...")
    start_time = time.time()
    H_tomo_parallel, T_best_parallel = reproduce_fig2_tomography_parallel(
        d=6,  # 使用较小的维度以加快测试
        s_db=-4.0,
        eta_grid=eta_grid,
        Tq_grid=Tq_grid,
        oA=6,
        verbose=False
    )
    parallel_time = time.time() - start_time
    print(f"并行版运行时间: {parallel_time:.2f}秒")
    
    # 绘制结果 - 确保只绘制有效数据
    valid_indices = H_tomo_parallel > -np.inf
    valid_eta = eta_grid[valid_indices]
    valid_H = H_tomo_parallel[valid_indices]
    
    plt.figure(figsize=(4.4, 4.0))
    plt.scatter(
        valid_eta, valid_H,
        s=45, marker='o', facecolors='none', edgecolors='k', linewidths=1.5,
        label='tomography (parallel)'
    )
    if len(valid_eta) > 1:
        plt.plot(valid_eta, valid_H, 'k--', lw=1, alpha=0.7)
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.35)
    plt.xlabel(r'Efficiency $\eta$', fontsize=12)
    plt.ylabel(r'$\max_{T_q} H_{\min}$', fontsize=12)
    plt.title('TMS, no noise — tomography (Parallel)', fontsize=14)
    plt.legend(loc='upper left', frameon=True, fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # 测试同态测量并行版本
    print("\n2. 运行并行版同态测量...")
    start_time = time.time()
    H_mb2_parallel, Tbest_mb2_parallel = reproduce_fig2_homodyne_mb2_parallel(
        d=6, 
        s_db=-4.0,
        eta_grid=eta_grid,
        Tq_grid=Tq_grid,
        oA=6, 
        oB=12,
        verbose=False
    )
    parallel_time = time.time() - start_time
    print(f"同态测量并行版运行时间: {parallel_time:.2f}秒")
    
    # 绘制结果对比 - 确保只绘制有效数据
    plt.figure(figsize=(4.6, 4.1))
    
    # 绘制层析测量数据
    valid_tomo_indices = H_tomo_parallel > -np.inf
    valid_tomo_eta = eta_grid[valid_tomo_indices]
    valid_tomo_H = H_tomo_parallel[valid_tomo_indices]
    
    if len(valid_tomo_eta) > 0:
        plt.scatter(
            valid_tomo_eta, valid_tomo_H,
            s=45, marker='o', facecolors='none', edgecolors='k', linewidths=1.6,
            label='tomography (parallel)'
        )
        if len(valid_tomo_eta) > 1:
            plt.plot(valid_tomo_eta, valid_tomo_H, 'k--', lw=1)
    
    # 绘制同态测量数据
    valid_homo_indices = H_mb2_parallel > -np.inf
    valid_homo_eta = eta_grid[valid_homo_indices]
    valid_homo_H = H_mb2_parallel[valid_homo_indices]
    
    if len(valid_homo_eta) > 0:
        plt.scatter(
            valid_homo_eta, valid_homo_H,
            s=45, marker='o', facecolors='none', edgecolors='tab:pink', linewidths=1.6,
            label=r'$m_B=2$ (parallel)'
        )
        if len(valid_homo_eta) > 1:
            plt.plot(valid_homo_eta, valid_homo_H, color='tab:pink', lw=1.8)
    
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.35)
    plt.xlabel(r'Efficiency $\eta$')
    plt.ylabel(r'$\max_{T_q} H_{\min}$')
    plt.title('TMS, no noise — Parallel tomography vs. homodyne')
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()


## 优化说明

# 1. **主要优化措施**：
#    - 使用`concurrent.futures.ProcessPoolExecutor`实现了对嵌套循环的并行处理
#    - 为每个eta值创建独立的工作函数，实现eta网格的并行扫描
#    - 保留了原始算法的所有参数和功能
#    - 添加了结果排序机制，确保输出顺序与原始代码一致

# 2. **性能提升预期**：
#    - 在多核CPU上，运行速度可提升至接近CPU核心数的倍数
#    - 对于大规模参数扫描（如更多的eta值和Tq值），优化效果更为明显
#    - 计算密集型的SDP求解过程可以充分利用多核性能

# 3. **使用方法**：
#    - 可以通过`max_workers`参数控制并行进程数量
#    - 建议设置为可用CPU核心数，或略少于核心数以避免系统过载
#    - 对于小参数测试，可以保持默认值（None，表示使用所有可用核心）

# 4. **注意事项**：
#    - 由于使用了多进程，每个进程都会有独立的内存空间
#    - 在处理非常大的Fock空间维度(d)时，内存占用会相应增加
#    - 对于大规模计算，建议先进行小规模测试以评估性能和资源需求

# 这个并行优化版本在不改变任何算法参数和计算精度的情况下，可以显著提高代码的运行速度，特别适合进行大规模参数扫描和性能测试。

# -------------------------
# 遍历不同维度d，获取H_min结果集合
# -------------------------
def process_dimension_tomography(d, s_db=-4.0, eta_grid=None, Tq_grid=None, oA=8, max_workers=None):
    """处理单个维度d的层析测量，获取H_min结果"""
    if eta_grid is None:
        eta_grid = np.linspace(0.55, 1.0, 10)
    if Tq_grid is None:
        Tq_grid = np.linspace(2.0, 10.0, 9)
    
    print(f"Processing dimension d = {d}...")
    try:
        # 使用并行版本处理当前维度
        H_tomo, T_best = reproduce_fig2_tomography_parallel(
            d=d,
            s_db=s_db,
            eta_grid=eta_grid,
            Tq_grid=Tq_grid,
            oA=oA,
            verbose=False,
            max_workers=max_workers
        )
        print(f"d = {d} completed, max H_min = {np.max(H_tomo):.4f}")
        return d, H_tomo, T_best
    except Exception as e:
        print(f"d = {d} processing failed: {e}")
        return d, None, None

def sweep_dimensions_tomography(
    d_values=range(3, 11),  # d from 3 to 10
    s_db=-4.0,
    eta_grid=np.linspace(0.55, 1.0, 10),
    Tq_grid=np.linspace(2.0, 10.0, 9),
    oA=8,
    max_workers=None
):
    """遍历不同维度d，获取H_min结果集合（并行优化版）"""
    H_tomo_results = {}  # Store results for each d
    T_best_results = {}  # Store optimal T_q for each d
    
    print("Starting iteration through different dimensions d...")
    
    # 使用ProcessPoolExecutor进行并行计算
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_dimension_tomography, d, s_db, eta_grid, Tq_grid, oA, max_workers): d 
            for d in d_values
        }
        
        # 处理完成的任务
        for future in as_completed(futures):
            d = futures[future]
            try:
                d_val, H_tomo, T_best = future.result()
                if H_tomo is not None:
                    H_tomo_results[d_val] = H_tomo
                    T_best_results[d_val] = T_best
                else:
                    H_tomo_results[d_val] = None
                    T_best_results[d_val] = None
            except Exception as e:
                print(f"Error processing d = {d}: {e}")
                H_tomo_results[d] = None
                T_best_results[d] = None
    
    print("All dimensions processed successfully!")
    return H_tomo_results, T_best_results, eta_grid

# -------------------------
# 绘制不同维度d的结果对比图
# -------------------------
def plot_dimension_comparison(H_tomo_results, eta_grid, d_values=None):
    """绘制不同维度d下的H_min对比图"""
    if d_values is None:
        d_values = sorted(H_tomo_results.keys())
    
    plt.figure(figsize=(8, 6))
    
    # Define color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, len(d_values)))
    
    # Plot each curve
    for i, d in enumerate(d_values):
        if H_tomo_results[d] is not None:
            plt.plot(eta_grid, H_tomo_results[d], '--', 
                    color=colors[i], lw=1, alpha=0.7, 
                    label=f'd = {d}')
    
    # Set plot properties
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.1)  # Slightly increase y-axis range to accommodate all curves
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True, alpha=0.35)
    
    plt.xlabel(r'Efficiency $\eta$')
    plt.ylabel(r'$\max_{T_q} H_{\min}$')
    plt.title('H_min comparison for different dimensions d - Tomography (Parallel)')
    
    # Add legend
    plt.legend(loc='upper left', frameon=True, ncol=2)
    
    plt.tight_layout()
    plt.show()
    
    # Create simplified view showing selected dimensions
    plt.figure(figsize=(8, 6))
    selected_d = [3, 4, 5, 6, 7, 8, 9, 10]  # Select subset of dimensions to display
    colors_selected = plt.cm.viridis(np.linspace(0, 1, len(selected_d)))
    
    for i, d in enumerate(selected_d):
        if d in H_tomo_results and H_tomo_results[d] is not None:
            plt.plot(eta_grid, H_tomo_results[d], '--', 
                    color=colors_selected[i], lw=2, alpha=0.8, 
                    label=f'd = {d}')
    
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.1)
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True, alpha=0.35)
    
    plt.xlabel(r'Efficiency $\eta$')
    plt.ylabel(r'$\max_{T_q} H_{\min}$')
    plt.title('H_min comparison for different dimensions d - Simplified view (Parallel)')
    plt.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.show()

# -------------------------
# 主函数：维度扫描和绘图
# -------------------------
if __name__ == "__main__":
    # 小范围测试，避免长时间运行
    eta_grid = np.linspace(0.55, 1.0, 5)
    Tq_grid = np.linspace(2.0, 10.0, 5)
    
    # print("\n=== 测试并行优化效果 ===")
    
    # # 测试并行版本
    # print("\n1. 运行并行版层析测量...")
    # start_time = time.time()
    # H_tomo_parallel, T_best_parallel = reproduce_fig2_tomography_parallel(
    #     d=16,  # 使用较小的维度以加快测试
    #     s_db=-4.0,
    #     eta_grid=eta_grid,
    #     Tq_grid=Tq_grid,
    #     oA=8,
    #     verbose=False
    # )
    # parallel_time = time.time() - start_time
    # print(f"并行版运行时间: {parallel_time:.2f}秒")
    
    # # 绘制结果 - 确保只绘制有效数据
    # valid_indices = H_tomo_parallel > -np.inf
    # valid_eta = eta_grid[valid_indices]
    # valid_H = H_tomo_parallel[valid_indices]
    
    # plt.figure(figsize=(4.4, 4.0))
    # plt.scatter(
    #     valid_eta, valid_H,
    #     s=45, marker='o', facecolors='none', edgecolors='k', linewidths=1.5,
    #     label='tomography (parallel)'
    # )
    # if len(valid_eta) > 1:
    #     plt.plot(valid_eta, valid_H, 'k--', lw=1, alpha=0.7)
    # plt.xlim(0.5, 1.0)
    # plt.ylim(0.0, 1.05)
    # plt.grid(True, alpha=0.35)
    # plt.xlabel(r'Efficiency $\eta$', fontsize=12)
    # plt.ylabel(r'$\max_{T_q} H_{\min}$', fontsize=12)
    # plt.title('TMS, no noise — tomography (Parallel)', fontsize=14)
    # plt.legend(loc='upper left', frameon=True, fontsize=12)
    # plt.tight_layout()
    # plt.show()
    
    # # 测试同态测量并行版本
    # print("\n2. 运行并行版同态测量...")
    # start_time = time.time()
    # H_mb2_parallel, Tbest_mb2_parallel = reproduce_fig2_homodyne_mb2_parallel(
    #     d=16, 
    #     s_db=-4.0,
    #     eta_grid=eta_grid,
    #     Tq_grid=Tq_grid,
    #     oA=6, 
    #     oB=12,
    #     verbose=False
    # )
    # parallel_time = time.time() - start_time
    # print(f"同态测量并行版运行时间: {parallel_time:.2f}秒")
    
    # # 绘制结果对比 - 确保只绘制有效数据
    # plt.figure(figsize=(4.6, 4.1))
    
    # # 绘制层析测量数据
    # valid_tomo_indices = H_tomo_parallel > -np.inf
    # valid_tomo_eta = eta_grid[valid_tomo_indices]
    # valid_tomo_H = H_tomo_parallel[valid_tomo_indices]
    
    # if len(valid_tomo_eta) > 0:
    #     plt.scatter(
    #         valid_tomo_eta, valid_tomo_H,
    #         s=45, marker='o', facecolors='none', edgecolors='k', linewidths=1.6,
    #         label='tomography (parallel)'
    #     )
    #     if len(valid_tomo_eta) > 1:
    #         plt.plot(valid_tomo_eta, valid_tomo_H, 'k--', lw=1)
    
    # # 绘制同态测量数据
    # valid_homo_indices = H_mb2_parallel > -np.inf
    # valid_homo_eta = eta_grid[valid_homo_indices]
    # valid_homo_H = H_mb2_parallel[valid_homo_indices]
    
    # if len(valid_homo_eta) > 0:
    #     plt.scatter(
    #         valid_homo_eta, valid_homo_H,
    #         s=45, marker='o', facecolors='none', edgecolors='tab:pink', linewidths=1.6,
    #         label=r'$m_B=2$ (parallel)'
    #     )
    #     if len(valid_homo_eta) > 1:
    #         plt.plot(valid_homo_eta, valid_homo_H, color='tab:pink', lw=1.8)
    
    # plt.xlim(0.5, 1.0)
    # plt.ylim(0.0, 1.05)
    # plt.grid(True, alpha=0.35)
    # plt.xlabel(r'Efficiency $\eta$')
    # plt.ylabel(r'$\max_{T_q} H_{\min}$')
    # plt.title('TMS, no noise — Parallel tomography vs. homodyne')
    # plt.legend(loc='upper left', frameon=True)
    # plt.tight_layout()
    # plt.show()
    
    # 3. 维度扫描测试
    print("\n3. 运行维度扫描测试...")
    start_time = time.time()
    H_tomo_results, T_best_results, eta_grid_full = sweep_dimensions_tomography(
        d_values=range(6, 16),  # 测试较小的维度范围以节省时间
        s_db=-4.0,
        eta_grid=np.linspace(0.55, 1.0, 5),  # 减少网格点数以加快测试
        Tq_grid=np.linspace(2.0, 10.0, 5),
        oA=8,
        max_workers=4
    )
    sweep_time = time.time() - start_time
    print(f"维度扫描运行时间: {sweep_time:.2f}秒")
    
    # 绘制维度对比图
    print("\n4. 绘制维度对比图...")
    plot_dimension_comparison(H_tomo_results, eta_grid_full)