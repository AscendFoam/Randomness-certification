%% 相位不敏感 APD 模型的原始 SDP（Primal SDP）求解器
% ============================================================================
% 功能说明：
%   本脚本通过求解半定规划（SDP）问题，计算在相位不敏感的雪崩光电二极管
%   （APD, Avalanche Photodiode）模型下的**最大猜测概率**（Maximum Guessing
%   Probability），进而得到**最小熵**（Min-Entropy）H_min。
%
%   最小熵是量子随机数发生器（QRNG）中衡量随机性质量的核心指标：
%     H_min = -log2(p_guess)
%   其中 p_guess 是敌手在最优策略下猜中测量结果的概率。
%
% 物理背景：
%   - 使用相干态（coherent state）作为输入，其光子数分布服从泊松分布。
%   - APD 的相位不敏感模型意味着测量结果仅依赖光子数的对角元。
%   - 通过 primal SDP 直接最大化猜测概率，得到随机性上界的严格估计。
%
% 使用说明：
%   1. 安装 CVX 工具箱及兼容的 SDP 求解器（强烈推荐 MOSEK）。
%   2. 确保当前目录下存在 Probability.mat 文件，其中应包含一个 9×256 的
%      概率分布表（9 种光强 × 256 种可能的探测器输出结果）。
%   3. 直接运行本脚本即可。
% ============================================================================

clear; clc;

%% ===================== 1. 基本参数配置 =====================
% selected_mu_list：从 9 个候选光强中选取用于优化的测试态光强（单位：0.01 photon）
%   - 100 对应平均光子数 μ=1.0，120 对应 μ=1.2，140 对应 μ=1.4
%   - 选取原则：选取信噪比较好、概率分布差异明显的光强组合
%   - 注意：由于 SDP 的计算复杂度为 O(N^(D+1))，D 不宜过大
selected_mu_list = [100, 120, 140];

% q_selected：每个测试态的先验概率（权重）
%   - 必须与 selected_mu_list 等长
%   - 这里分配 1/4、1/4、1/2 的概率给三个测试态
%   - 权重之和应为 1
q_selected = [1/4, 1/4, 1/2];

% M：Fock 空间的截断维度
%   - 将无限维的光子数空间截断为 {0, 1, 2, ..., M-1}
%   - 需要足够大以覆盖泊松分布的绝大部分概率质量
%   - 对于 μ≈1.4，泊松分布的尾部在 n=280 处已可忽略
M = 280;

% full_mu：完整的 9 个候选光强列表
%   - 对应实验中制备的 9 个不同强度的相干态
%   - 包含 0（真空态）到 160（μ=1.6）共 9 个离散光强
full_mu = [0, 20, 40, 60, 80, 100, 120, 140, 160];

% 验证选定的光强是否在候选列表中
if ~all(ismember(selected_mu_list, full_mu))
    error('selected_mu_list must be a subset of [%s].', num2str(full_mu));
end

% selected_full_indices：选定光强在完整列表中的位置索引（从 1 开始）
%   - 后续从 Probability.mat 中提取对应行的概率数据时使用
selected_full_indices = find(ismember(full_mu, selected_mu_list));

% D：选定的测试态数量
D = length(selected_mu_list);

% 验证先验概率向量的长度与测试态数量是否一致
if length(q_selected) ~= D
    error('Length of q_selected (%d) does not match number of selected test states (%d).', ...
        length(q_selected), D);
end

%% ===================== 2. 初始化变量 =====================
% N：APD 探测器的输出结果数（即测量结果的离散取值个数）
%   - 例如 N=4 表示探测器有 4 种可能的输出
N = 4;

% p：条件概率矩阵，大小为 D×N
%   - p(i, k) 表示在使用第 i 个测试态时，测量结果为 k 的概率
%   - 将由 Probability.mat 中的数据粗粒化得到
p = zeros(D, N);

% shift：索引偏移量
%   - 用于从 Probability.mat 中选取正确的行
%   - 默认为 0（无偏移）
shift = 0;

%% ===================== 3. 稳定构建 rho_diag（密度矩阵对角元） =====================
% 相干态 |α⟩ 的密度矩阵在 Fock 基下是对角的，其对角元服从泊松分布：
%   ρ_nn = exp(-μ) * μ^n / n!
%
% 数值稳定性考虑：
%   - 当 n ≥ 171 时，n! 会超出 double 的表示范围（Inf）
%   - 因此在对数空间中计算 log(p_n)，再通过 exp 还原
%   - log(p_n) = -μ + n*ln(μ) - ln(Γ(n+1))
%   - gammaln(x) = ln(Γ(x))，对于正整数 x，Γ(x) = (x-1)!
%
% rho_diag：大小为 D×M 的矩阵
%   - rho_diag(i, :) 是第 i 个测试态的光子数概率分布（归一化后）
rho_diag = zeros(D, M);

% photon_numbers：光子数向量 [0, 1, 2, ..., M-1]
photon_numbers = 0:(M-1);

for i = 1:D
    % 获取第 i 个测试态的平均光子数 μ_i
    mu_i = selected_mu_list(i);

    if mu_i < 0
        error('Mean photon numbers must be non-negative.');
    end

    if mu_i == 0
        % μ=0 对应真空态，所有光子都在 n=0
        diag_i = zeros(1, M);
        diag_i(1) = 1;
    else
        % 在对数空间中计算泊松概率，避免阶乘溢出
        % log_probs(n) = -μ + n*ln(μ) - ln(Γ(n+1))
        log_probs = -mu_i + photon_numbers * log(mu_i) - gammaln(photon_numbers + 1);
        % 通过 exp 从对数概率恢复实际概率值
        diag_i = exp(log_probs);
    end

    % 检查是否存在非有限值（NaN 或 Inf）
    if any(~isfinite(diag_i))
        error('rho_diag contains non-finite values. Check the input parameters.');
    end

    % 将由于数值精度导致的微小负值置零
    diag_i(diag_i < 0) = 0;

    % 计算迹（所有对角元之和），用于归一化
    trace_i = sum(diag_i);
    if ~isfinite(trace_i) || trace_i <= 0
        error('Invalid trace encountered while building rho_diag.');
    end

    % 归一化：确保 ρ 的迹为 1（概率和为 1）
    rho_diag(i, :) = diag_i / trace_i;
end

% 归一化后的全局检查：确保没有 NaN
if any(isnan(rho_diag(:)))
    error('rho_diag contains NaN entries after normalization.');
end

% 验证每个测试态的迹与 1 的偏差（截断误差）
%   - 由于 Fock 空间截断，泊松分布的尾部被忽略
%   - 如果偏差 > 1e-10，说明截断维度 M 可能不够大
trace_error = max(abs(sum(rho_diag, 2) - 1));
if trace_error > 1e-10
    warning('rho_diag normalization error is %.3e. Consider increasing M.', trace_error);
end

%% ===================== 4. 加载概率数据并进行粗粒化 =====================
% Probability.mat 包含实验测得的概率分布表
%   - 大小为 9×256（9 种光强，256 种可能的原始输出）
%   - 需要将 256 种原始输出合并（粗粒化）为 N=4 类
try
    % 尝试加载 Probability.mat 文件
    mat_data = load('Probability.mat');
    % 获取文件中第一个变量的名称和内容
    var_names = fieldnames(mat_data);
    ProbData = mat_data.(var_names{1});
catch
    % 如果文件不存在，使用随机数据作为演示回退方案
    %   - 注意：随机数据仅供调试，不具备物理意义
    warning('Probability.mat not found. Falling back to random demo data.');
    ProbData = rand(9, 256);
    % 对每行归一化，确保每行的概率之和为 1
    ProbData = ProbData ./ sum(ProbData, 2);
end

% block_size：每个粗粒化块的大小
%   - 将 256 个原始结果等分为 N=4 块，每块 64 个结果
block_size = 256 / N;
block_size = round(block_size);

% 对每个测试态，将其对应的 256 个概率值粗粒化为 N 个概率
for i = 1:D
    % 提取第 i 个测试态对应的原始 256 维概率向量
    %   - selected_full_indices(i) 是该测试态在完整 9 光强列表中的位置
    prob_256 = ProbData(selected_full_indices(i) + shift, :);

    for k = 1:N
        % 第 k 块的起止索引
        idx_start = (k - 1) * block_size + 1;
        idx_end = k * block_size;
        % 粗粒化：将第 k 块内的所有概率求和
        p(i, k) = sum(prob_256(idx_start:idx_end));
    end
end

%% ===================== 5. 生成策略索引矩阵 =====================
% SDP 的核心变量是测量算符 M_{y|λ}，其中：
%   - y ∈ {1,...,N} 是测量输出（共 N 个取值）
%   - λ 是"策略"（即 side information），由 D+1 个取值确定：
%       λ = (λ_0, λ_1, ..., λ_D)，每个分量 ∈ {1,...,N}
%   - λ_0 对应 eve 的猜测，λ_i (i≥1) 对应第 i 个测试态的预测输出
%
% num_strategies = N^(D+1)：所有可能的策略总数
%   - D=3, N=4 时，num_strategies = 4^4 = 256
fprintf('Generating strategy indices... (D=%d, N=%d)\n', D, N);
num_strategies = N^(D + 1);

% 如果策略数量过多，发出内存警告
if num_strategies > 100000
    warning('Number of strategies is %d. Memory usage may be large.', num_strategies);
end

% 使用 ndgrid 生成所有 (D+1) 维策略组合
%   - args = {1:N, 1:N, ..., 1:N}，共 D+1 个
%   - ndgrid 生成所有可能的组合
args = repmat({1:N}, 1, D + 1);
[grids{1:D+1}] = ndgrid(args{:});

% LambdaIndices：大小为 num_strategies × (D+1) 的矩阵
%   - 每一行代表一个策略 λ
%   - 第 1 列是 λ_0，第 j+1 列是 λ_j
LambdaIndices = zeros(num_strategies, D + 1);
for i = 1:D+1
    LambdaIndices(:, i) = grids{i}(:);
end

%% ===================== 6. 使用 CVX 求解 SDP =====================
% SDP 问题（Primal 形式）：
%
%   最大化：∑_{x=1}^{D} q_x * ∑_{y=1}^{N} Tr(ρ_x * M_{y|λ: λ_{x+1}=y})
%
%   约束条件：
%   1. 正定性：M_{y|λ} ≥ 0（对所有 y, λ）
%   2. 对于每个策略 λ，测量算符在 y 上的求和与 y 无关（无信号条件）：
%      ∑_y M_{y|λ} 的第 n 个对角元对所有 y 相同
%   3. 归一化：Tr(ρ_x * ∑_λ M_{y|λ}) = p(x, y)
%      即总测量概率与实验观测一致
%
fprintf('Starting CVX solve...\n');

% 尝试切换到 MOSEK 求解器（精度和速度均优于 SDPT3/SeDuMi）
try
    cvx_solver mosek
catch solver_err
    warning('Could not switch to MOSEK. CVX will use its default solver instead. Reason: %s', ...
        solver_err.message);
end

cvx_begin
    % 优化变量：M_elements，大小为 M × N × num_strategies 的三维数组
    %   - M_elements(n, y, k) 表示 M_{y|λ_k} 的第 n 个对角元
    %   - nonnegative：保证所有元素非负（物理上要求测量算符为正算符）
    variable M_elements(M, N, num_strategies) nonnegative

    % 构建目标函数：
    %   目标 = ∑_{x} q_x * ∑_{y} Tr(ρ_x * ∑_{λ: λ_{x+1}=y} M_{y|λ})
    %
    % 物理含义：在最优测量策略下，敌手猜中输入态对应的测量结果的概率
    %   - 对每个测试态 x，以先验概率 q_x 加权
    %   - 对每个输出 y，累加所有"预测正确"的策略（λ_{x+1} = y）的贡献
    obj_expr = 0;
    for x_idx = 1:D
        % 第 x 个测试态的先验概率
        qx = q_selected(x_idx);
        % 第 x 个测试态的光子数概率分布（1×M 向量）
        rho_vec = rho_diag(x_idx, :);
        % 所有策略中，λ_{x+1}（即对第 x 个测试态的预测输出）的取值
        target_y_indices = LambdaIndices(:, x_idx + 1);

        for y_val = 1:N
            % 找出所有"预测第 x 个测试态输出为 y_val"的策略索引
            k_indices = (target_y_indices == y_val);
            if any(k_indices)
                % 对这些策略，将对应的测量算符在 y=y_val 上求和
                %   M_sum_for_y 是 M×1 向量，表示 ∑_{λ: λ_{x+1}=y} M_{y|λ}
                M_sum_for_y = sum(M_elements(:, y_val, k_indices), 3);
                % 计算 Tr(ρ_x * M_sum_for_y) 并累加到目标函数
                %   rho_vec * M_sum_for_y 是内积，等于 ∑_n ρ_x(n) * M_sum(n)
                obj_expr = obj_expr + qx * (rho_vec * M_sum_for_y);
            end
        end
    end

    maximize(obj_expr)

    subject to
        % ===== 约束 1：无信号条件（No-Signaling） =====
        % 对每个策略 λ，∑_y M_{y|λ} 的所有对角元必须相同
        % 即测量算符之和 ∑_y M_{y|λ} 必须是单位矩阵的倍数
        %
        % 物理含义：不同输出 y 的测量算符加在一起应该不携带关于策略 λ 的信息
        %   - sum_over_y(n, 1, k) = ∑_y M_{y|λ_k}(n)：策略 k 下第 n 个对角元之和
        %   - vec_k(2:end) == vec_k(1:end-1) 要求所有对角元彼此相等
        sum_over_y = sum(M_elements, 2);  % 大小为 M × 1 × num_strategies
        for k = 1:num_strategies
            vec_k = sum_over_y(:, 1, k);  % 策略 k 的对角元和向量，M×1
            vec_k(2:end) == vec_k(1:end-1);  % 等价于 vec_k 的所有元素相等
        end

        % ===== 约束 2：测量概率匹配 =====
        % 对每个测试态 x 和输出 y，总测量概率必须等于实验观测值 p(x, y)
        %
        % M_total(n, y) = ∑_λ M_{y|λ}(n)：所有策略在输出 y 上的测量算符之和
        % Tr(ρ_x * M_total(:, y)) = ∑_n ρ_x(n) * M_total(n, y) = p(x, y)
        M_total = sum(M_elements, 3);  % 大小为 M × N
        for x_idx = 1:D
            for y_idx = 1:N
                rho_diag(x_idx, :) * M_total(:, y_idx) == p(x_idx, y_idx);
            end
        end
cvx_end

%% ===================== 7. 输出结果 =====================
% cvx_status：CVX 的求解状态
%   - 'Solved'：成功求解
%   - 'Inaccurate/Solved'：求解成功但精度有限
%   - 其他值（如 'Infeasible'、'Unbounded'）表示失败
if strcmpi(cvx_status, 'Solved') || strcmpi(cvx_status, 'Inaccurate/Solved')
    % cvx_optval：目标函数的最优值，即最大猜测概率 p_guess
    p_guess = cvx_optval;

    % 最小熵 H_min = -log2(p_guess)
    %   - H_min 越大，随机性越好
    %   - 理想情况下 p_guess = 1/N（均匀分布），H_min = log2(N)
    H_min = -log2(p_guess);

    fprintf('\n========================================\n');
    fprintf('Optimization Status: %s\n', cvx_status);
    fprintf('Max Guessing Probability: %.6f\n', p_guess);
    fprintf('Min-Entropy (H_min): %.6f bits\n', H_min);
    fprintf('========================================\n');
else
    fprintf('Optimization Failed: %s\n', cvx_status);
end
