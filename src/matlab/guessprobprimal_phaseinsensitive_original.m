%% 前提说明：
% 1. 需安装CVX工具箱（https://cvxr.com/cvx/）及兼容的SDP求解器（如SDPT3/MOSEK）
% 2. 确保Probability.mat文件为9×256数组（对应9个测试态的256个原始概率）
% 3. 手动指定选中的测试态强度、先验概率和密度矩阵维度M

clear; clc;
%% ===================== 1. 基础参数配置（用户需根据需求修改）=====================
% !!! 注意：由于算法复杂度为 O(N^(D+1))，请勿一次性使用过多测试态 !!!
% 建议 D <= 3 (例如只选 [20, 80, 160])，否则变量将超过百万级导致内存溢出
selected_mu_list = [100, 120, 140];  % 示例：仅选择3个强度
q_selected = [1/4, 1/4,1/2];      % 选中测试态的先验概率 q_x
M = 280;                            % 截断维数（光子数空间）

full_mu = [0,20,40,60,80,100,120,140,160];  % 原始9个测试态的强度（固定）

% 校验输入的选中强度是否在原始列表中
if ~all(ismember(selected_mu_list, full_mu))
    error('选中的强度不在原始列表[%s]中！', num2str(full_mu));
end
% 找到选中强度在原始列表中的索引（用于读取ProbData对应行）
selected_full_indices = find(ismember(full_mu, selected_mu_list));
% 选中测试态的数量（后续计算的核心维度D）
D = length(selected_mu_list);  
% 校验概率向量长度匹配
if length(q_selected) ~= D
    error('概率向量长度(%d)与选中测试态数量(%d)不匹配！', length(q_selected), D);
end   

%% ===================== 2. 输入参数校验与初始化 =====================
% 初始化变量
N = 8;                                        % 粗粒化后的输出数量
p = zeros(D, N);                              % p(x,y)：x=测试态，y=输出
rho = zeros(D, M, M);                         % 输入态密度矩阵
shift = 0;                                    % 数据偏移

% ===================== 3. 构建输入态密度矩阵rho（对角矩阵）=====================
for i = 1:D
    alpha = sqrt(selected_mu_list(i));        % 相干态参数α=√μ
    coeff = zeros(M, 1);                      % 光子数基下的展开系数
    for n = 0:M-1
        % 相干态|α>在光子数态|n>下的系数
        coeff(n+1) = exp(-abs(alpha)^2 / 2) * (alpha^n) / sqrt(factorial(n));
    end
    rho(i,:,:) = coeff * coeff';              % 密度矩阵|α><α|
end
% 提取rho的对角元素（因为题目要求对角约束，只需保留对角线信息）
% rho_diag 维度: D x M
rho_diag = zeros(D, M);
for i = 1:D
    rho_i_2d = squeeze(rho(i,:,:));
    rho_diag(i,:) = diag(rho_i_2d);
end

% %% ===================== 3. 构建输入态密度矩阵rho（数值稳定版）=====================
% for i = 1:D
%     alpha = sqrt(selected_mu_list(i));        % 相干态参数α=√μ
%     coeff = zeros(M, 1);                      % 光子数基下的展开系数
% 
%     for n = 0:M-1
%         if alpha == 0
%             % 特殊处理真空态 mu=0
%             if n == 0, coeff(n+1) = 1; else, coeff(n+1) = 0; end
%         else
%             % === 修改开始：使用对数域计算避免 factorial(300) 溢出 ===
%             % 原公式: exp(-|α|^2/2) * α^n / sqrt(n!)
%             % 对数公式: -|α|^2/2 + n*ln(α) - 0.5*ln(n!)
%             % 注意：gammaln(n+1) 等于 ln(n!)
% 
%             log_c = -abs(alpha)^2 / 2 + n * log(alpha) - 0.5 * gammaln(n+1);
%             coeff(n+1) = exp(log_c);
%             % === 修改结束 ===
%         end
%     end
% 
%     % 归一化校验（防止数值误差累积）
%     coeff = coeff / norm(coeff); 
% 
%     rho(i,:,:) = coeff * coeff';              % 密度矩阵|α><α|
% end
% 
% % 提取rho的对角元素
% rho_diag = zeros(D, M);
% for i = 1:D
%     rho_i_2d = squeeze(rho(i,:,:));
%     rho_diag(i,:) = diag(rho_i_2d);
% end
% 
% % 检查是否仍有NaN（调试用）
% if any(isnan(rho_diag(:)))
%     error('rho_diag 中仍然存在 NaN，请检查输入强度是否为负数或异常值。');
% end

%% ===================== 4. 读取概率数据并粗粒化（获取p(y|x)）=====================
% 读取Probability.mat文件
try
    mat_data = load('Probability.mat');
    var_names = fieldnames(mat_data);
    ProbData = mat_data.(var_names{1});
catch
    warning('未找到Probability.mat，使用随机数据代替以演示代码逻辑');
    ProbData = rand(9, 256);
    ProbData = ProbData ./ sum(ProbData, 2); % 归一化
end

% 粗粒化处理
block_size = 256 / N;
block_size = round(block_size);
for i = 1:D
    prob_256 = ProbData(selected_full_indices(i) + shift, :);  
    for k = 1:N
        idx_start = (k-1)*block_size + 1;
        idx_end = k*block_size;
        p(i, k) = sum(prob_256(idx_start:idx_end));
    end
end

%% ===================== 5. 准备SDP/LP 求解所需索引 =====================
% 问题描述基于文档公式(3)和(4)。
% 变量是 M_y^{lambda_0...lambda_D}。
% 总共有 N^(D+1) 种确定性策略 (Strategies)。
% 每一列 lambda_vec 代表一组 (lambda_0, lambda_1, ..., lambda_D)

fprintf('生成策略索引... (D=%d, N=%d)\n', D, N);
num_strategies = N^(D+1);
if num_strategies > 100000
    warning('警告：策略数量为 %d，内存可能不足。建议减少D的数量。', num_strategies);
end

% 生成所有 lambda 的组合。维度：num_strategies x (D+1)
% 列1对应 lambda_0，列2对应 lambda_1 ... 列D+1 对应 lambda_D
args = repmat({1:N}, 1, D+1);
[grids{1:D+1}] = ndgrid(args{:});
LambdaIndices = zeros(num_strategies, D+1);
for i = 1:D+1
    LambdaIndices(:,i) = grids{i}(:);
end

%% ===================== 6. CVX 优化求解 =====================
fprintf('开始CVX求解...\n');

cvx_begin


    % 变量定义：
    % M_elements: 对应文档中的 M_y^{lambda...} 的对角元素
    % 维度说明: M(光子数空间) x N(输出值y) x num_strategies(策略数)
    % 由于是对角矩阵约束，我们只需要存储对角线上的向量
    variable M_elements(M, N, num_strategies) nonnegative
    
    % --- 目标函数 (公式 2) ---
    % max sum_x q_x * sum_lambda tr(rho_x * M_{lambda_x}^{lambda})
    % 在对角约束下，tr(rho * M) = rho_diag * M_diag
    
    obj_expr = 0;
    % 由于直接在CVX中写巨大循环很慢，我们先预计算系数或者分块求和
    % 这里为了代码清晰，使用通过中间变量累加的方式
    
    for x_idx = 1:D
        % 对应当前 x 的先验概率
        qx = q_selected(x_idx);
        rho_vec = rho_diag(x_idx, :); % 1 x M
        
        % 对于特定的 x，攻击者猜中的输出值由 lambda_x 决定
        % 注意：LambdaIndices 的第 (x_idx+1) 列对应 lambda_x (因为第1列是lambda_0)
        target_y_indices = LambdaIndices(:, x_idx+1); 
        
        % 我们需要求和： sum_k ( rho_vec * M_elements(:, target_y_k, k) )
        % 为了利用矩阵运算加速，构造掩码或直接索引稍显复杂，
        % 这里使用循环遍历 N 种可能的输出 y
        
        for y_val = 1:N
            % 找出所有使得 lambda_x == y_val 的策略索引 k
            k_indices = (target_y_indices == y_val);
            
            if any(k_indices)
                % 提取这些策略对应的 M 变量
                % M_subset 维度: M x 1 x sum(k_indices) -> 我们可以求和降维
                % 聚合所有满足条件的策略的 M 变量
                M_sum_for_y = sum(M_elements(:, y_val, k_indices), 3); % 结果为 M x 1
                
                % 累加到目标函数: q_x * (rho_vec . M_sum)
                obj_expr = obj_expr + qx * (rho_vec * M_sum_for_y);
            end
        end
    end
    
    maximize(obj_expr)
    
    subject to
        % --- 约束 1: 归一化约束 (公式 3 第三行 / 公式 48) ---
        % sum_y M_y^{lambda} = (1/M) * tr(...) * I
        % 物理含义：对于每一个策略 lambda，所有输出 y 的 POVM 元素之和必须正比于单位阵。
        % 在对角且非负变量下，这意味着 sum_y M_elements(:, y, k) 必须是平坦向量 (所有元素相等)。
        
        % 计算每个策略 k 的 sum_y
        sum_over_y = sum(M_elements, 2); % 维度: M x 1 x num_strategies
        
        % 约束：对于每个 k，该向量的所有元素必须相等
        % 实现技巧：向量平均值 == 向量本身，或者 v(1) == v(2) ...
        % 这里使用差分约束： v(2:end) == v(1:end-1)
        for k = 1:num_strategies
             vec_k = sum_over_y(:, 1, k);
             vec_k(2:end) == vec_k(1:end-1);
        end
        
        % --- 约束 2: 统计数据兼容 (公式 3 最后一行 / 公式 50) ---
        % sum_lambda tr(rho_x * M_y^{lambda}) = p(y|x)
        % 即：对所有策略求和后，必须匹配实验概率
        
        % 先对策略维度 (dim 3) 求和，得到“平均”POVM
        M_total = sum(M_elements, 3); % 维度: M x N
        
        for x_idx = 1:D
            for y_idx = 1:N
                % rho_x . M_total_y == p(x,y)
                rho_diag(x_idx, :) * M_total(:, y_idx) == p(x_idx, y_idx);
            end
        end

cvx_end

%% ===================== 7. 结果输出 =====================
if strcmpi(cvx_status, 'Solved') || strcmpi(cvx_status, 'Inaccurate/Solved')
    p_guess = cvx_optval;
    H_min = -log2(p_guess);
    fprintf('\n========================================\n');
    fprintf('Optimization Status: %s\n', cvx_status);
    fprintf('Max Guessing Probability: %.6f\n', p_guess);
    fprintf('Min-Entropy (H_min): %.6f bits\n', H_min);
    fprintf('========================================\n');
else
    fprintf('Optimization Failed: %s\n', cvx_status);
end