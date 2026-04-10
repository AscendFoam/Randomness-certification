%% Route4-ex-constrained 的 Matlab 单文件参考脚本
% =========================================================================
% 目标：
%   给导师提供一份“拿到后可直接在 Matlab + CVX 中运行”的单文件版本，
%   用来复现当前 route4-ex-constrained 主线的正式计算。
%
% 这份脚本刻意保持与原始脚本 guessprobprimal_phaseinsensitive_original.m
% 相近的阅读顺序：
%   1. 参数配置
%   2. 输入检查与初始化
%   3. 构造 trusted input states
%   4. 读取 Probability.mat 并 coarse-graining
%   5. 生成 LambdaIndices
%   6. 求解 diagonal primal（可选）
%   7. 求解 full primal（主结果）
%   8. 输出结果
%
% 与原始 Matlab route4 的关键区别：
%   - Probability.mat 仍然是同一份实验概率表；
%   - selected_mu_list / q_selected 仍保留同样角色；
%   - 但 trusted input 不再只取 Fock 对角，而是完整截断相干态 |alpha><alpha|；
%   - 正式结果以 full primal（一般 Hermitian PSD POVM）为主。
%
% 当前默认参数对应 Python 中已经 formal 复核过的 constrained 主线：
%   selected_mu_list = [100, 120, 140]
%   q_selected      = [1, 0, 0]
%   custom_edges    = [0, 121, 132, 256]
%   alpha_values    = [0.54, 0.66 i, -0.72]
%   M               = 6
%
% 预期主结果（MOSEK）：
%   full primal: H_min ≈ 1.2275008643
%
% 运行前提：
%   1. 已安装 CVX；
%   2. 已配置可用的 SDP 求解器，推荐 MOSEK；
%   3. 当前目录下或脚本同目录下存在 Probability.mat。
%
% 如需快速了解本脚本相对原始 Matlab route4 的变化，可先阅读：
%   docs/route4_ex_constrained_matlab_onepage_summary_cn.md
% 若需要更细的逐段对照说明，可进一步阅读：
%   docs/route4_ex_constrained_matlab_comparison_cn.md
% =========================================================================

clear; clc;

%% ===================== 1. 基本参数配置 =====================
% selected_mu_list：
%   从原始 9 个光强标签中选取当前要放进 SDP 的输入窗口。
%   这里固定为 [100, 120, 140]，对应当前 constrained 主线。
selected_mu_list = [100, 120, 140];

% q_selected：
%   生成轮目标函数中的输入权重。这里取 [1,0,0]，意味着目标函数只关心
%   第一个输入（mu_label = 100）上的 guessing probability。
%   这并不意味着后两个输入“不使用”，它们仍然通过统计约束进入 SDP，
%   只是在目标函数中不直接加权。
q_selected = [1, 0, 0];

% M：
%   trusted input 截断到的 Fock 维数。这里取 6，是当前 constrained 主线
%   已经验证过可行、且计算规模较小的一组设置。
M = 6;

% full_mu：
%   Probability.mat 中 9 个固定输入标签的完整菜单。
full_mu = [0, 20, 40, 60, 80, 100, 120, 140, 160];

% shift：
%   保留与原 route4 Matlab 代码一致的“行偏移”接口。
%   当前主线不使用偏移，因此保持为 0。
shift = 0;

% custom_edges 使用 Python/JSON 里的“零基边界”记法：
% 第 k 个 coarse bin 覆盖原始列 custom_edges(k)+1 : custom_edges(k+1)
% 这里的 [0, 121, 132, 256] 对应 3 输出 coarse-graining：
%   - 输出1覆盖原始列 1:121
%   - 输出2覆盖原始列 122:132
%   - 输出3覆盖原始列 133:256
custom_edges = [0, 121, 132, 256];

% radii / phases / alpha_values：
%   当前 constrained 主线固定采用的 trusted coherent alphabet。
%   这里不再做自由搜索，而是直接写死当前正式结果对应的一组参数。
%   alpha_values = radii .* exp(i * phases)
%   具体为：
%     alpha_1 = 0.54
%     alpha_2 = 0.66 i
%     alpha_3 = -0.72
radii = [0.54, 0.66, 0.72];
phases = [0, pi/2, pi];
alpha_values = radii .* exp(1i * phases);

% prob_floor：
%   概率正则化下限。若 coarse-grained 概率里有 0，则把它抬到该值后再归一化，
%   以避免后续数值病态。当前默认 1e-12。
prob_floor = 1e-12;

% probability_variable_name / probability_filename：
%   读取 Probability.mat 时优先取名为 Probability 的变量。
probability_variable_name = 'Probability';
probability_filename = 'Probability.mat';

% run_diagonal_primal：
%   是否同时跑“只允许 Fock 对角测量元”的对照问题。
%   在当前默认点下它通常 infeasible，但保留它有助于直观看到：
%   非对角 trusted input 的确给 full primal 带来了新的认证能力。
run_diagonal_primal = true;   % 仅用于对照；当前默认点通常 infeasible

% run_full_primal：
%   是否运行当前真正关心的主问题。这里应始终保持 true。
run_full_primal = true;       % 主结果

% save_result_mat：
%   若设为 true，则会把结果结构体保存为 .mat 文件，方便后续复查。
save_result_mat = false;
save_result_filename = 'route4_ex_constrained_result.mat';

%% ===================== 2. 输入检查与初始化 =====================
% 第一步先检查选中的 mu 标签是否真的存在于完整菜单中。
if ~all(ismember(selected_mu_list, full_mu))
    error('selected_mu_list 必须是 [%s] 的子集。', num2str(full_mu));
end

% D：当前进入 SDP 的输入数量。
D = length(selected_mu_list);

% 检查 q_selected 的长度是否和输入数量一致。
if length(q_selected) ~= D
    error('q_selected 的长度 (%d) 必须与 selected_mu_list 的长度 (%d) 一致。', ...
        length(q_selected), D);
end

% 检查 alpha_values 的长度是否和输入数量一致。
if length(alpha_values) ~= D
    error('alpha_values 的长度 (%d) 必须与 selected_mu_list 的长度 (%d) 一致。', ...
        length(alpha_values), D);
end

% q_selected 必须是非负权重。
if any(q_selected < 0)
    error('q_selected 不能包含负数。');
end

% q_selected 总和必须大于 0，否则不能归一化，也没有物理意义。
if sum(q_selected) <= 0
    error('q_selected 的总和必须为正。');
end

% 归一化 q_selected，使其真正成为概率分布。
q_selected = q_selected / sum(q_selected);

% 检查 custom_edges 的首尾是否正确覆盖全部 256 个原始 bin。
if custom_edges(1) ~= 0 || custom_edges(end) ~= 256
    error('custom_edges 必须从 0 开始，并以 256 结束。');
end

% 检查 custom_edges 是否严格递增，避免区间重叠或空区间。
if any(diff(custom_edges) <= 0)
    error('custom_edges 必须严格递增。');
end

% N：coarse-graining 之后的输出数。
N = length(custom_edges) - 1;

% p_raw：
%   记录从 Probability.mat 直接 coarse-grain 得到的原始输出概率。
% p：
%   若需要正则化，则记录正则化后的概率；否则 p = p_raw。
p_raw = zeros(D, N);
p = zeros(D, N);

% rho：
%   记录完整 trusted input 密度矩阵，大小为 M x M x D。
% rho_diag：
%   仅提取每个 rho 的对角线，主要给 diagonal primal 使用。
rho = zeros(M, M, D);
rho_diag = zeros(D, M);

% selected_full_indices：
%   把 selected_mu_list 中每个 mu 映射回 full_mu 里的位置索引。
%   注意这里使用 Matlab 的 1-based 索引。
selected_full_indices = zeros(1, D);
for i = 1:D
    idx = find(full_mu == selected_mu_list(i), 1);
    if isempty(idx)
        error('未能在 full_mu 中找到 selected_mu_list(%d) = %g', i, selected_mu_list(i));
    end
    selected_full_indices(i) = idx;
end

fprintf('Route4-ex-constrained Matlab 参考脚本启动...\n');
fprintf('D = %d, N = %d, M = %d\n', D, N, M);

%% ===================== 3. 构造 trusted input states =====================
% 对每个固定 alpha_x 构造一个截断相干态：
%   |alpha_x> = sum_n coeff_n |n>
% 并进一步得到：
%   rho_x = |alpha_x><alpha_x|
%   rho_diag_x = diag(rho_x)
for i = 1:D
    [rho_i, rho_diag_i, coeff_i] = build_truncated_coherent_density(alpha_values(i), M);
    rho(:, :, i) = rho_i;
    rho_diag(i, :) = rho_diag_i;

    % 打印每个 trusted input 的摘要，便于检查 alpha 是否如预期。
    fprintf('Input %d: mu_label=%g, alpha=%.6f%+.6fi, ||coeff||_2=%.12f\n', ...
        i, selected_mu_list(i), real(alpha_values(i)), imag(alpha_values(i)), norm(coeff_i));
end

%% ===================== 4. 读取 Probability.mat 并 coarse-graining =====================
% 先定位 Probability.mat 文件。优先读当前工作目录；若没有，则读脚本同目录。
probability_path = resolve_probability_path(probability_filename);
fprintf('Loading probability table from: %s\n', probability_path);

% 载入 .mat 文件。
mat_data = load(probability_path);

% 优先读取名为 Probability 的变量；若该变量不存在，则退回到首个变量。
if isfield(mat_data, probability_variable_name)
    ProbData = mat_data.(probability_variable_name);
else
    variable_names = fieldnames(mat_data);
    if isempty(variable_names)
        error('Probability.mat 中没有可读取的变量。');
    end
    warning('未找到变量 %s，改用首个变量 %s。', probability_variable_name, variable_names{1});
    ProbData = mat_data.(variable_names{1});
    probability_variable_name = variable_names{1};
end

% 这里要求外部概率表必须是“若干输入 x 256 原始输出”的二维矩阵。
if ~ismatrix(ProbData) || size(ProbData, 2) ~= 256
    error('Probability.mat 中的概率表必须是二维数组，且列数为 256。');
end

% 检查文件是否确实包含了所需的那些输入行。
if size(ProbData, 1) < max(selected_full_indices + shift)
    error('Probability.mat 行数不足，无法读取所需输入。');
end

% 逐个输入读取其原始 256 维概率，并按 custom_edges 做 coarse-graining。
for i = 1:D
    % 先取出当前输入对应的原始直方图行。
    prob_256 = ProbData(selected_full_indices(i) + shift, :);

    % 保险起见先做一次行归一化，确保数值上总和为 1。
    row_sum = sum(prob_256);
    if row_sum <= 0
        error('第 %d 个输入选中的原始概率行总和非正。', i);
    end
    prob_256 = prob_256 / row_sum;

    % custom_edges 使用零基边界，因此在 Matlab 中要转换成 1-based 列范围。
    for k = 1:N
        idx_start = custom_edges(k) + 1;
        idx_end = custom_edges(k + 1);
        p_raw(i, k) = sum(prob_256(idx_start:idx_end));
    end
end

% 统计有多少 coarse-grained 输出在正则化前恰好为 0。
regularized_entries = sum(p_raw(:) == 0);

% 若启用 prob_floor，则把所有 0 概率抬到 prob_floor，再逐行归一化。
if ~isempty(prob_floor) && prob_floor > 0
    p = max(p_raw, prob_floor);
    p = p ./ sum(p, 2);
else
    p = p_raw;
end

%% ===================== 5. 生成 LambdaIndices =====================
% LambdaIndices 的每一行对应一个确定性策略 lambda：
%   lambda = (lambda_0, lambda_1, ..., lambda_D)
% 每个分量都属于 {1, ..., N}。
% 因此总策略数为 N^(D+1)。
fprintf('Generating strategy indices... (D=%d, N=%d)\n', D, N);
num_strategies = N^(D + 1);

% 使用 ndgrid 枚举全部策略组合。
args = repmat({1:N}, 1, D + 1);
[grids{1:D+1}] = ndgrid(args{:});
LambdaIndices = zeros(num_strategies, D + 1);
for i = 1:D+1
    LambdaIndices(:, i) = grids{i}(:);
end

%% ===================== 6. diagonal primal（可选对照） =====================
diagonal_result = struct();
if run_diagonal_primal
    fprintf('Starting diagonal primal...\n');

    % 尽量优先用 MOSEK；若当前 Matlab/CVX 环境没有配置好，则退回默认求解器。
    try
        cvx_solver mosek
    catch solver_err
        warning('无法切换到 MOSEK，将使用 CVX 默认求解器。原因：%s', solver_err.message);
    end

    cvx_begin quiet
        % M_diag(n, y, lambda)：
        %   第 lambda 个策略下、第 y 个输出对应 POVM 元的第 n 个 Fock 对角元。
        % 这是“只允许 Fock 对角测量元”的对照问题。
        variable M_diag(M, N, num_strategies) nonnegative
        expression obj_diag
        obj_diag = 0;

        % 目标函数：
        %   max sum_x q_x * sum_{lambda: lambda_x = y} Tr(rho_x * M_{y,lambda})
        % 在 diagonal primal 中，这里的 Tr(...) 退化成 rho_diag 与 M_diag 的内积。
        for x_idx = 1:D
            qx = q_selected(x_idx);
            rho_vec = rho_diag(x_idx, :);
            target_y_indices = LambdaIndices(:, x_idx + 1);

            for y_val = 1:N
                k_indices = find(target_y_indices == y_val);
                if ~isempty(k_indices)
                    % 把所有“对输入 x 来说会猜成 y”的策略先加总。
                    M_sum_for_y = sum(M_diag(:, y_val, k_indices), 3);
                    obj_diag = obj_diag + qx * (rho_vec * M_sum_for_y);
                end
            end
        end

        maximize(obj_diag)
        subject to
            % 对每个策略 lambda，sum_y M_{y,lambda} 必须正比于单位阵。
            % 在对角模型里，这意味着对应对角向量必须“所有分量相等”。
            sum_over_y = sum(M_diag, 2);
            for k = 1:num_strategies
                vec_k = sum_over_y(:, 1, k);
                vec_k(2:end) == vec_k(1:end-1);
            end

            % 统计约束：
            %   对每个输入 x 和输出 y，总 POVM 元必须复现实验 coarse-grained 概率 p(x,y)。
            M_total_diag = sum(M_diag, 3);
            for x_idx = 1:D
                for y_idx = 1:N
                    rho_diag(x_idx, :) * M_total_diag(:, y_idx) == p(x_idx, y_idx);
                end
            end
    cvx_end

    diagonal_result = build_result_struct( ...
        'route4_ex_constrained_diagonal_primal_matlab', ...
        cvx_status, cvx_optval, M, N, D, num_strategies);
end

%% ===================== 7. full primal（主结果） =====================
full_result = struct();
if run_full_primal
    fprintf('Starting full primal...\n');

    % full primal 中，每个输出 y、每个策略 lambda 都对应一个 M x M 的 Hermitian PSD 矩阵。
    num_operator_variables = N * num_strategies;

    try
        cvx_solver mosek
    catch solver_err
        warning('无法切换到 MOSEK，将使用 CVX 默认求解器。原因：%s', solver_err.message);
    end

    cvx_begin quiet
        % M_full(:,:,op_idx)：
        %   full primal 中的 POVM 矩阵变量。
        % 这里不再限制它必须是 Fock 对角，因此它可以利用 trusted input 的非对角结构。
        variable M_full(M, M, num_operator_variables) hermitian semidefinite

        % s_lambda(lambda_idx)：
        %   每个策略的标量权重，使得 sum_y M_{y,lambda} = s_lambda * I。
        variable s_lambda(num_strategies) nonnegative
        expression obj_full
        obj_full = 0;

        % 构造 full primal 的目标函数。
        for x_idx = 1:D
            qx = q_selected(x_idx);
            rho_x = rho(:, :, x_idx);
            target_y_indices = LambdaIndices(:, x_idx + 1);

            for y_val = 1:N
                strategy_ids = find(target_y_indices == y_val);
                if ~isempty(strategy_ids)
                    % 对固定的 x 和 y，把所有满足 lambda_x = y 的策略对应矩阵求和。
                    expression M_sum_for_y(M, M)
                    M_sum_for_y = zeros(M, M);
                    for t = 1:length(strategy_ids)
                        lambda_idx = strategy_ids(t);
                        op_idx = (lambda_idx - 1) * N + y_val;
                        M_sum_for_y = M_sum_for_y + M_full(:, :, op_idx);
                    end
                    obj_full = obj_full + qx * real(trace(rho_x * M_sum_for_y));
                end
            end
        end

        maximize(obj_full)
        subject to
            % 归一化 / 完备性约束：
            %   对每个策略 lambda，有 sum_y M_{y,lambda} = s_lambda I。
            % 这是 full primal 里与 diagonal primal 最对应的一组结构约束。
            for lambda_idx = 1:num_strategies
                expression strategy_sum(M, M)
                strategy_sum = zeros(M, M);
                for y_val = 1:N
                    op_idx = (lambda_idx - 1) * N + y_val;
                    strategy_sum = strategy_sum + M_full(:, :, op_idx);
                end
                strategy_sum == s_lambda(lambda_idx) * eye(M);
            end

            % 统计匹配约束：
            %   对每个输入 x 和输出 y，总 POVM 元必须满足
            %   Tr(rho_x * M_y) = p(x,y)。
            for x_idx = 1:D
                rho_x = rho(:, :, x_idx);
                for y_val = 1:N
                    expression total_element_y(M, M)
                    total_element_y = zeros(M, M);
                    for lambda_idx = 1:num_strategies
                        op_idx = (lambda_idx - 1) * N + y_val;
                        total_element_y = total_element_y + M_full(:, :, op_idx);
                    end
                    real(trace(rho_x * total_element_y)) == p(x_idx, y_val);
                end
            end
    cvx_end

    full_result = build_result_struct( ...
        'route4_ex_constrained_full_primal_matlab', ...
        cvx_status, cvx_optval, M, N, D, num_strategies);

    % 补充一些规模信息，便于和 Python 版本的 size estimate 对照。
    full_result.num_operator_variables = num_operator_variables;
    full_result.hermitian_scalar_count = num_operator_variables * M * M;
end

%% ===================== 8. 结果输出 =====================
% 把关键配置和结果收进一个 result 结构体，便于后续 save(...) 或人工检查。
result = struct();
result.route = 'route4_ex_constrained_matlab_script';
result.config = struct();
result.config.selected_mu_list = selected_mu_list;
result.config.selected_full_indices_one_based = selected_full_indices;
result.config.q_selected = q_selected;
result.config.M = M;
result.config.D = D;
result.config.N = N;
result.config.shift = shift;
result.config.custom_edges = custom_edges;
result.config.block_widths = diff(custom_edges);
result.config.radii = radii;
result.config.phases = phases;
result.config.alpha_values = alpha_values;
result.config.prob_floor = prob_floor;
result.config.probability_path = probability_path;
result.config.probability_variable_name = probability_variable_name;
result.config.regularized_entries = regularized_entries;
result.probabilities_raw = p_raw;
result.probabilities = p;
result.rho_diag = rho_diag;
result.diagonal_result = diagonal_result;
result.full_result = full_result;

fprintf('\n========================================\n');
fprintf('Route4-ex-constrained Matlab script done.\n');
fprintf('selected_mu_list = [%s]\n', num2str(selected_mu_list));
fprintf('custom_edges     = [%s]\n', num2str(custom_edges));
fprintf('alpha_values     = [');
for i = 1:D
    fprintf(' %.6f%+.6fi ', real(alpha_values(i)), imag(alpha_values(i)));
end
fprintf(']\n');
if isfield(full_result, 'status')
    fprintf('Full primal status: %s\n', full_result.status);
end
if isfield(full_result, 'H_min') && ~isempty(full_result.H_min)
    fprintf('Full primal H_min: %.12f bits\n', full_result.H_min);
end
if isfield(diagonal_result, 'status')
    fprintf('Diagonal primal status: %s\n', diagonal_result.status);
end
fprintf('========================================\n');

if save_result_mat
    save(save_result_filename, 'result');
    fprintf('Saved result struct to %s\n', save_result_filename);
end

%% ===================== Local Functions =====================
function [rho_i, rho_diag_i, coeff] = build_truncated_coherent_density(alpha, M)
% 在截断 Fock 空间中构造单个相干态的密度矩阵与对角元。
%
% 输入：
%   alpha : 相干态振幅
%   M     : Fock 截断维数
%
% 输出：
%   rho_i      : M x M 的截断密度矩阵 |alpha><alpha|
%   rho_diag_i : rho_i 的对角线（行向量）
%   coeff      : |alpha> 在截断 Fock 基下的展开系数
%
% 说明：
%   这里使用对数域公式计算 coeff，避免直接用 factorial(n) 带来的数值溢出。

    % 先为 Fock 展开系数分配空间。
    coeff = zeros(M, 1);

    % 真空态 alpha = 0 时，只有 |0> 分量非零。
    if abs(alpha) == 0
        coeff(1) = 1;
    else
        % 一般相干态时，使用
        %   coeff_n = exp(-|alpha|^2/2) * alpha^n / sqrt(n!)
        % 的对数版本来计算，数值更稳定。
        for n = 0:M-1
            log_coeff = -abs(alpha)^2 / 2 + n * log(alpha) - 0.5 * gammaln(n + 1);
            coeff(n + 1) = exp(log_coeff);
        end
    end

    % 归一化，确保截断后的态向量范数为 1。
    coeff_norm = norm(coeff);
    if ~isfinite(coeff_norm) || coeff_norm <= 0
        error('相干态展开系数的范数非法，请检查 alpha 和 M。');
    end
    coeff = coeff / coeff_norm;

    % 构造密度矩阵和其对角线。
    rho_i = coeff * coeff';
    rho_diag_i = real(diag(rho_i)).';
end

function probability_path = resolve_probability_path(probability_filename)
% 优先在当前工作目录查找 Probability.mat；若不存在，则回退到脚本同目录。
%
% 这样导师既可以在 matlab 当前目录直接运行，也可以从别处调用脚本。

    if exist(probability_filename, 'file')
        probability_path = probability_filename;
        return;
    end

    current_file = mfilename('fullpath');
    current_dir = fileparts(current_file);
    fallback_path = fullfile(current_dir, probability_filename);
    if exist(fallback_path, 'file')
        probability_path = fallback_path;
        return;
    end

    error('未找到 %s。请将其放在当前工作目录或脚本同目录下。', probability_filename);
end

function result_struct = build_result_struct(route_name, cvx_status_value, cvx_optval_value, M, N, D, num_strategies)
% 把 CVX 的状态和最优值整理成统一结构体。
%
% 这里的逻辑非常简单：
%   - 若 CVX 给出的是“成功求解”状态，且最优值是正且有限的，
%     则记录 p_guess 和 H_min；
%   - 否则把这两个字段留空，避免误把失败态当成正式结果。

    result_struct = struct();
    result_struct.route = route_name;
    result_struct.status = cvx_status_value;
    result_struct.M = M;
    result_struct.N = N;
    result_struct.D = D;
    result_struct.num_strategies = num_strategies;

    if is_cvx_solved(cvx_status_value) && isfinite(cvx_optval_value) && cvx_optval_value > 0
        result_struct.p_guess = cvx_optval_value;
        result_struct.H_min = -log2(cvx_optval_value);
    else
        result_struct.p_guess = [];
        result_struct.H_min = [];
    end
end

function flag = is_cvx_solved(status_text)
% 判断 CVX 状态是否属于“成功求解”。
%
% 当前只把以下两类状态视为成功：
%   - Solved
%   - Inaccurate/Solved

    flag = strcmpi(status_text, 'Solved') || strcmpi(status_text, 'Inaccurate/Solved');
end
