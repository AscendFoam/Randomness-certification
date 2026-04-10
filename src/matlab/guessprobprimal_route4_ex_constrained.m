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
% =========================================================================

clear; clc;

%% ===================== 1. 基本参数配置 =====================
selected_mu_list = [100, 120, 140];
q_selected = [1, 0, 0];
M = 6;

full_mu = [0, 20, 40, 60, 80, 100, 120, 140, 160];
shift = 0;

% custom_edges 使用 Python/JSON 里的“零基边界”记法：
% 第 k 个 coarse bin 覆盖原始列 custom_edges(k)+1 : custom_edges(k+1)
custom_edges = [0, 121, 132, 256];

% 当前 constrained 主线固定使用的 trusted coherent alpha
radii = [0.54, 0.66, 0.72];
phases = [0, pi/2, pi];
alpha_values = radii .* exp(1i * phases);

prob_floor = 1e-12;
probability_variable_name = 'Probability';
probability_filename = 'Probability.mat';

run_diagonal_primal = true;   % 仅用于对照；当前默认点通常 infeasible
run_full_primal = true;       % 主结果
save_result_mat = false;
save_result_filename = 'route4_ex_constrained_result.mat';

%% ===================== 2. 输入检查与初始化 =====================
if ~all(ismember(selected_mu_list, full_mu))
    error('selected_mu_list 必须是 [%s] 的子集。', num2str(full_mu));
end

D = length(selected_mu_list);
if length(q_selected) ~= D
    error('q_selected 的长度 (%d) 必须与 selected_mu_list 的长度 (%d) 一致。', ...
        length(q_selected), D);
end
if length(alpha_values) ~= D
    error('alpha_values 的长度 (%d) 必须与 selected_mu_list 的长度 (%d) 一致。', ...
        length(alpha_values), D);
end
if any(q_selected < 0)
    error('q_selected 不能包含负数。');
end
if sum(q_selected) <= 0
    error('q_selected 的总和必须为正。');
end
q_selected = q_selected / sum(q_selected);

if custom_edges(1) ~= 0 || custom_edges(end) ~= 256
    error('custom_edges 必须从 0 开始，并以 256 结束。');
end
if any(diff(custom_edges) <= 0)
    error('custom_edges 必须严格递增。');
end

N = length(custom_edges) - 1;
p_raw = zeros(D, N);
p = zeros(D, N);
rho = zeros(M, M, D);
rho_diag = zeros(D, M);

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
for i = 1:D
    [rho_i, rho_diag_i, coeff_i] = build_truncated_coherent_density(alpha_values(i), M);
    rho(:, :, i) = rho_i;
    rho_diag(i, :) = rho_diag_i;

    fprintf('Input %d: mu_label=%g, alpha=%.6f%+.6fi, ||coeff||_2=%.12f\n', ...
        i, selected_mu_list(i), real(alpha_values(i)), imag(alpha_values(i)), norm(coeff_i));
end

%% ===================== 4. 读取 Probability.mat 并 coarse-graining =====================
probability_path = resolve_probability_path(probability_filename);
fprintf('Loading probability table from: %s\n', probability_path);

mat_data = load(probability_path);
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

if ~ismatrix(ProbData) || size(ProbData, 2) ~= 256
    error('Probability.mat 中的概率表必须是二维数组，且列数为 256。');
end
if size(ProbData, 1) < max(selected_full_indices + shift)
    error('Probability.mat 行数不足，无法读取所需输入。');
end

for i = 1:D
    prob_256 = ProbData(selected_full_indices(i) + shift, :);
    row_sum = sum(prob_256);
    if row_sum <= 0
        error('第 %d 个输入选中的原始概率行总和非正。', i);
    end
    prob_256 = prob_256 / row_sum;

    for k = 1:N
        idx_start = custom_edges(k) + 1;
        idx_end = custom_edges(k + 1);
        p_raw(i, k) = sum(prob_256(idx_start:idx_end));
    end
end

regularized_entries = sum(p_raw(:) == 0);
if ~isempty(prob_floor) && prob_floor > 0
    p = max(p_raw, prob_floor);
    p = p ./ sum(p, 2);
else
    p = p_raw;
end

%% ===================== 5. 生成 LambdaIndices =====================
fprintf('Generating strategy indices... (D=%d, N=%d)\n', D, N);
num_strategies = N^(D + 1);
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

    try
        cvx_solver mosek
    catch solver_err
        warning('无法切换到 MOSEK，将使用 CVX 默认求解器。原因：%s', solver_err.message);
    end

    cvx_begin quiet
        variable M_diag(M, N, num_strategies) nonnegative
        expression obj_diag
        obj_diag = 0;

        for x_idx = 1:D
            qx = q_selected(x_idx);
            rho_vec = rho_diag(x_idx, :);
            target_y_indices = LambdaIndices(:, x_idx + 1);

            for y_val = 1:N
                k_indices = find(target_y_indices == y_val);
                if ~isempty(k_indices)
                    M_sum_for_y = sum(M_diag(:, y_val, k_indices), 3);
                    obj_diag = obj_diag + qx * (rho_vec * M_sum_for_y);
                end
            end
        end

        maximize(obj_diag)
        subject to
            sum_over_y = sum(M_diag, 2);
            for k = 1:num_strategies
                vec_k = sum_over_y(:, 1, k);
                vec_k(2:end) == vec_k(1:end-1);
            end

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

    num_operator_variables = N * num_strategies;

    try
        cvx_solver mosek
    catch solver_err
        warning('无法切换到 MOSEK，将使用 CVX 默认求解器。原因：%s', solver_err.message);
    end

    cvx_begin quiet
        variable M_full(M, M, num_operator_variables) hermitian semidefinite
        variable s_lambda(num_strategies) nonnegative
        expression obj_full
        obj_full = 0;

        for x_idx = 1:D
            qx = q_selected(x_idx);
            rho_x = rho(:, :, x_idx);
            target_y_indices = LambdaIndices(:, x_idx + 1);

            for y_val = 1:N
                strategy_ids = find(target_y_indices == y_val);
                if ~isempty(strategy_ids)
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
            for lambda_idx = 1:num_strategies
                expression strategy_sum(M, M)
                strategy_sum = zeros(M, M);
                for y_val = 1:N
                    op_idx = (lambda_idx - 1) * N + y_val;
                    strategy_sum = strategy_sum + M_full(:, :, op_idx);
                end
                strategy_sum == s_lambda(lambda_idx) * eye(M);
            end

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
    full_result.num_operator_variables = num_operator_variables;
    full_result.hermitian_scalar_count = num_operator_variables * M * M;
end

%% ===================== 8. 结果输出 =====================
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

    coeff = zeros(M, 1);
    if abs(alpha) == 0
        coeff(1) = 1;
    else
        for n = 0:M-1
            log_coeff = -abs(alpha)^2 / 2 + n * log(alpha) - 0.5 * gammaln(n + 1);
            coeff(n + 1) = exp(log_coeff);
        end
    end

    coeff_norm = norm(coeff);
    if ~isfinite(coeff_norm) || coeff_norm <= 0
        error('相干态展开系数的范数非法，请检查 alpha 和 M。');
    end
    coeff = coeff / coeff_norm;

    rho_i = coeff * coeff';
    rho_diag_i = real(diag(rho_i)).';
end

function probability_path = resolve_probability_path(probability_filename)
% 优先在当前工作目录查找 Probability.mat；若不存在，则回退到脚本同目录。

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

    flag = strcmpi(status_text, 'Solved') || strcmpi(status_text, 'Inaccurate/Solved');
end
