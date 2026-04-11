%% Route4-ex 的 Matlab 单文件参考脚本
% =========================================================================
% 目标：
%   给导师和实验室提供一份“尽量贴近现有 Matlab route4 脚本阅读体验”的
%   route4-ex 单文件版本，用来承载 route4-ex Python 主线中的核心建模逻辑。
%
% 这份脚本不是把 route4-ex 的所有搜索脚本逐字翻译成 Matlab，而是把
%   src/python/qrng_routes/route4_ex/prototype.py
% 中真正决定物理模型与 SDP 结构的主流程，压缩成一个便于直接运行和逐段
% 对照的 Matlab 参考实现。
%
% 支持的实例模式：
%   1. toy
%      - 使用二元 coherent-projector POVM 构造一个小型理论例子；
%      - 用于快速验证“非对角 trusted input 会让 diagonal/full primal 分叉”。
%
%   2. apdlike
%      - 使用带位移的 APD-like 计数 POVM；
%      - 可模拟探测效率、暗计数、raw histogram，再做 coarse-graining。
%
%   3. external
%      - 从外部概率表（当前重点是 Probability.mat）读取实验概率；
%      - 再与非对角 trusted coherent inputs 结合，进入 diagonal/full primal。
%
% 支持的求解模式：
%   - diagonal : 只跑 Fock 对角测量元的对照问题
%   - full     : 只跑一般 Hermitian PSD POVM 的主问题
%   - compare  : 二者都跑，用于比较
%
% 与原始 guessprobprimal_phaseinsensitive.m 的关系：
%   - 同样是 primal SDP；
%   - 同样保留 LambdaIndices 的离散策略枚举；
%   - 同样可直接吃 Probability.mat；
%   - 但 route4-ex 不再把 trusted input 限制为 Fock 对角 Poisson 分布，
%     而是允许完整截断相干态 |alpha><alpha|；
%   - 因此 full primal 真的会用到输入态的非对角结构。
%
% 与 guessprobprimal_route4_ex_constrained.m 的关系：
%   - constrained 脚本可看作本脚本在 external 模式下的一条固定主线；
%   - 本脚本则是更一般的 route4-ex Matlab 核心版，额外支持 toy / apdlike，
%     也支持更灵活的 coarse-graining 与外部概率表注入。
%
% 当前默认配置：
%   - instance_mode = 'external'
%   - solve_mode    = 'compare'
%   - selected_mu_list = [100, 120, 140]
%   - q_selected      = [1, 0, 0]
%   - alpha_values    = [0.54, 0.66 i, -0.72]
%   - M = 6
%   - external_custom_edges = [0, 121, 132, 256]
%
% 这组默认参数对应 route4-ex external 主线上一个已 formal 验证过的例子，
% 方便导师先直接运行，再根据需要切换 instance_mode 或调整参数。
%
% 运行前提：
%   1. 已安装 CVX；
%   2. 已配置可用的 SDP 求解器，推荐 MOSEK；
%   3. 若使用 external 模式，应保证 Probability.mat 在当前目录或脚本同目录。
% =========================================================================

clear; clc;

%% ===================== 1. 运行模式与基本参数 =====================
% instance_mode：
%   选择 route4-ex 的哪一类概率实例。
%   可选：
%     - 'toy'
%     - 'apdlike'
%     - 'external'
instance_mode = 'external';

% solve_mode：
%   选择运行 diagonal primal、full primal，还是二者都跑。
%   可选：
%     - 'diagonal'
%     - 'full'
%     - 'compare'
solve_mode = 'compare';

% preferred_solver：
%   若环境支持，优先切到该求解器。推荐 'mosek'。
%   设为空字符串则使用 CVX 默认求解器。
preferred_solver = 'mosek';

% save_result_mat：
%   若设为 true，则把结果结构体保存为 .mat 文件。
save_result_mat = false;
save_result_filename = 'route4_ex_result.mat';

%% ===================== 2. route4-ex 共享物理参数 =====================
% selected_mu_list：
%   外部实验数据模式下使用的光强标签窗口。
%   在 toy/apdlike 模式下，这些标签主要用于打印摘要，不直接参与理论概率构造。
selected_mu_list = [100, 120, 140];

% q_selected：
%   生成轮目标函数中的输入权重。
%   这里默认取 [1,0,0]，即主目标只关心第一个输入上的 guessing probability；
%   其余输入仍然通过统计约束进入 SDP。
q_selected = [1, 0, 0];

% M：
%   Fock 截断维数。
%   toy/apdlike 模式下它决定概率近似的 Hilbert 空间大小；
%   external 模式下它决定 trusted coherent input 的表示维度以及 SDP 变量规模。
M = 6;

% full_mu：
%   若使用 external + Probability.mat，这里给出当前 9 个固定实验光强标签。
full_mu = [0, 20, 40, 60, 80, 100, 120, 140, 160];

% shift：
%   保留与原始 route4 Matlab 相同的行偏移接口。
shift = 0;

% alpha_values：
%   route4-ex 的 trusted coherent alphabet。
%   与原始 route4 最大不同在于：这里不只保留 Poisson 对角，而是把完整
%   截断相干态 |alpha><alpha| 送入 full primal。
alpha_values = [0.54 + 0.0i, 0.0 + 0.66i, -0.72 + 0.0i];

% prob_floor：
%   若 coarse-grained 概率中出现 0，则把它抬到 prob_floor 再归一化，
%   以减轻数值病态。
prob_floor = 1e-12;

%% ===================== 3. toy 模式参数 =====================
% probe_alpha：
%   toy coherent-projector POVM 中使用的探针相干态振幅。
probe_alpha = 0.4 + 0.4i;

%% ===================== 4. APD-like 模式参数 =====================
% displacement_alpha：
%   先对输入做位移，再使用 APD 风格计数 POVM，构成 route4-ex 的 APD-like 前端。
displacement_alpha = 0.35 + 0.35i;

% apdlike_raw_num_bins：
%   APD-like 模式中，位移计数直方图的原始 bin 数。
apdlike_raw_num_bins = 16;

% apdlike_num_outputs：
%   APD-like raw histogram 粗粒化后的输出数。
apdlike_num_outputs = 4;

% apdlike_custom_edges：
%   若非空，则用自定义边界做 coarse-graining；为空则使用等覆盖边界。
apdlike_custom_edges = [];

% detection_efficiency / dark_count_mean：
%   APD-like 对角计数模型的两个主要实验参数。
detection_efficiency = 0.6;
dark_count_mean = 0.02;

%% ===================== 5. external 模式参数 =====================
% probability_filename / probability_variable_name：
%   external 模式下默认读取的实验概率表。
probability_filename = 'Probability.mat';
probability_variable_name = 'Probability';

% external_table_already_coarse：
%   若为 true，则假设外部表已经是 coarse-grained 输出，直接进入 SDP；
%   若为 false，则还需要再做一次 coarse-graining。
external_table_already_coarse = false;

% external_num_outputs：
%   当 external 表尚未 coarse-grained 且没有给 custom_edges 时，使用等覆盖
%   边界把原始表压缩为 external_num_outputs 个输出。
external_num_outputs = 3;

% external_custom_edges：
%   external 模式下的自定义 coarse-graining 边界。
%   当前默认值 [0,121,132,256] 对应 3 输出边界：
%     - 输出1覆盖原始列   1:121
%     - 输出2覆盖原始列 122:132
%     - 输出3覆盖原始列 133:256
external_custom_edges = [0, 121, 132, 256];

% external_row_indices_override：
%   若为空，则按 selected_mu_list 在 full_mu 中的位置自动取行；
%   若非空，则直接把这里的 Matlab 1-based 行号当作外部表选行结果。
external_row_indices_override = [];

%% ===================== 6. 输入检查与初始化 =====================
% 统一把 solve_mode 转成两个布尔开关，便于后续逻辑复用。
run_diagonal_primal = strcmpi(solve_mode, 'diagonal') || strcmpi(solve_mode, 'compare');
run_full_primal = strcmpi(solve_mode, 'full') || strcmpi(solve_mode, 'compare');

% alpha_values / q_selected 必须长度一致，因为每个 trusted input 都要同时有
% 一个相干态振幅和一个生成轮权重。
D = length(alpha_values);
if length(q_selected) ~= D
    error('q_selected 的长度 (%d) 必须与 alpha_values 的长度 (%d) 一致。', ...
        length(q_selected), D);
end

% selected_mu_list 在 external 模式下还承担“映射外部概率表行”的角色，因此其长度
% 也必须与 trusted input 数量一致。为了让脚本三种模式下的摘要都统一，这里对
% 所有模式都做同样检查。
if length(selected_mu_list) ~= D
    error('selected_mu_list 的长度 (%d) 必须与 alpha_values 的长度 (%d) 一致。', ...
        length(selected_mu_list), D);
end

% q_selected 归一化。
if any(q_selected < 0)
    error('q_selected 不能包含负值。');
end
if sum(q_selected) <= 0
    error('q_selected 的总和必须为正。');
end
q_selected = q_selected / sum(q_selected);

% 检查模式字符串是否合法。
valid_instance_modes = {'toy', 'apdlike', 'external'};
if ~any(strcmpi(instance_mode, valid_instance_modes))
    error('instance_mode 必须属于 {%s}。', strjoin(valid_instance_modes, ', '));
end

valid_solve_modes = {'diagonal', 'full', 'compare'};
if ~any(strcmpi(solve_mode, valid_solve_modes))
    error('solve_mode 必须属于 {%s}。', strjoin(valid_solve_modes, ', '));
end

if M <= 0 || round(M) ~= M
    error('M 必须是正整数。');
end

fprintf('Route4-ex Matlab 参考脚本启动...\n');
fprintf('instance_mode = %s, solve_mode = %s\n', instance_mode, solve_mode);
fprintf('D = %d, M = %d\n', D, M);

% 统一预分配一些后续结果变量，便于最终汇总到 result 结构体。
rho = zeros(M, M, D);
rho_diag = zeros(D, M);
raw_probability_table = [];
raw_probability_table_before_normalization = [];
p_raw = [];
p = [];
coarse_grain_edges = [];
selected_external_rows_one_based = [];
raw_num_bins = [];
input_model = 'exact_truncated_coherent_states';
probability_model = '';
probability_path = '';
regularized_entries = 0;
distribution_only_p_guess_raw = [];
distribution_only_p_guess = [];
distribution_only_H_min_raw = [];
distribution_only_H_min = [];
input_offdiagonal_metrics = struct();
instance_extra = struct();

%% ===================== 7. 构造 trusted coherent inputs =====================
% route4-ex 的 trusted input 是完整截断相干态，而不是原 route4 的 Fock 对角
% Poisson 分布。这一块是两条路线最关键的建模分歧。
for x_idx = 1:D
    [rho_x, rho_diag_x, coeff_x] = build_truncated_coherent_density(alpha_values(x_idx), M);
    rho(:, :, x_idx) = rho_x;
    rho_diag(x_idx, :) = rho_diag_x;

    fprintf('Input %d: mu_label=%g, alpha=%.6f%+.6fi, ||coeff||_2=%.12f\n', ...
        x_idx, selected_mu_list(x_idx), real(alpha_values(x_idx)), imag(alpha_values(x_idx)), norm(coeff_x));
end

% 记录 trusted input 的非对角强度指标，方便后续报告中解释“full primal 为什么
% 真的看见了新的结构信息”。
input_offdiagonal_metrics = build_input_offdiagonal_metrics(rho);

%% ===================== 8. 按 instance_mode 生成概率表 =====================
switch lower(instance_mode)
    case 'toy'
        % toy 模式直接由一个二元 coherent-projector POVM 生成概率表。
        toy_povm = build_binary_coherent_projector_povm(M, probe_alpha);
        raw_probability_table_before_normalization = probabilities_from_povm(rho, toy_povm);
        raw_probability_table = raw_probability_table_before_normalization;
        p_raw = raw_probability_table;
        coarse_grain_edges = 0:size(p_raw, 2);
        raw_num_bins = size(raw_probability_table_before_normalization, 2);
        probability_model = 'binary_coherent_projector_povm';

        instance_extra.probe_alpha = probe_alpha;
        instance_extra.toy_povm = toy_povm;

    case 'apdlike'
        % APD-like 模式先生成带位移的 raw histogram，再做 coarse-graining。
        raw_povm = build_displaced_apd_povm( ...
            M, displacement_alpha, apdlike_raw_num_bins, detection_efficiency, dark_count_mean);
        raw_probability_table_before_normalization = probabilities_from_povm(rho, raw_povm);

        if isempty(apdlike_custom_edges)
            [p_raw, coarse_grain_edges] = coarse_grain_probability_table_equal( ...
                raw_probability_table_before_normalization, apdlike_num_outputs);
        else
            [p_raw, coarse_grain_edges] = coarse_grain_probability_table_custom( ...
                raw_probability_table_before_normalization, apdlike_custom_edges);
        end

        raw_probability_table = p_raw;
        raw_num_bins = size(raw_probability_table_before_normalization, 2);
        probability_model = 'displaced_apd_count_histogram_coarse_grained';

        instance_extra.displacement_alpha = displacement_alpha;
        instance_extra.raw_povm = raw_povm;
        instance_extra.detection_efficiency = detection_efficiency;
        instance_extra.dark_count_mean = dark_count_mean;
        instance_extra.raw_histogram_probabilities = raw_probability_table_before_normalization;

    case 'external'
        % external 模式下，当前重点是接入 Probability.mat，然后按 selected_mu_list
        % 或显式给定的行号取出所需输入窗口。
        probability_path = resolve_probability_path(probability_filename);
        fprintf('Loading external probability table from: %s\n', probability_path);

        mat_data = load(probability_path);
        [ProbData, probability_variable_name] = extract_probability_variable(mat_data, probability_variable_name);

        if isempty(external_row_indices_override)
            selected_external_rows_one_based = map_mu_labels_to_row_indices(selected_mu_list, full_mu, shift);
        else
            selected_external_rows_one_based = external_row_indices_override(:).';
        end

        if any(selected_external_rows_one_based < 1) || any(selected_external_rows_one_based > size(ProbData, 1))
            error('external 选中的行号超出 Probability 表范围。');
        end

        selected_rows = ProbData(selected_external_rows_one_based, :);
        raw_probability_table_before_normalization = selected_rows;
        raw_num_bins = size(selected_rows, 2);

        if external_table_already_coarse
            p_raw = selected_rows;
            coarse_grain_edges = [];
        else
            if isempty(external_custom_edges)
                [p_raw, coarse_grain_edges] = coarse_grain_probability_table_equal(selected_rows, external_num_outputs);
            else
                [p_raw, coarse_grain_edges] = coarse_grain_probability_table_custom(selected_rows, external_custom_edges);
            end
        end

        raw_probability_table = p_raw;
        probability_model = 'external_probability_table';

        instance_extra.probability_variable_name = probability_variable_name;
        instance_extra.external_table_shape = size(ProbData);
        instance_extra.external_selected_rows_raw = selected_rows;
        instance_extra.external_table_already_coarse = external_table_already_coarse;
end

%% ===================== 9. 概率归一化与 prob_floor 正则化 =====================
% 不管概率来自 toy、apdlike 还是 external，后续 SDP 都只依赖最终的
% coarse-grained 条件概率 p(c|x)。因此这里先逐行归一化，再按需做 prob_floor。
[p_raw, raw_row_sums_before_normalization] = normalize_probability_rows(raw_probability_table);

regularized_entries = sum(p_raw(:) == 0);
if ~isempty(prob_floor) && prob_floor > 0
    p = max(p_raw, prob_floor);
    p = p ./ sum(p, 2);
else
    p = p_raw;
end

distribution_only_p_guess_raw = distribution_only_guessing_probability(p_raw, q_selected);
distribution_only_p_guess = distribution_only_guessing_probability(p, q_selected);
if distribution_only_p_guess_raw > 0
    distribution_only_H_min_raw = -log2(distribution_only_p_guess_raw);
end
if distribution_only_p_guess > 0
    distribution_only_H_min = -log2(distribution_only_p_guess);
end

N = size(p, 2);
fprintf('Prepared coarse-grained probabilities with N = %d outputs.\n', N);

%% ===================== 10. 生成 LambdaIndices =====================
% 每个 lambda = (lambda_0, lambda_1, ..., lambda_D)。
% 其中 lambda_{x}（在代码里对应 x+1 列）表示：对输入 x，策略预测的输出标签。
fprintf('Generating strategy indices... (D=%d, N=%d)\n', D, N);
num_strategies = N^(D + 1);
args = repmat({1:N}, 1, D + 1);
[grids{1:D+1}] = ndgrid(args{:});
LambdaIndices = zeros(num_strategies, D + 1);
for idx = 1:D+1
    LambdaIndices(:, idx) = grids{idx}(:);
end

%% ===================== 11. diagonal primal（可选） =====================
diagonal_result = struct();
if run_diagonal_primal
    fprintf('Starting diagonal primal...\n');
    try_set_cvx_solver(preferred_solver);

    cvx_begin quiet
        % M_diag(n, y, lambda)：
        %   只允许 Fock 对角测量元时的变量。
        variable M_diag(M, N, num_strategies) nonnegative
        expression obj_diag
        obj_diag = 0;

        % 目标函数：
        %   sum_x q_x * sum_{lambda: lambda_x = y} <rho_diag_x, M_{y,lambda}>
        for x_idx = 1:D
            qx = q_selected(x_idx);
            rho_vec = rho_diag(x_idx, :);
            target_y_indices = LambdaIndices(:, x_idx + 1);

            for y_val = 1:N
                strategy_ids = find(target_y_indices == y_val);
                if ~isempty(strategy_ids)
                    M_sum_for_y = sum(M_diag(:, y_val, strategy_ids), 3);
                    obj_diag = obj_diag + qx * (rho_vec * M_sum_for_y);
                end
            end
        end

        maximize(obj_diag)
        subject to
            % 对每个策略，sum_y M_{y,lambda} 必须正比于单位阵。
            % 在对角模型里，这等价于所有对角元彼此相等。
            sum_over_y = sum(M_diag, 2);
            for lambda_idx = 1:num_strategies
                vec_lambda = sum_over_y(:, 1, lambda_idx);
                vec_lambda(2:end) == vec_lambda(1:end-1);
            end

            % 统计匹配：
            %   rho_diag 与总对角测量元的内积必须复现实验/理论概率。
            M_total_diag = sum(M_diag, 3);
            for x_idx = 1:D
                for y_val = 1:N
                    rho_diag(x_idx, :) * M_total_diag(:, y_val) == p(x_idx, y_val);
                end
            end
    cvx_end

    diagonal_result = build_result_struct( ...
        'route4_ex_diagonal_primal_matlab', cvx_status, cvx_optval, M, N, D, num_strategies);
    diagonal_result.measurement_constraint = 'Fock_diagonal_only';
end

%% ===================== 12. full primal（主问题） =====================
full_result = struct();
if run_full_primal
    fprintf('Starting full primal...\n');
    try_set_cvx_solver(preferred_solver);

    % N * num_strategies 个 Hermitian PSD 矩阵变量。
    num_operator_variables = N * num_strategies;

    cvx_begin quiet
        % M_full(:,:,op_idx)：
        %   每个输出 y、每个策略 lambda 对应一个 M x M 的 Hermitian PSD 矩阵。
        variable M_full(M, M, num_operator_variables) hermitian semidefinite

        % s_lambda：
        %   每个策略对应的标量权重，使得 sum_y M_{y,lambda} = s_lambda I。
        variable s_lambda(num_strategies) nonnegative

        expression obj_full
        obj_full = 0;

        % full primal 的目标函数会完整使用 trusted coherent input 的非对角结构。
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
            % 对每个策略，sum_y M_{y,lambda} = s_lambda I。
            for lambda_idx = 1:num_strategies
                expression strategy_sum(M, M)
                strategy_sum = zeros(M, M);
                for y_val = 1:N
                    op_idx = (lambda_idx - 1) * N + y_val;
                    strategy_sum = strategy_sum + M_full(:, :, op_idx);
                end
                strategy_sum == s_lambda(lambda_idx) * eye(M);
            end

            % 概率匹配约束：
            %   对每个输入 x 和输出 y，总 POVM 元必须满足
            %   Re Tr(rho_x M_y) = p(y|x)。
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
        'route4_ex_full_primal_matlab', cvx_status, cvx_optval, M, N, D, num_strategies);
    full_result.measurement_constraint = 'general_Hermitian_PSD';
    full_result.num_operator_variables = num_operator_variables;
    full_result.hermitian_scalar_count = num_operator_variables * M * M;
end

%% ===================== 13. 汇总结果 =====================
% 为了让导师可以像查看 Python 原型结果一样快速理解这次运行，这里把配置、
% 概率表、输入态诊断和 primal 结果统一收进 result 结构体。
result = struct();
result.route = 'route4_ex_matlab_script';

result.config = struct();
result.config.instance_mode = instance_mode;
result.config.solve_mode = solve_mode;
result.config.preferred_solver = preferred_solver;
result.config.selected_mu_list = selected_mu_list;
result.config.q_selected = q_selected;
result.config.M = M;
result.config.D = D;
result.config.N = N;
result.config.full_mu = full_mu;
result.config.shift = shift;
result.config.alpha_values = alpha_values;
result.config.prob_floor = prob_floor;
result.config.probe_alpha = probe_alpha;
result.config.displacement_alpha = displacement_alpha;
result.config.apdlike_raw_num_bins = apdlike_raw_num_bins;
result.config.apdlike_num_outputs = apdlike_num_outputs;
result.config.apdlike_custom_edges = apdlike_custom_edges;
result.config.detection_efficiency = detection_efficiency;
result.config.dark_count_mean = dark_count_mean;
result.config.probability_filename = probability_filename;
result.config.probability_variable_name = probability_variable_name;
result.config.external_table_already_coarse = external_table_already_coarse;
result.config.external_num_outputs = external_num_outputs;
result.config.external_custom_edges = external_custom_edges;
result.config.external_row_indices_override = external_row_indices_override;

result.instance_summary = struct();
result.instance_summary.input_model = input_model;
result.instance_summary.probability_model = probability_model;
result.instance_summary.raw_num_bins = raw_num_bins;
result.instance_summary.coarse_grain_edges = coarse_grain_edges;
result.instance_summary.raw_row_sums_before_normalization = raw_row_sums_before_normalization;
result.instance_summary.regularized_entries = regularized_entries;
result.instance_summary.input_offdiagonal_metrics = input_offdiagonal_metrics;
result.instance_summary.selected_external_rows_one_based = selected_external_rows_one_based;
result.instance_summary.external_probability_path = probability_path;
result.instance_summary.distribution_only_p_guess_raw = distribution_only_p_guess_raw;
result.instance_summary.distribution_only_H_min_raw = distribution_only_H_min_raw;
result.instance_summary.distribution_only_p_guess = distribution_only_p_guess;
result.instance_summary.distribution_only_H_min = distribution_only_H_min;
result.instance_summary.extra = instance_extra;

result.rho = rho;
result.rho_diag = rho_diag;
result.raw_probability_table_before_normalization = raw_probability_table_before_normalization;
result.probabilities_raw = p_raw;
result.probabilities = p;
result.LambdaIndices = LambdaIndices;
result.diagonal_result = diagonal_result;
result.full_result = full_result;

fprintf('\n========================================\n');
fprintf('Route4-ex Matlab script done.\n');
fprintf('instance_mode = %s\n', instance_mode);
fprintf('solve_mode    = %s\n', solve_mode);
fprintf('selected_mu_list = [%s]\n', num2str(selected_mu_list));
fprintf('alpha_values     = [');
for x_idx = 1:D
    fprintf(' %.6f%+.6fi ', real(alpha_values(x_idx)), imag(alpha_values(x_idx)));
end
fprintf(']\n');
fprintf('N outputs        = %d\n', N);
if run_full_primal && isfield(full_result, 'status')
    fprintf('Full primal status: %s\n', full_result.status);
end
if run_full_primal && isfield(full_result, 'H_min') && ~isempty(full_result.H_min)
    fprintf('Full primal H_min: %.12f bits\n', full_result.H_min);
end
if run_diagonal_primal && isfield(diagonal_result, 'status')
    fprintf('Diagonal primal status: %s\n', diagonal_result.status);
end
if run_diagonal_primal && isfield(diagonal_result, 'H_min') && ~isempty(diagonal_result.H_min)
    fprintf('Diagonal primal H_min: %.12f bits\n', diagonal_result.H_min);
end
fprintf('========================================\n');

if save_result_mat
    save(save_result_filename, 'result');
    fprintf('Saved result struct to %s\n', save_result_filename);
end

%% ===================== Local Functions =====================
function [rho_i, rho_diag_i, coeff] = build_truncated_coherent_density(alpha, M)
% 在截断 Fock 空间中构造单个相干态 |alpha> 的密度矩阵。
%
% 输入：
%   alpha : 相干态振幅
%   M     : Fock 截断维数
%
% 输出：
%   rho_i      : M x M 截断密度矩阵 |alpha><alpha|
%   rho_diag_i : rho_i 的 Fock 对角（行向量）
%   coeff      : |alpha> 在截断 Fock 基下的展开系数

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
        error('相干态展开系数的范数非法，请检查 alpha 与 M。');
    end
    coeff = coeff / coeff_norm;

    rho_i = coeff * coeff';
    rho_diag_i = real(diag(rho_i)).';
end

function metrics = build_input_offdiagonal_metrics(rho)
% 统计每个 trusted input 的非对角强度指标。
%
% 输出结构体字段：
%   - per_input(input_idx).fro_norm
%   - per_input(input_idx).offdiag_fro_norm
%   - per_input(input_idx).offdiag_over_fro
%   - per_input(input_idx).max_abs_offdiag
%   - max_offdiag_over_fro
%   - mean_offdiag_over_fro
%   - max_abs_offdiag

    D = size(rho, 3);
    ratios = zeros(1, D);
    max_entries = zeros(1, D);
    per_input = struct([]);

    for x_idx = 1:D
        rho_x = rho(:, :, x_idx);
        diagonal_x = diag(diag(rho_x));
        offdiag_x = rho_x - diagonal_x;
        fro_norm = norm(rho_x, 'fro');
        offdiag_norm = norm(offdiag_x, 'fro');
        if fro_norm == 0
            ratio = 0;
        else
            ratio = offdiag_norm / fro_norm;
        end

        abs_offdiag = abs(offdiag_x);
        abs_offdiag(1:size(abs_offdiag,1)+1:end) = 0;
        max_abs_offdiag = max(abs_offdiag(:));

        ratios(x_idx) = ratio;
        max_entries(x_idx) = max_abs_offdiag;
        per_input(x_idx).input_index = x_idx;
        per_input(x_idx).fro_norm = fro_norm;
        per_input(x_idx).offdiag_fro_norm = offdiag_norm;
        per_input(x_idx).offdiag_over_fro = ratio;
        per_input(x_idx).max_abs_offdiag = max_abs_offdiag;
    end

    metrics = struct();
    metrics.per_input = per_input;
    metrics.max_offdiag_over_fro = max(ratios);
    metrics.mean_offdiag_over_fro = mean(ratios);
    metrics.max_abs_offdiag = max(max_entries);
end

function povm = build_binary_coherent_projector_povm(M, probe_alpha)
% 构造 toy 模式中的二元 coherent-projector POVM。
%
% POVM 由两部分组成：
%   E_1 = |probe><probe|
%   E_2 = I - |probe><probe|

    [projector, ~, ~] = build_truncated_coherent_density(probe_alpha, M);
    povm = zeros(M, M, 2);
    povm(:, :, 1) = projector;
    povm(:, :, 2) = eye(M) - projector;
end

function U = build_displacement_unitary(M, displacement_alpha)
% 构造截断 Fock 空间中的位移算符 D(alpha)。
%
% 使用公式：
%   D(alpha) = exp(alpha a^\dagger - alpha^* a)

    a = destroy_operator(M);
    adag = a';
    generator = displacement_alpha * adag - conj(displacement_alpha) * a;
    U = expm(generator);
end

function a = destroy_operator(M)
% 构造 M 维 Fock 截断空间中的湮灭算符 a。
%
% a|n> = sqrt(n)|n-1>

    a = zeros(M, M);
    for n = 2:M
        a(n - 1, n) = sqrt(n - 1);
    end
end

function povm = build_apd_count_povm(M, raw_num_bins, detection_efficiency, dark_count_mean)
% 构造 APD 风格计数 POVM 的 Fock 对角近似。
%
% 对每个输入光子数 n：
%   1. 用二项分布描述探测效率引起的“被探测到的真实光子数”；
%   2. 用 Poisson 分布描述暗计数；
%   3. 对二者卷积得到最终点击计数分布；
%   4. 把超过 raw_num_bins-1 的尾部质量并到最后一个 overflow bin。

    if raw_num_bins < 2
        error('raw_num_bins 至少应为 2。');
    end
    if detection_efficiency < 0 || detection_efficiency > 1
        error('detection_efficiency 必须落在 [0,1]。');
    end
    if dark_count_mean < 0
        error('dark_count_mean 不能为负。');
    end

    exact_count_cap = raw_num_bins - 1;
    diagonal_probabilities = zeros(raw_num_bins, M);

    for n = 0:M-1
        exact_probabilities = zeros(1, exact_count_cap);

        for detected_count = 0:exact_count_cap-1
            total = 0;
            upper = min(n, detected_count);
            for transmitted = 0:upper
                binomial_weight = nchoosek(n, transmitted) * ...
                    (detection_efficiency ^ transmitted) * ...
                    ((1 - detection_efficiency) ^ (n - transmitted));

                dark_count = detected_count - transmitted;
                poisson_weight = poisson_pmf_nonnegative(dark_count, dark_count_mean);
                total = total + binomial_weight * poisson_weight;
            end
            exact_probabilities(detected_count + 1) = total;
        end

        tail_probability = max(0, 1 - sum(exact_probabilities));
        diagonal_probabilities(1:exact_count_cap, n + 1) = exact_probabilities;
        diagonal_probabilities(exact_count_cap + 1, n + 1) = tail_probability;
    end

    povm = zeros(M, M, raw_num_bins);
    for output = 1:raw_num_bins
        povm(:, :, output) = diag(diagonal_probabilities(output, :));
    end
end

function weight = poisson_pmf_nonnegative(k, lambda)
% 计算 Poisson(lambda) 在非负整数 k 处的概率质量。
%
% 对 lambda = 0 做专门处理，避免 0 * log(0) 这类数值问题。

    if k < 0 || round(k) ~= k
        weight = 0;
        return;
    end

    if lambda == 0
        if k == 0
            weight = 1;
        else
            weight = 0;
        end
        return;
    end

    log_weight = -lambda + k * log(lambda) - gammaln(k + 1);
    weight = exp(log_weight);
end

function povm = build_displaced_apd_povm(M, displacement_alpha, raw_num_bins, detection_efficiency, dark_count_mean)
% 构造 route4-ex APD-like 模式中的带位移 POVM。
%
% 先构造对角 APD count POVM，再做位移共轭：
%   E'_c = D(alpha)^\dagger E_c D(alpha)

    diagonal_povm = build_apd_count_povm(M, raw_num_bins, detection_efficiency, dark_count_mean);
    displacement = build_displacement_unitary(M, displacement_alpha);
    povm = zeros(M, M, raw_num_bins);
    for output = 1:raw_num_bins
        povm(:, :, output) = displacement' * diagonal_povm(:, :, output) * displacement;
    end
end

function probabilities = probabilities_from_povm(rho, povm)
% 计算 P(c|x) = Tr(rho_x E_c) 并逐行归一化。
%
% 输入：
%   rho  : M x M x D 的输入态密度矩阵数组
%   povm : M x M x N 的 POVM 元数组
%
% 输出：
%   probabilities : D x N 的条件概率表

    D = size(rho, 3);
    N = size(povm, 3);
    probabilities = zeros(D, N);

    for x_idx = 1:D
        for y_val = 1:N
            value = real(trace(rho(:, :, x_idx) * povm(:, :, y_val)));
            probabilities(x_idx, y_val) = value;
        end
    end

    probabilities(probabilities < 0) = 0;
    row_sums = sum(probabilities, 2);
    if any(row_sums <= 0)
        error('由 POVM 生成的概率表中存在零和行，请检查截断维数或测量参数。');
    end
    probabilities = probabilities ./ row_sums;
end

function edges = build_equal_cover_edges(num_raw_bins, num_outputs)
% 构造等覆盖 coarse-graining 边界。
%
% 第 k 个区间覆盖：
%   raw_bins(edges(k)+1 : edges(k+1))

    if num_outputs <= 0
        error('num_outputs 必须为正。');
    end
    if num_outputs > num_raw_bins
        error('num_outputs 不能超过原始 bin 数。');
    end

    edges = zeros(1, num_outputs + 1);
    for k = 0:num_outputs
        edges(k + 1) = floor(k * num_raw_bins / num_outputs);
    end

    if edges(1) ~= 0 || edges(end) ~= num_raw_bins
        error('内部错误：等覆盖边界未能覆盖全部 raw bins。');
    end
    if any(diff(edges) <= 0)
        error('内部错误：等覆盖边界必须严格递增。');
    end
end

function resolved_edges = validate_custom_edges(edges, num_raw_bins)
% 校验自定义 coarse-graining 边界是否合法。

    resolved_edges = edges(:).';
    if length(resolved_edges) < 2
        error('custom_edges 至少要包含两个端点。');
    end
    if resolved_edges(1) ~= 0 || resolved_edges(end) ~= num_raw_bins
        error('custom_edges 必须从 0 开始，并以 num_raw_bins 结束。');
    end
    if any(diff(resolved_edges) <= 0)
        error('custom_edges 必须严格递增。');
    end
end

function [coarse, edges] = coarse_grain_probability_table_equal(raw_table, num_outputs)
% 用等覆盖边界把二维 raw probability table 压缩成较少输出。

    if ndims(raw_table) ~= 2
        error('raw_table 必须是二维数组。');
    end
    edges = build_equal_cover_edges(size(raw_table, 2), num_outputs);
    coarse = zeros(size(raw_table, 1), num_outputs);
    for output = 1:num_outputs
        idx_start = edges(output) + 1;
        idx_end = edges(output + 1);
        coarse(:, output) = sum(raw_table(:, idx_start:idx_end), 2);
    end
end

function [coarse, edges] = coarse_grain_probability_table_custom(raw_table, custom_edges)
% 用显式给定的边界压缩二维 raw probability table。

    if ndims(raw_table) ~= 2
        error('raw_table 必须是二维数组。');
    end
    edges = validate_custom_edges(custom_edges, size(raw_table, 2));
    num_outputs = length(edges) - 1;
    coarse = zeros(size(raw_table, 1), num_outputs);
    for output = 1:num_outputs
        idx_start = edges(output) + 1;
        idx_end = edges(output + 1);
        coarse(:, output) = sum(raw_table(:, idx_start:idx_end), 2);
    end
end

function [normalized, row_sums] = normalize_probability_rows(table)
% 对二维概率表逐行归一化。
%
% 同时返回归一化前的行和，用于结果摘要。

    table = double(table);
    row_sums = sum(table, 2);
    if any(row_sums <= 0)
        error('概率表中存在和非正的行。');
    end
    normalized = table ./ row_sums;
end

function p_guess = distribution_only_guessing_probability(probabilities, q_selected)
% 计算 distribution-only guessing probability：
%   p_guess = sum_x q_x * max_c P(c|x)

    p_guess = q_selected * max(probabilities, [], 2);
end

function probability_path = resolve_probability_path(probability_filename)
% 优先在当前工作目录查找外部概率文件；若不存在，则回退到脚本同目录。

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

function [ProbData, resolved_name] = extract_probability_variable(mat_data, preferred_name)
% 从 load(...) 得到的结构体中提取外部概率表变量。
%
% 逻辑：
%   - 若 preferred_name 存在，则优先使用；
%   - 否则退回到首个非内部变量。

    if isfield(mat_data, preferred_name)
        ProbData = mat_data.(preferred_name);
        resolved_name = preferred_name;
        if ~ismatrix(ProbData)
            error('外部概率表变量 %s 必须是二维数组。', resolved_name);
        end
        ProbData = double(ProbData);
        return;
    end

    variable_names = fieldnames(mat_data);
    variable_names = variable_names(~startsWith(variable_names, '__'));
    if isempty(variable_names)
        error('外部 .mat 文件中没有可读取的数组变量。');
    end

    resolved_name = variable_names{1};
    warning('未找到变量 %s，改用首个变量 %s。', preferred_name, resolved_name);
    ProbData = mat_data.(resolved_name);

    if ~ismatrix(ProbData)
        error('外部概率表变量 %s 必须是二维数组。', resolved_name);
    end
    ProbData = double(ProbData);
end

function row_indices = map_mu_labels_to_row_indices(selected_mu_list, full_mu, shift)
% 把光强标签 selected_mu_list 映射到 full_mu 中的位置，再加上 shift。
%
% 返回：
%   Matlab 1-based 行号。

    if ~all(ismember(selected_mu_list, full_mu))
        error('selected_mu_list 必须是 full_mu = [%s] 的子集。', num2str(full_mu));
    end

    row_indices = zeros(1, length(selected_mu_list));
    for idx = 1:length(selected_mu_list)
        found = find(full_mu == selected_mu_list(idx), 1);
        if isempty(found)
            error('未能在 full_mu 中找到 mu label = %g', selected_mu_list(idx));
        end
        row_indices(idx) = found + shift;
    end
end

function try_set_cvx_solver(preferred_solver)
% 若 preferred_solver 非空，则尝试切换 CVX 求解器。

    if isempty(preferred_solver)
        return;
    end

    try
        eval(sprintf('cvx_solver %s', preferred_solver));
    catch solver_err
        warning('无法切换到求解器 %s，将使用 CVX 默认求解器。原因：%s', ...
            preferred_solver, solver_err.message);
    end
end

function result_struct = build_result_struct(route_name, cvx_status_value, cvx_optval_value, M, N, D, num_strategies)
% 把 CVX 的状态与目标值整理成统一结构体。

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
% 判断 CVX 是否给出了“成功求解”的状态。

    flag = strcmpi(status_text, 'Solved') || strcmpi(status_text, 'Inaccurate/Solved');
end
