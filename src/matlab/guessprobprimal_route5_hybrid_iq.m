%% Route5 的 Matlab 单文件参考脚本
% =========================================================================
% 目标：
%   给导师和实验室提供一份“拿到后可直接在 Matlab + CVX 中运行”的 Route5
%   单文件参考实现。脚本尽量保持与现有 Matlab 文件一致的阅读风格，同时在
%   逻辑上对齐 Python 中的 run_route5(...) 主流程。
%
% 本脚本对应的 Python 主入口是：
%   src/python/qrng_routes/route5/hybrid_iq.py 中的 run_route5(...)
%
% 它保留的核心逻辑是：
%   1. 生成 generalized coherent alphabet
%   2. 构造联合输入态并投影到有效支持空间
%   3. 使用 balanced beamsplitter + dual-homodyne / IQ coarse-graining
%      计算中心测量概率
%   4. 计算 raw min-entropy 排序
%   5. 用 single-device prepare-and-measure MDI SDP 做正式认证
%
% 与 route4 / route4-ex 的关系：
%   - route4 依赖一维 Probability.mat 和 phase-insensitive / diagonal 模型；
%   - route4-ex 依赖 external / toy / APD-like 概率后端；
%   - route5 则是“广义相干态字母表 + IQ 平面物理分箱 + 单设备 SDP”。
%
% 这份脚本刻意只对齐 Route5 的“single”正式运行流程，不把 Python 里的
% partition-search / alphabet-search 调度器整体翻译过来。原因是：
%   - 这份 Matlab 文件主要服务于导师直接检查 Route5 的主协议逻辑；
%   - 搜索器属于外围调度层，而不是 Route5 的物理/SDP 核心。
%
% 当前默认参数直接对应一条已经 formal 验证过的强点：
%   cutoff            = 4
%   radii             = [0.0, 0.85, 1.25]
%   phase_values      = 8 个均匀相位
%   num_x_bins        = 6
%   num_p_bins        = 2
%   quadrature_range  = 1.8
%   boundary_gamma    = 1.0
%   max_inputs_to_certify = 3
%
% 对应的 Python 结果文件可参考：
%   output/qrng_routes/route5_local_refine_queue_mosek_v1/r0.0000_0.8500_1.2500.json
%
% 运行前提：
%   1. 已安装 CVX；
%   2. 已配置可用的 SDP 求解器，推荐 MOSEK；
%   3. 不依赖 Probability.mat，本脚本内部直接按 Route5 的物理模型生成概率。
% =========================================================================

clear; clc;

%% ===================== 1. 运行与保存选项 =====================
% preferred_solver：
%   优先尝试的 CVX 求解器。若环境中已有 MOSEK，建议保持为 'mosek'。
preferred_solver = 'mosek';

% verbose_solver：
%   是否让 CVX 输出详细日志。默认 false，对应 quiet 模式。
verbose_solver = false;

% save_result_mat：
%   若设为 true，则把结果结构体保存为 .mat 文件，方便后续复查。
save_result_mat = false;
save_result_filename = 'route5_hybrid_iq_result.mat';

%% ===================== 2. Route5 共享物理参数 =====================
% cutoff：
%   截断 Fock 维数。Route5 当前主线常用较低 cutoff（如 4），
%   因为它会先把输入态压缩到实际支持空间，再进入正式 SDP。
cutoff = 4;

% alpha_values：
%   如果非空，则直接把这里的复振幅列表当作 trusted coherent alphabet。
%   如果留空，则使用 radii × phase_values 自动生成，并去重。
alpha_values = [];

% radii / phase_values：
%   当 alpha_values 为空时，用这组半径和相位生成 generalized coherent alphabet。
%   默认就是当前已 formal 验证过的 Route5 强点。
radii = [0.0, 0.85, 1.25];
phase_values = 2 * pi * (0:7) / 8;

% num_x_bins / num_p_bins：
%   IQ 平面上的轴对齐分箱结构。总输出数为 num_x_bins × num_p_bins。
num_x_bins = 6;
num_p_bins = 2;

% x_bounds / p_bounds：
%   若留空，则按 power_spaced_bounds 自动生成；
%   若显式给出，则直接使用外部边界。
x_bounds = [];
p_bounds = [];

% quadrature_range / boundary_gamma：
%   用于自动生成 IQ 分箱边界。
%   gamma = 1 时得到等间距有限边界；
%   gamma > 1 会让中心区域更细；
%   gamma < 1 会让边缘区域更细。
quadrature_range = 1.8;
boundary_gamma = 1.0;

% num_quadrature_nodes：
%   dual-homodyne 概率数值积分节点数。
%   留空时使用 Route3/Route5 主线一致的默认策略：max(400, 60*cutoff)。
num_quadrature_nodes = [];

% max_inputs_to_certify：
%   只对 raw_H_min 最高的前若干个 target inputs 做正式 SDP 认证。
%   当前强点文件使用的是 3。
max_inputs_to_certify = 3;

%% ===================== 3. 输入检查与初始化 =====================
if cutoff <= 0 || round(cutoff) ~= cutoff
    error('cutoff 必须是正整数。');
end

if ~isempty(alpha_values)
    if ~isvector(alpha_values) || isempty(alpha_values)
        error('alpha_values 若提供，必须是非空向量。');
    end
    local_alphas = deduplicate_alphas(alpha_values(:).');
else
    if isempty(radii) || isempty(phase_values)
        error('当 alpha_values 为空时，radii 和 phase_values 必须非空。');
    end
    local_alphas = build_alpha_values_from_grid(radii, phase_values);
end

if num_x_bins < 2 || num_p_bins < 2
    error('num_x_bins 和 num_p_bins 都至少应为 2。');
end

fprintf('Route5 Matlab 参考脚本启动...\n');
fprintf('cutoff = %d\n', cutoff);
fprintf('num_local_states = %d\n', length(local_alphas));
fprintf('num_x_bins = %d, num_p_bins = %d\n', num_x_bins, num_p_bins);

%% ===================== 4. 构造 generalized coherent alphabet =====================
% local_states：
%   每个本地相干态的完整截断密度矩阵。
% local_kets：
%   相干态对应的截断态矢量。
num_local_states = length(local_alphas);
local_states = cell(1, num_local_states);
local_kets = cell(1, num_local_states);

for idx = 1:num_local_states
    [rho_i, ~, ket_i] = build_truncated_coherent_density(local_alphas(idx), cutoff);
    local_states{idx} = rho_i;
    local_kets{idx} = ket_i;
    fprintf('Local state %d: alpha = %.6f%+.6fi\n', ...
        idx, real(local_alphas(idx)), imag(local_alphas(idx)));
end

% local_basis：
%   这些本地 coherent states 张成的正交归一支持空间基。
local_basis = support_basis_from_vectors(local_kets);
local_rank = size(local_basis, 2);

% reduced_local_states：
%   将本地输入态投影到支持空间后的有效态。
reduced_local_states = cell(1, num_local_states);
for idx = 1:num_local_states
    reduced_local_states{idx} = project_density_to_basis(local_states{idx}, local_basis);
end

% joint_states / labels：
%   构造所有本地态对的联合输入态。
joint_states = {};
labels = zeros(num_local_states^2, 2);
row_counter = 1;
for x_idx = 1:num_local_states
    for y_idx = 1:num_local_states
        joint_states{row_counter} = kron(reduced_local_states{x_idx}, reduced_local_states{y_idx});
        labels(row_counter, :) = [x_idx - 1, y_idx - 1];
        row_counter = row_counter + 1;
    end
end

joint_basis = kron(local_basis, local_basis);
joint_dim = size(joint_states{1}, 1);
local_operator_span_rank = operator_span_rank_from_states(reduced_local_states);
joint_operator_span_rank = operator_span_rank_from_states(joint_states);

%% ===================== 5. 构造 IQ 分箱边界 =====================
if isempty(x_bounds)
    resolved_x_bounds = power_spaced_bounds(num_x_bins, quadrature_range, boundary_gamma);
else
    resolved_x_bounds = x_bounds(:).';
end

if isempty(p_bounds)
    resolved_p_bounds = power_spaced_bounds(num_p_bins, quadrature_range, boundary_gamma);
else
    resolved_p_bounds = p_bounds(:).';
end

%% ===================== 6. 生成 dual-homodyne / IQ 概率 =====================
[probabilities, output_labels, x_bounds_out, p_bounds_out] = dual_homodyne_probabilities_route5( ...
    joint_states, joint_basis, cutoff, num_x_bins, num_p_bins, ...
    resolved_x_bounds, resolved_p_bounds, quadrature_range, num_quadrature_nodes);

raw_h = -log2(max(max(probabilities, [], 2), 1e-15));
[raw_best_H_min, raw_best_index_one_based] = max(raw_h);

candidate_order = sort_target_indices_desc(raw_h);
if ~isempty(max_inputs_to_certify)
    candidate_order = candidate_order(1:min(max_inputs_to_certify, length(candidate_order)));
end

%% ===================== 7. 正式认证 target inputs =====================
[best_result, target_scan] = certify_target_inputs_route5( ...
    joint_states, probabilities, labels, local_alphas, candidate_order, preferred_solver, verbose_solver);

%% ===================== 8. 汇总结果 =====================
result = struct();
result.route = 'route5_cv_generalized_iq_matlab';

result.config = struct();
result.config.preferred_solver = preferred_solver;
result.config.cutoff = cutoff;
result.config.alpha_values = serialize_complex_vector(local_alphas);
result.config.radii = radii;
result.config.phase_values = phase_values;
result.config.num_local_states = num_local_states;
result.config.num_x_bins = num_x_bins;
result.config.num_p_bins = num_p_bins;
result.config.num_outputs = size(probabilities, 2);
result.config.quadrature_range = quadrature_range;
result.config.boundary_gamma = boundary_gamma;
result.config.num_quadrature_nodes = resolve_num_quadrature_nodes(cutoff, num_quadrature_nodes);
result.config.max_inputs_to_certify = max_inputs_to_certify;

result.instance_summary = struct();
result.instance_summary.local_rank = local_rank;
result.instance_summary.local_operator_span_rank = local_operator_span_rank;
result.instance_summary.local_operator_space_dim = local_rank^2;
result.instance_summary.joint_dim = joint_dim;
result.instance_summary.operator_span_rank = joint_operator_span_rank;
result.instance_summary.operator_space_dim = joint_dim^2;
result.instance_summary.output_labels = output_labels;
result.instance_summary.x_bounds = x_bounds_out;
result.instance_summary.p_bounds = p_bounds_out;

result.probabilities = probabilities;
result.labels = labels;
result.raw_best_target_index = raw_best_index_one_based - 1;
result.raw_best_target = labels(raw_best_index_one_based, :);
result.raw_best_target_alphas = serialize_complex_vector([ ...
    local_alphas(labels(raw_best_index_one_based, 1) + 1), ...
    local_alphas(labels(raw_best_index_one_based, 2) + 1)]);
result.raw_best_H_min = raw_best_H_min;
result.target_scan = target_scan;
result.num_inputs = length(joint_states);
result.num_inputs_certified = length(target_scan);

result.best = best_result;
result.H_min = best_result.H_min;
result.p_guess = best_result.p_guess;
result.certified_best_target_index = best_result.target_index;
result.certified_best_target = best_result.target_input;
result.certified_best_target_alphas = best_result.target_alphas;

fprintf('\n========================================\n');
fprintf('Route5 Matlab script done.\n');
fprintf('num_local_states = %d\n', num_local_states);
fprintf('num_inputs       = %d\n', result.num_inputs);
fprintf('num_outputs      = %d\n', result.config.num_outputs);
fprintf('raw best target  = [%d, %d]\n', result.raw_best_target(1), result.raw_best_target(2));
fprintf('raw best H_min   = %.12f bits\n', result.raw_best_H_min);
if isfield(best_result, 'status')
    fprintf('formal status    = %s\n', best_result.status);
end
if isfield(best_result, 'H_min') && ~isempty(best_result.H_min)
    fprintf('formal H_min     = %.12f bits\n', best_result.H_min);
end
fprintf('========================================\n');

if save_result_mat
    save(save_result_filename, 'result');
    fprintf('Saved result struct to %s\n', save_result_filename);
end

%% ===================== Local Functions =====================
function unique_alphas = deduplicate_alphas(alpha_values)
% 去除复振幅列表中的重复值，保持原始顺序。
    tol = 1e-12;
    unique_alphas = [];
    for idx = 1:length(alpha_values)
        alpha = alpha_values(idx);
        is_duplicate = false;
        for j = 1:length(unique_alphas)
            if abs(alpha - unique_alphas(j)) <= tol
                is_duplicate = true;
                break;
            end
        end
        if ~is_duplicate
            unique_alphas(end + 1) = alpha; %#ok<AGROW>
        end
    end
end

function alpha_values = build_alpha_values_from_grid(radii, phase_values)
% 用 radii × phase_values 生成 generalized coherent alphabet，并去重。
    alpha_values = [];
    for r_idx = 1:length(radii)
        for p_idx = 1:length(phase_values)
            alpha_values(end + 1) = radii(r_idx) * exp(1i * phase_values(p_idx)); %#ok<AGROW>
        end
    end
    alpha_values = deduplicate_alphas(alpha_values);
end

function [rho_i, rho_diag_i, ket] = build_truncated_coherent_density(alpha, M)
% 在截断 Fock 空间中构造相干态 |alpha>、其密度矩阵和对角元。
    ket = zeros(M, 1);
    if abs(alpha) == 0
        ket(1) = 1;
    else
        for n = 0:M-1
            log_coeff = -abs(alpha)^2 / 2 + n * log(alpha) - 0.5 * gammaln(n + 1);
            ket(n + 1) = exp(log_coeff);
        end
    end
    ket_norm = norm(ket);
    if ~isfinite(ket_norm) || ket_norm <= 0
        error('相干态展开系数范数非法，请检查 alpha 与 M。');
    end
    ket = ket / ket_norm;
    rho_i = ket * ket';
    rho_diag_i = real(diag(rho_i)).';
end

function basis = support_basis_from_vectors(vectors)
% 计算给定态矢量张成子空间的正交归一基。
    stacked = [];
    for idx = 1:length(vectors)
        stacked = [stacked, vectors{idx}(:)]; %#ok<AGROW>
    end
    [u, singular_values_matrix, ~] = svd(stacked, 'econ');
    singular_values = diag(singular_values_matrix);
    rank_value = sum(singular_values > 1e-9);
    basis = u(:, 1:rank_value);
end

function projected = project_density_to_basis(rho, basis)
% 将高维密度矩阵投影到给定的正交归一基。
    projected = basis' * rho * basis;
    projected = 0.5 * (projected + projected');
end

function rank_value = operator_span_rank_from_states(states)
% 计算一组算符张成空间的秩。
    flattened = [];
    for idx = 1:length(states)
        flattened = [flattened; reshape(states{idx}, 1, [])]; %#ok<AGROW>
    end
    singular_values = svd(flattened, 'econ');
    rank_value = sum(singular_values > 1e-9);
end

function bounds = power_spaced_bounds(num_bins, finite_range, gamma)
% 构造对称的幂次间隔 IQ 边界。
    if num_bins < 2
        error('num_bins 至少应为 2。');
    end
    if finite_range <= 0
        error('finite_range 必须为正。');
    end
    if gamma <= 0
        error('gamma 必须为正。');
    end

    if num_bins == 2
        bounds = [-inf, 0.0, inf];
        return;
    end

    normalized = linspace(-1.0, 1.0, num_bins + 1);
    bounds = sign(normalized) .* (abs(normalized) .^ gamma) * finite_range;
    bounds(1) = -inf;
    bounds(end) = inf;
    bounds(abs(bounds) < 1e-12) = 0.0;
end

function [probabilities, output_labels, x_edges, p_edges] = dual_homodyne_probabilities_route5( ...
    joint_states, joint_basis, cutoff, num_x_bins, num_p_bins, x_bounds, p_bounds, quadrature_range, num_nodes)
% Route5 的中心测量概率：
%   balanced beamsplitter + X/P quadrature POVM + IQ coarse-graining。
    [full_povm, output_labels, x_edges, p_edges] = dual_homodyne_povm_route5( ...
        cutoff, num_x_bins, num_p_bins, x_bounds, p_bounds, quadrature_range, num_nodes);
    reduced_povm = project_povm_to_basis_list(full_povm, joint_basis);
    probabilities = measurement_probabilities_from_states(joint_states, reduced_povm);
end

function [povm, labels, x_edges, p_edges] = dual_homodyne_povm_route5( ...
    cutoff, num_x_bins, num_p_bins, x_bounds, p_bounds, quadrature_range, num_nodes)
% 构造 Route5 的 full-space POVM：B^\dagger (F_x ⊗ F_p) B。
    if isempty(x_bounds)
        x_edges = power_spaced_bounds(num_x_bins, quadrature_range, 1.0);
    else
        x_edges = x_bounds;
    end
    if isempty(p_bounds)
        p_edges = power_spaced_bounds(num_p_bins, quadrature_range, 1.0);
    else
        p_edges = p_bounds;
    end

    x_povms = quadrature_povms_from_bounds_route5(cutoff, 0.0, x_edges, num_nodes);
    p_povms = quadrature_povms_from_bounds_route5(cutoff, pi / 2.0, p_edges, num_nodes);
    beamsplitter = balanced_beamsplitter_unitary_route5(cutoff);

    povm = {};
    labels = zeros(num_x_bins * num_p_bins, 2);
    row_counter = 1;
    for x_idx = 1:length(x_povms)
        for p_idx = 1:length(p_povms)
            output_effect = kron(x_povms{x_idx}, p_povms{p_idx});
            povm{row_counter} = beamsplitter' * output_effect * beamsplitter;
            labels(row_counter, :) = [x_idx - 1, p_idx - 1];
            row_counter = row_counter + 1;
        end
    end
end

function projected = project_povm_to_basis_list(povm, basis)
% 将 POVM 列表投影到输入态实际支持空间。
    projected = cell(1, length(povm));
    for idx = 1:length(povm)
        reduced = basis' * povm{idx} * basis;
        projected{idx} = 0.5 * (reduced + reduced');
    end
end

function probabilities = measurement_probabilities_from_states(states, povm)
% 计算 Born 概率 P(c|s) = Tr(rho_s E_c)。
    probabilities = zeros(length(states), length(povm));
    for s_idx = 1:length(states)
        rho_s = states{s_idx};
        for c_idx = 1:length(povm)
            probabilities(s_idx, c_idx) = real(trace(povm{c_idx} * rho_s));
        end
    end
end

function povms = quadrature_povms_from_bounds_route5(cutoff, theta, bounds, num_nodes)
% 从连续边界构造 quadrature POVM。
    edges = bounds(:).';
    if length(edges) < 2
        error('bounds 至少需要两个边界。');
    end
    if any(diff(edges) < 0)
        error('bounds 必须单调不减。');
    end

    nodes_count = resolve_num_quadrature_nodes(cutoff, num_nodes);
    nodes = quadrature_hermite_data_route5(cutoff, nodes_count);
    nodes_x = nodes.nodes;

    num_bins = length(edges) - 1;
    masks = zeros(num_bins, length(nodes_x));
    for idx = 1:num_bins
        lower = edges(idx);
        upper = edges(idx + 1);
        mask = true(size(nodes_x));
        if isfinite(lower)
            mask = mask & (nodes_x >= lower);
        end
        if isfinite(upper)
            if idx == num_bins
                mask = mask & (nodes_x <= upper);
            else
                mask = mask & (nodes_x < upper);
            end
        end
        masks(idx, mask) = 1.0;
    end

    povms = quadrature_povms_from_node_masks_route5(cutoff, theta, masks, nodes_count);
end

function povms = quadrature_povms_from_node_masks_route5(cutoff, theta, masks, num_nodes)
% 用高斯-厄米特节点掩码构造粗粒化 quadrature POVM。
    data = quadrature_hermite_data_route5(cutoff, num_nodes);
    weights = data.weights;
    values = data.values;

    if size(masks, 2) ~= length(weights)
        error('node masks 的列数必须和 num_nodes 对应。');
    end

    weighted_values = values .* sqrt(weights(:).');
    num_bins = size(masks, 1);
    base_elements = cell(1, num_bins);
    for idx = 1:num_bins
        masked_values = weighted_values .* sqrt(masks(idx, :));
        base_elements{idx} = masked_values * masked_values.';
    end

    number_indices = (0:cutoff-1).';
    phase = exp(-1i * theta * number_indices);
    rotated = cell(1, num_bins);
    for idx = 1:num_bins
        element = base_elements{idx};
        rotated{idx} = (phase * phase') .* element;
    end

    povms = complete_povm_via_whitening_route5(rotated);
end

function corrected = complete_povm_via_whitening_route5(povm)
% 通过白化变换数值修正 POVM 完备性。
    total = zeros(size(povm{1}));
    for idx = 1:length(povm)
        total = total + povm{idx};
    end
    total = 0.5 * (total + total');
    [basis, values] = eig(total);
    eigenvalues = real(diag(values));
    clipped = max(eigenvalues, 1e-12);
    inv_sqrt = basis * diag(1 ./ sqrt(clipped)) * basis';

    corrected = cell(1, length(povm));
    for idx = 1:length(povm)
        element = inv_sqrt * povm{idx} * inv_sqrt';
        corrected{idx} = 0.5 * (element + element');
    end
end

function data = quadrature_hermite_data_route5(dimension, num_nodes)
% 计算高斯-厄米特求积节点、权重和归一化厄米特多项式值。
    [nodes, weights] = roots_hermite_golub_welsch(num_nodes);
    values = zeros(dimension, num_nodes);
    prefactor = pi^(-0.25);

    hermite_values = zeros(dimension, num_nodes);
    hermite_values(1, :) = 1;
    if dimension >= 2
        hermite_values(2, :) = 2 * nodes.';
    end
    for n = 3:dimension
        hermite_values(n, :) = 2 * nodes.' .* hermite_values(n - 1, :) - 2 * (n - 2) * hermite_values(n - 2, :);
    end

    for n = 0:dimension-1
        norm_factor = prefactor / sqrt((2^n) * factorial(n));
        values(n + 1, :) = norm_factor * hermite_values(n + 1, :);
    end

    data = struct();
    data.nodes = nodes.';
    data.weights = weights.';
    data.values = values;
end

function [nodes, weights] = roots_hermite_golub_welsch(num_nodes)
% 用 Golub-Welsch 方法计算 physicists' Hermite 多项式的节点与权重。
    if num_nodes <= 0 || round(num_nodes) ~= num_nodes
        error('num_nodes 必须是正整数。');
    end

    beta = sqrt((1:num_nodes-1) / 2);
    jacobi = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(jacobi);
    nodes = diag(D);
    [nodes, order] = sort(nodes);
    V = V(:, order);
    weights = sqrt(pi) * (V(1, :) .^ 2).';
end

function n = resolve_num_quadrature_nodes(cutoff, requested)
% Route3/Route5 默认节点数策略：max(400, 60*cutoff)。
    if isempty(requested)
        n = max(400, 60 * cutoff);
    else
        n = requested;
    end
end

function U = balanced_beamsplitter_unitary_route5(dimension)
% 构造 50:50 balanced beamsplitter 的幺正算符。
    a = destroy_operator_route5(dimension);
    b = destroy_operator_route5(dimension);
    generator = (pi / 4.0) * (kron(create_operator_route5(dimension), b) - kron(a, create_operator_route5(dimension)));
    U = expm(generator);
end

function a = destroy_operator_route5(dimension)
% 构造湮灭算符 a。
    a = zeros(dimension, dimension);
    for n = 2:dimension
        a(n - 1, n) = sqrt(n - 1);
    end
end

function adag = create_operator_route5(dimension)
% 构造产生算符 a^\dagger。
    adag = destroy_operator_route5(dimension)';
end

function [best_result, scan] = certify_target_inputs_route5(states, probabilities, labels, local_alphas, target_indices, preferred_solver, verbose_solver)
% 对若干 target inputs 做正式 SDP 认证，并返回最佳结果与完整扫描。
    scan = struct([]);
    raw_h = -log2(max(max(probabilities, [], 2), 1e-15));
    best_result = empty_target_result_route5();

    for idx = 1:length(target_indices)
        target_index_one_based = target_indices(idx);
        current = solve_single_device_guessing_route5(states, probabilities, target_index_one_based, preferred_solver, verbose_solver);
        current.target_index = target_index_one_based - 1;
        current.target_input = labels(target_index_one_based, :);
        current.target_alphas = serialize_complex_vector([ ...
            local_alphas(labels(target_index_one_based, 1) + 1), ...
            local_alphas(labels(target_index_one_based, 2) + 1)]);
        current.raw_H_min = raw_h(target_index_one_based);
        current.raw_p_guess = max(probabilities(target_index_one_based, :));
        current = normalize_target_result_route5(current);

        if isempty(scan)
            scan = current;
        else
            scan(end + 1) = current; %#ok<AGROW>
        end

        if isempty(best_result.H_min)
            best_result = current;
        elseif ~isempty(current.H_min) && (isempty(best_result.H_min) || current.H_min > best_result.H_min)
            best_result = current;
        end
    end
end

function result_struct = solve_single_device_guessing_route5(states, probabilities, target_input_one_based, preferred_solver, verbose_solver)
% Route5 正式认证后端：single-device prepare-and-measure MDI SDP。
    num_inputs = length(states);
    num_outputs = size(probabilities, 2);
    dimension = size(states{1}, 1);
    identity_matrix = eye(dimension);
    num_operator_variables = num_outputs * num_outputs;

    result_struct = empty_target_result_route5();
    result_struct.solver = preferred_solver;

    try
        try_set_cvx_solver_route5(preferred_solver);

        if verbose_solver
            cvx_begin
        else
            cvx_begin quiet
        end
            variable M_ops(dimension, dimension, num_operator_variables) hermitian semidefinite
            variable p_e(num_outputs) nonnegative

            rho_star = states{target_input_one_based};
            expression objective_value
            objective_value = 0;
            for c_idx = 1:num_outputs
                op_idx = operator_index_route5(c_idx, c_idx, num_outputs);
                objective_value = objective_value + real(trace(M_ops(:, :, op_idx) * rho_star));
            end
            maximize(objective_value)
            subject to
                for s_idx = 1:num_inputs
                    rho_s = states{s_idx};
                    for c_idx = 1:num_outputs
                        expression stats_sum
                        stats_sum = 0;
                        for e_idx = 1:num_outputs
                            op_idx = operator_index_route5(c_idx, e_idx, num_outputs);
                            stats_sum = stats_sum + real(trace(M_ops(:, :, op_idx) * rho_s));
                        end
                        stats_sum == probabilities(s_idx, c_idx);
                    end
                end

                for e_idx = 1:num_outputs
                    expression complete_sum(dimension, dimension)
                    complete_sum = zeros(dimension, dimension);
                    for c_idx = 1:num_outputs
                        op_idx = operator_index_route5(c_idx, e_idx, num_outputs);
                        complete_sum = complete_sum + M_ops(:, :, op_idx);
                    end
                    complete_sum == p_e(e_idx) * identity_matrix;
                end

                sum(p_e) == 1;
        cvx_end

        result_struct.status = cvx_status;
        if is_cvx_solved_route5(cvx_status) && isfinite(cvx_optval) && cvx_optval > 0
            result_struct.p_guess = cvx_optval;
            result_struct.H_min = -log2(cvx_optval);
        else
            result_struct.p_guess = [];
            result_struct.H_min = [];
        end
    catch solve_err
        result_struct.status = 'ERROR';
        result_struct.error_message = solve_err.message;
        result_struct.p_guess = [];
        result_struct.H_min = [];
    end
end

function result_struct = empty_target_result_route5()
% 构造一个字段固定、顺序固定的 target 结果结构体。
    result_struct = struct( ...
        'solver', [], ...
        'status', [], ...
        'error_message', [], ...
        'p_guess', [], ...
        'H_min', [], ...
        'target_index', [], ...
        'target_input', [], ...
        'target_alphas', struct([]), ...
        'raw_H_min', [], ...
        'raw_p_guess', []);
end

function normalized = normalize_target_result_route5(input_struct)
% 把 target 结果规范化成固定字段顺序，避免 Matlab struct array 拼接报错。
    normalized = empty_target_result_route5();
    field_names = fieldnames(normalized);
    for idx = 1:length(field_names)
        field_name = field_names{idx};
        if isfield(input_struct, field_name)
            normalized.(field_name) = input_struct.(field_name);
        end
    end
end

function op_idx = operator_index_route5(c_idx, e_idx, num_outputs)
% 将 (c,e) 二元索引映射到 3D 变量切片索引。
    op_idx = (e_idx - 1) * num_outputs + c_idx;
end

function order = sort_target_indices_desc(values)
% 返回按数值降序排列的索引。
    [~, order] = sort(values, 'descend');
end

function serialized = serialize_complex_vector(alpha_values)
% 把复振幅向量转成便于查看的结构体数组。
    serialized = struct([]);
    for idx = 1:length(alpha_values)
        alpha = alpha_values(idx);
        serialized(idx).real = real(alpha); %#ok<AGROW>
        serialized(idx).imag = imag(alpha);
        serialized(idx).abs = abs(alpha);
        serialized(idx).phase = angle(alpha);
    end
end

function try_set_cvx_solver_route5(preferred_solver)
% 尝试切换到指定的 CVX 求解器。
    if isempty(preferred_solver)
        return;
    end
    try
        eval(sprintf('cvx_solver %s', preferred_solver));
    catch solver_err
        warning('无法切换到 %s，将使用 CVX 默认求解器。原因：%s', preferred_solver, solver_err.message);
    end
end

function flag = is_cvx_solved_route5(status_text)
% 判断 CVX 是否成功求解。
    flag = strcmpi(status_text, 'Solved') || strcmpi(status_text, 'Inaccurate/Solved');
end
