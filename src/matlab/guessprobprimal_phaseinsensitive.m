%% Primal SDP for the phase-insensitive APD model
% Notes:
% 1. Install CVX and a compatible SDP solver (MOSEK is strongly recommended).
% 2. Probability.mat should contain a 9x256 probability table.
% 3. This script keeps the original optimization logic, but fixes the
%    coherent-state diagonal construction so that it is numerically stable.

clear; clc;

%% ===================== 1. Basic configuration =====================
% Due to O(N^(D+1)) scaling, avoid using too many test states at once.
selected_mu_list = [100, 120, 140];
q_selected = [1/4, 1/4, 1/2];
M = 280;

full_mu = [0, 20, 40, 60, 80, 100, 120, 140, 160];

if ~all(ismember(selected_mu_list, full_mu))
    error('selected_mu_list must be a subset of [%s].', num2str(full_mu));
end

selected_full_indices = find(ismember(full_mu, selected_mu_list));
D = length(selected_mu_list);

if length(q_selected) ~= D
    error('Length of q_selected (%d) does not match number of selected test states (%d).', ...
        length(q_selected), D);
end

%% ===================== 2. Initialization =====================
N = 4;
p = zeros(D, N);
shift = 0;

%% ===================== 3. Stable construction of rho_diag =====================
% Only the diagonal of the input states is used by the optimization.
% Build the Poisson photon-number distribution directly in log-space:
%   p_n = exp(-mu) * mu^n / n!
% This avoids factorial overflow for n >= 171 and keeps each input state
% properly normalized.
rho_diag = zeros(D, M);
photon_numbers = 0:(M-1);

for i = 1:D
    mu_i = selected_mu_list(i);

    if mu_i < 0
        error('Mean photon numbers must be non-negative.');
    end

    if mu_i == 0
        diag_i = zeros(1, M);
        diag_i(1) = 1;
    else
        log_probs = -mu_i + photon_numbers * log(mu_i) - gammaln(photon_numbers + 1);
        diag_i = exp(log_probs);
    end

    if any(~isfinite(diag_i))
        error('rho_diag contains non-finite values. Check the input parameters.');
    end

    diag_i(diag_i < 0) = 0;
    trace_i = sum(diag_i);
    if ~isfinite(trace_i) || trace_i <= 0
        error('Invalid trace encountered while building rho_diag.');
    end

    rho_diag(i, :) = diag_i / trace_i;
end

if any(isnan(rho_diag(:)))
    error('rho_diag contains NaN entries after normalization.');
end

trace_error = max(abs(sum(rho_diag, 2) - 1));
if trace_error > 1e-10
    warning('rho_diag normalization error is %.3e. Consider increasing M.', trace_error);
end

%% ===================== 4. Load Probability.mat and coarse grain =====================
try
    mat_data = load('Probability.mat');
    var_names = fieldnames(mat_data);
    ProbData = mat_data.(var_names{1});
catch
    warning('Probability.mat not found. Falling back to random demo data.');
    ProbData = rand(9, 256);
    ProbData = ProbData ./ sum(ProbData, 2);
end

block_size = 256 / N;
block_size = round(block_size);
for i = 1:D
    prob_256 = ProbData(selected_full_indices(i) + shift, :);
    for k = 1:N
        idx_start = (k - 1) * block_size + 1;
        idx_end = k * block_size;
        p(i, k) = sum(prob_256(idx_start:idx_end));
    end
end

%% ===================== 5. Prepare strategy indices =====================
fprintf('Generating strategy indices... (D=%d, N=%d)\n', D, N);
num_strategies = N^(D + 1);
if num_strategies > 100000
    warning('Number of strategies is %d. Memory usage may be large.', num_strategies);
end

args = repmat({1:N}, 1, D + 1);
[grids{1:D+1}] = ndgrid(args{:});
LambdaIndices = zeros(num_strategies, D + 1);
for i = 1:D+1
    LambdaIndices(:, i) = grids{i}(:);
end

%% ===================== 6. Solve with CVX =====================
fprintf('Starting CVX solve...\n');

% MOSEK is preferred for this problem. If it is unavailable, CVX will keep
% using its current default solver.
try
    cvx_solver mosek
catch solver_err
    warning('Could not switch to MOSEK. CVX will use its default solver instead. Reason: %s', ...
        solver_err.message);
end

cvx_begin
    variable M_elements(M, N, num_strategies) nonnegative

    obj_expr = 0;
    for x_idx = 1:D
        qx = q_selected(x_idx);
        rho_vec = rho_diag(x_idx, :);
        target_y_indices = LambdaIndices(:, x_idx + 1);

        for y_val = 1:N
            k_indices = (target_y_indices == y_val);
            if any(k_indices)
                M_sum_for_y = sum(M_elements(:, y_val, k_indices), 3);
                obj_expr = obj_expr + qx * (rho_vec * M_sum_for_y);
            end
        end
    end

    maximize(obj_expr)

    subject to
        sum_over_y = sum(M_elements, 2);
        for k = 1:num_strategies
            vec_k = sum_over_y(:, 1, k);
            vec_k(2:end) == vec_k(1:end-1);
        end

        M_total = sum(M_elements, 3);
        for x_idx = 1:D
            for y_idx = 1:N
                rho_diag(x_idx, :) * M_total(:, y_idx) == p(x_idx, y_idx);
            end
        end
cvx_end

%% ===================== 7. Report results =====================
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
