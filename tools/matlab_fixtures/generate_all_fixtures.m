% generate_all_fixtures.m
% =======================
% Reproducibly generate MATLAB reference outputs for complexbox parity tests.
%
% Prerequisites:
%   * MATLAB R2019b or later
%   * MVGC2 toolbox on the path (https://github.com/lcbarnett/MVGC2)
%   * SSDI-1 toolbox on the path (https://github.com/lcbarnett/ssdi)
%   * ELPH toolbox on the path (https://gitlab.com/concog/elph)
%
% Usage (from the tools/matlab_fixtures/ directory):
%   >> generate_all_fixtures
%
% This writes .mat files into ../../tests/fixtures/, which the Python parity
% suite (pytest -m fixture) loads to assert agreement.

clear; clc;

% Resolve fixtures directory relative to this file
this_dir   = fileparts(mfilename('fullpath'));
fixture_dir = fullfile(this_dir, '..', '..', 'tests', 'fixtures');
if ~exist(fixture_dir, 'dir')
    mkdir(fixture_dir);
end

% Deterministic seed
rng(20260516, 'twister');

%% ============================================================
%% MVGC2 fixtures
%% ============================================================
fprintf('Generating MVGC2 fixtures...\n');

% A small, well-conditioned 3-variable VAR(2) — used across many tests.
A      = zeros(3, 3, 2);
A(:,:,1) = [ 0.50  0.20  0.00;
             0.10  0.30  0.15;
             0.00  0.00  0.40];
A(:,:,2) = [-0.10  0.05  0.00;
             0.00 -0.20  0.10;
             0.05  0.00  0.10];
V = diag([1.0, 0.8, 1.2]);

% var_to_autocov (algebraic Lyapunov solution)
[G, q_ac]   = var_to_autocov(A, V, -10);

% autocov_to_var round-trip
[A_back, V_back] = autocov_to_var(G);

% State-space form and CPSD
[A_ss, C, K] = var_to_ss(A, V);
[G_ss, ~]    = ss_to_autocov(A_ss, C, K, V, -10);

fres = 128;
S_var = var_to_cpsd(A, V, fres);
S_ss  = ss_to_cpsd(A_ss, C, K, V, fres);

% Pairwise-conditional GC (time and frequency)
F_var = var_to_pwcgc(A, V);
F_ss  = ss_to_pwcgc(A_ss, C, K, V);

% Simulate and re-estimate
nobs    = 50000;
ntrials = 1;
X = var_to_tsdata(A, V, nobs, ntrials);

[A_lwr, V_lwr] = tsdata_to_var(X, 2, 'LWR');
[A_ols, V_ols] = tsdata_to_var(X, 2, 'OLS');
F_lwr = var_to_pwcgc(A_lwr, V_lwr);

% Model-order selection
[moaic, mobic, mohqc, ~] = tsdata_to_varmo(X, 8, 'LWR', [], false);

% Statistics: p-value and critical value
F_test_stat = F_lwr(1, 2);  % causality 2 → 1
pval_F  = mvgc_pval(F_test_stat, 'F',  1, 1, 1, 2, nobs, ntrials);
cval_F  = mvgc_cval(0.05,        'F',  1, 1, 1, 2, nobs, ntrials);
pval_LR = mvgc_pval(F_test_stat, 'LR', 1, 1, 1, 2, nobs, ntrials);

% Whiteness and consistency
E = X(:, 3:end) - reshape(A(:,:,1), 3, 3) * X(:, 2:end-1) ...
                - reshape(A(:,:,2), 3, 3) * X(:, 1:end-2);
[dw_stat, dw_pval] = whiteness(X, E);
cons = consistency(X, E);

save(fullfile(fixture_dir, 'mvgc_basic.mat'), ...
    'A', 'V', 'G', 'A_back', 'V_back', 'A_ss', 'C', 'K', 'G_ss', ...
    'S_var', 'S_ss', 'F_var', 'F_ss', 'fres', ...
    'X', 'A_lwr', 'V_lwr', 'A_ols', 'V_ols', 'F_lwr', ...
    'moaic', 'mobic', 'mohqc', ...
    'F_test_stat', 'pval_F', 'cval_F', 'pval_LR', ...
    'dw_stat', 'dw_pval', 'cons', '-v7');

fprintf('  -> tests/fixtures/mvgc_basic.mat\n');

%% ============================================================
%% SSDI-1 fixtures
%% ============================================================
fprintf('Generating SSDI fixtures...\n');

n_obs = 5;
r_state = 8;
[A_ss2, C2, K2, rhob] = iss_rand(n_obs, r_state, 0.9);
V2 = eye(n_obs);

CAK = iss2cak(A_ss2, C2, K2);

% Random orthonormal projection (m=2)
L_rand = rand_orthonormal(n_obs, 2, 1);

dd_proxy = cak2ddx(L_rand, CAK);
dd_exact = iss2dd(L_rand, A_ss2, C2, K2);
[grad, gmag] = cak2ddxgrad(L_rand, CAK);

% Transfer function and spectral DD
fres_ss = 256;
H = ss2trfun(A_ss2, C2, K2, fres_ss);
dd_spec_arr = trfun2dd(L_rand, H);
dd_spec = dd_spec_arr(1);

% Perfect DD solutions (only when r < n)
if r_state < n_obs
    [LC, LK] = iss_perfect_dd(C2, K2, true);
    dd_LC = iss2dd(LC, A_ss2, C2, K2);
    dd_LK = iss2dd(LK, A_ss2, C2, K2);
else
    LC = []; LK = []; dd_LC = NaN; dd_LK = NaN;
end

% Gradient-descent optimisation, one run
L0 = rand_orthonormal(n_obs, 2, 1);
[dd_opt, L_opt, ~, ~, iters_opt, ~] = opt_gd2_ddx(CAK, L0, 5000, 1e-3, 2, 1e-9, false);

save(fullfile(fixture_dir, 'ssdi_basic.mat'), ...
    'A_ss2', 'C2', 'K2', 'V2', 'CAK', 'L_rand', ...
    'dd_proxy', 'dd_exact', 'grad', 'gmag', ...
    'fres_ss', 'H', 'dd_spec', ...
    'LC', 'LK', 'dd_LC', 'dd_LK', ...
    'L0', 'dd_opt', 'L_opt', 'iters_opt', '-v7');

fprintf('  -> tests/fixtures/ssdi_basic.mat\n');

%% ============================================================
%% ELPH fixtures
%% ============================================================
fprintf('Generating ELPH fixtures...\n');

% Bivariate normal with known correlation
T_mi = 50000;
rho = 0.6;
Sigma = [1.0, rho; rho, 1.0];
L_ch = chol(Sigma, 'lower');
Z = L_ch * randn(2, T_mi);

% Closed-form Gaussian MI ground truth
mi_true = -0.5 * log(1 - rho^2);

% PhiID on simple VAR data
A_phi = zeros(2, 2, 1);
A_phi(:,:,1) = [0.5, 0.0; 0.2, 0.5];
V_phi = eye(2);
X_phi = var_to_tsdata(A_phi, V_phi, 5000, 1);

% NOTE: PhiIDFull uses JIDT. If you don't have it set up locally, comment
% out the next block and run only the MVGC / SSDI fixtures.
try
    phi_struct_mmi = PhiIDFull(X_phi, 1, 'MMI');
    phi_struct_ccs = PhiIDFull(X_phi, 1, 'CCS');
    have_jidt = true;
catch
    fprintf(2, '  (Warning: JIDT not available, skipping PhiID fixtures)\n');
    phi_struct_mmi = struct(); phi_struct_ccs = struct();
    have_jidt = false;
end

% LZ76 on a fixed binary sequence (optional with ELPH)
lz_seq = double([0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1] > 0);
if exist('LZ76', 'file')
    lz_val = LZ76(logical(lz_seq));
    have_lz76 = true;
else
    fprintf(2, '  (Warning: LZ76 not available, skipping its reference value)\n');
    lz_val = NaN;
    have_lz76 = false;
end

save(fullfile(fixture_dir, 'elph_basic.mat'), ...
    'Z', 'mi_true', 'X_phi', 'phi_struct_mmi', 'phi_struct_ccs', ...
    'have_jidt', 'lz_seq', 'lz_val', 'have_lz76', '-v7');

fprintf('  -> tests/fixtures/elph_basic.mat\n');

fprintf('\nAll fixtures generated. You may now run "pytest -m fixture".\n');
