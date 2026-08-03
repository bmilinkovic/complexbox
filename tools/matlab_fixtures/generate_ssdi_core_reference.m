function generate_ssdi_core_reference()
%GENERATE_SSDI_CORE_REFERENCE Small deterministic SSDI-1 parity fixture.
%
% Run with the original MVGC2 and SSDI-1 toolboxes on the MATLAB path.  The
% fixed matrices avoid any dependency on MATLAB/NumPy random-number streams.

this_dir = fileparts(mfilename('fullpath'));
fixture_dir = fullfile(this_dir, '..', '..', 'tests', 'fixtures');
if ~exist(fixture_dir, 'dir')
    mkdir(fixture_dir);
end

A = [0.40, 0.05, 0.00; ...
     0.00, 0.30, 0.02; ...
     0.01, 0.00, 0.20];
C = [1.00, 0.20, 0.00; ...
     0.00, 1.00, 0.10; ...
     0.10, 0.00, 1.00];
K = [0.10, 0.00, 0.02; ...
     0.01, 0.12, 0.00; ...
     0.00, 0.02, 0.08];
V = [1.00, 0.20, 0.10; ...
     0.20, 0.80, 0.05; ...
     0.10, 0.05, 1.20];
L = orthonormalise([1.00, 0.00; 0.20, 1.00; 0.30, -0.10]);

rho_A = specnorm(A);
rho_B = specnorm(A-K*C);
[CI, DD] = iss2ce(L, A, C, K, V);
[Gpre, Ppre] = iss2ce_precomp(A, C, K, V);
[CI_pre, DD_pre] = iss2ce(L, A, C, K, V, Gpre, Ppre);

fres = 64;
H = ss2trfun(A, C, K, fres);
[D, d] = trfun2dd(L, H);
[grad, gmag] = trfun2ddgrad(L, H);
[fres_fast, ierr_fast] = ss2fres(A, C, K, V, true);
[fres_adaptive, ierr_adaptive] = ss2fres(A, C, K, V, false);

AVAR = zeros(2, 2, 2);
AVAR(:, :, 1) = [0.45, 0.10; -0.05, 0.30];
AVAR(:, :, 2) = [-0.12, 0.03; 0.02, -0.08];
VVAR = [1.00, 0.15; 0.15, 0.90];
rho_VAR = specnorm(AVAR);
[var_fres_fast, var_ierr_fast] = var2fres(AVAR, VVAR, true);
[var_fres_adaptive, var_ierr_adaptive] = var2fres(AVAR, VVAR, false);

[dd_opt, L_opt, conv, sig, iters, hist] = ...
    opt_gd2_dds(H, L, 250, 1e-3, 2, 1e-10, true);

save(fullfile(fixture_dir, 'ssdi_core_reference.mat'), ...
    'A', 'C', 'K', 'V', 'L', 'rho_A', 'rho_B', ...
    'CI', 'DD', 'Gpre', 'Ppre', 'CI_pre', 'DD_pre', ...
    'fres', 'H', 'D', 'd', 'grad', 'gmag', ...
    'fres_fast', 'ierr_fast', 'fres_adaptive', 'ierr_adaptive', ...
    'AVAR', 'VVAR', 'rho_VAR', ...
    'var_fres_fast', 'var_ierr_fast', 'var_fres_adaptive', 'var_ierr_adaptive', ...
    'dd_opt', 'L_opt', 'conv', 'sig', 'iters', 'hist', '-v7');

fprintf('Wrote tests/fixtures/ssdi_core_reference.mat\n');
end
