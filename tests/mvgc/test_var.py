"""Self-consistency tests for VAR estimation, simulation, and round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc


def test_specnorm_simple():
    A = np.array([[[0.5]]])
    assert mvgc.specnorm(A) == pytest.approx(0.5, abs=1e-12)


def test_var_rand_target_rho(rng):
    A = mvgc.var_rand(4, 3, rho=0.85, rng=rng)
    assert mvgc.specnorm(A) == pytest.approx(0.85, rel=1e-10)


def test_simulate_then_recover_lwr(rng):
    """LWR should recover A within Monte-Carlo noise."""
    A_true = np.zeros((2, 2, 2))
    A_true[:, :, 0] = [[0.5, 0.2], [0.1, 0.3]]
    A_true[:, :, 1] = [[-0.1, 0.05], [0.0, -0.2]]
    V_true = np.diag([1.0, 0.8])
    X, _ = mvgc.var_to_tsdata(A_true, V_true, m=200_000, N=1, rng=rng)
    fit = mvgc.tsdata_to_var(X, p=2, regmode="LWR")
    assert np.max(np.abs(fit.A - A_true)) < 0.01
    assert np.max(np.abs(fit.V - V_true)) < 0.01


def test_simulate_then_recover_ols(rng):
    A_true = np.zeros((2, 2, 2))
    A_true[:, :, 0] = [[0.5, 0.2], [0.1, 0.3]]
    A_true[:, :, 1] = [[-0.1, 0.05], [0.0, -0.2]]
    V_true = np.diag([1.0, 0.8])
    X, _ = mvgc.var_to_tsdata(A_true, V_true, m=200_000, N=1, rng=rng)
    fit = mvgc.tsdata_to_var(X, p=2, regmode="OLS")
    assert np.max(np.abs(fit.A - A_true)) < 0.01
    assert np.max(np.abs(fit.V - V_true)) < 0.01


def test_var_to_autocov_roundtrip(rng):
    A = mvgc.var_rand(3, 2, 0.6, rng=rng)
    V = mvgc.corr_rand(3, rng=rng)
    G, q = mvgc.var_to_autocov(A, V, qmax=-5)
    A_back, V_back = mvgc.autocov_to_var(G)
    # First 2 lags should be machine-precision identical
    assert np.max(np.abs(A_back[..., :2] - A)) < 1e-10
    assert np.max(np.abs(V_back - V)) < 1e-10


def test_var_to_ss_to_autocov_matches_var_to_autocov(rng):
    A = mvgc.var_rand(4, 3, 0.85, rng=rng)
    V = mvgc.corr_rand(4, rng=rng)
    G_var, _ = mvgc.var_to_autocov(A, V, qmax=-15)
    A_ss, C, K, _ = mvgc.var_to_ss(A, V)
    G_ss, _ = mvgc.ss_to_autocov(A_ss, C, K, V, qmax=-15)
    assert np.max(np.abs(G_var - G_ss)) < 1e-12


def test_var_to_cpsd_real_positive(rng):
    A = mvgc.var_rand(3, 2, 0.7, rng=rng)
    V = mvgc.corr_rand(3, rng=rng)
    S = mvgc.var_to_cpsd(A, V, fres=64)
    # CPSD diagonal must be real, non-negative
    diag = np.real(np.einsum("iif->if", S))
    assert np.all(diag >= 0)
    # Off-diagonal Hermitian symmetry
    for j in range(S.shape[2]):
        assert np.allclose(S[:, :, j], S[:, :, j].conj().T, atol=1e-12)


def test_lyap_aitr_matches_dlyap(rng):
    """Smith iteration agrees with the Schur method."""
    n = 4
    A = 0.7 * mvgc.specnorm(rng.standard_normal((n, n, 1)), 0.7)[0][:, :, 0]
    Q = rng.standard_normal((n, n))
    Q = Q @ Q.T + np.eye(n)
    X_smith, _ = mvgc.dlyap_aitr(A, Q)
    X_schur = mvgc.dlyap(A, Q)
    assert np.max(np.abs(X_smith - X_schur)) < 1e-8
