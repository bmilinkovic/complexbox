"""Granger-causality consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc


def test_var_gc_matches_ss_gc(rng):
    """Time-domain GC computed from VAR coefficients vs from companion SS."""
    A = mvgc.var_rand(4, 3, 0.85, rng=rng)
    V = mvgc.corr_rand(4, rng=rng)
    A_ss, C, K, _ = mvgc.var_to_ss(A, V)
    F_var = mvgc.var_to_pwcgc(A, V)
    F_ss = mvgc.ss_to_pwcgc(A_ss, C, K, V)
    diff = np.nanmax(np.abs(F_var - F_ss))
    assert diff < 1e-10


def test_pwcgc_consistent_with_mvgc(rng):
    A = mvgc.var_rand(5, 3, 0.85, rng=rng)
    V = mvgc.corr_rand(5, rng=rng)
    F_full = mvgc.var_to_pwcgc(A, V)
    # Compare an off-diagonal entry against direct mvgc call
    target, source = 0, 2
    F_pair = mvgc.var_to_mvgc(A, V, x=[target], y=[source])
    assert F_full[target, source] == pytest.approx(F_pair, abs=1e-12)


def test_pval_cval_consistency(rng):
    """If GC = critical value, p-value should equal alpha."""
    alpha = 0.05
    cval = mvgc.mvgc_cval(alpha, "F", nx=1, ny=1, nz=2, p=3, m=5000)
    pval = mvgc.mvgc_pval(cval, "F", nx=1, ny=1, nz=2, p=3, m=5000)
    assert pval == pytest.approx(alpha, abs=1e-10)
    cval = mvgc.mvgc_cval(alpha, "LR", nx=1, ny=1, nz=2, p=3, m=5000)
    pval = mvgc.mvgc_pval(cval, "LR", nx=1, ny=1, nz=2, p=3, m=5000)
    assert pval == pytest.approx(alpha, abs=1e-10)


def test_zero_gc_under_independence(rng):
    """Independent VAR processes should have ~ zero GC."""
    n, p = 3, 2
    # Make a block-diagonal VAR so the variables are independent
    A1 = mvgc.var_rand(1, p, 0.5, rng=rng)
    A2 = mvgc.var_rand(1, p, 0.5, rng=rng)
    A3 = mvgc.var_rand(1, p, 0.5, rng=rng)
    A = np.zeros((n, n, p))
    A[0, 0, :] = A1[0, 0, :]
    A[1, 1, :] = A2[0, 0, :]
    A[2, 2, :] = A3[0, 0, :]
    V = np.eye(n)
    F = mvgc.var_to_pwcgc(A, V)
    # All off-diagonal entries should be exactly zero
    np.fill_diagonal(F, 0.0)
    assert np.max(np.abs(F)) < 1e-10


def test_infocrit_formulas():
    """Cross-check info-crit formulas against the MATLAB definitions in MVGC2.

    With K = k/m, AIC = -2L + 2K, BIC = -2L + K log m, HQC = -2L + 2K log log m.
    """
    L, k, m = -1000.0, 10, 500
    aic, bic, hqc = mvgc.infocrit(L, k, m)
    K = k / m
    assert float(aic) == pytest.approx(-2 * L + 2 * K, abs=1e-10)
    assert float(bic) == pytest.approx(-2 * L + K * np.log(m), abs=1e-10)
    assert float(hqc) == pytest.approx(-2 * L + 2 * K * np.log(np.log(m)), abs=1e-10)


def test_infocrit_selects_correct_order_for_var(rng):
    """For data simulated from VAR(p_true), the chosen order should be near p_true."""
    p_true = 3
    A = mvgc.var_rand(3, p_true, 0.6, rng=rng)
    V = np.eye(3)
    X, _ = mvgc.var_to_tsdata(A, V, m=5000, N=1, rng=rng)
    result = mvgc.tsdata_to_varmo(X, pmax=8, regmode="LWR")
    # Allow ±1 lag due to finite sample
    assert abs(result.p_bic - p_true) <= 1
    assert abs(result.p_hqc - p_true) <= 1
