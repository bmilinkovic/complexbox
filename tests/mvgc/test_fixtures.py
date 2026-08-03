"""Parity tests against MATLAB-generated fixtures (skipped if missing)."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc

pytestmark = pytest.mark.fixture


def test_var_to_autocov_matches_matlab(mvgc_fixture):
    A = np.asarray(mvgc_fixture["A"])
    V = np.asarray(mvgc_fixture["V"])
    G_ref = np.asarray(mvgc_fixture["G"])
    G, _ = mvgc.var_to_autocov(A, V, qmax=-(G_ref.shape[2] - 1))
    np.testing.assert_allclose(G, G_ref, rtol=0.0, atol=1e-12)


def test_autocov_to_var_matches_matlab(mvgc_fixture):
    G = np.asarray(mvgc_fixture["G"])
    A_ref = np.asarray(mvgc_fixture["A_back"])
    V_ref = np.asarray(mvgc_fixture["V_back"])
    A_back, V_back = mvgc.autocov_to_var(G)
    np.testing.assert_allclose(A_back, A_ref, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(V_back, V_ref, rtol=0.0, atol=1e-10)


def test_var_to_ss_matches_matlab(mvgc_fixture):
    A = np.asarray(mvgc_fixture["A"])
    V = np.asarray(mvgc_fixture["V"])
    A_ss_ref = np.asarray(mvgc_fixture["A_ss"])
    C_ref = np.asarray(mvgc_fixture["C"])
    K_ref = np.asarray(mvgc_fixture["K"])
    A_ss, C, K, _ = mvgc.var_to_ss(A, V)
    np.testing.assert_allclose(A_ss, A_ss_ref, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(C, C_ref, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(K, K_ref, rtol=0.0, atol=1e-12)


def test_var_pwcgc_matches_matlab(mvgc_fixture):
    A = np.asarray(mvgc_fixture["A"])
    V = np.asarray(mvgc_fixture["V"])
    F_ref = np.asarray(mvgc_fixture["F_var"])
    F = mvgc.var_to_pwcgc(A, V)
    # NaN diagonal entries must agree
    assert np.all(np.isnan(F) == np.isnan(F_ref))
    mask = ~np.isnan(F_ref)
    np.testing.assert_allclose(F[mask], F_ref[mask], rtol=0.0, atol=1e-10)


def test_mvgc_pval_matches_matlab(mvgc_fixture):
    pval_F = float(mvgc_fixture["pval_F"])
    cval_F = float(mvgc_fixture["cval_F"])
    pval_LR = float(mvgc_fixture["pval_LR"])
    F_stat = float(mvgc_fixture["F_test_stat"])
    p_py = mvgc.mvgc_pval(F_stat, "F", 1, 1, 1, 2, m=50000, N=1)
    p_py_LR = mvgc.mvgc_pval(F_stat, "LR", 1, 1, 1, 2, m=50000, N=1)
    c_py = mvgc.mvgc_cval(0.05, "F", 1, 1, 1, 2, m=50000, N=1)
    np.testing.assert_allclose(p_py, pval_F, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(p_py_LR, pval_LR, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(c_py, cval_F, rtol=0.0, atol=1e-10)
