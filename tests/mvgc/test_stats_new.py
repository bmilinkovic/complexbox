"""Tests for the new statistics functions (GGC, LBQ, Mardia, MW, empirical)."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc


def test_ggc_pval_cval_consistency():
    """If GGC stat equals critical value, p-value equals alpha."""
    alpha = 0.05
    cv = mvgc.ggc_cval(alpha, "F", nx=2, nz=2, p=2, m=5000)
    pv = mvgc.ggc_pval(cv, "F", nx=2, nz=2, p=2, m=5000)
    assert pv == pytest.approx(alpha, abs=1e-10)


def test_mvgc_icdf_inverse(rng):
    """``icdf(cdf(x)) == x`` over the meaningful x range.

    The F-statistic for GC is scaled by ``sf = d2/d`` (often ~500), so
    we sweep over the lower CDF region to avoid saturating to 1.
    """
    cdf = mvgc.mvgc_cdf("F", 1, 1, 0, 2, m=1000)
    icdf = mvgc.mvgc_icdf("F", 1, 1, 0, 2, m=1000)
    p = np.linspace(0.01, 0.99, 20)
    x = icdf(p)
    p_back = cdf(x)
    assert np.max(np.abs(p_back - p)) < 1e-8


def test_mardia_zero_for_gaussian(rng):
    """Mardia statistics should be small for IID Gaussian data."""
    X = rng.standard_normal((4, 10_000))
    S, K, J = mvgc.mardia(X, debias=True)
    assert abs(S) < 0.1
    assert abs(K) < 1.0
    assert J >= 0


def test_mann_whitney_shifted(rng):
    """Mann-Whitney z should be strongly positive when x2 is right-shifted."""
    x1 = rng.standard_normal(200)
    x2 = rng.standard_normal(200) + 1.0
    z, _ = mvgc.mann_whitney(x1, x2)
    assert z > 3.0


def test_empirical_cdf_recovers_gaussian(rng):
    samples = rng.standard_normal(10_000)
    cdf_vals = mvgc.empirical_cdf(np.array([-1.0, 0.0, 1.0]), samples)
    # Standard normal quantiles
    assert abs(cdf_vals[0] - 0.1587) < 0.02
    assert abs(cdf_vals[1] - 0.5) < 0.02
    assert abs(cdf_vals[2] - 0.8413) < 0.02


def test_empirical_cdfi_inverts_cdf(rng):
    samples = rng.standard_normal(10_000)
    P = np.array([0.1, 0.5, 0.9])
    x = mvgc.empirical_cdfi(P, samples)
    P_back = mvgc.empirical_cdf(x, samples)
    assert np.max(np.abs(P_back - P)) < 0.02


def test_tsdata_permute_preserves_variance(rng):
    A = mvgc.var_rand(3, 2, rho=0.6, rng=rng)
    V = np.eye(3)
    X, _ = mvgc.var_to_tsdata(A, V, m=2000, rng=rng)
    Y_shuf = mvgc.tsdata_permute(X, method="shuffle", rng=rng)
    Y_bs = mvgc.tsdata_permute(X, method="block_shuffle", block_length=50, rng=rng)
    assert Y_shuf.shape == X.shape
    assert Y_bs.shape == X.shape
    # Total variance preserved
    assert np.allclose(X.var(axis=1), Y_shuf.var(axis=1), rtol=1e-10)


def test_lbqtest_runs(rng):
    A = mvgc.var_rand(2, 2, rho=0.6, rng=rng)
    V = np.eye(2)
    X, _ = mvgc.var_to_tsdata(A, V, m=2000, rng=rng)
    Q = mvgc.lbqtest(X, p=2, hmax=10)
    assert np.all(np.isnan(Q[:2]))
    assert np.all(np.isfinite(Q[2:]))
    pvals = mvgc.lbqtest_pval(Q, n=2, p=2, h=np.arange(1, 11))
    assert np.all(np.isnan(pvals[:2]))
