"""Tests for the β-statistic inference + haxa Monte-Carlo null distribution."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import ssdi


def test_habeta_sums_to_m(rng):
    L = ssdi.rand_orthonormal(8, 3, rng=rng)
    beta = ssdi.habeta(L)
    assert beta.sum() == pytest.approx(3.0, abs=1e-12)


def test_habeta_statinf_returns_expected_shapes(rng):
    L = ssdi.rand_orthonormal(8, 3, rng=rng)
    beta = ssdi.habeta(L)
    res_one = ssdi.habeta_statinf(beta, n=8, m=3, slevel=0.05, tails="right")
    assert res_one.sig.shape == (8,)
    res_two = ssdi.habeta_statinf(beta, n=8, m=3, slevel=0.05, tails="both")
    assert res_two.sig.shape == (8, 2)


def test_habeta_two_tail_level_matches_matlab_without_bonferroni(rng):
    beta = ssdi.habeta(ssdi.rand_orthonormal(8, 3, rng=rng))
    left = ssdi.habeta_statinf(beta, n=8, m=3, slevel=0.05, tails="left", mhtc=False)
    right = ssdi.habeta_statinf(beta, n=8, m=3, slevel=0.05, tails="right", mhtc=False)
    both = ssdi.habeta_statinf(beta, n=8, m=3, slevel=0.05, tails="both", mhtc=False)

    np.testing.assert_allclose(both.cval, [left.cval, right.cval], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(both.sig[:, 0], left.sig)
    np.testing.assert_array_equal(both.sig[:, 1], right.sig)


def test_axis_aligned_L_has_significant_beta(rng):
    """When L is exactly aligned with the first m coordinate axes, β = 1 for
    those nodes and β = 0 for the rest — both should be flagged 'right' tail
    significant (or 'left' for the zeros)."""
    n, m = 8, 3
    L = np.zeros((n, m))
    L[:m, :] = np.eye(m)
    beta = ssdi.habeta(L)
    assert np.allclose(beta[:m], 1.0)
    assert np.allclose(beta[m:], 0.0)
    res = ssdi.habeta_statinf(beta, n=n, m=m, slevel=0.05, tails="right", mhtc=True)
    sig = res.sig if res.sig.ndim == 1 else res.sig[:, 1]
    assert np.all(sig[:m])  # the m occupied axes are significant


def test_haxa_dist_shape(rng):
    theta = ssdi.haxa_dist(n=8, N=1000, rng=rng)
    assert theta.shape == (1000, 4)
    # All angles should lie in [0, pi/2]
    assert np.all(theta >= 0)
    assert np.all(theta <= np.pi / 2 + 1e-12)


def test_get_haxa_cvals_runs(rng):
    stats = ssdi.make_haxa_stats(nmax=6, N=2000, rng=rng)
    cv = ssdi.get_haxa_cvals(n=6, stats=stats, mdim=[1, 2, 3], slev=[0.05, 0.5, 0.95])
    assert cv.shape == (3, 3)
    # Critical angles should increase with significance level
    assert np.all(np.diff(cv, axis=1) >= 0)


def test_make_haxa_stats_matches_matlab_principal_angle_reflection():
    seed = 803
    expected_rng = np.random.default_rng(seed)
    ssdi.haxa_dist(n=2, N=500, rng=expected_rng)
    theta_half = ssdi.haxa_dist(n=3, N=500, rng=expected_rng)

    stats = ssdi.make_haxa_stats(nmax=3, N=500, rng=np.random.default_rng(seed))[3]
    assert stats.theta.shape == (500, 2)
    np.testing.assert_allclose(stats.theta[:, 0], theta_half[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        stats.theta[:, 1], np.pi / 2 - theta_half[:, 0], atol=1e-15, rtol=0.0
    )


def test_haxa_high_beta_threshold_uses_lower_angle_quantile():
    levels = np.array([0.05, 0.5, 0.95])
    stats = ssdi.make_haxa_stats(
        nmax=6,
        N=5001,
        slevels=levels,
        rng=np.random.default_rng(804),
    )
    theta_cval = ssdi.get_haxa_cvals(n=6, stats=stats, mdim=[2], slev=[0.05])[0, 0]
    beta_threshold = np.cos(theta_cval) ** 2
    beta_samples = np.cos(stats[6].theta[:, 1]) ** 2
    assert beta_threshold == pytest.approx(np.quantile(beta_samples, 0.95), abs=1e-14)


def test_get_haxa_cvals_validates_table_and_levels(rng):
    stats = ssdi.make_haxa_stats(nmax=4, N=100, rng=rng)
    with pytest.raises(ValueError, match="not n = 3"):
        ssdi.get_haxa_cvals(n=3, stats=stats[4])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ssdi.get_haxa_cvals(n=4, stats=stats, slev=[1.1])


def test_rand_involution(rng):
    Q = ssdi.rand_involution(8, 3, rng=rng)
    # Q² = I (involution)
    assert np.allclose(Q @ Q, np.eye(8), atol=1e-10)
    # Signature 2m - n = -2
    assert pytest.approx(np.trace(Q), abs=1e-10) == 2 * 3 - 8


def test_subspaceb_agrees_with_subspacea(rng):
    """subspaceb computes the largest principal angle, matching subspacea.max()."""
    A = ssdi.rand_orthonormal(8, 3, rng=rng)
    B = ssdi.rand_orthonormal(8, 3, rng=rng)
    theta_a = ssdi.subspacea(A, B)
    theta_b = ssdi.subspaceb(A, B)
    assert theta_b == pytest.approx(theta_a.max(), abs=1e-8)


def test_gmetricsx_returns_n_values(rng):
    L = ssdi.rand_orthonormal(8, 3, rng=rng)
    d = ssdi.gmetricsx(L)
    assert d.shape == (8,)
    assert np.all(d >= 0)
    assert np.all(d <= 1.0)
