"""Dynamical-dependence self-consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc, ssdi


def test_orthonormal_round_trip(rng):
    L = ssdi.rand_orthonormal(8, 3, rng=rng)
    assert np.allclose(L.T @ L, np.eye(3), atol=1e-12)


def test_perfect_dd_is_zero(rng):
    """Closed-form perfect projections give DD = 0 exactly."""
    n, r = 8, 3
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.9, rng=rng)
    LC, LK = ssdi.iss_perfect_dd(C, K, uniq=True)
    assert ssdi.iss2dd(LC, A, C, K) == pytest.approx(0.0, abs=1e-12)
    assert ssdi.iss2dd(LK, A, C, K) == pytest.approx(0.0, abs=1e-12)


def test_iss2dd_matches_trfun2dd(rng):
    """Time-domain DD == frequency-integrated spectral DD."""
    n, r = 4, 8
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.9, rng=rng)
    A, C, K, V = ssdi.transform_ss(A, C, K, np.eye(n))
    L = ssdi.rand_orthonormal(n, 2, rng=rng)
    H = mvgc.ss2trfun(A, C, K, fres=512)
    d_ss = ssdi.iss2dd(L, A, C, K)
    d_sp, _ = ssdi.trfun2dd(L, H)
    assert d_ss == pytest.approx(d_sp, abs=1e-8)


def test_pointwise_spectral_dd_integrates_to_broadband_dd(rng):
    """Paper Eqs. (25)-(27) agree with the legacy Eq. (24) objective."""
    rng = np.random.default_rng(20260516)
    n, r, fres = 4, 7, 512
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.8, rng=rng)
    H = mvgc.ss2trfun(A, C, K, fres=fres)
    L = ssdi.rand_orthonormal(n, 2, rng=rng)
    frequencies = np.linspace(0.0, np.pi, fres + 1)

    broadband, _ = ssdi.trfun2dd(L, H)
    band_dd, pointwise = ssdi.trfun2dd_band(L, H, frequencies)
    np.testing.assert_allclose(pointwise, ssdi.trfun2dd_pointwise(L, H), atol=1e-12)
    assert band_dd == pytest.approx(broadband, abs=1e-8)


def test_band_spectral_gradient_matches_finite_difference(rng):
    rng = np.random.default_rng(20260517)
    n, r, fres = 4, 7, 48
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.75, rng=rng)
    H = mvgc.ss2trfun(A, C, K, fres=fres)
    frequencies = np.linspace(0.0, np.pi, fres + 1)
    band = (frequencies[5], frequencies[31])
    L = ssdi.rand_orthonormal(n, 2, rng=rng)
    direction = rng.standard_normal(L.shape)
    direction -= L @ (L.T @ direction)
    direction /= np.linalg.norm(direction)

    gradient, _ = ssdi.trfun2dd_bandgrad(L, H, frequencies, band)
    eps = 1e-6
    Lplus = ssdi.orthonormalise(L + eps * direction)
    Lminus = ssdi.orthonormalise(L - eps * direction)
    dplus = ssdi.trfun2dd_band(Lplus, H, frequencies, band)[0]
    dminus = ssdi.trfun2dd_band(Lminus, H, frequencies, band)[0]

    assert float(np.sum(gradient * direction)) == pytest.approx(
        (dplus - dminus) / (2.0 * eps), rel=2e-4, abs=2e-6
    )
    np.testing.assert_allclose(L.T @ gradient, np.zeros((2, 2)), atol=1e-11)


def test_off_grid_band_interpolates_exact_boundaries(rng):
    n, r, fres = 4, 7, 32
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.75, rng=rng)
    H = mvgc.ss2trfun(A, C, K, fres=fres)
    frequencies = np.linspace(0.0, np.pi, fres + 1)
    L = ssdi.rand_orthonormal(n, 2, rng=rng)
    pointwise = ssdi.trfun2dd_pointwise(L, H)

    low = 5.25 * np.pi / fres
    high = 5.75 * np.pi / fres
    band_dd, boundary_values = ssdi.trfun2dd_band(L, H, frequencies, (low, high))
    expected_values = np.interp([low, high], frequencies, pointwise)

    np.testing.assert_allclose(boundary_values, expected_values, atol=1e-14, rtol=0.0)
    assert band_dd == pytest.approx(float(np.mean(expected_values)), abs=1e-14, rel=0.0)


def test_off_grid_band_gradient_matches_finite_difference(rng):
    n, r, fres = 4, 7, 48
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.75, rng=rng)
    H = mvgc.ss2trfun(A, C, K, fres=fres)
    frequencies = np.linspace(0.0, np.pi, fres + 1)
    band = (5.3 * np.pi / fres, 30.6 * np.pi / fres)
    L = ssdi.rand_orthonormal(n, 2, rng=rng)
    direction = rng.standard_normal(L.shape)
    direction -= L @ (L.T @ direction)
    direction /= np.linalg.norm(direction)

    gradient, _ = ssdi.trfun2dd_bandgrad(L, H, frequencies, band)
    eps = 1e-6
    Lplus = ssdi.orthonormalise(L + eps * direction)
    Lminus = ssdi.orthonormalise(L - eps * direction)
    dplus = ssdi.trfun2dd_band(Lplus, H, frequencies, band)[0]
    dminus = ssdi.trfun2dd_band(Lminus, H, frequencies, band)[0]

    assert float(np.sum(gradient * direction)) == pytest.approx(
        (dplus - dminus) / (2.0 * eps), rel=2e-4, abs=2e-6
    )
    np.testing.assert_allclose(L.T @ gradient, np.zeros((2, 2)), atol=1e-11, rtol=0.0)


def test_band_rejects_zero_width_and_out_of_grid(rng):
    A, C, K, _ = mvgc.iss_rand(3, 5, rhoa=0.7, rng=rng)
    H = mvgc.ss2trfun(A, C, K, fres=16)
    frequencies = np.linspace(0.0, np.pi, 17)
    L = ssdi.rand_orthonormal(3, 1, rng=rng)

    with pytest.raises(ValueError, match="low < high"):
        ssdi.trfun2dd_band(L, H, frequencies, (0.2, 0.2))
    with pytest.raises(ValueError, match="within"):
        ssdi.trfun2dd_band(L, H, frequencies, (-0.1, 0.2))


def test_gradient_descent_decreases_dd(rng):
    """A single GD step should not increase the proxy DD."""
    n, r = 6, 10
    A, C, K, _ = mvgc.iss_rand(n, r, rhoa=0.9, rng=rng)
    A, C, K, _ = ssdi.transform_ss(A, C, K, np.eye(n))
    CAK = ssdi.iss2cak(A, C, K)
    L0 = ssdi.rand_orthonormal(n, 2, rng=rng)
    res = ssdi.opt_gd2_ddx(CAK, L0, maxiters=500, gdsig0=1e-3)
    assert res.dd <= ssdi.cak2ddx(L0, CAK) + 1e-12


def test_grassmannian_metric_symmetric(rng):
    n, m = 6, 3
    L1 = ssdi.rand_orthonormal(n, m, rng=rng)
    L2 = ssdi.rand_orthonormal(n, m, rng=rng)
    assert ssdi.gmetric(L1, L2) == pytest.approx(ssdi.gmetric(L2, L1), abs=1e-12)
    assert ssdi.gmetric(L1, L1) == pytest.approx(0.0, abs=1e-10)


def test_l2q_round_trip(rng):
    L = ssdi.rand_orthonormal(7, 3, rng=rng)
    Q = ssdi.L2Q(L)
    assert np.allclose(Q @ Q, np.eye(7), atol=1e-12)  # Q is involution
    L_back = ssdi.Q2L(Q, m=3)
    assert ssdi.gmetric(L, L_back) < 1e-10


def test_transform_ss_yields_identity_V(rng):
    A, C, K, _ = mvgc.iss_rand(5, 8, rhoa=0.9, rng=rng)
    V = mvgc.corr_rand(5, rng=rng)
    A_d, C_d, K_d, V_d = ssdi.transform_ss(A, C, K, V)
    assert np.allclose(V_d, np.eye(5), atol=1e-12)


def test_test_networks_have_expected_shape():
    for net in (ssdi.tnet9a, ssdi.tnet9b, ssdi.tnet9c, ssdi.tnet9d):
        G = net()
        assert G.shape == (9, 9)
        assert np.all(np.diag(G) == 1)  # self-loops always present


def test_tnet243_structure():
    """tnet243: 2-module → 4-module, isolated 3-module."""
    G = ssdi.tnet243()
    assert G.shape == (9, 9)
    # 2-module (rows 0-1) gets nothing from 4- or 3-modules
    assert np.all(G[0:2, 2:] == 0)
    # 4-module (rows 2-5) receives from 2-module (cols 0-1) and itself, nothing from 3-module
    assert np.all(G[2:6, 6:] == 0)
    assert np.all(G[2:6, 0:2] == 1)
    # 3-module (rows 6-8) is fully isolated (only intra-module edges)
    assert np.all(G[6:9, 0:6] == 0)
    assert np.all(G[6:9, 6:9] == 1)
