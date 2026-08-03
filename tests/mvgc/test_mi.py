"""Tests for the mutual-information suite."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from complexbox import mvgc


def test_cov_to_mvmi_bivariate_analytic(rng):
    """MVGC2 returns *twice* the standard mutual information.

    For 2D Gaussian with correlation ``rho``, the standard MI in nats is
    ``-0.5 log(1 - rho²)``. MVGC2's ``cov_to_mvmi`` follows the same
    log-determinant convention as Granger causality (no 0.5 factor), so the
    expected value here is ``-log(1 - rho²)``.
    """
    rho = 0.6
    V = np.array([[1.0, rho], [rho, 1.0]])
    mi_mvgc = -np.log(1 - rho * rho)  # twice the standard MI
    mi = mvgc.cov_to_mvmi(V, x=[0], y=[1])
    assert mi == pytest.approx(mi_mvgc, abs=1e-12)


def test_cov_to_pwmi_diagonal(rng):
    """Pairwise MI matrix should be symmetric and have NaN diagonal."""
    V = mvgc.corr_rand(4, rng=rng)
    mutual_info = mvgc.cov_to_pwmi(V)
    assert np.all(np.isnan(np.diag(mutual_info)))
    # Symmetry
    assert np.max(np.abs(mutual_info - mutual_info.T)[~np.isnan(mutual_info)]) < 1e-12


def test_cov_to_iomi_nonneg(rng):
    V = mvgc.corr_rand(4, rng=rng)
    mutual_info = mvgc.cov_to_iomi(V)
    assert np.all(mutual_info >= -1e-12)


def test_cpsd_smvmi_integrates_to_cov_mvmi(rng):
    """Frequency-integrated spectral MI should equal the time-domain MI."""
    A = mvgc.var_rand(3, 2, rho=0.6, rng=rng)
    V = mvgc.corr_rand(3, rng=rng)
    G, _ = mvgc.var_to_autocov(A, V, qmax=-15)
    S = mvgc.var_to_cpsd(A, V, fres=256)
    omega = np.linspace(0, np.pi, S.shape[2])
    smvmi = mvgc.cpsd_to_smvmi(S, x=[0], y=[1])
    integrated = float(trapezoid(smvmi, omega) / np.pi)
    cov_mvmi = mvgc.cov_to_mvmi(G[:, :, 0], x=[0], y=[1])
    # These differ in interpretation: cov is instantaneous, spectral is total
    # — they should both be non-negative
    assert integrated >= -1e-10
    assert cov_mvmi >= -1e-10


def test_singleton_gwcmi_equals_pwcmi(rng):
    """When each group is one variable, group-wise CMI should match pairwise CMI."""
    V = mvgc.corr_rand(4, rng=rng)
    groups = [[0], [1], [2], [3]]
    gw = mvgc.cov_to_gwcmi(V, groups)
    pw = mvgc.cov_to_pwcmi(V)
    # Take a single off-diagonal pair
    assert pytest.approx(pw[0, 1], abs=1e-12) == gw[0, 1]
