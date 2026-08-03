"""Tests for the GC completers (autocov_to_* and cpsd_to_* paths)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from complexbox import mvgc


@pytest.fixture
def small_var(rng):
    A = mvgc.var_rand(4, 2, rho=0.7, rng=rng)
    V = mvgc.corr_rand(4, rng=rng)
    return A, V


def test_var_pwcgc_matches_autocov_pwcgc(small_var):
    A, V = small_var
    # Use plenty of autocov lags so the Whittle LWR is well-converged.
    # Finite-lag truncation in autocov_to_var limits agreement to ~1e-5
    # when the VAR has rho ≈ 0.7.
    G, _ = mvgc.var_to_autocov(A, V, qmax=-80)
    F_var = mvgc.var_to_pwcgc(A, V)
    F_ac = mvgc.autocov_to_pwcgc(G)
    diff = np.nanmax(np.abs(F_var - F_ac))
    assert diff < 1e-5


def test_var_mvgc_matches_autocov_mvgc(small_var):
    A, V = small_var
    G, _ = mvgc.var_to_autocov(A, V, qmax=-15)
    F_var = mvgc.var_to_mvgc(A, V, x=[0], y=[2])
    F_ac = mvgc.autocov_to_mvgc(G, x=[0], y=[2])
    assert F_var == pytest.approx(F_ac, abs=1e-10)


def test_var_pwcgc_matches_cpsd_pwcgc(small_var):
    A, V = small_var
    S = mvgc.var_to_cpsd(A, V, fres=256)
    F_var = mvgc.var_to_pwcgc(A, V)
    F_cpsd = mvgc.cpsd_to_pwcgc(S, tol=1e-10)
    diff = np.nanmax(np.abs(F_var - F_cpsd))
    assert diff < 1e-6


def test_cpsd_specfac_roundtrip(small_var):
    A, V = small_var
    S = mvgc.var_to_cpsd(A, V, fres=256)
    H, V_rec, conv, _relerr, _niter = mvgc.cpsd_specfac(S, tol=1e-10)
    assert conv
    assert np.max(np.abs(V_rec - V)) < 1e-8


def test_integrated_spectral_matches_time_domain(small_var):
    A, V = small_var
    fres = 1024  # higher fres needed for trapezoidal accuracy
    F_t = mvgc.var_to_pwcgc(A, V)
    F_s = mvgc.var_to_spwcgc(A, V, fres=fres)
    omega = np.linspace(0, np.pi, fres + 1)
    F_int = trapezoid(np.nan_to_num(F_s, nan=0), omega, axis=2) / np.pi
    F_t0 = np.nan_to_num(F_t, nan=0)
    # Trapezoidal rule on sharp spectral peaks: 1e-2 is the practical bound.
    assert np.max(np.abs(F_int - F_t0)) < 5e-2
