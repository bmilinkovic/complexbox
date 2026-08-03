"""Deterministic parity checks against the original MATLAB SSDI-1 toolbox."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from complexbox import mvgc, ssdi

REFERENCE = Path(__file__).parents[1] / "fixtures" / "ssdi_core_reference.mat"


@pytest.fixture(scope="module")
def matlab_reference():
    return loadmat(REFERENCE, squeeze_me=True, struct_as_record=False)


def _principal_angle(L1: np.ndarray, L2: np.ndarray) -> float:
    singular = np.linalg.svd(L1.T @ L2, compute_uv=False)
    return float(np.max(np.arccos(np.clip(singular, -1.0, 1.0))))


def test_spectral_radii_and_frequency_resolution_match_matlab(matlab_reference):
    ref = matlab_reference
    A, C, K, V = (np.asarray(ref[name], dtype=float) for name in ("A", "C", "K", "V"))

    assert mvgc.specnorm(A) == pytest.approx(float(ref["rho_A"]), abs=1e-14, rel=0.0)
    assert mvgc.specnorm(A - K @ C) == pytest.approx(float(ref["rho_B"]), abs=1e-14, rel=0.0)

    fast, fast_err = mvgc.ss2fres(A, C, K, V, fast=True, return_error=True)
    adaptive, adaptive_err = mvgc.ss2fres(A, C, K, V, return_error=True)
    assert fast == int(ref["fres_fast"])
    assert adaptive == int(ref["fres_adaptive"])
    assert fast_err == pytest.approx(float(ref["ierr_fast"]), abs=1e-14, rel=0.0)
    assert adaptive_err == pytest.approx(float(ref["ierr_adaptive"]), abs=1e-14, rel=0.0)

    AVAR = np.asarray(ref["AVAR"], dtype=float)
    VVAR = np.asarray(ref["VVAR"], dtype=float)
    assert mvgc.specnorm(AVAR) == pytest.approx(float(ref["rho_VAR"]), abs=1e-14, rel=0.0)
    var_fast, var_fast_err = mvgc.var2fres(AVAR, VVAR, fast=True, return_error=True)
    var_adaptive, var_adaptive_err = mvgc.var2fres(AVAR, VVAR, return_error=True)
    assert var_fast == int(ref["var_fres_fast"])
    assert var_adaptive == int(ref["var_fres_adaptive"])
    assert var_fast_err == pytest.approx(float(ref["var_ierr_fast"]), abs=1e-14, rel=0.0)
    assert var_adaptive_err == pytest.approx(float(ref["var_ierr_adaptive"]), abs=1e-14, rel=0.0)


def test_causal_emergence_matches_matlab_with_and_without_precompute(matlab_reference):
    ref = matlab_reference
    A, C, K, V, L = (np.asarray(ref[name], dtype=float) for name in ("A", "C", "K", "V", "L"))
    ci, dd = ssdi.iss2ce(L, A, C, K, V)
    G, P = ssdi.iss2ce_precomp(A, C, K, V)
    ci_pre, dd_pre = ssdi.iss2ce(L, A, C, K, V, G, P)

    np.testing.assert_allclose(G, np.asarray(ref["Gpre"]), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(P, np.asarray(ref["Ppre"]), rtol=0.0, atol=2e-13)
    assert ci == pytest.approx(float(ref["CI"]), abs=2e-13, rel=0.0)
    assert dd == pytest.approx(float(ref["DD"]), abs=2e-13, rel=0.0)
    assert ci_pre == pytest.approx(float(ref["CI_pre"]), abs=2e-13, rel=0.0)
    assert dd_pre == pytest.approx(float(ref["DD_pre"]), abs=2e-13, rel=0.0)


def test_spectral_dd_and_gradient_match_matlab(matlab_reference):
    ref = matlab_reference
    L = np.asarray(ref["L"], dtype=float)
    H = np.asarray(ref["H"])
    D, d = ssdi.trfun2dd(L, H)
    grad, gmag = ssdi.trfun2ddgrad(L, H)

    assert D == pytest.approx(float(ref["D"]), abs=2e-14, rel=0.0)
    np.testing.assert_allclose(d, np.asarray(ref["d"]), rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(grad, np.asarray(ref["grad"]), rtol=0.0, atol=2e-13)
    assert gmag == pytest.approx(float(ref["gmag"]), abs=2e-13, rel=0.0)
    np.testing.assert_allclose(L.T @ grad, np.zeros((L.shape[1], L.shape[1])), rtol=0.0, atol=2e-13)


def test_spectral_optimizer_matches_matlab(matlab_reference):
    ref = matlab_reference
    L = np.asarray(ref["L"], dtype=float)
    H = np.asarray(ref["H"])
    result = ssdi.opt_gd2_dds(
        H,
        L,
        maxiters=250,
        gdsig0=1e-3,
        gdls=2.0,
        tol=1e-10,
        history=True,
    )

    assert result.dd == pytest.approx(float(ref["dd_opt"]), abs=2e-12, rel=0.0)
    assert result.converged == int(ref["conv"])
    assert result.sig == pytest.approx(float(ref["sig"]), abs=1e-15, rel=0.0)
    assert result.iters == int(ref["iters"])
    np.testing.assert_allclose(result.history, np.asarray(ref["hist"]), rtol=0.0, atol=2e-11)
    assert _principal_angle(result.L, np.asarray(ref["L_opt"])) < 5e-7
