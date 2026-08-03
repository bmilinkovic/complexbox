"""Parity checks against the deterministic MATLAB SSDI-1 fixture."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import ssdi

pytestmark = pytest.mark.fixture


def test_ssdi_values_and_gradients_match_matlab(ssdi_fixture):
    ref = ssdi_fixture
    A, C, K = (np.asarray(ref[name], dtype=float) for name in ("A_ss2", "C2", "K2"))
    CAK = np.asarray(ref["CAK"], dtype=float)
    L = np.asarray(ref["L_rand"], dtype=float)
    H = np.asarray(ref["H"])

    np.testing.assert_allclose(ssdi.iss2cak(A, C, K), CAK, rtol=0.0, atol=2e-13)
    assert ssdi.cak2ddx(L, CAK) == pytest.approx(float(ref["dd_proxy"]), abs=2e-13, rel=0.0)
    assert ssdi.iss2dd(L, A, C, K) == pytest.approx(float(ref["dd_exact"]), abs=2e-12, rel=0.0)
    gradient, magnitude = ssdi.cak2ddxgrad(L, CAK)
    np.testing.assert_allclose(gradient, np.asarray(ref["grad"]), rtol=0.0, atol=2e-12)
    assert magnitude == pytest.approx(float(ref["gmag"]), abs=2e-12, rel=0.0)
    spectral, _ = ssdi.trfun2dd(L, H)
    assert spectral == pytest.approx(float(ref["dd_spec"]), abs=2e-12, rel=0.0)


def test_proxy_optimizer_matches_matlab(ssdi_fixture):
    ref = ssdi_fixture
    result = ssdi.opt_gd2_ddx(
        np.asarray(ref["CAK"], dtype=float),
        np.asarray(ref["L0"], dtype=float),
        maxiters=5_000,
        gdsig0=1e-3,
        gdls=2.0,
        tol=1e-9,
    )
    assert result.dd == pytest.approx(float(ref["dd_opt"]), abs=2e-11, rel=0.0)
    assert result.iters == int(ref["iters_opt"])
    reference_projector = np.asarray(ref["L_opt"]) @ np.asarray(ref["L_opt"]).T
    np.testing.assert_allclose(result.L @ result.L.T, reference_projector, rtol=0.0, atol=2e-9)
