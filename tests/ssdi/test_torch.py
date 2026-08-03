"""Parity and lifecycle tests for the optional batched Torch SSDI backend."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from complexbox.ssdi import _torch as torch_backend
from complexbox.ssdi.dd import cak2ddx, cak2ddxgrad, trfun2dd, trfun2ddgrad
from complexbox.ssdi.optimise import opt_gd_dds_mruns, opt_gd_ddx_mruns


def _random_bases(rng: np.random.Generator, n: int, m: int, runs: int) -> np.ndarray:
    bases = np.empty((n, m, runs))
    for run in range(runs):
        bases[:, :, run] = np.linalg.svd(rng.standard_normal((n, m)), full_matrices=False)[0]
    return bases


def _well_conditioned_transfer(rng: np.random.Generator, n: int, nfreq: int) -> np.ndarray:
    transfer = np.empty((n, n, nfreq), dtype=np.complex128)
    for frequency in range(nfreq):
        perturbation = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        transfer[:, :, frequency] = np.eye(n) + 0.05 * perturbation
    return transfer


def _projectors(bases: np.ndarray) -> np.ndarray:
    return np.einsum("imr,jmr->ijr", bases, bases)


def test_proxy_primitives_match_numpy_and_chunking(rng):
    n, m, runs, lags = 6, 2, 5, 7
    bases = _random_bases(rng, n, m, runs)
    cak = 0.2 * rng.standard_normal((n, n, lags))

    values = torch_backend.cak2ddx_torch(bases, cak, run_chunk_size=2, lag_chunk_size=3)
    gradients, magnitudes = torch_backend.cak2ddxgrad_torch(
        bases, cak, run_chunk_size=3, lag_chunk_size=2
    )
    values_unchunked = torch_backend.cak2ddx_torch(bases, cak)
    gradients_unchunked, magnitudes_unchunked = torch_backend.cak2ddxgrad_torch(bases, cak)

    for run in range(runs):
        value_ref = cak2ddx(bases[:, :, run], cak)
        gradient_ref, magnitude_ref = cak2ddxgrad(bases[:, :, run], cak)
        assert values[run] == pytest.approx(value_ref, abs=1e-12)
        np.testing.assert_allclose(gradients[:, :, run], gradient_ref, rtol=1e-12, atol=1e-12)
        assert magnitudes[run] == pytest.approx(magnitude_ref, abs=1e-12)
        assert np.linalg.norm(bases[:, :, run].T @ gradients[:, :, run]) < 1e-12

    np.testing.assert_allclose(values, values_unchunked, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(gradients, gradients_unchunked, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(magnitudes, magnitudes_unchunked, rtol=0.0, atol=3e-15)


def test_spectral_primitives_match_numpy_and_chunking(rng):
    n, m, runs, nfreq = 5, 2, 4, 11
    bases = _random_bases(rng, n, m, runs)
    transfer = _well_conditioned_transfer(rng, n, nfreq)

    values = torch_backend.trfun2dd_torch(bases, transfer, run_chunk_size=2, frequency_chunk_size=3)
    gradients, magnitudes = torch_backend.trfun2ddgrad_torch(
        bases, transfer, run_chunk_size=3, frequency_chunk_size=4
    )
    values_unchunked = torch_backend.trfun2dd_torch(bases, transfer)
    gradients_unchunked, magnitudes_unchunked = torch_backend.trfun2ddgrad_torch(bases, transfer)

    for run in range(runs):
        value_ref, _ = trfun2dd(bases[:, :, run], transfer)
        gradient_ref, magnitude_ref = trfun2ddgrad(bases[:, :, run], transfer)
        assert values[run] == pytest.approx(value_ref, abs=1e-12)
        np.testing.assert_allclose(gradients[:, :, run], gradient_ref, rtol=1e-12, atol=1e-12)
        assert magnitudes[run] == pytest.approx(magnitude_ref, abs=1e-12)
        assert np.linalg.norm(bases[:, :, run].T @ gradients[:, :, run]) < 1e-12

    np.testing.assert_allclose(values, values_unchunked, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(gradients, gradients_unchunked, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(magnitudes, magnitudes_unchunked, rtol=0.0, atol=3e-15)


@pytest.mark.parametrize("objective", ["proxy", "spectral"])
def test_analytic_gradient_matches_tangent_finite_difference(rng, objective):
    n, m = 5, 2
    basis = _random_bases(rng, n, m, 1)[:, :, 0]
    direction = rng.standard_normal((n, m))
    direction = direction - basis @ (basis.T @ direction)
    direction /= np.linalg.norm(direction)
    eps = 1e-6

    def retract(X):
        return np.linalg.svd(X, full_matrices=False)[0]

    if objective == "proxy":
        data = 0.15 * rng.standard_normal((n, n, 6))
        gradient = torch_backend.cak2ddxgrad_torch(basis, data)[0][:, :, 0]
        plus = torch_backend.cak2ddx_torch(retract(basis + eps * direction), data)[0]
        minus = torch_backend.cak2ddx_torch(retract(basis - eps * direction), data)[0]
    else:
        data = _well_conditioned_transfer(rng, n, 9)
        gradient = torch_backend.trfun2ddgrad_torch(basis, data)[0][:, :, 0]
        plus = torch_backend.trfun2dd_torch(retract(basis + eps * direction), data)[0]
        minus = torch_backend.trfun2dd_torch(retract(basis - eps * direction), data)[0]

    finite_difference = (plus - minus) / (2.0 * eps)
    analytic = float(np.sum(gradient * direction))
    assert finite_difference == pytest.approx(analytic, rel=2e-6, abs=2e-8)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("objective", ["proxy", "spectral"])
def test_optimisers_match_numpy(rng, variant, objective):
    n, m, runs = 5, 2, 4
    bases = _random_bases(rng, n, m, runs)
    common = dict(
        maxiters=10,
        variant=variant,
        gdsig0=1e-3,
        gdls=2.0,
        tol=(1e-14, -1.0, 1e-14),
        history=True,
    )

    if objective == "proxy":
        data = 0.15 * rng.standard_normal((n, n, 6))
        numpy_result = opt_gd_ddx_mruns(data, bases, **common)
        torch_result = torch_backend.opt_gd_ddx_mruns_torch(
            data,
            bases,
            run_chunk_size=2,
            lag_chunk_size=2,
            **common,
        )
    else:
        data = _well_conditioned_transfer(rng, n, 9)
        numpy_result = opt_gd_dds_mruns(data, bases, **common)
        torch_result = torch_backend.opt_gd_dds_mruns_torch(
            data,
            bases,
            run_chunk_size=2,
            frequency_chunk_size=3,
            **common,
        )

    np.testing.assert_allclose(torch_result[0], numpy_result[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        _projectors(torch_result[1]),
        _projectors(numpy_result[1]),
        rtol=1e-10,
        atol=1e-11,
    )
    assert torch_result[2] == numpy_result[2]
    for torch_history, numpy_history in zip(torch_result[3], numpy_result[3]):
        np.testing.assert_allclose(torch_history, numpy_history, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("variant", [1, 2])
def test_optimizer_chunking_is_invariant(rng, variant):
    n, m, runs, lags = 6, 2, 5, 7
    bases = _random_bases(rng, n, m, runs)
    cak = 0.2 * rng.standard_normal((n, n, lags))
    common = dict(
        maxiters=8,
        variant=variant,
        gdsig0=1e-3,
        tol=(1e-14, -1.0, 1e-14),
        history=True,
    )
    full = torch_backend.opt_gd_ddx_mruns_torch(cak, bases, **common)
    chunked = torch_backend.opt_gd_ddx_mruns_torch(
        cak, bases, run_chunk_size=1, lag_chunk_size=2, **common
    )

    np.testing.assert_allclose(chunked[0], full[0], rtol=1e-12, atol=2e-14)
    np.testing.assert_allclose(
        _projectors(chunked[1]), _projectors(full[1]), rtol=1e-11, atol=2e-12
    )
    assert chunked[2] == full[2]
    for chunked_history, full_history in zip(chunked[3], full[3]):
        np.testing.assert_allclose(chunked_history, full_history, rtol=1e-11, atol=2e-13)


@pytest.mark.parametrize("variant", [1, 2])
def test_converged_run_is_frozen_while_other_runs_continue(variant):
    rng = np.random.default_rng(19)
    n, m, lags = 6, 2, 7
    cak = 0.2 * rng.standard_normal((n, n, lags))
    candidates = _random_bases(rng, n, m, 64)
    candidate_values = np.array(
        [cak2ddx(candidates[:, :, run], cak) for run in range(candidates.shape[2])]
    )
    low = int(np.argmin(candidate_values))
    high = int(np.argmax(candidate_values))
    bases = candidates[:, :, [low, high]]
    dtol = candidate_values[low] + 0.1 * (candidate_values[high] - candidate_values[low])
    common = dict(
        variant=variant,
        gdsig0=1e-5,
        gdls=2.0,
        tol=(1e-20, float(dtol), 1e-20),
        history=True,
        run_chunk_size=2,
        lag_chunk_size=3,
    )

    stopped_at_two = torch_backend.opt_gd_ddx_mruns_torch(cak, bases, maxiters=2, **common)
    continued = torch_backend.opt_gd_ddx_mruns_torch(cak, bases, maxiters=12, **common)

    stopped_idx = [i for i, code in enumerate(continued[2]) if code == 2]
    assert stopped_idx == [0]
    idx = stopped_idx[0]
    assert continued[3][idx].shape == (2, 3)
    np.testing.assert_array_equal(continued[1][:, :, idx], stopped_at_two[1][:, :, idx])
    np.testing.assert_array_equal(continued[3][idx], stopped_at_two[3][idx])
    assert continued[0][idx] == stopped_at_two[0][idx]
    # The higher-DD restart remained active, proving this was per-run freezing
    # rather than an all-runs early exit.
    assert continued[3][1].shape == (12, 3)
    assert continued[0][1] < stopped_at_two[0][1]


def test_history_false_matches_numpy_shape(rng):
    bases = _random_bases(rng, 4, 2, 3)
    cak = 0.1 * rng.standard_normal((4, 4, 5))
    result = torch_backend.opt_gd_ddx_mruns_torch(cak, bases, maxiters=2, history=False)
    assert result[0].shape == (3,)
    assert result[1].shape == bases.shape
    assert len(result[2]) == 3
    assert result[3] == [None, None, None]


def test_public_optimizer_backend_dispatch(rng):
    bases = _random_bases(rng, 4, 2, 3)
    cak = 0.1 * rng.standard_normal((4, 4, 5))
    direct = torch_backend.opt_gd_ddx_mruns_torch(
        cak, bases, maxiters=4, run_chunk_size=2, lag_chunk_size=2
    )
    public = opt_gd_ddx_mruns(
        cak,
        bases,
        maxiters=4,
        backend="torch",
        run_chunk_size=2,
        lag_chunk_size=2,
    )
    np.testing.assert_allclose(public[0], direct[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_projectors(public[1]), _projectors(direct[1]), atol=0.0)
    assert public[2:] == direct[2:]
