"""Parity and batching tests for the optional Torch ELPH backend."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from complexbox import elph
from complexbox.elph._torch import (
    emergence_delta_torch,
    emergence_gamma_torch,
    emergence_psi_torch,
    gaussian_local_mi_torch,
    gaussian_mi_torch,
)


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch")


def _batched_correlated_data(
    rng: np.random.Generator, batches: int = 5, samples: int = 600
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((batches, 2, samples))
    noise = rng.standard_normal((batches, 3, samples))
    Y = np.empty_like(noise)
    Y[:, 0] = 0.70 * X[:, 0] + 0.25 * X[:, 1] + 0.45 * noise[:, 0]
    Y[:, 1] = -0.30 * X[:, 0] + 0.55 * X[:, 1] + 0.60 * noise[:, 1]
    Y[:, 2] = 0.20 * X[:, 0] + 0.75 * noise[:, 2]
    return X, Y


def test_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import complexbox.elph._torch; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unbatched_gaussian_mi_and_locals_match_numpy(rng, torch_module):
    Xb, Yb = _batched_correlated_data(rng)
    X, Y = Xb[0], Yb[0]

    expected_mi = elph.gaussian_mi(X, Y)
    expected_local = elph.gaussian_local_mi(X, Y)
    actual_mi = gaussian_mi_torch(X, Y, device="cpu", dtype="float64")
    actual_local = gaussian_local_mi_torch(X, Y, device="cpu", dtype="float64")

    assert isinstance(actual_mi, float)
    assert actual_local.shape == (X.shape[1],)
    assert actual_mi == pytest.approx(expected_mi, rel=2e-11, abs=2e-11)
    np.testing.assert_allclose(actual_local, expected_local, rtol=2e-11, atol=2e-11)
    assert float(np.mean(actual_local)) == pytest.approx(actual_mi, abs=2e-12)


def test_batched_parity_broadcast_and_chunking(rng, torch_module):
    X, Y = _batched_correlated_data(rng)
    expected_mi = np.array([elph.gaussian_mi(x, y) for x, y in zip(X, Y)])
    expected_local = np.stack([elph.gaussian_local_mi(x, y) for x, y in zip(X, Y)])

    unchunked_mi = gaussian_mi_torch(X, Y, batch_size=None)
    chunked_mi = gaussian_mi_torch(X, Y, batch_size=2)
    reverse_mi = gaussian_mi_torch(Y, X, batch_size=2)
    chunked_local = gaussian_local_mi_torch(X, Y, batch_size=2)

    np.testing.assert_allclose(unchunked_mi, expected_mi, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(chunked_mi, unchunked_mi, rtol=0, atol=2e-13)
    np.testing.assert_allclose(reverse_mi, chunked_mi, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(chunked_local, expected_local, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(chunked_local.mean(axis=1), chunked_mi, atol=2e-12)

    # A single Y is broadcast across the independent X batches.
    expected_broadcast = np.array([elph.gaussian_mi(x, Y[0]) for x in X])
    actual_broadcast = gaussian_mi_torch(X, Y[0], batch_size=3)
    np.testing.assert_allclose(actual_broadcast, expected_broadcast, rtol=2e-11, atol=2e-11)


def test_base_and_float32_are_explicit(rng, torch_module):
    X, Y = _batched_correlated_data(rng, batches=3, samples=800)
    nats = gaussian_mi_torch(X, Y, dtype="float64", batch_size=2)
    bits = gaussian_mi_torch(X, Y, base=2.0, dtype="float64", batch_size=2)
    np.testing.assert_allclose(bits, nats / np.log(2.0), rtol=2e-12, atol=2e-12)

    local32 = gaussian_local_mi_torch(X, Y, dtype="float32", batch_size=2)
    assert local32.dtype == np.float32
    expected = np.stack([elph.gaussian_local_mi(x, y) for x, y in zip(X, Y)])
    np.testing.assert_allclose(local32, expected, rtol=2e-4, atol=2e-4)


def test_emergence_gaussian_parity_and_local_shapes(rng, torch_module):
    dimensions, samples = 4, 700
    innovations = rng.standard_normal((dimensions, samples))
    X = np.zeros_like(innovations)
    X[:, 0] = innovations[:, 0]
    coupling = np.array(
        [
            [0.45, 0.10, 0.00, 0.00],
            [0.00, 0.35, 0.15, 0.00],
            [0.10, 0.00, 0.40, 0.05],
            [0.00, 0.10, 0.00, 0.30],
        ]
    )
    for sample in range(1, samples):
        X[:, sample] = coupling @ X[:, sample - 1] + innovations[:, sample]
    V = np.array([0.5, -0.2, 0.3, 0.4]) @ X + 0.1 * rng.standard_normal(samples)

    cases = [
        (elph.emergence_psi, emergence_psi_torch, {"psi", "v_mi", "x_mi"}),
        (elph.emergence_delta, emergence_delta_torch, {"delta", "v_mi", "x_mi"}),
        (elph.emergence_gamma, emergence_gamma_torch, {"gamma"}),
    ]
    for numpy_fn, torch_fn, local_keys in cases:
        expected = numpy_fn(X, V, tau=2, method="gaussian", return_locals=True)
        actual = torch_fn(
            X,
            V,
            tau=2,
            return_locals=True,
            device="cpu",
            dtype="float64",
            batch_size=2,
        )
        assert actual.value == pytest.approx(expected.value, rel=3e-11, abs=3e-11)
        assert actual.v_mi == pytest.approx(expected.v_mi, rel=3e-11, abs=3e-11)
        assert actual.x_mi == pytest.approx(expected.x_mi, rel=3e-11, abs=3e-11)
        assert actual.locals_ is not None
        assert expected.locals_ is not None
        assert set(actual.locals_) == local_keys
        for key in local_keys:
            assert actual.locals_[key].shape == expected.locals_[key].shape
            np.testing.assert_allclose(
                actual.locals_[key], expected.locals_[key], rtol=3e-11, atol=3e-11
            )


def test_invalid_configuration_never_falls_back(rng, torch_module):
    X, Y = _batched_correlated_data(rng, batches=2, samples=20)
    with pytest.raises(ValueError, match="dtype"):
        gaussian_mi_torch(X, Y, dtype="float16")
    with pytest.raises(ValueError, match="batch_size"):
        gaussian_mi_torch(X, Y, batch_size=0)
    with pytest.raises(ValueError, match="invalid Torch device"):
        gaussian_mi_torch(X, Y, device="not-a-device")
    if not torch_module.cuda.is_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            gaussian_mi_torch(X, Y, device="cuda")


def test_input_validation(rng, torch_module):
    X, Y = _batched_correlated_data(rng, batches=3, samples=20)
    with pytest.raises(ValueError, match="number of samples"):
        gaussian_local_mi_torch(X, Y[..., :-1])
    with pytest.raises(ValueError, match="batch dimensions"):
        gaussian_mi_torch(X, np.repeat(Y, 2, axis=0))
    with pytest.raises(ValueError, match="tau"):
        emergence_psi_torch(X[0], Y[0, 0], tau=0)
