"""Parity and validation tests for the optional MVGC Torch kernels."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc
from complexbox.mvgc import _torch as torch_kernels
from complexbox.mvgc.ss import ss2itrfun, ss2trfun, ss_to_cpsd
from complexbox.mvgc.var import var2itrfun, var2trfun, var_to_cpsd

VAR_A = np.array(
    [
        [[0.35, -0.10], [0.08, 0.02]],
        [[-0.04, 0.01], [0.25, -0.08]],
    ]
).transpose(0, 2, 1)
VAR_V = np.array([[1.0, 0.15], [0.15, 0.8]])

SS_A = np.array(
    [
        [0.40, 0.03, 0.00],
        [-0.02, 0.30, 0.04],
        [0.00, -0.01, 0.20],
    ]
)
SS_C = np.array([[0.50, 0.10, -0.05], [0.02, 0.35, 0.08]])
SS_K = np.array([[0.15, -0.02], [0.03, 0.12], [-0.01, 0.05]])
SS_V = np.array([[0.9, -0.1], [-0.1, 1.1]])

FRES = 31
KERNEL_NAMES = (
    "var2trfun",
    "var2itrfun",
    "ss2trfun",
    "ss2itrfun",
    "var_to_cpsd",
    "ss_to_cpsd",
)


@pytest.fixture(scope="module")
def torch_runtime():
    return pytest.importorskip("torch")


def _reference(name: str) -> np.ndarray:
    if name == "var2trfun":
        return var2trfun(VAR_A, FRES)
    if name == "var2itrfun":
        return var2itrfun(VAR_A, FRES)
    if name == "ss2trfun":
        return ss2trfun(SS_A, SS_C, SS_K, FRES)
    if name == "ss2itrfun":
        return ss2itrfun(SS_A, SS_C, SS_K, FRES)
    if name == "var_to_cpsd":
        return var_to_cpsd(VAR_A, VAR_V, FRES)
    if name == "ss_to_cpsd":
        return ss_to_cpsd(SS_A, SS_C, SS_K, SS_V, FRES)
    raise AssertionError(f"unknown test kernel {name}")


def _accelerated(name: str, **options) -> np.ndarray:
    if name == "var2trfun":
        return torch_kernels.var2trfun(VAR_A, FRES, **options)
    if name == "var2itrfun":
        return torch_kernels.var2itrfun(VAR_A, FRES, **options)
    if name == "ss2trfun":
        return torch_kernels.ss2trfun(SS_A, SS_C, SS_K, FRES, **options)
    if name == "ss2itrfun":
        return torch_kernels.ss2itrfun(SS_A, SS_C, SS_K, FRES, **options)
    if name == "var_to_cpsd":
        return torch_kernels.var_to_cpsd(VAR_A, VAR_V, FRES, **options)
    if name == "ss_to_cpsd":
        return torch_kernels.ss_to_cpsd(SS_A, SS_C, SS_K, SS_V, FRES, **options)
    raise AssertionError(f"unknown test kernel {name}")


def test_module_import_is_lazy():
    """The optional module must not retain a module-level Torch dependency."""
    assert "torch" not in vars(torch_kernels)


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_cpu_float64_matches_numpy(name, torch_runtime):
    expected = _reference(name)
    actual = _accelerated(name)

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.complex128
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_public_backend_dispatch(name, torch_runtime):
    if name == "var2trfun":
        actual = mvgc.var2trfun(VAR_A, FRES, backend="torch", batch_size=5)
    elif name == "var2itrfun":
        actual = mvgc.var2itrfun(VAR_A, FRES, backend="torch", batch_size=5)
    elif name == "ss2trfun":
        actual = mvgc.ss2trfun(SS_A, SS_C, SS_K, FRES, backend="torch", batch_size=5)
    elif name == "ss2itrfun":
        actual = mvgc.ss2itrfun(SS_A, SS_C, SS_K, FRES, backend="torch", batch_size=5)
    elif name == "var_to_cpsd":
        actual = mvgc.var_to_cpsd(VAR_A, VAR_V, FRES, backend="torch", batch_size=5)
    else:
        actual = mvgc.ss_to_cpsd(SS_A, SS_C, SS_K, SS_V, FRES, backend="torch", batch_size=5)
    np.testing.assert_allclose(actual, _reference(name), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("name", KERNEL_NAMES)
@pytest.mark.parametrize("batch_size", [1, 2, 7, FRES + 1, FRES + 20])
def test_frequency_chunking_matches_single_batch(name, batch_size, torch_runtime):
    expected = _accelerated(name, device="cpu", dtype="float64", batch_size=None)
    actual = _accelerated(
        name,
        device=torch_runtime.device("cpu"),
        dtype=torch_runtime.complex128,
        batch_size=batch_size,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-14)


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_cpu_float32_has_complex64_output_and_expected_accuracy(name, torch_runtime):
    expected = _reference(name)
    actual = _accelerated(name, device="cpu", dtype=np.float32, batch_size=5)

    assert actual.dtype == np.complex64
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("dtype", [None, "float16", np.float16, np.int64, "not-a-dtype"])
def test_invalid_dtype_raises(dtype, torch_runtime):
    with pytest.raises(ValueError, match="dtype"):
        torch_kernels.var2trfun(VAR_A, FRES, dtype=dtype)


@pytest.mark.parametrize("device", ["not-a-device", "meta"])
def test_invalid_device_raises(device, torch_runtime):
    with pytest.raises(ValueError, match="device"):
        torch_kernels.var2trfun(VAR_A, FRES, device=device)


def test_unavailable_cuda_raises_instead_of_falling_back(torch_runtime):
    if torch_runtime.cuda.is_available():
        pytest.skip("CUDA is available on this runner")
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        torch_kernels.var2trfun(VAR_A, FRES, device="cuda")


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_raises(batch_size, torch_runtime):
    error = TypeError if isinstance(batch_size, (float, bool)) else ValueError
    with pytest.raises(error, match="batch_size"):
        torch_kernels.var2trfun(VAR_A, FRES, batch_size=batch_size)


def test_invalid_model_shapes_raise(torch_runtime):
    with pytest.raises(ValueError, match=r"\(n, n, p\)"):
        torch_kernels.var2trfun(np.eye(2), FRES)
    with pytest.raises(ValueError, match=r"\(r, n\)"):
        torch_kernels.ss2trfun(SS_A, SS_C, SS_K[:, :1], FRES)
    with pytest.raises(ValueError, match=r"\(n, n\)"):
        torch_kernels.var_to_cpsd(VAR_A, np.eye(3), FRES)


@pytest.mark.parametrize("fres", [0, -1, 2.5, True])
def test_invalid_frequency_resolution_raises(fres, torch_runtime):
    error = TypeError if isinstance(fres, (float, bool)) else ValueError
    with pytest.raises(error, match="fres"):
        torch_kernels.var2trfun(VAR_A, fres)
