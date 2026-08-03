"""Optional PyTorch acceleration for MVGC frequency-domain kernels.

The public functions in this module deliberately accept and return NumPy
arrays.  PyTorch is imported only when a function is called, so importing
``complexbox`` does not make Torch a runtime dependency.  Computation defaults
to CPU double precision (``float64`` inputs and ``complex128`` spectra).

No function silently changes device or precision.  An unavailable device or
unsupported dtype raises an explicit exception.
"""

from __future__ import annotations

import importlib
import operator
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "var2trfun",
    "var2itrfun",
    "ss2trfun",
    "ss2itrfun",
    "var_to_cpsd",
    "ss_to_cpsd",
]


def _import_torch() -> Any:
    """Import Torch on first use and provide an actionable optional-dependency error."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:  # pragma: no cover - exercised without Torch
        raise ImportError(
            "PyTorch is required for complexbox.mvgc._torch; install a compatible "
            "PyTorch build for the requested device"
        ) from exc


def _resolve_device(torch: Any, device: str | object) -> Any:
    """Resolve a supported Torch device without falling back to CPU."""
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid Torch device {device!r}") from exc

    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {resolved.index} is not available")
        return resolved
    if resolved.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build")
        return resolved
    raise ValueError("device must be a CPU, CUDA, or MPS Torch device")


def _resolve_dtype(torch: Any, dtype: object) -> tuple[Any, Any, np.dtype, np.dtype]:
    """Return paired Torch/NumPy real and complex dtypes."""
    if dtype is None:
        bits = 0
    elif dtype is torch.float64 or dtype is torch.complex128:
        bits = 64
    elif dtype is torch.float32 or dtype is torch.complex64:
        bits = 32
    else:
        if isinstance(dtype, str):
            key = dtype.strip().lower().removeprefix("torch.").removeprefix("numpy.")
            aliases = {
                "float64": 64,
                "double": 64,
                "complex128": 64,
                "cdouble": 64,
                "float32": 32,
                "single": 32,
                "complex64": 32,
                "cfloat": 32,
            }
            bits = aliases.get(key, 0)
        else:
            try:
                numpy_dtype = np.dtype(dtype)
            except TypeError:
                numpy_dtype = None
            if numpy_dtype in (np.dtype(np.float64), np.dtype(np.complex128)):
                bits = 64
            elif numpy_dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
                bits = 32
            else:
                bits = 0

    if bits == 64:
        return torch.float64, torch.complex128, np.dtype(np.float64), np.dtype(np.complex128)
    if bits == 32:
        return torch.float32, torch.complex64, np.dtype(np.float32), np.dtype(np.complex64)
    raise ValueError("dtype must be float32/complex64 or float64/complex128")


def _backend(device: str | object, dtype: object) -> tuple[Any, Any, Any, Any, np.dtype, np.dtype]:
    torch = _import_torch()
    resolved_device = _resolve_device(torch, device)
    real_dtype, complex_dtype, np_real, np_complex = _resolve_dtype(torch, dtype)
    if resolved_device.type == "mps" and real_dtype is torch.float64:
        raise ValueError(
            "MPS does not support the requested float64/complex128 computation; "
            "pass dtype='float32' explicitly or choose another device"
        )
    return torch, resolved_device, real_dtype, complex_dtype, np_real, np_complex


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


def _frequency_batches(nfreq: int, batch_size: int | None) -> Iterator[tuple[int, int]]:
    if batch_size is None:
        size = nfreq
    else:
        size = _positive_integer(batch_size, "batch_size")
    for start in range(0, nfreq, size):
        yield start, min(start + size, nfreq)


def _real_array(value: npt.ArrayLike, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.any(np.imag(array) != 0):
            raise ValueError(f"{name} must be real-valued")
        array = np.real(array)
    try:
        return np.ascontiguousarray(array, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc


def _var_array(value: npt.ArrayLike, dtype: np.dtype) -> np.ndarray:
    A = _real_array(value, "A", dtype)
    if A.ndim != 3 or A.shape[0] == 0 or A.shape[0] != A.shape[1] or A.shape[2] == 0:
        raise ValueError("A must have shape (n, n, p) with n >= 1 and p >= 1")
    return A


def _ss_arrays(
    A: npt.ArrayLike,
    C: npt.ArrayLike,
    K: npt.ArrayLike,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_array = _real_array(A, "A", dtype)
    C_array = _real_array(C, "C", dtype)
    K_array = _real_array(K, "K", dtype)
    if A_array.ndim != 2 or A_array.shape[0] == 0 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError("A must have shape (r, r) with r >= 1")
    r = A_array.shape[0]
    if C_array.ndim != 2 or C_array.shape[0] == 0 or C_array.shape[1] != r:
        raise ValueError("C must have shape (n, r) matching A")
    n = C_array.shape[0]
    if K_array.shape != (r, n):
        raise ValueError("K must have shape (r, n) matching A and C")
    return A_array, C_array, K_array


def _covariance(value: npt.ArrayLike, n: int, dtype: np.dtype) -> np.ndarray:
    V = _real_array(value, "V", dtype)
    if V.shape != (n, n):
        raise ValueError("V must have shape (n, n) matching the observation dimension")
    return V


def _var_polynomial_batches(
    A: np.ndarray,
    fres: int,
    batches: Iterator[tuple[int, int]],
    torch: Any,
    device: Any,
    real_dtype: Any,
    complex_dtype: Any,
) -> Iterator[tuple[int, int, Any]]:
    """Yield ``I - sum_k A_k exp(-i k omega)`` with frequency first."""
    n, _, p = A.shape
    At = torch.as_tensor(A, dtype=complex_dtype, device=device)
    eye = torch.eye(n, dtype=complex_dtype, device=device)
    lags = torch.arange(1, p + 1, dtype=real_dtype, device=device)
    step = np.pi / fres
    for start, stop in batches:
        omega = torch.arange(start, stop, dtype=real_dtype, device=device) * step
        angles = -omega[:, None] * lags[None, :]
        phase = torch.polar(torch.ones_like(angles), angles)
        polynomial = eye[None, :, :] - torch.einsum("ijp,bp->bij", At, phase)
        yield start, stop, polynomial


def _ss_transfer_batches(
    A: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    fres: int,
    batches: Iterator[tuple[int, int]],
    inverse: bool,
    torch: Any,
    device: Any,
    real_dtype: Any,
    complex_dtype: Any,
) -> Iterator[tuple[int, int, Any]]:
    """Yield innovations-form SS transfer or inverse transfer batches."""
    r = A.shape[0]
    n = C.shape[0]
    At = torch.as_tensor(A, dtype=complex_dtype, device=device)
    Ct = torch.as_tensor(C, dtype=complex_dtype, device=device)
    Kt = torch.as_tensor(K, dtype=complex_dtype, device=device)
    transition = At - Kt @ Ct if inverse else At
    eye_r = torch.eye(r, dtype=complex_dtype, device=device)
    eye_n = torch.eye(n, dtype=complex_dtype, device=device)
    step = np.pi / fres
    for start, stop in batches:
        omega = torch.arange(start, stop, dtype=real_dtype, device=device) * step
        z = torch.polar(torch.ones_like(omega), omega)
        system = z[:, None, None] * eye_r[None, :, :] - transition[None, :, :]
        rhs = Kt[None, :, :].expand(stop - start, -1, -1)
        solved = torch.linalg.solve(system, rhs)
        correction = Ct[None, :, :] @ solved
        transfer = eye_n[None, :, :] - correction if inverse else eye_n[None, :, :] + correction
        yield start, stop, transfer


def _store_frequency_batch(output: np.ndarray, batch: Any, start: int, stop: int) -> None:
    output[:, :, start:stop] = batch.permute(1, 2, 0).detach().cpu().numpy()


def var2itrfun(
    A: npt.ArrayLike,
    fres: int,
    *,
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Calculate a VAR inverse transfer function using batched Torch operations.

    The frequency grid and output layout match :func:`complexbox.mvgc.var2itrfun`:
    ``omega[j] = pi*j/fres`` and output shape ``(n, n, fres + 1)``.
    """
    fres = _positive_integer(fres, "fres")
    torch, dev, real, complex_, np_real, np_complex = _backend(device, dtype)
    A_array = _var_array(A, np_real)
    n = A_array.shape[0]
    nfreq = fres + 1
    output = np.empty((n, n, nfreq), dtype=np_complex)
    batches = _frequency_batches(nfreq, batch_size)
    with torch.inference_mode():
        for start, stop, polynomial in _var_polynomial_batches(
            A_array, fres, batches, torch, dev, real, complex_
        ):
            _store_frequency_batch(output, polynomial, start, stop)
    return output


def var2trfun(
    A: npt.ArrayLike,
    fres: int,
    *,
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Calculate a VAR transfer function using batched Torch linear solves."""
    fres = _positive_integer(fres, "fres")
    torch, dev, real, complex_, np_real, np_complex = _backend(device, dtype)
    A_array = _var_array(A, np_real)
    n = A_array.shape[0]
    nfreq = fres + 1
    output = np.empty((n, n, nfreq), dtype=np_complex)
    batches = _frequency_batches(nfreq, batch_size)
    eye = torch.eye(n, dtype=complex_, device=dev)
    with torch.inference_mode():
        for start, stop, polynomial in _var_polynomial_batches(
            A_array, fres, batches, torch, dev, real, complex_
        ):
            rhs = eye[None, :, :].expand(stop - start, -1, -1)
            transfer = torch.linalg.solve(polynomial, rhs)
            _store_frequency_batch(output, transfer, start, stop)
    return output


def ss2trfun(
    A: npt.ArrayLike,
    C: npt.ArrayLike,
    K: npt.ArrayLike,
    fres: int,
    *,
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Calculate an innovations-form SS transfer function with Torch."""
    fres = _positive_integer(fres, "fres")
    torch, dev, real, complex_, np_real, np_complex = _backend(device, dtype)
    A_array, C_array, K_array = _ss_arrays(A, C, K, np_real)
    n = C_array.shape[0]
    nfreq = fres + 1
    output = np.empty((n, n, nfreq), dtype=np_complex)
    batches = _frequency_batches(nfreq, batch_size)
    with torch.inference_mode():
        for start, stop, transfer in _ss_transfer_batches(
            A_array,
            C_array,
            K_array,
            fres,
            batches,
            False,
            torch,
            dev,
            real,
            complex_,
        ):
            _store_frequency_batch(output, transfer, start, stop)
    return output


def ss2itrfun(
    A: npt.ArrayLike,
    C: npt.ArrayLike,
    K: npt.ArrayLike,
    fres: int,
    *,
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Calculate an innovations-form SS inverse transfer function with Torch."""
    fres = _positive_integer(fres, "fres")
    torch, dev, real, complex_, np_real, np_complex = _backend(device, dtype)
    A_array, C_array, K_array = _ss_arrays(A, C, K, np_real)
    n = C_array.shape[0]
    nfreq = fres + 1
    output = np.empty((n, n, nfreq), dtype=np_complex)
    batches = _frequency_batches(nfreq, batch_size)
    with torch.inference_mode():
        for start, stop, transfer in _ss_transfer_batches(
            A_array,
            C_array,
            K_array,
            fres,
            batches,
            True,
            torch,
            dev,
            real,
            complex_,
        ):
            _store_frequency_batch(output, transfer, start, stop)
    return output


def var_to_cpsd(
    A: npt.ArrayLike,
    V: npt.ArrayLike,
    fres: int,
    *,
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Calculate ``H(omega) V H(omega)*`` for a VAR model with Torch."""
    fres = _positive_integer(fres, "fres")
    torch, dev, real, complex_, np_real, np_complex = _backend(device, dtype)
    A_array = _var_array(A, np_real)
    n = A_array.shape[0]
    V_array = _covariance(V, n, np_real)
    Vt = torch.as_tensor(V_array, dtype=complex_, device=dev)
    eye = torch.eye(n, dtype=complex_, device=dev)
    nfreq = fres + 1
    output = np.empty((n, n, nfreq), dtype=np_complex)
    batches = _frequency_batches(nfreq, batch_size)
    with torch.inference_mode():
        for start, stop, polynomial in _var_polynomial_batches(
            A_array, fres, batches, torch, dev, real, complex_
        ):
            rhs = eye[None, :, :].expand(stop - start, -1, -1)
            transfer = torch.linalg.solve(polynomial, rhs)
            spectrum = transfer @ Vt[None, :, :] @ transfer.mH
            _store_frequency_batch(output, spectrum, start, stop)
    return output


def ss_to_cpsd(
    A: npt.ArrayLike,
    C: npt.ArrayLike,
    K: npt.ArrayLike,
    V: npt.ArrayLike,
    fres: int,
    *,
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Calculate ``H(omega) V H(omega)*`` for an innovations SS model with Torch."""
    fres = _positive_integer(fres, "fres")
    torch, dev, real, complex_, np_real, np_complex = _backend(device, dtype)
    A_array, C_array, K_array = _ss_arrays(A, C, K, np_real)
    n = C_array.shape[0]
    V_array = _covariance(V, n, np_real)
    Vt = torch.as_tensor(V_array, dtype=complex_, device=dev)
    nfreq = fres + 1
    output = np.empty((n, n, nfreq), dtype=np_complex)
    batches = _frequency_batches(nfreq, batch_size)
    with torch.inference_mode():
        for start, stop, transfer in _ss_transfer_batches(
            A_array,
            C_array,
            K_array,
            fres,
            batches,
            False,
            torch,
            dev,
            real,
            complex_,
        ):
            spectrum = transfer @ Vt[None, :, :] @ transfer.mH
            _store_frequency_batch(output, spectrum, start, stop)
    return output
