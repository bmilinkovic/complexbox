r"""Optional Torch acceleration for Gaussian ELPH kernels.

The functions in this module keep NumPy as the public data boundary and load
Torch only when an accelerated function is called.  They support independent
batches of Gaussian mutual-information problems, which is useful for PhiID and
for the :math:`\Psi`, :math:`\Delta`, and :math:`\Gamma` emergence measures.

Batch convention
----------------
An unbatched variable is shaped ``(dimensions, samples)`` (or ``(samples,)``
for a scalar variable).  A batch is shaped ``(batch, dimensions, samples)``.
If only one argument is batched, the unbatched argument is broadcast across
that batch.  ``batch_size`` bounds the number of independent problems moved to
the requested Torch device at once; it never changes the statistical sample
axis.

There is deliberately no automatic device fallback.  Requesting an
unavailable device raises an error so that callers can trust where a
calculation ran.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .emergence import EmergenceResult

__all__ = [
    "gaussian_mi_torch",
    "gaussian_local_mi_torch",
    "emergence_psi_torch",
    "emergence_delta_torch",
    "emergence_gamma_torch",
]


def _require_torch():
    """Import Torch on demand with an actionable optional-dependency error."""
    try:
        import torch
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "Torch acceleration is optional; install complexbox with its Torch "
            "extra or install `torch` directly."
        ) from exc
    return torch


def _resolve_backend(device: str, dtype: str | np.dtype | type):
    """Resolve and validate a Torch device/dtype without falling back."""
    torch = _require_torch()

    try:
        numpy_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError("dtype must be float32 or float64") from exc
    if numpy_dtype == np.dtype(np.float64):
        torch_dtype = torch.float64
    elif numpy_dtype == np.dtype(np.float32):
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype must be float32 or float64")

    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid Torch device {device!r}") from exc

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested Torch device {device!r} is unavailable")
    if torch_device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(f"requested Torch device {device!r} is unavailable")
        if torch_dtype == torch.float64:
            raise ValueError("Torch MPS does not support float64; request dtype='float32'")

    # This also catches an invalid device index and less-common unavailable
    # backends without silently moving the calculation to CPU.
    try:
        torch.empty(0, dtype=torch_dtype, device=torch_device)
    except Exception as exc:  # pragma: no cover - backend-specific exception types
        raise RuntimeError(f"requested Torch device {device!r} is unavailable") from exc

    return torch, torch_device, torch_dtype, numpy_dtype


def _as_batched_pair(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    numpy_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Normalise two variables to ``(batch, dimensions, samples)`` arrays."""

    def normalise(value: npt.ArrayLike, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(value, dtype=numpy_dtype)
        explicitly_batched = array.ndim == 3
        if array.ndim == 1:
            array = array[None, None, :]
        elif array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim != 3:
            raise ValueError(
                f"{name} must have shape (samples,), (dimensions, samples), or "
                "(batch, dimensions, samples)"
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty batch and dimension axes")
        if array.shape[2] < 2:
            raise ValueError("Gaussian MI requires at least two samples")
        return array, explicitly_batched

    Xb, X_explicit = normalise(X, "X")
    Yb, Y_explicit = normalise(Y, "Y")
    if Xb.shape[2] != Yb.shape[2]:
        raise ValueError("X and Y must have the same number of samples")

    batches = max(Xb.shape[0], Yb.shape[0])
    if Xb.shape[0] not in (1, batches) or Yb.shape[0] not in (1, batches):
        raise ValueError("X and Y batch dimensions must match or be one")
    if Xb.shape[0] == 1 and batches > 1:
        Xb = np.broadcast_to(Xb, (batches, *Xb.shape[1:]))
    if Yb.shape[0] == 1 and batches > 1:
        Yb = np.broadcast_to(Yb, (batches, *Yb.shape[1:]))
    return Xb, Yb, X_explicit or Y_explicit


def _validate_batch_size(batch_size: int | None, batches: int) -> int:
    if batch_size is None:
        return batches
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be a positive integer or None")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return min(int(batch_size), batches)


def _covariance(centered: Any) -> Any:
    samples = centered.shape[-1]
    return centered @ centered.transpose(-1, -2) / (samples - 1)


def _cholesky_logdet(torch: Any, covariance: Any) -> tuple[Any, Any, Any]:
    """Batched HPD log determinant and a safe factor for failed entries."""
    factor, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
    valid = info == 0
    dimensions = covariance.shape[-1]
    identity = torch.eye(dimensions, dtype=covariance.dtype, device=covariance.device).expand(
        covariance.shape[0], dimensions, dimensions
    )
    safe_factor = torch.where(valid[:, None, None], factor, identity)
    logdet = 2.0 * torch.log(torch.diagonal(safe_factor, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet = torch.where(valid, logdet, torch.full_like(logdet, float("nan")))
    return logdet, safe_factor, valid


def _logpdf(torch: Any, centered: Any, covariance: Any) -> Any:
    logdet, factor, valid = _cholesky_logdet(torch, covariance)
    solved = torch.cholesky_solve(centered, factor)
    mahalanobis = (centered * solved).sum(dim=1)
    dimensions = centered.shape[1]
    values = -0.5 * dimensions * np.log(2.0 * np.pi) - 0.5 * logdet[:, None] - 0.5 * mahalanobis
    return torch.where(valid[:, None], values, torch.full_like(values, float("nan")))


def _local_mi_chunk(torch: Any, X: Any, Y: Any) -> Any:
    dimensions_x = X.shape[1]
    Z = torch.cat((X, Y), dim=1)
    centered = Z - Z.mean(dim=-1, keepdim=True)
    Xc = centered[:, :dimensions_x, :]
    Yc = centered[:, dimensions_x:, :]
    return (
        _logpdf(torch, centered, _covariance(centered))
        - _logpdf(torch, Xc, _covariance(Xc))
        - _logpdf(torch, Yc, _covariance(Yc))
    )


def gaussian_local_mi_torch(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    *,
    device: str = "cpu",
    dtype: str | np.dtype | type = "float64",
    batch_size: int | None = None,
) -> npt.NDArray[np.floating]:
    """Pointwise Gaussian mutual information using batched Torch linear algebra.

    The result has shape ``(samples,)`` for two unbatched inputs and
    ``(batch, samples)`` if either input explicitly carries a batch axis.
    Values are in nats, matching :func:`complexbox.elph.gaussian_local_mi`.
    Rank-deficient covariance batches return ``NaN`` rather than being
    regularised implicitly.
    """
    torch, torch_device, torch_dtype, numpy_dtype = _resolve_backend(device, dtype)
    Xb, Yb, explicitly_batched = _as_batched_pair(X, Y, numpy_dtype)
    chunk_size = _validate_batch_size(batch_size, Xb.shape[0])

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, Xb.shape[0], chunk_size):
            stop = min(start + chunk_size, Xb.shape[0])
            Xt = torch.as_tensor(
                np.ascontiguousarray(Xb[start:stop]), dtype=torch_dtype, device=torch_device
            )
            Yt = torch.as_tensor(
                np.ascontiguousarray(Yb[start:stop]), dtype=torch_dtype, device=torch_device
            )
            outputs.append(_local_mi_chunk(torch, Xt, Yt).cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    return result if explicitly_batched else result[0]


def _mi_chunk(torch: Any, X: Any, Y: Any, log_base: float) -> Any:
    Z = torch.cat((X, Y), dim=1)
    Xc = X - X.mean(dim=-1, keepdim=True)
    Yc = Y - Y.mean(dim=-1, keepdim=True)
    Zc = Z - Z.mean(dim=-1, keepdim=True)
    logdet_x, _, valid_x = _cholesky_logdet(torch, _covariance(Xc))
    logdet_y, _, valid_y = _cholesky_logdet(torch, _covariance(Yc))
    logdet_z, _, valid_z = _cholesky_logdet(torch, _covariance(Zc))
    values = 0.5 * (logdet_x + logdet_y - logdet_z) / log_base
    valid = valid_x & valid_y & valid_z
    return torch.where(valid, values, torch.full_like(values, float("nan")))


def gaussian_mi_torch(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    base: float = np.e,
    *,
    device: str = "cpu",
    dtype: str | np.dtype | type = "float64",
    batch_size: int | None = None,
) -> float | npt.NDArray[np.floating]:
    """Gaussian mutual information with optional independent batch axes.

    Returns a Python ``float`` for unbatched inputs and a ``(batch,)`` NumPy
    array for explicitly batched inputs.  ``base=np.e`` returns nats and
    ``base=2`` returns bits.
    """
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("base must be finite, positive, and different from one")
    torch, torch_device, torch_dtype, numpy_dtype = _resolve_backend(device, dtype)
    Xb, Yb, explicitly_batched = _as_batched_pair(X, Y, numpy_dtype)
    chunk_size = _validate_batch_size(batch_size, Xb.shape[0])

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, Xb.shape[0], chunk_size):
            stop = min(start + chunk_size, Xb.shape[0])
            Xt = torch.as_tensor(
                np.ascontiguousarray(Xb[start:stop]), dtype=torch_dtype, device=torch_device
            )
            Yt = torch.as_tensor(
                np.ascontiguousarray(Yb[start:stop]), dtype=torch_dtype, device=torch_device
            )
            outputs.append(_mi_chunk(torch, Xt, Yt, float(np.log(base))).cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    return result if explicitly_batched else float(result[0])


def _validate_emergence_inputs(
    X: npt.ArrayLike,
    V: npt.ArrayLike,
    tau: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    X_array = np.atleast_2d(np.asarray(X, dtype=float))
    V_array = np.asarray(V, dtype=float).ravel()
    if X_array.ndim != 2:
        raise ValueError("X must have shape (micro_variables, samples)")
    dimensions, samples = X_array.shape
    if dimensions == 0 or samples == 0:
        raise ValueError("X must be non-empty")
    if V_array.size != samples:
        raise ValueError("X and V must have the same time-length")
    if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)):
        raise TypeError("tau must be an integer")
    if tau < 1 or tau >= samples - 1:
        raise ValueError("tau must satisfy 1 <= tau < samples - 1")
    return X_array, V_array, dimensions, samples


def emergence_psi_torch(
    X: npt.ArrayLike,
    V: npt.ArrayLike,
    tau: int = 1,
    *,
    return_locals: bool = False,
    device: str = "cpu",
    dtype: str | np.dtype | type = "float64",
    batch_size: int | None = None,
) -> EmergenceResult:
    r"""Gaussian :math:`\Psi` emergence with the micro MI terms batched."""
    X, V, _dimensions, samples = _validate_emergence_inputs(X, V, tau)
    V_past = V[: samples - tau]
    V_future = V[tau:]
    v_lmi = gaussian_local_mi_torch(
        V_past,
        V_future,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    x_terms = gaussian_local_mi_torch(
        X[:, None, : samples - tau],
        V_future,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    x_lmi = x_terms.sum(axis=0)
    local_psi = v_lmi - x_lmi
    return EmergenceResult(
        value=float(np.mean(local_psi)),
        v_mi=float(np.mean(v_lmi)),
        x_mi=float(np.mean(x_lmi)),
        locals_={"psi": local_psi, "v_mi": v_lmi, "x_mi": x_lmi} if return_locals else None,
    )


def emergence_delta_torch(
    X: npt.ArrayLike,
    V: npt.ArrayLike,
    tau: int = 1,
    *,
    return_locals: bool = False,
    device: str = "cpu",
    dtype: str | np.dtype | type = "float64",
    batch_size: int | None = None,
) -> EmergenceResult:
    r"""Gaussian :math:`\Delta` emergence with future targets batched."""
    X, V, dimensions, samples = _validate_emergence_inputs(X, V, tau)
    X_past = X[:, : samples - tau]
    X_future = X[:, tau:]
    V_past = V[: samples - tau]

    v_lmi = gaussian_local_mi_torch(
        V_past,
        X_future[:, None, :],
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    ).T
    x_lmi_sum = np.zeros_like(v_lmi)
    # Each call evaluates every future target for one micro-level source.  This
    # reduces D^2 Python scalar calls to D batched calls without materialising a
    # D^2 x T input tensor.
    for source in range(dimensions):
        source_terms = gaussian_local_mi_torch(
            X_past[source],
            X_future[:, None, :],
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        x_lmi_sum += source_terms.T

    local_delta = v_lmi - x_lmi_sum
    max_index = int(np.argmax(np.mean(local_delta, axis=0)))
    local_delta_max = local_delta[:, max_index]
    return EmergenceResult(
        value=float(np.mean(local_delta_max)),
        v_mi=float(np.mean(v_lmi[:, max_index])),
        x_mi=float(np.mean(x_lmi_sum[:, max_index])),
        locals_={"delta": local_delta_max, "v_mi": v_lmi, "x_mi": x_lmi_sum}
        if return_locals
        else None,
    )


def emergence_gamma_torch(
    X: npt.ArrayLike,
    V: npt.ArrayLike,
    tau: int = 1,
    *,
    return_locals: bool = False,
    device: str = "cpu",
    dtype: str | np.dtype | type = "float64",
    batch_size: int | None = None,
) -> EmergenceResult:
    r"""Gaussian :math:`\Gamma` emergence with future targets batched."""
    X, V, _dimensions, samples = _validate_emergence_inputs(X, V, tau)
    gamma_terms = gaussian_local_mi_torch(
        V[: samples - tau],
        X[:, None, tau:],
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    ).T
    max_index = int(np.argmax(np.mean(gamma_terms, axis=0)))
    local_gamma = gamma_terms[:, max_index]
    value = float(np.mean(local_gamma))
    return EmergenceResult(
        value=value,
        v_mi=value,
        x_mi=0.0,
        locals_={"gamma": local_gamma} if return_locals else None,
    )
