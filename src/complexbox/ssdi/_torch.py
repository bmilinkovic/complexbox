"""Optional batched Torch kernels for SSDI optimisation.

This module deliberately imports :mod:`torch` lazily.  The NumPy routines in
``dd.py`` and ``optimise.py`` remain the MATLAB-parity reference; the helpers
here accept NumPy arrays and return the same NumPy-facing multi-run tuple as
``opt_gd_dd{x,s}_mruns``.

All calculations use float64/complex128.  A requested device is never changed
silently: unsupported devices or dtypes raise a clear error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "cak2ddx_torch",
    "cak2ddxgrad_torch",
    "trfun2dd_torch",
    "trfun2ddgrad_torch",
    "opt_gd_ddx_mruns_torch",
    "opt_gd_dds_mruns_torch",
]


def _require_torch():
    """Import Torch only when a Torch backend function is requested."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "The Torch SSDI backend requires PyTorch. Install the optional "
            "Torch dependency and retry with backend='torch'."
        ) from exc
    return torch


def _resolve_device(torch, device: str | None, *, complex_required: bool):
    """Resolve and validate a device without falling back or changing dtype."""
    resolved = torch.device("cpu" if device is None else device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested Torch device {resolved!s} is not available")
    if resolved.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(f"requested Torch device {resolved!s} is not available")

    try:
        torch.empty((), dtype=torch.float64, device=resolved)
        if complex_required:
            torch.empty((), dtype=torch.complex128, device=resolved)
    except (RuntimeError, TypeError) as exc:
        required = "float64/complex128" if complex_required else "float64"
        raise RuntimeError(
            f"Torch device {resolved!s} does not support the required {required} SSDI parity dtype"
        ) from exc
    return resolved


def _positive_chunk_size(value: int | None, total: int, name: str) -> int:
    if value is None:
        return max(total, 1)
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_bases(L: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    bases = np.asarray(L, dtype=np.float64)
    if bases.ndim == 2:
        bases = bases[:, :, None]
    if bases.ndim != 3:
        raise ValueError("L must have shape (n, m) or (n, m, runs)")
    if bases.shape[1] > bases.shape[0]:
        raise ValueError("subspace dimension m must not exceed ambient dimension n")
    if bases.shape[2] < 1:
        raise ValueError("L must contain at least one run")
    return bases


def _validate_proxy_inputs(
    L: npt.NDArray[np.floating], CAK: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    bases = _validate_bases(L)
    cak = np.asarray(CAK, dtype=np.float64)
    if cak.ndim != 3 or cak.shape[0] != cak.shape[1]:
        raise ValueError("CAK must have shape (n, n, lags)")
    if cak.shape[0] != bases.shape[0]:
        raise ValueError("CAK and L must have the same ambient dimension")
    if cak.shape[2] < 1:
        raise ValueError("CAK must contain at least one lag")
    return bases, cak


def _validate_spectral_inputs(
    L: npt.NDArray[np.floating], H: npt.NDArray[np.complexfloating]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    bases = _validate_bases(L)
    transfer = np.asarray(H, dtype=np.complex128)
    if transfer.ndim != 3 or transfer.shape[0] != transfer.shape[1]:
        raise ValueError("H must have shape (n, n, frequencies)")
    if transfer.shape[0] != bases.shape[0]:
        raise ValueError("H and L must have the same ambient dimension")
    if transfer.shape[2] < 2:
        raise ValueError("H must contain at least two frequency bins")
    return bases, transfer


def _parse_gdls(gdls: float | tuple[float, float]) -> tuple[float, float]:
    if np.isscalar(gdls):
        ifac = float(gdls)
        if ifac <= 0.0:
            raise ValueError("gdls must be positive")
        return ifac, 1.0 / ifac
    if len(gdls) != 2:
        raise ValueError("gdls must be a scalar or a two-element tuple")
    ifac, nfac = float(gdls[0]), float(gdls[1])
    if ifac <= 0.0 or nfac <= 0.0:
        raise ValueError("gdls factors must be positive")
    return ifac, nfac


def _parse_tol(
    tol: float | tuple[float, float, float], *, spectral: bool
) -> tuple[float, float, float]:
    if np.isscalar(tol):
        value = float(tol)
        # MATLAB SSDI uses tol for all spectral criteria, but tol/10 for the
        # proxy-gradient criterion.
        return value, value, value if spectral else value / 10.0
    if len(tol) != 3:
        raise ValueError("tol must be a scalar or a three-element tuple")
    return float(tol[0]), float(tol[1]), float(tol[2])


def _as_batched_basis(torch, bases: npt.NDArray[np.float64], device):
    return torch.as_tensor(bases, dtype=torch.float64, device=device).permute(2, 0, 1).contiguous()


def _orthonormalise_batch(torch, X):
    """SVD retraction matching SSDI-1's ``orthonormalise.m``."""
    U, _, _ = torch.linalg.svd(X, full_matrices=False)
    return U


def _proxy_value_tensor(torch, L, Q, lag_chunk_size: int):
    """Proxy objective for ``L=(runs,n,m)``, ``Q=(lags,n,n)``."""
    out = torch.zeros(L.shape[0], dtype=torch.float64, device=L.device)
    LT = L.transpose(1, 2)
    for start in range(0, Q.shape[0], lag_chunk_size):
        q = Q[start : start + lag_chunk_size]
        lq = torch.einsum("bmi,kij->bkmj", LT, q)
        lql = torch.einsum("bkmi,bin->bkmn", lq, L)
        out = out + torch.sum(lq * lq, dim=(1, 2, 3))
        out = out - torch.sum(lql * lql, dim=(1, 2, 3))
    return out


def _proxy_grad_tensor(torch, L, Q, lag_chunk_size: int):
    """Analytic Grassmann gradient without materialising ``runs*lags*n*n``."""
    qqt = torch.zeros((Q.shape[1], Q.shape[2]), dtype=torch.float64, device=Q.device)
    for start in range(0, Q.shape[0], lag_chunk_size):
        q = Q[start : start + lag_chunk_size]
        qqt = qqt + torch.sum(torch.matmul(q, q.transpose(1, 2)), dim=0)

    raw = torch.matmul(qqt.unsqueeze(0), L)
    for start in range(0, Q.shape[0], lag_chunk_size):
        q = Q[start : start + lag_chunk_size]
        q_l = torch.einsum("kij,bjm->bkim", q, L)
        qt_l = torch.einsum("kji,bjm->bkim", q, L)
        lt_q_l = torch.einsum("bim,bkin->bkmn", L, q_l)
        lt_qt_l = torch.einsum("bim,bkin->bkmn", L, qt_l)
        raw = raw - torch.sum(torch.einsum("bkim,bkmn->bkin", qt_l, lt_q_l), dim=1)
        raw = raw - torch.sum(torch.einsum("bkim,bkmn->bkin", q_l, lt_qt_l), dim=1)

    grad = 2.0 * raw
    grad = grad - torch.matmul(L, torch.matmul(L.transpose(1, 2), grad))
    magnitude = torch.linalg.vector_norm(grad, dim=(1, 2))
    return grad, magnitude


def _frequency_weights(torch, nfreq: int, device):
    weights = torch.ones(nfreq, dtype=torch.float64, device=device)
    weights[0] = 0.5
    weights[-1] = 0.5
    return weights / float(nfreq - 1)


def _spectral_terms(torch, L, H_chunk):
    """Return ``H*L`` and its Hermitian Gram matrix."""
    Lc = L.to(dtype=torch.complex128)
    HH = H_chunk.conj().transpose(1, 2)
    hl = torch.einsum("fij,bjm->bfim", HH, Lc)
    gram = torch.einsum("bfim,bfin->bfmn", hl.conj(), hl)
    # Remove only roundoff-level non-Hermiticity; this does not add jitter or
    # otherwise conceal a singular projected transfer matrix.
    gram = 0.5 * (gram + gram.conj().transpose(-2, -1))
    chol, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    if bool(torch.any(info != 0).item()):
        bad = torch.nonzero(info != 0, as_tuple=False)[0].detach().cpu().tolist()
        raise np.linalg.LinAlgError(
            "projected spectral matrix is not positive definite "
            f"(batch index {bad[0]}, frequency-chunk index {bad[1]})"
        )
    return hl, chol


def _spectral_value_tensor(torch, L, H, frequency_chunk_size: int):
    nfreq = H.shape[0]
    weights = _frequency_weights(torch, nfreq, L.device)
    out = torch.zeros(L.shape[0], dtype=torch.float64, device=L.device)
    for start in range(0, nfreq, frequency_chunk_size):
        stop = min(start + frequency_chunk_size, nfreq)
        _, chol = _spectral_terms(torch, L, H[start:stop])
        diag = torch.real(torch.diagonal(chol, dim1=-2, dim2=-1))
        logdet = 2.0 * torch.sum(torch.log(diag), dim=-1)
        out = out + torch.sum(logdet * weights[start:stop].unsqueeze(0), dim=1)
    return out


def _spectral_grad_tensor(torch, L, H, frequency_chunk_size: int):
    nfreq = H.shape[0]
    weights = _frequency_weights(torch, nfreq, L.device)
    grad = torch.zeros_like(L)
    for start in range(0, nfreq, frequency_chunk_size):
        stop = min(start + frequency_chunk_size, nfreq)
        h = H[start:stop]
        hl, chol = _spectral_terms(torch, L, h)
        hhl = torch.einsum("fij,bfjm->bfim", h, hl)
        # Right division by the Hermitian Gram matrix, implemented through a
        # Cholesky solve rather than inverse/pseudoinverse.
        rhs = hhl.conj().transpose(-2, -1)
        solved = torch.cholesky_solve(rhs, chol, upper=False)
        right_divided = solved.conj().transpose(-2, -1)
        weighted = weights[start:stop].reshape(1, -1, 1, 1)
        grad = grad + 2.0 * torch.sum(weighted * torch.real(right_divided), dim=1)

    # MATLAB trfun2ddgrad: the integrated Euclidean derivative has
    # L'G = 2I, so this single subtraction is exactly the Grassmann projection.
    grad = grad - 2.0 * L
    magnitude = torch.linalg.vector_norm(grad, dim=(1, 2))
    return grad, magnitude


def _batched_primitive(
    bases: npt.NDArray[np.float64],
    *,
    torch,
    device,
    run_chunk_size: int,
    kernel: Callable[[Any], Any],
    gradient: bool,
):
    values: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    for start in range(0, bases.shape[2], run_chunk_size):
        stop = min(start + run_chunk_size, bases.shape[2])
        L = _as_batched_basis(torch, bases[:, :, start:stop], device)
        result = kernel(L)
        if gradient:
            grad, magnitude = result
            values.append(grad.permute(1, 2, 0).detach().cpu().numpy())
            magnitudes.append(magnitude.detach().cpu().numpy())
        else:
            values.append(result.detach().cpu().numpy())
    if gradient:
        return np.concatenate(values, axis=2), np.concatenate(magnitudes)
    return np.concatenate(values)


def cak2ddx_torch(
    L: npt.NDArray[np.floating],
    CAK: npt.NDArray[np.floating],
    *,
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    lag_chunk_size: int | None = None,
) -> npt.NDArray[np.float64]:
    """Batched proxy DD values, returned as a ``(runs,)`` NumPy array."""
    bases, cak = _validate_proxy_inputs(L, CAK)
    torch = _require_torch()
    resolved = _resolve_device(torch, device, complex_required=False)
    run_chunk = _positive_chunk_size(run_chunk_size, bases.shape[2], "run_chunk_size")
    lag_chunk = _positive_chunk_size(lag_chunk_size, cak.shape[2], "lag_chunk_size")
    Q = torch.as_tensor(cak, dtype=torch.float64, device=resolved).permute(2, 0, 1).contiguous()
    with torch.no_grad():
        return _batched_primitive(
            bases,
            torch=torch,
            device=resolved,
            run_chunk_size=run_chunk,
            kernel=lambda lb: _proxy_value_tensor(torch, lb, Q, lag_chunk),
            gradient=False,
        )


def cak2ddxgrad_torch(
    L: npt.NDArray[np.floating],
    CAK: npt.NDArray[np.floating],
    *,
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    lag_chunk_size: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Batched analytic proxy gradients and their Frobenius magnitudes."""
    bases, cak = _validate_proxy_inputs(L, CAK)
    torch = _require_torch()
    resolved = _resolve_device(torch, device, complex_required=False)
    run_chunk = _positive_chunk_size(run_chunk_size, bases.shape[2], "run_chunk_size")
    lag_chunk = _positive_chunk_size(lag_chunk_size, cak.shape[2], "lag_chunk_size")
    Q = torch.as_tensor(cak, dtype=torch.float64, device=resolved).permute(2, 0, 1).contiguous()
    with torch.no_grad():
        return _batched_primitive(
            bases,
            torch=torch,
            device=resolved,
            run_chunk_size=run_chunk,
            kernel=lambda lb: _proxy_grad_tensor(torch, lb, Q, lag_chunk),
            gradient=True,
        )


def trfun2dd_torch(
    L: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
    *,
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    frequency_chunk_size: int | None = None,
) -> npt.NDArray[np.float64]:
    """Batched spectral DD values, returned as a ``(runs,)`` NumPy array."""
    bases, transfer = _validate_spectral_inputs(L, H)
    torch = _require_torch()
    resolved = _resolve_device(torch, device, complex_required=True)
    run_chunk = _positive_chunk_size(run_chunk_size, bases.shape[2], "run_chunk_size")
    freq_chunk = _positive_chunk_size(
        frequency_chunk_size, transfer.shape[2], "frequency_chunk_size"
    )
    Ht = (
        torch.as_tensor(transfer, dtype=torch.complex128, device=resolved)
        .permute(2, 0, 1)
        .contiguous()
    )
    with torch.no_grad():
        return _batched_primitive(
            bases,
            torch=torch,
            device=resolved,
            run_chunk_size=run_chunk,
            kernel=lambda lb: _spectral_value_tensor(torch, lb, Ht, freq_chunk),
            gradient=False,
        )


def trfun2ddgrad_torch(
    L: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
    *,
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    frequency_chunk_size: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Batched MATLAB-parity spectral gradients and Frobenius magnitudes."""
    bases, transfer = _validate_spectral_inputs(L, H)
    torch = _require_torch()
    resolved = _resolve_device(torch, device, complex_required=True)
    run_chunk = _positive_chunk_size(run_chunk_size, bases.shape[2], "run_chunk_size")
    freq_chunk = _positive_chunk_size(
        frequency_chunk_size, transfer.shape[2], "frequency_chunk_size"
    )
    Ht = (
        torch.as_tensor(transfer, dtype=torch.complex128, device=resolved)
        .permute(2, 0, 1)
        .contiguous()
    )
    with torch.no_grad():
        return _batched_primitive(
            bases,
            torch=torch,
            device=resolved,
            run_chunk_size=run_chunk,
            kernel=lambda lb: _spectral_grad_tensor(torch, lb, Ht, freq_chunk),
            gradient=True,
        )


def _optimise_tensor_batch(
    torch,
    L,
    *,
    objective: Callable[[Any], Any],
    gradient: Callable[[Any], tuple[Any, Any]],
    maxiters: int,
    variant: int,
    gdsig0: float,
    ifac: float,
    nfac: float,
    stol: float,
    dtol: float,
    gtol: float,
    history: bool,
):
    """Optimise one restart chunk while freezing individually stopped runs."""
    nruns = L.shape[0]
    dd = objective(L)
    grad, gmag = gradient(L)
    sigma = torch.full((nruns,), float(gdsig0), dtype=torch.float64, device=L.device)
    active = torch.ones(nruns, dtype=torch.bool, device=L.device)
    convergence = torch.zeros(nruns, dtype=torch.int64, device=L.device)
    stop_iters = torch.full((nruns,), int(maxiters), dtype=torch.int64, device=L.device)

    hist_tensor = None
    if history:
        hist_tensor = torch.empty((maxiters, nruns, 3), dtype=torch.float64, device=L.device)
        hist_tensor[0, :, :] = torch.stack((dd, sigma, gmag), dim=1)

    for iteration in range(2, maxiters + 1):
        safe_mag = torch.where(gmag > 0.0, gmag, torch.ones_like(gmag))
        step = sigma[:, None, None] * grad / safe_mag[:, None, None]
        step = torch.where(active[:, None, None], step, torch.zeros_like(step))
        candidate_all = _orthonormalise_batch(torch, L - step)
        candidate = torch.where(active[:, None, None], candidate_all, L)

        if variant == 1:
            dd_try = objective(candidate)
            accept = active & (dd_try < dd)
            L = torch.where(accept[:, None, None], candidate, L)
            dd = torch.where(accept, dd_try, dd)
            sigma_active = torch.where(accept, sigma * ifac, sigma * nfac)
            sigma = torch.where(active, sigma_active, sigma)
            grad_try, gmag_try = gradient(L)
            grad = torch.where(accept[:, None, None], grad_try, grad)
            gmag = torch.where(accept, gmag_try, gmag)
        else:
            L = candidate
            grad_new, gmag_new = gradient(L)
            dd_new = objective(L)
            improve = active & (dd_new < dd)
            dd = torch.where(improve, dd_new, dd)
            sigma_active = torch.where(improve, sigma * ifac, sigma * nfac)
            sigma = torch.where(active, sigma_active, sigma)
            grad = torch.where(active[:, None, None], grad_new, grad)
            gmag = torch.where(active, gmag_new, gmag)

        if hist_tensor is not None:
            hist_tensor[iteration - 1, :, :] = torch.stack((dd, sigma, gmag), dim=1)

        # Preserve MATLAB's ordered elseif semantics.
        c1 = active & (sigma < stol)
        remaining = active & ~c1
        c2 = remaining & (dd < dtol)
        remaining = remaining & ~c2
        c3 = remaining & (gmag < gtol)
        stopped = c1 | c2 | c3
        convergence = torch.where(c1, torch.ones_like(convergence), convergence)
        convergence = torch.where(c2, torch.full_like(convergence, 2), convergence)
        convergence = torch.where(c3, torch.full_like(convergence, 3), convergence)
        stop_iters = torch.where(stopped, torch.full_like(stop_iters, iteration), stop_iters)
        active = active & ~stopped
        if not bool(torch.any(active).item()):
            break

    histories: list[npt.NDArray[np.float64] | None]
    if hist_tensor is None:
        histories = [None] * nruns
    else:
        hist_np = hist_tensor.detach().cpu().numpy()
        stop_np = stop_iters.detach().cpu().numpy()
        histories = [hist_np[: int(stop_np[k]), k, :].copy() for k in range(nruns)]

    return (
        dd.detach().cpu().numpy(),
        L.permute(1, 2, 0).detach().cpu().numpy(),
        convergence.detach().cpu().numpy().astype(int),
        histories,
    )


def _optimise_mruns(
    bases: npt.NDArray[np.float64],
    *,
    torch,
    device,
    run_chunk_size: int,
    objective: Callable[[Any], Any],
    gradient: Callable[[Any], tuple[Any, Any]],
    maxiters: int,
    variant: int,
    gdsig0: float,
    ifac: float,
    nfac: float,
    stol: float,
    dtol: float,
    gtol: float,
    history: bool,
):
    dds_parts: list[np.ndarray] = []
    bases_parts: list[np.ndarray] = []
    conv_parts: list[np.ndarray] = []
    histories: list[npt.NDArray[np.float64] | None] = []

    with torch.no_grad():
        for start in range(0, bases.shape[2], run_chunk_size):
            stop = min(start + run_chunk_size, bases.shape[2])
            L = _as_batched_basis(torch, bases[:, :, start:stop], device)
            dds, optimised, convergence, chunk_histories = _optimise_tensor_batch(
                torch,
                L,
                objective=objective,
                gradient=gradient,
                maxiters=maxiters,
                variant=variant,
                gdsig0=gdsig0,
                ifac=ifac,
                nfac=nfac,
                stol=stol,
                dtol=dtol,
                gtol=gtol,
                history=history,
            )
            dds_parts.append(dds)
            bases_parts.append(optimised)
            conv_parts.append(convergence)
            histories.extend(chunk_histories)

    dds_all = np.concatenate(dds_parts)
    bases_all = np.concatenate(bases_parts, axis=2)
    conv_all = np.concatenate(conv_parts)
    order = np.argsort(dds_all, kind="stable")
    return (
        dds_all[order],
        bases_all[:, :, order],
        [int(conv_all[i]) for i in order],
        [histories[i] for i in order],
    )


def opt_gd_ddx_mruns_torch(
    CAK: npt.NDArray[np.floating],
    L0_all: npt.NDArray[np.floating],
    maxiters: int = 10_000,
    variant: int = 2,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
    *,
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    lag_chunk_size: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[int],
    list[npt.NDArray[np.float64] | None],
]:
    """Batched Torch proxy-DD multi-restart optimisation.

    The returned tuple exactly follows ``opt_gd_ddx_mruns``:
    ``(sorted_dds, sorted_Ls, convergence_codes, histories)``.
    """
    if maxiters < 1:
        raise ValueError("maxiters must be at least 1")
    if variant not in (1, 2):
        raise ValueError("variant must be 1 or 2")
    bases, cak = _validate_proxy_inputs(L0_all, CAK)
    torch = _require_torch()
    resolved = _resolve_device(torch, device, complex_required=False)
    run_chunk = _positive_chunk_size(run_chunk_size, bases.shape[2], "run_chunk_size")
    lag_chunk = _positive_chunk_size(lag_chunk_size, cak.shape[2], "lag_chunk_size")
    ifac, nfac = _parse_gdls(gdls)
    stol, dtol, gtol = _parse_tol(tol, spectral=False)
    Q = torch.as_tensor(cak, dtype=torch.float64, device=resolved).permute(2, 0, 1).contiguous()
    return _optimise_mruns(
        bases,
        torch=torch,
        device=resolved,
        run_chunk_size=run_chunk,
        objective=lambda lb: _proxy_value_tensor(torch, lb, Q, lag_chunk),
        gradient=lambda lb: _proxy_grad_tensor(torch, lb, Q, lag_chunk),
        maxiters=maxiters,
        variant=variant,
        gdsig0=gdsig0,
        ifac=ifac,
        nfac=nfac,
        stol=stol,
        dtol=dtol,
        gtol=gtol,
        history=history,
    )


def opt_gd_dds_mruns_torch(
    H: npt.NDArray[np.complexfloating],
    L0_all: npt.NDArray[np.floating],
    maxiters: int = 10_000,
    variant: int = 2,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
    *,
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    frequency_chunk_size: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[int],
    list[npt.NDArray[np.float64] | None],
]:
    """Batched Torch spectral-DD multi-restart optimisation."""
    if maxiters < 1:
        raise ValueError("maxiters must be at least 1")
    if variant not in (1, 2):
        raise ValueError("variant must be 1 or 2")
    bases, transfer = _validate_spectral_inputs(L0_all, H)
    torch = _require_torch()
    resolved = _resolve_device(torch, device, complex_required=True)
    run_chunk = _positive_chunk_size(run_chunk_size, bases.shape[2], "run_chunk_size")
    freq_chunk = _positive_chunk_size(
        frequency_chunk_size, transfer.shape[2], "frequency_chunk_size"
    )
    ifac, nfac = _parse_gdls(gdls)
    stol, dtol, gtol = _parse_tol(tol, spectral=True)
    Ht = (
        torch.as_tensor(transfer, dtype=torch.complex128, device=resolved)
        .permute(2, 0, 1)
        .contiguous()
    )
    return _optimise_mruns(
        bases,
        torch=torch,
        device=resolved,
        run_chunk_size=run_chunk,
        objective=lambda lb: _spectral_value_tensor(torch, lb, Ht, freq_chunk),
        gradient=lambda lb: _spectral_grad_tensor(torch, lb, Ht, freq_chunk),
        maxiters=maxiters,
        variant=variant,
        gdsig0=gdsig0,
        ifac=ifac,
        nfac=nfac,
        stol=stol,
        dtol=dtol,
        gtol=gtol,
        history=history,
    )
