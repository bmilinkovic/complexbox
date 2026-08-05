"""Innovations-form state-space conversions and simulation.

Ports of MVGC2's ``core/ss_to_autocov.m``, ``core/ss_to_cpsd.m``,
``core/ss_to_tsdata.m``, and ``core/tsdata_to_ss.m``.

Innovations-form notation: ``z(t+1) = A z(t) + K e(t)``,
``x(t) = C z(t) + e(t)`` with ``e ~ N(0, V)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

from ._lyap import dlyap
from ._utils import specnorm

__all__ = [
    "ss_to_autocov",
    "ss_to_cpsd",
    "ss_to_tsdata",
    "ss2trfun",
    "ss2itrfun",
    "ss_check_fres",
    "ss2fres",
    "tsdata_to_ss",
]


def ss2itrfun(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    fres: int,
    *,
    backend: str = "numpy",
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Inverse transfer function ``H^{-1}(z) = I - C (zI - (A - KC))^{-1} K``.

    Port of ``utils/ss2itrfun.m``. Returns array of shape ``(n, n, fres+1)``.
    """
    if backend == "torch":
        from ._torch import ss2itrfun as ss2itrfun_torch

        return ss2itrfun_torch(A, C, K, fres, device=device, dtype=dtype, batch_size=batch_size)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    A, C, K = (np.asarray(x, dtype=float) for x in (A, C, K))
    n, r = C.shape
    h = fres + 1
    B = A - K @ C
    omega = np.linspace(0.0, np.pi, h)
    z = np.exp(1j * omega)
    Ir = np.eye(r)
    In = np.eye(n)
    J = np.empty((n, n, h), dtype=complex)
    for k in range(h):
        J[:, :, k] = In - C @ np.linalg.solve(z[k] * Ir - B, K)
    return J


def ss_to_autocov(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    qmax: int,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.floating], int]:
    """Autocovariance sequence for an innovations-form SS model.

    Port of ``ss_to_autocov.m``. Recursion::

        Ω = A Ω A' + K V K'
        Γ_0 = C Ω C' + V
        Λ_1 = A Ω C' + K V
        Γ_k = C A^{k-1} Λ_1   for k ≥ 1
    """
    A, C, K, V = (np.asarray(x, dtype=float) for x in (A, C, K, V))
    n = C.shape[0]
    M = dlyap(A, K @ V @ K.T)
    G0 = C @ M @ C.T + V

    if qmax == 0:
        return G0.reshape(n, n, 1), 0

    alags = qmax < 0
    q = -qmax if alags else qmax

    Lam = A @ M @ C.T + K @ V  # Λ_1
    G_list = [G0, C @ Lam]
    if alags:
        for _ in range(2, q + 1):
            Lam = A @ Lam
            G_list.append(C @ Lam)
        return np.stack(G_list, axis=2), q

    if tol is None:
        tol = np.finfo(float).eps * float(np.max(np.abs(G0)))
    k = 1
    while np.max(np.abs(G_list[-1])) > tol:
        if k + 1 > qmax:
            import warnings

            warnings.warn(
                "ss_to_autocov: covariance sequence failed to converge",
                stacklevel=2,
            )
            return np.stack(G_list, axis=2), k
        k += 1
        Lam = A @ Lam
        G_list.append(C @ Lam)
    return np.stack(G_list, axis=2), k


def ss2trfun(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    fres: int,
    *,
    backend: str = "numpy",
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Transfer function ``H(z) = I + C (zI - A)^{-1} K`` of an innovations SS.

    Port of MVGC2's ``ss2trfun.m``. Frequencies ``ω = πj/fres`` for
    ``j = 0, ..., fres`` so output has ``fres + 1`` slices.
    """
    if backend == "torch":
        from ._torch import ss2trfun as ss2trfun_torch

        return ss2trfun_torch(A, C, K, fres, device=device, dtype=dtype, batch_size=batch_size)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    A, C, K = (np.asarray(x, dtype=float) for x in (A, C, K))
    n = C.shape[0]
    r = A.shape[0]
    h = fres + 1
    omega = np.linspace(0.0, np.pi, h)
    z = np.exp(1j * omega)
    Ir = np.eye(r)
    In = np.eye(n)
    H = np.empty((n, n, h), dtype=complex)
    for j in range(h):
        H[:, :, j] = In + C @ np.linalg.solve(z[j] * Ir - A, K)
    return H


def ss_to_cpsd(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    fres: int,
    *,
    backend: str = "numpy",
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """CPSD of an innovations-form SS: ``S(ω) = H(ω) V H(ω)*``."""
    if backend == "torch":
        from ._torch import ss_to_cpsd as ss_to_cpsd_torch

        return ss_to_cpsd_torch(A, C, K, V, fres, device=device, dtype=dtype, batch_size=batch_size)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    H = ss2trfun(A, C, K, fres)
    V = np.asarray(V, dtype=float)
    n, _, h = H.shape
    S = np.empty((n, n, h), dtype=complex)
    for j in range(h):
        S[:, :, j] = H[:, :, j] @ V @ H[:, :, j].conj().T
    return S


def ss_to_tsdata(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    m: int,
    N: int = 1,
    mtrunc: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Simulate an innovations-form state-space process.

    Port of ``core/ss_to_tsdata.m``. Returns ``(X, Z, E)`` — observations,
    state, and innovations.
    """
    if rng is None:
        rng = np.random.default_rng()
    A, C, K, V = (np.asarray(x, dtype=float) for x in (A, C, K, V))
    n = C.shape[0]
    r = A.shape[0]

    rho = specnorm(A)
    if mtrunc is None:
        if rho >= 1.0:
            mtrunc = 0
        else:
            mtrunc = int(np.ceil(-np.log(np.finfo(float).eps) / -np.log(rho)))
    M = m + mtrunc

    L = cholesky(V, lower=True)
    X = np.zeros((n, M, N))
    Z = np.zeros((r, M, N))
    E = np.zeros((n, M, N))
    for trial in range(N):
        E[:, :, trial] = L @ rng.standard_normal((n, M))
        z = np.zeros(r)
        for t in range(M):
            Z[:, t, trial] = z
            X[:, t, trial] = C @ z + E[:, t, trial]
            z = A @ z + K @ E[:, t, trial]
    X = X[:, mtrunc:, :]
    Z = Z[:, mtrunc:, :]
    E = E[:, mtrunc:, :]
    if N == 1:
        return X[:, :, 0], Z[:, :, 0], E[:, :, 0]
    return X, Z, E


def ss_check_fres(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating] | None,
    fres: int,
) -> float:
    """Check the MVGC2 spectral log-determinant integration identity.

    For an innovations-form model, the normalised integral of
    ``log det S(ω)`` must equal ``log det V``.  The returned absolute error is
    the criterion used by MATLAB ``ss2fres.m`` in its adaptive mode.
    """
    A, C, K = (np.asarray(x, dtype=float) for x in (A, C, K))
    n = C.shape[0]
    if V is None:
        L = np.eye(n)
    else:
        try:
            L = cholesky(np.asarray(V, dtype=float), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return float("nan")
    ldv = 2.0 * float(np.sum(np.log(np.diag(L))))
    H = ss2trfun(A, C, K, fres)
    half_logdet = np.empty(fres + 1)
    for k in range(fres + 1):
        HL = H[:, :, k] @ L
        try:
            R = cholesky(HL @ HL.conj().T, lower=False, check_finite=False)
        except np.linalg.LinAlgError:
            return float("nan")
        half_logdet[k] = float(np.real(np.sum(np.log(np.diag(R)))))
    integrated = float(np.sum(half_logdet[:-1] + half_logdet[1:]) / fres)
    return abs(integrated - ldv)


def ss2fres(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating] | None = None,
    K: npt.NDArray[np.floating] | None = None,
    V: npt.NDArray[np.floating] | None = None,
    *,
    fast: bool = False,
    integration: tuple[float, int, int] = (1e-12, 6, 14),
    return_error: bool = False,
) -> int | tuple[int, float]:
    r"""Return the MVGC2-recommended SS frequency resolution.

    The paper requires both stationarity and invertibility,

    .. math::

       \rho = \max\{\max_{\lambda\in\operatorname{eig}(A)}|\lambda|,
       \max_{\lambda\in\operatorname{eig}(A-KC)}|\lambda|\}<1.

    This is an eigenvalue spectral *radius*, not a largest singular value.
    MATLAB's fast mode chooses the next power of two after
    ``log(eps) / log(rho)`` and clamps it to ``2**min_power`` through
    ``2**max_power``.  Its default adaptive mode instead selects the first
    such power whose :func:`ss_check_fres` error is within ``tolerance``.

    Supplying only ``A`` retains the old convenience API and necessarily uses
    fast mode based on ``rho(A)`` alone.  Full MATLAB parity requires ``C``
    and ``K``.
    """
    import warnings

    A = np.asarray(A, dtype=float)
    tol, min_power, max_power = integration
    if tol <= 0 or min_power < 0 or max_power < min_power:
        raise ValueError("integration must be (positive tolerance, min_power, max_power)")
    if (C is None) != (K is None):
        raise ValueError("C and K must either both be supplied or both be omitted")

    if C is None:
        fast = True
    else:
        C = np.asarray(C, dtype=float)
        K = np.asarray(K, dtype=float)

    if fast:
        rho = float(specnorm(A))
        if C is not None and K is not None:
            rho = max(rho, float(specnorm(A - K @ C)))
        if not np.isfinite(rho) or rho >= 1.0:
            warnings.warn(
                "SS model is not stable and invertible; using maximum frequency resolution",
                RuntimeWarning,
                stacklevel=2,
            )
            power = max_power
        elif rho <= 0.0:
            power = min_power
        else:
            decay_samples = np.log(np.finfo(float).eps) / np.log(rho)
            power = int(np.ceil(np.log2(decay_samples)))
            if power > max_power:
                warnings.warn(
                    f"frequency resolution exceeds 2**{max_power}; clamping",
                    RuntimeWarning,
                    stacklevel=2,
                )
                power = max_power
            elif power < min_power:
                power = min_power
        fres = 2**power
        ierr = (
            ss_check_fres(A, C, K, V, fres)
            if return_error and C is not None and K is not None
            else float("nan")
        )
        return (fres, ierr) if return_error else fres

    assert C is not None and K is not None
    ierr = float("nan")
    fres = 2**max_power
    for power in range(min_power, max_power + 1):
        fres = 2**power
        ierr = ss_check_fres(A, C, K, V, fres)
        if np.isfinite(ierr) and ierr <= tol:
            break
    else:
        warnings.warn(
            f"frequency resolution exceeds 2**{max_power}; clamping",
            RuntimeWarning,
            stacklevel=2,
        )
    return (fres, ierr) if return_error else fres


def tsdata_to_ss(
    X: npt.NDArray[np.floating],
    pf: int,
    r: int | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Identify an innovations-form SS model via CCA subspace method.

    Port of MVGC2's ``core/tsdata_to_ss.m`` (Larimore's Canonical Correlations
    Analysis). ``pf`` is the past/future horizon; ``r`` the state dimension
    (auto-selected via Bauer's SVC if ``None``; constrained to ``r >= n``).

    Returns ``(A, C, K, V)``.
    """
    from ._utils import demean

    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape
    if pf >= m // 2:
        raise ValueError("past/future horizon too large for series length")
    X = demean(X)

    p = pf
    f = pf
    M = N * (m - p - f + 1)

    # Build past and future Hankel-stacked matrices
    P = np.zeros((p * n, M))
    F = np.zeros((f * n, M))
    col = 0
    for trial in range(N):
        for t in range(p, m - f + 1):
            for k in range(p):
                P[k * n : (k + 1) * n, col] = X[:, t - 1 - k, trial]
            for k in range(f):
                F[k * n : (k + 1) * n, col] = X[:, t + k, trial]
            col += 1

    # Covariances
    Spp = (P @ P.T) / M
    Sff = (F @ F.T) / M
    Spf = (P @ F.T) / M
    # Canonical correlations via whitened cross-cov
    Lp = cholesky(Spp + 1e-12 * np.eye(p * n), lower=True)
    Lf = cholesky(Sff + 1e-12 * np.eye(f * n), lower=True)
    M_mat = np.linalg.solve(Lf, Spf.T) @ np.linalg.inv(Lp.T)
    U, S, Vt = np.linalg.svd(M_mat, full_matrices=False)

    rmax = n * min(p, f)
    if r is None:
        from .modelorder import bauer_svc

        r, _ = bauer_svc(S, n_obs=n, n_eff=M, min_order=n)
    else:
        if not isinstance(r, (int, np.integer)):
            raise TypeError("r must be an integer or None")
        if not n <= r <= rmax:
            raise ValueError(
                f"state dimension must satisfy {n} <= r <= {rmax}"
            )

    # State estimate
    Z_hat = np.diag(S[:r]) @ Vt[:r, :] @ np.linalg.inv(Lp).T @ P  # shape (r, M)

    # Estimate (A, C, K, V) via linear regressions
    # State equation:   Z(t+1) = A Z(t) + K e(t)
    # Output equation:  X(t)   = C Z(t) + e(t)
    Z0 = Z_hat[:, :-1]
    Z1 = Z_hat[:, 1:]
    A_hat = Z1 @ np.linalg.pinv(Z0)

    # Reconstruct observations corresponding to states (use first block of future)
    X_obs = F[:n, : M - 1]
    C_hat = X_obs @ np.linalg.pinv(Z_hat[:, : M - 1])
    E_hat = X_obs - C_hat @ Z_hat[:, : M - 1]
    V_hat = (E_hat @ E_hat.T) / (M - 1)
    # Kalman gain via regressing state innovations on observation innovations
    W = Z1 - A_hat @ Z0
    # Solve K from W = K @ E_hat (least squares)
    K_hat = W @ np.linalg.pinv(E_hat)
    return A_hat, C_hat, K_hat, V_hat
