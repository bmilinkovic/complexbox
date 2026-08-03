"""VAR model estimation, simulation, and conversions.

Ports of MVGC2's ``core/tsdata_to_var.m``, ``core/var_to_tsdata.m``,
``core/var_to_ss.m``, ``core/var_to_autocov.m``, ``core/var_to_cpsd.m`` and
``core/autocov_to_var.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

from ._lyap import dlyap
from ._utils import demean, specnorm

__all__ = [
    "VARFit",
    "tsdata_to_var",
    "var_to_tsdata",
    "var_to_ss",
    "var_to_autocov",
    "autocov_to_var",
    "var_to_cpsd",
    "var2trfun",
    "var2itrfun",
    "var_check_fres",
    "var2fres",
]


@dataclass
class VARFit:
    """Result of fitting a VAR(p) model.

    Attributes
    ----------
    A : (n, n, p) ndarray
        Coefficient array; ``A[:, :, k]`` is the lag-(k+1) block.
    V : (n, n) ndarray
        Residuals covariance matrix (unbiased estimator, divisor ``M - 1``
        where ``M = N*(m - p)``).
    E : (n, m - p, N) ndarray
        Residuals time series.
    """

    A: npt.NDArray[np.floating]
    V: npt.NDArray[np.floating]
    E: npt.NDArray[np.floating]


def tsdata_to_var(X: npt.NDArray[np.floating], p: int, regmode: str = "LWR") -> VARFit:
    """Fit a VAR(p) to multi-trial time series ``X``.

    Direct port of ``tsdata_to_var.m``. Two regression modes:

    - ``'LWR'``  — Morf's recursive Lattice-Whitening-Regression algorithm,
      which is guaranteed to yield a stable estimate.
    - ``'OLS'``  — ordinary least squares via QR decomposition; valid for
      unstable processes (e.g., unit roots).

    Parameters
    ----------
    X : array of shape (n, m) or (n, m, N)
    p : int — model order (number of lags)
    regmode : 'LWR' or 'OLS'

    Returns
    -------
    fit : VARFit with attributes ``A``, ``V``, ``E``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.ndim != 3:
        raise ValueError("X must be 2-D or 3-D")
    n, m, N = X.shape
    if not (0 <= p < m):
        raise ValueError(f"bad model order (p={p}, m={m})")

    M = N * (m - p)

    if p == 0:
        A = np.zeros((n, n, 0))
        X0 = demean(X).reshape(n, m * N)
        V = (X0 @ X0.T) / (M - 1)
        return VARFit(A=A, V=V, E=X.copy())

    X = demean(X)

    if regmode.upper() == "OLS":
        obs = np.arange(p, m)  # 0-indexed (MATLAB: p+1:m)
        # X0 = X[:, obs, :] flattened to (n, M) — the "current" observation matrix.
        X0 = X[:, obs, :].reshape(n, M)
        # Stack lagged observations so the flattened rows are organised as
        # [X(t-1); X(t-2); ...; X(t-p)] — i.e., lag varies slowest, variable fastest.
        XL = np.zeros((p, n, M))
        for k in range(p):
            XL[k, :, :] = X[:, obs - (k + 1), :].reshape(n, M)
        XL = XL.reshape(p * n, M)
        # OLS via least squares: A @ XL = X0  →  solve XL.T @ A.T = X0.T
        sol, *_ = np.linalg.lstsq(XL.T, X0.T, rcond=None)
        A_flat = sol.T  # shape (n, p*n) with column blocks [A_1 | A_2 | ... | A_p]
        E = X0 - A_flat @ XL
        A = np.empty((n, n, p))
        for k in range(p):
            A[:, :, k] = A_flat[:, k * n : (k + 1) * n]
    elif regmode.upper() == "LWR":
        identity = np.eye(n)
        p1 = p + 1
        p1n = p1 * n

        # Stack lagged observations in lag-major order to mirror MATLAB's
        # column-major reshape: XX[k, i, t, r] = X[i, t - k, r] when t - k >= 0.
        XX = np.zeros((p1, n, m + p, N))
        for k in range(p1):
            XX[k, :, k : k + m, :] = X

        EE = X.reshape(n, N * m)
        try:
            C = cholesky(EE @ EE.T, lower=True)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Covariance matrix not positive-definite — likely colinearity in X"
            ) from exc
        IC = np.linalg.inv(C)

        k = 1
        kn = k * n
        Meff = N * (m - k)
        kk_end = k
        kf_end = kn
        kb_start = p1n - kn

        AF = np.zeros((n, p1n))
        AF[:, :kf_end] = IC
        AB = np.zeros((n, p1n))
        AB[:, kb_start:] = IC

        while k <= p:
            # forward prediction errors — slice (kk_end, n, m-k, N) → (kn, Meff)
            block_f = XX[:kk_end, :, k:m, :]
            EF = AF[:, :kf_end] @ block_f.reshape(kn, Meff)
            # backward prediction errors
            block_b = XX[:kk_end, :, k - 1 : m - 1, :]
            EB = AB[:, kb_start:] @ block_b.reshape(kn, Meff)

            # normalised reflection coefficients
            cEF = cholesky(EF @ EF.T, lower=True)
            cEB = cholesky(EB @ EB.T, lower=True)
            R = np.linalg.solve(cEF, EF) @ np.linalg.solve(cEB, EB).T

            k += 1
            kn = k * n
            Meff = N * (m - k) if k <= m else 0
            kk_end = k
            kf_end = kn
            kb_start = p1n - kn

            AFPREV = AF[:, :kf_end].copy()
            ABPREV = AB[:, kb_start:].copy()

            cF = cholesky(identity - R @ R.T, lower=True)
            cB = cholesky(identity - R.T @ R, lower=True)
            AF[:, :kf_end] = np.linalg.solve(cF, AFPREV - R @ ABPREV)
            AB[:, kb_start:] = np.linalg.solve(cB, ABPREV - R.T @ AFPREV)

        A0 = AF[:, :n]
        flat = -np.linalg.solve(A0, AF[:, n:p1n])  # (n, p*n), columns [A_1|...|A_p]
        A = np.empty((n, n, p))
        for k in range(p):
            A[:, :, k] = flat[:, k * n : (k + 1) * n]
        E = np.linalg.solve(A0, EF)
    else:
        raise ValueError(f"unknown regression mode {regmode!r}")

    V = (E @ E.T) / (M - 1)
    E = E.reshape(n, m - p, N)
    return VARFit(A=A, V=V, E=E)


def var_to_tsdata(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    m: int,
    N: int = 1,
    mtrunc: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Simulate a stationary VAR(p) process.

    Port of ``core/var_to_tsdata.m``.

    Parameters
    ----------
    A : (n, n, p) coefficient array
    V : (n, n) residuals covariance
    m : number of time points per trial
    N : number of trials
    mtrunc : transient burn-in length (default: ``ceil(-log(eps) / -log(rho))``)
    rng : numpy Generator (default: ``np.random.default_rng()``)

    Returns
    -------
    X : (n, m, N) simulated time series
    E : (n, m, N) innovations
    """
    if rng is None:
        rng = np.random.default_rng()
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    if A.ndim != 3:
        raise ValueError("A must be 3-D (n, n, p)")
    n, _, p = A.shape

    rho = specnorm(A)
    if mtrunc is None:
        if rho >= 1.0:
            mtrunc = 0
        else:
            mtrunc = int(np.ceil(-np.log(np.finfo(float).eps) / -np.log(rho)))
    M = m + mtrunc

    L = cholesky(V, lower=True)
    X = np.zeros((n, M, N))
    E = np.zeros((n, M, N))
    for r in range(N):
        E[:, :, r] = L @ rng.standard_normal((n, M))
        for t in range(M):
            x = E[:, t, r].copy()
            for k in range(p):
                if t - k - 1 >= 0:
                    x = x + A[:, :, k] @ X[:, t - k - 1, r]
            X[:, t, r] = x
    X = X[:, mtrunc:, :]
    E = E[:, mtrunc:, :]
    if N == 1:
        return X[:, :, 0], E[:, :, 0]
    return X, E


def var_to_ss(
    A: npt.NDArray[np.floating], V: npt.NDArray[np.floating] | None = None
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
]:
    """Convert VAR coefficients to innovations-form state space.

    Port of ``core/var_to_ss.m``. Returns ``(A_ss, C, K, V)`` where::

        A_ss = [[A1 A2 ... Ap], [I 0 ... 0], [0 I ... 0], ..., [0 ... I 0]]
        C    = [I, 0, ..., 0]   (n, p*n)
        K    = [I; 0; ...; 0]   (p*n, n)
    """
    A = np.asarray(A, dtype=float)
    n, n1, p = A.shape
    if n1 != n:
        raise ValueError("VAR coefficient matrix has bad shape")
    pn1 = (p - 1) * n
    # C = [A(:,:,1) A(:,:,2) ... A(:,:,p)]
    C = np.concatenate([A[:, :, k] for k in range(p)], axis=1)
    A_ss = np.zeros((p * n, p * n))
    A_ss[:n, :] = C
    if pn1 > 0:
        A_ss[n:, :pn1] = np.eye(pn1)
    K = np.zeros((p * n, n))
    K[:n, :] = np.eye(n)
    return A_ss, C, K, (V if V is None else np.asarray(V, dtype=float))


def var_to_autocov(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    qmax: int,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.floating], int]:
    """Autocovariance sequence of a VAR process.

    Port of ``core/var_to_autocov.m``.

    Returns ``(G, q)`` where ``G[:, :, k]`` is the lag-``k`` autocovariance,
    for ``k = 0, ..., q``. Pass ``qmax > 0`` for adaptive convergence-based
    truncation or ``qmax < 0`` to request exactly ``|qmax|`` lags.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, p = A.shape
    pn = p * n
    pn1 = (p - 1) * n

    A_flat = np.concatenate([A[:, :, k] for k in range(p)], axis=1)  # (n, pn)
    A1 = np.zeros((pn, pn))
    A1[:n, :] = A_flat
    if pn1 > 0:
        A1[n:, :pn1] = np.eye(pn1)
    V1 = np.zeros((pn, pn))
    V1[:n, :n] = V

    G1 = dlyap(A1, V1)  # solve X = A1 X A1' + V1
    G = np.empty((n, n, p))
    for k in range(p):
        G[:, :, k] = G1[:n, k * n : (k + 1) * n]

    alags = qmax < 0
    q = -qmax if alags else qmax
    q1 = q + 1

    if q < p:
        return G[:, :, : q + 1].copy(), q

    # Initialise reverse covariance sequence
    R = np.zeros((pn, n))
    for k in range(p):
        R[k * n : (k + 1) * n, :] = G[:, :, p - 1 - k]

    if alags:
        G_full = np.zeros((n, n, q1))
        G_full[:, :, :p] = G
        for k in range(p, q1):
            G_full[:, :, k] = A_flat @ R
            R = np.concatenate([G_full[:, :, k], R[: (p - 1) * n, :]], axis=0)
        return G_full, q
    else:
        if tol is None:
            tol = np.finfo(float).eps * float(np.max(np.abs(G[:, :, 0])))
        G_list = [G[:, :, k].copy() for k in range(p)]
        k = p - 1  # last index in G_list, MATLAB k=p before loop
        # MATLAB increments: while maxabs(G(:,:,k)) > tol; bump k++
        while np.max(np.abs(G_list[-1])) > tol:
            if k + 1 > qmax:
                import warnings

                warnings.warn(
                    "var_to_autocov: covariance sequence failed to converge",
                    stacklevel=2,
                )
                return np.stack(G_list, axis=2), k
            k += 1
            new = A_flat @ R
            G_list.append(new)
            R = np.concatenate([new, R[: (p - 1) * n, :]], axis=0)
        return np.stack(G_list, axis=2), k


def autocov_to_var(
    G: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Whittle's recursive LWR algorithm: autocov sequence → VAR + residuals cov.

    Port of ``core/autocov_to_var.m``.
    """
    G = np.asarray(G, dtype=float)
    n, _, q1 = G.shape
    q = q1 - 1
    qn = q * n

    G0 = G[:, :, 0]
    # GF: forward autocov sequence (qn, n) = [G(1).T; G(2).T; ...; G(q).T]
    # MATLAB: reshape(G(:,:,2:end), n, qn)'  — column-major reshape then transpose.
    GF = np.concatenate([G[:, :, k + 1] for k in range(q)], axis=1).T  # (qn, n)
    # GB: backward, MATLAB reshape(permute(flipdim(G(:,:,2:end),3),[1,3,2]), qn, n)
    # Trace through column-major indexing: GB[0:n, :] = G(q), GB[n:2n, :] = G(q-1), etc.
    # (No transpose, unlike GF — the permute-flip-reshape produces the *non-transposed* form.)
    GB = np.concatenate([G[:, :, q - k] for k in range(q)], axis=0)  # (qn, n)

    AF = np.zeros((n, qn))
    AB = np.zeros((n, qn))

    k = 1
    r = q - k
    kf_end = k * n
    kb_start = r * n

    AF[:, :kf_end] = np.linalg.solve(G0.T, GB[kb_start:, :].T).T
    AB[:, kb_start:] = np.linalg.solve(G0.T, GF[:kf_end, :].T).T

    for k in range(2, q + 1):
        # AAF = (GB((r-1)*n+1:r*n,:) - AF(:,kf)*GB(kb,:)) / (G0 - AB(:,kb)*GB(kb,:))
        block_f = GB[(r - 1) * n : r * n, :]  # (n, n)
        denom_f = G0 - AB[:, kb_start:] @ GB[kb_start:, :]
        AAF = np.linalg.solve(denom_f.T, (block_f - AF[:, :kf_end] @ GB[kb_start:, :]).T).T

        block_b = GF[(k - 1) * n : k * n, :]
        denom_b = G0 - AF[:, :kf_end] @ GF[:kf_end, :]
        AAB = np.linalg.solve(denom_b.T, (block_b - AB[:, kb_start:] @ GF[:kf_end, :]).T).T

        AFPREV = AF[:, :kf_end].copy()
        ABPREV = AB[:, kb_start:].copy()

        r = q - k
        kf_end = k * n
        kb_start = r * n
        AF[:, :kf_end] = np.concatenate([AFPREV - AAF @ ABPREV, AAF], axis=1)
        AB[:, kb_start:] = np.concatenate([AAB, ABPREV - AAB @ AFPREV], axis=1)

    V = G0 - AF @ GF
    A = np.empty((n, n, q))
    for k in range(q):
        A[:, :, k] = AF[:, k * n : (k + 1) * n]
    return A, V


def var2itrfun(
    A: npt.NDArray[np.floating],
    fres: int,
    *,
    backend: str = "numpy",
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Inverse transfer function ``H^{-1}(ω) = I - sum_k A_k e^{-ikω}``.

    Port of ``utils/var2itrfun.m``. Returns array of shape ``(n, n, fres+1)``.
    """
    if backend == "torch":
        from ._torch import var2itrfun as var2itrfun_torch

        return var2itrfun_torch(A, fres, device=device, dtype=dtype, batch_size=batch_size)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    A = np.asarray(A, dtype=float)
    n, _, p = A.shape
    # Stack [I, -A_1, -A_2, ..., -A_p] along the lag axis, then FFT
    block = np.zeros((n, n, 2 * fres), dtype=float)
    block[:, :, 0] = np.eye(n)
    for k in range(p):
        block[:, :, k + 1] = -A[:, :, k]
    J = np.fft.fft(block, axis=-1)
    return J[:, :, : fres + 1]


def var2trfun(
    A: npt.NDArray[np.floating],
    fres: int,
    *,
    backend: str = "numpy",
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Transfer function H(e^{iω}) = (I - sum_k A_k e^{-ikω})^{-1}.

    Port of MVGC2's ``var2trfun.m``. Frequencies are ``ω = πj/fres`` for
    ``j = 0, ..., fres``, so the returned array has ``fres + 1`` slices.

    Returns
    -------
    H : (n, n, fres + 1) complex array
    """
    if backend == "torch":
        from ._torch import var2trfun as var2trfun_torch

        return var2trfun_torch(A, fres, device=device, dtype=dtype, batch_size=batch_size)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    A = np.asarray(A, dtype=float)
    n, _, p = A.shape
    h = fres + 1
    omega = np.linspace(0.0, np.pi, h)
    H = np.empty((n, n, h), dtype=complex)
    identity = np.eye(n)
    z = np.exp(-1j * omega)
    for j in range(h):
        Apoly = identity.copy().astype(complex)
        zk = 1.0
        for k in range(p):
            zk = zk * z[j]
            Apoly = Apoly - A[:, :, k] * zk
        H[:, :, j] = np.linalg.inv(Apoly)
    return H


def var_to_cpsd(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    fres: int,
    *,
    backend: str = "numpy",
    device: str | object = "cpu",
    dtype: object = np.float64,
    batch_size: int | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Cross-power spectral density of a VAR process.

    S(ω) = H(ω) V H(ω)*. Port of ``core/var_to_cpsd.m``.
    """
    if backend == "torch":
        from ._torch import var_to_cpsd as var_to_cpsd_torch

        return var_to_cpsd_torch(A, V, fres, device=device, dtype=dtype, batch_size=batch_size)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    H = var2trfun(A, fres)
    n, _, h = H.shape
    S = np.empty((n, n, h), dtype=complex)
    for j in range(h):
        S[:, :, j] = H[:, :, j] @ V @ H[:, :, j].conj().T
    return S


def var_check_fres(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating] | None,
    fres: int,
) -> float:
    """Check the MVGC2 VAR spectral log-determinant integration identity."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if V is None:
        L = np.eye(n)
    else:
        try:
            L = cholesky(np.asarray(V, dtype=float), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return float("nan")
    ldv = 2.0 * float(np.sum(np.log(np.diag(L))))
    H = var2trfun(A, fres)
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


def var2fres(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating] | None = None,
    *,
    fast: bool = False,
    integration: tuple[float, int, int] = (1e-12, 6, 14),
    return_error: bool = False,
) -> int | tuple[int, float]:
    r"""Return the MVGC2-recommended VAR frequency resolution.

    Stability uses the maximum eigenvalue modulus of the VAR companion matrix,
    ``rho = specnorm(A) < 1``.  Fast mode selects the next power of two after
    ``log(eps) / log(rho)``; adaptive mode selects the first configured power
    satisfying :func:`var_check_fres`.
    """
    import warnings

    A = np.asarray(A, dtype=float)
    tol, min_power, max_power = integration
    if tol <= 0 or min_power < 0 or max_power < min_power:
        raise ValueError("integration must be (positive tolerance, min_power, max_power)")

    if fast:
        rho = float(specnorm(A))
        if not np.isfinite(rho) or rho >= 1.0:
            warnings.warn(
                "VAR model is not stable; using maximum frequency resolution",
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
        ierr = var_check_fres(A, V, fres) if return_error else float("nan")
        return (fres, ierr) if return_error else fres

    ierr = float("nan")
    fres = 2**max_power
    for power in range(min_power, max_power + 1):
        fres = 2**power
        ierr = var_check_fres(A, V, fres)
        if np.isfinite(ierr) and ierr <= tol:
            break
    else:
        warnings.warn(
            f"frequency resolution exceeds 2**{max_power}; clamping",
            RuntimeWarning,
            stacklevel=2,
        )
    return (fres, ierr) if return_error else fres
