"""Wilson's spectral factorisation algorithm.

Port of MVGC2's ``utils/cpsd_specfac.m``: factorises a Hermitian, positive
cross-power spectral density ``S(ω)`` into ``H(ω) V H(ω)*`` with ``H``
minimum-phase (analytic in the upper half-plane). Implemented as the
fixed-point iteration of Wilson (1972), itself a multivariate generalisation
of the scalar spectral factorisation.

The MVGC2 source attributes its implementation to M. Dhamala and G.
Rangarajan (2006), with modifications by S. K. Mody (2016), revisions by M.
Dhamala (2017), and adaptation for MVGC2 by L. Barnett (2019).

References
----------
Wilson, G. T. (1972). "The factorization of matricial spectral densities",
SIAM J. Appl. Math. 23(4), 420-426.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

__all__ = ["cpsd_specfac"]


def cpsd_specfac(
    S: npt.NDArray[np.complexfloating],
    tol: float | None = None,
    maxi: int | None = None,
) -> tuple[
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating],
    bool,
    float,
    int,
]:
    """Factorise a one-sided CPSD ``S`` as ``H V H^*``.

    Parameters
    ----------
    S : (n, n, h) complex array (one-sided spectrum)
    tol : convergence tolerance (default 1e-7)
    maxi : maximum iterations (default ``min(100, round(1/sqrt(tol)))``)

    Returns
    -------
    H : (n, n, h) minimum-phase transfer function
    V : (n, n) innovations covariance
    converged : bool
    relerr : final relative error
    niter : iterations actually performed
    """
    if tol is None:
        tol = 1e-7
    if maxi is None:
        maxi = min(100, round(1.0 / np.sqrt(tol)))
    S = np.asarray(S)
    n, _, h = S.shape
    # Build two-sided spectrum (negative frequencies via conjugation)
    SS = np.concatenate([S, np.conj(S[:, :, h - 2 : 0 : -1])], axis=2)
    m = SS.shape[2]
    identity = np.eye(n)

    L = _l_initial(SS, n).astype(complex)
    K = np.repeat(L[:, :, None], h, axis=2).astype(complex)
    K = np.concatenate([K, np.conj(K[:, :, h - 2 : 0 : -1])], axis=2)

    # Cholesky of SS at each frequency (upper triangular)
    U = np.empty_like(SS)
    for j in range(m):
        SSj = 0.5 * (SS[:, :, j] + SS[:, :, j].conj().T)
        try:
            U[:, :, j] = cholesky(SSj, lower=False)
        except np.linalg.LinAlgError:
            U[:, :, j] = cholesky(SSj + 1e-12 * identity, lower=False)

    niter = 0
    converged = False
    relerr = np.inf
    g = np.empty_like(SS)
    while niter < maxi and not converged:
        for k in range(m):
            W = np.linalg.solve(K[:, :, k], U[:, :, k].conj().T)
            g[:, :, k] = W @ W.conj().T + identity
        gp, gp0 = _plus_operator(g, n)
        T = -np.tril(gp0, -1)
        T = T - T.conj().T

        K_prev = K.copy()
        for k in range(m):
            K[:, :, k] = K[:, :, k] @ (gp[:, :, k] + T)
        L_prev = L.copy()
        L = L @ (gp0 + T)

        relerr_K = _relerr(K, K_prev)
        relerr_L = _relerr(L, L_prev)
        relerr = max(relerr_K, relerr_L)
        if relerr < tol:
            converged = True
        niter += 1

    H = np.empty((n, n, h), dtype=complex)
    Linv = np.linalg.inv(L)
    for k in range(h):
        H[:, :, k] = K[:, :, k] @ Linv
    V = np.real(L @ L.conj().T)
    return H, V, converged, float(relerr), int(niter)


def _l_initial(SS: npt.NDArray, n: int) -> npt.NDArray:
    """Initial Cholesky factor from the zero-lag autocovariance."""
    m = SS.shape[2]
    flat = SS.reshape(n * n, m)
    gamma = np.fft.ifft(flat, axis=-1)
    gamma0 = gamma[:, 0].reshape(n, n)
    gamma0 = np.real(0.5 * (gamma0 + gamma0.conj().T))
    return cholesky(gamma0, lower=False)


def _plus_operator(g: npt.NDArray, n: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Project ``g`` onto the analytic (causal, "plus") part.

    Returns ``(g_plus_freq, g_plus_zero_lag)``.
    """
    m = g.shape[2]
    h = (m + 1) // 2
    flat = g.reshape(n * n, m)
    gamma = np.real(np.fft.ifft(flat.T, axis=0))  # shape (m, n*n)
    gamma = gamma.T.reshape(n, n, m)
    gamma[:, :, 0] = 0.5 * gamma[:, :, 0]
    gp0 = gamma[:, :, 0].copy()
    gamma[:, :, h:] = 0
    flat = gamma.reshape(n * n, m)
    gp = np.fft.fft(flat.T, axis=0).T.reshape(n, n, m)
    return gp, gp0


def _relerr(A: npt.NDArray, B: npt.NDArray) -> float:
    nA = np.linalg.norm(A.reshape(-1))
    nB = np.linalg.norm((A - B).reshape(-1))
    return float(nB / max(nA, 1e-30))
