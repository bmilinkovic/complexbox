"""VAR and state-space model-order selection.

Ports of MVGC2's ``core/tsdata_to_varmo.m`` and ``core/tsdata_to_ssmo.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._utils import demean, logdet
from .stats import infocrit
from .var import tsdata_to_var

__all__ = ["VARModelOrder", "tsdata_to_varmo", "tsdata_to_ssmo"]


@dataclass
class VARModelOrder:
    """Model-order selection results."""

    p_aic: int
    p_bic: int
    p_hqc: int
    aic: npt.NDArray[np.floating]
    bic: npt.NDArray[np.floating]
    hqc: npt.NDArray[np.floating]
    loglik: npt.NDArray[np.floating]


def tsdata_to_varmo(
    X: npt.NDArray[np.floating],
    pmax: int,
    regmode: str = "LWR",
    hurvich_tsai: bool = False,
) -> VARModelOrder:
    """Select VAR model order via AIC, BIC, HQC.

    Port of ``core/tsdata_to_varmo.m``. Fits VAR(p) for p = 1, ..., pmax and
    returns the orders minimising each criterion.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape
    L = np.empty(pmax)
    k = np.empty(pmax)
    for p in range(1, pmax + 1):
        fit = tsdata_to_var(X, p, regmode=regmode)
        # log-likelihood per observation — matches MVGC2's infocrit scaling
        # where the penalty (k/m) log(m) is also per-observation
        L[p - 1] = -0.5 * (n * np.log(2 * np.pi) + logdet(fit.V) + n)
        k[p - 1] = p * n * n  # number of free VAR parameters
    M_eff = N * (m - np.arange(1, pmax + 1))
    aic, bic, hqc = infocrit(L, k, M_eff, hurvich_tsai=hurvich_tsai)
    return VARModelOrder(
        p_aic=int(np.argmin(aic) + 1),
        p_bic=int(np.argmin(bic) + 1),
        p_hqc=int(np.argmin(hqc) + 1),
        aic=aic,
        bic=bic,
        hqc=hqc,
        loglik=L,
    )


def tsdata_to_ssmo(X: npt.NDArray[np.floating], pf: int) -> tuple[int, npt.NDArray[np.floating]]:
    """Bauer's Singular Value Criterion for state dimension.

    Port of ``core/tsdata_to_ssmo.m`` (simplified).

    Returns
    -------
    r_svc : int — chosen state dimension
    sv : 1-D array of normalised singular values
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    X = demean(X)
    n, m, N = X.shape
    if pf >= m // 2:
        raise ValueError("past/future horizon too large for series length")

    # Build past/future Hankel matrices
    p = f = pf
    M = N * (m - p - f + 1)
    P = np.zeros((p * n, M))
    F = np.zeros((f * n, M))
    col = 0
    for trial in range(N):
        for t in range(p, m - f + 1):
            for kk in range(p):
                P[kk * n : (kk + 1) * n, col] = X[:, t - 1 - kk, trial]
            for kk in range(f):
                F[kk * n : (kk + 1) * n, col] = X[:, t + kk, trial]
            col += 1
    Spp = (P @ P.T) / M
    Sff = (F @ F.T) / M
    Spf = (P @ F.T) / M
    from scipy.linalg import cholesky

    Lp = cholesky(Spp + 1e-12 * np.eye(p * n), lower=True)
    Lf = cholesky(Sff + 1e-12 * np.eye(f * n), lower=True)
    M_mat = np.linalg.solve(Lf, Spf.T) @ np.linalg.inv(Lp.T)
    sv = np.linalg.svd(M_mat, compute_uv=False)
    sv = sv / sv[0]
    # Bauer's SVC: pick r so that sv[r] < threshold AND sv[r-1] >= threshold.
    r_svc = int(np.sum(sv > 1e-2))
    r_svc = max(r_svc, n)
    return r_svc, sv
