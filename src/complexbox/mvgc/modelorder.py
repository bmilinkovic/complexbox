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

__all__ = ["VARModelOrder", "bauer_svc", "tsdata_to_varmo", "tsdata_to_ssmo"]


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


def bauer_svc(
    canonical_correlations: npt.NDArray[np.floating],
    n_obs: int,
    n_eff: int,
    *,
    min_order: int | None = None,
) -> tuple[int, npt.NDArray[np.floating]]:
    """Select SS state dimension using Bauer's Singular Value Criterion.

    For candidate state dimension ``r``, the criterion is

    ``SVC(r) = rho[r + 1]**2 + 2*n_obs*r*log(n_eff)/n_eff``,

    where ``rho[r + 1]`` is the first canonical correlation omitted by a
    rank-``r`` model. The omitted correlation is defined as zero at the
    maximum candidate order.

    Candidate orders default to ``n_obs, ..., rmax`` because ComplexBox's
    Granger-causality workflow requires state dimension ``r >= n_obs``.

    Parameters
    ----------
    canonical_correlations
        CVA singular values in descending order. Do not normalise them by the
        leading singular value before applying this criterion.
    n_obs
        Number of observed variables.
    n_eff
        Effective number of past/future Hankel columns.
    min_order
        Smallest candidate order. Defaults to ``n_obs`` and may not be less
        than ``n_obs``.

    Returns
    -------
    r_svc
        State dimension minimising the criterion.
    svc
        Criterion values for candidate orders ``min_order, ..., rmax``.
    """
    rho = np.asarray(canonical_correlations, dtype=float)
    if rho.ndim != 1 or rho.size == 0:
        raise ValueError("canonical_correlations must be a non-empty 1-D array")
    if not np.all(np.isfinite(rho)):
        raise ValueError("canonical_correlations must be finite")
    if n_obs < 1:
        raise ValueError("n_obs must be a positive integer")
    if n_eff <= 1:
        raise ValueError("n_eff must be greater than one")

    rmax = int(rho.size)
    if min_order is None:
        min_order = n_obs
    if min_order < n_obs:
        raise ValueError("min_order must be at least n_obs")
    if min_order > rmax:
        raise ValueError(
            f"minimum order {min_order} exceeds maximum identifiable order {rmax}"
        )

    # Canonical correlations are theoretically in [0, 1]. Clipping only
    # removes small floating-point excursions outside that interval.
    rho = np.clip(rho, 0.0, 1.0)
    orders = np.arange(min_order, rmax + 1, dtype=int)

    # For candidate order r, rho[r] is the zero-based representation of the
    # first omitted canonical correlation rho_{r+1}. At rmax, set it to zero.
    omitted = np.zeros(orders.size, dtype=float)
    mask = orders < rmax
    omitted[mask] = rho[orders[mask]]

    penalty = 2.0 * n_obs * orders * np.log(float(n_eff)) / float(n_eff)
    svc = omitted**2 + penalty
    r_svc = int(orders[np.argmin(svc)])
    return r_svc, svc


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


def tsdata_to_ssmo(
    X: npt.NDArray[np.floating],
    pf: int,
) -> tuple[int, npt.NDArray[np.floating]]:
    """Select SS state dimension with Larimore CCA and Bauer's SVC.

    Parameters
    ----------
    X
        Time-series data with shape ``(n, m)`` or ``(n, m, N)``.
    pf
        Common past/future horizon.

    Returns
    -------
    r_svc
        Bauer-SVC state dimension, constrained to ``r_svc >= n``.
    sv
        Canonical correlations normalised by the leading value, preserving
        the function's existing return convention. Bauer's criterion itself
        is evaluated on the unnormalised canonical correlations.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.ndim != 3:
        raise ValueError("X must have shape (n, m) or (n, m, N)")

    X = demean(X)
    n, m, N = X.shape
    if pf < 1:
        raise ValueError("past/future horizon must be positive")
    if pf >= m // 2:
        raise ValueError("past/future horizon too large for series length")

    p = f = pf
    M = N * (m - p - f + 1)
    rmax = n * min(p, f)
    if rmax < n:
        raise ValueError("past/future horizon cannot support state order r >= n")

    # Build past/future Hankel matrices.
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
    sv_raw = np.linalg.svd(M_mat, compute_uv=False)

    r_svc, _ = bauer_svc(sv_raw, n_obs=n, n_eff=M, min_order=n)
    if sv_raw[0] > 0.0:
        sv = sv_raw / sv_raw[0]
    else:
        sv = sv_raw.copy()
    return r_svc, sv

