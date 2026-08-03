"""Mutual-information measures from covariance / cross-spectral density.

Ports of MVGC2's ``mi/`` directory: every entry is a log-determinant
ratio between blocks of a Hermitian positive-definite matrix.

Conventions
-----------
- Covariance input ``V`` is shape ``(n, n)``.
- CPSD input ``S`` is shape ``(n, n, h)`` (one-sided).
- For groupwise variants, ``groups`` is a list of index arrays.
- Outputs are in nats (multiply by ``1 / np.log(2)`` for bits).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._utils import check_group, logdet, symmetrise

__all__ = [
    "cov_to_mvmi",
    "cov_to_pwmi",
    "cov_to_cmii",
    "cov_to_pwcmi",
    "cov_to_iomi",
    "cov_to_gwcmi",
    "cov_to_gwcmii",
    "cov_to_gwiomi",
    "cpsd_to_smvmi",
    "cpsd_to_spwcmi",
    "cpsd_to_sgwcmi",
    "cpsd_to_sgwgmi",
    "cpsd_to_sgwiomi",
    "cpsd_to_scwiomi",
]


# ---------------------------------------------------------------------------
# Covariance-based MI
# ---------------------------------------------------------------------------


def cov_to_mvmi(V, x, y) -> float:
    """Conditional MI ``I(X; Y | Z)`` where ``Z`` = all-other-variables.

    Port of ``mi/cov_to_mvmi.m``.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    z = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    xz = np.concatenate([x, z])
    yz = np.concatenate([y, z])
    if z.size == 0:
        return logdet(V[np.ix_(xz, xz)]) + logdet(V[np.ix_(yz, yz)]) - logdet(V)
    return (
        logdet(V[np.ix_(xz, xz)]) + logdet(V[np.ix_(yz, yz)]) - logdet(V[np.ix_(z, z)]) - logdet(V)
    )


def cov_to_pwmi(V) -> npt.NDArray[np.floating]:
    """Pairwise (unconditional) MI between every variable pair.

    Port of ``mi/cov_to_pwmi.m``. Returns symmetric matrix with NaN diagonal.
    Uses the MVGC2 convention (no 0.5 prefactor — see :func:`cov_to_mvmi`).
    """
    V = np.asarray(V, dtype=float)
    DV = np.diag(V)
    LDV = np.log(DV)
    denom = DV[:, None] * DV[None, :] - V * V
    np.fill_diagonal(denom, 1.0)  # avoid log(0) on the diagonal
    mutual_info = LDV[:, None] + LDV[None, :] - np.log(denom)
    np.fill_diagonal(mutual_info, np.nan)
    return mutual_info


def cov_to_cmii(V, x=None) -> float:
    """Conditional multi-information among ``x`` conditioned on the rest.

    Port of ``mi/cov_to_cmii.m``.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    if x is None:
        x = np.arange(n)
    x = np.atleast_1d(np.asarray(x, dtype=int))
    ox = np.setdiff1d(np.arange(n), x)
    nx = x.size
    LDVOI = 0.0
    for i in x:
        iox = np.concatenate([[i], ox])
        LDVOI += logdet(V[np.ix_(iox, iox)])
    return LDVOI - logdet(V) - (nx - 1) * logdet(V[np.ix_(ox, ox)])


def cov_to_pwcmi(V) -> npt.NDArray[np.floating]:
    """Pairwise-conditional MI: ``I(X_i; X_j | rest)``.

    Port of ``mi/cov_to_pwcmi.m``. Symmetric, NaN diagonal.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    LDVI = np.zeros(n)
    for i in range(n):
        oi = np.array([k for k in range(n) if k != i])
        LDVI[i] = logdet(V[np.ix_(oi, oi)])
    LDV = logdet(V)
    mutual_info = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            oij = np.array([k for k in range(n) if k != i and k != j])
            LDVIJ = logdet(V[np.ix_(oij, oij)]) if oij.size > 0 else 0.0
            mutual_info[i, j] = LDVI[i] + LDVI[j] - LDVIJ - LDV
    return symmetrise(mutual_info)


def cov_to_iomi(V) -> npt.NDArray[np.floating]:
    """Information-optimised MI per variable: ``I(X_i; rest)``.

    Port of ``mi/cov_to_iomi.m``.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    LDVOI = np.zeros(n)
    for i in range(n):
        oi = np.array([k for k in range(n) if k != i])
        LDVOI[i] = logdet(V[np.ix_(oi, oi)])
    LDV = logdet(V)
    return np.log(np.diag(V)) + LDVOI - LDV


def cov_to_gwcmi(V, groups: list) -> npt.NDArray[np.floating]:
    """Pairwise conditional MI between groups. Port of ``cov_to_gwcmi.m``."""
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, _ = check_group(groups, n)
    LDV = logdet(V)
    LDVG = np.zeros(g)
    for a in range(g):
        goa = np.array([i for i in range(n) if i not in groups[a]])
        LDVG[a] = logdet(V[np.ix_(goa, goa)])
    mutual_info = np.full((g, g), np.nan)
    for a in range(g):
        for b in range(a + 1, g):
            goab = np.array([i for i in range(n) if i not in groups[a] and i not in groups[b]])
            block = logdet(V[np.ix_(goab, goab)]) if goab.size > 0 else 0.0
            mutual_info[a, b] = LDVG[a] + LDVG[b] - block - LDV
    return symmetrise(mutual_info)


def cov_to_gwcmii(V, groups: list) -> npt.NDArray[np.floating]:
    """Conditional multi-information within each group. Port of ``cov_to_gwcmii.m``."""
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, gsiz = check_group(groups, n)
    LDV = logdet(V)
    mutual_info = np.zeros(g)
    for a in range(g):
        goa = np.array([i for i in range(n) if i not in groups[a]])
        block = logdet(V[np.ix_(goa, goa)]) if goa.size > 0 else 0.0
        Ia = -LDV - (gsiz[a] - 1) * block
        for i in np.atleast_1d(np.asarray(groups[a], dtype=int)):
            igoa = np.concatenate([[i], goa])
            Ia += logdet(V[np.ix_(igoa, igoa)])
        mutual_info[a] = Ia
    return mutual_info


def cov_to_gwiomi(V, groups: list, *_args) -> npt.NDArray[np.floating]:
    """Group-level information-optimised MI. Port of ``cov_to_gwiomi.m``."""
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, _ = check_group(groups, n)
    LDV = logdet(V)
    mutual_info = np.zeros(g)
    for a in range(g):
        ga = np.atleast_1d(np.asarray(groups[a], dtype=int))
        goa = np.array([i for i in range(n) if i not in ga])
        mutual_info[a] = logdet(V[np.ix_(ga, ga)]) + logdet(V[np.ix_(goa, goa)]) - LDV
    return mutual_info


# ---------------------------------------------------------------------------
# Spectral MI (CPSD-based)
# ---------------------------------------------------------------------------


def cpsd_to_smvmi(S, x, y) -> npt.NDArray[np.floating]:
    """Spectral conditional MI from CPSD. Port of ``cpsd_to_smvmi.m``."""
    S = np.asarray(S)
    n, _, h = S.shape
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    z = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    xz = np.concatenate([x, z])
    yz = np.concatenate([y, z])
    f = np.empty(h)
    for k in range(h):
        if z.size == 0:
            f[k] = np.real(
                logdet(S[np.ix_(xz, xz)][:, :, k])
                + logdet(S[np.ix_(yz, yz)][:, :, k])
                - logdet(S[:, :, k])
            )
        else:
            f[k] = np.real(
                logdet(S[np.ix_(xz, xz)][:, :, k])
                + logdet(S[np.ix_(yz, yz)][:, :, k])
                - logdet(S[np.ix_(z, z)][:, :, k])
                - logdet(S[:, :, k])
            )
    return f


def cpsd_to_spwcmi(S) -> npt.NDArray[np.floating]:
    """Spectral pairwise-conditional MI. Port of ``cpsd_to_spwcmi.m``."""
    S = np.asarray(S)
    n, _, h = S.shape
    LDS = np.array([np.real(logdet(S[:, :, k])) for k in range(h)])
    LDSI = np.zeros((n, h))
    for i in range(n):
        oi = np.array([j for j in range(n) if j != i])
        for k in range(h):
            LDSI[i, k] = np.real(logdet(S[np.ix_(oi, oi)][:, :, k]))
    C = np.full((n, n, h), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            oij = np.array([z for z in range(n) if z != i and z != j])
            for k in range(h):
                block = np.real(logdet(S[np.ix_(oij, oij)][:, :, k])) if oij.size > 0 else 0.0
                v = LDSI[i, k] + LDSI[j, k] - block - LDS[k]
                C[i, j, k] = v
                C[j, i, k] = v
    return C


def cpsd_to_sgwcmi(S, groups: list) -> npt.NDArray[np.floating]:
    """Spectral pairwise groupwise conditional MI. Port of ``cpsd_to_sgwcmi.m``."""
    S = np.asarray(S)
    n, _, h = S.shape
    g, _ = check_group(groups, n)
    LDS = np.array([np.real(logdet(S[:, :, k])) for k in range(h)])
    LDSG = np.zeros((g, h))
    for a in range(g):
        goa = np.array([i for i in range(n) if i not in groups[a]])
        for k in range(h):
            LDSG[a, k] = np.real(logdet(S[np.ix_(goa, goa)][:, :, k]))
    C = np.full((g, g, h), np.nan)
    for a in range(g):
        for b in range(a + 1, g):
            goab = np.array([i for i in range(n) if i not in groups[a] and i not in groups[b]])
            for k in range(h):
                block = np.real(logdet(S[np.ix_(goab, goab)][:, :, k])) if goab.size > 0 else 0.0
                v = LDSG[a, k] + LDSG[b, k] - block - LDS[k]
                C[a, b, k] = v
                C[b, a, k] = v
    return C


def cpsd_to_sgwgmi(S, groups: list) -> npt.NDArray[np.floating]:
    """Spectral conditional multi-information per group. Port of ``cpsd_to_sgwgmi.m``."""
    S = np.asarray(S)
    n, _, h = S.shape
    g, gsiz = check_group(groups, n)
    LDS = np.array([np.real(logdet(S[:, :, k])) for k in range(h)])
    C = np.zeros((g, h))
    for a in range(g):
        ga = np.atleast_1d(np.asarray(groups[a], dtype=int))
        goa = np.array([i for i in range(n) if i not in ga])
        for k in range(h):
            block = np.real(logdet(S[np.ix_(goa, goa)][:, :, k])) if goa.size > 0 else 0.0
            C[a, k] = -LDS[k] - (gsiz[a] - 1) * block
        for i in ga:
            igoa = np.concatenate([[i], goa])
            for k in range(h):
                C[a, k] += np.real(logdet(S[np.ix_(igoa, igoa)][:, :, k]))
    return C


def cpsd_to_sgwiomi(S, groups: list) -> npt.NDArray[np.floating]:
    """Spectral group-level IO MI. Port of ``cpsd_to_sgwiomi.m``."""
    S = np.asarray(S)
    n, _, h = S.shape
    g, _ = check_group(groups, n)
    LDS = np.array([np.real(logdet(S[:, :, k])) for k in range(h)])
    C = np.zeros((g, h))
    for a in range(g):
        ga = np.atleast_1d(np.asarray(groups[a], dtype=int))
        gb = np.array([i for i in range(n) if i not in ga])
        for k in range(h):
            C[a, k] = (
                np.real(logdet(S[np.ix_(ga, ga)][:, :, k]))
                + np.real(logdet(S[np.ix_(gb, gb)][:, :, k]))
                - LDS[k]
            )
    return C


def cpsd_to_scwiomi(S) -> npt.NDArray[np.floating]:
    """Spectral channel-wise IO MI. Port of ``cpsd_to_scwiomi.m``."""
    S = np.asarray(S)
    n, _, h = S.shape
    LDS = np.array([np.real(logdet(S[:, :, k])) for k in range(h)])
    C = np.zeros((n, h))
    for i in range(n):
        oi = np.array([j for j in range(n) if j != i])
        for k in range(h):
            C[i, k] = (
                np.log(np.real(S[i, i, k])) + np.real(logdet(S[np.ix_(oi, oi)][:, :, k])) - LDS[k]
            )
    return C
