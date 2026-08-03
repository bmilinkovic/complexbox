"""Conditional, groupwise, and information-optimised Granger causality.

Ports of MVGC2's ``var_to_cggc``, ``var_to_gwcgc``, ``var_to_gwcggc``,
``var_to_gwiogc``, ``var_to_iogc``, their ``ss_to_*`` counterparts, and the
spectral ``var_to_s*`` / ``ss_to_s*`` variants.

Conventions
-----------
- A *group* is a Python list of integer index arrays, e.g. ``[[0, 1], [2, 3], [4]]``.
- ``F[a, b]`` for groupwise GC matrices is causality **from group ``b`` to
  group ``a``** (mirrors MVGC2's order).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

from ._lyap import mdare
from ._utils import check_group, logdet, parcov
from .gc import _vardare
from .ss import ss2itrfun, ss2trfun
from .var import var2itrfun, var2trfun

__all__ = [
    # VAR-based, time-domain
    "var_to_iogc",
    "var_to_cggc",
    "var_to_gwcgc",
    "var_to_gwcggc",
    "var_to_gwiogc",
    # SS-based, time-domain
    "ss_to_iogc",
    "ss_to_cggc",
    "ss_to_gwcgc",
    "ss_to_gwcggc",
    "ss_to_gwiogc",
    # VAR-based, spectral
    "var_to_siogc",
    "var_to_sgwcgc",
    "var_to_sgwiogc",
    # SS-based, spectral
    "ss_to_siogc",
    "ss_to_sgwcgc",
    "ss_to_sgwiogc",
]


# ---------------------------------------------------------------------------
# Time-domain — VAR variants
# ---------------------------------------------------------------------------


def var_to_iogc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    inout: str = "in",
) -> npt.NDArray[np.floating]:
    """Information-optimised GC for every variable.

    Port of ``gc/var/var_to_iogc.m``. With ``inout='in'``, returns the GC
    from all-other-variables → ``i`` for each ``i``. With ``inout='out'``,
    returns the GC from ``i`` → all-others.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    F = np.full(n, np.nan)
    for i in range(n):
        if gcin:
            y = np.array([j for j in range(n) if j != i])
            r = np.array([i])
        else:
            y = np.array([i])
            r = np.array([j for j in range(n) if j != i])
        _, VR, rep = _vardare(A, V, y, r)
        if rep < 0:
            continue
        F[i] = (
            logdet(VR) - logdet(V[np.ix_(r, r)])
            if r.size > 1
            else (np.log(VR[0, 0]) - np.log(V[r[0], r[0]]))
        )
    return F


def var_to_cggc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    x: npt.NDArray | list[int] | None = None,
) -> float:
    """Conditional groupwise GC: a *single* scalar measuring how much of the
    joint dependence inside group ``x`` is explained by external sources.

    Port of ``gc/var/var_to_cggc.m``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    if x is None:
        x = np.arange(n)
    x = np.atleast_1d(np.asarray(x, dtype=int))
    z = np.array([i for i in range(n) if i not in x])
    DV = np.diag(V)
    nx = x.size
    VRx = np.empty(nx)
    for i in range(nx):
        y = np.array([x[j] for j in range(nx) if j != i])
        r = np.concatenate([[x[i]], z])
        _, VR, rep = _vardare(A, V, y, r)
        if rep < 0:
            return float("nan")
        VRx[i] = VR[0, 0]
    return float(np.sum(np.log(VRx)) - np.sum(np.log(DV[x])))


def var_to_gwcgc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    groups: list,
) -> npt.NDArray[np.floating]:
    """Pairwise-conditional GC between groups.

    Port of ``gc/var/var_to_gwcgc.m``. Returns a ``g × g`` matrix where
    ``F[a, b]`` is GC from group ``b`` to group ``a``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, _ = check_group(groups, n)
    F = np.full((g, g), np.nan)
    for b in range(g):
        y = np.atleast_1d(np.asarray(groups[b], dtype=int))
        r = np.array([i for i in range(n) if i not in y])
        _, VR, rep = _vardare(A, V, y, r)
        if rep < 0:
            continue
        for a in range(g):
            if a == b:
                continue
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            # indices of x within r
            xr = np.array([np.where(r == xi)[0][0] for xi in x])
            F[a, b] = logdet(VR[np.ix_(xr, xr)]) - logdet(V[np.ix_(x, x)])
    return F


def var_to_gwcggc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    groups: list,
) -> npt.NDArray[np.floating]:
    """Conditional groupwise GC per group (scalar per group).

    Port of ``gc/var/var_to_gwcggc.m``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, _ = check_group(groups, n)
    F = np.full(g, np.nan)
    DV = np.diag(V)
    for a in range(g):
        x = np.atleast_1d(np.asarray(groups[a], dtype=int))
        z = np.array([i for i in range(n) if i not in x])
        nx = x.size
        VRx = np.empty(nx)
        ok = True
        for i in range(nx):
            y = np.array([x[j] for j in range(nx) if j != i])
            r = np.concatenate([[x[i]], z])
            _, VR, rep = _vardare(A, V, y, r)
            if rep < 0:
                ok = False
                break
            VRx[i] = VR[0, 0]
        if ok:
            F[a] = float(np.sum(np.log(VRx)) - np.sum(np.log(DV[x])))
    return F


def var_to_gwiogc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    groups: list,
    inout: str = "in",
) -> npt.NDArray[np.floating]:
    """Information-optimised GC at the group level.

    Port of ``gc/var/var_to_gwiogc.m``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, _ = check_group(groups, n)
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    F = np.full(g, np.nan)
    for a in range(g):
        if gcin:
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            y = np.array([i for i in range(n) if i not in x])
        else:
            y = np.atleast_1d(np.asarray(groups[a], dtype=int))
            x = np.array([i for i in range(n) if i not in y])
        _, VR, rep = _vardare(A, V, y, x)
        if rep < 0:
            continue
        F[a] = logdet(VR) - logdet(V[np.ix_(x, x)])
    return F


# ---------------------------------------------------------------------------
# Time-domain — SS variants
# ---------------------------------------------------------------------------


def _ss_reduced_dare(A, C, K, V, y, r):
    """Solve the DARE for an innovations-form SS model reduced to outputs ``r``
    after removing source-output indices ``y``."""
    KVK = K @ V @ K.T
    R = V[np.ix_(r, r)]
    S = K @ V[:, r]
    Cr = C[r, :]
    return mdare(A, Cr, KVK, R, S)


def ss_to_iogc(A, C, K, V, inout: str = "in") -> npt.NDArray[np.floating]:
    """Information-optimised GC from an SS model. Port of ``ss_to_iogc.m``."""
    A, C, K, V = (np.asarray(x, dtype=float) for x in (A, C, K, V))
    n = C.shape[0]
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    F = np.full(n, np.nan)
    for i in range(n):
        if gcin:
            r = np.array([i])
        else:
            r = np.array([j for j in range(n) if j != i])
        _, VR, rep, _ = _ss_reduced_dare(A, C, K, V, None, r)
        if rep < 0:
            continue
        if r.size == 1:
            F[i] = float(np.log(VR[0, 0]) - np.log(V[r[0], r[0]]))
        else:
            F[i] = logdet(VR) - logdet(V[np.ix_(r, r)])
    return F


def ss_to_cggc(A, C, K, V, x: npt.NDArray | list[int] | None = None) -> float:
    """Conditional groupwise GC, scalar form. Port of ``ss_to_cggc.m``."""
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    if x is None:
        x = np.arange(n)
    x = np.atleast_1d(np.asarray(x, dtype=int))
    z = np.array([i for i in range(n) if i not in x])
    F_total = 0.0
    for i in range(x.size):
        r = np.concatenate([[x[i]], z])
        _, VR, rep, _ = _ss_reduced_dare(A, C, K, V, None, r)
        if rep < 0:
            return float("nan")
        F_total += float(np.log(VR[0, 0]) - np.log(V[x[i], x[i]]))
    return F_total


def ss_to_gwcgc(A, C, K, V, groups: list) -> npt.NDArray[np.floating]:
    """Pairwise-conditional groupwise GC from SS. Port of ``ss_to_gwcgc.m``."""
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    g, _ = check_group(groups, n)
    F = np.full((g, g), np.nan)
    for b in range(g):
        y = np.atleast_1d(np.asarray(groups[b], dtype=int))
        r = np.array([i for i in range(n) if i not in y])
        _, VR, rep, _ = _ss_reduced_dare(A, C, K, V, y, r)
        if rep < 0:
            continue
        for a in range(g):
            if a == b:
                continue
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            xr = np.array([np.where(r == xi)[0][0] for xi in x])
            F[a, b] = logdet(VR[np.ix_(xr, xr)]) - logdet(V[np.ix_(x, x)])
    return F


def ss_to_gwcggc(A, C, K, V, groups: list) -> npt.NDArray[np.floating]:
    """Per-group conditional groupwise GC.

    State-space analogue of MVGC2's ``var_to_gwcggc.m``, using the reduced
    DARE construction from ``ss_to_cggc.m``. MVGC2 has no function named
    ``ss_to_gwcggc.m``.
    """
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    g, _ = check_group(groups, n)
    F = np.full(g, np.nan)
    DV = np.diag(V)
    for a in range(g):
        x = np.atleast_1d(np.asarray(groups[a], dtype=int))
        z = np.array([i for i in range(n) if i not in x])
        total = 0.0
        ok = True
        for i in range(x.size):
            r = np.concatenate([[x[i]], z])
            _, VR, rep, _ = _ss_reduced_dare(A, C, K, V, None, r)
            if rep < 0:
                ok = False
                break
            total += float(np.log(VR[0, 0]) - np.log(DV[x[i]]))
        if ok:
            F[a] = total
    return F


def ss_to_gwiogc(A, C, K, V, groups: list, inout: str = "in") -> npt.NDArray[np.floating]:
    """Group-level information-optimised GC from an SS model.

    State-space analogue of MVGC2's ``var_to_gwiogc.m``, using the reduced
    DARE construction from ``ss_to_iogc.m``. MVGC2 has no function named
    ``ss_to_gwiogc.m``.
    """
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    g, _ = check_group(groups, n)
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    F = np.full(g, np.nan)
    for a in range(g):
        if gcin:
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
        else:
            y = np.atleast_1d(np.asarray(groups[a], dtype=int))
            x = np.array([i for i in range(n) if i not in y])
        _, VR, rep, _ = _ss_reduced_dare(A, C, K, V, None, x)
        if rep < 0:
            continue
        if x.size == 1:
            F[a] = float(np.log(VR[0, 0]) - np.log(V[x[0], x[0]]))
        else:
            F[a] = logdet(VR) - logdet(V[np.ix_(x, x)])
    return F


# ---------------------------------------------------------------------------
# Spectral variants
# ---------------------------------------------------------------------------


def var_to_siogc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    inout: str,
    fres: int,
    H: npt.NDArray[np.complexfloating] | None = None,
) -> npt.NDArray[np.floating]:
    """Frequency-domain information-optimised GC.

    Port of ``gc/var/var_to_siogc.m``. Returns a ``(n, fres+1)`` array.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    h = fres + 1
    f = np.full((n, h), np.nan)
    if H is None:
        H = var2trfun(A, fres)
    VL = cholesky(V, lower=True)
    if gcin:
        for x in range(n):
            y = np.array([i for i in range(n) if i != x])
            PVL = cholesky(parcov(V, y, [x]), lower=True)
            for k in range(h):
                HVL = H[x : x + 1, :, k] @ VL
                SR = HVL @ HVL.conj().T
                HR = H[x : x + 1, y, k] @ PVL
                f[x, k] = np.real(np.log(SR[0, 0]) - np.log((SR - HR @ HR.conj().T)[0, 0]))
    else:
        for y in range(n):
            x = np.array([i for i in range(n) if i != y])
            PVL = cholesky(parcov(V, [y], x), lower=True)
            for k in range(h):
                HVL = H[x, :, k] @ VL
                SR = HVL @ HVL.conj().T
                HR = H[np.ix_(x, [y])][:, :, k] @ PVL
                f[y, k] = np.real(logdet(SR) - logdet(SR - HR @ HR.conj().T))
    return f


def var_to_sgwcgc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    groups: list,
    fres: int,
) -> npt.NDArray[np.floating]:
    """Frequency-domain groupwise conditional GC. Port of ``var_to_sgwcgc.m``.

    Returns shape ``(g, g, fres + 1)``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, p = A.shape
    g, _ = check_group(groups, n)
    h = fres + 1
    f = np.full((g, g, h), np.nan)
    H = var2trfun(A, fres)
    # Pre-compute partial-covariance Cholesky factors
    PVL = []
    for a in range(g):
        x = np.atleast_1d(np.asarray(groups[a], dtype=int))
        w = np.array([i for i in range(n) if i not in x])
        PVL.append(cholesky(parcov(V, w, x), lower=True))
    for b in range(g):
        y = np.atleast_1d(np.asarray(groups[b], dtype=int))
        r = np.array([i for i in range(n) if i not in y])
        nr = r.size
        # Reduced-VAR SS form via vardare
        _, VR, rep = _vardare(A, V, y, r)
        if rep < 0:
            continue
        # Build a "reduced" SS to feed ss2itrfun
        # Companion: AR = [A_flat; I, 0] of size (p*n, p*n)
        pn = p * n
        pn1 = pn - n
        AR = np.zeros((pn, pn))
        AR[:n, :] = np.concatenate([A[:, :, k] for k in range(p)], axis=1)
        if pn1 > 0:
            AR[n:, :pn1] = np.eye(pn1)
        CR = np.zeros((nr, pn))
        CR[:, :] = np.concatenate([A[np.ix_(r, np.arange(n))][:, :, k] for k in range(p)], axis=1)
        # KR will be set per-frequency from the vardare result; use ss2itrfun on
        # the canonical companion-form pair with the recovered KT (Kalman gain)
        # which vardare returns. For groupwise spectral GC we re-derive via
        # var2itrfun on the reduced VAR (the simpler path used by MVGC2).
        # Use autocov_to_var to estimate the reduced VAR coefficients:
        from .var import autocov_to_var, var_to_autocov

        G_full, _ = var_to_autocov(A, V, qmax=-max(2 * p, 8))
        try:
            AR_red, _ = autocov_to_var(G_full[np.ix_(r, r)])
        except Exception:
            continue
        BR = var2itrfun(AR_red, fres)
        for a in range(g):
            if a == b:
                continue
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            xr = np.array([np.where(r == xi)[0][0] for xi in x])
            w = np.array([i for i in range(n) if i not in x])
            SR = VR[np.ix_(xr, xr)]
            LDSR = logdet(SR)
            PVLa = PVL[a]
            for k in range(h):
                HR = BR[xr, :, k] @ H[np.ix_(r, w)][:, :, k] @ PVLa
                f[a, b, k] = np.real(LDSR - logdet(SR - HR @ HR.conj().T))
    return f


def var_to_sgwiogc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    groups: list,
    inout: str,
    fres: int,
) -> npt.NDArray[np.floating]:
    """Spectral group-level IO GC. Port of ``var_to_sgwiogc.m``.

    Returns shape ``(g, fres + 1)``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    g, _ = check_group(groups, n)
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    h = fres + 1
    f = np.full((g, h), np.nan)
    H = var2trfun(A, fres)
    VL = cholesky(V, lower=True)
    for a in range(g):
        if gcin:
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            y = np.array([i for i in range(n) if i not in x])
        else:
            y = np.atleast_1d(np.asarray(groups[a], dtype=int))
            x = np.array([i for i in range(n) if i not in y])
        PVL = cholesky(parcov(V, y, x), lower=True)
        for k in range(h):
            HVL = H[x, :, k] @ VL
            SR = HVL @ HVL.conj().T
            HR = H[np.ix_(x, y)][:, :, k] @ PVL
            f[a, k] = np.real(logdet(SR) - logdet(SR - HR @ HR.conj().T))
    return f


def ss_to_siogc(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    inout: str,
    fres: int,
) -> npt.NDArray[np.floating]:
    """SS-based spectral IO GC."""
    H = ss2trfun(A, C, K, fres)
    return _spectral_iogc_from_H(H, V, inout)


def ss_to_sgwcgc(A, C, K, V, groups: list, fres: int) -> npt.NDArray[np.floating]:
    """SS-based spectral groupwise conditional GC.

    Falls back to the VAR-based implementation by re-deriving the
    reduced-VAR per-frequency components from the companion form. For
    ``ss_to_sgwcgc`` we use ``ss2trfun`` on the reduced model, computed from
    the DARE-reduced parameters.
    """
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    g, _ = check_group(groups, n)
    h = fres + 1
    f = np.full((g, g, h), np.nan)
    H = ss2trfun(A, C, K, fres)
    PVL = []
    for a in range(g):
        x = np.atleast_1d(np.asarray(groups[a], dtype=int))
        w = np.array([i for i in range(n) if i not in x])
        PVL.append(cholesky(parcov(V, w, x), lower=True))
    for b in range(g):
        y = np.atleast_1d(np.asarray(groups[b], dtype=int))
        r = np.array([i for i in range(n) if i not in y])
        KR, VR, rep, _ = _ss_reduced_dare(A, C, K, V, y, r)
        if rep < 0:
            continue
        BR = ss2itrfun(A, C[r, :], KR, fres)
        for a in range(g):
            if a == b:
                continue
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            xr = np.array([np.where(r == xi)[0][0] for xi in x])
            w = np.array([i for i in range(n) if i not in x])
            SR = VR[np.ix_(xr, xr)]
            LDSR = logdet(SR)
            PVLa = PVL[a]
            for k in range(h):
                HR = BR[xr, :, k] @ H[np.ix_(r, w)][:, :, k] @ PVLa
                f[a, b, k] = np.real(LDSR - logdet(SR - HR @ HR.conj().T))
    return f


def ss_to_sgwiogc(A, C, K, V, groups: list, inout: str, fres: int) -> npt.NDArray[np.floating]:
    """SS-based spectral group-level IO GC."""
    H = ss2trfun(A, C, K, fres)
    return _spectral_gwiogc_from_H(H, V, groups, inout)


def _spectral_iogc_from_H(
    H: npt.NDArray[np.complexfloating], V: npt.NDArray[np.floating], inout: str
) -> npt.NDArray[np.floating]:
    n = V.shape[0]
    h = H.shape[2]
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    f = np.full((n, h), np.nan)
    VL = cholesky(V, lower=True)
    if gcin:
        for x in range(n):
            y = np.array([i for i in range(n) if i != x])
            PVL = cholesky(parcov(V, y, [x]), lower=True)
            for k in range(h):
                HVL = H[x : x + 1, :, k] @ VL
                SR = HVL @ HVL.conj().T
                HR = H[x : x + 1, y, k] @ PVL
                f[x, k] = np.real(np.log(SR[0, 0]) - np.log((SR - HR @ HR.conj().T)[0, 0]))
    else:
        for y in range(n):
            x = np.array([i for i in range(n) if i != y])
            PVL = cholesky(parcov(V, [y], x), lower=True)
            for k in range(h):
                HVL = H[x, :, k] @ VL
                SR = HVL @ HVL.conj().T
                HR = H[np.ix_(x, [y])][:, :, k] @ PVL
                f[y, k] = np.real(logdet(SR) - logdet(SR - HR @ HR.conj().T))
    return f


def _spectral_gwiogc_from_H(
    H: npt.NDArray[np.complexfloating],
    V: npt.NDArray[np.floating],
    groups: list,
    inout: str,
) -> npt.NDArray[np.floating]:
    n = V.shape[0]
    h = H.shape[2]
    g, _ = check_group(groups, n)
    gcin = inout.lower() == "in"
    if not gcin and inout.lower() != "out":
        raise ValueError("inout must be 'in' or 'out'")
    f = np.full((g, h), np.nan)
    VL = cholesky(V, lower=True)
    for a in range(g):
        if gcin:
            x = np.atleast_1d(np.asarray(groups[a], dtype=int))
            y = np.array([i for i in range(n) if i not in x])
        else:
            y = np.atleast_1d(np.asarray(groups[a], dtype=int))
            x = np.array([i for i in range(n) if i not in y])
        PVL = cholesky(parcov(V, y, x), lower=True)
        for k in range(h):
            HVL = H[x, :, k] @ VL
            SR = HVL @ HVL.conj().T
            HR = H[np.ix_(x, y)][:, :, k] @ PVL
            f[a, k] = np.real(logdet(SR) - logdet(SR - HR @ HR.conj().T))
    return f
