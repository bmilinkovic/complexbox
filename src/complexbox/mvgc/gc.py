"""Granger causality from VAR / SS / autocovariance / CPSD parameters.

Ports of the time-domain and spectral GC routines in MVGC2's ``gc/`` tree.
The matrix sign convention is the MATLAB one: ``F[i, j]`` is the causality
**from j to i**; diagonals are NaN.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._lyap import mdare
from ._utils import logdet
from .ss import ss2itrfun, ss2trfun
from .var import var2itrfun, var2trfun

__all__ = [
    "var_to_mvgc",
    "var_to_pwcgc",
    "ss_to_mvgc",
    "ss_to_pwcgc",
    "var_to_smvgc",
    "var_to_spwcgc",
    "ss_to_smvgc",
    "ss_to_spwcgc",
    "autocov_to_mvgc",
    "autocov_to_pwcgc",
    "autocov_to_smvgc",
    "autocov_to_spwcgc",
    "cpsd_to_mvgc",
    "cpsd_to_pwcgc",
    "cpsd_to_smvgc",
    "cpsd_to_spwcgc",
]


def _vardare(A, V, y, r):
    """Solve the DARE for the SS model derived from a reduced VAR.

    Port of MVGC2's ``vardare.m``.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, p = A.shape
    ny = len(y)
    nr = len(r)
    pny = p * ny
    pny1 = pny - ny

    Ass = np.zeros((pny, pny))
    Ass[:ny, :] = np.concatenate([A[np.ix_(y, y)][:, :, k] for k in range(p)], axis=1)
    if pny1 > 0:
        Ass[ny:, :pny1] = np.eye(pny1)
    C = np.zeros((nr, pny))
    C[:, :] = np.concatenate([A[np.ix_(r, y)][:, :, k] for k in range(p)], axis=1)
    Q = np.zeros((pny, pny))
    Q[:ny, :ny] = V[np.ix_(y, y)]
    S = np.zeros((pny, nr))
    S[:ny, :] = V[np.ix_(y, r)]
    R = V[np.ix_(r, r)]
    K, VR, rep, _ = mdare(Ass, C, Q, R, S)
    return K, VR, rep


def var_to_mvgc(A, V, x, y) -> float:
    """Multivariate Granger causality from sources ``y`` to targets ``x``.

    Port of ``gc/var/var_to_mvgc.m``. ``x`` and ``y`` are integer lists of
    variable indices (no overlap). All remaining variables are conditioned
    out.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, _p = A.shape
    x = np.atleast_1d(x).astype(int).ravel()
    y = np.atleast_1d(y).astype(int).ravel()
    xy = np.concatenate([x, y])
    if len(np.unique(xy)) != len(xy):
        raise ValueError("x and y must be disjoint")
    z = np.setdiff1d(np.arange(n), xy)
    r = np.concatenate([x, z])
    nx = len(x)
    xr = np.arange(nx)
    _, VR, rep = _vardare(A, V, y, r)
    if rep < 0:
        return float("nan")
    return logdet(VR[np.ix_(xr, xr)]) - logdet(V[np.ix_(x, x)])


def var_to_pwcgc(A, V) -> npt.NDArray[np.floating]:
    """Pairwise-conditional GC for every (target, source) pair.

    Port of ``gc/var/var_to_pwcgc.m``. ``F[i, j]`` = causality j → i.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    DV = np.diag(V)
    LDV = np.log(DV)
    F = np.full((n, n), np.nan)
    for y in range(n):
        r = np.array([i for i in range(n) if i != y])
        _, VR, rep = _vardare(A, V, [y], r)
        if rep < 0:
            continue
        F[r, y] = np.log(np.diag(VR)) - LDV[r]
    return F


def ss_to_mvgc(A, C, K, V, x, y) -> float:
    """MVGC from an innovations-form SS model. Port of ``ss_to_mvgc.m``."""
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    x = np.atleast_1d(x).astype(int).ravel()
    y = np.atleast_1d(y).astype(int).ravel()
    z_idx = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    r = np.concatenate([x, z_idx])
    nx = len(x)
    xr = np.arange(nx)

    # Reduced model: drop rows of C and K corresponding to y, but state
    # transition stays. Use DARE on (A, C_r) with covariances
    KVK = K @ V @ K.T
    Q = KVK
    R = V[np.ix_(r, r)]
    S = K @ V[:, r]
    Cr = C[r, :]
    _, VR, rep, _ = mdare(A, Cr, Q, R, S)
    if rep < 0:
        return float("nan")
    return logdet(VR[np.ix_(xr, xr)]) - logdet(V[np.ix_(x, x)])


def ss_to_pwcgc(A, C, K, V) -> npt.NDArray[np.floating]:
    """Pairwise-conditional GC for an innovations-form SS model."""
    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    F = np.full((n, n), np.nan)
    KVK = K @ V @ K.T
    for y in range(n):
        r = np.array([i for i in range(n) if i != y])
        Cr = C[r, :]
        S = K @ V[:, r]
        R = V[np.ix_(r, r)]
        _, VR, rep, _ = mdare(A, Cr, KVK, R, S)
        if rep < 0:
            continue
        F[r, y] = np.log(np.diag(VR)) - np.log(np.diag(V)[r])
    return F


def _var_companion(A: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Build the companion-form state transition matrix from a VAR(p) array."""
    n, _, p = A.shape
    pn = p * n
    pn1 = pn - n
    AR = np.zeros((pn, pn))
    AR[:n, :] = np.concatenate([A[:, :, k] for k in range(p)], axis=1)
    if pn1 > 0:
        AR[n:, :pn1] = np.eye(pn1)
    return AR


def _build_reduced_KR(KT, y, r, n, p):
    """Construct the reduced-SS Kalman gain ``KR`` from vardare's ``KT``.

    Mirrors MATLAB's loop in ``var_to_spwcgc.m`` / ``var_to_smvgc.m``::

        kn = 0
        for k1 in 0..p-1:
            KR[kn + y, :] = KT[k1*ny : (k1+1)*ny, :]
            kn += n
    """
    ny = len(y)
    nr = len(r)
    pn = p * n
    KR = np.zeros((pn, nr))
    KR[r, :] = np.eye(nr)
    for k1 in range(p):
        KR[k1 * n + np.asarray(y), :] = KT[k1 * ny : (k1 + 1) * ny, :]
    return KR


def var_to_smvgc(
    A: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    x,
    y,
    fres: int,
) -> npt.NDArray[np.floating]:
    """Pairwise-conditional spectral GC from sources ``y`` to targets ``x``.

    Port of ``gc/var/var_to_smvgc.m``. Uses the DARE-based reduced-state-space
    construction to obtain the spectral residual covariance.
    """
    from ._utils import parcov

    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, p = A.shape
    x = np.atleast_1d(np.asarray(x, dtype=int)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=int)).ravel()
    z_idx = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    r = np.concatenate([x, z_idx])
    xr = np.arange(x.size)
    h = fres + 1

    KT, VR, rep = _vardare(A, V, y, r)
    if rep < 0:
        return np.full(h, np.nan)

    AR_ss = _var_companion(A)
    CR = np.concatenate([A[r, :, k] for k in range(p)], axis=1)
    KR = _build_reduced_KR(KT, y, r, n, p)
    BR = ss2itrfun(AR_ss, CR, KR, fres)
    H = var2trfun(A, fres)
    # w = [y, z] indices in the *full* model
    w = np.concatenate([y, z_idx])
    PVL = np.linalg.cholesky(parcov(V, w, x))
    f = np.full(h, np.nan)
    SR_xx = VR[np.ix_(xr, xr)]
    LDSR = logdet(SR_xx)
    for k in range(h):
        HRk = BR[xr, :, k] @ H[np.ix_(r, w)][:, :, k] @ PVL
        f[k] = np.real(LDSR - logdet(SR_xx - HRk @ HRk.conj().T))
    return f


def var_to_spwcgc(
    A: npt.NDArray[np.floating], V: npt.NDArray[np.floating], fres: int
) -> npt.NDArray[np.floating]:
    """Pairwise-conditional spectral GC matrix from a VAR.

    Port of ``gc/var/var_to_spwcgc.m``. Returns shape ``(n, n, fres + 1)``.
    """
    from ._utils import parcov

    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, p = A.shape
    h = fres + 1
    f = np.full((n, n, h), np.nan)
    AR_ss = _var_companion(A)
    H = var2trfun(A, fres)
    # Pre-compute partial-cov Cholesky factors per target x
    PVL = np.zeros((n - 1, n - 1, n))
    for xi in range(n):
        rest = np.array([i for i in range(n) if i != xi])
        PVL[:, :, xi] = np.linalg.cholesky(parcov(V, rest, [xi]))
    for yi in range(n):
        r = np.array([i for i in range(n) if i != yi])
        KT, VR, rep = _vardare(A, V, [yi], r)
        if rep < 0:
            continue
        CR = np.concatenate([A[r, :, k] for k in range(p)], axis=1)
        KR = _build_reduced_KR(KT, [yi], r, n, p)
        BR = ss2itrfun(AR_ss, CR, KR, fres)
        for xr_idx in range(n - 1):
            xi = int(r[xr_idx])
            w = np.array([i for i in range(n) if i != xi])
            SR = float(VR[xr_idx, xr_idx])
            LSR = np.log(SR)
            PVLx = PVL[:, :, xi]
            for k in range(h):
                HRk = BR[xr_idx : xr_idx + 1, :, k] @ H[np.ix_(r, w)][:, :, k] @ PVLx
                f[xi, yi, k] = np.real(LSR - np.log(SR - (HRk @ HRk.conj().T)[0, 0]))
    return f


def ss_to_smvgc(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    x,
    y,
    fres: int,
) -> npt.NDArray[np.floating]:
    """Pairwise-conditional spectral GC from an innovations-form SS model.

    Port of ``gc/ss/ss_to_smvgc.m`` (uses the same parcov / BR construction
    as the VAR path, with the reduced Kalman gain solved directly from the
    full-SS DARE).
    """
    from ._utils import parcov

    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    x = np.atleast_1d(np.asarray(x, dtype=int)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=int)).ravel()
    z_idx = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    r = np.concatenate([x, z_idx])
    w = np.concatenate([y, z_idx])
    xr = np.arange(x.size)
    h = fres + 1

    KVK = K @ V @ K.T
    S = K @ V[:, r]
    R = V[np.ix_(r, r)]
    Cr = C[r, :]
    KR, VR, rep, _ = mdare(A, Cr, KVK, R, S)
    if rep < 0:
        return np.full(h, np.nan)

    BR = ss2itrfun(A, Cr, KR, fres)
    H = ss2trfun(A, C, K, fres)
    PVL = np.linalg.cholesky(parcov(V, w, x))
    SR_xx = VR[np.ix_(xr, xr)]
    LDSR = logdet(SR_xx)
    f = np.full(h, np.nan)
    for k in range(h):
        HRk = BR[xr, :, k] @ H[np.ix_(r, w)][:, :, k] @ PVL
        f[k] = np.real(LDSR - logdet(SR_xx - HRk @ HRk.conj().T))
    return f


def ss_to_spwcgc(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    fres: int,
) -> npt.NDArray[np.floating]:
    """Pairwise-conditional spectral GC from an innovations-form SS model."""
    from ._utils import parcov

    A, C, K, V = (np.asarray(z, dtype=float) for z in (A, C, K, V))
    n = C.shape[0]
    h = fres + 1
    f = np.full((n, n, h), np.nan)
    H = ss2trfun(A, C, K, fres)
    KVK = K @ V @ K.T
    PVL = np.zeros((n - 1, n - 1, n))
    for xi in range(n):
        rest = np.array([i for i in range(n) if i != xi])
        PVL[:, :, xi] = np.linalg.cholesky(parcov(V, rest, [xi]))
    for yi in range(n):
        r = np.array([i for i in range(n) if i != yi])
        Cr = C[r, :]
        S = K @ V[:, r]
        R = V[np.ix_(r, r)]
        KR, VR, rep, _ = mdare(A, Cr, KVK, R, S)
        if rep < 0:
            continue
        BR = ss2itrfun(A, Cr, KR, fres)
        for xr_idx in range(n - 1):
            xi = int(r[xr_idx])
            w = np.array([i for i in range(n) if i != xi])
            SR = float(VR[xr_idx, xr_idx])
            LSR = np.log(SR)
            PVLx = PVL[:, :, xi]
            for k in range(h):
                HRk = BR[xr_idx : xr_idx + 1, :, k] @ H[np.ix_(r, w)][:, :, k] @ PVLx
                f[xi, yi, k] = np.real(LSR - np.log(SR - (HRk @ HRk.conj().T)[0, 0]))
    return f


# ---------------------------------------------------------------------------
# Autocovariance-based GC (port of gc/autocov/autocov_to_*.m)
# ---------------------------------------------------------------------------


def autocov_to_mvgc(G, x, y) -> float:
    """MVGC from an autocovariance sequence (Whittle's LWR reduction).

    Port of ``gc/autocov/autocov_to_mvgc.m``.
    """
    from .var import autocov_to_var

    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    z = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    r = np.concatenate([x, z])
    _, V = autocov_to_var(G)
    _, VR = autocov_to_var(G[np.ix_(r, r)])
    xr = np.arange(x.size)
    return logdet(VR[np.ix_(xr, xr)]) - logdet(V[np.ix_(x, x)])


def autocov_to_pwcgc(G) -> npt.NDArray[np.floating]:
    """Pairwise-conditional GC from autocovariance. Port of ``autocov_to_pwcgc.m``."""
    from .var import autocov_to_var

    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    _, V = autocov_to_var(G)
    LSIG = np.log(np.diag(V))
    F = np.full((n, n), np.nan)
    for j in range(n):
        jo = np.array([i for i in range(n) if i != j])
        _, Vj = autocov_to_var(G[np.ix_(jo, jo)])
        F[jo, j] = np.log(np.diag(Vj)) - LSIG[jo]
    return F


def autocov_to_smvgc(G, x, y, fres: int) -> npt.NDArray[np.floating]:
    """Spectral MVGC from autocovariance. Port of ``autocov_to_smvgc.m``."""
    from ._utils import parcov
    from .var import autocov_to_var, var2trfun, var_to_cpsd

    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    z = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    A, V = autocov_to_var(G)
    h = fres + 1
    f = np.full(h, np.nan)
    if z.size == 0:
        S = var_to_cpsd(A, V, fres)
        H = var2trfun(A, fres)
        PSIGSR = np.linalg.cholesky(parcov(V, y, x))
        for k in range(h):
            Hk = H[np.ix_(x, y)][:, :, k] @ PSIGSR
            f[k] = np.real(
                logdet(S[np.ix_(x, x)][:, :, k])
                - logdet(S[np.ix_(x, x)][:, :, k] - Hk @ Hk.conj().T)
            )
    else:
        xz = np.concatenate([x, z])
        yz = np.concatenate([y, z])
        xr = np.arange(x.size)
        H = var2trfun(A, fres)
        AR, VR = autocov_to_var(G[np.ix_(xz, xz)])
        BR = var2itrfun(AR, fres)
        SRxx = VR[np.ix_(xr, xr)]
        LDSRxx = logdet(SRxx)
        PSIGSR = np.linalg.cholesky(parcov(V, yz, x))
        for k in range(h):
            HRk = BR[xr, :, k] @ H[np.ix_(xz, yz)][:, :, k] @ PSIGSR
            f[k] = np.real(LDSRxx - logdet(SRxx - HRk @ HRk.conj().T))
    return f


def autocov_to_spwcgc(G, fres: int) -> npt.NDArray[np.floating]:
    """Pairwise spectral GC from autocovariance. Port of ``autocov_to_spwcgc.m``."""
    from ._utils import parcov
    from .var import autocov_to_var, var2trfun

    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    A, V = autocov_to_var(G)
    H = var2trfun(A, fres)
    h = fres + 1
    f = np.full((n, n, h), np.nan)
    for j in range(n):
        oj = np.array([i for i in range(n) if i != j])
        AR, VR = autocov_to_var(G[np.ix_(oj, oj)])
        BR = var2itrfun(AR, fres)
        for ii in range(n - 1):
            i = int(oj[ii])
            oi = np.array([t for t in range(n) if t != i])
            SRii = float(VR[ii, ii])
            LSRii = np.log(SRii)
            PSIGSR = np.linalg.cholesky(parcov(V, oi, [i]))
            for k in range(h):
                HRk = BR[ii : ii + 1, :, k] @ H[np.ix_(oj, oi)][:, :, k] @ PSIGSR
                f[i, j, k] = np.real(LSRii - np.log(SRii - (HRk @ HRk.conj().T)[0, 0]))
    return f


# ---------------------------------------------------------------------------
# CPSD-based GC (port of gc/cpsd/cpsd_to_*.m), uses Wilson's factorisation.
# ---------------------------------------------------------------------------


def cpsd_to_mvgc(
    S: npt.NDArray[np.complexfloating],
    x,
    y,
    tol: float = 1e-7,
    maxi: int | None = None,
) -> float:
    """MVGC from a CPSD via Wilson spectral factorisation.

    Port of ``gc/cpsd/cpsd_to_mvgc.m``.
    """
    from ._specfac import cpsd_specfac

    S = np.asarray(S)
    n = S.shape[0]
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    z = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    r = np.concatenate([x, z])
    xr = np.arange(x.size)

    _, V, conv_f, *_ = cpsd_specfac(S, tol=tol, maxi=maxi)
    if not conv_f:
        return float("nan")
    _, VR, conv_r, *_ = cpsd_specfac(S[np.ix_(r, r)], tol=tol, maxi=maxi)
    if not conv_r:
        return float("nan")
    return logdet(VR[np.ix_(xr, xr)]) - logdet(V[np.ix_(x, x)])


def cpsd_to_pwcgc(
    S: npt.NDArray[np.complexfloating],
    tol: float = 1e-7,
    maxi: int | None = None,
) -> npt.NDArray[np.floating]:
    """Pairwise-conditional GC from CPSD. Port of ``cpsd_to_pwcgc.m``."""
    from ._specfac import cpsd_specfac

    S = np.asarray(S)
    n = S.shape[0]
    F = np.full((n, n), np.nan)
    _, V, conv, *_ = cpsd_specfac(S, tol=tol, maxi=maxi)
    if not conv:
        return F
    LDV = np.log(np.diag(V))
    for y in range(n):
        r = np.array([i for i in range(n) if i != y])
        _, VR, conv_r, *_ = cpsd_specfac(S[np.ix_(r, r)], tol=tol, maxi=maxi)
        if not conv_r:
            continue
        F[r, y] = np.log(np.diag(VR)) - LDV[r]
    return F


def cpsd_to_smvgc(
    S: npt.NDArray[np.complexfloating],
    x,
    y,
    tol: float = 1e-7,
    maxi: int | None = None,
) -> npt.NDArray[np.floating]:
    """Spectral MVGC from CPSD. Port of ``cpsd_to_smvgc.m``."""
    from ._specfac import cpsd_specfac
    from ._utils import parcov

    S = np.asarray(S)
    n, _, h = S.shape
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    z = np.setdiff1d(np.arange(n), np.concatenate([x, y]))
    r = np.concatenate([x, z])
    w = np.concatenate([y, z])
    xr = np.arange(x.size)
    f = np.full(h, np.nan)

    H, V, conv_f, *_ = cpsd_specfac(S, tol=tol, maxi=maxi)
    if not conv_f:
        return f
    PL = np.linalg.cholesky(parcov(V, w, x))
    if z.size == 0:
        # Use the directly-factorised H
        for k in range(h):
            HL = H[x, :, k] @ np.linalg.cholesky(V)
            SR = HL @ HL.conj().T
            HR = H[np.ix_(x, y)][:, :, k] @ PL
            f[k] = np.real(logdet(SR) - logdet(SR - HR @ HR.conj().T))
    else:
        HR_full, VR, conv_r, *_ = cpsd_specfac(S[np.ix_(r, r)], tol=tol, maxi=maxi)
        if not conv_r:
            return f
        nr = r.size
        BR = np.empty((nr, nr, h), dtype=complex)
        for k in range(h):
            BR[:, :, k] = np.linalg.inv(HR_full[:, :, k])
        SR_block = VR[np.ix_(xr, xr)]
        LDSR = logdet(SR_block)
        for k in range(h):
            HRk = BR[xr, :, k] @ H[np.ix_(r, w)][:, :, k] @ PL
            f[k] = np.real(LDSR - logdet(SR_block - HRk @ HRk.conj().T))
    return f


def cpsd_to_spwcgc(
    S: npt.NDArray[np.complexfloating],
    tol: float = 1e-7,
    maxi: int | None = None,
) -> npt.NDArray[np.floating]:
    """Pairwise spectral GC from CPSD. Port of ``cpsd_to_spwcgc.m``."""
    from ._specfac import cpsd_specfac
    from ._utils import parcov

    S = np.asarray(S)
    n, _, h = S.shape
    f = np.full((n, n, h), np.nan)
    H, V, conv_f, *_ = cpsd_specfac(S, tol=tol, maxi=maxi)
    if not conv_f:
        return f
    nr = n - 1
    PL = np.zeros((nr, nr, n))
    for xi in range(n):
        w = np.array([i for i in range(n) if i != xi])
        PL[:, :, xi] = np.linalg.cholesky(parcov(V, w, [xi]))
    for y in range(n):
        r = np.array([i for i in range(n) if i != y])
        HR_full, VR, conv_r, *_ = cpsd_specfac(S[np.ix_(r, r)], tol=tol, maxi=maxi)
        if not conv_r:
            continue
        BR = np.empty((nr, nr, h), dtype=complex)
        for k in range(h):
            BR[:, :, k] = np.linalg.inv(HR_full[:, :, k])
        for xr in range(n - 1):
            xi = int(r[xr])
            w = np.array([i for i in range(n) if i != xi])
            SR = float(VR[xr, xr])
            LSR = np.log(SR)
            PLx = PL[:, :, xi]
            for k in range(h):
                HRk = BR[xr : xr + 1, :, k] @ H[np.ix_(r, w)][:, :, k] @ PLx
                f[xi, y, k] = np.real(LSR - np.log(SR - (HRk @ HRk.conj().T)[0, 0]))
    return f
