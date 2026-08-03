"""Linear-algebra and statistical helpers used throughout ``complexbox.mvgc``.

Direct ports of MVGC2's ``utils/`` and ``stats/`` mini-utilities. The exposed
functions mirror MATLAB names where useful for code review against the original
toolbox.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky, eigvals

__all__ = [
    "logdet",
    "specnorm",
    "var_decay",
    "demean",
    "symmetrise",
    "maxabs",
    "is_pos_def",
    "parcov",
    "check_group",
]


def parcov(
    V: npt.NDArray[np.floating],
    x: npt.NDArray | list[int] | int,
    y: npt.NDArray | list[int] | int,
) -> npt.NDArray[np.floating]:
    """Partial covariance of ``V[x, x]`` given ``V[y, y]``.

    Port of MVGC2's ``utils/parcov.m``::

        P = V[x, x] - V[x, y] @ V[y, y]^{-1} @ V[y, x]

    Computed numerically via Cholesky for stability.
    """
    V = np.asarray(V, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=int))
    y = np.atleast_1d(np.asarray(y, dtype=int))
    Vyy = V[np.ix_(y, y)]
    Vyx = V[np.ix_(y, x)]
    if Vyy.shape == (1, 1):
        U = Vyx / np.sqrt(Vyy[0, 0])
    else:
        from scipy.linalg import cholesky, solve_triangular

        L = cholesky(Vyy, lower=True)
        U = solve_triangular(L, Vyx, lower=True)
    return V[np.ix_(x, x)] - U.T @ U


def check_group(group: list, n: int | None = None) -> tuple[int, npt.NDArray[np.intp]]:
    """Validate a groupwise index specification.

    Port of MVGC2's ``utils/check_group.m``. ``group`` is a list-of-lists
    of variable indices. Returns ``(num_groups, group_sizes)``.
    """
    if not isinstance(group, (list, tuple)):
        raise TypeError("group must be a list of index arrays")
    flat = np.concatenate([np.atleast_1d(np.asarray(g, dtype=int)) for g in group])
    if len(np.unique(flat)) != len(flat):
        raise ValueError("group indices must be unique and non-overlapping")
    if n is not None:
        if np.any(flat < 0) or np.any(flat >= n):
            raise ValueError("some group indices are out of range")
    sizes = np.array([len(np.atleast_1d(g)) for g in group], dtype=np.intp)
    return len(group), sizes


def logdet(V: npt.NDArray[np.floating]) -> float:
    """Log-determinant of a symmetric positive-definite matrix via Cholesky.

    Mirrors MVGC2's ``logdet.m``: ``LD = 2 * sum(log(diag(chol(V))))`` if ``V``
    is HPD, else falls back to ``log(det(V))``. Complex matrices fall back to
    the real part of the determinant.
    """
    V = np.asarray(V)
    if np.iscomplexobj(V):
        # Hermitian path: symmetrise to ensure Hermitian, then Cholesky
        V = 0.5 * (V + V.conj().T)
    try:
        L = cholesky(V, lower=True, check_finite=False)
        d = np.diag(L)
        if np.iscomplexobj(d):
            return 2.0 * float(np.sum(np.log(np.abs(d))))
        return 2.0 * float(np.sum(np.log(d)))
    except np.linalg.LinAlgError:
        det = np.linalg.det(V)
        if np.iscomplexobj(det) and abs(det.imag) > np.sqrt(np.finfo(float).eps):
            return float("nan")
        return float(np.log(np.real(det)))


def maxabs(X: npt.NDArray) -> float:
    """Maximum absolute value of array entries (MVGC2 ``maxabs.m``)."""
    return float(np.max(np.abs(X)))


def is_pos_def(V: npt.NDArray[np.floating]) -> bool:
    """True iff ``V`` is symmetric positive-definite (Cholesky test)."""
    try:
        cholesky(V, lower=True, check_finite=False)
        return True
    except np.linalg.LinAlgError:
        return False


def symmetrise(X: npt.NDArray, hermitian: bool = False, ut2lt: bool = True) -> npt.NDArray:
    """Symmetrise (or Hermitianise) a square matrix in place semantics.

    Mirrors MVGC2's ``symmetrise.m``. By default copies the upper triangle to
    the lower (``ut2lt=True``); the result is symmetric if ``hermitian=False``
    else Hermitian.
    """
    X = np.array(X, copy=True)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("symmetrise expects a square matrix")
    n = X.shape[0]
    iu = np.triu_indices(n, k=1)
    il = (iu[1], iu[0])
    if ut2lt:
        X[il] = np.conj(X[iu]) if hermitian else X[iu]
    else:
        X[iu] = np.conj(X[il]) if hermitian else X[il]
    return X


def specnorm(
    A: npt.NDArray[np.floating], new_rho: float | None = None
) -> tuple[npt.NDArray[np.floating], float] | float:
    """Spectral radius of a matrix or VAR coefficient array.

    Mirrors the historically named MVGC2 ``specnorm.m``. This is the largest
    absolute eigenvalue, *not* the largest singular value. For ``A`` shaped
    ``(n, n, p)``, the eigenvalues are taken from its companion matrix.
    If ``new_rho`` is supplied, the coefficients are decayed (``var_decay``)
    so the new spectral radius equals ``new_rho``; returns
    ``(A_new, old_rho)``.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        # 1-lag VAR-like polynomial
        p = A.size
        a = A.ravel()
        A1 = np.zeros((p, p))
        A1[0, :] = a
        if p > 1:
            A1[1:, :-1] = np.eye(p - 1)
    else:
        if A.ndim == 2:
            A = A[..., None]
        n, n1, p = A.shape
        if n1 != n:
            raise ValueError("VAR coefficient matrix has bad shape")
        # companion matrix [A_flat; I, 0]
        A_flat = A.transpose(0, 2, 1).reshape(n, p * n)
        # The MATLAB flat order is [A(:,:,1) A(:,:,2) ... A(:,:,p)]:
        A_flat = np.concatenate([A[:, :, k] for k in range(p)], axis=1)
        pn1 = (p - 1) * n
        A1 = np.zeros((p * n, p * n))
        A1[:n, :] = A_flat
        if pn1 > 0:
            A1[n:, :pn1] = np.eye(pn1)
    rho = float(np.max(np.abs(eigvals(A1))))
    if new_rho is None:
        return rho
    return var_decay(A, new_rho / rho), rho


def var_decay(A: npt.NDArray[np.floating], factor: float) -> npt.NDArray[np.floating]:
    """Apply geometric decay to VAR coefficients (mirrors ``var_decay.m``).

    Each lag-``k`` block is multiplied by ``factor**k``.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        out = A.copy()
        for k in range(out.size):
            out[k] = out[k] * factor ** (k + 1)
        return out
    if A.ndim != 3:
        raise ValueError("A must be 1-D or 3-D")
    out = A.copy()
    for k in range(out.shape[2]):
        out[:, :, k] *= factor ** (k + 1)
    return out


def demean(X: npt.NDArray[np.floating], normalise: bool = False) -> npt.NDArray[np.floating]:
    """Temporally demean time-series data (mirrors ``demean.m``).

    For multi-trial data of shape ``(n, m, N)`` the mean is computed across
    the concatenated trials (i.e., ``mean(X[:, :], axis=1)``).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        flat = X
        n, m = flat.shape
        N = 1
        flat_view = flat
    elif X.ndim == 3:
        n, m, N = X.shape
        flat_view = X.reshape(n, m * N)
    else:
        raise ValueError("X must be 2-D or 3-D")
    mu = flat_view.mean(axis=1, keepdims=True)
    Y = flat_view - mu
    if normalise:
        sd = Y.std(axis=1, ddof=1, keepdims=True)
        Y = Y / sd
    if X.ndim == 3:
        Y = Y.reshape(n, m, N)
    return Y
