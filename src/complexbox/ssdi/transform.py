"""Residual-decorrelation transforms and subspace push-forward / pull-back.

Ports of SSDI-1's ``utils/transform_ss.m``, ``utils/transform_var.m``,
``utils/transform_subspace.m``, ``utils/itransform_subspace.m``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

from ._grassmann import orthonormalise

__all__ = [
    "transform_ss",
    "transform_var",
    "transform_subspace",
    "itransform_subspace",
]


def transform_ss(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Decorrelate-and-normalise innovations of an SS model.

    Port of ``transform_ss.m``. Returns ``(A, C', K', I)`` where the new
    residual covariance is the identity.
    """
    A, C, K, V = (np.asarray(x, dtype=float) for x in (A, C, K, V))
    n = C.shape[0]
    SQRTV = cholesky(V, lower=True)
    ISQRTV = np.linalg.inv(SQRTV)
    Cn = ISQRTV @ C
    Kn = K @ SQRTV
    return A, Cn, Kn, np.eye(n)


def transform_var(
    A: npt.NDArray[np.floating], V: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Decorrelate-and-normalise innovations of a VAR.

    Port of ``transform_var.m``. Each lag block is multiplied by the inverse
    Cholesky factor on the left and by the Cholesky factor on the right.
    """
    A = np.asarray(A, dtype=float)
    V = np.asarray(V, dtype=float)
    n, _, p = A.shape
    SQRTV = cholesky(V, lower=True)
    ISQRTV = np.linalg.inv(SQRTV)
    An = np.empty_like(A)
    for k in range(p):
        An[:, :, k] = ISQRTV @ A[:, :, k] @ SQRTV
    return An, np.eye(n)


def transform_subspace(
    L0: npt.NDArray[np.floating], V0: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Push an orthonormal basis through the residual decorrelation map.

    Port of ``transform_subspace.m``. ``L = orthonormalise(V0_chol.T @ L0)``.
    Supports batched input of shape ``(n, m)`` or ``(n, m, R)``.
    """
    L0 = np.asarray(L0, dtype=float)
    V0 = np.asarray(V0, dtype=float)
    V0LCT = cholesky(V0, lower=True).T
    if L0.ndim == 2:
        return orthonormalise(V0LCT @ L0)
    n, m, R = L0.shape
    L = np.empty_like(L0)
    for k in range(R):
        L[:, :, k] = orthonormalise(V0LCT @ L0[:, :, k])
    return L


def itransform_subspace(
    L: npt.NDArray[np.floating], V0: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Pull an orthonormal basis back to the original residual coordinates.

    Port of ``itransform_subspace.m``.
    """
    L = np.asarray(L, dtype=float)
    V0 = np.asarray(V0, dtype=float)
    IV0LCT = np.linalg.inv(cholesky(V0, lower=True).T)
    if L.ndim == 2:
        return orthonormalise(IV0LCT @ L)
    n, m, R = L.shape
    L0 = np.empty_like(L)
    for k in range(R):
        L0[:, :, k] = orthonormalise(IV0LCT @ L[:, :, k])
    return L0
