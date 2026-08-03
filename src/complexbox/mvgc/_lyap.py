"""Discrete-time Lyapunov and DARE solvers.

``dlyap_aitr`` is a direct port of MVGC2's Smith accelerated iteration
(``utils/dlyap_aitr.m``); ``dlyap`` is a wrapper around SciPy's Schur-based
solver that matches MATLAB's Control System Toolbox interface; ``mdare`` is a
port of MVGC2's stripped-down DARE solver (``utils/mdare.m``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

__all__ = ["dlyap_aitr", "dlyap", "mdare"]


def dlyap_aitr(
    A: npt.NDArray[np.floating],
    Q: npt.NDArray[np.floating],
    maxiters: int = 100,
    maxrelerr: float = 1e-8,
) -> tuple[npt.NDArray[np.floating], int]:
    """Solve ``X = A X A' + Q`` by Smith's accelerated iteration.

    Direct port of ``dlyap_aitr.m``. Caller must ensure ``rho(A) < 1`` and that
    ``Q`` is symmetric positive-definite.

    Parameters
    ----------
    A : (r, r) array
    Q : (r, r) array — symmetric positive-definite
    maxiters : int
    maxrelerr : float

    Returns
    -------
    X : (r, r) array — Lyapunov solution
    iters : int — number of iterations performed

    Raises
    ------
    RuntimeError if convergence is not reached in ``maxiters`` steps.
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Q.shape != A.shape:
        raise ValueError("Q must have the same shape as A")

    X = Q.copy()
    AA = A.copy()
    snorm = np.linalg.norm(Q, "fro")
    minrelerr = np.finfo(float).max
    relerr = np.inf
    iters = 0
    for it in range(1, maxiters + 2):
        iters = it
        relerr = np.linalg.norm(X - A @ X @ A.T - Q, "fro") / snorm
        if relerr < maxrelerr:
            if relerr >= minrelerr:
                break
        if relerr < minrelerr:
            minrelerr = relerr
        X = AA @ X @ AA.T + X
        AA = AA @ AA
    if iters > maxiters:
        raise RuntimeError(f"dlyap_aitr exceeded max iterations (max rel err = {relerr:.3e})")
    return X, iters


def dlyap(A: npt.NDArray[np.floating], Q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Solve the discrete Lyapunov equation ``X = A X A' + Q`` (Schur method).

    Identical to MATLAB's ``dlyap``; uses SciPy's
    :func:`scipy.linalg.solve_discrete_lyapunov` with the Bartels-Stewart
    method.
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return la.solve_discrete_lyapunov(A, Q, method="bilinear")


def mdare(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    Q: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating] | None = None,
    S: npt.NDArray[np.floating] | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    npt.NDArray[np.floating],
]:
    """Solve the discrete algebraic Riccati equation for innovations form.

    Returns ``(K, V, rep, P)`` where ``K`` is the Kalman gain, ``V`` the
    innovations covariance, ``P`` the DARE solution, and ``rep`` the relative
    residual (or a negative error code, mirroring MVGC2 conventions:
    ``-1`` = eigenvalues on/near unit circle, ``-2`` = no stabilising
    solution).

    Solves the standard general-form Kalman DARE::

        P = A P A' - (A P C' + S)(C P C' + R)^{-1}(A P C' + S)' + Q
        K = (A P C' + S)(C P C' + R)^{-1}
        V = C P C' + R

    Backed by :func:`scipy.linalg.solve_discrete_are` (Schur method, real),
    which is mathematically equivalent to the QZ-based MATLAB implementation
    in ``mdare.m``.
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    Q = np.asarray(Q, dtype=float)
    r = A.shape[0]
    n = C.shape[0]
    if R is None:
        R = np.eye(n)
    else:
        R = np.asarray(R, dtype=float)
    if S is None:
        S = np.zeros((r, n))
    else:
        S = np.asarray(S, dtype=float)

    # SciPy's solve_discrete_are signature: a, b, q, r, e=None, s=None.
    # We pass B = C.T (because SciPy uses state-feedback form), with the
    # appropriate sign convention. The standard Kalman DARE for innovations
    # form is dual to the LQR DARE: solve P = A' P A - A' P B (R + B' P B)^-1
    # B' P A + Q. To map to the Kalman form we transpose A and swap roles.
    try:
        P = la.solve_discrete_are(A.T, C.T, Q, R, s=S)
    except Exception:
        # Fall back to error sentinel; caller must check
        return (
            np.empty((r, n)),
            np.empty((n, n)),
            -2.0,
            np.empty((r, r)),
        )

    P = 0.5 * (P + P.T)
    U = A @ P @ C.T + S
    V = C @ P @ C.T + R
    try:
        K = la.solve(V, U.T, assume_a="pos").T
    except np.linalg.LinAlgError:
        return (np.empty((r, n)), np.empty((n, n)), -1.0, P)

    APA = A @ P @ A.T - P
    UK = U @ K.T
    rep = float(
        np.linalg.norm(APA - UK + Q, 1)
        / (1.0 + np.linalg.norm(APA, 1) + np.linalg.norm(UK, 1) + np.linalg.norm(Q, 1))
    )
    return K, V, rep, P
