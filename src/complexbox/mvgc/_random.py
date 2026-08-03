"""Random model generation (VAR, correlation matrix, innovations-form SS).

Ports of MVGC2's ``utils/var_rand.m``, ``utils/corr_rand.m``,
``utils/iss_rand.m``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky, qr

from ._utils import specnorm

__all__ = ["var_rand", "corr_rand", "iss_rand"]


def var_rand(
    n: int | npt.NDArray,
    p: int | None,
    rho: float,
    w: float | None = None,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating]:
    """Generate a random VAR(p) with target spectral radius ``rho``.

    Port of ``utils/var_rand.m``. ``n`` may be a scalar (number of variables)
    or a connectivity matrix / 3-D array shaping the sparsity pattern.
    """
    if rng is None:
        rng = np.random.default_rng()
    if np.isscalar(n):
        A = rng.standard_normal((int(n), int(n), int(p)))
    else:
        C = np.asarray(n)
        if C.ndim == 2:
            if p is None:
                raise ValueError("Need lag count p when n is 2-D connectivity")
            C3 = np.broadcast_to(C[:, :, None], (*C.shape, p)).astype(float)
        elif C.ndim == 3:
            C3 = C.astype(float)
            p = C3.shape[2]
        else:
            raise ValueError("n must be a scalar or a 2/3-D connectivity array")
        n_var = C3.shape[0]
        A = C3 * rng.standard_normal((n_var, n_var, p))
    if w is not None:
        A = np.exp(-w * np.sqrt(p)) * A
    A_new, _ = specnorm(A, rho)
    return A_new


def corr_rand(
    n: int,
    g: float | None = None,
    vexp: float = 2.0,
    tol: float = float(np.sqrt(np.finfo(float).eps)),
    maxretries: int = 1000,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating]:
    """Random correlation matrix with target multi-information ``g``.

    Port of ``utils/corr_rand.m``. ``g = -log|R|``. If ``g is None``, samples
    uniformly from the manifold of correlation matrices via the onion method.
    """
    if rng is None:
        rng = np.random.default_rng()
    if g is None:
        return _onion(n, rng)
    if abs(g) < np.finfo(float).eps:
        return np.eye(n)
    if g < 0:
        g = -0.5 * n * np.log(1.0 - g * g)
    gtarget = g

    g_curr = -np.inf
    D = v = M = None
    for _ in range(maxretries + 1):
        Q, R = qr(rng.standard_normal((n, n)))
        v = np.abs(rng.standard_normal(n)) ** vexp
        M = Q @ np.diag(np.sign(np.diag(R)))
        V = M @ np.diag(v) @ M.T
        g_curr = float(np.sum(np.log(np.diag(V)) - np.log(v)))
        if g_curr >= gtarget:
            break
    if g_curr < gtarget:
        raise RuntimeError("corr_rand: target multi-information not reachable")
    D = np.diag(M @ np.diag(v) @ M.T)

    # Binary chop on c such that V = M diag(v + c) M' has g = gtarget.
    c = 1.0
    while True:
        g_curr = float(np.sum(np.log(D + c) - np.log(v + c)))
        if g_curr <= gtarget:
            break
        c *= 2.0
    chi, clo = c, 0.0
    while chi - clo > tol:
        c = 0.5 * (clo + chi)
        g_curr = float(np.sum(np.log(D + c) - np.log(v + c)))
        if g_curr < gtarget:
            chi = c
        elif g_curr > gtarget:
            clo = c
        else:
            break
    V = M @ np.diag(v + c) @ M.T
    L = cholesky(V, lower=True)
    # Normalise rows of L to unit length → correlation matrix
    L = (1.0 / np.sqrt(np.sum(L * L, axis=1)))[:, None] * L
    return L @ L.T


def _onion(n: int, rng: np.random.Generator) -> npt.NDArray[np.floating]:
    """Onion method (Lewandowski et al., 2009) for uniform correlation matrices."""
    R = np.array([[1.0]])
    for k in range(2, n + 1):
        # sample y uniform on (k-1)-ball with appropriate Jacobian
        beta = 0.5 * (n - k + 1)
        u = rng.beta(0.5 * (k - 1), beta)
        u = np.sqrt(u)
        z = rng.standard_normal(k - 1)
        z = z / np.linalg.norm(z) * u
        w = cholesky(R, lower=True) @ z
        R_new = np.zeros((k, k))
        R_new[: k - 1, : k - 1] = R
        R_new[: k - 1, k - 1] = w
        R_new[k - 1, : k - 1] = w
        R_new[k - 1, k - 1] = 1.0
        R = R_new
    return R


def iss_rand(
    n: int,
    m: int,
    rhoa: float,
    rng: np.random.Generator | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
]:
    """Random stable, minimum-phase innovations-form SS model.

    Port of ``utils/iss_rand.m``.

    Returns ``(A, C, K, rhob)`` where ``rhob = ρ(A - K C)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rhoa >= 1.0:
        raise ValueError("rhoa must be < 1 for a stable model")
    A_raw = rng.standard_normal((m, m))
    A, _ = specnorm(A_raw[:, :, None], rhoa)
    A = A[:, :, 0]
    C = rng.standard_normal((n, m))
    K = rng.standard_normal((m, n))
    M = K @ C
    rmin = _speclim(A, M, -1.0, 0.0)
    rmax = _speclim(A, M, 1.0, 0.0)
    r = rmin + (rmax - rmin) * rng.random()
    sqrtr = np.sqrt(abs(r))
    C = sqrtr * C
    K = np.sign(r) * sqrtr * K
    rhob = float(np.max(np.abs(np.linalg.eigvals(A - K @ C))))
    return A, C, K, rhob


def _speclim(
    A: npt.NDArray[np.floating], M: npt.NDArray[np.floating], r1: float, r2: float
) -> float:
    """Binary search for r ∈ (r1, r2) such that ρ(A - r M) just crosses 1."""

    def rho(r):
        return float(np.max(np.abs(np.linalg.eigvals(A - r * M))))

    assert rho(r1) > 1 and rho(r2) < 1
    while abs(r1 - r2) > np.finfo(float).eps:
        r = 0.5 * (r1 + r2)
        if rho(r) > 1:
            r1 = r
        else:
            r2 = r
    return r
