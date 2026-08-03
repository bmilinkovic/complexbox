"""Grassmannian-manifold utilities: orthonormalisation, principal angles,
distance metrics, Plücker coordinates, β-statistic, L↔Q involution map.

Ports of SSDI-1's ``utils/orthonormalise.m``, ``utils/rand_orthonormal.m``,
``utils/L2Q.m``, ``utils/Q2L.m``, ``metrics/subspacea.m``, ``metrics/gmetric.m``,
``metrics/gmetrics.m``, ``metrics/gmetricsxx.m``, ``metrics/habeta.m``,
``metrics/plucker.m``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import numpy.typing as npt

__all__ = [
    "orthonormalise",
    "rand_orthonormal",
    "subspacea",
    "subspaceb",
    "subspacec",
    "gmetric",
    "gmetrics",
    "gmetrics1",
    "gmetricsx",
    "gmetricsxx",
    "habeta",
    "habetax",
    "plucker",
    "L2Q",
    "Q2L",
    "rand_involution",
]


def orthonormalise(
    X: npt.NDArray[np.floating], return_complement: bool = False
) -> npt.NDArray[np.floating] | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """SVD-based orthonormal basis for the range of ``X``.

    Port of ``orthonormalise.m``. ``L = orthonormalise(X)`` returns ``L`` with
    ``L.T @ L = I``. If ``return_complement=True``, also returns an
    orthonormal basis ``M`` for the complement subspace.
    """
    X = np.asarray(X, dtype=float)
    if not return_complement:
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        return U
    n, m = X.shape
    U, _, _ = np.linalg.svd(X, full_matrices=True)
    return U[:, :m], U[:, m:]


def rand_orthonormal(
    n: int,
    m: int,
    runs: int = 1,
    rng: np.random.Generator | None = None,
    return_complement: bool = False,
) -> npt.NDArray[np.floating] | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return ``runs`` random ``n × m`` orthonormal bases (and optionally
    their ``n × (n - m)`` complements).

    Port of ``rand_orthonormal.m``. Output shape: ``(n, m, runs)`` if
    ``runs > 1`` else ``(n, m)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.standard_normal((n, m, runs))
    L = np.zeros((n, m, runs))
    if return_complement:
        M = np.zeros((n, n - m, runs))
        for k in range(runs):
            L[:, :, k], M[:, :, k] = orthonormalise(raw[:, :, k], return_complement=True)
        if runs == 1:
            return L[:, :, 0], M[:, :, 0]
        return L, M
    for k in range(runs):
        L[:, :, k] = orthonormalise(raw[:, :, k])
    if runs == 1:
        return L[:, :, 0]
    return L


def subspacea(F: npt.NDArray[np.floating], G: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Principal angles between two subspaces (no metric).

    Port of ``subspacea.m`` (Knyazev-Argentati 2002), with the standard
    A = I inner product. The full A-based variant is omitted; SSDI-1 always
    calls subspacea with two arguments. See ``THIRD_PARTY_LICENSES.md`` for
    its BSD notices.

    Returns
    -------
    theta : array of principal angles in [0, π/2] sorted in arbitrary order.
    """
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    if F.shape[0] != G.shape[0]:
        raise ValueError("subspaces must lie in matching ambient dimensions")

    # Column scaling for numerical stability
    eps = np.finfo(float).eps
    thresh_scale = eps**0.981
    F = F.copy()
    G = G.copy()
    for i in range(F.shape[1]):
        norm = np.max(np.abs(F[:, i]))
        if norm > thresh_scale:
            F[:, i] = F[:, i] / norm
    for i in range(G.shape[1]):
        norm = np.max(np.abs(G[:, i]))
        if norm > thresh_scale:
            G[:, i] = G[:, i] / norm

    QF = orthonormalise(F)
    QG = orthonormalise(G)
    q = min(QF.shape[1], QG.shape[1])

    Ys, s, _Zs = np.linalg.svd(QF.T @ QG, full_matrices=False)
    s = np.minimum(s, 1.0)
    theta = np.maximum(np.arccos(s), 0.0)
    threshold = np.sqrt(2) / 2
    indexsmall = s > threshold
    if np.any(indexsmall):
        RG = (QG @ _Zs)[:, indexsmall]
        _Yx, x, _Zx = np.linalg.svd(RG - QF @ (QF.T @ RG), full_matrices=False)
        thetasmall = np.maximum(np.arcsin(np.minimum(x, 1.0)), 0.0)
        thetasmall = thetasmall[::-1]
        theta_idx = np.where(indexsmall)[0]
        for j, ti in enumerate(theta_idx):
            if j < thetasmall.size:
                theta[ti] = thetasmall[j]
    return theta[:q]


def subspaceb(F: npt.NDArray[np.floating], G: npt.NDArray[np.floating]) -> float:
    """Largest principal angle via the simpler eigenvalue formulation.

    Port of ``metrics/subspaceb.m``. Computes ``acos(sqrt(λ_min(P^T P)))``
    where ``P = F^T G`` (or ``G^T F`` if F has more columns). Less accurate
    than :func:`subspacea` for small angles but cheap.
    """
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    if F.shape[1] < G.shape[1]:
        P = G.T @ F
    else:
        P = F.T @ G
    eigvals = np.linalg.eigvalsh(P.T @ P)
    return float(np.arccos(np.sqrt(max(eigvals.min(), 0.0))))


def subspacec(F: npt.NDArray[np.floating], G: npt.NDArray[np.floating]) -> float:
    """Largest principal angle via the residual-norm formulation.

    Port of ``metrics/subspacec.m``. Computes ``asin(||F - G G^T F||)`` (or
    the symmetric form when ``G`` is taller). More accurate than
    :func:`subspaceb` for angles close to 0 or π/2.
    """
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    if F.shape[1] < G.shape[1]:
        val = float(np.linalg.norm(F - G @ (G.T @ F)))
    else:
        val = float(np.linalg.norm(G - F @ (F.T @ G)))
    return float(np.arcsin(min(val, 1.0)))


def rand_involution(
    n: int,
    m: int,
    runs: int = 1,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating]:
    """Sample random ``(n × n)`` involutions of signature ``(m, n - m)``.

    Port of ``utils/rand_involution.m``. Constructs ``Q = V J V^T`` where
    ``V`` is a random orthonormal matrix and ``J = diag(1×m, -1×(n-m))``.
    Useful as an alternative parameterisation of Grassmannian points
    (compare :func:`L2Q`).
    """
    if rng is None:
        rng = np.random.default_rng()
    J = np.eye(n)
    J[m:, m:] = -np.eye(n - m)
    Q = np.empty((n, n, runs))
    for k in range(runs):
        V = orthonormalise(rng.standard_normal((n, n)))
        Q[:, :, k] = V @ J @ V.T
    if runs == 1:
        return Q[:, :, 0]
    return Q


def gmetric(
    L1: npt.NDArray[np.floating],
    L2: npt.NDArray[np.floating],
    max_angle: bool = True,
) -> float:
    """Grassmannian distance ∈ [0, 1] between two subspaces.

    Port of ``gmetric.m``. If ``max_angle=True`` returns the largest
    principal angle / (π/2); else the RMS / (π/2).
    """
    theta = subspacea(L1, L2)
    if max_angle:
        return float(np.max(theta) / (np.pi / 2))
    return float(np.sqrt(np.mean(theta**2)) / (np.pi / 2))


def gmetrics(L: npt.NDArray[np.floating], max_angle: bool = True) -> npt.NDArray[np.floating]:
    """Pairwise Grassmannian distance matrix.

    Port of ``gmetrics.m``. ``L`` has shape ``(n, m, R)`` for ``R`` subspaces.
    Returns symmetric matrix of shape ``(R, R)`` with zero diagonal.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim == 2:
        L = L[:, :, None]
    R = L.shape[2]
    D = np.zeros((R, R))
    for i in range(R):
        for j in range(i + 1, R):
            D[i, j] = D[j, i] = gmetric(L[:, :, i], L[:, :, j], max_angle=max_angle)
    return D


def gmetrics1(L: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Per-row angles of orthonormal basis ``L`` against the coordinate axes.

    Port of ``metrics/gmetrics1.m``. ``d_i = acos(sqrt(sum_j L_ij^2)) / (π/2)``,
    normalised to [0, 1]. Each row of ``L`` is one channel; the result tells
    you how *close* each channel is to lying in the subspace spanned by
    ``L``.
    """
    L = np.asarray(L, dtype=float)
    return np.arccos(np.sqrt(np.sum(L * L, axis=1))) / (np.pi / 2)


def gmetricsx(L: npt.NDArray[np.floating], max_angle: bool = True) -> npt.NDArray[np.floating]:
    """Distance from ``L`` to each *coordinate-axis* (1-D) subspace.

    Port of ``metrics/gmetricsx.m``. Returns an ``(n,)`` vector of normalised
    Grassmannian distances.
    """
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    d = np.empty(n)
    for i in range(n):
        v = np.zeros((n, 1))
        v[i, 0] = 1.0
        d[i] = gmetric(L, v, max_angle=max_angle)
    return d


def gmetricsxx(L: npt.NDArray[np.floating], max_angle: bool = True) -> npt.NDArray[np.floating]:
    """Grassmannian distance from each L to every coordinate-axis subspace.

    Port of ``gmetricsxx.m``. Returns array of shape ``(R, n_choose_m)``.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim == 2:
        L = L[:, :, None]
    n, m, R = L.shape
    combos = list(combinations(range(n), m))
    D = np.zeros((R, len(combos)))
    for r in range(R):
        for c, idx in enumerate(combos):
            E = np.zeros((n, m))
            for j, ax in enumerate(idx):
                E[ax, j] = 1.0
            D[r, c] = gmetric(L[:, :, r], E, max_angle=max_angle)
    return D


def habeta(L: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """β-statistic per channel: squared cosines of angles with coord axes.

    Port of ``habeta.m``. Returns ``sum(L**2, axis=1)``, shape ``(n,)``.
    Under the null of a uniformly-random hyperplane on the Grassmannian
    ``G(n, m)``, each entry is marginally ``Beta(m/2, (n - m)/2)`` (though
    the n entries are not independent — they sum to ``m``).
    """
    L = np.asarray(L, dtype=float)
    return np.sum(L * L, axis=1)


def habetax(
    n: int,
    m1: int,
    m2: int,
    N: int = 100_000,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.floating], tuple[float, float]]:
    """Monte-Carlo β-statistic distribution for a *nested* axis-subspace test.

    Port of ``metrics/habetax.m``. Samples ``N`` random orthonormal bases of
    dimension ``m1`` in ambient dimension ``n``, computes the smallest
    eigenvalue of ``L[:m2].T @ L[:m2]`` for each, and returns the empirical
    sample plus a Beta(α, β) maximum-likelihood fit.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (m1 > 0 and m2 < n):
        raise ValueError("require m1 > 0 and m2 < n")
    if m1 > m2:
        raise ValueError("m1 must be <= m2")
    samples = np.empty(N)
    raw = rng.standard_normal((n, m1, N))
    for k in range(N):
        L_k = orthonormalise(raw[:, :, k])
        P = L_k[:m2]
        samples[k] = float(np.linalg.eigvalsh(P.T @ P).min())
    from scipy.stats import beta as _beta

    a, b, _loc, _sc = _beta.fit(samples, floc=0.0, fscale=1.0)
    return samples, (float(a), float(b))


def plucker(L: npt.NDArray[np.floating], normalise: bool = True) -> npt.NDArray[np.floating]:
    """Plücker embedding of an orthonormal basis. Port of ``plucker.m``.

    Returns a vector of length ``nCm`` of m×m sub-determinants.
    """
    L = np.asarray(L, dtype=float)
    n, m = L.shape
    combos = list(combinations(range(n), m))
    p = np.array([np.linalg.det(L[list(c), :]) for c in combos])
    if normalise:
        p = p / np.sqrt(np.sum(p * p))
    return p


def L2Q(L: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Convert orthonormal basis L to involution ``Q = 2 L L' - I``.

    Port of ``L2Q.m`` (Lai-Lim-Ye 2020). Accepts ``L`` of shape ``(n, m)`` or
    ``(n, m, R)``.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim == 2:
        n = L.shape[0]
        return 2.0 * L @ L.T - np.eye(n)
    n, m, R = L.shape
    Q = np.empty((n, n, R))
    for k in range(R):
        Q[:, :, k] = 2.0 * L[:, :, k] @ L[:, :, k].T - np.eye(n)
    return Q


def Q2L(Q: npt.NDArray[np.floating], m: int) -> npt.NDArray[np.floating]:
    """Inverse of :func:`L2Q`: extract an orthonormal basis from involution.

    Port of ``Q2L.m``.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 2:
        n = Q.shape[0]
        U, _, _ = np.linalg.svd((np.eye(n) + Q) / 2.0)
        return U[:, :m]
    n, _, R = Q.shape
    L = np.empty((n, m, R))
    for k in range(R):
        U, _, _ = np.linalg.svd((np.eye(n) + Q[:, :, k]) / 2.0)
        L[:, :, k] = U[:, :m]
    return L
