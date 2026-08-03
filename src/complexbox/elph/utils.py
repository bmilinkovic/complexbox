"""Graph and discretisation helpers.

Ports of ELPH's ``elph_base/`` utilities: ``MaximalCliques.m``,
``CovarianceSelectionModel.m``, ``QuasiBayesMI.m``, ``TransferEntropy.m``,
``isdiscrete.m``, ``discretize_oct.m``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._mi import gaussian_entropy

__all__ = [
    "isdiscrete",
    "discretize_quantile",
    "maximal_cliques",
    "covariance_selection_model",
    "quasi_bayes_mi",
    "transfer_entropy_gaussian",
]


def isdiscrete(X: npt.NDArray, tol: float = 1e-10) -> bool:
    """True if ``X`` is integer-, logical-, or float-within-tolerance integer-valued.

    Port of ``isdiscrete.m``.
    """
    X = np.asarray(X)
    if X.dtype.kind in {"b", "i", "u"}:
        return True
    if X.dtype.kind == "f":
        return bool(np.all(np.abs(X - np.round(X)) < tol))
    return False


def discretize_quantile(
    x: npt.NDArray[np.floating], n: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """Quantile-based discretisation into ``n`` equiprobable bins.

    Port of ``discretize_oct.m``. Returns ``(bins, edges)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    edges = np.quantile(x, np.linspace(0, 1, n + 1))
    bins = np.digitize(x, edges[1:-1], right=False).astype(np.intp)
    bins = np.clip(bins, 0, n - 1)
    return bins, edges


def maximal_cliques(adj: npt.NDArray[np.intp]) -> list[list[int]]:
    """Maximal cliques of an undirected graph via Bron-Kerbosch (pivot variant).

    Port of Jeffrey Wildman's ``MaximalCliques.m`` (v2). Input is an ``n × n``
    boolean / 0-1 adjacency matrix (undirected, no self-loops). Returns a list
    of lists of node indices. See ``THIRD_PARTY_LICENSES.md`` for its BSD
    notice.
    """
    adj = np.asarray(adj, dtype=bool).copy()
    n = adj.shape[0]
    np.fill_diagonal(adj, False)
    adj |= adj.T  # undirected

    neighbours = [set(np.where(adj[i])[0].tolist()) for i in range(n)]

    cliques: list[list[int]] = []

    def bron_kerbosch(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            cliques.append(sorted(R))
            return
        # Pivot: choose u in P∪X maximising |P ∩ N(u)|
        pivot = max(P | X, key=lambda u: len(P & neighbours[u]))
        for v in list(P - neighbours[pivot]):
            bron_kerbosch(
                R | {v},
                P & neighbours[v],
                X & neighbours[v],
            )
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())
    return cliques


def covariance_selection_model(
    Sigma: npt.NDArray[np.floating],
    A: npt.NDArray[np.intp],
    max_iters: int = 5000,
    tol: float = 1e-4,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fit a sparse covariance with prescribed zero-pattern (graphical Gaussian).

    Port of ``CovarianceSelectionModel.m`` (Speed-Kiiveri 1986 cyclic
    coordinate descent). Returns ``(K, K_inv)`` where ``K`` is the
    structured covariance matching ``Sigma`` on the support of ``A`` and
    ``K_inv`` has zeros off-support.

    Parameters
    ----------
    Sigma : symmetric positive-definite covariance to be approximated
    A : boolean adjacency (1 = edge / variable dependency, 0 = forced zero in
        the precision matrix). Diagonal is forced to True.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    A = np.asarray(A, dtype=bool).copy()
    n = Sigma.shape[0]
    np.fill_diagonal(A, True)
    # Initialise with identity covariance scaled to match Sigma diagonal
    K = np.diag(np.diag(Sigma))
    K_inv = np.linalg.inv(K)
    cliques = maximal_cliques(A.astype(np.intp))
    if not cliques:
        return K, K_inv
    for _ in range(max_iters):
        K_prev = K.copy()
        for cl in cliques:
            idx = np.array(cl)
            comp = np.array([i for i in range(n) if i not in cl])
            # Solve so that K[idx, idx] = Sigma[idx, idx] and K_inv off-cluster
            # rows/cols are unchanged. Standard IPS update:
            S_cc = Sigma[np.ix_(idx, idx)]
            if comp.size == 0:
                K[np.ix_(idx, idx)] = S_cc
                K_inv[np.ix_(idx, idx)] = np.linalg.inv(S_cc)
                continue
            K_inv_co = K_inv[np.ix_(idx, comp)]
            K_oo = K[np.ix_(comp, comp)]
            try:
                inv_S_cc = np.linalg.inv(S_cc)
            except np.linalg.LinAlgError:
                continue
            K_inv_new_cc = inv_S_cc + K_inv_co @ K_oo @ K_inv_co.T
            K_inv[np.ix_(idx, idx)] = K_inv_new_cc
        K = np.linalg.inv(K_inv)
        delta = np.max(np.abs(K - K_prev))
        if delta < tol:
            break
    return K, K_inv


def quasi_bayes_mi(X: npt.NDArray, Y: npt.NDArray, base: float = 2.0) -> float:
    """Quasi-Bayesian mutual information via Miller-Madow correction.

    Approximation to ELPH's ``QuasiBayesMI.m`` (which uses the NSB estimator
    via a Java library). The Miller-Madow correction
    ``H_MM = H_plug-in + (K - 1) / (2 N ln 2)`` is a fast, well-known
    quasi-Bayesian fix for the plug-in bias. For NSB itself, install
    ``ndd`` (https://github.com/simomarsili/ndd) and replace this function.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    N = X.shape[-1] if X.ndim > 0 else X.size

    def _miller_madow_entropy(arr):
        if arr.ndim == 1:
            _, counts = np.unique(arr, return_counts=True)
        else:
            view = np.ascontiguousarray(arr.T)
            _, counts = np.unique(view, axis=0, return_counts=True)
        K = counts.size
        p = counts / counts.sum()
        H = -np.sum(p * np.log(p))
        correction = (K - 1) / (2.0 * N)
        return (H + correction) / np.log(base)

    return float(
        _miller_madow_entropy(X)
        + _miller_madow_entropy(Y)
        - _miller_madow_entropy(np.vstack([np.atleast_2d(X), np.atleast_2d(Y)]))
    )


def transfer_entropy_gaussian(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    tau: int = 1,
) -> float:
    """Gaussian transfer entropy from X to Y at lag ``tau``.

    Closed-form replacement for ELPH's JIDT-backed ``TransferEntropy.m``::

        TE(X → Y) = I(Y_t ; X_{t-τ} | Y_{t-τ})
                  = H(Y_t, Y_{t-τ}) + H(X_{t-τ}, Y_{t-τ})
                    - H(Y_{t-τ}) - H(Y_t, X_{t-τ}, Y_{t-τ})

    For multivariate ``X`` (shape ``(D_X, T)``) or ``Y`` (shape ``(D_Y, T)``)
    we compute the joint Gaussian TE using the covariance of the stacked
    past-target/past-source/future-target block.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of samples")
    T = X.shape[1]
    if tau >= T:
        raise ValueError("tau must be smaller than the time-series length")
    Yt = Y[:, tau:]  # future target
    Yp = Y[:, :-tau]  # past target
    Xp = X[:, :-tau]  # past source

    def H(*blocks):
        Z = np.vstack(blocks)
        return gaussian_entropy(np.cov(Z, ddof=1))

    return float(H(Yt, Yp) + H(Xp, Yp) - H(Yp) - H(Yt, Xp, Yp))
