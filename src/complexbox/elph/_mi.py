"""Closed-form and plug-in mutual-information estimators.

Pure-Python replacements for ELPH's JIDT-backed Gaussian and discrete MI
calculators (``private/GaussianMI.m`` and ``private/DiscreteMI.m``), plus the
ELPH ``private/gauss_mi.m`` helper.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..mvgc._utils import logdet

__all__ = [
    "gaussian_entropy",
    "gaussian_mi",
    "gaussian_local_mi",
    "discrete_mi",
    "discrete_entropy",
]


def gaussian_entropy(cov: npt.NDArray[np.floating]) -> float:
    """Differential entropy of a multivariate Gaussian.

    ``H(X) = 0.5 * (k log(2π e) + log|Σ|)`` in nats. Mirrors ELPH's inline
    helper ``h(C)`` used throughout the toolbox.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        return 0.5 * (np.log(2 * np.pi * np.e) + np.log(float(cov)))
    k = cov.shape[0]
    return 0.5 * (k * np.log(2 * np.pi * np.e) + logdet(cov))


def gaussian_mi(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    base: float = np.e,
) -> float:
    """Mutual information between two jointly-Gaussian variables.

    Port of ELPH's ``gauss_mi.m`` (and the JIDT Gaussian MI calculator
    fallback). ``X``, ``Y`` are ``(d_X, T)`` and ``(d_Y, T)`` arrays. Returns
    MI in nats by default; pass ``base=2`` for bits.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of samples")
    Z = np.vstack([X, Y])
    SX = np.cov(X, ddof=1) if X.shape[0] > 1 else np.array([[float(np.var(X, ddof=1))]])
    SY = np.cov(Y, ddof=1) if Y.shape[0] > 1 else np.array([[float(np.var(Y, ddof=1))]])
    SZ = np.cov(Z, ddof=1)
    mi = 0.5 * (logdet(SX) + logdet(SY) - logdet(SZ))
    return float(mi / np.log(base))


def gaussian_local_mi(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Pointwise local Gaussian MI per sample (length ``T``).

    Returns ``i(x_t, y_t) = log p(x_t, y_t) / (p(x_t) p(y_t))`` for each ``t``.
    Used by PhiID and the emergence calculators.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    T = X.shape[1]
    Z = np.vstack([X, Y])
    mu = Z.mean(axis=1, keepdims=True)
    Zc = Z - mu
    Sxx = np.cov(X, ddof=1) if X.shape[0] > 1 else np.array([[float(np.var(X, ddof=1))]])
    Syy = np.cov(Y, ddof=1) if Y.shape[0] > 1 else np.array([[float(np.var(Y, ddof=1))]])
    Szz = np.cov(Z, ddof=1)

    def _logpdf(samples, S):
        samples = np.atleast_2d(samples)
        if S.ndim == 0 or (S.ndim == 2 and S.shape == (1, 1)):
            v = float(S) if S.ndim == 0 else float(S[0, 0])
            sq = samples[0] ** 2 if samples.shape[0] == 1 else (samples**2).sum(axis=0)
            return -0.5 * np.log(2 * np.pi * v) - 0.5 * sq / v
        d = S.shape[0]
        try:
            Sinv = np.linalg.inv(S)
            mahal = np.einsum("ij,jt,it->t", Sinv, samples, samples)
            return -0.5 * d * np.log(2 * np.pi) - 0.5 * logdet(S) - 0.5 * mahal
        except np.linalg.LinAlgError:
            return np.full(T, np.nan)

    lp_x = _logpdf(Zc[: X.shape[0]], Sxx)
    lp_y = _logpdf(Zc[X.shape[0] :], Syy)
    lp_z = _logpdf(Zc, Szz)
    return lp_z - lp_x - lp_y


def discrete_entropy(X: npt.NDArray, base: float = np.e) -> float:
    """Plug-in (empirical) entropy of discrete data.

    Treats columns of ``X`` (shape ``(T,)`` or ``(D, T)``) as joint symbols.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        _, counts = np.unique(X, return_counts=True)
    else:
        # combine D rows into a single symbol per column via lexicographic hash
        view = np.ascontiguousarray(X.T)
        _, counts = np.unique(view, axis=0, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(base))


def discrete_mi(
    X: npt.NDArray,
    Y: npt.NDArray,
    base: float = np.e,
) -> float:
    """Plug-in MI between discrete random variables.

    ``I(X; Y) = H(X) + H(Y) - H(X, Y)``.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    return (
        discrete_entropy(X, base=base)
        + discrete_entropy(Y, base=base)
        - discrete_entropy(
            np.vstack([np.atleast_2d(X), np.atleast_2d(Y)]),
            base=base,
        )
    )
