"""Sample autocovariance estimation from time-series data.

Port of MVGC2's ``core/tsdata_to_autocov.m`` and inverse FFT helpers.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._utils import demean

__all__ = ["tsdata_to_autocov", "tsdata_to_cov", "cpsd_to_autocov", "autocov_to_cpsd"]


def tsdata_to_autocov(X: npt.NDArray[np.floating], q: int) -> npt.NDArray[np.floating]:
    """Biased sample autocovariance sequence Γ_0, ..., Γ_q.

    Port of ``core/tsdata_to_autocov.m``. Returns array of shape
    ``(n, n, q + 1)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape
    X = demean(X)
    G = np.zeros((n, n, q + 1))
    for k in range(q + 1):
        for trial in range(N):
            Xt = X[:, k:, trial]
            Xt_lag = X[:, : m - k, trial]
            G[:, :, k] += Xt @ Xt_lag.T
    G /= N * m
    return G


def tsdata_to_cov(X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Sample covariance Γ_0 only."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape
    X = demean(X).reshape(n, m * N)
    return (X @ X.T) / (m * N - 1)


def cpsd_to_autocov(
    S: npt.NDArray[np.complexfloating], q: int | None = None
) -> npt.NDArray[np.floating]:
    """Inverse FFT of a one-sided CPSD into autocovariance lags.

    Mirror image of MVGC2's ``cpsd_to_autocov.m``.
    """
    S = np.asarray(S)
    n = S.shape[0]
    h = S.shape[2]
    nfft = 2 * (h - 1)
    # Build two-sided spectrum by mirroring
    S_full = np.empty((n, n, nfft), dtype=complex)
    S_full[:, :, :h] = S
    S_full[:, :, h:] = np.conj(S[:, :, 1 : h - 1][..., ::-1])
    G_full = np.real(np.fft.ifft(S_full, axis=-1))
    if q is None:
        q = nfft // 2 - 1
    return G_full[:, :, : q + 1].copy()


def autocov_to_cpsd(G: npt.NDArray[np.floating], fres: int) -> npt.NDArray[np.complexfloating]:
    """Forward FFT of an autocovariance sequence into a one-sided CPSD."""
    G = np.asarray(G, dtype=float)
    n, _, q1 = G.shape
    nfft = 2 * fres
    if q1 > nfft:
        raise ValueError(f"autocov length ({q1}) exceeds NFFT ({nfft})")
    Gpad = np.zeros((n, n, nfft))
    Gpad[:, :, :q1] = G
    Gpad[:, :, nfft - (q1 - 1) :] = G[:, :, 1:][..., ::-1]
    S_full = np.fft.fft(Gpad, axis=-1)
    return S_full[:, :, : fres + 1]
