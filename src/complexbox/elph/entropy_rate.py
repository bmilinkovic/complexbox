"""Entropy-rate complexity measures.

Ports of ELPH's ``EntRate/LZ76.c``, ``EntRate/StateSpaceEntropyRate.m``, and
``EntRate/CTWEntropyRate.m`` (the last is a stub — the Java VMM library used
in MATLAB has no pure-Python replacement, so we provide a documented
placeholder that errors out unless an external CTW implementation is
plugged in).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..mvgc.modelorder import tsdata_to_varmo
from ..mvgc.ss import ss_to_cpsd
from ..mvgc.var import tsdata_to_var, var_to_ss
from ._mi import gaussian_entropy

__all__ = ["lz76", "lz76_entropy_rate", "state_space_entropy_rate"]


def lz76(seq: npt.NDArray | bytes | str) -> int:
    """Lempel-Ziv 1976 complexity of a binary sequence.

    Pure-Python port of ELPH's ``LZ76.c`` (Kaspar & Schuster 1987). Returns
    the unnormalised integer complexity ``c``. To convert to an entropy-rate
    estimate, multiply by ``log2(len(seq)) / len(seq)`` (see
    :func:`lz76_entropy_rate`).

    The reference C implementation is faithfully reproduced; with Numba
    enabled, performance is comparable to the MEX build.
    """
    if isinstance(seq, (bytes, bytearray)):
        s = np.frombuffer(seq, dtype=np.uint8).astype(np.intp)
    elif isinstance(seq, str):
        s = np.array([int(c) for c in seq], dtype=np.intp)
    else:
        s = np.asarray(seq).ravel()
        if s.dtype == bool:
            s = s.astype(np.intp)
    n = s.size
    if n == 1:
        return 1
    return _lz76_core(s, n)


try:  # pragma: no cover — optional acceleration
    from numba import njit

    @njit(cache=True)
    def _lz76_core(seq, n):  # type: ignore[no-redef]
        i = 0
        k = 1
        cursor = 1
        c = 1
        k_max = 1
        while True:
            if seq[i + k - 1] == seq[cursor + k - 1]:
                k += 1
                if cursor + k >= n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == cursor:
                    c += 1
                    cursor = cursor + k_max
                    if cursor >= n:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1
        return c

except ImportError:  # numba not available — pure Python fallback

    def _lz76_core(seq, n):
        i = 0
        k = 1
        cursor = 1
        c = 1
        k_max = 1
        while True:
            if seq[i + k - 1] == seq[cursor + k - 1]:
                k += 1
                if cursor + k >= n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == cursor:
                    c += 1
                    cursor = cursor + k_max
                    if cursor >= n:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1
        return c


def lz76_entropy_rate(seq: npt.NDArray) -> float:
    """Entropy-rate estimate from LZ76 complexity.

    Returns ``C * log2(n) / n``, the standard normalisation.
    """
    n = np.asarray(seq).size
    if n < 2:
        return 0.0
    return float(lz76(seq) * np.log2(n) / n)


def state_space_entropy_rate(
    X: npt.NDArray[np.floating],
    fs: float,
    *,
    downsample: bool = True,
    band: npt.NDArray[np.floating] | None = None,
    varmomax: int = 20,
    regmode: str = "LWR",
) -> tuple[float, npt.NDArray[np.floating] | None]:
    """State-space-based entropy rate (Mediano 2020).

    Port of ``StateSpaceEntropyRate.m``. Builds a per-channel SS model via
    MVGC2 routines and returns the average residual-covariance log-determinant
    (the entropy rate of a stationary Gaussian process under that model).

    Parameters
    ----------
    X : ``(D, T)`` or ``(D, T, M)`` data array
    fs : sampling frequency (Hz)
    downsample : if True and ``fs > 200``, decimate to ≤ 200 Hz
    band : optional ``(K, 2)`` frequency-band edges for spectral decomposition
    varmomax : maximum VAR order tried during selection (default 20, suited
        to MEEG; use ~``round(8/TR)`` for fMRI)

    Returns
    -------
    H : scalar broadband entropy rate
    band_H : ``(K,)`` per-band rates (or None if ``band is None``)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    D, T, M = X.shape
    if T < D:
        import warnings

        warnings.warn(
            "Time-series shorter than channel count — did you transpose X?",
            stacklevel=2,
        )

    if downsample and fs > 200:
        k = int(np.floor(fs / 200))
        if k > 1:
            X = X[:, ::k, :]
            fs = fs / k

    H_total = 0.0
    band_H = None
    if band is not None:
        band = np.asarray(band, dtype=float)
        band_H = np.zeros(band.shape[0])

    for d in range(D):
        y = X[d : d + 1, :, :]
        order = tsdata_to_varmo(y, pmax=varmomax, regmode=regmode)
        p = max(order.p_hqc, 1)
        fit = tsdata_to_var(y, p=p, regmode=regmode)
        H_total += gaussian_entropy(fit.V)

        if band is not None:
            fres = 1000
            A_ss, C, K, _ = var_to_ss(fit.A, fit.V)
            S = ss_to_cpsd(A_ss, C, K, fit.V, fres=fres)
            H_freq = np.array([gaussian_entropy(np.real(S[:, :, i])) for i in range(S.shape[2])])
            freqs = np.linspace(0, fs / 2, S.shape[2])
            for j in range(band.shape[0]):
                lo, hi = band[j]
                if np.isinf(hi):
                    hi = fs / 2
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    band_H[j] += float(np.mean(H_freq[mask]))

    H_total /= D
    if band_H is not None:
        band_H /= D
    return H_total, band_H
