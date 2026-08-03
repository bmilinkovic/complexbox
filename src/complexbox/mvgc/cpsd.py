"""Cross-power spectral density estimation from time series (Welch method).

Port of MVGC2's ``core/tsdata_to_cpsd.m``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal

__all__ = ["tsdata_to_cpsd"]


def tsdata_to_cpsd(
    X: npt.NDArray[np.floating],
    fs: float = 1.0,
    window: int | npt.NDArray | None = None,
    overlap: float | None = None,
    fres: int | None = None,
    return_onesided: bool = True,
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.floating]]:
    """Estimate CPSD by averaging across trials and Welch segments.

    Parameters
    ----------
    X : (n, m) or (n, m, N) array
    fs : sampling frequency in Hz
    window : segment length (int) or window array (default: ``min(256, m//8)``)
    overlap : fractional overlap in [0, 1) (default: 0.5)
    fres : if provided, NFFT = ``2 * fres``; output spectrum has ``fres + 1``
        bins from 0 to Nyquist.
    return_onesided : keep only positive frequencies

    Returns
    -------
    S : (n, n, h) complex array of cross-spectra
    f : (h,) frequency vector in Hz
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape

    if window is None:
        wlen = min(256, max(8, m // 8))
    elif np.isscalar(window):
        wlen = int(window)
    else:
        wlen = int(len(window))
    if overlap is None:
        overlap = 0.5
    noverlap = int(np.floor(wlen * overlap))
    if fres is not None:
        nfft = 2 * fres
    else:
        nfft = max(wlen, 256)
    if np.isscalar(window) or window is None:
        win = signal.windows.hann(wlen)
    else:
        win = np.asarray(window, dtype=float)

    # Concatenate trials along time
    Xc = X.reshape(n, m * N)
    f, Pxy = signal.csd(
        Xc[:, None, :],
        Xc[None, :, :],
        fs=fs,
        window=win,
        nperseg=wlen,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=return_onesided,
        scaling="density",
        axis=-1,
    )
    # csd broadcasting may not be supported on all SciPy versions; fall back to
    # the manual loop for correctness:
    S = np.empty((n, n, f.size), dtype=complex)
    for i in range(n):
        for j in range(n):
            _, Sij = signal.csd(
                Xc[i],
                Xc[j],
                fs=fs,
                window=win,
                nperseg=wlen,
                noverlap=noverlap,
                nfft=nfft,
                return_onesided=return_onesided,
                scaling="density",
            )
            S[i, j, :] = Sij
    return S, f
