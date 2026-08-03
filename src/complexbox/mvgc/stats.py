"""Statistical inference for Granger-causality estimates.

Ports of MVGC2's ``stats/`` directory: theoretical CDFs/p-values/critical
values under the F and likelihood-ratio chi-square null distributions, plus
the AIC/BIC/HQC information criteria, Durbin-Watson whiteness test, model
consistency, and significance corrections.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import stats as _stats

from ._utils import demean

__all__ = [
    "mvgc_cdf",
    "mvgc_icdf",
    "mvgc_pval",
    "mvgc_cval",
    "mvgc_bias",
    "ggc_cdf",
    "ggc_icdf",
    "ggc_pval",
    "ggc_cval",
    "ggc_bias",
    "infocrit",
    "whiteness",
    "consistency",
    "significance",
    "lbqtest",
    "lbqtest_pval",
    "lbqtest_cval",
    "mardia",
    "mann_whitney",
    "mann_whitney_group",
    "empirical_cdf",
    "empirical_cdfi",
    "empirical_pval",
    "empirical_cval",
    "empirical_confint",
    "tsdata_permute",
]


def mvgc_cdf(
    tstat: str,
    nx: int,
    ny: int,
    nz: int,
    p: int,
    m: int,
    N: int = 1,
) -> Callable[[npt.NDArray | float], npt.NDArray | float]:
    """Return a CDF function under the null for the GC test statistic.

    Port of ``stats/mvgc_cdf.m``.
    """
    if m <= p:
        raise ValueError("Insufficient observations for statistical tests")
    d = p * nx * ny  # degrees of freedom
    M = N * (m - p)
    if tstat.upper() == "F":
        n = nx + ny + nz
        pn = p * n
        if M <= pn:
            raise ValueError("Insufficient observations for F-test")
        d2 = nx * (M - pn)
        sf = d2 / d
        return lambda stat: _stats.f.cdf(sf * np.asarray(stat), d, d2)
    elif tstat.upper() == "LR":
        sf = M
        return lambda stat: _stats.chi2.cdf(sf * np.asarray(stat), d)
    else:
        raise ValueError(f"Unknown test statistic {tstat!r}")


def mvgc_pval(
    stat: npt.NDArray | float,
    tstat: str,
    nx: int,
    ny: int,
    nz: int,
    p: int,
    m: int,
    N: int = 1,
) -> npt.NDArray | float:
    """p-value for the GC test statistic. Port of ``stats/mvgc_pval.m``."""
    cdf = mvgc_cdf(tstat, nx, ny, nz, p, m, N)
    return 1.0 - cdf(stat)


def mvgc_icdf(
    tstat: str,
    nx: int,
    ny: int,
    nz: int,
    p: int,
    m: int,
    N: int = 1,
) -> Callable[[npt.NDArray | float], npt.NDArray | float]:
    """Inverse CDF (quantile function) for the GC test statistic.

    Port of ``stats/mvgc_icdf.m``.
    """
    if m <= p:
        raise ValueError("Insufficient observations for statistical tests")
    d = p * nx * ny
    M = N * (m - p)
    if tstat.upper() == "F":
        n = nx + ny + nz
        pn = p * n
        if M <= pn:
            raise ValueError("Insufficient observations for F-test")
        d2 = nx * (M - pn)
        sf = d2 / d
        return lambda prob: _stats.f.ppf(np.asarray(prob), d, d2) / sf
    if tstat.upper() == "LR":
        sf = M
        return lambda prob: _stats.chi2.ppf(np.asarray(prob), d) / sf
    raise ValueError(f"Unknown test statistic {tstat!r}")


def mvgc_cval(
    pcrit: float,
    tstat: str,
    nx: int,
    ny: int,
    nz: int,
    p: int,
    m: int,
    N: int = 1,
) -> float:
    """Critical value for the GC test statistic. Port of ``stats/mvgc_cval.m``."""
    d = p * nx * ny
    M = N * (m - p)
    if tstat.upper() == "F":
        n = nx + ny + nz
        d2 = nx * (M - p * n)
        sf = d2 / d
        return float(_stats.f.ppf(1.0 - pcrit, d, d2) / sf)
    elif tstat.upper() == "LR":
        sf = M
        return float(_stats.chi2.ppf(1.0 - pcrit, d) / sf)
    else:
        raise ValueError(f"Unknown test statistic {tstat!r}")


def mvgc_bias(
    tstat: str,
    nx: int,
    ny: int,
    nz: int,
    p: int,
    m: int,
    N: int = 1,
) -> float:
    """Asymptotic null-mean bias of the GC test statistic.

    Port of ``stats/mvgc_bias.m``. Useful as a debiased point estimator under
    the null.
    """
    d = p * nx * ny
    M = N * (m - p)
    if tstat.upper() == "F":
        n = nx + ny + nz
        d2 = nx * (M - p * n)
        if d2 <= 2:
            return float("nan")
        return float(d / (d2 - 2))
    elif tstat.upper() == "LR":
        return float(d / M)
    else:
        raise ValueError(f"Unknown test statistic {tstat!r}")


def infocrit(
    L: float | npt.NDArray,
    k: int | npt.NDArray,
    m: int | npt.NDArray,
    hurvich_tsai: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """AIC, BIC, HQC information criteria. Port of ``stats/infocrit.m``."""
    L = np.asarray(L, dtype=float)
    k = np.asarray(k, dtype=float)
    m = np.asarray(m, dtype=float)
    K = k / m
    if hurvich_tsai:
        fac = m / (m - k - 1)
        aic = -2.0 * L + 2.0 * K * fac
        aic = np.where(fac <= 0, np.nan, aic)
    else:
        aic = -2.0 * L + 2.0 * K
    bic = -2.0 * L + K * np.log(m)
    hqc = -2.0 * L + 2.0 * K * np.log(np.log(m))
    return aic, bic, hqc


def whiteness(
    X: npt.NDArray[np.floating], E: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Durbin-Watson test for residual whiteness, per variable.

    Port of ``stats/whiteness.m``.
    """
    X = np.asarray(X, dtype=float)
    E = np.asarray(E, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if E.ndim == 2:
        E = E[:, :, None]
    n, m, N = X.shape
    p = m - E.shape[1]
    if p <= 0:
        raise ValueError("bad number of lags inferred from residuals length")

    X = demean(X)
    M = N * (m - p)
    Xf = X[:, p:m, :].reshape(n, M)
    Ef = E.reshape(n, M)

    dw = np.empty(n)
    pval = np.empty(n)
    for i in range(n):
        dw[i], pval[i] = _durbin_watson(Xf, Ef[i])
    return dw, pval


def _durbin_watson(X, E):
    """Approximate Durbin-Watson statistic and p-value."""
    n, m = X.shape
    dw = float(np.sum(np.diff(E) ** 2) / np.sum(E**2))

    A = X @ X.T
    # B = filter([-1, 2, -1], 1, X')
    B = np.zeros_like(X.T)
    Bt = np.zeros_like(X.T)
    Xt = X.T
    Bt[0] = -Xt[0]
    if m > 1:
        Bt[1] = 2 * Xt[0] - Xt[1]
    for t in range(2, m):
        Bt[t] = -Xt[t - 2] + 2 * Xt[t - 1] - Xt[t]
    B = Bt
    # Boundary fix: B([1,m]) = (X(:,[1,m]) - X(:,[2,m-1]))'
    B[0] = X[:, 0] - X[:, 1]
    B[m - 1] = X[:, m - 1] - X[:, m - 2]
    # D = B / A
    D = np.linalg.solve(A.T, B.T).T
    C = X @ D
    nu1 = 2 * (m - 1) - np.trace(C)
    nu2 = 2 * (3 * m - 4) - 2 * np.trace(B.T @ D) + np.trace(C @ C)
    mu = nu1 / (m - n)
    sigma = np.sqrt(2.0 / ((m - n) * (m - n + 2)) * (nu2 - nu1 * mu))
    pval = _stats.norm.cdf(dw, loc=mu, scale=sigma)
    pval = 2.0 * min(pval, 1.0 - pval)
    return dw, pval


def consistency(X: npt.NDArray[np.floating], E: npt.NDArray[np.floating]) -> float:
    """Ding-Bressler-Yang-Liang consistency statistic.

    Port of ``stats/consistency.m``. Values > 0.8 indicate adequate fit.
    """
    X = np.asarray(X, dtype=float)
    E = np.asarray(E, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    if E.ndim == 2:
        E = E[:, :, None]
    n, m, N = X.shape
    p = m - E.shape[1]
    X = demean(X)
    M = N * (m - p)
    Xf = X[:, p:m, :].reshape(n, M)
    Ef = E.reshape(n, M)
    Y = Xf - Ef
    Rr = (Xf @ Xf.T) / (M - 1)
    Rs = (Y @ Y.T) / (M - 1)
    return float(1.0 - np.linalg.norm(Rs - Rr) / np.linalg.norm(Rr))


# ---------------------------------------------------------------------------
# Groupwise-GC F-distribution under the null
# ---------------------------------------------------------------------------


def ggc_cdf(
    tstat: str, nx: int, nz: int, p: int, m: int, N: int = 1
) -> Callable[[npt.NDArray | float], npt.NDArray | float]:
    """CDF for the conditional-GGC F-statistic. Port of ``stats/ggc_cdf.m``.

    Only the ``'F'`` test is defined for GGC. Degrees of freedom are
    ``d = p*nx*(nx-1)`` and ``d2 = nx*(M - p*(nx+nz))``.
    """
    if tstat.upper() != "F":
        raise ValueError("Only F-test available for GGC")
    if m <= p:
        raise ValueError("Insufficient observations for statistical test")
    M = N * (m - p)
    n = nx + nz
    pn = p * n
    if M <= pn:
        raise ValueError("Insufficient observations for F-test")
    d = p * nx * (nx - 1)
    d2 = nx * (M - pn)
    sf = d2 / d
    return lambda stat: _stats.f.cdf(sf * np.asarray(stat), d, d2)


def ggc_icdf(
    tstat: str, nx: int, nz: int, p: int, m: int, N: int = 1
) -> Callable[[npt.NDArray | float], npt.NDArray | float]:
    """Inverse CDF for the GGC F-statistic. Port of ``stats/ggc_icdf.m``."""
    if tstat.upper() != "F":
        raise ValueError("Only F-test available for GGC")
    M = N * (m - p)
    n = nx + nz
    d = p * nx * (nx - 1)
    d2 = nx * (M - p * n)
    sf = d2 / d
    return lambda prob: _stats.f.ppf(np.asarray(prob), d, d2) / sf


def ggc_pval(
    stat: npt.NDArray | float,
    tstat: str,
    nx: int,
    nz: int,
    p: int,
    m: int,
    N: int = 1,
) -> npt.NDArray | float:
    """p-value for the GGC F-statistic. Port of ``stats/ggc_pval.m``."""
    cdf = ggc_cdf(tstat, nx, nz, p, m, N)
    return 1.0 - cdf(stat)


def ggc_cval(pcrit: float, tstat: str, nx: int, nz: int, p: int, m: int, N: int = 1) -> float:
    """Critical value of the GGC F-statistic. Port of ``stats/ggc_cval.m``."""
    icdf = ggc_icdf(tstat, nx, nz, p, m, N)
    return float(icdf(1.0 - pcrit))


def ggc_bias(tstat: str, nx: int, nz: int, p: int, m: int, N: int = 1) -> float:
    """Asymptotic null-mean bias of the GGC F-statistic. Port of ``stats/ggc_bias.m``."""
    if tstat.upper() != "F":
        raise ValueError("Only F-test available for GGC")
    M = N * (m - p)
    n = nx + nz
    pn = p * n
    if M <= pn:
        raise ValueError("Insufficient observations for F-test")
    d = p * nx * (nx - 1)
    d2 = nx * (M - pn)
    if d2 <= 2:
        return float("nan")
    return float(d / (d2 - 2))


# ---------------------------------------------------------------------------
# Whiteness alternative: Ljung-Box-Q
# ---------------------------------------------------------------------------


def lbqtest(
    X: npt.NDArray[np.floating],
    p: int,
    hmax: int,
    standardised: bool = False,
) -> npt.NDArray[np.floating]:
    """Cumulative multivariate Ljung-Box-Q autocorrelation statistic.

    Port of ``stats/lbqtest.m``. Returns the Q statistic for each lag
    ``h = 1, ..., hmax``; entries with ``h <= p`` are NaN.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape
    if hmax <= p:
        raise ValueError("hmax must exceed model order p")
    if hmax >= m:
        raise ValueError("hmax must be smaller than the number of observations")
    T = N * m
    # Lag-0 covariance (averaged over trials)
    C = np.zeros((n, n))
    for i in range(N):
        C = C + (X[:, :, i] @ X[:, :, i].T) / m
    C = C / N
    inverse_cov = np.linalg.inv(C)
    Q = np.zeros(hmax)
    for h in range(1, hmax + 1):
        Ch = np.zeros((n, n))
        for i in range(N):
            Ch = Ch + (X[:, h:, i] @ X[:, : m - h, i].T) / m
        Ch = Ch / N
        Q[h - 1] = np.trace(Ch @ inverse_cov @ Ch.T @ inverse_cov) / (T - h)
    Q = T * (T + 2) * np.cumsum(Q)
    Q[:p] = np.nan
    if standardised:
        df = (n * n) * np.arange(1, hmax - p + 1)
        Q[p:] = (Q[p:] - df) / np.sqrt(2 * df)
    return Q


def lbqtest_pval(
    Q: npt.NDArray[np.floating],
    n: int,
    p: int,
    h: npt.NDArray[np.intp] | int,
) -> npt.NDArray[np.floating]:
    """p-value for the Ljung-Box-Q statistic. Port of ``stats/lbqtest_pval.m``."""
    Q = np.asarray(Q, dtype=float)
    h_arr = np.atleast_1d(np.asarray(h, dtype=float))
    mask = h_arr > p
    pval = np.full(h_arr.shape, np.nan)
    dof = (n * n) * (h_arr - p)
    pval[mask] = 1.0 - _stats.chi2.cdf(Q[mask], dof[mask])
    return pval


def lbqtest_cval(
    n: int, p: int, h: npt.NDArray[np.intp] | int, alpha: float
) -> npt.NDArray[np.floating]:
    """Critical value for Ljung-Box-Q. Port of ``stats/lbqtest_cval.m``."""
    h_arr = np.atleast_1d(np.asarray(h, dtype=float))
    out = np.full(h_arr.shape, np.nan)
    mask = h_arr > p
    out[mask] = _stats.chi2.ppf(1.0 - alpha, (n * n) * (h_arr[mask] - p))
    return out


# ---------------------------------------------------------------------------
# Multivariate normality + non-parametric tests
# ---------------------------------------------------------------------------


def mardia(X: npt.NDArray[np.floating], debias: bool = False) -> tuple[float, float, float | None]:
    """Mardia's multivariate skewness and kurtosis. Port of ``stats/mardia.m``.

    Returns ``(S, K, J)`` where ``S`` is the multivariate skewness, ``K``
    the centred kurtosis ``K = E[||w||^4] - n(n+2)`` for whitened ``w``,
    and ``J = S/6 + K^2/(8 n (n+2))`` an omnibus chi-square statistic.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:
        n, m, N = X.shape
        X = X.reshape(n, m * N)
    else:
        n, m = X.shape
    d = n * (n + 2)
    X = X - X.mean(axis=1, keepdims=True)
    V = (X @ X.T) / m
    try:
        from scipy.linalg import cholesky

        U = cholesky(V, lower=True)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), None
    w = np.linalg.solve(U, X)
    W = w.T @ w
    W3 = W**3
    S = float(np.mean(W3))
    K = float(np.mean(np.diag(W) ** 2) - d)
    if debias:
        S = S - d * ((m + 1) * (n + 1) - 6) / ((m + 1) * (m + 3))
        K = K + 2 * d / (m + 1)
    J = S / 6.0 + (K * K) / (8.0 * d)
    return S, K, J


def mann_whitney(x1: npt.NDArray[np.floating], x2: npt.NDArray[np.floating]) -> tuple[float, int]:
    """Mann-Whitney U statistic and normal-approximation z-score.

    Port of ``stats/mann_whitney.m``. Uses the asymptotic normal
    approximation; for ties or small samples use SciPy's ``mannwhitneyu``.
    """
    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    n1 = x1.size
    n2 = x2.size
    U = int(np.sum(x2[:, None] > x1[None, :]))
    m = (n1 * n2) / 2.0
    v = (m * (n1 + n2 + 1)) / 6.0
    z = (U - m) / np.sqrt(v)
    return float(z), U


def mann_whitney_group(
    x1: list[npt.NDArray[np.floating]], x2: list[npt.NDArray[np.floating]]
) -> tuple[float, int]:
    """Mann-Whitney U pooled across paired groups. Port of ``stats/mann_whitney_group.m``."""
    if len(x1) != len(x2):
        raise ValueError("Paired group lists must have matching length")
    Ng = len(x1)
    u = np.zeros(Ng)
    n1 = np.zeros(Ng)
    n2 = np.zeros(Ng)
    for g in range(Ng):
        a = np.asarray(x1[g]).ravel()
        b = np.asarray(x2[g]).ravel()
        n1[g] = a.size
        n2[g] = b.size
        u[g] = int(np.sum(b[:, None] > a[None, :]))
    U = int(np.sum(u))
    m = float(np.sum((n1 * n2) / 2.0))
    v = float(np.sum((m * (n1 + n2 + 1)) / 6.0))
    z = (U - m) / np.sqrt(v)
    return float(z), U


# ---------------------------------------------------------------------------
# Empirical / bootstrap inference
# ---------------------------------------------------------------------------


def empirical_cdf(
    x: npt.NDArray[np.floating],
    X: npt.NDArray[np.floating],
    ptails: tuple[float, float] = (0.0, 1.0),
    ksmooth: bool = False,
) -> npt.NDArray[np.floating]:
    """Empirical CDF evaluated at points ``x`` from samples ``X``.

    Pure-Python replacement for ``stats/empirical_cdf.m``. The Pareto-tails
    refinement of MATLAB's ``paretotails`` is approximated by the standard
    empirical CDF (which is conservative); kernel smoothing is available via
    the ``ksmooth`` flag (Gaussian kernel, Silverman bandwidth).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    X = np.asarray(X, dtype=float).ravel()
    X_sorted = np.sort(X)
    if ksmooth:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(X)
        # Numerical CDF via cumulative integration of the KDE
        grid = np.linspace(X_sorted.min(), X_sorted.max(), 1024)
        pdf = kde(grid)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        return np.interp(x, grid, cdf, left=0.0, right=1.0)
    return np.searchsorted(X_sorted, x, side="right") / X_sorted.size


def empirical_cdfi(
    P: npt.NDArray[np.floating],
    X: npt.NDArray[np.floating],
    ptails: tuple[float, float] = (0.0, 1.0),
    ksmooth: bool = False,
) -> npt.NDArray[np.floating]:
    """Inverse empirical CDF (quantile function). Port of ``empirical_cdfi.m``."""
    P = np.atleast_1d(np.asarray(P, dtype=float))
    P = np.clip(P, 0.0, 1.0)
    X = np.asarray(X, dtype=float).ravel()
    return np.quantile(X, P)


def empirical_pval(
    x: npt.NDArray[np.floating],
    XNULL: npt.NDArray[np.floating],
    ptails: tuple[float, float] = (0.0, 1.0),
    ksmooth: bool = False,
) -> npt.NDArray[np.floating]:
    """One-sided permutation p-value from empirical null. Port of ``empirical_pval.m``.

    ``XNULL`` has shape ``(B, ...)`` where ``B`` is the number of surrogates
    and the trailing axes match ``x``'s shape.
    """
    x_arr = np.asarray(x, dtype=float)
    XNULL = np.asarray(XNULL, dtype=float)
    nullarr = XNULL.reshape(XNULL.shape[0], -1)
    flat_x = x_arr.ravel()
    pval = np.full(flat_x.shape, np.nan)
    for i in range(flat_x.size):
        if np.isnan(flat_x[i]):
            continue
        col = nullarr[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        pval[i] = 1.0 - empirical_cdf(flat_x[i], col, ptails, ksmooth)[0]
    return pval.reshape(x_arr.shape)


def empirical_cval(
    alpha: float | npt.NDArray[np.floating],
    XNULL: npt.NDArray[np.floating],
    ptails: tuple[float, float] = (0.0, 1.0),
    ksmooth: bool = False,
) -> npt.NDArray[np.floating]:
    """Empirical critical value at significance ``alpha``. Port of ``empirical_cval.m``."""
    XNULL = np.asarray(XNULL, dtype=float)
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
    leading = XNULL.shape[1:]
    flat = XNULL.reshape(XNULL.shape[0], -1)
    out = np.full((flat.shape[1], alpha_arr.size), np.nan)
    for i in range(flat.shape[1]):
        col = flat[:, i]
        if np.any(np.isnan(col)):
            continue
        out[i, :] = empirical_cdfi(1.0 - alpha_arr, col, ptails, ksmooth)
    return out.reshape((*leading, alpha_arr.size))


def empirical_confint(
    alpha: float,
    X: npt.NDArray[np.floating],
    ptails: tuple[float, float] = (0.0, 1.0),
    ksmooth: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Two-sided empirical confidence interval. Port of ``empirical_confint.m``."""
    X = np.asarray(X, dtype=float)
    alpha = alpha / 2.0
    leading = X.shape[1:]
    flat = X.reshape(X.shape[0], -1)
    xup = np.full(flat.shape[1], np.nan)
    xlo = np.full(flat.shape[1], np.nan)
    for i in range(flat.shape[1]):
        col = flat[:, i]
        if np.any(np.isnan(col)):
            continue
        if np.max(np.abs(col - col.mean())) < 1e-10:
            xup[i] = xlo[i] = col.mean()
        else:
            xup[i] = empirical_cdfi(1.0 - alpha, col, ptails, ksmooth)[0]
            xlo[i] = empirical_cdfi(alpha, col, ptails, ksmooth)[0]
    return xup.reshape(leading), xlo.reshape(leading)


def tsdata_permute(
    X: npt.NDArray[np.floating],
    method: str = "block_shuffle",
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating]:
    """Generate a surrogate time series for permutation tests.

    Methods:
      - ``'shuffle'``: permute the time samples (destroys autocorrelation).
      - ``'block_shuffle'`` (default): permute contiguous blocks of length
        ``block_length`` (preserves short-range autocorrelation; required
        for GC null distributions).
      - ``'phase'``: Theiler-style amplitude-adjusted Fourier transform.

    Python surrogate implementation inspired by MVGC2's
    ``utils/tsdata_permute.m``, plus the standard phase-randomisation surrogate
    of Theiler et al. (1992). The block method deliberately uses ordinary
    NumPy block permutation rather than MVGC2's vendored derangement helper.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    n, m, N = X.shape
    if method == "shuffle":
        Y = X.copy()
        for r in range(N):
            order = rng.permutation(m)
            Y[:, :, r] = X[:, order, r]
    elif method == "block_shuffle":
        if block_length is None:
            block_length = max(1, m // 20)
        nblocks = m // block_length
        if nblocks < 2:
            raise ValueError("block_length too long for series — pick a smaller block")
        Y = np.zeros_like(X)
        for r in range(N):
            order = rng.permutation(nblocks)
            for new_idx, old_idx in enumerate(order):
                Y[
                    :,
                    new_idx * block_length : (new_idx + 1) * block_length,
                    r,
                ] = X[
                    :,
                    old_idx * block_length : (old_idx + 1) * block_length,
                    r,
                ]
            # Append leftover samples without shuffling
            leftover = m - nblocks * block_length
            if leftover:
                Y[:, nblocks * block_length :, r] = X[:, nblocks * block_length :, r]
    elif method == "phase":
        Y = np.zeros_like(X)
        for r in range(N):
            Z = np.fft.fft(X[:, :, r], axis=1)
            mag = np.abs(Z)
            half = m // 2
            phases = rng.uniform(0, 2 * np.pi, size=(n, half - 1))
            new_phase = np.zeros((n, m))
            new_phase[:, 1:half] = phases
            new_phase[:, half + 1 :] = -phases[:, ::-1]
            Z_surr = mag * np.exp(1j * new_phase)
            Y[:, :, r] = np.real(np.fft.ifft(Z_surr, axis=1))
    else:
        raise ValueError(f"unknown surrogate method {method!r}")
    if N == 1:
        return Y[:, :, 0]
    return Y


def significance(
    pvals: npt.NDArray[np.floating],
    alpha: float = 0.05,
    method: str = "FDR",
) -> npt.NDArray[np.bool_]:
    """Multiple-comparison correction. Port of ``stats/significance.m``.

    The FDR branch adapts David M. Groppe's ``fdr_bh`` helper. See
    ``THIRD_PARTY_LICENSES.md`` for its BSD notice.

    Supported methods:
        - ``'none'``: raw thresholding ``p < alpha``
        - ``'Bonferroni'``: threshold ``alpha / N``
        - ``'FDR'`` (Benjamini-Hochberg): step-up procedure
    """
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    finite = ~np.isnan(flat)
    valid = flat[finite]
    N = valid.size
    if N == 0:
        return np.zeros_like(p, dtype=bool)
    if method.lower() == "none":
        sig = valid < alpha
    elif method.lower() == "bonferroni":
        sig = valid < alpha / N
    elif method.lower() in {"fdr", "bh"}:
        order = np.argsort(valid)
        ranked = valid[order]
        thresh = (np.arange(1, N + 1) / N) * alpha
        below = ranked <= thresh
        if not np.any(below):
            sig_sorted = np.zeros(N, dtype=bool)
        else:
            kmax = np.max(np.where(below)[0])
            sig_sorted = np.zeros(N, dtype=bool)
            sig_sorted[: kmax + 1] = True
        sig = np.empty(N, dtype=bool)
        sig[order] = sig_sorted
    else:
        raise ValueError(f"Unknown significance method {method!r}")
    out = np.zeros_like(flat, dtype=bool)
    out[finite] = sig
    return out.reshape(p.shape)
