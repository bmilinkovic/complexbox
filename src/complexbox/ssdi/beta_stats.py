"""β-statistic significance testing for hyperplane projections.

Port of SSDI-1's ``utils/habeta_statinf.m``, ``utils/get_haxa_cvals.m``, plus
the Monte-Carlo null distribution generators ``batch/gen_haxa_dist.m`` and
``batch/make_haxa_stats.m`` rolled into a single in-memory function
(:func:`haxa_dist`).

The β-statistic of an orthonormal basis ``L`` of shape ``(n, m)`` is the
vector of squared row norms ``β_i = ||L_i||²``. Under the null hypothesis
that ``L`` is uniformly distributed on the Grassmannian ``G(n, m)``, each
``β_i`` is marginally ``Beta(m/2, (n - m)/2)``. The dependence between
entries (they sum to ``m``) is the reason SSDI-1 provides the Monte-Carlo
``haxa_*`` machinery: the joint distribution of the maximum / k-th-largest β
is not closed-form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import beta as _beta_dist

__all__ = [
    "habeta_statinf",
    "haxa_dist",
    "make_haxa_stats",
    "get_haxa_cvals",
    "BetaStatResult",
    "HaxaStats",
]


@dataclass
class BetaStatResult:
    """Per-channel β-statistic inference output."""

    cval: npt.NDArray[np.floating]
    pval: npt.NDArray[np.floating]
    sig: npt.NDArray[np.bool_]


def habeta_statinf(
    beta: npt.NDArray[np.floating],
    n: int,
    m: int,
    slevel: float = 0.05,
    tails: str = "both",
    mhtc: bool = True,
) -> BetaStatResult:
    """β-statistic significance test under the marginal Beta(m/2, (n-m)/2) null.

    Port of ``utils/habeta_statinf.m``.

    Parameters
    ----------
    beta : 1-D array of squared row norms of an orthonormal basis
    n, m : ambient and subspace dimensions
    slevel : significance level (default 0.05)
    tails : ``'left'``, ``'right'``, or ``'both'``
    mhtc : Bonferroni correct for ``len(beta)`` hypotheses (default True)

    Returns
    -------
    BetaStatResult with critical values, p-values, and significance mask.
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    nbeta = beta.size
    a = m / 2.0
    b = (n - m) / 2.0
    bcdf = _beta_dist.cdf(beta, a, b)
    tails = tails.lower()
    if tails == "left":
        slevel_use = slevel / nbeta if mhtc else slevel
        cval = _beta_dist.ppf(slevel_use, a, b)
        pval = bcdf
    elif tails == "right":
        slevel_use = slevel / nbeta if mhtc else slevel
        cval = _beta_dist.ppf(1.0 - slevel_use, a, b)
        pval = 1.0 - bcdf
    elif tails == "both":
        # MATLAB only splits alpha across tails as part of its optional
        # Bonferroni correction. With mhtc=False it applies ``slevel`` to
        # each tail independently.
        slevel_use = slevel / (2 * nbeta) if mhtc else slevel
        cval = np.array(
            [
                _beta_dist.ppf(slevel_use, a, b),
                _beta_dist.ppf(1.0 - slevel_use, a, b),
            ]
        )
        # 2-column p-value: lower- and upper-tail
        pval = np.column_stack([bcdf, 1.0 - bcdf])
    else:
        raise ValueError(f"unknown tails {tails!r}; use 'left', 'right', 'both'")
    sig = pval <= slevel_use
    return BetaStatResult(cval=cval, pval=pval, sig=sig)


@dataclass
class HaxaStats:
    """Pre-computed null distribution of the hyperplane-axis angle statistic.

    ``theta`` has shape ``(N, n - 1)``. Column ``m - 1`` contains the
    principal angle between a random ``m``-dimensional hyperplane and a
    coordinate axis. Following MATLAB ``make_haxa_stats.m``, dimensions above
    ``floor(n / 2)`` are obtained by complementary reflection
    ``theta_m = pi / 2 - theta_(n-m)``.

    ``cvals`` has shape ``(len(slevels), n - 1)`` and stores the corresponding
    empirical critical angles. Since ``beta = cos(theta)**2``, a high-beta
    threshold at confidence ``1 - alpha`` uses the *lower* angle quantile
    ``alpha``.
    """

    n: int
    N: int
    theta: npt.NDArray[np.floating]
    slevels: npt.NDArray[np.floating]
    cvals: npt.NDArray[np.floating]


def haxa_dist(
    n: int,
    N: int = 100_000,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating]:
    """Monte-Carlo null distribution of hyperplane-axis principal angles.

    Port of ``batch/gen_haxa_dist.m`` (in-memory, no file I/O). For each
    ``m = 1, ..., floor(n/2)`` we sample ``N`` random unit vectors in
    ``R^n`` and compute the principal angle between the first-``m``-axis
    subspace and the random direction. The full distribution table is
    returned as a ``(N, floor(n/2))`` array.
    """
    if rng is None:
        rng = np.random.default_rng()
    h = n // 2
    theta = np.empty((N, h))
    for m in range(1, h + 1):
        v = rng.standard_normal((n, N))
        ratio = np.sum(v[:m] ** 2, axis=0) / np.sum(v**2, axis=0)
        theta[:, m - 1] = np.arccos(np.sqrt(np.clip(ratio, 0.0, 1.0)))
    return theta


def make_haxa_stats(
    nmax: int,
    N: int = 100_000,
    slevels: npt.NDArray[np.floating] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[int, HaxaStats]:
    """Aggregate per-(n, m) Monte-Carlo statistics from :func:`haxa_dist`.

    Port of ``batch/make_haxa_stats.m``. Returns a dict ``{n: HaxaStats}``
    for ``n = 2, ..., nmax``. Each ``HaxaStats`` carries the empirical
    quantile table needed by :func:`get_haxa_cvals`.

    The returned table follows MATLAB and explicitly materialises all
    dimensions ``m = 1, ..., n - 1`` after generating only the independent
    half.
    """
    if rng is None:
        rng = np.random.default_rng()
    if slevels is None:
        # MATLAB's haxa_slev grid is dense near both tails
        seg1 = np.arange(0.0001, 0.0005, 0.0001)
        seg2 = np.arange(0.0005, 0.0105, 0.0005)
        seg3 = np.arange(0.015, 0.105, 0.005)
        seg4 = np.arange(0.125, 0.300, 0.025)
        slevels = np.concatenate(
            [
                [0.0],
                seg1,
                seg2,
                seg3,
                seg4,
                1 - seg4[::-1],
                1 - seg3[::-1],
                1 - seg2[::-1],
                1 - seg1[::-1],
                [1.0],
            ]
        )
    slevels = np.asarray(slevels, dtype=float)
    if slevels.ndim != 1 or slevels.size == 0:
        raise ValueError("slevels must be a non-empty vector")
    if np.any(~np.isfinite(slevels)) or np.any((slevels < 0.0) | (slevels > 1.0)):
        raise ValueError("slevels must contain finite values in [0, 1]")
    if np.any(np.diff(slevels) <= 0.0):
        raise ValueError("slevels must be strictly increasing")

    stats = {}
    for n in range(2, nmax + 1):
        theta_half = haxa_dist(n, N=N, rng=rng)
        h = n // 2
        theta = np.empty((N, n - 1))
        theta[:, :h] = theta_half
        # Direct translation of MATLAB:
        # theta(:, n-(1:h)) = pi/2 - theta(:, 1:h)
        complement_columns = n - np.arange(1, h + 1) - 1
        theta[:, complement_columns] = np.pi / 2 - theta_half
        cvals = np.quantile(theta, slevels, axis=0)
        stats[n] = HaxaStats(
            n=n,
            N=N,
            theta=theta,
            slevels=slevels,
            cvals=cvals,
        )
    return stats


def get_haxa_cvals(
    n: int,
    stats: dict[int, HaxaStats] | HaxaStats,
    mdim: npt.NDArray[np.intp] | list[int] | None = None,
    slev: npt.NDArray[np.floating] | tuple[float, ...] = (0.05, 0.95),
) -> npt.NDArray[np.floating]:
    """Look up Monte-Carlo critical values from a pre-computed table.

    Port of ``utils/get_haxa_cvals.m``.

    Parameters
    ----------
    n : ambient dimension
    stats : output of :func:`make_haxa_stats`, or a single ``HaxaStats``
    mdim : hyperplane dimensions to look up (default ``1..n-1``)
    slev : significance levels (default ``(0.05, 0.95)`` two-tailed)

    Returns
    -------
    ``(len(mdim), len(slev))`` array of critical angles in radians.
    """
    if isinstance(stats, dict):
        if n not in stats:
            raise KeyError(f"no HaxaStats for n = {n}; call make_haxa_stats first")
        S = stats[n]
    else:
        S = stats
    if S.n != n:
        raise ValueError(f"HaxaStats is for n = {S.n}, not n = {n}")
    if mdim is None:
        mdim = np.arange(1, n)
    mdim = np.atleast_1d(np.asarray(mdim, dtype=int))
    slev = np.atleast_1d(np.asarray(slev, dtype=float))
    if np.any(~np.isfinite(slev)) or np.any((slev < 0.0) | (slev > 1.0)):
        raise ValueError("slev must contain finite values in [0, 1]")
    cvaln = S.cvals  # (len(slevels), n - 1)
    out = np.empty((mdim.size, slev.size))
    for i, m in enumerate(mdim):
        if not (1 <= m < n):
            raise ValueError(f"m = {m} out of range 1..n-1")
        for k, alpha in enumerate(slev):
            out[i, k] = float(np.interp(alpha, S.slevels, cvaln[:, m - 1]))
    return out
