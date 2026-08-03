"""Integrated Information Decomposition (PhiID).

Port of ELPH's ``PhiID/PhiIDFull.m`` and ``PhiID/PhiIDFullDiscrete.m``,
following Mediano, Rosas et al. (2021) — "Towards an extended taxonomy of
information dynamics via Integrated Information Decomposition".

The decomposition produces 16 atoms ``QtP`` for ``Q, P ∈ {r, x, y, s}``
(redundancy, unique-X, unique-Y, synergy). Two redundancy measures are
supported: ``'MMI'`` (min mutual info) and ``'CCS'`` (common-change-in-surprisal).
The MIB step that ELPH delegates to JIDT for ``D > 2`` is implemented here by
exhaustive search over non-trivial bipartitions (only feasible for small
``D``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from ._mi import gaussian_local_mi

__all__ = ["PhiIDResult", "phi_id_full", "phi_id_full_discrete"]


ATOM_NAMES = (
    "rtr",
    "rtx",
    "rty",
    "rts",
    "xtr",
    "xtx",
    "xty",
    "xts",
    "ytr",
    "ytx",
    "yty",
    "yts",
    "str",
    "stx",
    "sty",
    "sts",
)


@dataclass
class PhiIDResult:
    """PhiID decomposition output.

    ``atoms`` is a dict from atom name → mean atom value. Local (per-sample)
    arrays are in ``locals_`` (None if not requested). Synergy = ``sts``,
    transfer = ``xty`` + ``ytx``, redundancy = ``rtr``, etc.
    """

    atoms: dict[str, float]
    locals_: dict[str, npt.NDArray[np.floating]] | None = None

    @property
    def synergy(self) -> float:
        return self.atoms["sts"]

    @property
    def redundancy(self) -> float:
        return self.atoms["rtr"]

    @property
    def transfer_xy(self) -> float:
        return self.atoms["xty"]

    @property
    def transfer_yx(self) -> float:
        return self.atoms["ytx"]

    def to_array(self) -> npt.NDArray[np.floating]:
        """16-vector of atom values in canonical order."""
        return np.array([self.atoms[name] for name in ATOM_NAMES])


# ---------------------------------------------------------------------------
# PhiID design matrix — copied directly from PhiIDFull.m.
# Atom order: rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts,
# str, stx, sty, sts
# Equations (rows): rtr, Rxyta, Rxytb, Rxytab, Rabtx, Rabty, Rabtxy,
# Ixta, Ixtb, Iyta, Iytb, Ixyta, Ixytb, Ixtab, Iytab, Ixytab.
# ---------------------------------------------------------------------------
_M = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rtr
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxyta
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytb
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytab
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rabtx
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Rabty
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Rabtxy
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixta
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixtb
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Iyta
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # Iytb
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # Ixyta
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Ixytb
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ixtab
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # Iytab
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ixytab
    ],
    dtype=float,
)


def _redundancy_mmi(mi1: np.ndarray, mi2: np.ndarray, _mi12: np.ndarray) -> np.ndarray:
    """MMI redundancy: pick the smaller-mean of the two single-source MIs."""
    return mi1 if np.mean(mi1) < np.mean(mi2) else mi2


def _redundancy_ccs(mi1: np.ndarray, mi2: np.ndarray, mi12: np.ndarray) -> np.ndarray:
    """Common-change-in-surprisal redundancy. Sign-coherent: returns ``-c``
    where ``c = mi12 - mi1 - mi2`` only where all four quantities share sign,
    else zero (pointwise).
    """
    c = mi12 - mi1 - mi2
    signs = np.stack([np.sign(mi1), np.sign(mi2), np.sign(mi12), np.sign(-c)], axis=1)
    agree = np.all(signs == signs[:, [0]], axis=1)
    return agree.astype(float) * (-c)


def _phi_id_four_vector(
    X1: npt.NDArray[np.floating],
    X2: npt.NDArray[np.floating],
    Y1: npt.NDArray[np.floating],
    Y2: npt.NDArray[np.floating],
    measure: str,
) -> tuple[dict[str, float], dict[str, npt.NDArray[np.floating]]]:
    """PhiID for an explicit 4-vector (X1, X2 ; Y1, Y2). Port of
    ``private_FourVectorPhiID``.
    """
    measure = measure.lower()
    if measure == "mmi":
        red_fn = _redundancy_mmi
    elif measure == "ccs":
        red_fn = _redundancy_ccs
    else:
        raise ValueError(f"Unknown PhiID measure {measure!r}; supported: 'MMI', 'CCS'")

    # Compute pairwise local MIs (T-length arrays)
    Ixta = gaussian_local_mi(X1, Y1)
    Ixtb = gaussian_local_mi(X1, Y2)
    Iyta = gaussian_local_mi(X2, Y1)
    Iytb = gaussian_local_mi(X2, Y2)

    Ixyta = gaussian_local_mi(np.vstack([X1, X2]), Y1)
    Ixytb = gaussian_local_mi(np.vstack([X1, X2]), Y2)
    Ixtab = gaussian_local_mi(X1, np.vstack([Y1, Y2]))
    Iytab = gaussian_local_mi(X2, np.vstack([Y1, Y2]))
    Ixytab = gaussian_local_mi(np.vstack([X1, X2]), np.vstack([Y1, Y2]))

    Rxyta = red_fn(Ixta, Iyta, Ixyta)
    Rxytb = red_fn(Ixtb, Iytb, Ixytb)
    Rxytab = red_fn(Ixtab, Iytab, Ixytab)
    Rabtx = red_fn(Ixta, Ixtb, Ixtab)
    Rabty = red_fn(Iyta, Iytb, Iytab)
    Rabtxy = red_fn(Ixyta, Ixytb, Ixytab)

    # Double redundancy (rtr atom):
    if measure == "mmi":
        # Lower-mean of all four single-source local MIs
        candidates = [Ixta, Ixtb, Iyta, Iytb]
        means = [np.mean(c) for c in candidates]
        rtr = candidates[int(np.argmin(means))]
    else:  # CCS variant: use sign-coherent triple-CCS
        c = Ixytab - Ixyta - Ixytb
        s = np.stack([np.sign(Ixyta), np.sign(Ixytb), np.sign(Ixytab), np.sign(-c)], axis=1)
        agree = np.all(s == s[:, [0]], axis=1)
        rtr = agree.astype(float) * (-c)

    reds = np.stack(
        [
            rtr,
            Rxyta,
            Rxytb,
            Rxytab,
            Rabtx,
            Rabty,
            Rabtxy,
            Ixta,
            Ixtb,
            Iyta,
            Iytb,
            Ixyta,
            Ixytb,
            Ixtab,
            Iytab,
            Ixytab,
        ],
        axis=0,
    )  # (16, T)

    # Solve M @ partials = reds for partials (16, T)
    partials = np.linalg.solve(_M, reds)
    locals_ = {name: partials[i] for i, name in enumerate(ATOM_NAMES)}
    means = {name: float(np.mean(arr[np.isfinite(arr)])) for name, arr in locals_.items()}
    return means, locals_


def _find_mib_gaussian_d2(_X: npt.NDArray[np.floating]) -> tuple[list[int], list[int]]:
    """Trivial 2-variable case: partition is {0} | {1}."""
    return [0], [1]


def _find_mib_gaussian(X: npt.NDArray[np.floating]) -> tuple[list[int], list[int]]:
    """Exhaustive minimum-information bipartition for small D.

    For each non-trivial bipartition, compute the integrated information
    (mutual information across the cut) and return the one with the smallest
    *normalised* MI. This is an approximation of ELPH/JIDT's atomic-partition
    Phi but should give the same MIB for typical data.
    """
    D, _T = X.shape
    if D == 2:
        return [0], [1]
    indices = list(range(D))
    best = None
    best_score = np.inf
    for k in range(1, D // 2 + 1):
        for p1 in combinations(indices, k):
            p2 = tuple(i for i in indices if i not in p1)
            mi_local = gaussian_local_mi(X[list(p1)], X[list(p2)])
            mi = float(np.mean(mi_local))
            norm = float(min(len(p1), len(p2)))
            score = mi / norm if norm > 0 else mi
            if score < best_score:
                best_score = score
                best = (list(p1), list(p2))
    assert best is not None
    return best


def phi_id_full(
    X: npt.NDArray[np.floating] | None = None,
    tau: int = 1,
    measure: str = "CCS",
    *,
    X1: npt.NDArray[np.floating] | None = None,
    X2: npt.NDArray[np.floating] | None = None,
    Y1: npt.NDArray[np.floating] | None = None,
    Y2: npt.NDArray[np.floating] | None = None,
    return_locals: bool = False,
) -> PhiIDResult:
    """Full PhiID decomposition under Gaussian assumption.

    Two calling forms:

    - ``phi_id_full(X, tau)`` — time-delayed PhiID with ``D × T`` data and
      integer ``tau ≥ 1``. The MIB is computed automatically (exhaustive
      search; only practical for ``D ≲ 8``).
    - ``phi_id_full(X1=..., X2=..., Y1=..., Y2=...)`` — explicit 4-vector
      form; each input is a ``d_k × T`` matrix.

    Parameters
    ----------
    measure : 'CCS' or 'MMI'
    return_locals : if True, attach pointwise per-sample arrays.
    """
    if X is not None:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        D, T = X.shape
        if T <= D:
            raise ValueError(
                f"X has {D} dims and {T} timesteps; need T > D for a valid covariance."
            )
        if not (isinstance(tau, (int, np.integer)) and tau >= 1):
            raise ValueError("tau must be a positive integer")
        # Scale to unit variance per row for numerical stability
        sd = X.std(axis=1, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        sX = X / sd
        p1, p2 = _find_mib_gaussian(sX)
        Xp = sX[:, :-tau]
        Xf = sX[:, tau:]
        means, locals_ = _phi_id_four_vector(Xp[p1], Xp[p2], Xf[p1], Xf[p2], measure)
    else:
        if X1 is None or X2 is None or Y1 is None or Y2 is None:
            raise ValueError("Provide either X (with optional tau) or all four of X1, X2, Y1, Y2")
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        Y1 = np.atleast_2d(np.asarray(Y1, dtype=float))
        Y2 = np.atleast_2d(np.asarray(Y2, dtype=float))
        if not (X1.shape[1] == X2.shape[1] == Y1.shape[1] == Y2.shape[1]):
            raise ValueError("All four inputs must have the same number of samples")
        stacked = np.vstack([X1, X2, Y1, Y2])
        sd = stacked.std(axis=1, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        offsets = np.cumsum([X1.shape[0], X2.shape[0], Y1.shape[0], Y2.shape[0]])
        sX = stacked / sd
        s1 = sX[: offsets[0]]
        s2 = sX[offsets[0] : offsets[1]]
        sY1 = sX[offsets[1] : offsets[2]]
        sY2 = sX[offsets[2] :]
        means, locals_ = _phi_id_four_vector(s1, s2, sY1, sY2, measure)
    return PhiIDResult(atoms=means, locals_=locals_ if return_locals else None)


def phi_id_full_discrete(
    X: npt.NDArray | None = None,
    tau: int = 1,
    measure: str = "MMI",
    *,
    X1: npt.NDArray | None = None,
    X2: npt.NDArray | None = None,
    Y1: npt.NDArray | None = None,
    Y2: npt.NDArray | None = None,
) -> PhiIDResult:
    """Plug-in (discrete) PhiID decomposition.

    Continuous data is auto-binarised via mean-thresholding, mirroring
    ELPH's ``PhiIDFullDiscrete.m`` fallback. Only mean atom values are
    reported — the discrete plug-in does not produce per-sample locals in
    the same way as the Gaussian path.
    """
    from ._mi import discrete_mi

    if X is not None:
        X = np.atleast_2d(np.asarray(X))
        if X.dtype != np.intp and X.dtype != bool:
            # Auto-binarise on the row mean (ELPH convention)
            X = (X > X.mean(axis=1, keepdims=True)).astype(np.intp)
        D, T = X.shape
        if D != 2:
            raise NotImplementedError(
                "Discrete PhiID for D > 2 requires JIDT; only D = 2 implemented."
            )
        Xp = X[:, :-tau]
        Xf = X[:, tau:]
        X1, X2 = Xp[0:1], Xp[1:2]
        Y1, Y2 = Xf[0:1], Xf[1:2]
    if X1 is None or X2 is None or Y1 is None or Y2 is None:
        raise ValueError("Either X or all four of X1/X2/Y1/Y2 must be supplied")
    X1, X2, Y1, Y2 = (np.atleast_2d(np.asarray(z)) for z in (X1, X2, Y1, Y2))
    if X1.dtype != np.intp:
        X1 = (X1 > X1.mean(axis=1, keepdims=True)).astype(np.intp)
    if X2.dtype != np.intp:
        X2 = (X2 > X2.mean(axis=1, keepdims=True)).astype(np.intp)
    if Y1.dtype != np.intp:
        Y1 = (Y1 > Y1.mean(axis=1, keepdims=True)).astype(np.intp)
    if Y2.dtype != np.intp:
        Y2 = (Y2 > Y2.mean(axis=1, keepdims=True)).astype(np.intp)

    # All MIs as plug-in scalars (no locals here)
    Ixta = discrete_mi(X1, Y1)
    Ixtb = discrete_mi(X1, Y2)
    Iyta = discrete_mi(X2, Y1)
    Iytb = discrete_mi(X2, Y2)
    Ixyta = discrete_mi(np.vstack([X1, X2]), Y1)
    Ixytb = discrete_mi(np.vstack([X1, X2]), Y2)
    Ixtab = discrete_mi(X1, np.vstack([Y1, Y2]))
    Iytab = discrete_mi(X2, np.vstack([Y1, Y2]))
    Ixytab = discrete_mi(np.vstack([X1, X2]), np.vstack([Y1, Y2]))

    def red_mmi(a, b, _ab):
        return min(a, b)

    if measure.upper() == "MMI":
        red = red_mmi
        rtr = min(Ixta, Ixtb, Iyta, Iytb)
    else:
        raise NotImplementedError(
            "Only 'MMI' implemented for discrete PhiID; CCS/Rmin require local estimators."
        )
    Rxyta = red(Ixta, Iyta, Ixyta)
    Rxytb = red(Ixtb, Iytb, Ixytb)
    Rxytab = red(Ixtab, Iytab, Ixytab)
    Rabtx = red(Ixta, Ixtb, Ixtab)
    Rabty = red(Iyta, Iytb, Iytab)
    Rabtxy = red(Ixyta, Ixytb, Ixytab)

    reds = np.array(
        [
            rtr,
            Rxyta,
            Rxytb,
            Rxytab,
            Rabtx,
            Rabty,
            Rabtxy,
            Ixta,
            Ixtb,
            Iyta,
            Iytb,
            Ixyta,
            Ixytb,
            Ixtab,
            Iytab,
            Ixytab,
        ]
    )
    partials = np.linalg.solve(_M, reds)
    atoms = {name: float(partials[i]) for i, name in enumerate(ATOM_NAMES)}
    return PhiIDResult(atoms=atoms, locals_=None)
