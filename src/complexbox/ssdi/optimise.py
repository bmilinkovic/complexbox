"""Gradient-descent optimisation of dynamical dependence over the Grassmannian.

Ports of SSDI-1's ``utils/opt_gd1_ddx.m``, ``utils/opt_gd2_ddx.m``,
``utils/opt_gd_ddx_mruns.m``, and the corresponding spectral-DD variants
``opt_gd{1,2}_dds`` / ``opt_gd_dds_mruns``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._grassmann import orthonormalise
from .dd import cak2ddx, cak2ddxgrad, trfun2dd, trfun2ddgrad

__all__ = [
    "GDResult",
    "opt_gd1_ddx",
    "opt_gd2_ddx",
    "opt_gd1_dds",
    "opt_gd2_dds",
    "opt_gd_ddx_mruns",
    "opt_gd_dds_mruns",
]


@dataclass
class GDResult:
    """Single-run gradient-descent result."""

    dd: float
    L: npt.NDArray[np.floating]
    converged: int
    sig: float
    iters: int
    history: npt.NDArray[np.floating] | None


def _parse_tol(
    tol: float | tuple[float, float, float], *, spectral: bool = False
) -> tuple[float, float, float]:
    """Expand MATLAB's scalar optimisation tolerance convention.

    SSDI-1 uses ``gtol = tol / 10`` for the CAK proxy optimisers but
    ``gtol = tol`` for the spectral optimisers.
    """
    if np.isscalar(tol):
        value = float(tol)
        return value, value, value if spectral else value / 10.0
    s, d, g = tol
    return float(s), float(d), float(g)


def _parse_gdls(gdls: float | tuple[float, float]) -> tuple[float, float]:
    if np.isscalar(gdls):
        return float(gdls), 1.0 / float(gdls)
    return float(gdls[0]), float(gdls[1])


def _opt_gd_ddx_generic(
    CAK: npt.NDArray[np.floating],
    L0: npt.NDArray[np.floating],
    maxiters: int,
    gdsig0: float,
    gdls: float | tuple[float, float],
    tol: float | tuple[float, float, float],
    history: bool,
    variant: int,
) -> GDResult:
    """Shared GD/ES loop for ddx variants 1 and 2 (proxy DD)."""
    ifac, nfac = _parse_gdls(gdls)
    stol, dtol, gtol = _parse_tol(tol)

    L = L0.copy()
    G, g = cak2ddxgrad(L, CAK)
    dd = cak2ddx(L, CAK)
    sig = gdsig0

    dhist: list[tuple[float, float, float]] = []
    if history:
        dhist.append((dd, sig, g))

    converged = 0
    iters = 1
    for iteration in range(2, maxiters + 1):
        iters = iteration
        Ltry = orthonormalise(L - sig * (G / g) if g > 0 else L)
        ddtry = cak2ddx(Ltry, CAK)

        if variant == 1:
            if ddtry < dd:
                L = Ltry
                G, g = cak2ddxgrad(L, CAK)
                dd = ddtry
                sig = ifac * sig
            else:
                sig = nfac * sig
        elif variant == 2:
            L = Ltry
            G, g = cak2ddxgrad(L, CAK)
            ddnew = cak2ddx(L, CAK)
            if ddnew < dd:
                dd = ddnew
                sig = ifac * sig
            else:
                sig = nfac * sig
        else:
            raise ValueError("variant must be 1 or 2")

        if history:
            dhist.append((dd, sig, g))
        if sig < stol:
            converged = 1
            break
        if dd < dtol:
            converged = 2
            break
        if g < gtol:
            converged = 3
            break

    return GDResult(
        dd=dd,
        L=L,
        converged=converged,
        sig=sig,
        iters=iters,
        history=np.array(dhist) if history else None,
    )


def opt_gd1_ddx(
    CAK: npt.NDArray[np.floating],
    L0: npt.NDArray[np.floating],
    maxiters: int = 10000,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> GDResult:
    """Proxy-DD gradient descent, variant 1 (accept-only)."""
    return _opt_gd_ddx_generic(CAK, L0, maxiters, gdsig0, gdls, tol, history, 1)


def opt_gd2_ddx(
    CAK: npt.NDArray[np.floating],
    L0: npt.NDArray[np.floating],
    maxiters: int = 10000,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> GDResult:
    """Proxy-DD gradient descent, variant 2 (always-step)."""
    return _opt_gd_ddx_generic(CAK, L0, maxiters, gdsig0, gdls, tol, history, 2)


def _opt_gd_dds_generic(
    H: npt.NDArray[np.complexfloating],
    L0: npt.NDArray[np.floating],
    maxiters: int,
    gdsig0: float,
    gdls: float | tuple[float, float],
    tol: float | tuple[float, float, float],
    history: bool,
    variant: int,
) -> GDResult:
    ifac, nfac = _parse_gdls(gdls)
    stol, dtol, gtol = _parse_tol(tol, spectral=True)

    L = L0.copy()
    G, g = trfun2ddgrad(L, H)
    dd, _ = trfun2dd(L, H)
    sig = gdsig0
    dhist: list[tuple[float, float, float]] = []
    if history:
        dhist.append((dd, sig, g))

    converged = 0
    iters = 1
    for iteration in range(2, maxiters + 1):
        iters = iteration
        Ltry = orthonormalise(L - sig * (G / g) if g > 0 else L)
        ddtry, _ = trfun2dd(Ltry, H)
        if variant == 1:
            if ddtry < dd:
                L = Ltry
                G, g = trfun2ddgrad(L, H)
                dd = ddtry
                sig = ifac * sig
            else:
                sig = nfac * sig
        elif variant == 2:
            L = Ltry
            G, g = trfun2ddgrad(L, H)
            ddnew, _ = trfun2dd(L, H)
            if ddnew < dd:
                dd = ddnew
                sig = ifac * sig
            else:
                sig = nfac * sig
        else:
            raise ValueError("variant must be 1 or 2")
        if history:
            dhist.append((dd, sig, g))
        if sig < stol:
            converged = 1
            break
        if dd < dtol:
            converged = 2
            break
        if g < gtol:
            converged = 3
            break

    return GDResult(
        dd=dd,
        L=L,
        converged=converged,
        sig=sig,
        iters=iters,
        history=np.array(dhist) if history else None,
    )


def opt_gd1_dds(
    H: npt.NDArray[np.complexfloating],
    L0: npt.NDArray[np.floating],
    maxiters: int = 10000,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> GDResult:
    """Spectral-DD gradient descent, variant 1."""
    return _opt_gd_dds_generic(H, L0, maxiters, gdsig0, gdls, tol, history, 1)


def opt_gd2_dds(
    H: npt.NDArray[np.complexfloating],
    L0: npt.NDArray[np.floating],
    maxiters: int = 10000,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
) -> GDResult:
    """Spectral-DD gradient descent, variant 2."""
    return _opt_gd_dds_generic(H, L0, maxiters, gdsig0, gdls, tol, history, 2)


def opt_gd_ddx_mruns(
    CAK: npt.NDArray[np.floating],
    L0_all: npt.NDArray[np.floating],
    maxiters: int = 10000,
    variant: int = 2,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
    *,
    backend: str = "numpy",
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    lag_chunk_size: int | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    list[int],
    list[npt.NDArray[np.floating] | None],
]:
    """Multi-run proxy DD optimisation. Port of ``opt_gd_ddx_mruns.m``.

    Parameters
    ----------
    CAK : (n, n, r) — output of :func:`iss2cak`, or the VAR coefficient array
        (the proxy-DD MATLAB convention accepts either).
    L0_all : (n, m, R) — initial bases for ``R`` runs
    variant : 1 or 2
    history : if True, retain per-run convergence trajectories

    Returns
    -------
    dds : (R,) sorted DD values
    Ls : (n, m, R) corresponding optimised projections (same sort order)
    convergence : list of convergence codes (1: step too small, 2: dd
        below tol, 3: gradient below tol, 0: unconverged)
    histories : list of per-run history arrays (or list of None if
        ``history=False``)
    """
    if backend == "torch":
        from ._torch import opt_gd_ddx_mruns_torch

        return opt_gd_ddx_mruns_torch(
            CAK,
            L0_all,
            maxiters,
            variant,
            gdsig0,
            gdls,
            tol,
            history,
            device=device,
            run_chunk_size=run_chunk_size,
            lag_chunk_size=lag_chunk_size,
        )
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")

    L0_all = np.asarray(L0_all, dtype=float)
    R = L0_all.shape[2]
    dds = np.zeros(R)
    Ls = np.zeros_like(L0_all)
    conv = [0] * R
    histories: list[npt.NDArray[np.floating] | None] = [None] * R
    fn = opt_gd1_ddx if variant == 1 else opt_gd2_ddx
    for k in range(R):
        res = fn(CAK, L0_all[:, :, k], maxiters, gdsig0, gdls, tol, history)
        dds[k] = res.dd
        Ls[:, :, k] = res.L
        conv[k] = res.converged
        histories[k] = res.history
    order = np.argsort(dds)
    return (
        dds[order],
        Ls[:, :, order],
        [conv[i] for i in order],
        [histories[i] for i in order],
    )


def opt_gd_dds_mruns(
    H: npt.NDArray[np.complexfloating],
    L0_all: npt.NDArray[np.floating],
    maxiters: int = 10000,
    variant: int = 2,
    gdsig0: float = 1e-3,
    gdls: float | tuple[float, float] = 2.0,
    tol: float | tuple[float, float, float] = 1e-9,
    history: bool = False,
    *,
    backend: str = "numpy",
    device: str | None = "cpu",
    run_chunk_size: int | None = None,
    frequency_chunk_size: int | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    list[int],
    list[npt.NDArray[np.floating] | None],
]:
    """Multi-run spectral DD refinement. Port of ``opt_gd_dds_mruns.m``."""
    if backend == "torch":
        from ._torch import opt_gd_dds_mruns_torch

        return opt_gd_dds_mruns_torch(
            H,
            L0_all,
            maxiters,
            variant,
            gdsig0,
            gdls,
            tol,
            history,
            device=device,
            run_chunk_size=run_chunk_size,
            frequency_chunk_size=frequency_chunk_size,
        )
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")

    L0_all = np.asarray(L0_all, dtype=float)
    R = L0_all.shape[2]
    dds = np.zeros(R)
    Ls = np.zeros_like(L0_all)
    conv = [0] * R
    histories: list[npt.NDArray[np.floating] | None] = [None] * R
    fn = opt_gd1_dds if variant == 1 else opt_gd2_dds
    for k in range(R):
        res = fn(H, L0_all[:, :, k], maxiters, gdsig0, gdls, tol, history)
        dds[k] = res.dd
        Ls[:, :, k] = res.L
        conv[k] = res.converged
        histories[k] = res.history
    order = np.argsort(dds)
    return (
        dds[order],
        Ls[:, :, order],
        [conv[i] for i in order],
        [histories[i] for i in order],
    )
