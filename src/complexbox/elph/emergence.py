"""Causal emergence metrics: Ψ, Δ, Γ.

Port of ELPH's ``Emergence/EmergencePsi.m``, ``EmergenceDelta.m``,
``EmergenceGamma.m`` following Rosas, Mediano et al. (2020) — "Reconciling
emergences: An information-theoretic approach to identify causal emergence
in multivariate data".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._mi import discrete_mi, gaussian_local_mi

__all__ = [
    "EmergenceResult",
    "emergence_psi",
    "emergence_delta",
    "emergence_gamma",
]


@dataclass
class EmergenceResult:
    """Output of an emergence calculation."""

    value: float
    v_mi: float
    x_mi: float
    locals_: dict[str, npt.NDArray[np.floating]] | None


def _infer_method(X: np.ndarray, V: np.ndarray) -> str:
    """Heuristic from EmergencePsi.m: discrete if integer-valued, else Gaussian."""
    if (np.sum(np.abs(X - np.round(X))) + np.sum(np.abs(V - np.round(V)))) < 1e-10:
        return "discrete"
    return "gaussian"


def _local_mi(X: np.ndarray, Y: np.ndarray, method: str) -> np.ndarray:
    """Pointwise local MI (for Gaussian) or repeated scalar (for discrete)."""
    if method == "gaussian":
        return gaussian_local_mi(X, Y)
    # Discrete: plug-in MI gives a scalar; broadcast to length T
    T = X.shape[-1] if X.ndim > 1 else X.size
    mi = discrete_mi(X, Y)
    return np.full(T, mi)


def emergence_psi(
    X: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    tau: int = 1,
    method: str | None = None,
    return_locals: bool = False,
) -> EmergenceResult:
    """Causal-emergence criterion Ψ = I(V_past; V_future) - Σ_j I(X_j^past; V_future).

    Port of ``EmergencePsi.m``. ``X`` is ``D × T`` micro data, ``V`` is the
    ``T``-length macro signal (1-D).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    V = np.asarray(V, dtype=float).ravel()
    if V.size != X.shape[1]:
        raise ValueError("X and V must have the same time-length")
    if method is None:
        method = _infer_method(X, V)
    D, T = X.shape

    Vp = V[: T - tau]
    Vf = V[tau:]
    v_lmi = _local_mi(Vp[None, :], Vf[None, :], method)
    x_lmi = np.zeros_like(v_lmi)
    for j in range(D):
        Xj_p = X[j, : T - tau]
        x_lmi = x_lmi + _local_mi(Xj_p[None, :], Vf[None, :], method)
    lpsi = v_lmi - x_lmi
    return EmergenceResult(
        value=float(np.mean(lpsi)),
        v_mi=float(np.mean(v_lmi)),
        x_mi=float(np.mean(x_lmi)),
        locals_={"psi": lpsi, "v_mi": v_lmi, "x_mi": x_lmi} if return_locals else None,
    )


def emergence_delta(
    X: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    tau: int = 1,
    method: str | None = None,
    return_locals: bool = False,
) -> EmergenceResult:
    """Downward-causation criterion Δ = max_j[I(V^past; X_j^future) - Σ_i I(X_i^past; X_j^future)].

    Port of ``EmergenceDelta.m``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    V = np.asarray(V, dtype=float).ravel()
    if V.size != X.shape[1]:
        raise ValueError("X and V must have the same time-length")
    if method is None:
        method = _infer_method(X, V)
    D, T = X.shape

    Vp = V[: T - tau]
    v_lmi = np.zeros((T - tau, D))
    for j in range(D):
        Xj_f = X[j, tau:]
        v_lmi[:, j] = _local_mi(Vp[None, :], Xj_f[None, :], method)

    x_lmi = np.zeros((T - tau, D, D))
    for i in range(D):
        for j in range(D):
            Xi_p = X[i, : T - tau]
            Xj_f = X[j, tau:]
            x_lmi[:, i, j] = _local_mi(Xi_p[None, :], Xj_f[None, :], method)
    x_lmi_sum = x_lmi.sum(axis=1)  # (T-tau, D)

    ldelta = v_lmi - x_lmi_sum
    max_idx = int(np.argmax(np.mean(ldelta, axis=0)))
    ldelta_max = ldelta[:, max_idx]
    return EmergenceResult(
        value=float(np.mean(ldelta_max)),
        v_mi=float(np.mean(v_lmi[:, max_idx])),
        x_mi=float(np.mean(x_lmi_sum[:, max_idx])),
        locals_={"delta": ldelta_max, "v_mi": v_lmi, "x_mi": x_lmi_sum} if return_locals else None,
    )


def emergence_gamma(
    X: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    tau: int = 1,
    method: str | None = None,
    return_locals: bool = False,
) -> EmergenceResult:
    """Causal-decoupling criterion Γ = max_j I(V^past; X_j^future).

    Port of ``EmergenceGamma.m``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    V = np.asarray(V, dtype=float).ravel()
    if V.size != X.shape[1]:
        raise ValueError("X and V must have the same time-length")
    if method is None:
        method = _infer_method(X, V)
    D, T = X.shape

    Vp = V[: T - tau]
    lgamma = np.zeros((T - tau, D))
    for j in range(D):
        Xj_f = X[j, tau:]
        lgamma[:, j] = _local_mi(Vp[None, :], Xj_f[None, :], method)
    max_idx = int(np.argmax(np.mean(lgamma, axis=0)))
    lgamma_max = lgamma[:, max_idx]
    return EmergenceResult(
        value=float(np.mean(lgamma_max)),
        v_mi=float(np.mean(lgamma_max)),
        x_mi=0.0,
        locals_={"gamma": lgamma_max} if return_locals else None,
    )
