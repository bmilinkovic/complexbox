"""Closed-form zero-DD solutions and DD-validation routines.

Ports of SSDI-1's ``utils/iss_perfect_dd.m`` and ``utils/dds_check.m``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._grassmann import orthonormalise, rand_orthonormal
from .dd import iss2dd, trfun2dd

__all__ = ["iss_perfect_dd", "dds_check"]


def iss_perfect_dd(
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    uniq: bool = True,
) -> tuple[
    npt.NDArray[np.floating] | list[npt.NDArray[np.floating] | None],
    npt.NDArray[np.floating] | list[npt.NDArray[np.floating] | None],
]:
    """Closed-form orthonormal projections L with exactly zero DD.

    Port of ``iss_perfect_dd.m``. If ``uniq=True``, returns the unique
    C-null (``m = n - r``) and K-null (``m = r``) solutions. Otherwise
    returns hierarchically-nested cell-array-style lists.
    """
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    n, r = C.shape
    if r >= n:
        raise ValueError("No generic perfect solutions when r >= n")

    if uniq:
        # C-null solution, m = n - r
        # LC = orthonormalise([-C(r+1:n, :) / C(1:r, :) eye(n-r)]')
        top = -np.linalg.solve(
            C[:r, :].T, C[r:, :].T
        ).T  # shape (n-r, r) — equivalent to MATLAB's right-divide
        stacked = np.concatenate([top, np.eye(n - r)], axis=1)  # (n-r, n)
        LC = orthonormalise(stacked.T)

        # K-null solution, m = r
        # [~, LK] = orthonormalise([-K(:, 1:r) \ K(:, r+1:n); eye(n-r)])
        top_k = -np.linalg.solve(K[:, :r], K[:, r:])  # shape (r, n-r)
        stacked_k = np.concatenate([top_k, np.eye(n - r)], axis=0)  # (n, n-r)
        _, LK = orthonormalise(stacked_k, return_complement=True)
        return LC, LK

    LC_list: list[npt.NDArray[np.floating] | None] = [None] * n
    LK_list: list[npt.NDArray[np.floating] | None] = [None] * n
    for m in range(1, n - r + 1):
        k = n - m
        top = -np.linalg.solve(C[:r, :].T, C[k:, :].T).T  # (m, r)
        block = np.concatenate([top, np.zeros((m, k - r)), np.eye(m)], axis=1)
        LC_list[m - 1] = orthonormalise(block.T)
    for m in range(r, n):
        k = n - m
        top = -np.linalg.solve(K[:, :r], K[:, m:])  # (r, k)
        block = np.concatenate([top, np.zeros((m - r, k)), np.eye(k)], axis=0)
        _, LKm = orthonormalise(block, return_complement=True)
        LK_list[m - 1] = LKm
    return LC_list, LK_list


def dds_check(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
    m: int,
    nsamples: int = 100,
    rng: np.random.Generator | None = None,
) -> float:
    """Validate spectral DD by comparison with state-space DD.

    Port of ``dds_check.m``. Returns the maximum absolute discrepancy across
    ``nsamples`` random projections. Values < 1e-12 are reasonable.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = C.shape[0]
    L_all = rand_orthonormal(n, m, runs=nsamples, rng=rng)
    if L_all.ndim == 2:
        L_all = L_all[:, :, None]
    derr = 0.0
    for k in range(nsamples):
        d_ss = iss2dd(L_all[:, :, k], A, C, K)
        d_sp, _ = trfun2dd(L_all[:, :, k], H)
        derr = max(derr, abs(d_ss - d_sp))
    return derr
