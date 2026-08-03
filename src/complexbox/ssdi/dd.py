"""Dynamical dependence: proxy and exact computations.

Ports of SSDI-1's ``utils/iss2cak.m``, ``utils/cak2ddx.m``, ``utils/cak2ddxgrad.m``,
``utils/iss2dd.m``, ``utils/iss2ce.m``, ``utils/iss2ce_precomp.m``, and the
spectral DD helpers ``trfun2dd``, ``trfun2ddgrad``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve, cholesky, svd

from ..mvgc._lyap import dlyap, mdare
from ..mvgc._utils import logdet

__all__ = [
    "iss2cak",
    "cak2ddx",
    "cak2ddxgrad",
    "iss2dd",
    "iss2ce_precomp",
    "iss2ce",
    "trfun2dd",
    "trfun2ddgrad",
    "trfun2dd_pointwise",
    "trfun2dd_band",
    "trfun2dd_bandgrad",
]


def iss2cak(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    lags: int | None = None,
) -> npt.NDArray[np.floating]:
    """Sequence CA^{k-1}K, k = 1, ..., r. Port of ``iss2cak.m``.

    Returns array of shape ``(n, n, lags)``. If ``lags`` is None, uses the
    state dimension ``r`` (matching the MATLAB default).
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    r = A.shape[0]
    n = C.shape[0]
    if lags is None:
        lags = r
    Ak = np.eye(r)
    CAK = np.zeros((n, n, lags))
    CAK[:, :, 0] = C @ K
    for k in range(1, lags):
        Ak = Ak @ A
        CAK[:, :, k] = C @ Ak @ K
    return CAK


def cak2ddx(L: npt.NDArray[np.floating], CAK: npt.NDArray[np.floating]) -> float:
    """Proxy dynamical dependence under identity residuals. Port of ``cak2ddx.m``.

    ``D = sum_k ||L' CAK_k||_F^2 - ||L' CAK_k L||_F^2``. Vectorised over
    lags via ``np.einsum``.
    """
    L = np.asarray(L, dtype=float)
    CAK = np.asarray(CAK, dtype=float)
    # L' CAK_k for all k:  (m, n, r) tensor
    LCAK = np.einsum("nm,nkr->mkr", L, CAK.transpose(0, 1, 2))
    # Above expects CAK of shape (n, n, r): so reshape semantics:
    # einsum("nm,nkr->mkr", L, CAK)  with L (n, m), CAK (n, n, r) gives (m, n, r)
    LCAKL = np.einsum("mkr,kj->mjr", LCAK, L)
    return float(np.sum(LCAK * LCAK) - np.sum(LCAKL * LCAKL))


def cak2ddxgrad(
    L: npt.NDArray[np.floating], CAK: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], float]:
    """Grassmannian gradient of the proxy DD. Port of ``cak2ddxgrad.m``.

    Returns ``(G, |G|)`` with ``G`` projected onto the Grassmannian tangent.
    Vectorised across lags.
    """
    L = np.asarray(L, dtype=float)
    CAK = np.asarray(CAK, dtype=float)
    P = L @ L.T
    # g = sum_k (Q Q' - Q' P Q - Q P Q')
    # Vectorised: contract over the lag axis directly.
    Q = CAK  # (n, n, r)
    # sum_k Q_k Q_k.T
    gA = np.einsum("ikr,jkr->ij", Q, Q)
    # sum_k Q_k.T P Q_k
    gB = np.einsum("kir,kl,ljr->ij", Q, P, Q)
    # sum_k Q_k P Q_k.T
    gC = np.einsum("ikr,kl,jlr->ij", Q, P, Q)
    g = gA - gB - gC
    G = 2.0 * g @ L
    G = G - P @ G  # project to Grassmannian tangent
    return G, float(np.sqrt(np.sum(G * G)))


def iss2dd(
    L: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
) -> float:
    """Exact dynamical dependence of projection L for innovations-form SS.

    Port of ``iss2dd.m``. Assumes identity residuals covariance.
    """
    L = np.asarray(L, dtype=float)
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    # Reduced model: project C → L'C, the noise driving the projected
    # innovations is L'·(KK')·L. DARE has cross-term S = K @ L.
    Cred = L.T @ C
    Q = K @ K.T
    S = K @ L
    R = np.eye(L.shape[1])  # L'·I·L = I (since L is orthonormal)
    _, V, rep, _ = mdare(A, Cred, Q, R, S)
    if rep < 0 or rep > 1e-8:
        return float("nan")
    return logdet(V)


def iss2ce_precomp(
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Precompute ``(G, P)`` for amortised :func:`iss2ce` calls.

    ``G`` is the observable covariance; ``P[:, :, i]`` are the per-output
    prediction-error DARE solutions.
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    n = C.shape[0]
    r = A.shape[0]
    M = dlyap(A, K @ V @ K.T)
    G = C @ M @ C.T + V
    P = np.zeros((r, r, n))
    for i in range(n):
        Ci = C[i : i + 1, :]
        Vi = V[i : i + 1, i : i + 1]
        Si = K @ V[:, i : i + 1]
        _, _, _, Pi = mdare(A, Ci, K @ V @ K.T, Vi, Si)
        P[:, :, i] = Pi
    return G, P


def iss2ce(
    L: npt.NDArray[np.floating],
    A: npt.NDArray[np.floating],
    C: npt.NDArray[np.floating],
    K: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    G: npt.NDArray[np.floating] | None = None,
    P: npt.NDArray[np.floating] | None = None,
) -> tuple[float, float]:
    """Return co-information and dynamical dependence for projection ``L``.

    This is a direct port of SSDI-1 ``iss2ce.m`` and does *not* assume an
    identity innovations covariance.  Causal emergence is ``CI - DD``.
    Precomputed ``G`` and ``P`` may be supplied independently.
    """
    L = np.asarray(L, dtype=float)
    A, C, K, V = (np.asarray(x, dtype=float) for x in (A, C, K, V))
    n = C.shape[0]

    Vchol = np.linalg.cholesky(V)
    KV = K @ Vchol
    KVK = KV @ KV.T
    LV = L.T @ Vchol
    LVL = LV @ LV.T

    if P is None:
        r = A.shape[0]
        P = np.zeros((r, r, n))
        for i in range(n):
            _, _, rep, Pi = mdare(
                A,
                C[i : i + 1, :],
                KVK,
                V[i : i + 1, i : i + 1],
                K @ V[:, i : i + 1],
            )
            if rep < 0:
                return float("nan"), float("nan")
            P[:, :, i] = Pi

    # I1: sum of projected per-element prediction-error log determinants.
    I1 = 0.0
    for i in range(n):
        V_i = C @ P[:, :, i] @ C.T + V
        I1 += logdet(L.T @ V_i @ L)

    # I2: innovations covariance of the reduced projected process.
    _, VR, rep, _ = mdare(A, L.T @ C, KVK, LVL, KV @ LV.T)
    if rep < 0:
        return float("nan"), float("nan")
    I2 = logdet(VR)

    if G is None:
        M = dlyap(A, KVK)
        G = C @ M @ C.T + V
    I3 = (n - 1) * logdet(L.T @ G @ L)
    I4 = logdet(LVL)

    ci = I1 - I4 - I3
    dd = I2 - I4
    return ci, dd


def trfun2dd(
    L: npt.NDArray[np.floating], H: npt.NDArray[np.complexfloating]
) -> tuple[float, npt.NDArray[np.floating]]:
    r"""Spectral (frequency-integrated) dynamical dependence.

    For identity innovations covariance, Eq. (24) of Barnett & Seth (2023)
    in this module's column-basis convention is

    .. math::

       D(L)=\frac{1}{\pi}\int_0^\pi
       \log\det\!\left[L^\mathsf{T}H(\omega)H(\omega)^*L\right]d\omega.

    The one-sided form is equivalent to the paper's ``1/(2π)`` integral over
    ``[0, 2π]``.  To match MATLAB SSDI-1 exactly, the returned frequency
    samples ``d`` are half-log-determinants and ``D`` uses the corresponding
    doubled trapezoid sum.  Assumes identity innovations covariance.
    """
    L = np.asarray(L, dtype=float)
    H = np.asarray(H)
    n, _, h = H.shape
    d = np.zeros(h)
    for k in range(h):
        Hk = H[:, :, k]
        HL = Hk.conj().T @ L
        M = HL.conj().T @ HL
        try:
            R = cholesky(M, lower=False, check_finite=False)
            d[k] = float(np.real(np.sum(np.log(np.diag(R)))))
        except np.linalg.LinAlgError:
            # Match the determinant continuously if numerical rank loss makes
            # Cholesky unavailable; valid SSDI inputs normally take the HPD path.
            singular = svd(HL, compute_uv=False, check_finite=False)
            d[k] = float(np.sum(np.log(singular)))
    # MATLAB: sum(d(1:end-1) + d(2:end)) / (h - 1).  Since d is half the
    # paper's integrand, this is its normalised one-sided trapezoidal average.
    D = float(np.sum(d[:-1] + d[1:]) / (h - 1))
    return D, d


def trfun2ddgrad(
    L: npt.NDArray[np.floating], H: npt.NDArray[np.complexfloating]
) -> tuple[npt.NDArray[np.floating], float]:
    r"""Grassmannian gradient of :func:`trfun2dd`.

    In the column-basis convention, Appendix D, Eq. (D5), becomes

    .. math::

       \nabla D = 2\left\langle
       \Re\{S(\omega)L[L^\mathsf{T}S(\omega)L]^{-1}\}\right\rangle_\omega
       -2L,

    where ``S(ω) = H(ω)H(ω)*``.  The ``-2L`` term is already the canonical
    Grassmann projection; applying another tangent projection first would
    subtract the normal component twice.  This follows MATLAB
    ``trfun2ddgrad.m`` exactly.
    """
    L = np.asarray(L, dtype=float)
    H = np.asarray(H)
    n, _, h = H.shape
    g_stack = np.zeros((h, *L.shape))
    for k in range(h):
        Hk = H[:, :, k]
        HL = Hk.conj().T @ L
        M = HL.conj().T @ HL
        numerator = Hk @ HL
        # MATLAB right division: numerator / M.  This is half the Euclidean
        # derivative; the doubled trapezoid sum below restores its factor 2.
        try:
            g_stack[k] = np.real(numerator @ np.linalg.inv(M))
        except np.linalg.LinAlgError:
            # MATLAB's right-division is undefined here; the pseudoinverse
            # keeps diagnostics finite for a rank-deficient trial basis.
            g_stack[k] = np.real(numerator @ np.linalg.pinv(M))
    g = np.sum(g_stack[:-1] + g_stack[1:], axis=0) / (h - 1) - 2.0 * L
    return g, float(np.sqrt(np.sum(g * g)))


def _band_nodes_and_weights(
    frequencies: npt.NDArray[np.floating],
    band: tuple[float, float] | None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
]:
    """Build exact-edge trapezoid nodes and their grid interpolation stencils."""
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if frequencies.size < 2 or np.any(~np.isfinite(frequencies)):
        raise ValueError("frequencies must be a finite vector with at least two values")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    if band is None:
        low, high = frequencies[0], frequencies[-1]
    else:
        low, high = map(float, band)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("band edges must be finite")
        if high <= low:
            raise ValueError("band must satisfy low < high")
        if low < frequencies[0] or high > frequencies[-1]:
            raise ValueError("band must lie within the supplied frequency interval")

    interior = frequencies[(frequencies > low) & (frequencies < high)]
    nodes = np.concatenate(([low], interior, [high]))
    spacing = np.diff(nodes)
    weights = np.empty(nodes.size)
    weights[0] = spacing[0] / 2.0
    weights[-1] = spacing[-1] / 2.0
    if nodes.size > 2:
        weights[1:-1] = (spacing[:-1] + spacing[1:]) / 2.0
    weights /= high - low

    # Each exact boundary/interior node is represented as a linear
    # interpolation of two adjacent supplied samples. Grid-aligned nodes have
    # fraction 0 (or 1 at the final endpoint), so this also covers the full
    # interval without a special case.
    left = np.searchsorted(frequencies, nodes, side="right") - 1
    left = np.clip(left, 0, frequencies.size - 2).astype(np.intp)
    right = left + 1
    fraction = (nodes - frequencies[left]) / (frequencies[right] - frequencies[left])
    return nodes, weights, left, right, fraction


def _pointwise_dd_value_grad(
    L: npt.NDArray[np.floating],
    Hk: npt.NDArray[np.complexfloating],
) -> tuple[float, npt.NDArray[np.floating]]:
    """Paper Eq. (25) value and Euclidean derivative at one frequency."""
    S = Hk @ Hk.conj().T
    numerator = L.T @ S @ L
    numerator = 0.5 * (numerator + numerator.conj().T)
    projected_transfer = L.T @ Hk @ L
    denominator = projected_transfer @ projected_transfer.conj().T
    denominator = 0.5 * (denominator + denominator.conj().T)
    try:
        chol_num = cholesky(numerator, lower=True, check_finite=False)
        chol_den = cholesky(denominator, lower=True, check_finite=False)
        logdet_num = 2.0 * float(np.sum(np.log(np.real(np.diag(chol_num)))))
        logdet_den = 2.0 * float(np.sum(np.log(np.real(np.diag(chol_den)))))
        numerator_inv = cho_solve(
            (chol_num, True), np.eye(L.shape[1], dtype=numerator.dtype), check_finite=False
        )
        projected_inv = np.linalg.solve(
            projected_transfer,
            np.eye(L.shape[1], dtype=projected_transfer.dtype),
        )
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "spectral DD is undefined for a singular projected transfer matrix"
        ) from exc

    value = logdet_num - logdet_den
    grad_numerator = 2.0 * np.real(S @ L @ numerator_inv)
    grad_denominator = 2.0 * np.real(
        Hk @ L @ projected_inv + Hk.conj().T @ L @ projected_inv.conj().T
    )
    return value, grad_numerator - grad_denominator


def trfun2dd_pointwise(
    L: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.floating]:
    r"""Return the paper's pointwise spectral DD, Eq. (25).

    In this package's column-basis convention,

    .. math::

       f_L(\omega)=\log\frac{\det[L^\mathsf{T}HH^*L]}
       {\det[(L^\mathsf{T}HL)(L^\mathsf{T}HL)^*]}.

    This is distinct from the half-log-determinant quadrature samples returned
    by legacy MATLAB-parity :func:`trfun2dd`.
    """
    L = np.asarray(L, dtype=float)
    H = np.asarray(H)
    return np.asarray([_pointwise_dd_value_grad(L, H[:, :, k])[0] for k in range(H.shape[2])])


def trfun2dd_band(
    L: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
    frequencies: npt.NDArray[np.floating],
    band: tuple[float, float] | None = None,
) -> tuple[float, npt.NDArray[np.floating]]:
    r"""Return band-limited spectral DD and selected pointwise values.

    This implements paper Eq. (26),
    ``(ω2-ω1)^-1 integral_[ω1,ω2] f_L(ω) dω``, with trapezoidal weights.
    Off-grid band edges are evaluated by linear interpolation of the adjacent
    pointwise Eq. (25) samples, so the normalisation uses the exact requested
    interval. ``band=None`` selects the full supplied interval (paper Eq. 27).

    The returned values are ordered at the augmented integration nodes: the
    exact lower edge, supplied grid points strictly inside the band, and the
    exact upper edge.
    """
    L = np.asarray(L, dtype=float)
    H = np.asarray(H)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if H.shape[2] != frequencies.size:
        raise ValueError("frequencies length must equal H.shape[2]")
    _, weights, left, right, fraction = _band_nodes_and_weights(frequencies, band)
    needed = np.unique(np.concatenate((left, right)))
    grid_values = {int(k): _pointwise_dd_value_grad(L, H[:, :, k])[0] for k in needed}
    values = np.asarray(
        [
            (1.0 - alpha) * grid_values[int(lo)] + alpha * grid_values[int(hi)]
            for lo, hi, alpha in zip(left, right, fraction)  # noqa: B905 - common node count
        ]
    )
    return float(weights @ values), values


def trfun2dd_bandgrad(
    L: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
    frequencies: npt.NDArray[np.floating],
    band: tuple[float, float] | None = None,
) -> tuple[npt.NDArray[np.floating], float]:
    """Grassmann gradient of :func:`trfun2dd_band`."""
    L = np.asarray(L, dtype=float)
    H = np.asarray(H)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if H.shape[2] != frequencies.size:
        raise ValueError("frequencies length must equal H.shape[2]")
    _, weights, left, right, fraction = _band_nodes_and_weights(frequencies, band)
    needed = np.unique(np.concatenate((left, right)))
    grid_gradients = {int(k): _pointwise_dd_value_grad(L, H[:, :, k])[1] for k in needed}
    euclidean = np.zeros_like(L)
    for weight, lo, hi, alpha in zip(  # noqa: B905 - common node count
        weights, left, right, fraction
    ):
        euclidean += weight * (
            (1.0 - alpha) * grid_gradients[int(lo)] + alpha * grid_gradients[int(hi)]
        )
    gradient = np.real(euclidean - L @ (L.T @ euclidean))
    return gradient, float(np.linalg.norm(gradient))
