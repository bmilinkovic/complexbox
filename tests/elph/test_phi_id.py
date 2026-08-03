"""PhiID and emergence self-consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import elph, mvgc


def test_gaussian_mi_recovers_analytic(rng):
    rho = 0.6
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = L @ rng.standard_normal((2, 100_000))
    X, Y = Z[0:1], Z[1:2]
    mi = elph.gaussian_mi(X, Y)
    mi_true = -0.5 * np.log(1.0 - rho * rho)
    assert mi == pytest.approx(mi_true, abs=1e-2)


def test_phi_id_atoms_sum_to_total_info(rng):
    """All 16 PhiID atoms should sum to the total time-delayed MI I(X^past; Y^future)."""
    A = np.zeros((2, 2, 1))
    A[:, :, 0] = [[0.5, 0.0], [0.2, 0.5]]
    V = np.eye(2)
    X, _ = mvgc.var_to_tsdata(A, V, m=10_000, N=1, rng=rng)
    phi = elph.phi_id_full(X, tau=1, measure="MMI")
    s = sum(phi.atoms.values())
    # Compute the total info directly
    sd = X.std(axis=1, ddof=1, keepdims=True)
    sX = X / sd
    mi_tot = float(np.mean(elph.gaussian_local_mi(sX[:, :-1], sX[:, 1:])))
    assert s == pytest.approx(mi_tot, abs=1e-2)


def test_lz76_all_zeros():
    assert elph.lz76(np.zeros(100, dtype=np.intp)) == 2


def test_lz76_alternating():
    # Sequence 0101...0101 of length 100: complexity should be 3
    assert elph.lz76(np.tile([0, 1], 50)) == 3


def test_lz76_random_is_high(rng):
    """Random sequence has LZ76 ≈ n/log2(n) asymptotically."""
    n = 2_000
    seq = rng.integers(0, 2, n)
    c = elph.lz76(seq)
    expected = n / np.log2(n)
    assert 0.5 * expected < c < 1.5 * expected


def test_transfer_entropy_directional(rng):
    """For X → Y AR coupling, TE(X→Y) >> TE(Y→X)."""
    T = 5000
    X = rng.standard_normal(T)
    Y = np.zeros(T)
    for t in range(1, T):
        Y[t] = 0.5 * Y[t - 1] + 0.7 * X[t - 1] + 0.1 * rng.standard_normal()
    te_xy = elph.transfer_entropy_gaussian(X[None, :], Y[None, :], tau=1)
    te_yx = elph.transfer_entropy_gaussian(Y[None, :], X[None, :], tau=1)
    assert te_xy > 0.5
    assert te_yx < 0.05
    assert te_xy > 10 * te_yx


def test_maximal_cliques_example():
    A = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    cliques = elph.maximal_cliques(A)
    assert sorted(map(tuple, cliques)) == [(0, 1, 2), (1, 2, 3), (3, 4)]


def test_discretize_quantile_uniform():
    x = np.arange(100, dtype=float)
    bins, _edges = elph.discretize_quantile(x, n=4)
    # Each bin should have ~25 elements
    for k in range(4):
        assert abs((bins == k).sum() - 25) <= 1


def test_emergence_psi_iid_micro_is_zero(rng):
    """For independent micro variables, Ψ should be near zero."""
    X = rng.standard_normal((3, 10_000))
    V = X.mean(axis=0)
    psi = elph.emergence_psi(X, V, tau=1)
    assert abs(psi.value) < 0.01
