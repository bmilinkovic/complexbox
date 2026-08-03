"""Tests for conditional / groupwise / information-optimised GC."""

from __future__ import annotations

import numpy as np
import pytest

from complexbox import mvgc


@pytest.fixture
def small_var(rng):
    A = mvgc.var_rand(5, 2, rho=0.7, rng=rng)
    V = mvgc.corr_rand(5, rng=rng)
    return A, V


def test_var_iogc_matches_ss_iogc(small_var):
    A, V = small_var
    A_ss, C, K, _ = mvgc.var_to_ss(A, V)
    F_var_in = mvgc.var_to_iogc(A, V, "in")
    F_ss_in = mvgc.ss_to_iogc(A_ss, C, K, V, "in")
    diff = np.nanmax(np.abs(F_var_in - F_ss_in))
    assert diff < 1e-10
    F_var_out = mvgc.var_to_iogc(A, V, "out")
    F_ss_out = mvgc.ss_to_iogc(A_ss, C, K, V, "out")
    assert np.nanmax(np.abs(F_var_out - F_ss_out)) < 1e-10


def test_var_gwcgc_matches_ss_gwcgc(small_var):
    A, V = small_var
    A_ss, C, K, _ = mvgc.var_to_ss(A, V)
    groups = [[0, 1], [2], [3, 4]]
    F_var = mvgc.var_to_gwcgc(A, V, groups)
    F_ss = mvgc.ss_to_gwcgc(A_ss, C, K, V, groups)
    diff = np.nanmax(np.abs(F_var - F_ss))
    assert diff < 1e-10


def test_singleton_gwcgc_equals_pwcgc(small_var):
    """When each group is a single variable, groupwise GC should equal pairwise GC."""
    A, V = small_var
    n = V.shape[0]
    groups = [[i] for i in range(n)]
    F_gw = mvgc.var_to_gwcgc(A, V, groups)
    F_pw = mvgc.var_to_pwcgc(A, V)
    diff = np.nanmax(np.abs(F_gw - F_pw))
    assert diff < 1e-10


def test_var_cggc_scalar(small_var):
    A, V = small_var
    val = mvgc.var_to_cggc(A, V, x=[0, 1, 2])
    assert np.isfinite(val)


def test_check_group_validates(rng):
    from complexbox.mvgc._utils import check_group

    # Overlapping indices should fail
    with pytest.raises(ValueError):
        check_group([[0, 1], [1, 2]])
    # Out-of-range indices should fail
    with pytest.raises(ValueError):
        check_group([[0, 5]], n=3)
    g, sizes = check_group([[0, 1], [2], [3, 4, 5]])
    assert g == 3
    assert sizes.tolist() == [2, 1, 3]
