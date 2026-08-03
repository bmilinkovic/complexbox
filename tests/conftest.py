"""Pytest configuration and shared fixtures for complexbox tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def rng() -> np.random.Generator:
    """Fresh deterministic generator so tests are independent of run order."""
    return np.random.default_rng(20260516)


def _load_fixture(name: str):
    """Load a MATLAB-generated .mat file, returning the dict.

    Tests are auto-skipped if SciPy or the file is unavailable.
    """
    from scipy.io import loadmat

    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(
            f"fixture {name} not generated (run tools/matlab_fixtures/generate_all_fixtures.m)"
        )
    return loadmat(path, struct_as_record=False, squeeze_me=True)


@pytest.fixture
def mvgc_fixture():
    return _load_fixture("mvgc_basic.mat")


@pytest.fixture
def ssdi_fixture():
    return _load_fixture("ssdi_basic.mat")


@pytest.fixture
def elph_fixture():
    return _load_fixture("elph_basic.mat")
