"""
test_discovery_sampling.py -- V2.1 Defect 3 / test requirement #10:
reproducible Sobol/LHS 3D sampling, discovery/confirmation seed
separation.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.discovery_sampling import (sobol_3d_samples, latin_hypercube_3d_samples,  # noqa: E402
                                               to_raw_candidates, discovery_seeds, confirmation_seeds,
                                               assert_disjoint_discovery_confirmation)
from decoupled_qrc.validation_utils import ControlRange  # noqa: E402


def test_sobol_samples_in_unit_cube():
    pts = sobol_3d_samples(16, seed=0)
    assert pts.shape == (16, 3)
    assert np.all(pts >= 0.0) and np.all(pts <= 1.0)


def test_sobol_samples_deterministic():
    a = sobol_3d_samples(8, seed=1)
    b = sobol_3d_samples(8, seed=1)
    assert np.array_equal(a, b)


def test_sobol_samples_different_seed_gives_different_points():
    a = sobol_3d_samples(8, seed=1)
    b = sobol_3d_samples(8, seed=2)
    assert not np.array_equal(a, b)


def test_sobol_non_power_of_two_count():
    pts = sobol_3d_samples(10, seed=0)
    assert pts.shape == (10, 3)


def test_latin_hypercube_samples_in_unit_cube_and_deterministic():
    a = latin_hypercube_3d_samples(12, seed=3)
    b = latin_hypercube_3d_samples(12, seed=3)
    assert a.shape == (12, 3)
    assert np.array_equal(a, b)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)


def test_to_raw_candidates_respects_ranges():
    m_range, g_range, J_range = ControlRange("m", 0.1, 1.0), ControlRange("g", 0.05, 0.6), ControlRange("J", 0.05, 0.6)
    pts = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    cands = to_raw_candidates(pts, m_range, g_range, J_range)
    assert cands[0]["m"] == pytest.approx(0.1)
    assert cands[0]["g"] == pytest.approx(0.325)
    assert cands[0]["J"] == pytest.approx(0.6)
    assert cands[1]["m"] == pytest.approx(1.0)
    assert cands[1]["J"] == pytest.approx(0.325)


def test_discovery_and_confirmation_seed_ranges_never_overlap():
    assert_disjoint_discovery_confirmation(n_discovery=50, n_confirmation=5)  # must not raise
    disc = {discovery_seeds(k).reservoir_idx for k in range(50)}
    conf = {confirmation_seeds(k).reservoir_idx for k in range(5)}
    assert disc.isdisjoint(conf)
    assert all(idx < 0 for idx in disc)
    assert all(idx >= 0 for idx in conf)


def test_discovery_seeds_deterministic_and_distinct():
    a = discovery_seeds(3)
    b = discovery_seeds(3)
    c = discovery_seeds(4)
    assert a == b
    assert a.reservoir_seed != c.reservoir_seed


def test_confirmation_seeds_match_plain_nested_seeds_convention():
    """Confirmation seeds must be ORDINARY `make_nested_seeds` calls (the
    same convention every other V2/V2.1 module already uses for real
    reservoir seeds) -- only discovery gets the special negative-index
    convention."""
    from decoupled_qrc.validation_utils import make_nested_seeds
    assert confirmation_seeds(2, input_idx=1) == make_nested_seeds(2, 1)
