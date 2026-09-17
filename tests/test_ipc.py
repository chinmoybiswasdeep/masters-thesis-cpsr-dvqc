"""
test_ipc.py -- orthogonality of the generated Legendre targets, and a
synthetic-signal recovery check where the true IPC is known analytically
(Part 7's required unit tests).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.ipc import (  # noqa: E402
    legendre_target, to_v, generate_profiles, build_target, compute_ipc, Profile,
)
from qrc_qiskit import chrono_split, random_input  # noqa: E402


def test_legendre_orthonormality_gauss_quadrature():
    """integral_{-1}^1 L_d(v) L_e(v) * (1/2) dv == delta_{de}, checked to high
    accuracy via Gauss-Legendre quadrature (exact for polynomials up to
    degree 2*n_nodes-1, so this is not merely 'close' the way a coarse
    trapezoid rule would be -- it would FAIL under a wrong normalization
    constant or a wrong recursion)."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    for d in range(0, 7):
        for e in range(0, 7):
            integral = 0.5 * np.sum(weights * legendre_target(nodes, d) * legendre_target(nodes, e))
            expected = 1.0 if d == e else 0.0
            assert abs(integral - expected) < 1e-9, (d, e, integral)


def test_legendre_l1_is_not_degenerate_under_wrong_sign():
    """A convention/sign bug in `to_v` or `legendre_target` would be invisible
    if this test only checked symmetric quantities -- explicitly check L1 is
    monotonically increasing and odd (L1(-v) == -L1(v)), which a sign flip
    would break."""
    v = np.linspace(-1, 1, 101)
    l1 = legendre_target(v, 1)
    assert np.all(np.diff(l1) > 0)
    assert np.allclose(l1, -legendre_target(-v, 1), atol=1e-12)


def test_to_v_maps_01_to_negative11():
    u = np.array([0.0, 0.5, 1.0])
    v = to_v(u)
    assert np.allclose(v, [-1.0, 0.0, 1.0])


def test_generate_profiles_no_duplicates_and_correct_degree():
    profiles = generate_profiles(max_delay=6, max_degree=4, max_targets_per_degree=200, seed=0)
    for d in range(1, 5):
        plist = profiles[d]
        assert len(plist) == len(set(plist)), f"duplicate profiles at degree {d}"
        for p in plist:
            assert p.total_degree == d
            delays = [tau for tau, _ in p.pairs]
            assert len(delays) == len(set(delays)), "profile reuses a delay"


def test_generate_profiles_known_counts_small_case():
    """max_delay=2 (delays 0,1,2), degree=2: partitions of 2 with parts<=2 are
    [2] (single delay, degree 2) and [1,1] (two distinct delays, degree 1
    each). [2]: 3 choices of delay (C(3,1)). [1,1]: C(3,2)=3 choices of an
    unordered delay pair. Total = 3 + 3 = 6 -- hand-computed ground truth."""
    profiles = generate_profiles(max_delay=2, max_degree=2, max_targets_per_degree=1000, seed=0)
    assert len(profiles[1]) == 3   # L1 at delay 0, 1, or 2
    assert len(profiles[2]) == 6


def test_ipc_recovers_known_linear_and_nonlinear_signal():
    """Build reservoir features that are EXACTLY [L1(v_{t-1}), L2(v_{t-3}),
    random noise columns], and a target-adjacent reservoir where the true
    IPC_1 and IPC_2 contributions are known: a feature set containing the
    L1(v_{t-1}) target itself must recover memory capacity ~1 at degree 1,
    and a feature set containing L2(v_{t-3}) must recover ~1 at degree 2 --
    this is the closest thing to an analytic ground truth compute_ipc can be
    checked against (real quantum-reservoir features have no known closed
    form)."""
    rng = np.random.RandomState(0)
    T = 400
    u = random_input(T, seed=1)
    v = to_v(u)

    l1_lag1 = np.zeros(T)
    l1_lag1[1:] = legendre_target(v[:-1], 1)
    l2_lag3 = np.zeros(T)
    l2_lag3[3:] = legendre_target(v[:-3], 2)
    noise = rng.normal(0, 1, size=(T, 5))
    X = np.column_stack([l1_lag1, l2_lag3, noise])

    train, val, test = chrono_split(T, washout=10, n_val=60, n_test=100, gap=9)
    result = compute_ipc(u, X, train, val, test, max_delay=6, max_degree=3,
                          max_targets_per_degree=40, n_surrogates=6, seed=0)

    assert result.ipc_by_degree[1] > 0.8, "should recover ~1 from the planted L1(v_{t-1}) feature"
    assert result.ipc_by_degree[2] > 0.8, "should recover ~1 from the planted L2(v_{t-3}) feature"
    assert result.memory == result.ipc_by_degree[1]
    assert result.nonlinearity == pytest.approx(
        sum(result.ipc_by_degree[d] for d in range(2, 4)))


def test_ipc_pure_noise_features_give_near_zero_capacity():
    """With reservoir features that are pure noise (no relation to any
    target), the significance filter should reject essentially all targets,
    giving IPC_d close to zero at every degree -- proving the significance
    filter isn't vacuous (Part 7's 'finite-data significance filtering'
    requirement)."""
    rng = np.random.RandomState(2)
    T = 300
    u = random_input(T, seed=3)
    X = rng.normal(0, 1, size=(T, 10))
    train, val, test = chrono_split(T, washout=10, n_val=50, n_test=80, gap=7)
    result = compute_ipc(u, X, train, val, test, max_delay=5, max_degree=2,
                          max_targets_per_degree=20, n_surrogates=6, seed=0)
    assert result.total < 0.5, f"pure-noise features should score near zero, got {result.total}"
