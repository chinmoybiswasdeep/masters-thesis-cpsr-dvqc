"""V4 IPC: target correctness, capacity recovery, null calibration, saturation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v4_ipc import (  # noqa: E402
    BoundedReadout, IPCConfig, Target, compute_metrics, cross_delay_targets,
    legendre_target, nlong_targets, null_threshold, saturation_report,
    single_delay_targets)


def test_legendre_targets_are_orthonormal_under_the_input_measure():
    nodes, w = np.polynomial.legendre.leggauss(64)
    w = w / 2.0
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            ip = float(np.sum(w * legendre_target(nodes, d1) * legendre_target(nodes, d2)))
            assert abs(ip - (1.0 if d1 == d2 else 0.0)) < 1e-10


def test_target_canonicalisation_prevents_double_counting():
    a = Target(((3, 1), (1, 2)))
    b = Target(((1, 2), (3, 1)))
    assert a == b and a.pairs == ((1, 2), (3, 1))
    with pytest.raises(ValueError):
        Target(((2, 1), (2, 3)))            # repeated delay


def test_cross_delay_target_values_are_the_product_of_their_factors():
    v = np.random.default_rng(0).uniform(-1, 1, 200)
    t = Target(((0, 2), (3, 1)))
    y = t.build(v)
    shifted = np.zeros_like(v)
    shifted[3:] = v[:-3]
    expect = legendre_target(v, 2) * legendre_target(shifted, 1)
    expect[:3] = 0.0
    assert np.allclose(y, expect, atol=1e-12)


def test_target_metadata_identifies_the_capability_boundary():
    assert Target(((0, 3), (4, 1))).reachable_by_product_architecture is True
    assert Target(((4, 2),)).reachable_by_product_architecture is False
    assert Target(((0, 2), (3, 1))).degree_at_positive_delay == 1
    assert Target(((3, 2),)).degree_at_positive_delay == 2


def test_nlong_library_contains_only_reachable_targets_by_default():
    ts = nlong_targets(2, 8, nlong_max_delay=5)
    assert ts and all(t.reachable_by_product_architecture for t in ts)
    assert all(t.max_delay > 2 for t in ts)
    both = nlong_targets(2, 8, reachable_only=False, nlong_max_delay=5)
    assert any(not t.reachable_by_product_architecture for t in both)


def test_target_libraries_have_no_duplicates():
    for lib in (single_delay_targets(4, 8), cross_delay_targets(6, seed=0),
                nlong_targets(2, 8)):
        keys = [t.key() for t in lib]
        assert len(keys) == len(set(keys))


def test_capacity_recovers_a_known_linear_system():
    """A delay line must show capacity exactly at its planted delays."""
    rng = np.random.default_rng(3)
    T = 900
    v = rng.uniform(-1, 1, T)
    X = np.column_stack([np.roll(v, k) for k in (0, 1, 2, 3)])
    X[:4] = 0.0
    cfg = IPCConfig(alpha=1e-3)
    tr, te = cfg.splits(T)
    rd = BoundedReadout(X, tr, te, 1e-3)
    for tau in (0, 1, 2, 3):
        assert rd.capacity(Target(((tau, 1),)).build(v)) > 0.95, tau
    assert rd.capacity(Target(((6, 1),)).build(v)) < 0.2


def test_capacity_is_one_minus_mse_over_var():
    rng = np.random.default_rng(1)
    T = 600
    v = rng.uniform(-1, 1, T)
    X = np.column_stack([v, rng.normal(0, 0.01, T)])
    cfg = IPCConfig(alpha=1e-6)
    tr, te = cfg.splits(T)
    rd = BoundedReadout(X, tr, te, 1e-6)
    y = legendre_target(v, 1)
    c = rd.capacity(y)
    assert 0.0 <= c <= 1.0 and c > 0.95


def test_bounded_readout_rejects_a_zero_penalty():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    with pytest.raises(ValueError):
        BoundedReadout(X, np.arange(200), np.arange(220, 300), 0.0)


def test_alpha_bounds_the_readout_and_so_grades_small_components():
    """The mechanism that makes N(g) graded instead of a step function."""
    rng = np.random.default_rng(5)
    T = 1200
    v = rng.uniform(-1, 1, T)
    cfg = IPCConfig()
    tr, te = cfg.splits(T)
    y = legendre_target(v, 2)
    caps = []
    for eps in (1e-3, 1e-2, 1e-1):
        X = np.column_stack([v, v + eps * legendre_target(v, 2)])
        caps.append(BoundedReadout(X, tr, te, 0.5).capacity(y))
    assert caps[0] < caps[1] < caps[2]         # amplitude-sensitive
    # with a vanishing penalty the same tiny component is fully recoverable
    X = np.column_stack([v, v + 1e-3 * legendre_target(v, 2)])
    assert BoundedReadout(X, tr, te, 1e-10).capacity(y) > caps[0]


def test_null_threshold_is_small_for_an_uninformative_readout():
    rng = np.random.default_rng(7)
    T = 800
    X = rng.normal(size=(T, 6))
    cfg = IPCConfig()
    tr, te = cfg.splits(T)
    rd = BoundedReadout(X, tr, te, 0.5)
    nul = null_threshold(rd, T, n_null=80, seed=0)
    assert 0.0 <= nul["threshold"] < 0.25
    assert nul["n_null"] == 80


def test_saturation_detection_fires_above_the_limit():
    assert saturation_report([0.1] * 10)["saturated"] is False
    bad = saturation_report([0.999] * 5 + [0.1] * 5)
    assert bad["saturated"] is True and bad["saturated_fraction"] == 0.5
    assert "ceiling-pinned" in bad["reason"]


def test_splits_are_disjoint_with_a_guard_gap():
    cfg = IPCConfig(washout=50, train_frac=0.6, gap=15)
    tr, te = cfg.splits(1000)
    assert set(tr).isdisjoint(set(te))
    assert min(te) - max(tr) >= 15
    assert min(tr) >= 50
    with pytest.raises(ValueError):
        cfg.splits(80)


def test_metrics_are_route_restricted():
    """M must come from R only and N from P only, so neither can move with the
    other control."""
    from decoupled_qrc.v4_architecture import V4Spec, MemorySpec, ProcessorSpec, run_v4
    spec = V4Spec(memory=MemorySpec(L_R=3), processor=ProcessorSpec(N_P=3), n_joint=12)
    cfg = IPCConfig(alpha=0.5, max_delay=5, nlong_max_delay=4, n_null=20)
    u = np.random.default_rng(4).uniform(-1, 1, 900)
    a = compute_metrics(run_v4(spec, u, m=0.7, g=0.2, seed=3), cfg)
    b = compute_metrics(run_v4(spec, u, m=0.7, g=0.9, seed=3), cfg)
    c = compute_metrics(run_v4(spec, u, m=0.2, g=0.2, seed=3), cfg)
    assert a.M == b.M                 # M cannot move with g
    assert a.N == c.N                 # N cannot move with m
    assert a.N != b.N and a.M != c.M  # the diagonals do move


def test_unreachable_probe_reports_near_zero_capacity():
    from decoupled_qrc.v4_architecture import V4Spec, MemorySpec, ProcessorSpec, run_v4
    spec = V4Spec(memory=MemorySpec(L_R=4), processor=ProcessorSpec(N_P=4), n_joint=36)
    cfg = IPCConfig(alpha=0.5, n_null=20)
    u = np.random.default_rng(6).uniform(-1, 1, 1200)
    res = compute_metrics(run_v4(spec, u, m=0.9, g=0.9, seed=3), cfg)
    caps = list(res.unreachable_probe["capacities"].values())
    assert caps and max(caps) < 0.15
