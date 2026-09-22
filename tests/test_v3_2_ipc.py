"""V3.2 IPC: equivalence with the repo estimator, target hygiene, classification."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc import ipc as ipc_repo  # noqa: E402
from decoupled_qrc.v3_2_ipc import (  # noqa: E402
    RidgeBank, adaptive_max_delay, compute_ipc_report, profile_capacities)
from qrc_qiskit import chrono_split  # noqa: E402


def _toy(T=600, seed=3):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, T)
    v = 2 * u - 1
    X = np.column_stack([np.roll(v, 1), np.roll(v, 2), np.roll(v, 3),
                         v ** 2, v, rng.normal(0, 0.05, T)])
    return u, X


def test_fast_scorer_matches_repo_estimator():
    """The optimisation must not be a redefinition: V3.2's fast path has to
    reproduce `ipc._capacity_score` to machine precision."""
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    bank = RidgeBank(X, tr, va, te)
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(10):
        y = np.roll(2 * u - 1, k) if k % 2 == 0 else rng.normal(size=len(u))
        worst = max(worst, abs(bank.score(y)[0] - ipc_repo._capacity_score(X, y, tr, va, te)))
    assert worst < 1e-10


def test_fixed_alpha_path_skips_the_validation_search():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    bank = RidgeBank(X, tr, va, te)
    y = np.roll(2 * u - 1, 1)
    _, chosen = bank.score(y)
    frozen, used = bank.score(y, alpha=chosen)
    assert used == chosen
    assert abs(frozen - bank.score(y)[0]) < 1e-12


def test_targets_are_orthonormal_under_the_input_measure():
    """Gauss-quadrature check of sqrt(2d+1) P_d under Uniform[-1,1]."""
    nodes, w = np.polynomial.legendre.leggauss(64)
    w = w / 2.0
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            a = ipc_repo.legendre_target(nodes, d1)
            b = ipc_repo.legendre_target(nodes, d2)
            assert abs(float(np.sum(w * a * b)) - (1.0 if d1 == d2 else 0.0)) < 1e-10


def test_no_duplicate_targets_are_generated():
    profiles = ipc_repo.generate_profiles(6, 4, 100, seed=0, always_include_single_delays=True)
    profiles.pop("_was_capped")
    seen = set()
    for d, plist in profiles.items():
        for p in plist:
            assert p.pairs not in seen
            seen.add(p.pairs)


def test_cross_delay_targets_are_never_classified_as_instantaneous():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    records, _, _, _ = profile_capacities(u, X, tr, va, te, max_delay=4, max_degree=3,
                                          max_targets_per_degree=40, n_surrogates=3)
    inst = [r for r in records if r.degree >= 2 and r.delays == (0,)]
    for r in inst:
        assert r.max_delay == 0 and r.n_delays == 1
    mixed = [r for r in records if r.degree >= 2 and 0 in r.delays and r.n_delays > 1]
    assert mixed, "the toy setup should produce at least one cross-delay target with a tau=0 factor"
    for r in mixed:
        assert r.delays != (0,)


def test_null_targets_have_near_zero_capacity():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    bank = RidgeBank(X, tr, va, te)
    rng = np.random.default_rng(1)
    scores = [bank.score(rng.normal(size=len(u)))[0] for _ in range(30)]
    assert float(np.mean(scores)) < 0.08


def test_synthetic_linear_memory_is_recovered_at_the_planted_delays():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=5, max_degree=2,
                             max_targets_per_degree=30, n_surrogates=9,
                             min_train_absolute=0)
    prof = rep.delay_profile
    assert prof[1] > 0.8 and prof[2] > 0.8 and prof[3] > 0.8
    assert prof[5] < 0.3


def test_synthetic_instantaneous_nonlinearity_is_detected():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=3, max_degree=3,
                             max_targets_per_degree=30, n_surrogates=9,
                             min_train_absolute=0)
    assert rep.degree_profile.get(2, 0.0) > 0.5      # v**2 was planted
    assert rep.NL_0 > 0.5


def test_a_purely_linear_feature_set_shows_no_instantaneous_nonlinearity():
    rng = np.random.default_rng(11)
    T = 600
    u = rng.uniform(0, 1, T)
    v = 2 * u - 1
    X = np.column_stack([v, np.roll(v, 1), rng.normal(0, 0.02, T)])   # affine only
    tr, va, te = chrono_split(T, washout=20, n_val=120, n_test=150, gap=6)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=3, max_degree=3,
                             max_targets_per_degree=20, n_surrogates=9, min_train_absolute=0)
    assert rep.NL_0 < 0.15


def test_report_carries_raw_null_bias_corrected_and_legacy():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=4, max_degree=3,
                             max_targets_per_degree=20, n_surrogates=9, min_train_absolute=0)
    assert rep.NL_0_raw >= rep.NL_0                      # bias correction only subtracts
    assert rep.NL_0_null >= 0.0
    assert rep.M_long_raw >= rep.M_long
    assert rep.NL_0_legacy >= 0.0                        # legacy metric still reported


def test_effective_rank_and_sample_size_requirement_are_enforced():
    u, X = _toy(T=600)
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=3, max_degree=2,
                             max_targets_per_degree=10, n_surrogates=5)
    assert rep.sample_size_required >= 500              # the absolute floor binds here
    assert rep.sample_size_ok is False                  # n_train is ~300: V3.1 ran on 43
    assert rep.numerical_rank <= X.shape[1]
    assert rep.effective_rank > 0


def test_ceiling_audit_is_attached_to_both_metrics():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=3, max_degree=2,
                             max_targets_per_degree=10, n_surrogates=5, min_train_absolute=0)
    assert rep.ceiling_NL0 and rep.ceiling_M


def test_adaptive_delay_stops_when_the_tail_dies():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    out = adaptive_max_delay(u, X, tr, va, te, start=4, cap=12, n_surrogates=5)
    assert out["reason"] in ("tail_decayed", "resource_cap")
    assert out["max_delay"] <= 12
    assert out["profile"][1]["above_band"]              # a planted delay is above the band


def test_report_labels_degrees_above_the_structural_bound():
    u, X = _toy()
    tr, va, te = chrono_split(len(u), washout=20, n_val=120, n_test=150, gap=10)
    rep = compute_ipc_report(u, X, tr, va, te, tau_min=1, max_delay=3, max_degree=4,
                             max_targets_per_degree=20, n_surrogates=5,
                             max_structural_degree=2, min_train_absolute=0)
    assert all(d > 2 for d in rep.degrees_above_structural_bound)
