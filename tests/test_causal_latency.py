"""
test_causal_latency.py -- V2.1 Phase 1 / test requirement #5: causal
latency recovery on synthetic delayed features, plus the literal-vs-local
NL distinction (never silently renaming tau=1 as tau=0).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.causal_latency import (causal_latency_profile, feature_input_alignment,  # noqa: E402
                                           circuit_timing_table)
from decoupled_qrc.ipc import to_v, legendre_target  # noqa: E402
from qrc_qiskit import chrono_split, random_input  # noqa: E402


def _delayed_features(T, true_lag, seed=0, n_noise=6):
    """Build X whose ONLY real signal is a degree-1 AND degree-2 function
    of u_{t-true_lag} (plus decoy noise columns), so C_{1,tau} and NL
    should both peak sharply at tau=true_lag."""
    u = random_input(T, seed=seed)
    v = to_v(u)
    rng = np.random.RandomState(seed)
    shifted = np.zeros(T)
    if true_lag == 0:
        shifted[:] = v
    else:
        shifted[true_lag:] = v[:T - true_lag]
    lin_feat = legendre_target(shifted, 1)
    quad_feat = legendre_target(shifted, 2)
    noise = rng.normal(size=(T, n_noise))
    X = np.column_stack([lin_feat, quad_feat, noise])
    return u, X


def test_causal_latency_recovers_known_delay_zero():
    T = 320
    u, X = _delayed_features(T, true_lag=0, seed=1)
    train, val, test = chrono_split(T, washout=20, n_val=70, n_test=90, gap=6)
    res = causal_latency_profile(u, X, train, val, test, max_delay=5, max_degree=2,
                                  max_targets_per_degree=8, n_surrogates=19, seed=1)
    assert res.ell_0 == 0, f"expected ell_0=0 for a delay-0 planted signal, got {res.ell_0}"
    assert res.NL_local == pytest.approx(res.NL_literal, abs=1e-9), (
        "at ell_0=0, NL_local and NL_literal must coincide exactly (same delay)")


def test_causal_latency_recovers_known_delay_three():
    T = 340
    u, X = _delayed_features(T, true_lag=3, seed=2)
    train, val, test = chrono_split(T, washout=20, n_val=70, n_test=90, gap=6)
    res = causal_latency_profile(u, X, train, val, test, max_delay=5, max_degree=2,
                                  max_targets_per_degree=25, n_surrogates=19, seed=2)
    assert res.ell_0 == 3, f"expected ell_0=3 for a delay-3 planted signal, got {res.ell_0}"
    # the NL signal lives at tau=3, so the literal (tau=0) reading must NOT
    # be confused with it -- NL_local (at the recovered ell_0) should be
    # substantially larger than NL_literal here (never silently renamed).
    assert res.NL_local_bc > res.NL_literal_bc, (
        f"expected NL_local_bc ({res.NL_local_bc}) > NL_literal_bc ({res.NL_literal_bc}) "
        f"when the real nonlinear signal sits at a nonzero delay")


def test_c1_by_delay_keeps_all_delays_not_just_the_argmax():
    T = 300
    u, X = _delayed_features(T, true_lag=2, seed=3)
    train, val, test = chrono_split(T, washout=20, n_val=60, n_test=80, gap=6)
    res = causal_latency_profile(u, X, train, val, test, max_delay=4, max_degree=1,
                                  max_targets_per_degree=6, n_surrogates=19, seed=3)
    assert set(res.C1_by_delay.keys()) == set(range(5))
    for tau, rec in res.C1_by_delay.items():
        assert set(rec.keys()) == {"delay", "legacy", "raw", "null", "bc", "signed", "n_profiles"}


def test_feature_input_alignment_probe_peaks_at_true_lag():
    T = 300
    true_lag = 2
    u, X = _delayed_features(T, true_lag=true_lag, seed=4)
    train, val, test = chrono_split(T, washout=20, n_val=60, n_test=80, gap=6)
    probe = feature_input_alignment(u, X, train, test, max_delay=4)
    best_tau = max(probe, key=lambda tau: probe[tau]["r2"] if not np.isnan(probe[tau]["r2"]) else -1)
    assert best_tau == true_lag, f"expected the linear probe R^2 to peak at tau={true_lag}, got {best_tau}"
    assert probe[true_lag]["r2"] > 0.5
    assert probe[true_lag]["mi_proxy_nats"] > 0


def test_circuit_timing_table_every_stage_depends_on_u_t():
    rows = circuit_timing_table("protected_integrable")
    assert len(rows) == 6
    assert all(r["depends_on_u_t"] for r in rows)
    assert rows[-1]["stage"] == "feature readout"
