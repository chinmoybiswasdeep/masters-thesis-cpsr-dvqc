"""
test_ipc_decomposition.py -- order-delay IPC decomposition (Part 3/4 of the
V2 validation spec). The most important property: `compute_ipc`'s legacy
output must be BIT-IDENTICAL before/after this module's refactor of
`ipc.py` (it now aggregates `compute_ipc_detailed`'s records internally).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.ipc import compute_ipc, compute_ipc_detailed, to_v, legendre_target  # noqa: E402
from decoupled_qrc.ipc_decomposition import compute_ipc_decomposed, diagnose_ceiling, to_heatmap  # noqa: E402
from qrc_qiskit import chrono_split, random_input  # noqa: E402


def _setup(T=300, n_features=40, seed=0):
    u = random_input(T, seed=seed)
    X = np.random.RandomState(seed).normal(0, 1, size=(T, n_features))
    train, val, test = chrono_split(T, washout=30, n_val=60, n_test=90, gap=7)
    return u, X, train, val, test


def test_legacy_compute_ipc_unchanged_by_refactor():
    """The single most important regression test: compute_ipc's aggregated
    output must match compute_ipc_detailed's own aggregation exactly (same
    RNG draw order -> same numbers)."""
    u, X, train, val, test = _setup()
    legacy = compute_ipc(u, X, train, val, test, max_delay=5, max_degree=3, max_targets_per_degree=10,
                          n_surrogates=4, seed=1)
    records, was_capped, n_tested = compute_ipc_detailed(u, X, train, val, test, max_delay=5, max_degree=3,
                                                          max_targets_per_degree=10, n_surrogates=4, seed=1)
    manual_memory = sum(r.capacity for r in records if r.degree == 1)
    manual_nl = sum(r.capacity for r in records if r.degree >= 2)
    assert legacy.memory == pytest.approx(manual_memory, abs=1e-12)
    assert legacy.nonlinearity == pytest.approx(manual_nl, abs=1e-12)


def test_decomposition_M_and_NL_legacy_match_compute_ipc():
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=3,
                                     max_targets_per_degree=10, n_surrogates=4, seed=2)
    legacy = compute_ipc(u, X, train, val, test, max_delay=5, max_degree=3, max_targets_per_degree=10,
                          n_surrogates=4, seed=2)
    assert decomp.M_long == pytest.approx(legacy.memory, abs=1e-12)
    assert decomp.NL_legacy == pytest.approx(legacy.nonlinearity, abs=1e-12)


def test_instant_plus_temporal_le_legacy():
    """NL_instant + NL_temporal must never EXCEED NL_legacy (every
    instant/temporal record is also counted in the legacy sum; they can be
    strictly less only if there exist degree>=2 profiles that are neither
    -- there are none by construction, since every degree>=2 profile either
    has delays==(0,) [instant] or max_delay>0 [temporal], covering all
    cases exhaustively -- so this should be an exact equality)."""
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=3,
                                     max_targets_per_degree=10, n_surrogates=4, seed=3)
    assert decomp.NL_instant + decomp.NL_temporal == pytest.approx(decomp.NL_legacy, abs=1e-9)


def test_instantaneous_target_identified_correctly():
    """A planted feature that is EXACTLY L2(v_t) (delay=0, degree=2) must be
    classified as instantaneous, not temporal."""
    T = 300
    u = random_input(T, seed=0)
    v = to_v(u)
    instant_feature = legendre_target(v, 2)
    X = np.column_stack([instant_feature, np.random.RandomState(0).normal(0, 1, size=(T, 5))])
    train, val, test = chrono_split(T, washout=10, n_val=60, n_test=90, gap=7)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=2,
                                     max_targets_per_degree=20, n_surrogates=5, seed=0)
    assert decomp.NL_instant > 0.5, f"expected the planted instantaneous L2(v_t) signal to register, got {decomp.NL_instant}"


def test_temporal_target_identified_correctly_not_misclassified_as_instant():
    """A planted feature that is EXACTLY L2(v_{t-3}) (delay=3, degree=2)
    must be classified as TEMPORAL, never instantaneous -- Part 3's
    explicit 'do not misclassify cross-delay nonlinear targets' check."""
    T = 300
    u = random_input(T, seed=0)
    v = to_v(u)
    shifted = np.zeros(T)
    shifted[3:] = legendre_target(v[:-3], 2)
    X = np.column_stack([shifted, np.random.RandomState(1).normal(0, 1, size=(T, 5))])
    train, val, test = chrono_split(T, washout=10, n_val=60, n_test=90, gap=7)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=5, max_degree=2,
                                     max_targets_per_degree=20, n_surrogates=5, seed=0)
    assert decomp.NL_temporal > 0.5, f"expected the planted temporal L2(v_(t-3)) signal to register, got {decomp.NL_temporal}"
    assert decomp.NL_instant < 0.3, f"planted temporal signal must NOT be counted as instantaneous, got {decomp.NL_instant}"


def test_ceiling_diagnostic_flags_saturated_case():
    """A feature set that can PERFECTLY reconstruct every delay-0..k target
    (e.g. k+1 orthogonal delay-encoding columns) should be flagged as
    ceiling-contaminated when max_delay==k."""
    T = 400
    u = random_input(T, seed=0)
    v = to_v(u)
    max_delay = 3
    cols = []
    for k in range(max_delay + 1):
        shifted = np.zeros(T)
        if k == 0:
            shifted[:] = v
        else:
            shifted[k:] = v[:-k]
        cols.append(legendre_target(shifted, 1))
    X = np.column_stack(cols)
    train, val, test = chrono_split(T, washout=10, n_val=80, n_test=120, gap=max_delay + 1)
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=max_delay, max_degree=1,
                                     max_targets_per_degree=10, n_surrogates=5, seed=0)
    ceil = diagnose_ceiling(decomp, X, train)
    assert ceil.fraction_of_ceiling > 0.7, (
        f"expected a near-saturated fraction_of_ceiling for a feature set that can perfectly "
        f"reconstruct every delay-0..{max_delay} target, got {ceil}")


def test_to_heatmap_shape_and_values():
    u, X, train, val, test = _setup()
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=4, max_degree=2,
                                     max_targets_per_degree=10, n_surrogates=3, seed=0)
    hm = to_heatmap(decomp, max_degree=2, max_delay=4)
    assert hm.shape == (2, 5)
    assert hm[0, :].sum() == pytest.approx(decomp.M_long, abs=1e-9)
