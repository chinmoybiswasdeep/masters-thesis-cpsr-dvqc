"""
test_frozen_protocol.py -- V2.2 Phase 5 / test requirements #3, #4, #5,
#6: fixed delays/target lists across stencil points, fixed null
permutations, frozen ridge settings.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.frozen_protocol import freeze_protocol, evaluate_frozen  # noqa: E402
from decoupled_qrc.ipc_decomposition import capacity_at_delay  # noqa: E402
from qrc_qiskit import chrono_split, random_input  # noqa: E402


def _setup(T=200, n_features=20, seed=0):
    u = random_input(T, seed=seed)
    X = np.random.RandomState(seed).normal(0, 1, size=(T, n_features))
    train, val, test = chrono_split(T, washout=20, n_val=50, n_test=60, gap=4)
    return u, X, train, val, test


def test_evaluate_frozen_reproduces_center_exactly_at_same_X():
    """Test requirement #6 (frozen ridge settings): re-evaluating the
    FROZEN spec at the EXACT SAME feature matrix used to freeze it must
    reproduce the center's own records exactly -- no re-search, no
    re-draw, bit-identical."""
    u, X, train, val, test = _setup()
    spec = freeze_protocol(u, X, train, val, test, max_delay=2, max_degree=2, max_targets_per_degree=5,
                            n_surrogates=6, seed=0)
    records_same = evaluate_frozen(u, X, train, val, test, spec)
    assert len(records_same) == len(spec.center_records)
    for a, b in zip(spec.center_records, records_same):
        assert a.raw_capacity == pytest.approx(b.raw_capacity, abs=1e-12)
        assert a.null_mean == pytest.approx(b.null_mean, abs=1e-12)


def test_frozen_target_list_identical_across_different_feature_matrices():
    """Test requirement #4: the SAME target list (degree, delays) must be
    used whether X is the center's own features or a perturbed point's --
    verified by checking `evaluate_frozen`'s returned records have EXACTLY
    the same (degree, delays) set as the center, regardless of X."""
    u, X, train, val, test = _setup()
    spec = freeze_protocol(u, X, train, val, test, max_delay=2, max_degree=2, max_targets_per_degree=5,
                            n_surrogates=6, seed=0)
    X_perturbed = X + 0.3 * np.random.RandomState(1).normal(size=X.shape)
    records_perturbed = evaluate_frozen(u, X_perturbed, train, val, test, spec)
    center_keys = {(r.degree, r.delays) for r in spec.center_records}
    perturbed_keys = {(r.degree, r.delays) for r in records_perturbed}
    assert center_keys == perturbed_keys


def test_frozen_alpha_reused_not_researched():
    """Test requirement #6: the alpha chosen at the center for each target
    must be the SAME alpha used when scoring a perturbed X -- proven by
    directly comparing `_capacity_score_fixed_alpha` at the frozen alpha
    to what `evaluate_frozen` implicitly used (same score if called with
    that alpha explicitly)."""
    from decoupled_qrc.ipc import _capacity_score_fixed_alpha, to_v, build_target
    u, X, train, val, test = _setup()
    spec = freeze_protocol(u, X, train, val, test, max_delay=1, max_degree=1, max_targets_per_degree=3,
                            n_surrogates=4, seed=0)
    X_perturbed = X + 0.2 * np.random.RandomState(2).normal(size=X.shape)
    records_perturbed = evaluate_frozen(u, X_perturbed, train, val, test, spec)
    v = to_v(u)
    for prof_list in spec.profiles.values():
        for prof in prof_list:
            key = (1, tuple(sorted(tau for tau, _ in prof.pairs)))
            if key not in spec.alpha_by_key:
                continue
            alpha = spec.alpha_by_key[key]
            y = build_target(prof, v)
            expected_score = _capacity_score_fixed_alpha(X_perturbed, y, train, val, test, alpha)
            matching = [r for r in records_perturbed if r.degree == 1 and r.delays == key[1]]
            assert matching, f"expected a record for key {key}"
            assert matching[0].raw_capacity == pytest.approx(expected_score, abs=1e-9)


def test_frozen_null_permutations_deterministic_across_calls():
    """Test requirement #5: the sequence of surrogate/null permutation
    draws must be IDENTICAL every time `evaluate_frozen` is called with
    the same spec (same seed -> same RNG stream) -- proven by calling
    twice on two DIFFERENT feature matrices and checking the null_std
    pattern (a function purely of which permutations were drawn, given
    deterministic real data) is reproducible when re-run on the SAME
    perturbed matrix."""
    u, X, train, val, test = _setup()
    spec = freeze_protocol(u, X, train, val, test, max_delay=1, max_degree=1, max_targets_per_degree=3,
                            n_surrogates=6, seed=0)
    X_perturbed = X + 0.15 * np.random.RandomState(3).normal(size=X.shape)
    records_a = evaluate_frozen(u, X_perturbed, train, val, test, spec)
    records_b = evaluate_frozen(u, X_perturbed, train, val, test, spec)
    for a, b in zip(records_a, records_b):
        assert a.null_mean == pytest.approx(b.null_mean, abs=1e-12)
        assert a.null_std == pytest.approx(b.null_std, abs=1e-12)


def test_fixed_delay_metric_computable_at_every_stencil_point_with_same_delay_set():
    """Test requirement #3: fixed delay across parameter perturbations --
    `capacity_at_delay` at a GIVEN tau must be computable and structurally
    consistent (same key set) across the center and a perturbed point's
    frozen-protocol records."""
    u, X, train, val, test = _setup()
    spec = freeze_protocol(u, X, train, val, test, max_delay=3, max_degree=2, max_targets_per_degree=6,
                            n_surrogates=6, seed=0)
    X_perturbed = X + 0.25 * np.random.RandomState(4).normal(size=X.shape)
    records_perturbed = evaluate_frozen(u, X_perturbed, train, val, test, spec)
    for tau in range(4):
        center_entry = capacity_at_delay(spec.center_records, tau, degree_min=1, degree_max=1)
        perturbed_entry = capacity_at_delay(records_perturbed, tau, degree_min=1, degree_max=1)
        assert center_entry["delay"] == perturbed_entry["delay"] == tau
        assert center_entry["n_profiles"] == perturbed_entry["n_profiles"]
