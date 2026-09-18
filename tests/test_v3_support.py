"""
test_v3_support.py -- V3 support layer: safe ratios, robust scoring,
operator-entanglement bounds, seeds, cache, target coverage, Colab paths
and checkpoints (spec test items 12-36).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.colab_support import (Checkpointer, Paths, resolve_paths, estimate_runtime,  # noqa: E402
                                          runtime_metadata, in_colab)
from decoupled_qrc.operator_bounds import (operator_entanglement, operator_entanglement_max,  # noqa: E402
                                            validate_operator_entanglement, audit_v22_operator_entanglement)
from decoupled_qrc.robust_scoring import (lower_confidence_bound, robust_candidate_score,  # noqa: E402
                                           seed_support)
from decoupled_qrc.safe_ratio import safe_ratio, retained_nl_safe  # noqa: E402
from decoupled_qrc.v3_cache import (canonical_key, cached_v3, environment_fingerprint,  # noqa: E402
                                     CACHE_SCHEMA_VERSION, set_cache_root, get_cache_root)
from decoupled_qrc.v3_seeds import (make_seeds, discovery_seeds, confirmation_seeds,  # noqa: E402
                                     assert_disjoint, NestedSeeds)


# --------------------------------------------------------------------------
# Safe ratio handling (spec tests 19, 20)
# --------------------------------------------------------------------------

def test_safe_ratio_valid_case():
    r = safe_ratio(0.5, 1.0, denominator_uncertainty=0.01)
    assert r.evaluable and r.ratio == pytest.approx(0.5)


def test_safe_ratio_rejects_negative_denominator():
    r = safe_ratio(0.038, -0.030)
    assert not r.evaluable and np.isnan(r.ratio)
    assert "not_positive" in r.rejection_reason


def test_safe_ratio_rejects_tiny_denominator_no_giant_ratio():
    """The exact V2.2 failure: numerator 0.038 over a denominator that had
    crossed zero produced eta = 3.75e7. It must now be NOT EVALUABLE."""
    r = safe_ratio(0.038, 1e-9)
    assert not r.evaluable
    assert np.isnan(r.ratio), "must never return an epsilon-rescued enormous ratio"


def test_safe_ratio_rejects_denominator_inside_its_own_uncertainty():
    r = safe_ratio(1.0, 0.1, denominator_uncertainty=0.09)
    assert not r.evaluable and "uncertainty" in r.rejection_reason


def test_retained_nl_safe_records_all_fields():
    r = retained_nl_safe(0.9, 1.2, standalone_uncertainty=0.05)
    d = r.as_dict()
    for key in ("numerator", "denominator", "denominator_uncertainty", "ratio",
                "evaluable", "rejection_reason"):
        assert key in d


def test_v22_exact_explosion_case_is_now_rejected():
    """Replays the saved V2.2 numbers (numerator +0.038, denominator
    -0.030) and asserts the V3 handler refuses them."""
    r = retained_nl_safe(0.0379, -0.0300)
    assert not r.evaluable and np.isnan(r.ratio)


# --------------------------------------------------------------------------
# Robust scoring (V2.2 zero-score and one-seed defects)
# --------------------------------------------------------------------------

def test_lcb_well_defined_for_single_candidate():
    """V2.2's min-max score gave a VALID single candidate exactly 0.0.
    The LCB score must return its actual value instead."""
    s = lower_confidence_bound([0.59])
    assert s.n == 1 and s.lcb == pytest.approx(0.59)


def test_lcb_penalises_seed_inconsistency():
    consistent = lower_confidence_bound([0.40, 0.42, 0.38])
    erratic = lower_confidence_bound([0.80, 0.00, 0.40])
    assert consistent.mean == pytest.approx(0.40, abs=0.02)
    assert erratic.mean == pytest.approx(0.40, abs=0.02)
    assert consistent.lcb > erratic.lcb, "equal means -- the erratic candidate must score lower"


def test_robust_candidate_score_nonzero_for_single_valid_candidate():
    s = robust_candidate_score([0.59], [5.1], [0.089])
    assert np.isfinite(s.lcb) and s.lcb != 0.0
    assert set(s.components) >= {"nl0_lcb", "backaction_penalty"}


def test_seed_support_detects_one_seed_driving_the_effect():
    """The exact V2.2 situation: 2 response seeds run, only 1 showed NL,
    yet the old check passed."""
    s = seed_support([0.714, -0.001], threshold=0.05)
    assert s.n_seeds_run == 2 and s.n_seeds_with_effect == 1
    assert s.driven_by_one_seed and not s.sufficient


def test_seed_support_accepts_consistent_seeds():
    s = seed_support([0.5, 0.45, 0.6], threshold=0.05)
    assert s.n_seeds_with_effect == 3 and not s.driven_by_one_seed and s.sufficient


# --------------------------------------------------------------------------
# Operator entanglement bounds (spec tests 31, 32)
# --------------------------------------------------------------------------

def test_operator_entanglement_of_identity_is_zero():
    n = 4
    assert operator_entanglement(np.eye(2 ** n, dtype=complex), n) == pytest.approx(0.0, abs=1e-9)


def test_operator_entanglement_of_product_unitary_is_zero():
    """A tensor product across the bipartition has zero operator
    entanglement by definition."""
    from scipy.stats import unitary_group
    ua = unitary_group.rvs(4, random_state=0)
    ub = unitary_group.rvs(4, random_state=1)
    assert operator_entanglement(np.kron(ua, ub), 4) == pytest.approx(0.0, abs=1e-9)


def test_operator_entanglement_of_swap_is_maximal_and_within_bound():
    """A SWAP across a balanced bipartition is maximally operator-
    entangling; it must SATURATE the correct bound, not exceed it."""
    n, n_a = 2, 1
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    val = operator_entanglement(swap, n, n_a)
    check = validate_operator_entanglement(val, n, n_a)
    assert check.valid
    assert val == pytest.approx(operator_entanglement_max(n, n_a), abs=1e-9)


def test_random_unitary_respects_the_bound():
    from scipy.stats import unitary_group
    n = 4
    u = unitary_group.rvs(2 ** n, random_state=7)
    val = operator_entanglement(u, n)
    assert validate_operator_entanglement(val, n).valid


def test_bound_violation_is_rejected_loudly():
    check = validate_operator_entanglement(99.0, 4)
    assert not check.valid and "EXCEEDS" in check.message


def test_v22_operator_entanglement_audit_reproduces_the_discrepancy():
    """The saved V2.2 value 1.7537 exceeded the bound the repository
    PRINTED (log(2^2)=1.386) but is legal under the CORRECT operator
    bound (2*log(4)=2.7726). The audit must say exactly that."""
    a = audit_v22_operator_entanglement()
    assert a["exceeds_repository_printed_bound"] is True
    assert a["correct_bound"] == pytest.approx(2 * np.log(4))
    assert a["valid_under_correct_bound"] is True


# --------------------------------------------------------------------------
# Seeds (spec tests 23, 25)
# --------------------------------------------------------------------------

def test_seed_streams_are_independent_and_complete():
    s = make_seeds(0, 0)
    streams = [s.hamiltonian, s.disorder, s.input_sequence, s.regression_split,
               s.null_surrogate, s.projection, s.bootstrap, s.shot_noise]
    assert len(set(streams)) == 8, "all eight streams must differ"
    assert set(s.as_dict()) >= {"hamiltonian", "disorder", "input_sequence", "regression_split",
                                 "null_surrogate", "projection", "bootstrap", "shot_noise"}


def test_seeds_deterministic_and_frozen():
    assert make_seeds(2, 1) == make_seeds(2, 1)
    with pytest.raises(Exception):
        make_seeds(0, 0).hamiltonian = 5     # frozen dataclass


def test_discovery_and_confirmation_seeds_are_disjoint():
    d = [discovery_seeds(k) for k in range(20)]
    c = [confirmation_seeds(k) for k in range(5)]
    assert_disjoint(d, c)
    assert all(s.reservoir_idx < 0 for s in d)
    assert all(s.reservoir_idx >= 0 for s in c)


def test_v3_seed_namespace_differs_from_v2():
    """A V3 seed must never coincide with the V2 seed for the same
    indices, so V3 runs cannot accidentally reuse V2 realizations."""
    from decoupled_qrc.validation_utils import make_nested_seeds as v2_seeds
    assert make_seeds(0, 0).input_sequence != v2_seeds(0, 0).input_seed


# --------------------------------------------------------------------------
# Cache (spec tests 26, 27)
# --------------------------------------------------------------------------

def test_cache_key_changes_when_any_relevant_field_changes():
    base = dict(architecture="dual_route_current", m=0.8, g=0.8, J=0.6, lam=0.0, tap_depth=4,
                n_surrogates=19, seeds=make_seeds(0, 0).as_dict())
    k0 = canonical_key(**base)
    for field, new in (("architecture", "serial"), ("m", 0.81), ("g", 0.9), ("J", 0.61),
                       ("lam", 0.02), ("tap_depth", 5), ("n_surrogates", 49)):
        assert canonical_key(**{**base, field: new}) != k0, f"{field} did not change the cache key"
    assert canonical_key(**{**base, "seeds": make_seeds(1, 0).as_dict()}) != k0


def test_cache_key_is_stable_for_identical_input():
    base = dict(architecture="serial", m=0.5)
    assert canonical_key(**base) == canonical_key(**base)


def test_cache_schema_version_is_v3_and_in_every_key():
    assert CACHE_SCHEMA_VERSION.startswith("v3")
    assert environment_fingerprint()["schema_version"] == CACHE_SCHEMA_VERSION
    # a schema bump must invalidate: keys embed the schema, so two different
    # schema versions cannot produce the same digest for the same payload
    import decoupled_qrc.v3_cache as vc
    original = vc.CACHE_SCHEMA_VERSION
    k_a = canonical_key(x=1)
    try:
        vc.CACHE_SCHEMA_VERSION = "v9.9.9"
        k_b = canonical_key(x=1)
    finally:
        vc.CACHE_SCHEMA_VERSION = original
    assert k_a != k_b


def test_cached_v3_hits_and_misses(tmp_path):
    set_cache_root(tmp_path / "cache")
    calls = []

    @cached_v3("unit_test")
    def compute(*, a, b):
        calls.append((a, b))
        return a + b

    assert compute(a=1, b=2) == 3
    assert compute(a=1, b=2) == 3
    assert len(calls) == 1, "identical arguments must hit the cache"
    assert compute(a=1, b=3) == 4
    assert len(calls) == 2, "a changed argument must miss the cache"


# --------------------------------------------------------------------------
# Target coverage -- the defect that silently zeroed every fixed-delay metric
# --------------------------------------------------------------------------

def test_single_delay_targets_always_retained_when_requested():
    """Without the flag, random subsampling can drop EVERY single-delay
    profile (verified at max_delay=5, max_degree=3, cap=6), forcing
    C_{d,tau} to exactly 0 for want of a target."""
    from decoupled_qrc.ipc import generate_profiles
    kwargs = dict(max_delay=5, max_degree=3, max_targets_per_degree=6, seed=3858801506)
    without = generate_profiles(**kwargs)
    without.pop("_was_capped")
    assert not any(len(p.pairs) == 1 for p in without[2]), "this seed is the known pathological case"

    with_flag = generate_profiles(**kwargs, always_include_single_delays=True)
    with_flag.pop("_was_capped")
    for degree in (2, 3):
        delays = {p.pairs[0][0] for p in with_flag[degree] if len(p.pairs) == 1}
        assert 0 in delays, f"degree {degree} must retain the delay-0 single-delay target"


def test_single_delay_inclusion_is_deterministic():
    from decoupled_qrc.ipc import generate_profiles
    kw = dict(max_delay=5, max_degree=3, max_targets_per_degree=6, seed=11,
              always_include_single_delays=True)
    a, b = generate_profiles(**kw), generate_profiles(**kw)
    a.pop("_was_capped"), b.pop("_was_capped")
    assert all([str(x) for x in a[d]] == [str(x) for x in b[d]] for d in a)


def test_default_target_generation_is_unchanged():
    """Backward compatibility: the flag defaults to False and must not
    alter what earlier passes computed."""
    from decoupled_qrc.ipc import generate_profiles
    kw = dict(max_delay=4, max_degree=2, max_targets_per_degree=5, seed=2)
    a = generate_profiles(**kw)
    b = generate_profiles(**kw, always_include_single_delays=False)
    a.pop("_was_capped"), b.pop("_was_capped")
    assert all([str(x) for x in a[d]] == [str(x) for x in b[d]] for d in a)


# --------------------------------------------------------------------------
# Colab support (spec tests 34, 35, 36)
# --------------------------------------------------------------------------

def test_resolve_paths_creates_local_tree(tmp_path):
    paths = resolve_paths(repo_root=tmp_path, use_drive=False)
    assert isinstance(paths, Paths)
    assert paths.results_root.exists() and paths.cache_root.exists() and paths.checkpoint_root.exists()
    assert paths.persistent is False


def test_resolve_paths_uses_drive_when_available(tmp_path):
    drive = tmp_path / "drive" / "MyDrive" / "dqrc_dual_route_v3"
    drive.parent.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(repo_root=tmp_path / "repo", drive_results_root=drive, use_drive=True)
    assert paths.persistent is True
    assert str(paths.results_root) == str(drive)


def test_checkpoint_roundtrip_and_key_validation(tmp_path):
    ck = Checkpointer(tmp_path)
    ck.save("stage1", "item", key="KEY_A", payload={"value": 42})
    assert ck.load("stage1", "item", key="KEY_A") == {"value": 42}
    assert ck.load("stage1", "item", key="KEY_B") is None, "a changed key must never reuse a checkpoint"
    assert "item" in ck.list_stage("stage1")


def test_checkpoint_force_recompute_ignores_existing(tmp_path):
    ck = Checkpointer(tmp_path)
    ck.save("s", "n", key="K", payload=1)
    assert Checkpointer(tmp_path, force_recompute=True).load("s", "n", key="K") is None


def test_runtime_estimate_reports_simulations_and_memory():
    est = estimate_runtime(n_simulations=40, seconds_per_simulation=2.5, n_qubits=8)
    assert est.estimated_seconds == pytest.approx(100.0)
    assert est.estimated_peak_mb > 0
    assert "simulations" in est.summary()


def test_runtime_metadata_is_json_serializable():
    """Spec test 36."""
    meta = runtime_metadata()
    json.loads(json.dumps(meta))
    assert "in_colab" in meta and isinstance(meta["in_colab"], bool)
    assert in_colab() is False or in_colab() is True
