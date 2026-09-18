"""
test_cache_v21.py -- V2.1 Defect 10 / test requirements #11, #12:
cache-key completeness (a miss on ANY scientifically relevant field
changing) and explicit cache schema invalidation.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.utils import cached  # noqa: E402
from decoupled_qrc.candidate_eval import (evaluate_candidate_cached, compute_provenance_tag,  # noqa: E402
                                           CACHE_SCHEMA_VERSION)
from decoupled_qrc.validation_utils import ControlRange, make_nested_seeds  # noqa: E402

_TEST_NAMESPACE = "test_cache_v21_synthetic"


def _clear_test_namespace():
    ns_dir = None
    # discover the namespace dir the same way `cached()` would build it
    from decoupled_qrc import utils as u
    ns_dir = u._CACHE_DIR / _TEST_NAMESPACE
    if ns_dir.exists():
        shutil.rmtree(ns_dir)
    return ns_dir


def test_cache_hit_on_identical_arguments_all_explicit():
    """A synthetic stand-in with the SAME 'many explicit kwargs' shape as
    `candidate_eval.evaluate_candidate` -- proves the underlying `cached()`
    mechanism (sha256 of repr(full args+kwargs tuple)) gives a cache HIT
    (same file reused, function body not re-executed) when every argument
    is identical."""
    ns_dir = _clear_test_namespace()
    calls = []

    @cached(_TEST_NAMESPACE)
    def fake_eval(m, g, J, seeds_tag, *, theta, phi, n_surrogates, provenance_tag):
        calls.append(1)
        return {"m": m, "g": g, "J": J}

    fake_eval(0.5, 0.3, 0.3, "seedsA", theta=0.2, phi=0.8, n_surrogates=19, provenance_tag="tagX")
    fake_eval(0.5, 0.3, 0.3, "seedsA", theta=0.2, phi=0.8, n_surrogates=19, provenance_tag="tagX")
    assert len(calls) == 1, "identical arguments must be a cache HIT (function body runs only once)"
    n_files = len(list(ns_dir.glob("*.pkl")))
    assert n_files == 1
    shutil.rmtree(ns_dir)


def test_cache_miss_when_any_single_relevant_field_changes():
    """Test requirement #11: changing ONE scientifically relevant field at
    a time (theta, n_surrogates, provenance_tag) must each independently
    produce a cache MISS (a new file, function body re-executed)."""
    ns_dir = _clear_test_namespace()
    calls = []

    @cached(_TEST_NAMESPACE)
    def fake_eval(m, g, J, seeds_tag, *, theta, phi, n_surrogates, provenance_tag):
        calls.append(1)
        return {"m": m, "g": g, "J": J}

    base = dict(m=0.5, g=0.3, J=0.3, seeds_tag="seedsA", theta=0.2, phi=0.8, n_surrogates=19,
                provenance_tag="tagX")
    fake_eval(base["m"], base["g"], base["J"], base["seeds_tag"], theta=base["theta"], phi=base["phi"],
              n_surrogates=base["n_surrogates"], provenance_tag=base["provenance_tag"])

    variants = [
        {**base, "theta": 0.25},
        {**base, "n_surrogates": 49},
        {**base, "provenance_tag": "tagY"},
        {**base, "seeds_tag": "seedsB"},
        {**base, "phi": 0.7},
    ]
    for v in variants:
        fake_eval(v["m"], v["g"], v["J"], v["seeds_tag"], theta=v["theta"], phi=v["phi"],
                   n_surrogates=v["n_surrogates"], provenance_tag=v["provenance_tag"])

    assert len(calls) == 1 + len(variants), "every single-field change must be an independent cache miss"
    n_files = len(list(ns_dir.glob("*.pkl")))
    assert n_files == 1 + len(variants)
    shutil.rmtree(ns_dir)


def test_cache_schema_version_is_explicit_and_present_in_provenance_tag():
    """Test requirement #12: an explicit schema version string, distinct
    from any prior pass's, so old (V2 or earlier V2.1) cache entries can
    never silently satisfy a lookup under a bumped schema."""
    assert CACHE_SCHEMA_VERSION.startswith("v2.1")
    tag = compute_provenance_tag()
    assert f"schema={CACHE_SCHEMA_VERSION}" in tag


def test_evaluate_candidate_cached_is_wrapped_with_gj_v2_1_namespace_distinct_from_v2():
    """`evaluate_candidate_cached` must live under its OWN cache namespace
    ('gj_v2_1'), never V2's 'gj_v2' -- so a V2.1 run can never silently
    read back a V2-era cached result computed under different metric
    definitions/seed schemes."""
    assert evaluate_candidate_cached.cache_dir.name == "gj_v2_1"
    assert evaluate_candidate_cached.cache_dir.name != "gj_v2"


def test_real_evaluate_candidate_cached_hit_avoids_recomputation():
    """One real (slower) end-to-end check: an actual `evaluate_candidate_cached`
    call, repeated with IDENTICAL arguments, must reuse the cached result
    (no new file added on the second call). Uses a fresh, time-unique
    `provenance_tag` each run so this test is idempotent across repeated
    pytest invocations (the on-disk cache persists between test runs by
    design -- a stale cache entry from a PRIOR run of this same test would
    otherwise make the 'first call is a miss' assumption below false)."""
    import time
    from decoupled_qrc import utils as u
    ns_dir = u._CACHE_DIR / "gj_v2_1"
    before = set(ns_dir.glob("*.pkl")) if ns_dir.exists() else set()

    m_range, g_range, J_range = ControlRange("m", 0.1, 1.0), ControlRange("g", 0.05, 0.6), ControlRange("J", 0.05, 0.6)
    seeds = make_nested_seeds(999, 0)
    kwargs = dict(N_M=2, N_P=5, theta=0.2, phi=0.8, ap_kind="xy", m_range=m_range, g_range=g_range,
                  J_range=J_range, h_m_tilde=0.05, h_g_tilde=0.05, h_J_tilde=0.05, T=70, washout=12,
                  n_val=15, n_test=20, max_delay=3, max_degree=2, max_targets_per_degree=5, n_surrogates=6,
                  provenance_tag=f"test_real_cache_hit_{time.time_ns()}")

    evaluate_candidate_cached(0.5, 0.3, 0.3, seeds, **kwargs)
    after_first = set(ns_dir.glob("*.pkl"))
    evaluate_candidate_cached(0.5, 0.3, 0.3, seeds, **kwargs)
    after_second = set(ns_dir.glob("*.pkl"))

    assert len(after_first) == len(before) + 1
    assert after_second == after_first, "an identical second call must not create a new cache file"
