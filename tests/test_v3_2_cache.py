"""V3.2 cache: key completeness, intentional aliasing, cached == uncached."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_architecture import ArchitectureSpec  # noqa: E402
from decoupled_qrc.v3_2_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION, architecture_key_fields, canonical_key, cached_v3_2,
    environment_fingerprint, memory_key_fields, processor_key_fields)
from decoupled_qrc.v3_2_memory import MemorySpec  # noqa: E402
from decoupled_qrc.v3_2_processor import ProcessorSpec  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DQRC_V3_2_CACHE_DIR", str(tmp_path / "cache"))
    import importlib

    import decoupled_qrc.v3_2_cache as mod
    importlib.reload(mod)
    yield mod


def test_schema_version_is_its_own_namespace():
    assert CACHE_SCHEMA_VERSION.startswith("v3.2")
    from decoupled_qrc.v3_cache import CACHE_SCHEMA_VERSION as v3
    assert CACHE_SCHEMA_VERSION != v3


def test_key_is_deterministic_and_order_independent():
    a = canonical_key(b=2, a=1, nested={"y": 2, "x": 1})
    b = canonical_key(a=1, b=2, nested={"x": 1, "y": 2})
    assert a == b


@pytest.mark.parametrize("field,value", [
    ("processor.g", 0.7), ("processor.J", 0.7), ("processor.ablation", "mixing_only"),
    ("processor.dt", 0.9), ("processor.R", 5), ("processor.readout", ["Z"]),
    ("processor.N_P", 6), ("processor.hamiltonian_seed", 999),
])
def test_changing_any_processor_field_changes_the_key(field, value):
    spec = ProcessorSpec(N_P=5)
    base = processor_key_fields(spec, "full", 0.6, 0.5, 11, 12)
    variant = dict(base)
    variant[field] = value
    assert canonical_key(**base) != canonical_key(**variant)


def test_intentional_alias_is_detected_for_the_hamiltonian_term_set():
    """The V3 key carried (m,g,J) and the seeds but NOT the Hamiltonian term
    set, encoder or observable set. A key missing those could serve an
    `encoder_only` result for a `full` request."""
    spec = ProcessorSpec(N_P=5)
    full = processor_key_fields(spec, "full", 0.6, 0.5, 11, 12)
    enc_only = processor_key_fields(spec, "encoder_only", 0.6, 0.5, 11, 12)
    assert canonical_key(**full) != canonical_key(**enc_only)

    aliased = {k: v for k, v in full.items() if k not in ("processor.ablation",)}
    aliased_enc = {k: v for k, v in enc_only.items() if k not in ("processor.ablation",)}
    assert canonical_key(**aliased) == canonical_key(**aliased_enc), (
        "this is the aliasing failure mode the real key must avoid")


def test_shot_budget_and_shot_seed_belong_to_the_key():
    base = dict(architecture_key_fields(ArchitectureSpec(), m=0.4, g=0.6, J=0.5,
                                        ablation="full", hamiltonian_seed=1, disorder_seed=2))
    a = canonical_key(**base, shots=10000, shot_seed=5)
    b = canonical_key(**base, shots=1000, shot_seed=5)
    c = canonical_key(**base, shots=10000, shot_seed=6)
    assert len({a, b, c}) == 3


def test_memory_mechanism_changes_the_key():
    a = memory_key_fields(MemorySpec(mechanism="fractional_swap"), 0.4, 3)
    b = memory_key_fields(MemorySpec(mechanism="leaky_collision"), 0.4, 3)
    assert canonical_key(**a) != canonical_key(**b)


def test_architecture_lambda_changes_the_key():
    a = architecture_key_fields(ArchitectureSpec(architecture="dual_route_weak", lam=0.35),
                                m=0.4, g=0.6, J=0.5, ablation="full",
                                hamiltonian_seed=1, disorder_seed=2)
    b = architecture_key_fields(ArchitectureSpec(architecture="dual_route_current"),
                                m=0.4, g=0.6, J=0.5, ablation="full",
                                hamiltonian_seed=1, disorder_seed=2)
    assert canonical_key(**a) != canonical_key(**b)


def test_environment_fingerprint_is_recorded():
    fp = environment_fingerprint()
    for k in ("schema", "python", "numpy", "scipy", "platform", "git_commit", "precision"):
        assert k in fp


def test_cached_equals_uncached(_tmp_cache):
    calls = {"n": 0}

    @_tmp_cache.cached_v3_2(lambda x, seed: {"x": x, "seed": seed})
    def expensive(x, seed):
        calls["n"] += 1
        rng = np.random.default_rng(seed)
        return float(x * rng.normal())

    first = expensive(2.0, 7)
    second = expensive(2.0, 7)
    assert calls["n"] == 1                       # served from disk the second time
    assert abs(first - second) <= 1e-12
    assert abs(first - expensive.uncached(2.0, 7)) <= 1e-12
    assert calls["n"] == 2                       # .uncached deliberately bypasses the cache
    expensive(3.0, 7)
    assert calls["n"] == 3                       # a different key really recomputes


def test_cache_serialisation_round_trips_a_dataclass(_tmp_cache):
    @_tmp_cache.cached_v3_2(lambda spec: {"spec": spec})
    def describe(spec):
        return spec.as_dict()

    spec = ProcessorSpec(N_P=5)
    assert describe(spec) == describe(spec) == spec.as_dict()
