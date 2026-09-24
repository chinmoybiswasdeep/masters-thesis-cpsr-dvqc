"""Tests for the independent audit engine (audit_ipc / audit_core / audit_checks)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc import audit_checks as C  # noqa: E402
from decoupled_qrc import audit_core as K  # noqa: E402
from decoupled_qrc import audit_ipc as A  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FROZEN = os.path.join(ROOT, "results", "v4", "frozen_config.json")


def _toy(T=800, seed=0):
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    X = np.column_stack([A.shift(u, k) for k in range(4)] + [A.legendre(u, 2)])
    return u, X


def test_independent_ridge_matches_v4_estimator():
    from decoupled_qrc.v4_ipc import BoundedReadout
    u, X = _toy()
    sp = A.Split.standard(len(u))
    lib = A.build_library(tau_max=5)
    Y = A.build_Y(u, lib)
    mine = A.score_all(X, Y, lib, sp, "ridge", alpha=0.5)
    bank = BoundedReadout(X, sp.train, sp.test, 0.5)
    for i, t in enumerate(lib):
        if t.score == "cap" and t.max_delay >= 0:
            assert abs(np.clip(mine[i], 0, 1) - bank.capacity(t.build(u))) < 1e-12


def test_ols_is_scale_invariant():
    u, X = _toy()
    sp = A.Split.standard(len(u))
    lib = A.build_library(tau_max=4)
    Y = A.build_Y(u, lib)
    a = A.score_all(X, Y, lib, sp, "ols_raw")
    b = A.score_all(X * np.array([1e-3, 1, 50, 7, 0.2]), Y, lib, sp, "ols_raw")
    assert np.allclose(a, b, atol=1e-8)


def test_raw_capacity_is_not_clipped():
    u, _ = _toy()
    X = np.random.default_rng(3).normal(size=(len(u), 60))
    sp = A.Split.standard(len(u))
    nul = A.null_targets(len(u), 30, seed=1)
    c = A.capacity(A.predict(X, nul, sp, "ols_std"), nul, sp)
    assert c.min() < 0.0                        # overfit readouts go negative


def test_delay_line_capacity_is_recovered_and_future_is_not():
    u, X = _toy()
    sp = A.Split.standard(len(u))
    lib = A.build_library(tau_max=6)
    Y = A.build_Y(u, lib)
    s = A.score_all(X, Y, lib, sp, "ols_std")
    nm = [t.name for t in lib]
    for t in range(4):
        assert s[nm.index(f"P1(t-{t})")] > 0.99
    assert s[nm.index("P1(t-6)")] < 0.05
    assert s[nm.index("P2(t-0)")] > 0.99
    assert max(s[i] for i, t in enumerate(lib) if t.cls == "SENTINEL_future") < 0.05


def test_function_subspace_is_exact():
    sub = A.function_subspace(lambda u: np.array([u, u ** 2]))
    cap = sub["degree_capacity"]
    assert abs(cap[1] - 1) < 1e-10 and abs(cap[2] - 1) < 1e-10 and cap[3] < 1e-10
    assert sub["rank"] == 2


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_encoder_identity_is_exact(n):
    assert C.symbolic_encoder_expansion(n)["verified_exactly"]
    assert C.numeric_encoder_expansion(n)["ok"]


@pytest.fixture(scope="module")
def v4():
    if not os.path.exists(FROZEN):
        pytest.skip("frozen V4 artifact absent")
    return K.V4Adapter(FROZEN)


def test_run_variant_reproduces_run_v4(v4):
    from decoupled_qrc.v4_architecture import run_v4
    u = np.random.default_rng(1).uniform(-1, 1, 30)
    R, P, J = C._v4_run_variant(v4.spec, u, 0.6, 0.4, 5)
    r = run_v4(v4.spec, u, m=0.6, g=0.4, seed=5)
    assert np.array_equal(R, r.X_R) and np.array_equal(P, r.X_P) and np.array_equal(J, r.X_J)


def test_structural_test_has_power(v4):
    assert C.structural_dependency(v4.spec, n_draws=4, seed=0)["isolated"]
    for kind in ("g_into_R", "m_into_P", "serial"):
        assert not C.structural_dependency(v4.spec, n_draws=4, seed=0, kind=kind)["isolated"]


def test_noise_streams_do_not_couple_routes(v4):
    """A shared RNG stream would make P's shot noise depend on m."""
    sh = C.finite_shot_adapter(v4, 200)
    u = np.random.default_rng(2).uniform(-1, 1, 40)
    a = sh.run(u, 0.2, 0.5, 9)["P"]
    b = sh.run(u, 0.9, 0.5, 9)["P"]
    assert np.array_equal(a, b)


def test_split_and_feature_count_detectors():
    sp = A.Split.standard(500)
    assert sp.check()["disjoint"]
    assert not A.Split(sp.train, sp.train).check()["disjoint"]
    rows = [{"feature_counts": {"P": 12}}, {"feature_counts": {"P": 11}}]
    assert not K.feature_count_invariance(rows)["invariant"]


def test_bootstrap_is_reproducible():
    v = [0.1, 0.2, 0.15, 0.3]
    assert K.boot(v, seed=3) == K.boot(v, seed=3)
