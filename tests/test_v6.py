"""V6: planted-feature detection, exact register moments, g = 0 affinity,
route isolation, and the V5.4 span limitation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc import audit_ipc as A  # noqa: E402
from decoupled_qrc import v6_algebra as ALG  # noqa: E402
from decoupled_qrc import v6_architecture as V6  # noqa: E402
from decoupled_qrc.v6_core import V6_CLASSES, v6_targets  # noqa: E402


@pytest.mark.parametrize("cls", V6_CLASSES)
def test_planted_class_is_detected_and_its_absence_is_not(cls):
    lib = v6_targets()
    planted = [t for t in lib if t.cls == cls]
    others = [t for t in lib if t.cls != cls and t.cls in V6_CLASSES]
    with_cls = ALG.reachability(lambda u: A.build_Y(u, planted + others), T=8000)
    without = ALG.reachability(lambda u: A.build_Y(u, others), T=8000)
    assert with_cls[cls]["status"] == "reachable"
    assert without[cls]["status"] == "unreachable"


def test_unreachable_sentinel_detects_a_planted_leak():
    lib = [t for t in v6_targets() if t.cls == "SENTINEL_unreachable"]
    assert not ALG.reachability(lambda u: A.build_Y(u, lib), T=8000)["sentinel_unreachable_null"]


@pytest.mark.parametrize("p", [0.0, 0.35, 1.0])
def test_q_register_moments_equal_full_distribution(p):
    w = np.random.default_rng(3).uniform(-1, 1, 40)
    a, b = V6.q_register(w, p, 4), V6.q_register_dm(w, p, 4)
    assert np.abs(a[0] - b[0]).max() < 1e-12 and np.abs(a[1] - b[1]).max() < 1e-12


def test_processor_is_linear_and_Y_channel_empty_at_g0():
    s = V6.V6Spec()
    aP = V6.processor_poly(0.0, 0.0, s.phi, "n")
    aY = V6.processor_poly(0.0, 0.0, s.phi, "Y")
    assert np.allclose(aP, [0, np.cos(s.phi), 0, 0], atol=1e-12)
    assert np.allclose(aY, 0, atol=1e-12)
    a1 = V6.processor_poly(s.theta_max, s.chi_max, s.phi, "n")
    assert abs(a1[2]) > 0.05 and abs(a1[3]) > 0.05          # degrees 2 AND 3 at g = 1


def test_every_operational_feature_is_exactly_affine_at_g0_and_not_at_g1():
    ad = V6.V6Adapter(V6.V6Spec(L_R=6, L_Q=6, q_pairs=((1, 2), (3, 5))))
    for m in (0.3, 1.0):
        assert ALG.exact_affinity(ALG.v6_operational(ad, m, 0.0), T=120)["affine"]
    assert not ALG.exact_affinity(ALG.v6_operational(ad, 1.0, 1.0), T=120)["affine"]


def test_route_isolation_and_contaminated_controls():
    from decoupled_qrc import audit_checks as C
    make = lambda kind: V6.V6Adapter(V6.V6Spec(L_R=6, L_Q=4, q_pairs=((1, 2),)), kind=kind)  # noqa: E731
    assert C.structural_dependency_adapter(make, n_draws=4, seed=0)["isolated"]
    for kind in ("g_into_R", "m_into_P", "serial"):
        assert not C.structural_dependency_adapter(make, n_draws=4, seed=0, kind=kind)["isolated"]


def test_v5_4_cannot_reach_C2_C3_C4():
    from decoupled_qrc.v5_architecture import V5Spec
    s = V5Spec(L=16, p_max=0.7, theta_max=0.38 * np.pi, phi=np.pi / 3, read_stride=2)
    r = ALG.reachability(ALG.v5_4_operational(s), T=12000)
    assert r["V6_C1"]["status"] == "reachable"
    for c in ("V6_C2", "V6_C3", "V6_C4"):
        assert r[c]["status"] == "unreachable", (c, r[c]["max"])


def test_shot_adapter_keeps_routes_isolated():
    ad = V6.V6ShotAdapter(V6.V6Adapter(V6.V6Spec(L_R=6, L_Q=4, q_pairs=((1, 2),))), 300)
    u = np.random.default_rng(5).uniform(-1, 1, 50)
    assert np.array_equal(ad.run(u, 0.2, 0.6, 3)["P"], ad.run(u, 0.9, 0.6, 3)["P"])
    assert np.array_equal(ad.run(u, 0.6, 0.1, 3)["R"], ad.run(u, 0.6, 0.9, 3)["R"])


def test_r_register_first_moments_match_v5_memory_exactly():
    from decoupled_qrc.v5_architecture import memory_features
    u = np.random.default_rng(8).uniform(-1, 1, 60)
    Z, _ = V6.q_register(u, 0.55, 6)
    assert np.abs(Z[:, 1:] - memory_features(u, 0.55, 6)).max() < 1e-14


def test_r_pair_readout_is_affine_at_g0_and_isolated():
    s = V6.V6Spec(L_R=6, L_Q=4, q_pairs=(), r_pairs=((1, 2), (1, 3)))
    ad = V6.V6Adapter(s)
    assert ALG.exact_affinity(ALG.v6_operational(ad, 0.8, 0.0), T=100)["affine"]
    u = np.random.default_rng(9).uniform(-1, 1, 40)
    assert np.array_equal(ad.run(u, 0.5, 0.2)["R"], ad.run(u, 0.5, 0.9)["R"])


def test_shot_budget_is_deterministic_isolated_and_shared_with_classical_baseline():
    s = V6.V6Spec(L_R=6, stride_R=2, L_Q=4, q_pairs=(), r_pairs=((1, 3),), shots=500)
    ad = V6.V6ShotAdapter(V6.V6Adapter(s), s.shots)
    u = np.random.default_rng(11).uniform(-1, 1, 60)
    a, b = ad.run(u, 0.6, 0.2, 4), ad.run(u, 0.6, 0.9, 4)
    assert np.array_equal(a["R"], b["R"]) and np.array_equal(a["Rall"], b["Rall"])
    assert np.array_equal(ad.run(u, 0.1, 0.5, 4)["P"], ad.run(u, 0.9, 0.5, 4)["P"])
    assert np.array_equal(a["R"], a["Rall"][:, 1::2])            # local readout == baseline marginal
    assert np.array_equal(a["Qz"], a["Qall"][:, np.array(s.rails_Q) - 1])
    assert np.array_equal(ad.run(u, 0.6, 0.2, 4)["Q"], a["Q"])     # deterministic per seed
