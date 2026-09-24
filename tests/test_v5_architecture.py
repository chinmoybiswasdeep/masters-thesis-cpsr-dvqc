"""V5: exact memory, closed-form processor, route isolation, control power."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc import audit_ipc as A  # noqa: E402
from decoupled_qrc import v5_architecture as V  # noqa: E402

U = np.random.default_rng(1).uniform(-1, 1, 40)


@pytest.mark.parametrize("p", [0.0, 0.3, 0.9, 1.0])
def test_marginal_recursion_equals_full_density_matrix(p):
    a = V.memory_features(U, p, 4)
    b = V.memory_features_dm(U, p, 4)
    assert np.abs(a - b).max() < 1e-12


def test_memory_is_exactly_linear_in_past_inputs():
    base = [0.3, -0.5, 0.15, 0.2]

    def f(us):
        return float(V.memory_features(np.array(us), 0.7, 4)[-1, 1])

    for i in range(4):
        for j in range(i + 1, 4):
            def v(a, b):
                x = list(base); x[i] = a; x[j] = b
                return f(x)
            mixed = (v(.5, .5) - v(.5, -.5) - v(-.5, .5) + v(-.5, -.5)) / 4
            assert abs(mixed) < 1e-15


def test_memory_readout_never_contains_the_current_input():
    """The joint layer must not be able to form u_t * u_t."""
    X = V.memory_features(U, 1.0, 3)
    X2 = V.memory_features(np.r_[U[:-1], -U[-1]], 1.0, 3)
    assert np.array_equal(X[-1], X2[-1])


@pytest.mark.parametrize("theta,phi", [(0.0, 0.7), (0.4, 0.785), (1.2, 1.0)])
def test_processor_matches_its_closed_form(theta, phi):
    spec = V.V5Spec(theta_max=1.3, phi=phi)
    f = V.processor_feature_fn(spec, theta / 1.3)
    for u in np.linspace(-1, 1, 11):
        want = np.cos(phi) * np.cos(theta) * u + np.sin(phi) * np.sin(theta) * u * u
        alt = np.cos(phi) * np.cos(theta) * u - np.sin(phi) * np.sin(theta) * u * u
        assert min(abs(f(u) - want), abs(f(u) - alt)) < 1e-12


def test_processor_is_affine_at_g0_and_nonlinear_share_grows_monotonically():
    spec = V.V5Spec()
    Ns = []
    for g in np.linspace(0, 1, 21):
        f = V.processor_feature_fn(spec, g)
        cap = A.function_subspace(lambda u: np.array([f(u)]))["degree_capacity"]
        Ns.append(np.mean([cap[2], cap[3], cap[4]]))
    assert Ns[0] < 1e-12
    assert np.all(np.diff(Ns) >= -1e-12)
    assert Ns[-1] > 0.25


def test_route_isolation_is_exact_and_contamination_is_detected():
    ad = V.V5Adapter(V.V5Spec(L=4))
    a, b = ad.run(U, 0.4, 0.2), ad.run(U, 0.4, 0.9)
    c = ad.run(U, 0.9, 0.2)
    assert np.array_equal(a["R"], b["R"]) and np.array_equal(a["P"], c["P"])
    for kind in ("g_into_R", "m_into_P", "serial"):
        bad = V.V5Adapter(V.V5Spec(L=4), kind=kind)
        dR = np.abs(bad.run(U, 0.4, 0.2)["R"] - bad.run(U, 0.4, 0.9)["R"]).max()
        dP = np.abs(bad.run(U, 0.4, 0.5)["P"] - bad.run(U, 0.9, 0.5)["P"]).max()
        assert max(dR, dP) > 1e-6, kind


def test_feature_counts_are_control_independent():
    ad = V.V5Adapter(V.V5Spec(L=5))
    shapes = {tuple(v.shape[1] for k, v in ad.run(U, m, g).items() if k in ("R", "P", "J"))
              for m in (0, .5, 1) for g in (0, .5, 1)}
    assert shapes == {(5, 1, 5)}


def test_invalid_specs_rejected():
    with pytest.raises(ValueError):
        V.V5Spec(theta_max=2.0)
    with pytest.raises(ValueError):
        V.V5Spec(p_max=0.0)


def test_generic_structural_test_has_power():
    from decoupled_qrc import audit_checks as C
    make = lambda kind: V.V5Adapter(V.V5Spec(L=4), kind=kind)  # noqa: E731
    assert C.structural_dependency_adapter(make, n_draws=5, seed=0)["isolated"]
    for kind in ("g_into_R", "m_into_P", "serial"):
        assert not C.structural_dependency_adapter(make, n_draws=5, seed=0, kind=kind)["isolated"]


def test_section6_holds_for_v5_from_data():
    from decoupled_qrc import audit_checks as C
    s6 = C.intrinsic_nonlinearity(V.V5Adapter(), seeds=[(1, 2)], T=800)
    assert s6["verdict"].endswith("HOLDS")
    assert s6["interior_trend"]["ols_raw"] > 0.10 and s6["interior_trend"]["ols_std"] > 0.10
    assert not s6["same_subspace_for_all_g_positive"]


def test_encoder_carries_no_operational_nonlinearity():
    enc = V.encoder_only_capacity(V.V5Spec())
    for k in ("1_memory_local", "2_nonlinear_local", "3_operational_all", "4_joint",
              "5_nonlinear_local_at_g0"):
        assert enc[k]["nonlinear_capacity"] < 1e-9, k
    assert enc["P_full_Z_algebra"]["nonlinear_capacity"] > 0.99   # the resource g unlocks


def test_shot_noise_streams_keep_routes_isolated():
    sh = V.V5ShotAdapter(V.V5Adapter(V.V5Spec(L=3)), 500, depol_R=0.01, depol_P=0.01,
                         readout_error=0.01)
    u = np.random.default_rng(4).uniform(-1, 1, 60)
    assert np.array_equal(sh.run(u, 0.2, 0.5, 9)["P"], sh.run(u, 0.9, 0.5, 9)["P"])
    assert np.array_equal(sh.run(u, 0.5, 0.1, 9)["R"], sh.run(u, 0.5, 0.9, 9)["R"])


def test_strided_readout_reads_exactly_the_declared_rails():
    s = V.V5Spec(L=6, read_stride=2)
    assert s.read_rails == [2, 4, 6]
    full = V.memory_features(U, 0.5, 6)
    X = V.V5Adapter(s).run(U, 1.0, 0.3)
    assert np.array_equal(X["R"], V.memory_features(U, s.p_max, 6)[:, [1, 3, 5]])
    assert X["J"].shape[1] == 3 and full.shape[1] == 6
    sh = V.V5ShotAdapter(V.V5Adapter(s), 200).run(U, 1.0, 0.3, 1)
    assert sh["R"].shape[1] == 3 and sh["J"].shape[1] == 3
