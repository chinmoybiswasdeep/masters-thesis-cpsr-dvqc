"""V4 architecture: route isolation, factorisation, negative controls, resources."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v4_architecture import (  # noqa: E402
    MemoryRoute, MemorySpec, ProcessorRoute, ProcessorSpec, V4Spec, route_independence,
    run_v4, verify_product_factorisation)

SMALL = dict(memory=MemorySpec(L_R=3), processor=ProcessorSpec(N_P=3), n_joint=12)
U = np.random.default_rng(0).uniform(-1, 1, 40)


def test_route_isolation_is_exactly_zero():
    """Not 'small' -- exactly zero. The registers are never coupled."""
    iso = route_independence(V4Spec(**SMALL), U, m=0.6, g=0.5, seed=3)
    assert iso["max_dXR_dg"] == 0.0
    assert iso["max_dXP_dm"] == 0.0
    assert iso["passed"] is True


def test_joint_observables_factorise_exactly():
    fac = verify_product_factorisation(V4Spec(**SMALL), U[:10], m=0.6, g=0.5, seed=3)
    assert fac["factorises"] is True
    assert fac["max_abs_deviation"] < 1e-12


def test_contaminated_control_MUST_be_detected():
    """If this passes isolation, the isolation test has no power and proves
    nothing. It must fail."""
    bad = route_independence(V4Spec(architecture="contaminated", contamination=0.4, **SMALL),
                             U, m=0.6, g=0.5, seed=3)
    assert bad["passed"] is False
    assert bad["max_dXP_dm"] > 1e-6


def test_serial_control_MUST_be_detected():
    ser = route_independence(V4Spec(architecture="serial", **SMALL), U, m=0.6, g=0.5, seed=3)
    assert ser["passed"] is False


def test_processor_is_memoryless_by_construction():
    """P is reset each step, so X_P(t) depends only on u_t."""
    spec = V4Spec(**SMALL)
    a = run_v4(spec, np.array([0.3, -0.7, 0.5]), m=0.6, g=0.5, seed=3)
    b = run_v4(spec, np.array([-0.9, 0.1, 0.5]), m=0.6, g=0.5, seed=3)
    assert np.allclose(a.X_P[-1], b.X_P[-1], atol=1e-14)   # same final input
    assert not np.allclose(a.X_R[-1], b.X_R[-1])           # memory differs


def test_processor_at_g_zero_is_affine_in_the_input():
    """The whole N(g=0)=0 claim rests on this."""
    from decoupled_qrc.v4_encoder import polynomial_degrees
    P = ProcessorRoute(ProcessorSpec(N_P=3), 0.0, 3)
    for k in range(len(P.labels)):
        a = polynomial_degrees(lambda u, k=k: float(P.step(u)[k]), deg_max=6)
        assert a["is_affine"], (P.labels[k], a["degrees_present"])


def test_processor_at_g_positive_is_not_affine():
    from decoupled_qrc.v4_encoder import polynomial_degrees
    P = ProcessorRoute(ProcessorSpec(N_P=3, g_scale=0.5), 1.0, 3)
    degs = [polynomial_degrees(lambda u, k=k: float(P.step(u)[k]), deg_max=6)["max_degree"]
            for k in range(len(P.labels))]
    assert max(degs) >= 2


def test_memory_injection_gain_does_not_depend_on_m():
    """Input gain is decoupled from retention, so m is retention ALONE."""
    vals = []
    for m in (0.0, 0.5, 1.0):
        r = MemoryRoute(MemorySpec(L_R=3), m, 3)
        r.reset()
        vals.append(r.step(0.8)[0])          # first step, nothing retained yet
    assert max(vals) - min(vals) < 1e-12


def test_m_maps_to_retention_through_m_max():
    spec = MemorySpec(L_R=3, m_max=0.9)
    assert abs(MemoryRoute(spec, 1.0, 3).retention - 0.9) < 1e-15
    assert abs(MemoryRoute(spec, 0.5, 3).retention - 0.45) < 1e-15


def test_states_stay_valid_density_matrices():
    from decoupled_qrc.v4_encoder import density_matrix_audit
    for ret in ("amplitude_damping", "depolarizing"):
        r = MemoryRoute(MemorySpec(L_R=3, retention=ret), 0.6, 3)
        for u in np.random.default_rng(2).uniform(-1, 1, 25):
            r.step(float(u))
        assert density_matrix_audit(r.rho)["ok"], ret


def test_runs_are_deterministic_and_use_common_random_numbers():
    spec = V4Spec(**SMALL)
    a = run_v4(spec, U, m=0.6, g=0.5, seed=3)
    b = run_v4(spec, U, m=0.6, g=0.5, seed=3)
    assert np.array_equal(a.X_R, b.X_R) and np.array_equal(a.X_P, b.X_P)
    c = run_v4(spec, U, m=0.6, g=0.9, seed=3)
    assert np.array_equal(a.X_R, c.X_R)          # memory untouched by g
    assert not np.array_equal(a.X_P, c.X_P)


def test_resources_are_equal_across_every_control_value():
    """g must not buy extra depth, features or copies."""
    spec = V4Spec(**SMALL)
    base = spec.resources()
    for m in (0.0, 0.5, 1.0):
        for g in (0.0, 0.5, 1.0):
            r = run_v4(spec, U[:8], m=m, g=g, seed=3)
            assert r.X_R.shape[1] == base["observables_R"]
            assert r.X_P.shape[1] == base["observables_P"]
            assert r.X_J.shape[1] == base["observables_joint"]
    assert base["input_copies"] == 1 + SMALL["processor"].N_P
    assert base["nonlinear_degree_R"] == 1


def test_baselines_expose_only_their_own_route():
    m_only = run_v4(V4Spec(architecture="memory_only", **SMALL), U, m=0.6, g=0.5, seed=3)
    p_only = run_v4(V4Spec(architecture="processor_only", **SMALL), U, m=0.6, g=0.5, seed=3)
    assert m_only.X_R.size > 0 and m_only.X_P.size == 0 and m_only.X_J.size == 0
    assert p_only.X_P.size > 0 and p_only.X_R.size == 0


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        V4Spec(architecture="nope")
    with pytest.raises(ValueError):
        MemorySpec(L_R=1)
    with pytest.raises(ValueError):
        ProcessorSpec(N_P=1)
    with pytest.raises(ValueError):
        MemoryRoute(MemorySpec(), 1.5, 3)
    with pytest.raises(ValueError):
        ProcessorRoute(ProcessorSpec(), -0.1, 3)
