"""
test_v3_architectures.py -- V3 core: memory bank, nonlinear processor,
the four architectures, and structural isolation (spec test items 1-11).
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.dual_route import (DualRouteConfig, run_architecture, mp_channel_unitary,  # noqa: E402
                                       ARCHITECTURES)
from decoupled_qrc.isolation import check_structural_isolation, inject_cross_dependency  # noqa: E402
from decoupled_qrc.memory_bank import (MemoryBankConfig, fractional_swap_matrix, run_memory_bank,  # noqa: E402
                                        to_signed)
from decoupled_qrc.nonlinear_processor import (ProcessorConfig, h0_matrix, input_operators,  # noqa: E402
                                                processor_unitary, run_processor, build_tap_buffer)
from decoupled_qrc.v3_seeds import make_seeds  # noqa: E402

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _base(m=0.8, g=0.8, J=0.6, L=3, N_P=3):
    return dict(memory=MemoryBankConfig(L=L, m=m), processor=ProcessorConfig(N_P=N_P, g=g, J=J, R=2))


# --------------------------------------------------------------------------
# Memory bank
# --------------------------------------------------------------------------

def test_fractional_swap_endpoints_and_unitarity():
    assert np.allclose(fractional_swap_matrix(1.0), SWAP)
    assert np.allclose(fractional_swap_matrix(0.0), np.eye(4))
    u = fractional_swap_matrix(0.37)
    assert np.allclose(u @ u.conj().T, np.eye(4))


def test_fractional_swap_half_squared_is_swap():
    half = fractional_swap_matrix(0.5)
    assert np.allclose(half @ half, SWAP)


def test_memory_parameter_changes_physical_memory_dynamics():
    """Spec test 11: `m` must actually change the memory's internal
    dynamics -- not merely relabel a horizon or a feature count."""
    seeds = make_seeds(0, 0)
    u = np.random.RandomState(0).uniform(0, 1, 30)
    a = run_memory_bank(MemoryBankConfig(L=3, m=0.9), u, seeds)
    b = run_memory_bank(MemoryBankConfig(L=3, m=0.3), u, seeds)
    assert a.X_M.shape == b.X_M.shape, "m must NOT change the feature count"
    assert not np.allclose(a.X_M, b.X_M), "m must change the memory's actual dynamics"


def test_memory_at_m_one_is_a_faithful_delay_line():
    """At m=1 the fractional SWAP is a true SWAP, so the bank is a perfect
    shift register and rail r+1 should carry u_{t-r} essentially exactly."""
    seeds = make_seeds(0, 0)
    u = np.random.RandomState(1).uniform(0, 1, 60)
    run = run_memory_bank(MemoryBankConfig(L=4, m=1.0), u, seeds)
    s = to_signed(u)
    z1 = run.X_M[:, run.labels.index("Z1")]
    assert abs(float(np.corrcoef(z1, s)[0, 1])) > 0.99


def test_to_signed_maps_canonical_unit_interval():
    assert to_signed(0.0) == pytest.approx(-1.0)
    assert to_signed(1.0) == pytest.approx(1.0)
    assert to_signed(0.5) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Nonlinear processor
# --------------------------------------------------------------------------

def test_noncommuting_hamiltonian_encoding_is_actually_noncommuting():
    """Spec test 10: [V_k, H_0] != 0, and [V_k, V_l] != 0 for some k != l."""
    cfg = ProcessorConfig(N_P=3, n_taps=1)
    h0, vs = h0_matrix(cfg), input_operators(cfg)
    assert np.linalg.norm(vs[0] @ h0 - h0 @ vs[0]) > 1e-9

    cfg_multi = ProcessorConfig(N_P=2, n_taps=4)
    v = input_operators(cfg_multi)
    pairs = [np.linalg.norm(v[i] @ v[j] - v[j] @ v[i]) for i in range(4) for j in range(i + 1, 4)]
    assert max(pairs) > 1e-9, "expected at least one noncommuting pair of input operators"


def test_processor_unitary_is_unitary():
    cfg = ProcessorConfig(N_P=3, R=3)
    u = processor_unitary(cfg, [0.4])
    assert np.allclose(u @ u.conj().T, np.eye(2 ** cfg.N_P), atol=1e-10)


def test_processor_microsteps_produce_instantaneous_nonlinearity():
    """Spec test 9: the within-timestep processor must be a genuinely
    NONLINEAR function of the current input -- a straight line fit through
    <Z_0>(u) must leave substantial residual."""
    cfg = ProcessorConfig(N_P=3, g=0.8, J=0.6, R=3, n_taps=1)
    us = np.linspace(0.0, 1.0, 11)
    run = run_processor(cfg, build_tap_buffer(us, 1), make_seeds(0, 0))
    z = run.X_P[:, run.labels.index("Z0")]
    resid = float(np.sum((np.polyval(np.polyfit(us, z, 1), us) - z) ** 2))
    assert resid > 1e-3, f"expected nonlinear response, linear-fit residual was only {resid}"


def test_processor_reset_removes_inter_timestep_persistence():
    """Spec test 8: with reset every macro timestep, X_P[t] depends ONLY
    on d_t -- so two sequences sharing a value at step t must agree at
    step t regardless of their different histories."""
    cfg = ProcessorConfig(N_P=3, R=2, n_taps=1)
    seeds = make_seeds(0, 0)
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 1, 12)
    b = rng.uniform(0, 1, 12)
    b[7] = a[7]                      # share only the value at t=7
    ra = run_processor(cfg, build_tap_buffer(a, 1), seeds)
    rb = run_processor(cfg, build_tap_buffer(b, 1), seeds)
    assert np.allclose(ra.X_P[7], rb.X_P[7], atol=1e-10)
    assert not np.allclose(ra.X_P[6], rb.X_P[6], atol=1e-6)


def test_processor_state_is_valid_density_matrix():
    """Spec test 7: reset composed with the drive must stay CPTP -- the
    resulting state is trace-1, Hermitian and positive semidefinite."""
    from qiskit import transpile
    from qiskit.quantum_info import DensityMatrix
    from qrc_qiskit import make_simulator
    from decoupled_qrc.nonlinear_processor import build_processor_circuit

    cfg = ProcessorConfig(N_P=3, R=2, n_taps=1)
    d = build_tap_buffer(np.random.RandomState(0).uniform(0, 1, 6), 1)
    qc, _, _ = build_processor_circuit(cfg, d, make_seeds(0, 0))
    qc.save_density_matrix(label="rho")
    sim = make_simulator(method="density_matrix")
    rho = np.asarray(DensityMatrix(np.asarray(
        sim.run(transpile(qc, sim, optimization_level=1), shots=1).result().data(0)["rho"])).data)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
    assert np.allclose(rho, rho.conj().T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-9)


def test_tap_buffer_layout_and_signing():
    u = np.array([0.0, 0.5, 1.0, 0.25])
    d = build_tap_buffer(u, 3)
    assert d.shape == (4, 3)
    assert d[:, 0] == pytest.approx(2 * u - 1)
    assert d[0, 1] == 0.0 and d[0, 2] == 0.0          # zero padding at the start
    assert d[2, 1] == pytest.approx(2 * u[1] - 1)      # one-step delay tap


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------

def test_all_architectures_run_and_produce_both_feature_groups():
    seeds = make_seeds(0, 0)
    for arch in ARCHITECTURES:
        run = run_architecture(DualRouteConfig(architecture=arch, lam=0.1, tap_depth=3, **_base()),
                                T=12, seeds=seeds)
        assert run.X_M.shape[0] == 12 and run.X_P.shape[0] == 12
        assert run.X_M.shape[1] > 0 and run.X_P.shape[1] > 0


def test_mp_channel_identity_at_zero_and_unitary_otherwise():
    assert np.allclose(mp_channel_unitary(0.0), np.eye(4))
    u = mp_channel_unitary(0.2)
    assert np.allclose(u @ u.conj().T, np.eye(4))


def test_lambda_zero_reproduces_architecture_b_exactly():
    """Spec test 5: at lambda = 0 the weakly-coupled dual route must
    reduce EXACTLY (bit-for-bit) to the direct-current dual route."""
    seeds = make_seeds(1, 0)
    b = run_architecture(DualRouteConfig(architecture="dual_route_current", **_base()), T=14, seeds=seeds)
    d0 = run_architecture(DualRouteConfig(architecture="dual_route_weak", lam=0.0, **_base()),
                           T=14, seeds=seeds)
    assert np.array_equal(b.X_M, d0.X_M)
    assert np.array_equal(b.X_P, d0.X_P)


def test_increasing_lambda_changes_the_channel():
    """Spec test 6."""
    seeds = make_seeds(2, 0)
    runs = {lam: run_architecture(DualRouteConfig(architecture="dual_route_weak", lam=lam, **_base()),
                                   T=14, seeds=seeds) for lam in (0.0, 0.05, 0.2)}
    assert not np.allclose(runs[0.0].X_P, runs[0.05].X_P)
    assert not np.allclose(runs[0.05].X_P, runs[0.2].X_P)


def test_deterministic_reruns():
    """Spec test 33."""
    seeds = make_seeds(3, 0)
    cfg = DualRouteConfig(architecture="dual_route_current", **_base())
    a = run_architecture(cfg, T=10, seeds=seeds)
    b = run_architecture(cfg, T=10, seeds=seeds)
    assert np.array_equal(a.X_M, b.X_M) and np.array_equal(a.X_P, b.X_P)


def test_common_random_numbers_across_parameter_perturbation():
    """Spec test 24: perturbing m must not change the input realization."""
    seeds = make_seeds(4, 0)
    cfg = DualRouteConfig(architecture="dual_route_current", **_base(m=0.8))
    a = run_architecture(cfg, T=10, seeds=seeds)
    b = run_architecture(replace(cfg, memory=replace(cfg.memory, m=0.6)), T=10, seeds=seeds)
    assert np.array_equal(a.u, b.u)


# --------------------------------------------------------------------------
# Structural isolation (the central V3 test)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["dual_route_current", "parallel_fixed_taps"])
def test_structural_isolation_holds_for_parallel_architectures(arch):
    """Spec tests 1, 2, 3: X_P invariant to m; X_M invariant to (g, J)."""
    seeds = make_seeds(0, 0)
    res = check_structural_isolation(DualRouteConfig(architecture=arch, tap_depth=3, **_base()),
                                      T=12, seeds=seeds)
    assert res.passed, res.failures
    assert res.max_abs_dXP_dm == 0.0
    assert res.max_abs_dXM_dg == 0.0
    assert res.max_abs_dXM_dJ == 0.0
    assert res.processor_hamiltonian_identical
    assert res.memory_config_identical


def test_injected_cross_dependency_makes_isolation_fail():
    """Spec test 4: the isolation check MUST be able to fail. A test that
    cannot fail proves nothing."""
    seeds = make_seeds(0, 0)
    res = check_structural_isolation(DualRouteConfig(architecture="dual_route_current", **_base()),
                                      T=12, seeds=seeds, run_fn=inject_cross_dependency(0.1))
    assert not res.passed
    assert res.max_abs_dXP_dm > res.atol
    assert any("X_P moved with m" in f for f in res.failures)


def test_serial_architecture_fails_structural_isolation_by_construction():
    """The negative control: in the serial route every processor input has
    passed through M(m), so X_P MUST depend on m -- the quantitative form
    of V3's core diagnosis."""
    seeds = make_seeds(0, 0)
    res = check_structural_isolation(
        DualRouteConfig(architecture="serial", lam_serial=0.5, **_base()), T=12, seeds=seeds)
    assert not res.passed
    assert res.max_abs_dXP_dm > 1e-6
