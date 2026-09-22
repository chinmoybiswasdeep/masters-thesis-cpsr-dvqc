"""V3.2 architecture: isolation, falsifiability, CRN, determinism, back-action."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_architecture import (  # noqa: E402
    ArchitectureSpec, apply_two_qubit, check_structural_isolation, evaluate_point,
    mp_channel_unitary, resource_row, run_architecture)
from decoupled_qrc.v3_2_memory import MemorySpec  # noqa: E402
from decoupled_qrc.v3_2_processor import ProcessorSpec  # noqa: E402
from decoupled_qrc.v3_2_readout import ShotBudget  # noqa: E402
from decoupled_qrc.v3_seeds import discovery_seeds  # noqa: E402

SMALL = dict(memory=MemorySpec(L=3), processor=ProcessorSpec(N_P=3))
U = np.random.default_rng(0).uniform(0, 1, 40)
KW = dict(m=0.4, g=0.6, J=0.5, disorder_seed=7, hamiltonian_seed=7)


def _dense_two_qubit(G4, a, b, n):
    D = 2 ** n
    full = np.zeros((D, D), dtype=complex)
    G = G4.reshape(2, 2, 2, 2)
    for i in range(D):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        for oa in range(2):
            for ob in range(2):
                amp = G[oa, ob, bits[a], bits[b]]
                if amp == 0:
                    continue
                nb = list(bits)
                nb[a], nb[b] = oa, ob
                j = sum(v << (n - 1 - k) for k, v in enumerate(nb))
                full[j, i] += amp
    return full


@pytest.mark.parametrize("a,b", [(0, 3), (1, 2), (3, 0), (0, 1)])
def test_apply_two_qubit_matches_the_dense_embedding(a, b):
    rng = np.random.default_rng(1)
    n, D = 4, 16
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    G = mp_channel_unitary(0.37)
    dense = _dense_two_qubit(G, a, b, n)
    assert np.max(np.abs(dense @ rho @ dense.conj().T - apply_two_qubit(rho, G, a, b, n))) < 1e-12


def test_coupling_is_exactly_identity_at_zero_lambda():
    assert np.allclose(mp_channel_unitary(0.0), np.eye(4))


def test_weak_architecture_reduces_to_current_at_zero_lambda():
    w = run_architecture(ArchitectureSpec(architecture="dual_route_weak", lam=0.0, **SMALL), U, **KW)
    c = run_architecture(ArchitectureSpec(architecture="dual_route_current", **SMALL), U, **KW)
    assert np.array_equal(w.X_M, c.X_M) and np.array_equal(w.X_P, c.X_P)


def test_structural_isolation_is_exact_at_zero_lambda():
    iso = check_structural_isolation(ArchitectureSpec(architecture="dual_route_current", **SMALL),
                                     U, **KW)
    assert iso["passed"]
    assert iso["max_dXP_dm"] == 0.0 and iso["max_dXM_dg"] == 0.0 and iso["max_dXM_dJ"] == 0.0


def test_injected_dependency_control_must_fail():
    """The isolation test has to be able to FAIL, or it proves nothing."""
    bad = check_structural_isolation(ArchitectureSpec(architecture="dual_route_current", **SMALL),
                                     U, m_leak_into_processor=0.3, **KW)
    assert not bad["passed"]
    assert bad["max_dXP_dm"] > 1e-6


def test_weak_coupling_creates_a_real_cross_dependence():
    """With lambda > 0 the off-diagonals are EMPIRICAL, not structural --
    which is what makes Gate F a real test rather than a vacuous one."""
    weak = check_structural_isolation(ArchitectureSpec(architecture="dual_route_weak", lam=0.35,
                                                       **SMALL), U, **KW)
    assert weak["max_dXP_dm"] > 1e-6


def test_tracing_out_the_processor_leaves_g_J_dependent_back_action_on_memory():
    """Directional in information flow does NOT mean the memory row is
    structurally zero: the joint unitary back-acts on M, and that back-action
    depends on the processor state, hence on (g, J).

    Must use N_P >= 4. At N_P=3 there are C(3,4)=0 quartets, so H_4 == 0 and
    the dXM/dJ entry is structurally zero -- an `or` here would pass while
    that entry proved nothing.
    """
    big = dict(memory=MemorySpec(L=3), processor=ProcessorSpec(N_P=5))
    weak = check_structural_isolation(ArchitectureSpec(architecture="dual_route_weak", lam=0.35,
                                                       **big), U, **KW)
    assert weak["max_dXM_dg"] > 1e-9 and weak["max_dXM_dJ"] > 1e-9
    assert weak["max_dXP_dm"] > 1e-6


def test_four_body_term_is_structurally_absent_below_four_qubits():
    """Pins why the previous test must not run at N_P=3."""
    import numpy as _np
    from decoupled_qrc.v3_2_processor import build_hamiltonians
    assert _np.linalg.norm(build_hamiltonians(ProcessorSpec(N_P=3), 3).H_4) == 0.0
    assert _np.linalg.norm(build_hamiltonians(ProcessorSpec(N_P=5), 3).H_4) > 1.0


def test_back_action_vanishes_as_lambda_goes_to_zero():
    strong = run_architecture(ArchitectureSpec(architecture="dual_route_weak", lam=0.6, **SMALL),
                              U, measure_back_action=True, **KW)
    weak = run_architecture(ArchitectureSpec(architecture="dual_route_weak", lam=0.1, **SMALL),
                            U, measure_back_action=True, **KW)
    assert strong.back_action["max_trace_distance"] > weak.back_action["max_trace_distance"]


def test_runs_are_deterministic():
    a = run_architecture(ArchitectureSpec(architecture="dual_route_weak", lam=0.3, **SMALL), U, **KW)
    b = run_architecture(ArchitectureSpec(architecture="dual_route_weak", lam=0.3, **SMALL), U, **KW)
    assert np.array_equal(a.X_M, b.X_M) and np.array_equal(a.X_P, b.X_P)


def test_common_random_numbers_isolate_the_varied_control():
    """Two points sharing every seed must differ ONLY through the control."""
    spec = ArchitectureSpec(architecture="dual_route_current", **SMALL)
    base = run_architecture(spec, U, **KW)
    same_m = run_architecture(spec, U, **{**KW, "g": 0.9})
    assert np.array_equal(base.X_M, same_m.X_M)          # memory untouched by g
    assert not np.array_equal(base.X_P, same_m.X_P)


def test_baselines_expose_only_their_own_module():
    m_only = run_architecture(ArchitectureSpec(architecture="memory_only", **SMALL), U, **KW)
    p_only = run_architecture(ArchitectureSpec(architecture="processor_only", **SMALL), U, **KW)
    assert m_only.X_M.size > 0 and m_only.X_P.size == 0
    assert p_only.X_P.size > 0 and p_only.X_M.size == 0


def test_evaluate_point_returns_module_specific_metrics():
    spec = ArchitectureSpec(architecture="dual_route_weak", lam=0.3, **SMALL)
    r = evaluate_point(spec, m=0.4, g=0.6, J=0.5, seeds=discovery_seeds(0, 0),
                       budget=ShotBudget(5000), T=420, washout=30, n_val=90, n_test=110,
                       tau_min=2, max_delay=4, max_degree=3, max_targets_per_degree=8,
                       n_surrogates=3, measure_back_action=True)
    assert "M_long" in r and "NL_0" in r
    assert r["dm_audit"]["ok"]
    assert r["back_action"]["max_trace_distance"] >= 0.0
    assert r["splits"]["gap"] >= 4                        # guard gap >= max_delay


def test_resource_row_counts_both_modules():
    spec = ArchitectureSpec(architecture="dual_route_weak", lam=0.3, **SMALL)
    run = run_architecture(spec, U, **KW)
    row = resource_row(spec, run, ShotBudget(10000))
    assert row["memory_qubits"] == 3 and row["processor_qubits"] == 3
    assert row["input_copies"] == 1 + 3                   # memory rail + processor copies
    assert row["shots_per_timestep"] > 0


def test_invalid_architecture_and_tap_are_rejected():
    with pytest.raises(ValueError):
        ArchitectureSpec(architecture="nope")
    with pytest.raises(ValueError):
        ArchitectureSpec(mem_tap=99, **SMALL)
