"""V3.2 processor: ablation distinctness, exact null linearity, degree bound."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_encoder import embed, expectation_values  # noqa: E402
from decoupled_qrc.v3_2_processor import (  # noqa: E402
    ABLATIONS, PROVEN_FEATURE_DEGENERATE, ProcessorSpec, ablation_distinctness,
    build_hamiltonians, exact_instantaneous_capacities, processor_unitary)

SPEC = ProcessorSpec(N_P=4, dt=0.5, R=2)


@pytest.fixture(scope="module")
def ham():
    return build_hamiltonians(SPEC, disorder_seed=3)


def test_hamiltonians_are_hermitian_and_H2_H4_commute(ham):
    a = ham.audit()
    assert a["herm_H_mix"] < 1e-12 and a["herm_H_2"] < 1e-12 and a["herm_H_4"] < 1e-12
    assert a["comm_H2_H4"] < 1e-12            # both Z-diagonal
    assert a["comm_Hmix_H2"] > 1e-6           # the required noncommutation
    assert a["comm_Hmix_H4"] > 1e-6


def test_null_processor_has_exactly_zero_instantaneous_nonlinearity(ham):
    """The central architectural claim. V3.1's encoder gave NL_0 = 0.7201 here."""
    caps = exact_instantaneous_capacities(SPEC, processor_unitary(SPEC, ham, 0.0, 0.0))
    assert caps["NL_0_exact"] < 1e-9
    assert caps["capacities"][1] > 0.99       # the linear signal is still present


def test_capacity_is_structurally_zero_above_the_input_copy_count(ham):
    caps = exact_instantaneous_capacities(SPEC, processor_unitary(SPEC, ham, 0.8, 0.8),
                                          max_degree=SPEC.N_P + 2)
    for d, v in caps["degrees_above_bound"].items():
        assert v < 1e-9, (d, v)


def test_interaction_creates_nonlinearity(ham):
    off = exact_instantaneous_capacities(SPEC, processor_unitary(SPEC, ham, 0.0, 0.0))["NL_0_exact"]
    on = exact_instantaneous_capacities(SPEC, processor_unitary(SPEC, ham, 0.6, 0.6))["NL_0_exact"]
    assert on > off + 0.5


def test_analytic_two_qubit_interaction_matches_its_closed_form():
    """exp(-i t Z1Z2) maps X_1 -> cos(2t) X_1 + sin(2t) Y_1 Z_2. Starting from
    a state with <Y_1> = 0 the output is exactly cos(2t)*s -- an analytic check
    that the interaction, not the encoder, is the nonlinearity source."""
    from scipy.linalg import expm
    n, t = 2, 0.3
    mix = expm(-1j * (np.pi / 4) * embed(n, {0: "Y"}))       # rotates Z -> X on qubit 0
    U = expm(-1j * t * embed(n, {0: "Z", 1: "Z"})) @ mix
    X1, Y1 = embed(n, {0: "X"}), embed(n, {0: "Y"})

    us = np.linspace(0, 1, 41)
    s = 2 * us - 1
    got = []
    for u in us:
        r = np.array([[(1 + (2 * u - 1)) / 2, 0], [0, (1 - (2 * u - 1)) / 2]], dtype=complex)
        rho = U @ np.kron(r, r) @ U.conj().T
        got.append(expectation_values(rho, [X1])[0])

    pure0 = np.array([[1, 0], [0, 0]], dtype=complex)
    pre = mix @ np.kron(pure0, pure0) @ mix.conj().T
    assert abs(expectation_values(pre, [Y1])[0]) < 1e-12
    assert np.allclose(np.array(got), np.cos(2 * t) * s, atol=1e-12)


def test_ablations_give_distinct_unitaries(ham):
    seen = {}
    for name in ABLATIONS:
        U = processor_unitary(SPEC, ham, 0.6, 0.6, ablation=name)
        for other, U2 in seen.items():
            assert np.linalg.norm(U - U2) > 1e-8, f"{name} and {other} share a unitary"
        seen[name] = U


def test_feature_identical_ablations_are_exactly_the_provable_ones(ham):
    """V3.1 concern 7. Identical FEATURES with different UNITARIES is expected
    only for Z-diagonal channels acting on the Z-diagonal encoder ensemble."""
    rep = ablation_distinctness(SPEC, ham, 0.6, 0.6, hamiltonian_seed=5)
    assert rep["unexplained_duplicates"] == []
    for d in rep["duplicates"]:
        assert frozenset((d["a"], d["b"])) <= PROVEN_FEATURE_DEGENERATE
        assert d["dU"] > 1e-6 and d["dFeatures"] < 1e-10


def test_distinctness_report_separates_hamiltonian_unitary_state_and_feature(ham):
    rep = ablation_distinctness(SPEC, ham, 0.6, 0.6, hamiltonian_seed=5)
    pair = next(p for p in rep["pairs"] if {p["a"], p["b"]} == {"encoder_only", "two_body_only"})
    assert pair["dH"] > 1e-6          # Hamiltonians differ
    assert pair["dU"] > 1e-6          # unitaries differ
    assert pair["dFeatures"] < 1e-10  # features coincide -- by proven symmetry
    assert pair["state_trace_distance"] < 1e-10


def test_randomized_control_has_no_g_J_dependence(ham):
    a = processor_unitary(SPEC, ham, 0.1, 0.1, ablation="randomized", hamiltonian_seed=9)
    b = processor_unitary(SPEC, ham, 0.9, 0.9, ablation="randomized", hamiltonian_seed=9)
    assert np.allclose(a, b)


def test_site_disorder_does_not_break_null_linearity():
    """Disorder is single-qubit, so it must not create nonlinearity at g=J=0."""
    spec = ProcessorSpec(N_P=4, field_amp=0.9, omega_lo=0.2, omega_hi=2.0)
    h = build_hamiltonians(spec, disorder_seed=21)
    assert exact_instantaneous_capacities(spec, processor_unitary(spec, h, 0.0, 0.0))["NL_0_exact"] < 1e-9


def test_unknown_ablation_is_rejected(ham):
    with pytest.raises(ValueError):
        processor_unitary(SPEC, ham, 0.5, 0.5, ablation="nope")
