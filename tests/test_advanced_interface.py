"""
test_advanced_interface.py -- exact-generator interfaces (Part 9) and
back-action metrics (Part 10).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.interfaces_advanced import (  # noqa: E402
    exchange_unitary, apply_advanced_interface, apply_multi_edge_interface, trace_distance,
    memory_disturbance, transfer_efficiency,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace


def test_exchange_unitary_is_unitary():
    for kind_coeffs in ({"xx": 1.0}, {"xx": 1.0, "yy": 1.0}, {"xx": 1.0, "yy": 1.0, "zz": 0.3},
                         {"xx": 0.5, "yy": 0.2, "zz": 0.1, "zx": 0.4}):
        U = exchange_unitary(0.37, **kind_coeffs)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)


def test_lam_zero_is_identity_for_every_kind():
    qc0 = QuantumCircuit(2)
    for kind in ("zx", "xy", "heisenberg", "multiaxis"):
        qc = QuantumCircuit(2)
        apply_advanced_interface(qc, [0], [1], lam=0.0, kind=kind)
        assert Operator(qc) == Operator(qc0)


def test_xy_exchange_swaps_a_single_excitation():
    """The XY (X_M X_P + Y_M Y_P) exchange at lam=pi/4 should act as a
    genuine hopping/beam-splitter term: starting with qubit0 excited (Qiskit
    little-endian bitstring '01'), a full-transfer coupling should move that
    excitation to qubit1 (bitstring '10') -- a real, hand-verifiable
    positive control (unlike a diagonal ZZ gate, which could never do this
    -- see docs/DQRC_ARCHITECTURE_REPAIR.md)."""
    qc = QuantumCircuit(2)
    qc.x(0)  # bitstring '01' (qubit0=1, qubit1=0)
    apply_advanced_interface(qc, [0], [1], lam=np.pi / 4, kind="xy")
    sv = DensityMatrix(qc)
    probs = sv.probabilities_dict()
    assert probs.get("10", 0.0) > 0.9, probs  # excitation fully transferred to qubit1


def test_zx_kind_matches_interface_module_convention():
    """`apply_advanced_interface(..., kind='zx')` should match
    `interface.apply_interface(..., kind='zx')`'s physical effect (both are
    exp(-i*lam*Z_M X_P)), even though one is built via expm and the other
    via H-RZZ-H -- a cross-check that this module's more general machinery
    reduces correctly to the simple case."""
    from decoupled_qrc.interface import apply_interface
    qc1 = QuantumCircuit(2)
    qc1.h(0); qc1.x(1)  # arbitrary nontrivial starting state
    apply_advanced_interface(qc1, [0], [1], lam=0.4, kind="zx")

    qc2 = QuantumCircuit(2)
    qc2.h(0); qc2.x(1)
    apply_interface(qc2, [0], [1], lambda_mp=0.4, kind="zx")

    assert np.allclose(np.asarray(Operator(qc1).data), np.asarray(Operator(qc2).data), atol=1e-9)


def test_multi_edge_interface_applies_independent_strengths():
    """Qubit3 is only ever named in the SECOND edge, whose strength is 0
    (a no-op) -- qubit3 must stay in an exact product state with everything
    else (purity 1), regardless of what happens on the (0,2) edge."""
    qc = QuantumCircuit(4)  # qubits 0,1 = memory; 2,3 = processor
    qc.h(0); qc.h(1)
    edges = [(0, 2, 0.3), (1, 3, 0.0)]  # second edge has zero strength -> no-op
    apply_multi_edge_interface(qc, edges, kind="xy")
    dm = DensityMatrix(qc)
    rho_3 = partial_trace(dm, [0, 1, 2])
    purity_3 = float(np.real(np.trace(np.asarray(rho_3.data) @ np.asarray(rho_3.data))))
    assert purity_3 == pytest.approx(1.0, abs=1e-9)


def test_trace_distance_zero_for_identical_states():
    qc = QuantumCircuit(2)
    qc.h(0)
    rho = DensityMatrix(qc)
    assert trace_distance(rho, rho) == pytest.approx(0.0, abs=1e-9)


def test_trace_distance_positive_for_orthogonal_states():
    from qiskit.quantum_info import DensityMatrix as DM
    rho0 = DM.from_label("0")
    rho1 = DM.from_label("1")
    assert trace_distance(rho0, rho1) == pytest.approx(1.0, abs=1e-9)


def test_memory_disturbance_zero_when_no_interface_difference():
    qc = QuantumCircuit(3)
    qc.h(0); qc.h(1)
    rho = DensityMatrix(qc)
    result = memory_disturbance(rho, rho, mem_qubits=[0, 1], all_qubits=[0, 1, 2])
    assert result["trace_distance"] == pytest.approx(0.0, abs=1e-9)
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-6)


def test_transfer_efficiency_ratio():
    assert transfer_efficiency(nl_delta=0.5, disturbance=0.25) == pytest.approx(2.0, rel=1e-3)
    assert transfer_efficiency(nl_delta=0.0, disturbance=0.0) == pytest.approx(0.0, abs=1e-3)
