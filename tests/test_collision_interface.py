"""
test_collision_interface.py -- the M -> A -> P collision channel (Part 4)
and its QND property (Part 14).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.collision_interface import (  # noqa: E402
    apply_collision_step, qnd_commutator_norm, AP_KINDS,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace


def test_qnd_commutator_is_machine_zero():
    for theta in (0.1, 0.3, 1.0, np.pi / 2):
        assert qnd_commutator_norm(theta) < 1e-12


def test_qnd_commutator_zero_at_theta_zero():
    assert qnd_commutator_norm(0.0) == 0.0


def test_ancilla_is_reset_before_and_after_collision():
    """The ancilla must be |0> both immediately before the M-A interaction
    (guaranteed by the leading reset) and after the A-P interaction
    (guaranteed by the trailing reset) -- verified by checking its reduced
    state is exactly |0><0| after `apply_collision_step`, regardless of
    what memory/processor were doing beforehand."""
    qc = QuantumCircuit(3)  # 0=M, 1=A, 2=P
    qc.h(0)  # arbitrary nontrivial memory state
    qc.x(1)  # ancilla starts "dirty" -- must still end up reset
    qc.h(2)  # arbitrary nontrivial processor state
    apply_collision_step(qc, m_qubit=0, a_qubit=1, p_qubit=2, theta=0.4, phi=0.6, ap_kind="xy")
    rho = DensityMatrix(qc)
    rho_a = partial_trace(rho, [0, 2])
    assert np.allclose(np.asarray(rho_a.data), np.array([[1, 0], [0, 0]]), atol=1e-9)


def test_theta_zero_skips_ma_interaction_phi_nonzero_still_acts():
    """theta=0 should leave M completely unentangled with A/P even though
    phi!=0 (A-P interaction can't touch M if U_MA was never applied)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    apply_collision_step(qc, m_qubit=0, a_qubit=1, p_qubit=2, theta=0.0, phi=0.6, ap_kind="xy")
    rho = DensityMatrix(qc)
    rho_m = partial_trace(rho, [1, 2])
    purity_m = float(np.real(np.trace(np.asarray(rho_m.data) @ np.asarray(rho_m.data))))
    assert purity_m == pytest.approx(1.0, abs=1e-9)


def test_phi_zero_leaves_processor_untouched():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(2)
    apply_collision_step(qc, m_qubit=0, a_qubit=1, p_qubit=2, theta=0.4, phi=0.0, ap_kind="xy")
    rho = DensityMatrix(qc)
    rho_p = partial_trace(rho, [0, 1])
    purity_p = float(np.real(np.trace(np.asarray(rho_p.data) @ np.asarray(rho_p.data))))
    assert purity_p == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("kind", AP_KINDS)
def test_every_ap_kind_can_entangle_ancilla_with_processor(kind):
    """Positive control: with an ancilla in a GENERIC (non-eigenstate-of-
    the-coupling-basis) superposition, each A-P kind should be able to
    entangle A with P for a generic nonzero phi -- proving the A-P stage is
    a real, working coupling for every advertised kind. Uses `ry(0.9)`
    rather than `h` deliberately: `h` puts the ancilla in an exact X
    eigenstate, which a PURE X_A X_P ('xx') coupling cannot entangle for
    the same diagonal-gate/eigenstate reason this module's own docstring
    documents for the M-A stage -- that would be a test-construction
    artifact, not a real failure of the 'xx' kind."""
    qc = QuantumCircuit(3)
    qc.ry(0.9, 1)  # generic ancilla superposition, not an eigenstate of X, Y, or Z
    from decoupled_qrc.collision_interface import _ap_coeffs
    from decoupled_qrc.interfaces_advanced import exchange_unitary
    from qiskit.circuit.library import UnitaryGate
    coeffs = _ap_coeffs(kind, phi_x=0.6, phi_y=0.3, phi_z=0.2)
    U = exchange_unitary(0.7, **coeffs)
    qc.append(UnitaryGate(U), [1, 2])
    rho = DensityMatrix(qc)
    rho_p = partial_trace(rho, [0, 1])
    purity_p = float(np.real(np.trace(np.asarray(rho_p.data) @ np.asarray(rho_p.data))))
    assert purity_p < 0.999, f"{kind} failed to entangle ancilla with processor"


def test_zz_ma_kind_cannot_transfer_to_fresh_ancilla():
    """THE central finding of this module (see its own docstring): Part
    4's literal U_MA = exp(-i*theta*Z_M*Z_A) is diagonal in BOTH Z_M and
    Z_A, so it can never move amplitude into a freshly-reset (exact
    Z-eigenstate) ancilla -- A's purity must stay EXACTLY 1.0 regardless of
    theta."""
    from decoupled_qrc.collision_interface import _ma_unitary
    from qiskit.circuit.library import UnitaryGate as _UG
    for theta in (0.3, 0.8, np.pi / 2):
        qc = QuantumCircuit(2)
        qc.h(0)  # M in a generic superposition
        U = _ma_unitary(theta, ma_kind="zz")
        qc.append(_UG(U), [0, 1])
        rho = DensityMatrix(qc)
        rho_a = partial_trace(rho, [0])
        purity_a = float(np.real(np.trace(np.asarray(rho_a.data) @ np.asarray(rho_a.data))))
        assert purity_a == pytest.approx(1.0, abs=1e-9), (theta, purity_a)


def test_zx_ma_kind_transfers_to_fresh_ancilla_while_staying_qnd():
    """The fix: 'zx' is still exactly QND for Z_M (commutator machine-zero,
    already checked by `test_qnd_commutator_is_machine_zero`) but DOES
    transfer real amplitude into a fresh ancilla."""
    from decoupled_qrc.collision_interface import _ma_unitary
    from qiskit.circuit.library import UnitaryGate as _UG
    qc = QuantumCircuit(2)
    qc.h(0)
    U = _ma_unitary(0.5, ma_kind="zx")
    qc.append(_UG(U), [0, 1])
    rho = DensityMatrix(qc)
    rho_a = partial_trace(rho, [0])
    purity_a = float(np.real(np.trace(np.asarray(rho_a.data) @ np.asarray(rho_a.data))))
    assert purity_a < 0.9, purity_a
