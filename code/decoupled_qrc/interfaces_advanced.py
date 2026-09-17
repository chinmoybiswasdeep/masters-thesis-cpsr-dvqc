"""
interfaces_advanced.py -- Part 9/10: richer memory<->processor couplings
than `interface.py`'s single RZZ/CP/ZX primitive, plus directionality/
back-action metrics.

Every interface below is built as an EXACT 4x4 unitary
(`scipy.linalg.expm` of the requested Pauli-sum generator, wrapped as a
`UnitaryGate`) rather than a Trotter product of native gates -- most of
these generators do NOT pairwise commute (unlike the XX+YY+ZZ combination,
which does), so an exact exponential avoids introducing Trotter error that
would otherwise confound "how much information transfers" with "how much
Trotter error we introduced."

    A. 'zx':         lambda * Z_M X_P
    B. 'xy':          lambda * (X_M X_P + Y_M Y_P)
    C. 'heisenberg':  lambda * (X_M X_P + Y_M Y_P + eta * Z_M Z_P)
    D. 'multiaxis':   lambda_xx X_M X_P + lambda_yy Y_M Y_P + lambda_zz Z_M Z_P + lambda_zx Z_M X_P
    E. multi-edge:    sum_i sum_j J_ij O_i O_j -- apply any of A-D across a
                       LIST of (memory_qubit, processor_qubit) edges, each
                       with its own strength (`apply_multi_edge_interface`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}

ADVANCED_KINDS = ("zx", "xy", "heisenberg", "multiaxis")


def _term(p1: str, p2: str) -> np.ndarray:
    """`p1` acts on the FIRST qubit passed to `qc.append(gate, [q1, q2])`
    (conventionally the memory qubit `m`), `p2` on the second (`p`). Qiskit
    orders a multi-qubit gate's matrix little-endian -- the LAST qubit in
    the `qargs` list is the MOST significant tensor factor -- so building
    this as `kron(P[p1], P[p2])` would silently swap which physical qubit
    each Pauli lands on for any ASYMMETRIC term (zx, xz); verified directly
    in `tests/test_advanced_interface.py::test_zx_kind_matches_interface_module_convention`,
    which caught exactly this bug during development."""
    return np.kron(_PAULI[p2], _PAULI[p1])


def exchange_unitary(lam: float, xx: float = 0.0, yy: float = 0.0, zz: float = 0.0,
                      zx: float = 0.0, xz: float = 0.0) -> np.ndarray:
    """exp(-i * lam * (xx*XX + yy*YY + zz*ZZ + zx*ZX + xz*XZ)) as an exact
    4x4 unitary (`scipy.linalg.expm`). Coefficients are dimensionless
    multipliers on `lam` (the overall coupling-strength control), so
    `exchange_unitary(lam, xx=1)` alone reduces to the standard
    RXX(2*lam)-equivalent single-term coupling."""
    H = xx * _term("X", "X") + yy * _term("Y", "Y") + zz * _term("Z", "Z") \
        + zx * _term("Z", "X") + xz * _term("X", "Z")
    return expm(-1j * lam * H)


def _kind_to_coeffs(kind: str, eta: float, lambda_xx: float, lambda_yy: float, lambda_zz: float,
                     lambda_zx: float) -> dict:
    if kind == "zx":
        return dict(zx=1.0)
    if kind == "xy":
        return dict(xx=1.0, yy=1.0)
    if kind == "heisenberg":
        return dict(xx=1.0, yy=1.0, zz=eta)
    if kind == "multiaxis":
        return dict(xx=lambda_xx, yy=lambda_yy, zz=lambda_zz, zx=lambda_zx)
    raise ValueError(f"kind must be one of {ADVANCED_KINDS}, got {kind!r}")


def apply_advanced_interface(qc: QuantumCircuit, memory_taps: Sequence[int], processor_entry: Sequence[int],
                              lam: float, kind: str = "xy", eta: float = 0.3, lambda_xx: float = 1.0,
                              lambda_yy: float = 1.0, lambda_zz: float = 0.0, lambda_zx: float = 0.0):
    """Apply the requested exchange unitary between each paired
    (memory_tap, processor_entry) qubit. `lam=0` is an exact identity for
    every kind (all generators vanish), matching `interface.apply_interface`'s
    convention."""
    if lam == 0:
        return
    if len(memory_taps) != len(processor_entry):
        raise ValueError("memory_taps and processor_entry must have equal length")
    coeffs = _kind_to_coeffs(kind, eta, lambda_xx, lambda_yy, lambda_zz, lambda_zx)
    U = exchange_unitary(lam, **coeffs)
    gate = UnitaryGate(U, label=f"MP_{kind}")
    for m, p in zip(memory_taps, processor_entry):
        qc.append(gate, [m, p])


def apply_multi_edge_interface(qc: QuantumCircuit, edges: Sequence[tuple], kind: str = "xy", eta: float = 0.3):
    """Part 9E: sum_i sum_j J_ij O_i O_j -- `edges` is a list of
    (memory_qubit, processor_qubit, J_ij) triples, each independently
    coupled with the SAME `kind` generator but its OWN strength `J_ij`
    (sparse vs. dense connectivity is just how many edges are passed)."""
    for m, p, j in edges:
        apply_advanced_interface(qc, [m], [p], lam=j, kind=kind, eta=eta)


# =============================================================================
# Part 10 -- directionality / back-action metrics
# =============================================================================

def trace_distance(rho1: DensityMatrix, rho2: DensityMatrix) -> float:
    """0.5 * sum(|eigenvalues(rho1 - rho2)|) -- the standard trace-distance
    metric between two density matrices of the same dimension (not provided
    directly by qiskit.quantum_info as a named function for two arbitrary
    DensityMatrix objects, hence computed here from their `.data`)."""
    diff = np.asarray(rho1.data) - np.asarray(rho2.data)
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def memory_disturbance(rho_full_with_interface: DensityMatrix, rho_full_without_interface: DensityMatrix,
                        mem_qubits: Sequence[int], all_qubits: Sequence[int]) -> dict:
    """Compares the memory register's OWN reduced state with vs. without
    the M->P interface active (same trajectory otherwise) -- an engineering
    metric (Part 10 explicitly: 'not fundamental physics'), not a claim
    about a physical law."""
    other = [q for q in all_qubits if q not in mem_qubits]
    rho_m_with = partial_trace(rho_full_with_interface, other)
    rho_m_without = partial_trace(rho_full_without_interface, other)
    td = trace_distance(rho_m_with, rho_m_without)
    fid = float(state_fidelity(rho_m_with, rho_m_without))
    purity_with = float(np.real(np.trace(np.asarray(rho_m_with.data) @ np.asarray(rho_m_with.data))))
    purity_without = float(np.real(np.trace(np.asarray(rho_m_without.data) @ np.asarray(rho_m_without.data))))
    return {"trace_distance": td, "fidelity": fid, "purity_with_interface": purity_with,
            "purity_without_interface": purity_without,
            "purity_change": purity_with - purity_without}


def transfer_efficiency(nl_delta: float, disturbance: float, eps: float = 1e-9) -> float:
    """(increase in processor predictive capacity) / (memory disturbance) --
    an ENGINEERING ratio (Part 10), not treated as fundamental."""
    return nl_delta / (disturbance + eps)
