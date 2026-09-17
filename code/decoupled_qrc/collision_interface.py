"""
collision_interface.py -- the M -> A -> P collision-model channel (Part 4).

**Important correction found while testing (docs/DQRC_DIRECTIONAL_DECOUPLING_RESULTS.md
question 3)**: Part 4's literal "default candidate" U_MA(theta) =
exp(-i*theta*Z_M*Z_A) is diagonal in BOTH Z_M and Z_A. Since the ancilla is
freshly reset to |0> (an EXACT Z_A eigenstate) every timestep, a diagonal
gate can only apply a conditional PHASE to M -- it can never move any
amplitude/population into A. Verified directly: with this generator, A's
reduced purity after the M-A stage stays EXACTLY 1.0 (no entanglement at
all) regardless of theta -- the exact same mechanism
`docs/DQRC_ARCHITECTURE_REPAIR.md` found for the direct `'rzz'`/`'cp'`
interface, now hitting the FIRST link of this two-link chain. This module
therefore uses U_MA(theta) = exp(-i*theta*Z_M*X_A) instead: still EXACTLY
QND for Z_M (verified: `[Z_M (x) I_A, U_MA] = 0` to machine precision,
`tests/test_collision_interface.py`), but no longer diagonal in Z_A, so it
CAN inject real amplitude into the fresh ancilla (verified: A's purity
drops to ~0.65 for theta=0.5 starting from A=|0>). Part 4's own "or
equivalent circuit decomposition" phrasing is read as licensing this swap;
the original Z_M Z_A generator is kept available (`ma_kind='zz'`) so the
naive, non-transferring version can still be reproduced and reported on
explicitly, never silently dropped.

A FRESH ancilla A is introduced every timestep: reset to |0>, coupled to
memory via U_MA(theta) (QND for Z_M), then coupled to the processor via a
NON-commuting interaction U_AP(phi) (xy / heisenberg / anisotropic, reused
from `interfaces_advanced.exchange_unitary` rather than re-derived), then
DISCARDED (reset) before the next timestep. This realizes the CPTP
collision-model channel

    rho_MP' = Tr_A[ U_AP U_MA (rho_MP (x) |0><0|_A) U_MA^dag U_AP^dag ]

by construction (Aer's `reset` is an exact partial trace + fresh-state
reinitialization, not an approximation of one).

Both stages are built as EXACT 4x4 unitaries via
`interfaces_advanced.exchange_unitary` (already proven correct and
regression-tested for qubit ordering in the previous DQRC-repair pass) --
no new low-level gate-ordering code is introduced here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from .interfaces_advanced import exchange_unitary, _kind_to_coeffs, ADVANCED_KINDS

AP_KINDS = ("xx", "xy", "anisotropic")


def _ap_coeffs(kind: str, phi_x: float, phi_y: float, phi_z: float) -> dict:
    if kind == "xx":
        return dict(xx=1.0)
    if kind == "xy":
        return dict(xx=1.0, yy=1.0)
    if kind == "anisotropic":
        return dict(xx=phi_x, yy=phi_y, zz=phi_z)
    raise ValueError(f"kind must be one of {AP_KINDS}, got {kind!r}")


def _ma_unitary(theta: float, ma_kind: str = "zx") -> np.ndarray:
    if ma_kind == "zx":
        return exchange_unitary(theta, zx=1.0)   # QND for Z_M, CAN transfer amplitude to A
    if ma_kind == "zz":
        return exchange_unitary(theta, zz=1.0)   # Part 4's literal default -- QND but CANNOT transfer (see module docstring)
    raise ValueError(f"ma_kind must be 'zx' or 'zz', got {ma_kind!r}")


def apply_collision_step(qc: QuantumCircuit, m_qubit: int, a_qubit: int, p_qubit: int, theta: float,
                          phi: float, ap_kind: str = "xy", phi_x: float = 1.0, phi_y: float = 1.0,
                          phi_z: float = 0.0, ma_kind: str = "zx"):
    """One full M -> A -> P collision, IN PLACE: reset A, apply U_MA(theta)
    (QND for Z_M; `ma_kind='zx'` -- the default, actually transfers
    amplitude to a fresh ancilla; `ma_kind='zz'` -- Part 4's literal
    default, which does NOT, see module docstring), apply U_AP(phi) (one of
    `AP_KINDS`, non-commuting), then reset A again (discard before the
    next timestep uses it). `theta=0` or `phi=0` are exact identities for
    their respective stage (same convention as `interface.py`/
    `interfaces_advanced.py`)."""
    qc.reset(a_qubit)
    if theta != 0:
        U_MA = _ma_unitary(theta, ma_kind=ma_kind)
        qc.append(UnitaryGate(U_MA, label="U_MA"), [m_qubit, a_qubit])
    if phi != 0:
        coeffs = _ap_coeffs(ap_kind, phi_x, phi_y, phi_z)
        U_AP = exchange_unitary(phi, **coeffs)
        qc.append(UnitaryGate(U_AP, label="U_AP"), [a_qubit, p_qubit])
    qc.reset(a_qubit)


def qnd_commutator_norm(theta: float, ma_kind: str = "zx") -> float:
    """||[U_MA(theta), Z_M (x) I_A]||_F -- should be near machine zero for
    a QND M-A interaction, by construction (Part 14's first,
    necessary-but-not-sufficient check). Computed directly from the exact
    4x4 matrix, not assumed. `Z_M` is built as `kron(I, Z)` because
    `qc.append(gate, [m_qubit, a_qubit])` places `m_qubit` as the LEAST
    significant (rightmost-in-kron) tensor factor under Qiskit's
    little-endian convention -- the same convention
    `interfaces_advanced._term` already accounts for; getting this backwards
    here would silently report a false QND pass/fail for any ASYMMETRIC
    generator (it happens to not matter for the symmetric 'zz' case, which
    is exactly why the earlier version of this function went unnoticed
    before the 'zx' generator was introduced)."""
    if theta == 0:
        return 0.0
    U = _ma_unitary(theta, ma_kind=ma_kind)
    Z_M = np.kron(np.eye(2, dtype=np.complex128), np.array([[1, 0], [0, -1]], dtype=np.complex128))
    comm = U @ Z_M - Z_M @ U
    return float(np.linalg.norm(comm))
