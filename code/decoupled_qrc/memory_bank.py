"""
memory_bank.py -- V3's memory module: an L-rail routed delay bank with a
GENUINE internal retention control `m`.

Design:
  - rail 0 is the input rail; rails 1..L-1 are storage.
  - each macro timestep: reset rail 0, encode rho(u_t) = (I + u_t Z)/2 on
    it (diagonal encoding, via Ry(theta) with cos(theta) = u_t), then
    route information down the rails with a FRACTIONAL SWAP.
  - `m` is the fractional-SWAP exponent: SWAP^m between adjacent rails.
      m = 1  -> full SWAP  -> a perfect shift register (maximal retention)
      m = 0  -> identity   -> no transfer at all (zero retention)
      0<m<1  -> partial transfer per step (tunable retention)

`m` is therefore a physically-implemented channel parameter that changes
the memory's INTERNAL transfer dynamics. It is emphatically NOT a change
to the target delay horizon, input sequence, readout regularization,
feature count, or number of measured observables -- the V3 spec's
explicit forbidden list.

The processor's controls (g, J) never appear anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli

from .utils import ensure_repo_code_on_path
from .v3_seeds import NestedSeeds

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator  # noqa: E402


@dataclass(frozen=True)
class MemoryBankConfig:
    L: int = 4                       # total rails (1 input + L-1 storage)
    m: float = 0.8                   # fractional-SWAP exponent -- the retention control
    max_weight: int = 2              # readout Pauli weight for memory features
    encoding: str = "diagonal_z"     # rho(u) = (I + u Z)/2

    def as_dict(self) -> dict:
        return {"L": self.L, "m": self.m, "max_weight": self.max_weight, "encoding": self.encoding}


def fractional_swap_matrix(m: float) -> np.ndarray:
    """SWAP^m in the computational basis. SWAP has eigenvalues +1 (triplet,
    3-fold) and -1 (singlet); SWAP^m = exp(i*pi*m*(SWAP-I)/2)-style
    fractional power, built here by diagonalizing exactly:
        SWAP^m = P_+ + exp(i*pi*m) P_-
    where P_± are the symmetric/antisymmetric projectors. m=1 recovers
    SWAP exactly (exp(i*pi) = -1); m=0 recovers the identity."""
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    identity = np.eye(4, dtype=complex)
    p_minus = (identity - swap) / 2.0     # antisymmetric (singlet) projector
    p_plus = (identity + swap) / 2.0      # symmetric (triplet) projector
    return p_plus + np.exp(1j * np.pi * m) * p_minus


def to_signed(u):
    """Map this repository's canonical input convention u in [0,1] (what
    `qrc_qiskit.random_input` produces and what every IPC target builder
    expects, since `ipc.to_v` applies v = 2u-1 internally) onto the
    signed [-1,1] range the diagonal encoding needs. Keeping [0,1] as THE
    canonical representation everywhere avoids the domain mismatch that
    would otherwise silently corrupt the Legendre targets."""
    return 2.0 * np.asarray(u, dtype=float) - 1.0


def encode_diagonal_z(qc: QuantumCircuit, qubit: int, u: float) -> None:
    """Prepare rho = (I + s Z)/2 on `qubit` (assumed freshly reset to
    |0>), where s = 2u-1 is the signed form of the canonical input
    u in [0,1]. A pure state with <Z> = s is obtained by Ry(theta) with
    cos(theta) = s."""
    s = float(np.clip(to_signed(u), -1.0, 1.0))
    qc.ry(float(np.arccos(s)), qubit)


def memory_feature_ops(rails, max_weight: int = 2):
    """Module-specific memory observables: single-rail X/Y/Z plus
    adjacent-rail two-body terms up to `max_weight`. Depends only on the
    rail layout -- never on m, g, or J (so the feature DEFINITION is
    identical across every control setting, per the spec's requirement
    that m must not change the feature count)."""
    import itertools
    paulis = ("Z", "X", "Y")
    labels, ops = [], []
    for q in rails:
        for p in paulis:
            labels.append(f"{p}{q}")
            ops.append((Pauli(p), [q]))
    if max_weight >= 2:
        for q1, q2 in itertools.combinations(rails, 2):
            for p1, p2 in itertools.product(paulis, repeat=2):
                labels.append(f"{p1}{q1}{p2}{q2}")
                # Qiskit's qargs convention makes the LAST qubit the MOST significant
                # tensor factor, so the Pauli string is written reversed relative to qargs.
                ops.append((Pauli(p2 + p1), [q1, q2]))
    return labels, ops


def build_memory_circuit(cfg: MemoryBankConfig, u_seq, seeds: NestedSeeds) -> tuple:
    """Builds the L-rail memory circuit for the whole input sequence,
    saving per-timestep expectation values of the memory observables."""
    rails = list(range(cfg.L))
    qc = QuantumCircuit(cfg.L)
    labels, ops = memory_feature_ops(rails, max_weight=cfg.max_weight)
    swap_gate = UnitaryGate(fractional_swap_matrix(cfg.m), label=f"SWAP^{cfg.m:.3f}")

    for t, u_t in enumerate(u_seq):
        qc.reset(0)
        encode_diagonal_z(qc, 0, u_t)
        # route DOWN the rails, far end first, so each step moves information
        # one rail further without overwriting what has not yet been moved.
        for r in range(cfg.L - 1, 0, -1):
            qc.append(swap_gate, [r - 1, r])
        for (pauli, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(pauli, qargs, label=f"mem_{lab}__t{t}")

    return qc, labels


@dataclass
class MemoryBankRun:
    u: np.ndarray
    labels: list
    X_M: np.ndarray
    cfg: MemoryBankConfig
    circuit_depth: int = 0


def run_memory_bank(cfg: MemoryBankConfig, u_seq, seeds: NestedSeeds,
                     method: str = "density_matrix", use_gpu: bool = False) -> MemoryBankRun:
    """Runs the memory bank alone. Note `seeds` is accepted (and its
    Hamiltonian/disorder streams are deliberately UNUSED -- this memory is
    disorder-free by design) so that callers always pass the full seed
    object and every saved row can record it."""
    if method != "density_matrix":
        raise ValueError("memory-bank circuits reset a rail every step -- use method='density_matrix'.")
    u = np.asarray(u_seq, dtype=float)
    qc, labels = build_memory_circuit(cfg, u, seeds)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    data = sim.run(tqc, shots=1).result().data(0)
    X_M = np.array([[float(np.real(data[f"mem_{lab}__t{t}"])) for lab in labels] for t in range(len(u))])
    return MemoryBankRun(u=u, labels=labels, X_M=X_M, cfg=cfg, circuit_depth=tqc.depth())
