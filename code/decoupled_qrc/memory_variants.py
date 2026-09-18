"""
memory_variants.py -- V3.1's two preregistered memory variants.

V3's memory (M0, fractional SWAP `SWAP^m`) showed a coarse scan rising
monotonically (M = 1.478 -> 3.450 as m went 0.3 -> 0.9) while the LOCAL
five-point derivative at m = 0.7 came out NEGATIVE (-4.85) and UNSTABLE
(relative spread 0.46). Those two facts cannot both describe a simple
retention knob, which suggests `m` in M0 changes several things at once
(retention, transport speed, and the phase accumulated per hop, since
`SWAP^m = P_+ + e^{i pi m} P_-` rotates the singlet sector).

M1 therefore implements retention as an EXPLICIT CPTP channel whose only
effect is how much of the previous state survives:

    E_m(rho) = m * U_shift rho U_shift^dag + (1 - m) * rho_reference

with `U_shift` the FULL SWAP shift register (fixed, m-independent) and
`rho_reference` a fixed reference state. This is manifestly trace
preserving and completely positive (a convex mixture of a unitary channel
and a state-preparation channel), and `m` has one unambiguous physical
meaning: the retained fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Kraus, Pauli

from .memory_bank import (MemoryBankConfig, memory_feature_ops, encode_diagonal_z,
                           fractional_swap_matrix, to_signed, run_memory_bank)
from .utils import ensure_repo_code_on_path
from .v3_seeds import NestedSeeds

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator  # noqa: E402

MEMORY_VARIANTS = ("M0", "M1")

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


@dataclass(frozen=True)
class MemoryVariantConfig:
    variant: str = "M1"
    L: int = 4
    m: float = 0.7               # retention: M0 -> SWAP exponent; M1 -> retained fraction
    max_weight: int = 2

    def as_dict(self) -> dict:
        return asdict(self)

    def to_bank_config(self) -> MemoryBankConfig:
        return MemoryBankConfig(L=self.L, m=self.m, max_weight=self.max_weight)


def retention_kraus(m: float) -> Kraus:
    """Kraus operators for the single-rail retention channel

        E_m(rho) = m * rho + (1-m) * |0><0|

    i.e. with probability (1-m) the rail is reset to the fixed reference
    state |0>. Kraus set: {sqrt(m) I, sqrt(1-m)|0><0|, sqrt(1-m)|0><1|},
    which satisfies sum_k K_k^dag K_k = I exactly, so the map is CPTP."""
    m = float(np.clip(m, 0.0, 1.0))
    k0 = np.sqrt(m) * np.eye(2, dtype=complex)
    k1 = np.sqrt(1.0 - m) * np.array([[1, 0], [0, 0]], dtype=complex)
    k2 = np.sqrt(1.0 - m) * np.array([[0, 1], [0, 0]], dtype=complex)
    return Kraus([k0, k1, k2])


def is_trace_preserving(kraus: Kraus, atol: float = 1e-10) -> bool:
    total = sum(np.asarray(k).conj().T @ np.asarray(k) for k in kraus.data)
    return bool(np.allclose(total, np.eye(total.shape[0]), atol=atol))


def build_memory_circuit_variant(cfg: MemoryVariantConfig, u_seq, seeds: NestedSeeds):
    """M0 delegates to the V3 fractional-SWAP bank. M1 uses a FIXED full-SWAP
    shift register plus an explicit per-rail retention channel controlled by
    `m`, so `m` affects retention and nothing else."""
    rails = list(range(cfg.L))
    labels, ops = memory_feature_ops(rails, max_weight=cfg.max_weight)
    qc = QuantumCircuit(cfg.L)

    if cfg.variant == "M0":
        shift = UnitaryGate(fractional_swap_matrix(cfg.m), label=f"SWAP^{cfg.m:.3f}")
    elif cfg.variant == "M1":
        shift = UnitaryGate(SWAP, label="SWAP")
        channel = retention_kraus(cfg.m)
    else:
        raise ValueError(f"unknown memory variant {cfg.variant!r}")

    for t, u_t in enumerate(u_seq):
        qc.reset(0)
        encode_diagonal_z(qc, 0, u_t)
        for r in range(cfg.L - 1, 0, -1):
            qc.append(shift, [r - 1, r])
        if cfg.variant == "M1":
            # retention applies to the STORAGE rails; rail 0 is re-encoded each step
            for r in rails[1:]:
                qc.append(channel.to_instruction(), [r])
        for (pauli, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(pauli, qargs, label=f"mem_{lab}__t{t}")

    return qc, labels


@dataclass
class MemoryVariantRun:
    u: np.ndarray
    labels: list
    X_M: np.ndarray
    cfg: MemoryVariantConfig
    circuit_depth: int = 0


def run_memory_variant(cfg: MemoryVariantConfig, u_seq, seeds: NestedSeeds,
                        method: str = "density_matrix", use_gpu: bool = False) -> MemoryVariantRun:
    if method != "density_matrix":
        raise ValueError("memory circuits reset a rail every step -- use method='density_matrix'.")
    u = np.asarray(u_seq, dtype=float)
    qc, labels = build_memory_circuit_variant(cfg, u, seeds)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    data = sim.run(tqc, shots=1).result().data(0)
    X_M = np.array([[float(np.real(data[f"mem_{lab}__t{t}"])) for lab in labels]
                     for t in range(len(u))])
    return MemoryVariantRun(u=u, labels=labels, X_M=X_M, cfg=cfg, circuit_depth=tqc.depth())


def audit_m0_semantics(L: int = 4, m_values=(0.3, 0.5, 0.7, 0.9, 1.0)) -> dict:
    """What does `m` actually change in M0? `SWAP^m = P_+ + e^{i pi m} P_-`,
    so it is a PHASE rotation in the singlet sector rather than a clean
    retention knob: the transferred amplitude and the accumulated phase move
    together, and the map is periodic in m rather than monotone. This
    function reports the transfer fidelity and the singlet phase so the
    ambiguity is documented rather than assumed."""
    out = {}
    for m in m_values:
        u = fractional_swap_matrix(m)
        # amplitude actually transferred |01> -> |10>
        transfer = float(abs(u[2, 1]) ** 2)
        retained = float(abs(u[1, 1]) ** 2)
        phase = float(np.angle(np.exp(1j * np.pi * m)))
        out[m] = {"transfer_probability": transfer, "retained_probability": retained,
                  "singlet_phase_rad": phase}
    out["_note"] = ("SWAP^m mixes transfer amplitude with a singlet-sector phase e^{i pi m}; "
                    "m is therefore NOT a pure retention parameter in M0, which is why M1 "
                    "implements retention as an explicit CPTP channel instead")
    return out
