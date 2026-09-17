"""
idqnn_memory.py -- IDQNN-INSPIRED Spatial Quantum Memory (ID-SQM): an
OPTIONAL shallow local-block transform applied on top of `spatial_memory.py`'s
plain shift register (Part 6).

TERMINOLOGY (Part 1, strictly enforced): this is called "ID-SQM" or
"IDQNN-inspired spatial memory" EVERYWHERE in this module and its outputs --
NEVER "IDQNN" outright. Huang et al.'s actual IDQNN construction requires an
explicit shallow-wide circuit whose output distribution is NUMERICALLY
VERIFIED to match a corresponding DEEP circuit's, with reported ancilla
overhead and (if the specific mapping needs it) measurement/feed-forward.
Nothing here does that -- this is a small, FIXED (not learned, not tuned to
maximize any downstream metric per Part 6's explicit instruction), local
entangling transform applied to the shift register's own qubits, motivated
by the IDQNN idea of "spatializing" information but not a verified instance
of it. The genuine (separate, much harder, optional, gated) IDQNN attempt
lives in `experimental/idqnn_memory_prototype.py`.

The transform is deliberately NOT tuned to maximize nonlinear IPC (Part 6:
"Do NOT tune this circuit to maximize nonlinear IPC. Tune or choose it to
preserve distinguishability / accessible memory.") -- `entangle_strength`
defaults to a small, fixed value chosen only to keep the transform close to
identity (weak local mixing), not searched over.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli

from .spatial_memory import apply_shift_step, amplitude_damping_kraus, memory_local_ops
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator  # noqa: E402


def block_partition(mem_qubits: Sequence[int], block_size: int) -> list:
    """[M0 M1] -> B0, [M2 M3] -> B1, ... -- the last block may be smaller."""
    return [list(mem_qubits[i:i + block_size]) for i in range(0, len(mem_qubits), block_size)]


def _idqnn_block_subcircuit(n: int, seed: int, entangle_strength: float) -> QuantumCircuit:
    """ONE shallow layer on a bare n-qubit block: a small FIXED (seeded,
    not learned) Ry+Rz on each qubit, then a chain of CP(entangle_strength)
    within the block. Depth is constant (3) regardless of block size --
    'keep physical depth small' (Part 6)."""
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n)
    theta = rng.uniform(-0.3, 0.3, n)
    phi = rng.uniform(-0.3, 0.3, n)
    for q in range(n):
        qc.ry(theta[q], q)
        qc.rz(phi[q], q)
    for q in range(n - 1):
        qc.cp(entangle_strength, q, q + 1)
    return qc


def apply_idqnn_spatial_transform(qc: QuantumCircuit, mem_qubits: Sequence[int], block_size: int = 2,
                                   seed: int = 0, entangle_strength: float = 0.3):
    """Apply `_idqnn_block_subcircuit` to each block of `block_partition`,
    IN PLACE, at the block's own global qubit indices."""
    blocks = block_partition(mem_qubits, block_size)
    for i, block in enumerate(blocks):
        if len(block) < 2:
            continue  # a size-1 remainder block has nothing to entangle
        sub = _idqnn_block_subcircuit(len(block), seed=seed + i, entangle_strength=entangle_strength)
        qc.compose(sub, qubits=block, inplace=True)


def build_id_sqm_circuit(L: int, u_seq: Sequence[float], gamma_M: float = 0.0, block_size: int = 2,
                          transform_seed: int = 0, entangle_strength: float = 0.3, input_qubit: int = 0):
    """`spatial_memory.build_spatial_memory_circuit`'s exact shift+damping
    step (via `apply_shift_step`, reused not reimplemented), with the
    ID-SQM transform applied once per timestep AFTER the shift/damping,
    before the readout snapshot."""
    mem_qubits = list(range(1, L)) if input_qubit == 0 else [q for q in range(L) if q != input_qubit]
    slots = [input_qubit] + mem_qubits
    labels, ops = memory_local_ops(mem_qubits)
    damping = amplitude_damping_kraus(gamma_M) if gamma_M > 0 else None

    qc = QuantumCircuit(L)
    for t, u_t in enumerate(u_seq):
        apply_shift_step(qc, slots, input_qubit, mem_qubits, u_t, damping)
        apply_idqnn_spatial_transform(qc, mem_qubits, block_size=block_size, seed=transform_seed,
                                       entangle_strength=entangle_strength)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f"{lab}__t{t}")
    return qc, labels, mem_qubits


def run_id_sqm(L: int, u_seq: Sequence[float], gamma_M: float = 0.0, block_size: int = 2,
               transform_seed: int = 0, entangle_strength: float = 0.3, method: str = "density_matrix",
               use_gpu: bool = False):
    if method != "density_matrix":
        raise ValueError("id_sqm circuits reset the input qubit every step -- use method='density_matrix' "
                          "(docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10).")
    qc, labels, mem_qubits = build_id_sqm_circuit(L, u_seq, gamma_M=gamma_M, block_size=block_size,
                                                   transform_seed=transform_seed,
                                                   entangle_strength=entangle_strength)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f"{lab}__t{t}"])
    return {"labels": labels, "X": X, "u": np.asarray(u_seq), "mem_qubits": mem_qubits,
            "circuit_depth": tqc.depth()}
