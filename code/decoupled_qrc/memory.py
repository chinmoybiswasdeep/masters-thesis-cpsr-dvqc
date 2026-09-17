"""
memory.py -- the quantum memory subsystem M (Part 2). Two genuinely distinct
constructions, per the audit's central finding (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md
section 5): the repo's existing SCIENCE_CONFIG QELM encoding stores memory in
a classical Python array (a re-encoded sliding window), not in the quantum
state -- DQRC's memory subsystem must NOT repeat that. Every construction
here keeps the memory qubits UNRESET across the whole trajectory; only the
designated input qubit q0 is reset+Ry(pi*u_t)-encoded each step, exactly
`qrc_qiskit.py`'s own comment-1 fading-memory convention, extended to a
multi-qubit memory register with tunable internal dynamics.

    A. PersistentMemoryRegister  (`build_persistent_memory_circuit`)
       q0 (input, reset each step) <-coupling-> M1..M_{Nm} (never reset,
       internal dynamics selectable via `memory_mode`).

    B. QuantumDelayRegister      (`build_quantum_delay_circuit`)
       An exact quantum shift register realized with a SWAP network:
       M0 <- encoded u_t, M1 <- old M0, M2 <- old M1, ... each step.

CRITICAL: both run under method='density_matrix' ONLY -- one qubit is reset
every step of one long circuit, exactly the pattern the confirmed Aer
statevector-reset bug corrupts (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md
section 10 / memory `aer-statevector-reset-bug`). `run_memory_register`
refuses `method='statevector'` outright rather than silently producing wrong
numbers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli

from .utils import ensure_repo_code_on_path, ResourceUsage

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, delay_target, chrono_split, select_and_eval_ridge, nrmse  # noqa: E402

MEMORY_MODES = ("swap", "integrable", "weak_xx")


def memory_feature_ops(n_qubits: int, qubits: Sequence[int]):
    """Single-qubit X/Y/Z on the given qubits -- sufficient to fully
    reconstruct each qubit's Bloch vector, which is what a delay-resolved
    memory-capacity scan (Part 15 Section 4) needs; kept deliberately small
    (not the full weight<=3 Pauli-string basis `mixed_syk_core` uses for the
    processor, which is about NONLINEAR readout richness, not memory)."""
    ops, labels = [], []
    for q in qubits:
        for name in ("Z", "X", "Y"):
            ops.append((Pauli(name), [q]))
            labels.append(f"{name}{q}")
    return labels, ops


def sample_memory_disorder(n_mem: int, seed: int, scale: float):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, scale, n_mem)


# =============================================================================
# A. Persistent memory register
# =============================================================================

def build_persistent_memory_circuit(N: int, u_seq: Sequence[float], memory_mode: str = "integrable",
                                     input_qubit: int = 0, lambda_im: float = 0.3,
                                     epsilon: float = 0.05, omega_scale: float = 1.0,
                                     disorder_seed: int = 0):
    """`N` total qubits: `input_qubit` (default 0) is reset+Ry(pi*u_t) each
    step; the remaining N-1 qubits are the never-reset memory register.

    Each step, in order:
      1. reset(input_qubit); ry(pi*u_t, input_qubit)
      2. input-memory coupling: an RXX+RYY exchange of strength `lambda_im`
         between input_qubit and the first memory qubit (mode='swap' instead
         uses an EXACT SWAP -- perfect, not partial, transport)
      3. internal memory-register dynamics, mode-dependent:
         - 'swap':        on-site Z disorder (omega_scale) + a WEAK RXX+RYY
                           exchange (angle=epsilon) between adjacent memory
                           qubits, so information deposited at M1 slowly
                           diffuses down the chain (a soft, partial-transport
                           analogue of the literal SWAP-chain shift register
                           in build_quantum_delay_circuit).
         - 'integrable':  H_M = sum_i omega_i Z_i + omega_scale * sum_i
                           (X_i X_{i+1} + Y_i Y_{i+1}) -- a free-fermion XY
                           hopping chain (exactly solvable via Jordan-Wigner,
                           hence genuinely non-chaotic), realized as one
                           Trotter step (RZ + RXX + RYY) per timestep.
         - 'weak_xx':     H_M = sum_i omega_i Z_i + epsilon * sum_i X_i
                           X_{i+1}, epsilon small -- the task spec's own
                           literal example Hamiltonian: strong on-site Z
                           precession/dephasing plus a WEAK XX-only coupling
                           (no YY), so dynamics stay close to diagonal in the
                           Z basis and mix much more slowly than 'integrable'.
      4. snapshot memory_feature_ops(memory qubits) exact expectation values.
    """
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {MEMORY_MODES}, got {memory_mode!r}")
    mem_qubits = [q for q in range(N) if q != input_qubit]
    n_mem = len(mem_qubits)
    omega = sample_memory_disorder(n_mem, disorder_seed, scale=(2 * np.pi if memory_mode != "swap" else omega_scale))
    labels, ops = memory_feature_ops(N, mem_qubits)

    qc = QuantumCircuit(N)
    for t, u_t in enumerate(u_seq):
        apply_memory_step(qc, input_qubit, mem_qubits, memory_mode, lambda_im, epsilon, omega_scale, omega, u_t)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f"{lab}__t{t}")
    return qc, labels, mem_qubits


def apply_memory_step(qc: QuantumCircuit, input_qubit: int, mem_qubits: Sequence[int], memory_mode: str,
                       lambda_im: float, epsilon: float, omega_scale: float, omega: np.ndarray, u_t: float):
    """One timestep's worth of gates (reset+encode input, input->memory
    coupling, internal memory-register dynamics) applied IN PLACE to `qc` at
    GLOBAL qubit indices `input_qubit`/`mem_qubits` -- factored out of
    `build_persistent_memory_circuit` so `experiments.py` can interleave this
    exact same memory-subsystem step with a processor step and an interface
    coupling inside one combined DQRC trajectory circuit, without
    reimplementing the memory dynamics."""
    edges = list(zip(mem_qubits[:-1], mem_qubits[1:]))
    qc.reset(input_qubit)
    qc.ry(np.pi * float(u_t), input_qubit)

    first_mem = mem_qubits[0]
    if memory_mode == "swap":
        qc.swap(input_qubit, first_mem)
    else:
        qc.rxx(2.0 * lambda_im, input_qubit, first_mem)
        qc.ryy(2.0 * lambda_im, input_qubit, first_mem)

    if memory_mode == "swap":
        for i, (a, b) in enumerate(edges):
            qc.rz(omega[i], a)
            qc.rxx(2.0 * epsilon, a, b)
            qc.ryy(2.0 * epsilon, a, b)
        qc.rz(omega[-1], mem_qubits[-1])
    elif memory_mode == "integrable":
        for q, w in zip(mem_qubits, omega):
            qc.rz(w, q)
        for a, b in edges:
            qc.rxx(2.0 * omega_scale, a, b)
            qc.ryy(2.0 * omega_scale, a, b)
    elif memory_mode == "weak_xx":
        for q, w in zip(mem_qubits, omega):
            qc.rz(w, q)
        for a, b in edges:
            qc.rxx(2.0 * epsilon, a, b)


# =============================================================================
# B. Quantum delay register -- exact SWAP-network shift register
# =============================================================================

def build_quantum_delay_circuit(N: int, u_seq: Sequence[float], input_qubit: int = 0):
    """An exact quantum shift register: each step, shift the WHOLE register
    one slot toward higher index via a chain of SWAPs (executed
    high-to-low so no slot's contents are clobbered before being moved),
    THEN reset+encode the input slot. After step t: slot 0 holds u_t, slot 1
    holds u_{t-1}, ..., slot N-1 holds u_{t-(N-1)} -- M_k <- old M_{k-1} for
    every k, realized entirely in the quantum register (no classical array
    of past inputs anywhere in this function), matching Part 2B exactly.
    """
    mem_qubits = [q for q in range(N) if q != input_qubit]
    labels, ops = memory_feature_ops(N, mem_qubits)
    slots = [input_qubit] + mem_qubits  # slot order: 0=most recent

    qc = QuantumCircuit(N)
    for t, u_t in enumerate(u_seq):
        for a, b in zip(slots[-2::-1], slots[-1:0:-1]):
            qc.swap(a, b)
        qc.reset(input_qubit)
        qc.ry(np.pi * float(u_t), input_qubit)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f"{lab}__t{t}")
    return qc, labels, mem_qubits


# =============================================================================
# Execution
# =============================================================================

@dataclass
class MemoryRun:
    labels: list
    X: np.ndarray
    u: np.ndarray
    mem_qubits: list
    architecture: str
    circuit_depth: int


def _execute(qc: QuantumCircuit, labels, T: int, method: str, use_gpu: bool):
    if method != "density_matrix":
        raise ValueError(
            f"memory.py circuits reset one qubit every step of one long trajectory circuit -- "
            f"the confirmed Aer bug (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10) makes "
            f"method={method!r} give silently wrong results for this exact pattern. Use "
            f"method='density_matrix'.")
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f"{lab}__t{t}"])
    return X, tqc.depth()


def run_memory_register(N: int, u_seq: Sequence[float], memory_mode: str = "integrable",
                         method: str = "density_matrix", use_gpu: bool = False, **kwargs) -> MemoryRun:
    qc, labels, mem_qubits = build_persistent_memory_circuit(N, u_seq, memory_mode=memory_mode, **kwargs)
    X, depth = _execute(qc, labels, len(u_seq), method, use_gpu)
    return MemoryRun(labels=labels, X=X, u=np.asarray(u_seq), mem_qubits=mem_qubits,
                      architecture=f"persistent_memory[{memory_mode}]", circuit_depth=depth)


def run_quantum_delay_register(N: int, u_seq: Sequence[float], method: str = "density_matrix",
                                use_gpu: bool = False, input_qubit: int = 0) -> MemoryRun:
    qc, labels, mem_qubits = build_quantum_delay_circuit(N, u_seq, input_qubit=input_qubit)
    X, depth = _execute(qc, labels, len(u_seq), method, use_gpu)
    return MemoryRun(labels=labels, X=X, u=np.asarray(u_seq), mem_qubits=mem_qubits,
                      architecture="quantum_delay_register", circuit_depth=depth)


# =============================================================================
# Delay-resolved memory capacity C(k) and memory lifetime (Part 15 Section 4)
# =============================================================================

def delay_resolved_capacity(run: MemoryRun, k_max: int, washout: int, n_val: int, n_test: int,
                             alphas=None):
    """C(k) for k=1..k_max: squared correlation between u_{t-k} and its ridge
    reconstruction from this memory run's features (same construction as
    `qrc_qiskit.memory_capacity`'s per-lag score). Returns (k_values, C).
    """
    from qrc_qiskit import DEFAULT_ALPHAS
    alphas = alphas or DEFAULT_ALPHAS
    T = len(run.u)
    gap = k_max + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)
    C = []
    for k in range(1, k_max + 1):
        y = delay_target(run.u, k)
        err, _, model = select_and_eval_ridge(run.X, y, train, val, test, alphas=alphas)
        pred = model.predict(run.X[test])
        cov = np.cov(pred, y[test])[0, 1] ** 2
        denom = np.var(y[test]) * np.var(pred) + 1e-12
        C.append(float(np.clip(cov / denom, 0.0, 1.0)))
    return list(range(1, k_max + 1)), C


def memory_lifetime(k_values, C, threshold: float = 0.1) -> float:
    """First delay k at which C(k) drops below `threshold` and stays below it
    for the rest of the scan (a simple, honest operational definition -- not
    a fitted exponential decay constant, which would need more delay points
    than a FAST_MODE scan provides to fit reliably)."""
    C = np.asarray(C)
    below = C < threshold
    for i in range(len(below)):
        if np.all(below[i:]):
            return float(k_values[i])
    return float(k_values[-1]) + 1.0  # never dropped below threshold within the scan
