"""
spatial_memory.py -- Spatial Quantum Memory (SQM): a genuine quantum
shift register, M0..M_{L-1}, with an OPTIONAL fading-memory control
`gamma_M` (Part 4/5 of the IDQNN-inspired-memory brief).

TERMINOLOGY (Part 1): this module is called "Spatial Quantum Memory", never
"IDQNN memory" -- it is a plain shift register plus an optional decoherence
channel, not Huang et al.'s shallow-wide/deep-circuit-equivalence
construction. The IDQNN-INSPIRED spatial transform that sits on top of this
register lives in `idqnn_memory.py`, and is likewise never called "IDQNN"
unless it is numerically verified against a deep-circuit target (which it
is not, by construction -- see that module's own docstring).

NO CLASSICAL DELAY LEAKAGE (Part 4/23 item 1): every past input lives ONLY
in the quantum register's own reduced density matrix. `u_seq` is consumed
strictly causally (each `u_t` is used only when it is encoded, never stored
in a Python array and re-read later); the register itself, not a side
buffer, is what's read out for features. `tests/test_spatial_memory.py`
verifies this directly against an independent from-scratch reference (same
pattern as `memory.py`'s own
`test_quantum_delay_register_is_exact_shift_no_classical_array`).

Reuses the exact SWAP-network shift mechanism already proven correct in
`memory.build_quantum_delay_circuit` (kept there unmodified, since that
function's own regression tests -- `tests/test_memory.py` -- already
protect its zero-decay behavior); this module adds the configurable
register length `L` and the fading channel on top, rather than
reimplementing the shift itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, Pauli

from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, chrono_split, select_and_eval_ridge, delay_target  # noqa: E402


def amplitude_damping_kraus(gamma: float) -> Kraus:
    """Standard single-qubit amplitude-damping channel:
    K0 = diag(1, sqrt(1-gamma)), K1 = [[0, sqrt(gamma)],[0,0]] -- population
    decays toward |0> at rate `gamma` per application. `gamma=0` is the
    identity channel exactly."""
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0 - gamma))]])
    K1 = np.array([[0.0, np.sqrt(max(0.0, gamma))], [0.0, 0.0]])
    return Kraus([K0, K1])


def memory_local_ops(mem_qubits: Sequence[int], paulis=("Z", "X", "Y")):
    ops, labels = [], []
    for q in mem_qubits:
        for p in paulis:
            ops.append((Pauli(p), [q]))
            labels.append(f"{p}{q}")
    return labels, ops


def build_spatial_memory_circuit(L: int, u_seq: Sequence[float], gamma_M: float = 0.0,
                                  input_qubit: int = 0):
    """A depth-L quantum shift register (`L` TOTAL slots including the input
    slot, so `L-1` genuine memory qubits M1..M_{L-1}): each step, shift the
    whole register one slot toward higher index via a SWAP chain (identical
    mechanism to `memory.build_quantum_delay_circuit`), then reset+encode
    the input slot, THEN (if `gamma_M>0`) apply one amplitude-damping
    application to every memory qubit -- so a value that has sat in the
    register for k steps has undergone k damping applications, giving an
    approximately exponential-in-k fading profile. `gamma_M=0` reduces
    EXACTLY to the hard (non-fading) shift register.
    """
    if L < 2:
        raise ValueError("L must be >= 2 (1 input slot + >=1 memory slot)")
    mem_qubits = list(range(1, L)) if input_qubit == 0 else [q for q in range(L) if q != input_qubit]
    slots = [input_qubit] + mem_qubits
    labels, ops = memory_local_ops(mem_qubits)
    damping = amplitude_damping_kraus(gamma_M) if gamma_M > 0 else None

    qc = QuantumCircuit(L)
    for t, u_t in enumerate(u_seq):
        apply_shift_step(qc, slots, input_qubit, mem_qubits, u_t, damping)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f"{lab}__t{t}")
    return qc, labels, mem_qubits


def apply_shift_step(qc: QuantumCircuit, slots: Sequence[int], input_qubit: int, mem_qubits: Sequence[int],
                      u_t: float, damping: Kraus = None):
    """One timestep's shift+encode(+damping) applied IN PLACE at GLOBAL
    qubit indices -- factored out of `build_spatial_memory_circuit` so
    `idqnn_memory.py` can interleave its own shallow-wide spatial transform
    with this exact same shift mechanism, without reimplementing it (the
    same "reusable step function" pattern `memory.apply_memory_step`
    established)."""
    for a, b in zip(slots[-2::-1], slots[-1:0:-1]):
        qc.swap(a, b)
    qc.reset(input_qubit)
    qc.ry(np.pi * float(u_t), input_qubit)
    if damping is not None:
        for q in mem_qubits:
            qc.append(damping, [q])


@dataclass
class SpatialMemoryRun:
    labels: list
    X: np.ndarray
    u: np.ndarray
    mem_qubits: list
    gamma_M: float
    circuit_depth: int


def run_spatial_memory(L: int, u_seq: Sequence[float], gamma_M: float = 0.0,
                        method: str = "density_matrix", use_gpu: bool = False) -> SpatialMemoryRun:
    if method != "density_matrix":
        raise ValueError("spatial_memory circuits reset the input qubit every step of one long "
                          "trajectory circuit -- the confirmed Aer statevector-reset bug's exact "
                          "trigger pattern (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10). Use "
                          "method='density_matrix'.")
    qc, labels, mem_qubits = build_spatial_memory_circuit(L, u_seq, gamma_M=gamma_M)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f"{lab}__t{t}"])
    return SpatialMemoryRun(labels=labels, X=X, u=np.asarray(u_seq), mem_qubits=mem_qubits,
                             gamma_M=gamma_M, circuit_depth=tqc.depth())


def delay_resolved_capacity(run: SpatialMemoryRun, k_max: int, washout: int, n_val: int, n_test: int,
                             alphas=None, n_shots: int = None, noise_seed: int = 0):
    """C(k) for k=1..k_max -- identical construction to
    `memory.delay_resolved_capacity`, duplicated here (not imported) only
    because it operates on this module's own `SpatialMemoryRun` dataclass;
    the underlying ridge/correlation logic is the same one-liner either way,
    not a second implementation of anything nontrivial.

    **Important finding (kept here, not buried in a report)**: with
    `n_shots=None` (the exact, noiseless expectation-value readout every
    other module in this repo uses), amplitude damping does NOT reduce C(k)
    -- a fixed-gamma damping channel applied k times is still an INVERTIBLE,
    deterministic, monotonic function of u_{t-k}, and ridge-regression
    capacity is invariant to any invertible rescaling of its target-
    correlated feature. Real "fading memory" in the everyday sense (older
    inputs become genuinely UNRECOVERABLE, not just rescaled) requires
    either (a) the damping to actually destroy rank/invertibility, or (b)
    finite measurement precision so a shrinking signal amplitude eventually
    drops below the noise floor. This function implements (b): passing
    `n_shots` adds i.i.d. Gaussian noise with the standard projective-
    measurement standard error, sqrt((1-<O>^2)/n_shots), to each feature
    BEFORE fitting -- a synthetic approximation of shot noise (not a full
    circuit-level shot simulation), sufficient to demonstrate the intended
    effect. `n_shots=None` reproduces the exact (gamma-invariant) result;
    see `tests/test_spatial_memory.py` for both regimes checked directly.
    """
    from qrc_qiskit import DEFAULT_ALPHAS
    alphas = alphas or DEFAULT_ALPHAS
    T = len(run.u)
    gap = k_max + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)

    X = run.X
    if n_shots is not None:
        rng = np.random.RandomState(noise_seed)
        se = np.sqrt(np.clip(1.0 - X ** 2, 0.0, 1.0) / n_shots)
        X = X + rng.normal(0.0, 1.0, size=X.shape) * se

    C = []
    for k in range(1, k_max + 1):
        y = delay_target(run.u, k)
        err, _, model = select_and_eval_ridge(X, y, train, val, test, alphas=alphas)
        pred = model.predict(X[test])
        cov = np.cov(pred, y[test])[0, 1] ** 2
        denom = np.var(y[test]) * np.var(pred) + 1e-12
        C.append(float(np.clip(cov / denom, 0.0, 1.0)))
    return list(range(1, k_max + 1)), C
