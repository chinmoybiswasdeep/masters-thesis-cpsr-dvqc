"""
directional_memory.py -- Protected quantum memory M for the directional
M -> A -> P architecture (Part 2). Two variants, BOTH thin wrappers around
already-verified mechanisms in this repo -- no new physics is reimplemented:

  A. Shift/SWAP memory: `spatial_memory.py`'s exact shift register,
     re-exported here under this architecture's own naming. Historical
     inputs persist ONLY in the quantum register (no `delay_taps`, no
     Python array of past inputs, no full-history re-encoding) -- already
     proven in `tests/test_spatial_memory.py` /
     `tests/test_no_classical_delay_leakage.py`.

  B. Protected/integrable memory: `H_M = sum_i omega_i Z_i + epsilon_M
     sum_i (X_i X_{i+1} + Y_i Y_{i+1})` -- EXACTLY `memory.py`'s existing
     `memory_mode='integrable'` branch (`memory.apply_memory_step`), reused
     verbatim with `omega_scale` renamed `epsilon_M` to match this
     architecture's own notation. `epsilon_M` small keeps the XY-hopping
     term weak, so the register's Z-content decays slowly (an
     approximately-conserved, not exactly-conserved, memory operator --
     the XY term does not commute with an individual site's Z, only the
     TOTAL magnetization is exactly conserved; "protected" here means
     "slow", not "exactly frozen", consistent with Part 2's own "or at
     least ... long autocorrelation time" phrasing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from . import memory as memmod
from . import spatial_memory as sqm
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, chrono_split, select_and_eval_ridge, delay_target  # noqa: E402

MEMORY_VARIANTS = ("shift", "protected_integrable")


@dataclass
class DirectionalMemoryRun:
    labels: list
    X: np.ndarray
    u: np.ndarray
    mem_qubits: list
    variant: str
    circuit_depth: int


def run_directional_memory(variant: str, N: int, u_seq: Sequence[float], epsilon_M: float = 0.05,
                            disorder_seed: int = 0, method: str = "density_matrix",
                            use_gpu: bool = False) -> DirectionalMemoryRun:
    """`N` = total qubits (1 input + N-1 memory qubits), matching every
    other memory builder in this repo. `variant='shift'` ignores
    `epsilon_M`/`disorder_seed` (exact shift register has no such
    parameters); `variant='protected_integrable'` uses them exactly as
    `memory.run_memory_register(memory_mode='integrable')` would."""
    if variant not in MEMORY_VARIANTS:
        raise ValueError(f"variant must be one of {MEMORY_VARIANTS}, got {variant!r}")
    if variant == "shift":
        run = sqm.run_spatial_memory(L=N, u_seq=u_seq, gamma_M=0.0, method=method, use_gpu=use_gpu)
        return DirectionalMemoryRun(labels=run.labels, X=run.X, u=run.u, mem_qubits=run.mem_qubits,
                                     variant=variant, circuit_depth=run.circuit_depth)
    run = memmod.run_memory_register(N=N, u_seq=u_seq, memory_mode="integrable", lambda_im=epsilon_M,
                                      omega_scale=epsilon_M, disorder_seed=disorder_seed, method=method,
                                      use_gpu=use_gpu)
    return DirectionalMemoryRun(labels=run.labels, X=run.X, u=run.u, mem_qubits=run.mem_qubits,
                                 variant=variant, circuit_depth=run.circuit_depth)


def delay_resolved_capacity(run: DirectionalMemoryRun, k_max: int, washout: int, n_val: int, n_test: int,
                             alphas=None):
    """Identical construction to `memory.delay_resolved_capacity` /
    `spatial_memory.delay_resolved_capacity` -- kept as a thin local
    function only because it operates on THIS module's own dataclass."""
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
    C = np.asarray(C)
    below = C < threshold
    for i in range(len(below)):
        if np.all(below[i:]):
            return float(k_values[i])
    return float(k_values[-1]) + 1.0


# =============================================================================
# Operator autocorrelation (Part 3/21) -- infinite-temperature Heisenberg-
# picture autocorrelation of a single memory qubit's Z operator under the
# memory register's OWN internal dynamics (no input encoding), a task-
# independent complement to the C(k) task-based lifetime above.
# =============================================================================

def _internal_layer_unitary(n_mem: int, epsilon_M: float, omega_seed: int) -> np.ndarray:
    """The `memory_mode='integrable'` INTERNAL layer only (no input qubit,
    no reset) -- built directly from the same edges/omega convention
    `memory.apply_memory_step` uses, so the operator being autocorrelated
    is exactly the one the real memory register evolves under."""
    omega = memmod.sample_memory_disorder(n_mem, omega_seed, scale=2 * np.pi)
    qc = QuantumCircuit(n_mem)
    for q, w in zip(range(n_mem), omega):
        qc.rz(w, q)
    for a, b in zip(range(n_mem - 1), range(1, n_mem)):
        qc.rxx(2.0 * epsilon_M, a, b)
        qc.ryy(2.0 * epsilon_M, a, b)
    return Operator(qc).data


def z_autocorrelation(n_mem: int, epsilon_M: float, omega_seed: int, n_steps: int, qubit: int = 0):
    """C_ZZ(t) = Tr[Z_qubit(t) Z_qubit(0)] / dim, t=0..n_steps, infinite-
    temperature operator autocorrelation under repeated application of the
    memory register's own internal layer (no external input -- this
    isolates the memory SECTOR's intrinsic dynamics, per Part 21). C_ZZ(0)=1
    by construction; a slowly-decaying curve is the signature of a
    'protected'/slow operator sector."""
    U = _internal_layer_unitary(n_mem, epsilon_M, omega_seed)
    dim = 2 ** n_mem
    Z = np.array([[1.0]])
    for i in range(n_mem - 1, -1, -1):
        Z = np.kron(Z, np.array([[1, 0], [0, -1]]) if i == qubit else np.eye(2))
    Z0 = Z.astype(np.complex128)
    Zt = Z0.copy()
    corr = [1.0]
    for _ in range(n_steps):
        Zt = U.conj().T @ Zt @ U
        corr.append(float(np.real(np.trace(Zt @ Z0))) / dim / (np.real(np.trace(Z0 @ Z0)) / dim))
    return list(range(n_steps + 1)), corr
