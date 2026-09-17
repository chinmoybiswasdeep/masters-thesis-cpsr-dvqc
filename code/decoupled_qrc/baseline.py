"""
baseline.py -- the monolithic-QRC control, built by WRAPPING (not
reimplementing) the finalized `mixed_syk_core.py`/`qrc_qiskit.py` reservoir,
per docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 9. This is the reservoir
DQRC must expand the memory-vs-nonlinearity Pareto frontier relative to.

Also provides the classical_delay_qrc positive control (Part 2's three
explicit controls: monolithic_qrc, classical_delay_qrc, and -- in memory.py
-- true_quantum_memory_dqrc). `classical_delay_qrc` is `qrc_qiskit.delay_taps`
alone: NEVER concatenated into quantum features (the audit's finding #5/#7 --
that concatenation pattern exists in the OLDER idcpsr.py/idcpsr_qiskit.py
lineage and is deliberately not reused here).

CRITICAL: this module always runs with method='density_matrix' for the
recurrent architecture, per the confirmed Aer statevector-reset bug (audit
finding #10) -- the recurrent trajectory circuit here resets one qubit every
step, exactly the pattern that bug corrupts under method='statevector'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import ensure_repo_code_on_path, ResourceUsage

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402
from qrc_qiskit import ReservoirConfig, delay_taps, random_input  # noqa: E402


@dataclass
class ReservoirRun:
    labels: list
    X: np.ndarray
    u: np.ndarray
    resources: ResourceUsage
    info: dict


def monolithic_qrc(N: int, kappa: float, T: int, reps: int = 2, max_weight: int = 3,
                    term_seed: int = 0, disorder_seed: int = 42, input_seed: int = 0,
                    G_MAX: float = 0.6, J_MAX: float = 0.6, method: str = "density_matrix",
                    use_gpu: bool = False) -> ReservoirRun:
    """The monolithic baseline: ONE mixed-SYK register, no memory/processor
    separation, genuinely recurrent (only the input qubit resets each step --
    `mixed_syk_core.build_trajectory_circuit_mixed`, the SAME recurrent
    pattern `qrc_qiskit.py`'s own Section 1 established, generalized to the
    mixed g/J channel). `kappa` sweeps SYK4-like <-> SYK2-like exactly as in
    the finalized notebook 4 (`kappa_to_gJ`) -- this is the single scan
    parameter whose trade-off DQRC must beat.
    """
    g, J = msc.kappa_to_gJ(kappa, G_MAX, J_MAX)
    cfg = ReservoirConfig(N=N, g=0.0, reps=reps, input_qubit=0, seed=disorder_seed)
    u = random_input(T, seed=input_seed)
    labels, X, info, (terms, couplings, paulis) = msc.run_reservoir_mixed(
        cfg, u, g=g, J=J, reps=reps, term_seed=term_seed, method=method,
        use_gpu=use_gpu, max_weight=max_weight,
    )
    resources = ResourceUsage(
        n_qubits_physical=N, n_ancilla=0, circuit_depth=info["circuit_depth"],
        two_qubit_gates=_estimate_two_qubit_gates(N, reps, len(terms), T),
        shots=0, n_features=len(labels),
        notes=f"monolithic_qrc kappa={kappa} g={g:.4f} J={J:.4f} n_terms={len(terms)}",
    )
    return ReservoirRun(labels=labels, X=X, u=u, resources=resources, info=info)


def monolithic_qrc_at_eoc(N: int, T: int, science_kappa: float = 0.960, **kwargs) -> ReservoirRun:
    """`monolithic_qrc` at notebook 4's own established EOC point (kappa=0.960,
    loaded from `eoc_config`, never re-derived -- audit finding #3/#6:
    'kappa=0.960 was selected by NARMA2 performance on notebook-4-internal
    data', so this is a reference/starting point, not re-tuned leakage)."""
    return monolithic_qrc(N=N, kappa=science_kappa, T=T, **kwargs)


def classical_delay_qrc(T: int, m: int, input_seed: int = 0) -> ReservoirRun:
    """Classical-only positive control: raw delay taps, zero quantum
    processing (`qrc_qiskit.delay_taps`). Used to check whether any DQRC/
    baseline advantage is just 'the delayed input alone already explains
    this' (Part 2's mandated third control, `qrc_qiskit.py:448-456`'s own
    documented purpose)."""
    u = random_input(T, seed=input_seed)
    X = delay_taps(u, m=m)
    labels = [f"delay_{j}" for j in range(m + 1)]
    resources = ResourceUsage(n_qubits_physical=0, n_ancilla=0, circuit_depth=0,
                               two_qubit_gates=0, shots=0, n_features=X.shape[1],
                               notes="classical_delay_qrc: zero quantum resources, by construction")
    return ReservoirRun(labels=labels, X=X, u=u, resources=resources,
                         info={"architecture": "classical_delay_control"})


def _estimate_two_qubit_gates(N: int, reps: int, n_terms: int, T: int) -> int:
    """Rough per-trajectory 2Q-gate count for `mixed_layer`: N-1 RXX + N-1 RYY
    per rep (chain), + 6 CNOTs per SYK4 term (the `zzzz_rotation` gadget,
    `mixed_syk_core.zzzz_rotation`), times reps, times T timesteps. Used for
    honest resource reporting (Part 9), not for circuit-depth-critical
    decisions -- the transpiled `circuit_depth` from Aer is the authoritative
    depth number."""
    per_layer = 2 * (N - 1) + 6 * n_terms
    return per_layer * reps * T
