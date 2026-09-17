"""
id_memory_eoc.py -- combined orchestration: Spatial Quantum Memory (SQM,
optionally with the IDQNN-inspired ID-SQM transform) + an advanced M->P
interface + the EOC processor, in ONE trajectory circuit, with explicit
X_memory / X_processor / X_cross feature groups (Part 14) and the primary
success metrics eta_M / eta_NL (Part 11).

Ties together: spatial_memory.py / idqnn_memory.py (memory side),
interfaces_advanced.py (coupling), processor.py (unchanged, reused
verbatim). Mirrors `experiments.build_dqrc_circuit`'s structure but with the
shift-register memory mechanism instead of the persistent-register one, and
the richer interface set instead of just `interface.py`'s basic three.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from . import processor as procmod
from . import interface as ifacemod
from . import interfaces_advanced as advmod
from .spatial_memory import apply_shift_step, amplitude_damping_kraus
from .idqnn_memory import apply_idqnn_spatial_transform
from .diagnostics import memory_local_ops, processor_local_ops, cross_ops, ipc_MN
from .utils import ensure_repo_code_on_path, ResourceUsage, make_seed_bundle

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input, delay_taps  # noqa: E402


@dataclass
class IDMemoryEOCConfig:
    L: int = 4                       # total memory-side register (1 input + L-1 memory qubits)
    use_idqnn_transform: bool = False
    block_size: int = 2
    transform_seed: int = 0
    entangle_strength: float = 0.3
    gamma_M: float = 0.0
    N_P: int = 5
    kappa_processor: float = 1.0
    reps_processor: int = 1
    interface_kind: str = "xy"       # 'rzz'/'cp'/'zx' (interface.py) or 'xy'/'heisenberg'/'multiaxis' (interfaces_advanced.py)
    lambda_mp: float = 0.3
    eta_heisenberg: float = 0.3
    n_taps: int = 1
    max_weight_proc: int = 3
    max_weight_mem: int = 2
    G_MAX: float = 0.6
    J_MAX: float = 0.6

    @property
    def n_mem_qubits(self) -> int:
        return self.L - 1

    @property
    def n_qubits_total(self) -> int:
        return self.L + self.N_P


@dataclass
class IDMemoryEOCRun:
    u: np.ndarray
    labels_mem: list
    labels_proc: list
    labels_cross: list
    X_mem: np.ndarray
    X_proc: np.ndarray
    X_cross: np.ndarray
    mem_qubits: list
    proc_qubits: list
    resources: ResourceUsage

    @property
    def X_combined(self) -> np.ndarray:
        return np.hstack([self.X_mem, self.X_proc, self.X_cross])


def build_id_memory_eoc_circuit(cfg: IDMemoryEOCConfig, u_seq: Sequence[float], seeds):
    mem_qubits = list(range(1, cfg.L))
    proc_qubits = list(range(cfg.L, cfg.L + cfg.N_P))
    slots = [0] + mem_qubits
    N_total = cfg.n_qubits_total
    damping = amplitude_damping_kraus(cfg.gamma_M) if cfg.gamma_M > 0 else None

    proc_params = procmod.sample_processor_params(
        cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
        disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor, G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
    proc_layer = procmod.processor_reps_subcircuit(proc_params)

    n_taps_eff = min(cfg.n_taps, cfg.n_mem_qubits, cfg.N_P)
    memory_taps = mem_qubits[-n_taps_eff:]
    processor_entry = proc_qubits[:n_taps_eff]

    labels_mem, ops_mem = memory_local_ops(mem_qubits, max_weight=cfg.max_weight_mem)
    labels_proc, ops_proc = processor_local_ops(proc_qubits, max_weight=cfg.max_weight_proc)
    labels_cross, ops_cross = cross_ops(memory_taps, processor_entry)

    qc = QuantumCircuit(N_total)
    for t, u_t in enumerate(u_seq):
        apply_shift_step(qc, slots, 0, mem_qubits, u_t, damping)
        if cfg.use_idqnn_transform:
            apply_idqnn_spatial_transform(qc, mem_qubits, block_size=cfg.block_size,
                                           seed=cfg.transform_seed, entangle_strength=cfg.entangle_strength)
        if cfg.lambda_mp != 0:
            if cfg.interface_kind in ifacemod.INTERFACE_KINDS:
                ifacemod.apply_interface(qc, memory_taps, processor_entry, cfg.lambda_mp, cfg.interface_kind)
            else:
                advmod.apply_advanced_interface(qc, memory_taps, processor_entry, cfg.lambda_mp,
                                                 kind=cfg.interface_kind, eta=cfg.eta_heisenberg)
        qc.compose(proc_layer, qubits=proc_qubits, inplace=True)
        for (op, qargs), lab in zip(ops_mem, labels_mem):
            qc.save_expectation_value(op, qargs, label=f"mem_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(op, qargs, label=f"proc_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_cross, labels_cross):
            qc.save_expectation_value(op, qargs, label=f"cross_{lab}__t{t}")

    return qc, (labels_mem, labels_proc, labels_cross), mem_qubits, proc_qubits


def run_id_memory_eoc(cfg: IDMemoryEOCConfig, T: int, master_seed: int = 0,
                       method: str = "density_matrix", use_gpu: bool = False) -> IDMemoryEOCRun:
    if method != "density_matrix":
        raise ValueError("id_memory_eoc circuits reset a qubit every step -- use method='density_matrix' "
                          "(docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10).")
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    qc, (labels_mem, labels_proc, labels_cross), mem_qubits, proc_qubits = build_id_memory_eoc_circuit(cfg, u, seeds)

    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)

    X_mem = np.array([[np.real(data[f"mem_{lab}__t{t}"]) for lab in labels_mem] for t in range(T)])
    X_proc = np.array([[np.real(data[f"proc_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])
    X_cross = np.array([[np.real(data[f"cross_{lab}__t{t}"]) for lab in labels_cross] for t in range(T)])

    resources = ResourceUsage(n_qubits_physical=cfg.n_qubits_total, n_ancilla=0, circuit_depth=tqc.depth(),
                               two_qubit_gates=0, shots=0,
                               n_features=len(labels_mem) + len(labels_proc) + len(labels_cross),
                               notes=f"id_memory_eoc L={cfg.L} N_P={cfg.N_P} interface={cfg.interface_kind} "
                                     f"lambda_mp={cfg.lambda_mp} idqnn_transform={cfg.use_idqnn_transform}")
    return IDMemoryEOCRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, labels_cross=labels_cross,
                           X_mem=X_mem, X_proc=X_proc, X_cross=X_cross, mem_qubits=mem_qubits,
                           proc_qubits=proc_qubits, resources=resources)


# =============================================================================
# Part 11 -- primary success metrics eta_M / eta_NL
# =============================================================================

def compute_etas(M_combined: float, NL_combined: float, M_memory_standalone: float,
                  NL_processor_standalone: float) -> dict:
    return {
        "eta_M": M_combined / M_memory_standalone if M_memory_standalone > 0 else float("nan"),
        "eta_NL": NL_combined / NL_processor_standalone if NL_processor_standalone > 0 else float("nan"),
        "M_combined": M_combined, "NL_combined": NL_combined,
        "M_memory_standalone": M_memory_standalone, "NL_processor_standalone": NL_processor_standalone,
    }


# =============================================================================
# Part 15 -- classical-delay-line + EOC-processor control
# =============================================================================

def classical_delay_plus_processor(m_delay: int, N_P: int, kappa_processor: float, T: int,
                                    reps: int = 1, max_weight: int = 3, master_seed: int = 0,
                                    G_MAX: float = 0.6, J_MAX: float = 0.6):
    """Part 15 item 6's essential control: classical `delay_taps(u, m_delay)`
    (zero quantum memory) CONCATENATED with a standalone EOC processor's own
    features (the processor sees only the instantaneous u_t, no real
    memory of its own beyond `reps` steps of internal dynamics) -- tests
    whether real quantum spatial memory earns its keep over 'classical
    memory + quantum nonlinearity'."""
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    X_classical = delay_taps(u, m=m_delay)
    labels, X_proc, u_proc, info = procmod.run_processor_standalone(
        N_p=N_P, kappa_processor=kappa_processor, T=T, reps=reps, max_weight=max_weight,
        term_seed=seeds.reservoir_seed, disorder_seed=seeds.reservoir_seed + 1, input_seed=seeds.dataset_seed)
    assert np.array_equal(u, u_proc), "classical and quantum sides must share the same input trajectory"
    X = np.hstack([X_classical, X_proc])
    resources = ResourceUsage(n_qubits_physical=N_P, n_ancilla=0, circuit_depth=info["circuit_depth"],
                               two_qubit_gates=0, shots=0, n_features=X.shape[1],
                               notes="classical_delay_plus_processor control")
    return {"u": u, "X": X, "resources": resources}
