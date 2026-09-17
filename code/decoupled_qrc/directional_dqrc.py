"""
directional_dqrc.py -- the primary directional architecture (Part 8):

    u_t -> protected memory M -> [collision: M -> fresh ancilla A -> P] -> EOC processor P -> features -> Ridge

Ties together `directional_memory.py` (memory step), `collision_interface.py`
(the M->A->P collision channel, fresh ancilla reset every step),
`directional_processor.py`/`processor.py` (EOC processor, reused verbatim),
and `diagnostics.py`'s feature-group helpers (`memory_local_ops`,
`processor_local_ops`, `cross_ops`, reused, not reimplemented).

Also exposes the two REFERENCE direct-coupling configurations from the
prior `id_memory_eoc.py` pass (`'zx'` and `'heisenberg'`), so Part 15's
"collision channel vs. direct coupling" comparison can be run with a single
consistent API, without duplicating `id_memory_eoc.py`'s own circuit code
(it is called directly for those two reference cases).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from . import directional_memory as dmem
from . import directional_processor as dproc
from .collision_interface import apply_collision_step, AP_KINDS
from .diagnostics import memory_local_ops, processor_local_ops, cross_ops
from .utils import ensure_repo_code_on_path, ResourceUsage, make_seed_bundle

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input  # noqa: E402


@dataclass
class DirectionalConfig:
    memory_variant: str = "shift"       # 'shift' or 'protected_integrable'
    N_M: int = 2                        # TOTAL memory-side qubits (1 input-receiving qubit + (N_M-1) storage
                                         # qubits) -- matches Part 9's resource accounting (N_M+N_A+N_P=N_total)
                                         # exactly; NOT "N_M storage qubits plus a separate input qubit".
    epsilon_M: float = 0.5              # only used by 'protected_integrable'
    N_P: int = 5
    kappa_processor: float = 1.0
    g_processor: float = None       # if set (together with J_processor), used INSTEAD of
    J_processor: float = None       # kappa_processor -- independent (g,J), docs/DQRC_GJ_EOC_AUDIT.md Part 2
    reps_processor: int = 1
    theta: float = 0.3                  # M -> A QND strength
    ma_kind: str = "zx"                 # 'zx' (transfers) or 'zz' (Part 4's literal default -- does not, see collision_interface.py)
    phi: float = 0.5                    # A -> P transfer strength
    ap_kind: str = "xy"                 # 'xx' / 'xy' / 'anisotropic'
    phi_x: float = 1.0
    phi_y: float = 1.0
    phi_z: float = 0.0
    max_weight_proc: int = 3
    max_weight_mem: int = 2
    G_MAX: float = 0.6
    J_MAX: float = 0.6

    @property
    def n_qubits_total(self) -> int:
        return self.N_M + 1 + self.N_P  # memory (incl. its own input qubit) + ancilla + processor


@dataclass
class DirectionalRun:
    u: np.ndarray
    labels_mem: list
    labels_proc: list
    labels_cross: list
    X_mem: np.ndarray
    X_proc: np.ndarray
    X_cross: np.ndarray
    mem_input_qubit: int
    mem_qubits: list
    ancilla_qubit: int
    proc_qubits: list
    resources: ResourceUsage

    @property
    def X_combined(self) -> np.ndarray:
        return np.hstack([self.X_mem, self.X_proc])

    @property
    def X_combined_with_cross(self) -> np.ndarray:
        return np.hstack([self.X_mem, self.X_proc, self.X_cross])


def build_directional_circuit(cfg: DirectionalConfig, u_seq: Sequence[float], seeds):
    mem_input = 0
    mem_qubits = list(range(1, cfg.N_M))            # N_M-1 storage qubits; mem_input is the N_M-th memory-side qubit
    ancilla = cfg.N_M
    proc_start = cfg.N_M + 1
    proc_qubits = list(range(proc_start, proc_start + cfg.N_P))
    N_total = cfg.n_qubits_total

    if cfg.g_processor is not None and cfg.J_processor is not None:
        proc_params = dproc.sample_params_gJ(cfg.N_P, cfg.g_processor, cfg.J_processor,
                                              term_seed=seeds.reservoir_seed + 1,
                                              disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor)
    else:
        proc_params = dproc.sample_params(cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
                                           disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor,
                                           G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
    proc_layer = dproc.layer_subcircuit(proc_params)

    m_tap = mem_qubits[-1]      # memory qubit tapped by the collision channel
    p_entry = proc_qubits[0]    # processor qubit the ancilla couples into

    labels_mem, ops_mem = memory_local_ops(mem_qubits, max_weight=cfg.max_weight_mem)
    labels_proc, ops_proc = processor_local_ops(proc_qubits, max_weight=cfg.max_weight_proc)
    labels_cross, ops_cross = cross_ops([m_tap], [p_entry])

    qc = QuantumCircuit(N_total)

    if cfg.memory_variant == "shift":
        from .spatial_memory import apply_shift_step
        slots = [mem_input] + mem_qubits

        def mem_step(u_t):
            apply_shift_step(qc, slots, mem_input, mem_qubits, u_t, damping=None)
    elif cfg.memory_variant == "protected_integrable":
        from .memory import apply_memory_step, sample_memory_disorder
        omega = sample_memory_disorder(cfg.N_M, seeds.reservoir_seed, scale=2 * np.pi)

        def mem_step(u_t):
            apply_memory_step(qc, mem_input, mem_qubits, "integrable", cfg.epsilon_M, cfg.epsilon_M,
                               cfg.epsilon_M, omega, u_t)
    else:
        raise ValueError(f"unknown memory_variant {cfg.memory_variant!r}")

    for t, u_t in enumerate(u_seq):
        mem_step(u_t)
        apply_collision_step(qc, m_tap, ancilla, p_entry, cfg.theta, cfg.phi, ap_kind=cfg.ap_kind,
                              phi_x=cfg.phi_x, phi_y=cfg.phi_y, phi_z=cfg.phi_z, ma_kind=cfg.ma_kind)
        qc.compose(proc_layer, qubits=proc_qubits, inplace=True)
        for (op, qargs), lab in zip(ops_mem, labels_mem):
            qc.save_expectation_value(op, qargs, label=f"mem_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(op, qargs, label=f"proc_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_cross, labels_cross):
            qc.save_expectation_value(op, qargs, label=f"cross_{lab}__t{t}")

    return qc, (labels_mem, labels_proc, labels_cross), mem_qubits, ancilla, proc_qubits


def run_directional_dqrc(cfg: DirectionalConfig, T: int, master_seed: int = 0,
                          method: str = "density_matrix", use_gpu: bool = False) -> DirectionalRun:
    if method != "density_matrix":
        raise ValueError("directional_dqrc circuits reset a qubit every step -- use "
                          "method='density_matrix' (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10).")
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    qc, (labels_mem, labels_proc, labels_cross), mem_qubits, ancilla, proc_qubits = build_directional_circuit(
        cfg, u, seeds)

    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)

    X_mem = np.array([[np.real(data[f"mem_{lab}__t{t}"]) for lab in labels_mem] for t in range(T)])
    X_proc = np.array([[np.real(data[f"proc_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])
    X_cross = np.array([[np.real(data[f"cross_{lab}__t{t}"]) for lab in labels_cross] for t in range(T)])

    resources = ResourceUsage(n_qubits_physical=cfg.n_qubits_total, n_ancilla=1, circuit_depth=tqc.depth(),
                               two_qubit_gates=0, shots=0, n_features=len(labels_mem) + len(labels_proc),
                               notes=f"directional_dqrc N_M={cfg.N_M} N_P={cfg.N_P} variant={cfg.memory_variant} "
                                     f"theta={cfg.theta} phi={cfg.phi} ap_kind={cfg.ap_kind}")
    return DirectionalRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, labels_cross=labels_cross,
                           X_mem=X_mem, X_proc=X_proc, X_cross=X_cross, mem_input_qubit=0,
                           mem_qubits=mem_qubits, ancilla_qubit=ancilla, proc_qubits=proc_qubits,
                           resources=resources)


def reference_direct_coupling_run(kind: str, N_M: int, N_P: int, kappa_processor: float, lambda_mp: float,
                                   T: int, master_seed: int = 0):
    """Part 15's reference configurations A ('zx') and B ('heisenberg') --
    calls `id_memory_eoc.py` directly (the prior pass's own module), never
    reimplemented here."""
    from .id_memory_eoc import IDMemoryEOCConfig, run_id_memory_eoc
    cfg = IDMemoryEOCConfig(L=N_M + 1, N_P=N_P, kappa_processor=kappa_processor, interface_kind=kind,
                             lambda_mp=lambda_mp, reps_processor=1, n_taps=1)
    run = run_id_memory_eoc(cfg, T=T, master_seed=master_seed)
    return run
