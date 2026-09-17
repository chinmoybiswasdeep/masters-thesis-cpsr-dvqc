"""
reset_ablation.py -- Part 12 of the V2 validation spec: does PERSISTENT
processor state (never reset across timesteps) cause the memory-control
coupling found in `docs/DQRC_GJ_DECOUPLING_VALIDATION.md` (large, stable
dNL/dm)? Tests K in {1, 2, 4, infinity} -- reset the processor register
every K timesteps (K=infinity = the original persistent processor,
unchanged).

Implemented as a NEW circuit builder (does not modify
`directional_dqrc.build_directional_circuit`, which stays exactly as
tested) that reuses every existing gate-application helper
(`memory.apply_memory_step`, `spatial_memory.apply_shift_step`,
`collision_interface.apply_collision_step`,
`directional_processor.sample_params`/`sample_params_gJ`,
`diagnostics.memory_local_ops`/`processor_local_ops`/`cross_ops`) -- only
the NEW periodic-reset instruction is added.

Qiskit's `qc.reset(q)` is itself a valid CPTP (trace-preserving) channel
(Aer implements it as an exact partial-trace-then-reinitialize, the same
mechanism this project's `spatial_memory`/`collision_interface` modules
already rely on for the ancilla and memory-shift-register reset each
step) -- reusing it here is not a new or ad hoc "reset by hand" operation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from . import directional_processor as dproc
from .collision_interface import apply_collision_step
from .diagnostics import memory_local_ops, processor_local_ops, cross_ops
from .directional_dqrc import DirectionalConfig, DirectionalRun
from .utils import ensure_repo_code_on_path, ResourceUsage, make_seed_bundle

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input  # noqa: E402


def build_directional_circuit_with_reset(cfg: DirectionalConfig, u_seq: Sequence[float], seeds,
                                          reset_period):
    """Same gate sequence as `directional_dqrc.build_directional_circuit`,
    with one addition: every `reset_period`-th timestep (1-indexed;
    `reset_period=1` resets EVERY step, `reset_period=None` never resets --
    the original persistent-processor behavior), reset every processor
    qubit to |0> immediately after that step's processor layer, before the
    readout snapshot."""
    mem_input = 0
    mem_qubits = list(range(1, cfg.N_M))
    ancilla = cfg.N_M
    proc_start = cfg.N_M + 1
    proc_qubits = list(range(proc_start, proc_start + cfg.N_P))
    N_total = cfg.n_qubits_total

    if cfg.memory_variant == "shift":
        from .spatial_memory import apply_shift_step
        slots = [mem_input] + mem_qubits

        def mem_step(qc, u_t):
            apply_shift_step(qc, slots, mem_input, mem_qubits, u_t, damping=None)
    elif cfg.memory_variant == "protected_integrable":
        from .memory import apply_memory_step, sample_memory_disorder
        omega = sample_memory_disorder(cfg.N_M, seeds.reservoir_seed, scale=2 * np.pi)

        def mem_step(qc, u_t):
            apply_memory_step(qc, mem_input, mem_qubits, "integrable", cfg.epsilon_M, cfg.epsilon_M,
                               cfg.epsilon_M, omega, u_t)
    else:
        raise ValueError(f"unknown memory_variant {cfg.memory_variant!r}")

    if cfg.g_processor is not None and cfg.J_processor is not None:
        proc_params = dproc.sample_params_gJ(cfg.N_P, cfg.g_processor, cfg.J_processor,
                                              term_seed=seeds.reservoir_seed + 1,
                                              disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor)
    else:
        proc_params = dproc.sample_params(cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
                                           disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor,
                                           G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
    proc_layer = dproc.layer_subcircuit(proc_params)

    m_tap = mem_qubits[-1] if mem_qubits else mem_input
    p_entry = proc_qubits[0]

    labels_mem, ops_mem = memory_local_ops(mem_qubits, max_weight=cfg.max_weight_mem)
    labels_proc, ops_proc = processor_local_ops(proc_qubits, max_weight=cfg.max_weight_proc)
    labels_cross, ops_cross = cross_ops([m_tap], [p_entry])

    qc = QuantumCircuit(N_total)
    for t, u_t in enumerate(u_seq):
        mem_step(qc, u_t)
        apply_collision_step(qc, m_tap, ancilla, p_entry, cfg.theta, cfg.phi, ap_kind=cfg.ap_kind,
                              phi_x=cfg.phi_x, phi_y=cfg.phi_y, phi_z=cfg.phi_z)
        qc.compose(proc_layer, qubits=proc_qubits, inplace=True)
        for (op, qargs), lab in zip(ops_mem, labels_mem):
            qc.save_expectation_value(op, qargs, label=f"mem_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(op, qargs, label=f"proc_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_cross, labels_cross):
            qc.save_expectation_value(op, qargs, label=f"cross_{lab}__t{t}")
        if reset_period is not None and (t + 1) % reset_period == 0:
            for q in proc_qubits:
                qc.reset(q)

    return qc, (labels_mem, labels_proc, labels_cross), mem_qubits, ancilla, proc_qubits


def run_directional_dqrc_with_reset(cfg: DirectionalConfig, T: int, reset_period, master_seed: int = 0,
                                     method: str = "density_matrix", use_gpu: bool = False) -> DirectionalRun:
    if method != "density_matrix":
        raise ValueError("reset-ablation circuits reset qubits every step -- use method='density_matrix' "
                          "(docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10).")
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    qc, (labels_mem, labels_proc, labels_cross), mem_qubits, ancilla, proc_qubits = \
        build_directional_circuit_with_reset(cfg, u, seeds, reset_period)

    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)

    X_mem = np.array([[np.real(data[f"mem_{lab}__t{t}"]) for lab in labels_mem] for t in range(T)])
    X_proc = np.array([[np.real(data[f"proc_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])
    X_cross = np.array([[np.real(data[f"cross_{lab}__t{t}"]) for lab in labels_cross] for t in range(T)])

    resources = ResourceUsage(n_qubits_physical=cfg.n_qubits_total, n_ancilla=1, circuit_depth=tqc.depth(),
                               two_qubit_gates=0, shots=0, n_features=len(labels_mem) + len(labels_proc),
                               notes=f"reset_ablation reset_period={reset_period}")
    return DirectionalRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, labels_cross=labels_cross,
                           X_mem=X_mem, X_proc=X_proc, X_cross=X_cross, mem_input_qubit=0,
                           mem_qubits=mem_qubits, ancilla_qubit=ancilla, proc_qubits=proc_qubits,
                           resources=resources)
