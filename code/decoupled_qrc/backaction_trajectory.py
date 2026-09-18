"""
backaction_trajectory.py -- V2.1 Phase 10: V1/V2 only ever measured back-
action `D_M` at the FINAL timestep of a short trajectory. This module
extends that to a full time-resolved `D_M(t)`, reusing every existing
gate-application helper (the same additive-reuse pattern
`reset_ablation.py` already established) and
`interfaces_advanced.memory_disturbance` for the actual on/off comparison
at each saved timestep -- no new physics, only a new circuit-annotation
(save the density matrix every step, not only at the end) and new
orchestration around the existing comparison.

Deliberately restricted to SMALL T (a handful of steps): saving a full
density matrix every step for an 8-qubit system is `2^8 x 2^8` complex
per save, so T is kept in the ~15-25 range this experiment actually needs
(matching V1/V2's own T=12 single-point back-action check), never the
T~150-1000 used for the main IPC circuits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix

from .collision_interface import apply_collision_step
from .directional_dqrc import DirectionalConfig
from . import directional_processor as dproc
from .interfaces_advanced import memory_disturbance
from .utils import ensure_repo_code_on_path
from .validation_utils import NestedSeeds

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input  # noqa: E402


def _build_trajectory_circuit(cfg: DirectionalConfig, u_seq: Sequence[float], seeds: NestedSeeds):
    """Same gate sequence as `directional_dqrc.build_directional_circuit`,
    but saves the FULL density matrix after every step (label `rho_t{t}`)
    instead of per-operator expectation values -- this experiment needs
    the whole state, not IPC features."""
    mem_input = 0
    mem_qubits = list(range(1, cfg.N_M))
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

    m_tap = mem_qubits[-1] if mem_qubits else mem_input
    p_entry = proc_qubits[0]

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
        qc.save_density_matrix(label=f"rho_t{t}")

    return qc, mem_qubits


@dataclass
class BackActionTrajectory:
    steps: list
    trace_distance: list
    fidelity: list
    purity_with_interface: list
    purity_without_interface: list
    purity_change: list
    mean_trace_distance: float
    max_trace_distance: float
    final_trace_distance: float


def compute_backaction_trajectory(cfg_on: DirectionalConfig, cfg_off: DirectionalConfig, T: int,
                                   seeds: NestedSeeds, method: str = "density_matrix") -> BackActionTrajectory:
    """Runs the interface-ON (`cfg_on`) and interface-OFF (`cfg_off`,
    typically theta=phi=0) trajectories on the SAME `seeds` (common random
    numbers: identical input sequence and Hamiltonian realization), and
    computes `interfaces_advanced.memory_disturbance` between the two
    memory reduced states at EVERY timestep, not just the last."""
    if method != "density_matrix":
        raise ValueError("backaction trajectory circuits need method='density_matrix'.")
    u = random_input(T, seed=seeds.input_seed)
    qc_on, mem_qubits = _build_trajectory_circuit(cfg_on, u, seeds)
    qc_off, _ = _build_trajectory_circuit(cfg_off, u, seeds)

    sim = make_simulator(method=method)
    tqc_on = transpile(qc_on, sim, optimization_level=1)
    tqc_off = transpile(qc_off, sim, optimization_level=1)
    result_on = sim.run(tqc_on, shots=1).result().data(0)
    result_off = sim.run(tqc_off, shots=1).result().data(0)

    all_qubits = list(range(qc_on.num_qubits))
    steps, td, fid, pur_with, pur_without, pur_change = [], [], [], [], [], []
    for t in range(T):
        rho_on = DensityMatrix(np.asarray(result_on[f"rho_t{t}"]))
        rho_off = DensityMatrix(np.asarray(result_off[f"rho_t{t}"]))
        d = memory_disturbance(rho_on, rho_off, mem_qubits, all_qubits)
        steps.append(t)
        td.append(d["trace_distance"])
        fid.append(d["fidelity"])
        pur_with.append(d["purity_with_interface"])
        pur_without.append(d["purity_without_interface"])
        pur_change.append(d["purity_change"])

    return BackActionTrajectory(
        steps=steps, trace_distance=td, fidelity=fid, purity_with_interface=pur_with,
        purity_without_interface=pur_without, purity_change=pur_change,
        mean_trace_distance=float(np.mean(td)), max_trace_distance=float(np.max(td)),
        final_trace_distance=float(td[-1]),
    )
