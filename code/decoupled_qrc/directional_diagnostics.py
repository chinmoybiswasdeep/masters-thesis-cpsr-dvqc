"""
directional_diagnostics.py -- back-action (D_M/F_M/purity), the engineering
transfer metric Q_transfer, eta_M/eta_NL, and the full-channel QND check for
the M -> A -> P architecture (Parts 5/6/11/14). Every metric here is a thin
composition of already-verified primitives (`interfaces_advanced.
trace_distance`/`memory_disturbance`, `id_memory_eoc.compute_etas`,
`diagnostics.ipc_MN`) -- nothing fundamentally new is computed from scratch.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy

from .directional_dqrc import DirectionalConfig, build_directional_circuit
from .id_memory_eoc import compute_etas  # noqa: F401 -- re-exported for convenience
from .interfaces_advanced import trace_distance, memory_disturbance  # noqa: F401
from .diagnostics import ipc_MN  # noqa: F401
from .utils import ensure_repo_code_on_path, make_seed_bundle

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input  # noqa: E402


def _final_density_matrix(cfg: DirectionalConfig, u_seq, seeds):
    qc, groups, mem_qubits, ancilla, proc_qubits = build_directional_circuit(cfg, u_seq, seeds)
    qc.save_density_matrix(label="rho")
    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    rho = DensityMatrix(np.asarray(result.data(0)["rho"]))
    all_qubits = list(range(qc.num_qubits))
    return rho, mem_qubits, ancilla, proc_qubits, all_qubits


def memory_back_action(cfg: DirectionalConfig, T_small: int, master_seed: int = 0) -> dict:
    """Part 5's D_M/F_M/purity-change: memory evolution WITHOUT the
    collision channel (theta=phi=0, i.e. memory never touches A/P at all)
    vs. WITH it (cfg's own theta/phi), same trajectory otherwise."""
    seeds = make_seed_bundle(master_seed)
    u = random_input(T_small, seed=seeds.dataset_seed)

    d = dict(cfg.__dict__)
    d["theta"] = 0.0
    d["phi"] = 0.0
    cfg_off = DirectionalConfig(**d)

    rho_on, mem_q, _, _, all_q = _final_density_matrix(cfg, u, seeds)
    rho_off, _, _, _, _ = _final_density_matrix(cfg_off, u, seeds)
    return memory_disturbance(rho_on, rho_off, mem_q, all_q)


def q_transfer(nl_delta: float, D_M: float, eps: float = 1e-9) -> float:
    """'Engineering transfer efficiency' (Part 6) -- explicitly NOT a
    fundamental quantity. Q_transfer = Delta NL_P / (D_M + eps)."""
    return nl_delta / (D_M + eps)


def full_channel_qnd_check(cfg: DirectionalConfig, T_small: int, master_seed: int = 0) -> dict:
    """Part 14's second, stronger QND check: the bare commutator
    `collision_interface.qnd_commutator_norm` only proves U_MA itself
    preserves Z_M in isolation -- it says nothing about whether the FULL
    channel (including the A-P interaction and ancilla discard, repeated
    every timestep) leaves <Z_M> undisturbed in practice. This runs the
    real trajectory and reports how much <Z_qubit> for the memory TAP
    qubit differs, at each snapshot step, between theta>0 (QND coupling
    active) and theta=0 (no M-A coupling at all) -- entanglement + discard
    CAN still perturb <Z_M> indirectly even with an exactly-QND U_MA,
    since discarding an ancilla ENTANGLED with M via U_MA is not itself
    the identity on M unless additionally protected."""
    seeds = make_seed_bundle(master_seed)
    u = random_input(T_small, seed=seeds.dataset_seed)
    m_tap = cfg.N_M  # last memory qubit index (mem_qubits[-1])

    def z_tap_trace(theta_val):
        d = dict(cfg.__dict__)
        d["theta"] = theta_val
        c = DirectionalConfig(**d)
        rho, mem_q, _, _, _ = _final_density_matrix(c, u, seeds)
        other = [q for q in range(rho.num_qubits) if q != m_tap]
        rho_tap = partial_trace(rho, other)
        return float(np.real(np.trace(np.asarray(rho_tap.data) @ np.array([[1, 0], [0, -1]]))))

    z_with_qnd = z_tap_trace(cfg.theta)
    z_without_ma = z_tap_trace(0.0)
    return {"z_tap_with_MA_coupling": z_with_qnd, "z_tap_no_MA_coupling": z_without_ma,
            "z_tap_disturbance": abs(z_with_qnd - z_without_ma)}
