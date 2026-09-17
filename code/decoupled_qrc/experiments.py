"""
experiments.py -- orchestration: builds the COMBINED DQRC trajectory circuit
(memory register + interface + EOC processor, optionally modular/stitched,
in ONE circuit), wraps it with the same (labels, X, resources) shape as
`baseline.py`, and runs the Part 11 ablation list / Part 9 fixed-resource
comparisons on top of the other modules. This is the only module that knows
about ALL of memory.py/processor.py/interface.py/stitching.py/shadows.py at
once -- everything else stays independent, which is what makes the Part 10
Jacobian experiment meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from . import memory as memmod
from . import processor as procmod
from . import interface as ifacemod
from . import stitching as stitchmod
from . import baseline as basemod
from .utils import ensure_repo_code_on_path, ResourceUsage, SeedBundle, make_seed_bundle

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402
from qrc_qiskit import make_simulator, random_input  # noqa: E402


@dataclass
class DQRCConfig:
    N_M: int = 3
    N_P: int = 4
    memory_mode: str = "integrable"
    lambda_im: float = 0.3
    epsilon: float = 0.05
    omega_scale: float = 1.0
    kappa_processor: float = 1.0
    reps_processor: int = 2
    lambda_mp: float = 0.3
    interface_kind: str = "rzz"
    n_taps: int = 1
    max_weight_readout: int = 3
    G_MAX: float = 0.6
    J_MAX: float = 0.6
    stitching_blocks: tuple = None       # e.g. ((2, kappa1), (2, kappa2)) -- None = monolithic processor
    lambda_stitch: float = 0.2
    input_qubit: int = 0

    @property
    def n_qubits_total(self) -> int:
        n_anc = stitchmod.n_ancillas_needed(len(self.stitching_blocks)) if self.stitching_blocks else 0
        return 1 + self.N_M + self.N_P + n_anc


@dataclass
class DQRCRun:
    labels: list
    X: np.ndarray
    u: np.ndarray
    mem_qubits: list
    proc_qubits: list
    ancilla_qubits: list
    resources: ResourceUsage
    circuit_depth: int


def build_dqrc_circuit(cfg: DQRCConfig, u_seq: Sequence[float], seeds: SeedBundle):
    mem_qubits = list(range(1, 1 + cfg.N_M))
    proc_start = 1 + cfg.N_M
    proc_qubits = list(range(proc_start, proc_start + cfg.N_P))
    ancilla_qubits = []
    N_total = cfg.n_qubits_total

    omega = memmod.sample_memory_disorder(
        cfg.N_M, seeds.reservoir_seed, scale=(2 * np.pi if cfg.memory_mode != "swap" else cfg.omega_scale))

    if cfg.stitching_blocks is None:
        proc_params = procmod.sample_processor_params(
            cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
            disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor, G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
        proc_layer = procmod.processor_reps_subcircuit(proc_params)
        proc_target_qubits = proc_qubits
    else:
        # Built entirely on a LOCAL (N_P + n_anc)-sized register (indices
        # 0..N_P-1 = blocks, N_P.. = ancillas); `qc.compose(..., qubits=
        # proc_target_qubits)` below remaps local -> global, the same
        # Qiskit-native pattern `processor_reps_subcircuit` relies on, so no
        # manual global-index arithmetic is needed here.
        sizes = [b[0] for b in cfg.stitching_blocks]
        kappas = [b[1] for b in cfg.stitching_blocks]
        n_anc = stitchmod.n_ancillas_needed(len(cfg.stitching_blocks))
        local_blocks = stitchmod.build_modular_blocks(sizes, kappas, term_seed_base=seeds.reservoir_seed + 1,
                                                        disorder_seed_base=seeds.reservoir_seed + 2,
                                                        reps=cfg.reps_processor, start_qubit=0,
                                                        G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
        local_ancillas = list(range(cfg.N_P, cfg.N_P + n_anc))
        proc_layer = stitchmod.modular_processor_subcircuit(local_blocks, local_ancillas, cfg.lambda_stitch,
                                                              n_total_qubits=(cfg.N_P + n_anc))
        ancilla_qubits = list(range(proc_start + cfg.N_P, proc_start + cfg.N_P + n_anc))
        proc_target_qubits = proc_qubits + ancilla_qubits
        proc_params = None

    n_taps_eff = min(cfg.n_taps, cfg.N_M, cfg.N_P)
    memory_taps = mem_qubits[-n_taps_eff:]
    processor_entry = proc_qubits[:n_taps_eff]

    labels, ops = msc.feature_ops_mem_all(N_total, input_qubit=cfg.input_qubit, max_weight=cfg.max_weight_readout)

    qc = QuantumCircuit(N_total)
    for t, u_t in enumerate(u_seq):
        memmod.apply_memory_step(qc, cfg.input_qubit, mem_qubits, cfg.memory_mode, cfg.lambda_im,
                                  cfg.epsilon, cfg.omega_scale, omega, u_t)
        if cfg.lambda_mp != 0:
            ifacemod.apply_interface(qc, memory_taps, processor_entry, cfg.lambda_mp, cfg.interface_kind)
        qc.compose(proc_layer, qubits=proc_target_qubits, inplace=True)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f"{lab}__t{t}")

    return qc, labels, mem_qubits, proc_qubits, ancilla_qubits


def run_dqrc(cfg: DQRCConfig, T: int, master_seed: int = 0, method: str = "density_matrix",
             use_gpu: bool = False) -> DQRCRun:
    if method != "density_matrix":
        raise ValueError("experiments.run_dqrc requires method='density_matrix' -- the combined "
                          "circuit resets the input qubit every step, the confirmed Aer bug's exact "
                          "trigger pattern (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10).")
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    qc, labels, mem_qubits, proc_qubits, ancilla_qubits = build_dqrc_circuit(cfg, u, seeds)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f"{lab}__t{t}"])

    two_q_est = _estimate_two_qubit_gates(cfg, T)
    resources = ResourceUsage(
        n_qubits_physical=cfg.n_qubits_total, n_ancilla=len(ancilla_qubits),
        circuit_depth=tqc.depth(), two_qubit_gates=two_q_est, shots=0, n_features=len(labels),
        notes=f"DQRC N_M={cfg.N_M} N_P={cfg.N_P} memory_mode={cfg.memory_mode} "
              f"kappa_processor={cfg.kappa_processor} lambda_mp={cfg.lambda_mp} "
              f"stitched={cfg.stitching_blocks is not None}",
    )
    return DQRCRun(labels=labels, X=X, u=u, mem_qubits=mem_qubits, proc_qubits=proc_qubits,
                   ancilla_qubits=ancilla_qubits, resources=resources, circuit_depth=tqc.depth())


def _estimate_two_qubit_gates(cfg: DQRCConfig, T: int) -> int:
    mem_2q_per_step = 2 + 2 * (cfg.N_M - 1)  # input-memory coupling (2 gates) + chain internal (~2 per edge)
    proc_terms = msc.default_n_sparse_terms(cfg.N_P)
    proc_2q_per_layer = 2 * (cfg.N_P - 1) + 6 * proc_terms
    proc_2q_per_step = proc_2q_per_layer * cfg.reps_processor
    iface_2q = ifacemod.interface_two_qubit_gate_count(min(cfg.n_taps, cfg.N_M, cfg.N_P)) if cfg.lambda_mp else 0
    stitch_2q = 2 * len(cfg.stitching_blocks) if cfg.stitching_blocks else 0
    return (mem_2q_per_step + proc_2q_per_step + iface_2q + stitch_2q) * T


# =============================================================================
# Part 11 -- ablation registry
# =============================================================================
# Each ablation is dispatched to whichever module actually implements that
# specific single- or combined-subsystem architecture, rather than forcing
# every case through `build_dqrc_circuit` (a "processor only" run has no
# memory register at all, so it is a `processor.run_processor_standalone`
# call, not a degenerate DQRCConfig with N_M=0). Every ablation returns the
# same (labels, X, u, resources) shape so `ipc.compute_ipc` and
# `metrics.frontier_comparison` never need to know which module produced it.

ABLATION_NAMES = (
    "monolithic_qrc", "monolithic_qrc_at_eoc", "classical_delay_control",
    "memory_only", "processor_only_at_eoc",
    "dqrc_no_stitch", "dqrc_with_stitch", "dqrc_randomized_processor",
)
# 8-9 (exact vs shadow readout) are handled separately by
# `shadows.shadow_convergence_sweep` on a representative DQRC state, not by
# re-running the full IPC pipeline twice -- see docs/DQRC_RESULTS.md question
# 6. 12 (genuine IDQNN) is not implemented -- see
# experimental/idqnn_memory_prototype.py. 13 (ideal_depth_compressed_control)
# is a standalone numerical comparison, not an IPC-pipeline ablation -- see
# that module directly.


def _replace(cfg: DQRCConfig, **kwargs) -> DQRCConfig:
    d = dict(cfg.__dict__)
    d.update(kwargs)
    return DQRCConfig(**d)


def run_ablation(name: str, T: int, master_seed: int, dqrc_cfg: DQRCConfig = None,
                  N_total: int = 8, science_kappa: float = 0.960):
    """Dispatch one named ablation (see ABLATION_NAMES) to its implementing
    module and return (labels, X, u, resources) -- a uniform shape regardless
    of which architecture actually ran. `N_total` is the FIXED total physical
    qubit budget (Part 9's matched-resource requirement) every single-register
    ablation uses in full; DQRC-combined ablations split it as
    `dqrc_cfg` specifies (`1 + N_M + N_P [+ ancillas]` must equal `N_total`
    for a fair comparison -- callers are responsible for constructing
    `dqrc_cfg` that way when a fixed-qubit comparison is the point)."""
    if name not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation {name!r}, expected one of {ABLATION_NAMES}")
    seeds = make_seed_bundle(master_seed)

    if name == "monolithic_qrc":
        run = basemod.monolithic_qrc(N=N_total, kappa=1.0, T=T, term_seed=seeds.reservoir_seed,
                                      disorder_seed=seeds.reservoir_seed + 1, input_seed=seeds.dataset_seed)
        return run.labels, run.X, run.u, run.resources

    if name == "monolithic_qrc_at_eoc":
        run = basemod.monolithic_qrc_at_eoc(N=N_total, T=T, science_kappa=science_kappa,
                                             term_seed=seeds.reservoir_seed,
                                             disorder_seed=seeds.reservoir_seed + 1, input_seed=seeds.dataset_seed)
        return run.labels, run.X, run.u, run.resources

    if name == "classical_delay_control":
        run = basemod.classical_delay_qrc(T=T, m=10, input_seed=seeds.dataset_seed)
        return run.labels, run.X, run.u, run.resources

    if name == "memory_only":
        mode = dqrc_cfg.memory_mode if dqrc_cfg else "integrable"
        u = random_input(T, seed=seeds.dataset_seed)
        run = memmod.run_memory_register(N=N_total, u_seq=u, memory_mode=mode,
                                          lambda_im=dqrc_cfg.lambda_im if dqrc_cfg else 0.3,
                                          epsilon=dqrc_cfg.epsilon if dqrc_cfg else 0.05,
                                          disorder_seed=seeds.reservoir_seed)
        n_mem = N_total - 1
        two_q = (2 + 2 * (n_mem - 1)) * T
        resources = ResourceUsage(n_qubits_physical=N_total, n_ancilla=0, circuit_depth=run.circuit_depth,
                                   two_qubit_gates=two_q, shots=0, n_features=len(run.labels),
                                   notes=f"memory_only mode={mode}")
        return run.labels, run.X, run.u, resources

    if name == "processor_only_at_eoc":
        labels, X, u, info = procmod.run_processor_standalone(
            N_p=N_total - 1, kappa_processor=science_kappa, T=T, term_seed=seeds.reservoir_seed,
            disorder_seed=seeds.reservoir_seed + 1, input_seed=seeds.dataset_seed)
        resources = ResourceUsage(n_qubits_physical=N_total - 1, n_ancilla=0,
                                   circuit_depth=info["circuit_depth"], two_qubit_gates=0, shots=0,
                                   n_features=info["n_features"], notes="processor_only_at_eoc")
        return labels, X, u, resources

    cfg = dqrc_cfg or DQRCConfig()
    if name == "dqrc_with_stitch" and cfg.stitching_blocks is None:
        half = cfg.N_P // 2
        cfg = _replace(cfg, N_P=2 * half, stitching_blocks=((half, cfg.kappa_processor), (cfg.N_P - half, cfg.kappa_processor)))
    elif name == "dqrc_no_stitch":
        cfg = _replace(cfg, stitching_blocks=None)
    elif name == "dqrc_randomized_processor":
        cfg = _replace(cfg, kappa_processor=1e-6, stitching_blocks=None)  # ~pure SYK4: maximally scrambling, non-EOC

    run = run_dqrc(cfg, T=T, master_seed=master_seed)
    return run.labels, run.X, run.u, run.resources
