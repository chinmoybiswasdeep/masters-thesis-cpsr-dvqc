"""
diagnostics.py -- architecture-repair diagnostics for DQRC (Phases 1-9 of the
"where does the processor's nonlinear information go?" investigation that
followed the FAST_MODE run in docs/DQRC_RESULTS.md).

Two real inconsistencies were found by hand before writing any of this
module's analysis code (documented in docs/DQRC_ARCHITECTURE_REPAIR.md
question 1):

  1. `BASE_DQRC_CFG` in the FAST_MODE notebook used `max_weight_readout=2`
     while `processor.run_processor_standalone`'s default is `max_weight=3`
     -- an unmatched readout dictionary, not a fair Part-9 resource-matched
     comparison.
  2. The `dqrc_no_stitch`/`dqrc_with_stitch` ablations ran at
     `kappa_processor=1.0` (DQRCConfig's dataclass default) while
     `processor_only_at_eoc` ran at `kappa=science_kappa=0.960` -- the two
     were never actually compared AT the same point on the kappa axis.

Every function below takes `max_weight_proc`/`kappa_processor` as EXPLICIT
arguments precisely so a caller cannot repeat that mistake silently.

This module also implements the GROUPED readout Phase 3 requires
(X_memory / X_processor / X_cross, all from real quantum observables, never
a classical delay-line concatenation) -- `mixed_syk_core.feature_ops_mem_all`
only captures ADJACENT-in-global-index Pauli strings, which for N_P>3 misses
non-adjacent processor-processor correlations that the SYK4 term sampling
(`itertools.combinations` over ALL 4-tuples, not just chain-adjacent ones)
can genuinely generate -- so the grouped processor readout here uses ALL
pairs/triples within the processor qubit set, not just adjacent ones.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, DensityMatrix, partial_trace, entropy

from . import memory as memmod
from . import processor as procmod
from . import interface as ifacemod
from . import stitching as stitchmod
from .experiments import DQRCConfig
from .utils import ensure_repo_code_on_path, ResourceUsage, SeedBundle, make_seed_bundle

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input, chrono_split, select_and_eval_ridge  # noqa: E402
from decoupled_qrc import ipc as ipcmod  # noqa: E402


# =============================================================================
# Grouped feature-op builders (real quantum observables only)
# =============================================================================

def _weight1_ops(qubits: Sequence[int], paulis=("Z", "X", "Y")):
    ops, labels = [], []
    for q in qubits:
        for p in paulis:
            ops.append((Pauli(p), [q]))
            labels.append(f"{p}{q}")
    return labels, ops


def _all_combo_ops(qubits: Sequence[int], weight: int, paulis=("Z", "X", "Y")):
    """ALL C(len(qubits), weight) combinations (not just adjacent-in-index),
    every Pauli assignment -- richer than `feature_ops_all_general`'s
    adjacent-only convention, needed because SYK4 terms are sampled over ALL
    4-tuples (`mixed_syk_core.sample_syk4_terms`, `itertools.combinations`),
    so non-adjacent correlations are physically real, not just noise."""
    ops, labels = [], []
    for combo in itertools.combinations(qubits, weight):
        for pcombo in itertools.product(paulis, repeat=weight):
            ops.append((Pauli("".join(pcombo)), list(combo)))
            labels.append("".join(f"{p}{q}" for p, q in zip(pcombo, combo)))
    return labels, ops


def processor_local_ops(proc_qubits: Sequence[int], max_weight: int = 3):
    labels, ops = [], []
    for w in range(1, min(max_weight, len(proc_qubits)) + 1):
        if w == 1:
            l, o = _weight1_ops(proc_qubits)
        else:
            l, o = _all_combo_ops(proc_qubits, w)
        labels += l
        ops += o
    return labels, ops


def memory_local_ops(mem_qubits: Sequence[int], max_weight: int = 2):
    labels, ops = [], []
    for w in range(1, min(max_weight, len(mem_qubits)) + 1):
        if w == 1:
            l, o = _weight1_ops(mem_qubits)
        else:
            l, o = _all_combo_ops(mem_qubits, w)
        labels += l
        ops += o
    return labels, ops


def cross_ops(mem_taps: Sequence[int], proc_taps: Sequence[int]):
    """Explicit <Z_M Z_P>, <X_M X_P>, <Z_M X_P>, <X_M Z_P> for every
    (mem_tap, proc_tap) pair -- Phase 3's own literal example."""
    ops, labels = [], []
    for m in mem_taps:
        for p in proc_taps:
            for pm, pp in (("Z", "Z"), ("X", "X"), ("Z", "X"), ("X", "Z")):
                ops.append((Pauli(pm + pp), [m, p]))
                labels.append(f"{pm}{m}{pp}{p}")
    return labels, ops


# =============================================================================
# Phase 3 -- grouped combined-circuit run (production-equivalent gate
# sequence, diagnostic readout)
# =============================================================================

@dataclass
class GroupedDQRCRun:
    u: np.ndarray
    labels_mem: list
    labels_proc: list
    labels_cross: list
    X_mem: np.ndarray
    X_proc: np.ndarray
    X_cross: np.ndarray
    resources: ResourceUsage


def run_dqrc_grouped(cfg: DQRCConfig, T: int, master_seed: int = 0, max_weight_proc: int = 3,
                      max_weight_mem: int = 2, method: str = "density_matrix") -> GroupedDQRCRun:
    """Same gate sequence as `experiments.build_dqrc_circuit` (memory step ->
    interface -> processor layer [-> stitching]), but reads out THREE
    separate, explicitly-grouped feature sets instead of one merged
    adjacent-only dictionary."""
    if method != "density_matrix":
        raise ValueError("run_dqrc_grouped requires method='density_matrix' -- see "
                          "docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10.")
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)

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

    n_taps_eff = min(cfg.n_taps, cfg.N_M, cfg.N_P)
    memory_taps = mem_qubits[-n_taps_eff:]
    processor_entry = proc_qubits[:n_taps_eff]

    labels_mem, ops_mem = memory_local_ops(mem_qubits, max_weight=max_weight_mem)
    labels_proc, ops_proc = processor_local_ops(proc_qubits, max_weight=max_weight_proc)
    labels_cross, ops_cross = cross_ops(memory_taps, processor_entry)

    qc = QuantumCircuit(N_total)
    for t, u_t in enumerate(u):
        memmod.apply_memory_step(qc, cfg.input_qubit, mem_qubits, cfg.memory_mode, cfg.lambda_im,
                                  cfg.epsilon, cfg.omega_scale, omega, u_t)
        if cfg.lambda_mp != 0:
            ifacemod.apply_interface(qc, memory_taps, processor_entry, cfg.lambda_mp, cfg.interface_kind)
        qc.compose(proc_layer, qubits=proc_target_qubits, inplace=True)
        for (op, qargs), lab in zip(ops_mem, labels_mem):
            qc.save_expectation_value(op, qargs, label=f"mem_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(op, qargs, label=f"proc_{lab}__t{t}")
        for (op, qargs), lab in zip(ops_cross, labels_cross):
            qc.save_expectation_value(op, qargs, label=f"cross_{lab}__t{t}")

    sim = make_simulator(method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)

    X_mem = np.array([[np.real(data[f"mem_{lab}__t{t}"]) for lab in labels_mem] for t in range(T)])
    X_proc = np.array([[np.real(data[f"proc_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])
    X_cross = np.array([[np.real(data[f"cross_{lab}__t{t}"]) for lab in labels_cross] for t in range(T)])

    resources = ResourceUsage(n_qubits_physical=N_total, n_ancilla=len(ancilla_qubits),
                               circuit_depth=tqc.depth(), two_qubit_gates=0, shots=0,
                               n_features=len(labels_mem) + len(labels_proc) + len(labels_cross),
                               notes="grouped diagnostic readout")
    return GroupedDQRCRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, labels_cross=labels_cross,
                           X_mem=X_mem, X_proc=X_proc, X_cross=X_cross, resources=resources)


# =============================================================================
# Phase 1 -- feature-flow instrumented circuit (memory / processor-pre-stitch
# / processor-post-stitch / final)
# =============================================================================

def run_feature_flow_audit(cfg: DQRCConfig, T: int, master_seed: int = 0, max_weight_proc: int = 3,
                            max_weight_mem: int = 2, method: str = "density_matrix") -> dict:
    """Snapshots processor-local observables at THREE points in the SAME
    trajectory circuit: immediately after the processor's own EOC layer
    (pre-stitch), immediately after stitching (post-stitch, identical to
    pre-stitch if `cfg.stitching_blocks is None`), and memory-local
    observables right after the memory step. Returns
    {'X_M', 'X_P_prestitch', 'X_P_poststitch', 'u'} -- X_FINAL is whichever
    of X_P_poststitch / the grouped final readout the caller wants to feed
    to `ipc.compute_ipc` afterwards (kept separate here since 'final exposed
    features' in production is the MERGED dictionary, computed separately by
    `run_dqrc_grouped`/`experiments.run_dqrc`, not duplicated in this function).
    """
    if cfg.stitching_blocks is not None and cfg.lambda_stitch == 0:
        raise ValueError("set cfg.lambda_stitch != 0 to see a real pre/post-stitch difference")

    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)

    mem_qubits = list(range(1, 1 + cfg.N_M))
    proc_start = 1 + cfg.N_M
    proc_qubits = list(range(proc_start, proc_start + cfg.N_P))
    N_total_no_anc = 1 + cfg.N_M + cfg.N_P
    n_anc = stitchmod.n_ancillas_needed(len(cfg.stitching_blocks)) if cfg.stitching_blocks else 0
    ancilla_qubits = list(range(N_total_no_anc, N_total_no_anc + n_anc))
    N_total = N_total_no_anc + n_anc

    omega = memmod.sample_memory_disorder(
        cfg.N_M, seeds.reservoir_seed, scale=(2 * np.pi if cfg.memory_mode != "swap" else cfg.omega_scale))

    if cfg.stitching_blocks is None:
        proc_params = procmod.sample_processor_params(
            cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
            disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor, G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
        proc_layer_only = procmod.processor_reps_subcircuit(proc_params)
        proc_only_target = proc_qubits
        stitch_pairs = []
    else:
        sizes = [b[0] for b in cfg.stitching_blocks]
        kappas = [b[1] for b in cfg.stitching_blocks]
        local_blocks = stitchmod.build_modular_blocks(sizes, kappas, term_seed_base=seeds.reservoir_seed + 1,
                                                        disorder_seed_base=seeds.reservoir_seed + 2,
                                                        reps=cfg.reps_processor, start_qubit=0,
                                                        G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
        proc_layer_only = QuantumCircuit(cfg.N_P)
        for b in local_blocks:
            proc_layer_only.compose(procmod.processor_reps_subcircuit(b.params), qubits=b.qubits, inplace=True)
        proc_only_target = proc_qubits
        global_blocks = []
        q = proc_start
        for b in local_blocks:
            global_blocks.append(list(range(q, q + len(b.qubits))))
            q += len(b.qubits)
        stitch_pairs = [(global_blocks[i][-1], global_blocks[i + 1][0]) for i in range(len(global_blocks) - 1)]

    n_taps_eff = min(cfg.n_taps, cfg.N_M, cfg.N_P)
    memory_taps = mem_qubits[-n_taps_eff:]
    processor_entry = proc_qubits[:n_taps_eff]

    labels_mem, ops_mem = memory_local_ops(mem_qubits, max_weight=max_weight_mem)
    labels_proc, ops_proc = processor_local_ops(proc_qubits, max_weight=max_weight_proc)

    qc = QuantumCircuit(N_total)
    for t, u_t in enumerate(u):
        memmod.apply_memory_step(qc, cfg.input_qubit, mem_qubits, cfg.memory_mode, cfg.lambda_im,
                                  cfg.epsilon, cfg.omega_scale, omega, u_t)
        for (op, qargs), lab in zip(ops_mem, labels_mem):
            qc.save_expectation_value(op, qargs, label=f"M_{lab}__t{t}")

        if cfg.lambda_mp != 0:
            ifacemod.apply_interface(qc, memory_taps, processor_entry, cfg.lambda_mp, cfg.interface_kind)
        qc.compose(proc_layer_only, qubits=proc_only_target, inplace=True)
        for (op, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(op, qargs, label=f"P_{lab}__t{t}")

        for i, (a, b) in enumerate(stitch_pairs):
            stitchmod.apply_ancilla_stitch(qc, a, b, ancilla_qubits[i], cfg.lambda_stitch)
        for (op, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(op, qargs, label=f"PS_{lab}__t{t}")

    sim = make_simulator(method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)

    X_M = np.array([[np.real(data[f"M_{lab}__t{t}"]) for lab in labels_mem] for t in range(T)])
    X_P = np.array([[np.real(data[f"P_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])
    X_PS = np.array([[np.real(data[f"PS_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])

    return {"u": u, "X_M": X_M, "X_P_prestitch": X_P, "X_P_poststitch": X_PS,
            "labels_mem": labels_mem, "labels_proc": labels_proc}


# =============================================================================
# IPC helper (shared config, avoids re-deriving chrono_split everywhere)
# =============================================================================

def ipc_MN(u, X, CFG, seed=0):
    gap = CFG.max_delay_ipc + 1
    train, val, test = chrono_split(len(u), CFG.washout, CFG.n_val, CFG.n_test, gap)
    r = ipcmod.compute_ipc(u, X, train, val, test, max_delay=CFG.max_delay_ipc, max_degree=CFG.max_degree_ipc,
                            max_targets_per_degree=CFG.max_targets_per_degree, n_surrogates=CFG.n_surrogates,
                            seed=seed)
    return r.memory, r.nonlinearity, r.total, r


# =============================================================================
# Phase 2 -- standalone vs embedded-disconnected vs embedded-connected,
# MATCHED N_P/kappa/reps/max_weight/input sequence.
# =============================================================================

def standalone_processor_matched(N_P: int, kappa_processor: float, T: int, reps: int, max_weight: int,
                                  master_seed: int, G_MAX: float = 0.6, J_MAX: float = 0.6) -> dict:
    """A standalone processor that (a) uses qubit 0 of its OWN register as a
    genuinely recurrent input qubit (reset+Ry(pi*u_t) every step, exactly
    the convention every other reservoir in this repo uses), and (b) is read
    out with the SAME `processor_local_ops` (all-combinations, not
    adjacent-only) dictionary `run_dqrc_grouped` uses for the embedded
    processor -- so any NL gap between this and the embedded cases is NOT
    attributable to a readout-dictionary mismatch."""
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    params = procmod.sample_processor_params(N_P, kappa_processor, term_seed=seeds.reservoir_seed,
                                              disorder_seed=seeds.reservoir_seed + 1, reps=reps,
                                              G_MAX=G_MAX, J_MAX=J_MAX)
    layer = procmod.processor_reps_subcircuit(params)
    labels, ops = processor_local_ops(list(range(N_P)), max_weight=max_weight)

    qc = QuantumCircuit(N_P)
    for t, u_t in enumerate(u):
        qc.reset(0)
        qc.ry(np.pi * float(u_t), 0)
        qc.compose(layer, inplace=True)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f"{lab}__t{t}")

    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    X = np.array([[np.real(data[f"{lab}__t{t}"]) for lab in labels] for t in range(T)])
    return {"u": u, "X": X, "labels": labels}


def embedding_consistency_test(N_M: int, N_P: int, kappa_processor: float, T: int, reps: int,
                                max_weight_proc: int, lambda_mp: float, master_seed: int,
                                CFG, **cfg_kwargs) -> dict:
    """Phase 2's three-way comparison at MATCHED (N_P, kappa, reps,
    max_weight, input seed). Reports M, NL, feature rank, condition number,
    mean feature variance, mean pairwise |correlation| for each of:
    standalone / embedded-disconnected (lambda_mp=0) / embedded-connected."""
    standalone = standalone_processor_matched(N_P, kappa_processor, T, reps, max_weight_proc, master_seed)

    base_kwargs = dict(N_M=N_M, N_P=N_P, kappa_processor=kappa_processor, reps_processor=reps, **cfg_kwargs)
    cfg_disc = DQRCConfig(lambda_mp=0.0, **base_kwargs)
    cfg_conn = DQRCConfig(lambda_mp=lambda_mp, **base_kwargs)

    disc = run_dqrc_grouped(cfg_disc, T=T, master_seed=master_seed, max_weight_proc=max_weight_proc)
    conn = run_dqrc_grouped(cfg_conn, T=T, master_seed=master_seed, max_weight_proc=max_weight_proc)

    out = {}
    for name, u, X in (("standalone", standalone["u"], standalone["X"]),
                        ("embedded_disconnected", disc.u, disc.X_proc),
                        ("embedded_connected", conn.u, conn.X_proc)):
        M, NL, total, _ = ipc_MN(u, X, CFG, seed=master_seed)
        out[name] = {"M": M, "NL": NL, "total": total, **feature_health(X)}
    return out


def feature_health(X: np.ndarray) -> dict:
    """Feature-space health diagnostics: numerical rank, condition number of
    the covariance matrix, mean per-feature variance, mean |pairwise
    correlation| (excluding the diagonal) -- cheap proxies for 'did the
    features collapse/homogenize', used throughout Phases 2/5/12."""
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0)
    nonconstant = std > 1e-10
    rank = int(np.linalg.matrix_rank(Xc))
    cov = np.cov(Xc, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-14]
    cond = float(eigvals.max() / eigvals.min()) if len(eigvals) > 1 else float("inf")
    mean_var = float(np.mean(std ** 2))
    if nonconstant.sum() > 1:
        Xn = Xc[:, nonconstant] / std[nonconstant]
        corr = np.corrcoef(Xn, rowvar=False)
        iu = np.triu_indices_from(corr, k=1)
        mean_abs_corr = float(np.mean(np.abs(corr[iu]))) if len(iu[0]) else float("nan")
    else:
        mean_abs_corr = float("nan")
    return {"rank": rank, "n_features": X.shape[1], "condition_number": cond,
            "mean_feature_variance": mean_var, "mean_abs_pairwise_correlation": mean_abs_corr}


# =============================================================================
# Phase 4 -- g_processor x lambda_MP 2D scan
# =============================================================================

def scan_g_lambda(N_M: int, N_P: int, kappa_grid, lambda_grid, T: int, master_seed: int, CFG,
                   reps: int = 1, max_weight_readout: int = 3) -> dict:
    """Grid of (kappa_processor, lambda_mp) -> (M, NL, total), using the
    PRODUCTION `experiments.run_dqrc` path (merged readout) -- this is the
    architecture DQRC actually ships, so the scan that decides on a
    lambda_mp default must use it, not the diagnostic grouped readout."""
    from . import experiments
    M_grid = np.zeros((len(kappa_grid), len(lambda_grid)))
    NL_grid = np.zeros_like(M_grid)
    Total_grid = np.zeros_like(M_grid)
    for i, kappa in enumerate(kappa_grid):
        for j, lam in enumerate(lambda_grid):
            cfg = DQRCConfig(N_M=N_M, N_P=N_P, kappa_processor=float(kappa), lambda_mp=float(lam),
                              reps_processor=reps, max_weight_readout=max_weight_readout)
            run = experiments.run_dqrc(cfg, T=T, master_seed=master_seed)
            M, NL, total, _ = ipc_MN(run.u, run.X, CFG, seed=master_seed)
            M_grid[i, j] = M
            NL_grid[i, j] = NL
            Total_grid[i, j] = total
    return {"kappa_grid": list(kappa_grid), "lambda_grid": list(lambda_grid),
            "M": M_grid, "NL": NL_grid, "Total": Total_grid}


# =============================================================================
# Phase 6 -- information-theoretic diagnostics (small systems only)
# =============================================================================

def info_theoretic_diagnostics(cfg: DQRCConfig, T_small: int, master_seed: int) -> dict:
    """Purity of M, purity of P, mutual information I(M:P), on the FINAL
    density matrix after a short trajectory (small T so the density matrix
    stays cheap: dim = 2**n_qubits_total). Uses Qiskit's own
    `partial_trace`/`entropy`/`mutual_information` -- not re-derived."""
    seeds = make_seed_bundle(master_seed)
    u = random_input(T_small, seed=seeds.dataset_seed)
    mem_qubits = list(range(1, 1 + cfg.N_M))
    proc_start = 1 + cfg.N_M
    proc_qubits = list(range(proc_start, proc_start + cfg.N_P))
    N_total = cfg.n_qubits_total

    omega = memmod.sample_memory_disorder(
        cfg.N_M, seeds.reservoir_seed, scale=(2 * np.pi if cfg.memory_mode != "swap" else cfg.omega_scale))
    proc_params = procmod.sample_processor_params(cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
                                                   disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor,
                                                   G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
    proc_layer = procmod.processor_reps_subcircuit(proc_params)
    n_taps_eff = min(cfg.n_taps, cfg.N_M, cfg.N_P)
    memory_taps = mem_qubits[-n_taps_eff:]
    processor_entry = proc_qubits[:n_taps_eff]

    qc = QuantumCircuit(N_total)
    for u_t in u:
        memmod.apply_memory_step(qc, cfg.input_qubit, mem_qubits, cfg.memory_mode, cfg.lambda_im,
                                  cfg.epsilon, cfg.omega_scale, omega, u_t)
        if cfg.lambda_mp != 0:
            ifacemod.apply_interface(qc, memory_taps, processor_entry, cfg.lambda_mp, cfg.interface_kind)
        qc.compose(proc_layer, qubits=proc_qubits, inplace=True)
    qc.save_density_matrix(label="rho")

    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    rho = DensityMatrix(np.asarray(result.data(0)["rho"]))

    all_qubits = list(range(N_total))
    other_than_mem = [q for q in all_qubits if q not in mem_qubits]
    other_than_proc = [q for q in all_qubits if q not in proc_qubits]
    other_than_both = [q for q in all_qubits if q not in mem_qubits and q not in proc_qubits]

    rho_M = partial_trace(rho, other_than_mem)
    rho_P = partial_trace(rho, other_than_proc)
    rho_MP = partial_trace(rho, other_than_both) if other_than_both != all_qubits else rho

    purity_M = float(np.real(np.trace(np.asarray(rho_M.data) @ np.asarray(rho_M.data))))
    purity_P = float(np.real(np.trace(np.asarray(rho_P.data) @ np.asarray(rho_P.data))))
    entropy_M = float(entropy(rho_M, base=2))
    entropy_P = float(entropy(rho_P, base=2))
    # I(M:P) = S(rho_M) + S(rho_P) - S(rho_MP) -- computed directly rather than via
    # qiskit's `mutual_information` helper, which requires its input to already be
    # grouped into exactly two subsystems (len(state.dims())==2); rho_MP here has
    # one subsystem per qubit, so the definition is applied by hand instead.
    entropy_MP = float(entropy(rho_MP, base=2))
    mi_MP = entropy_M + entropy_P - entropy_MP

    return {"purity_M": purity_M, "purity_P": purity_P, "entropy_M": entropy_M, "entropy_P": entropy_P,
            "mutual_information_MP": mi_MP}


# =============================================================================
# Phase 7 -- clean, resource-decoupled scans
# =============================================================================

def fixed_total_qubit_scan(N_total: int, N_M_grid, kappa_processor: float, T: int, master_seed: int, CFG,
                            reps: int = 1, max_weight_readout: int = 3) -> list:
    """N_M varies, N_P = N_total - 1 - N_M (Part 9's fair, resource-matched
    trade-off -- explicitly conflates 'more memory' with 'less processor',
    reported as such)."""
    from . import experiments
    out = []
    for n_m in N_M_grid:
        n_p = N_total - 1 - n_m
        if n_p < 1:
            continue
        cfg = DQRCConfig(N_M=n_m, N_P=n_p, kappa_processor=kappa_processor, reps_processor=reps,
                          max_weight_readout=max_weight_readout)
        run = experiments.run_dqrc(cfg, T=T, master_seed=master_seed)
        M, NL, total, _ = ipc_MN(run.u, run.X, CFG, seed=master_seed)
        out.append({"N_M": n_m, "N_P": n_p, "M": M, "NL": NL})
    return out


def fixed_processor_size_scan(N_P: int, N_M_grid, kappa_processor: float, T: int, master_seed: int, CFG,
                               reps: int = 1, max_weight_readout: int = 3) -> list:
    """N_P held FIXED, N_M increases independently -- total qubits grows, so
    this is NOT a fair resource comparison, but isolates whether the memory
    control itself changes NL (Part 7 in the follow-up spec: a clean
    dNL/d(memory) probe)."""
    from . import experiments
    out = []
    for n_m in N_M_grid:
        cfg = DQRCConfig(N_M=n_m, N_P=N_P, kappa_processor=kappa_processor, reps_processor=reps,
                          max_weight_readout=max_weight_readout)
        run = experiments.run_dqrc(cfg, T=T, master_seed=master_seed)
        M, NL, total, _ = ipc_MN(run.u, run.X, CFG, seed=master_seed)
        out.append({"N_M": n_m, "N_P": N_P, "M": M, "NL": NL})
    return out


def processor_only_g_scan(N_M_fixed_cfg: DQRCConfig, kappa_grid, T: int, master_seed: int, CFG) -> list:
    """N_M (and everything else) held fixed, ONLY kappa_processor varies --
    the clean partial-NL/partial-g probe to pair with
    `fixed_processor_size_scan`'s partial-NL/partial-m probe for an
    independent-control Jacobian (Phase 8)."""
    from . import experiments
    out = []
    for kappa in kappa_grid:
        d = dict(N_M_fixed_cfg.__dict__)
        d["kappa_processor"] = float(kappa)
        cfg = DQRCConfig(**d)
        run = experiments.run_dqrc(cfg, T=T, master_seed=master_seed)
        M, NL, total, _ = ipc_MN(run.u, run.X, CFG, seed=master_seed)
        out.append({"kappa_processor": float(kappa), "M": M, "NL": NL})
    return out


def independent_control_jacobian(base_cfg: DQRCConfig, N_total_fixed: int, m_grid_extra_N_M: int,
                                  g0: float, dg: float, T: int, master_seed: int, CFG) -> dict:
    """A Jacobian using genuinely independent controls: m = "add one more
    memory qubit while holding N_P fixed" (a discrete control -- finite
    difference over N_M=base.N_M -> base.N_M + m_grid_extra_N_M, N_P
    UNCHANGED), g = kappa_processor (continuous, N_M/N_P both unchanged).
    Unlike the FAST_MODE notebook's Section 7/8 (which traded N_M against
    N_P to hold N_total fixed, conflating 'more memory' with 'less
    processor'), dM/dm and dNL/dm here isolate the memory control alone."""
    from . import experiments

    def MN(n_m, kappa):
        d = dict(base_cfg.__dict__)
        d["N_M"] = n_m
        d["kappa_processor"] = kappa
        cfg = DQRCConfig(**d)
        run = experiments.run_dqrc(cfg, T=T, master_seed=master_seed)
        M, NL, total, _ = ipc_MN(run.u, run.X, CFG, seed=master_seed)
        return M, NL

    M0, NL0 = MN(base_cfg.N_M, g0)
    M_dm, NL_dm = MN(base_cfg.N_M + m_grid_extra_N_M, g0)
    M_dg, NL_dg = MN(base_cfg.N_M, g0 + dg)

    dM_dm = (M_dm - M0) / m_grid_extra_N_M
    dNL_dm = (NL_dm - NL0) / m_grid_extra_N_M
    dM_dg = (M_dg - M0) / dg
    dNL_dg = (NL_dg - NL0) / dg
    eps = 1e-9
    cross = (abs(dM_dg) + abs(dNL_dm)) / (abs(dM_dm) + abs(dNL_dg) + eps)
    return {"M0": M0, "NL0": NL0, "dM_dm": dM_dm, "dM_dg": dM_dg, "dNL_dm": dNL_dm, "dNL_dg": dNL_dg,
            "cross_coupling": cross, "decoupling_score": 1.0 / (1.0 + cross),
            "note": "m = +1 memory qubit at fixed N_P (N_total grows) -- NOT the fixed-N_total Jacobian; "
                    "see fixed_total_qubit_scan for that resource-matched (but conflated) version."}


# =============================================================================
# Phase 9 -- extended memory characterization, k=1..20
# =============================================================================

def memory_design_comparison(N: int, k_max: int, T: int, master_seed: int, CFG) -> dict:
    from . import memory as memmod2
    seeds = make_seed_bundle(master_seed)
    u = random_input(T, seed=seeds.dataset_seed)
    out = {}
    for mode in memmod2.MEMORY_MODES:
        run = memmod2.run_memory_register(N=N, u_seq=u, memory_mode=mode, disorder_seed=seeds.reservoir_seed)
        k, C = memmod2.delay_resolved_capacity(run, k_max=k_max, washout=CFG.washout, n_val=CFG.n_val,
                                                n_test=CFG.n_test)
        M, NL, total, _ = ipc_MN(u, run.X, CFG, seed=master_seed)
        out[mode] = {"k": k, "C": C, "C_sum": float(np.sum(C)), "IPC1": M, "NL": NL}
    delay_run = memmod2.run_quantum_delay_register(N=N, u_seq=u)
    k_d, C_d = memmod2.delay_resolved_capacity(delay_run, k_max=k_max, washout=CFG.washout, n_val=CFG.n_val,
                                                n_test=CFG.n_test)
    M_d, NL_d, total_d, _ = ipc_MN(u, delay_run.X, CFG, seed=master_seed)
    out["delay_register"] = {"k": k_d, "C": C_d, "C_sum": float(np.sum(C_d)), "IPC1": M_d, "NL": NL_d}
    return out
