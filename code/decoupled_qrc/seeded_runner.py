"""
seeded_runner.py -- V2.1 Defect 9 fix: "seed separation was demonstrated
but not propagated". `directional_dqrc.run_directional_dqrc` and
`reset_ablation.run_directional_dqrc_with_reset` both take a single
`master_seed: int` and internally re-derive term/disorder/input seeds via
the OLD `utils.make_seed_bundle` system -- the carefully-designed
`validation_utils.NestedSeeds` (7 independent streams) built for V2 was
unit-tested but never actually wired into any real circuit run (V2's own
notebook only ever called `make_nested_seeds` in a throwaway print-only
cell; every real circuit evaluation still went through the old
`master_seed: int` path).

This module is the fix: `run_directional_dqrc_seeded` takes a `NestedSeeds`
object directly and threads `.reservoir_seed` (Hamiltonian/disorder draws)
and `.input_seed` (the input sequence) through to the actual circuit build.
Both `directional_dqrc.build_directional_circuit` and
`reset_ablation.build_directional_circuit_with_reset` already consume
`seeds.reservoir_seed` duck-typed (verified by reading both functions
directly -- neither accesses any OTHER attribute of the `seeds` object),
so no changes to either tested, existing circuit builder were needed. Two
calls sharing the SAME `NestedSeeds` object are now guaranteed common
random numbers (identical Hamiltonian realization AND identical input
sequence) even as `cfg` (m/g/J/reset_period) varies -- exactly the
"plus/minus response points share all randomness" / "reset variants share
all randomness" property V1/V2 never actually enforced at the circuit
level.
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile

from .directional_dqrc import DirectionalConfig, DirectionalRun
from .reset_ablation import build_directional_circuit_with_reset
from .utils import ResourceUsage, ensure_repo_code_on_path
from .validation_utils import NestedSeeds

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input  # noqa: E402


def run_directional_dqrc_seeded(cfg: DirectionalConfig, T: int, seeds: NestedSeeds, reset_period=None,
                                 method: str = "density_matrix", use_gpu: bool = False) -> DirectionalRun:
    """Same physics/circuit as `directional_dqrc.run_directional_dqrc`
    (reset_period=None) or `reset_ablation.run_directional_dqrc_with_reset`
    (reset_period set), but driven by an explicit `NestedSeeds` object
    instead of a bare int master_seed -- the actual Defect-9 fix. NOT
    expected to be bit-identical to either legacy function at any
    particular `master_seed` int (this is a deliberate CHANGE of
    seed-derivation scheme -- `make_nested_seeds` and `make_seed_bundle`
    use different SHA256 tag strings on purpose, so V2.1 results can never
    be silently confused with V1/V2 cache entries -- see Defect 10)."""
    if method != "density_matrix":
        raise ValueError("directional circuits reset a qubit every step -- use method='density_matrix' "
                          "(docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 10).")
    u = random_input(T, seed=seeds.input_seed)
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
                               notes=f"seeded_runner reset_period={reset_period} "
                                     f"reservoir_idx={seeds.reservoir_idx} input_idx={seeds.input_idx}")
    return DirectionalRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, labels_cross=labels_cross,
                           X_mem=X_mem, X_proc=X_proc, X_cross=X_cross, mem_input_qubit=0,
                           mem_qubits=mem_qubits, ancilla_qubit=ancilla, proc_qubits=proc_qubits,
                           resources=resources)


def run_directional_dqrc_with_explicit_input(cfg: DirectionalConfig, u_seq, seeds: NestedSeeds,
                                              reset_period=None, method: str = "density_matrix",
                                              use_gpu: bool = False) -> DirectionalRun:
    """V2.2 Phase 1: identical to `run_directional_dqrc_seeded` except the
    input sequence `u_seq` is supplied by the CALLER instead of being
    derived from `seeds.input_seed` -- needed for the intervention-based
    causal-latency test, which requires two input sequences that are
    IDENTICAL except at one timestep, run on the exact same Hamiltonian/
    disorder realization (`seeds.reservoir_seed` still drives that, via
    `build_directional_circuit_with_reset`, unchanged)."""
    if method != "density_matrix":
        raise ValueError("directional circuits reset a qubit every step -- use method='density_matrix'.")
    u = np.asarray(u_seq, dtype=float)
    qc, (labels_mem, labels_proc, labels_cross), mem_qubits, ancilla, proc_qubits = \
        build_directional_circuit_with_reset(cfg, u, seeds, reset_period)

    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    T = len(u)

    X_mem = np.array([[np.real(data[f"mem_{lab}__t{t}"]) for lab in labels_mem] for t in range(T)])
    X_proc = np.array([[np.real(data[f"proc_{lab}__t{t}"]) for lab in labels_proc] for t in range(T)])
    X_cross = np.array([[np.real(data[f"cross_{lab}__t{t}"]) for lab in labels_cross] for t in range(T)])

    resources = ResourceUsage(n_qubits_physical=cfg.n_qubits_total, n_ancilla=1, circuit_depth=tqc.depth(),
                               two_qubit_gates=0, shots=0, n_features=len(labels_mem) + len(labels_proc),
                               notes=f"explicit_input reset_period={reset_period} reservoir_idx={seeds.reservoir_idx}")
    return DirectionalRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, labels_cross=labels_cross,
                           X_mem=X_mem, X_proc=X_proc, X_cross=X_cross, mem_input_qubit=0,
                           mem_qubits=mem_qubits, ancilla_qubit=ancilla, proc_qubits=proc_qubits,
                           resources=resources)


def matched_processor_params(cfg: DirectionalConfig, seeds: NestedSeeds):
    """The SAME `(term_seed, disorder_seed)` derivation
    `build_directional_circuit`/`build_directional_circuit_with_reset` use
    internally (`seeds.reservoir_seed+1`, `seeds.reservoir_seed+2`),
    exposed here so a chaos/EOC diagnostic can be built on the EXACT SAME
    Hamiltonian realization as the actual task circuit -- V2.1 test
    requirement #21 ('matched Hamiltonians between chaos and task
    simulations'). V2's own Part 9 chaos diagnostics used hardcoded
    `term_seed=0, disorder_seed=0` regardless of which seed the candidate's
    own circuit run used -- a real mismatch this function fixes by
    construction (callers can no longer accidentally pass mismatched
    seeds, since there is only one way to derive them from a given
    `(cfg, seeds)` pair)."""
    from . import directional_processor as dproc
    if cfg.g_processor is not None and cfg.J_processor is not None:
        return dproc.sample_params_gJ(cfg.N_P, cfg.g_processor, cfg.J_processor,
                                       term_seed=seeds.reservoir_seed + 1, disorder_seed=seeds.reservoir_seed + 2,
                                       reps=cfg.reps_processor)
    return dproc.sample_params(cfg.N_P, cfg.kappa_processor, term_seed=seeds.reservoir_seed + 1,
                                disorder_seed=seeds.reservoir_seed + 2, reps=cfg.reps_processor,
                                G_MAX=cfg.G_MAX, J_MAX=cfg.J_MAX)
