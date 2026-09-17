"""
processor.py -- the EOC nonlinear processor subsystem P (Part 3). Built on
the finalized mixed-SYK layer (`mixed_syk_core.mixed_layer`,
`eoc_config.sample_syk4_supports`) restricted to the processor's own qubits,
never redefining EOC arbitrarily.

Two INDEPENDENT chaos diagnostics are computed fresh, on the processor's own
kappa grid, rather than trusting `eoc_config.NOTEBOOK4_QELM_EOC_KAPPA=0.960`
as physics-only truth -- the audit found that value was selected by NARMA2
task performance on notebook-4-internal data (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md
sections 3/6), so DQRC reproduces its own EOC diagnostic rather than
inheriting that number blindly:

    1. operator_entanglement  -- half-chain operator entanglement entropy of
       the single-layer unitary (`mixed_syk_core.operator_entanglement`).
    2. level_spacing_ratio    -- mean adjacent-gap ratio <r> of the T-step
       unitary's eigenphases, compared against Poisson/COE/CUE reference
       ensembles (`mixed_syk_core.level_spacing_ratio` /
       `sample_reference_r_statistics`).

`g_processor` in the task spec corresponds to this repo's established
`kappa` interpolation parameter (SYK4-like <-> SYK2-like, via
`mixed_syk_core.kappa_to_gJ`) -- named `kappa_processor` here to avoid
confusion with the OLDER, unrelated single-scalar `g` in `qrc_qiskit.py`'s
non-EOC lineage (docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from .utils import ensure_repo_code_on_path, cached

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402
import eoc_config as ec  # noqa: E402
from qrc_qiskit import ReservoirConfig, make_simulator, random_input  # noqa: E402


@dataclass
class ProcessorParams:
    N_p: int
    kappa_processor: float  # derived (= g/J when G_MAX==J_MAX, see docs/DQRC_GJ_EOC_AUDIT.md Q5)
                             # for gJ-direct construction, NOT an independent input there
    g: float
    J: float
    reps: int
    terms: list
    couplings: np.ndarray
    paulis: np.ndarray
    bias_z: np.ndarray


def gJ_to_kappa(g: float, J: float) -> float:
    """Derived quantity only (docs/DQRC_GJ_EOC_AUDIT.md Q5): kappa = g/J,
    exactly equal to `mixed_syk_core.kappa_to_gJ`'s own kappa when
    G_MAX==J_MAX (true of every config in this project). Not used to
    constrain (g,J) -- purely for reporting/backward-compatible bookkeeping
    when a caller supplies (g,J) directly instead of kappa."""
    return float(g / J) if J != 0 else float("inf")


def sample_processor_params_gJ(N_p: int, g: float, J: float, term_seed: int = 0,
                                disorder_seed: int = 0, reps: int = 2) -> ProcessorParams:
    """Part 2: g and J supplied INDEPENDENTLY -- no `kappa_to_gJ` call, no
    constraint that (g,J) lie on the anti-diagonal line
    `g/G_MAX + J/J_MAX = 1` that every kappa-based scan in this project has
    been implicitly confined to (docs/DQRC_GJ_EOC_AUDIT.md Q7). This is the
    SAME underlying construction `sample_processor_params` uses (identical
    `sample_syk4_supports` call, identical `ReservoirConfig` disorder
    sampling) -- `sample_processor_params` is now a thin wrapper around this
    function, not a separate implementation, so nothing about the finalized
    EOC processor's own physics changes."""
    n_terms = msc.default_n_sparse_terms(N_p)
    supports = ec.sample_syk4_supports(N_p, n_terms, term_seed, J)
    cfg = ReservoirConfig(N=N_p, g=0.0, reps=reps, seed=disorder_seed)
    bias_z, _ = cfg.sample_disorder()
    return ProcessorParams(N_p=N_p, kappa_processor=gJ_to_kappa(g, J), g=g, J=J, reps=reps,
                            terms=supports["terms"], couplings=supports["couplings"],
                            paulis=supports["paulis"], bias_z=bias_z)


def sample_processor_params(N_p: int, kappa_processor: float, term_seed: int = 0,
                             disorder_seed: int = 0, reps: int = 2,
                             G_MAX: float = 0.6, J_MAX: float = 0.6) -> ProcessorParams:
    """Sample the processor's own mixed-SYK term set, using the SAME
    support-count fix as the finalized SCIENCE_CONFIG (`eoc_config.
    sample_syk4_supports`, not the raw `mixed_syk_core.sample_syk4_terms` +
    `sample_syk4_couplings` pattern the audit flagged as latently buggy at
    small N). UNCHANGED signature/behavior (backward compatible) -- now
    implemented by deriving (g,J) via `kappa_to_gJ` and delegating to
    `sample_processor_params_gJ`."""
    g, J = msc.kappa_to_gJ(kappa_processor, G_MAX, J_MAX)
    params = sample_processor_params_gJ(N_p, g, J, term_seed=term_seed, disorder_seed=disorder_seed, reps=reps)
    params.kappa_processor = kappa_processor  # the CALLER's literal kappa, not the derived g/J round-trip
    return params


def processor_layer_subcircuit(params: ProcessorParams) -> QuantumCircuit:
    """ONE mixed-layer application on a bare N_p-qubit circuit (local qubit
    indices 0..N_p-1) -- meant to be `compose`d onto the processor's actual
    qubits within a larger DQRC circuit (interface.py does this), which is
    the standard Qiskit-native way to remap a subcircuit's local qubit
    indices onto a global register without hand-rolling index arithmetic."""
    qc = QuantumCircuit(params.N_p)
    msc.mixed_layer(qc, params.N_p, params.g, params.terms, params.couplings,
                     params.paulis, params.bias_z)
    return qc


def processor_reps_subcircuit(params: ProcessorParams) -> QuantumCircuit:
    """`params.reps` copies of `processor_layer_subcircuit`, composed once."""
    layer = processor_layer_subcircuit(params)
    qc = QuantumCircuit(params.N_p)
    for _ in range(params.reps):
        qc.compose(layer, inplace=True)
    return qc


# =============================================================================
# Independent EOC/chaos diagnostics (reused verbatim from mixed_syk_core --
# not redefined -- but computed FRESH on the processor's own kappa grid).
# =============================================================================

def processor_chaos_diagnostics(params: ProcessorParams, n_ref_trials: int = 20,
                                 ref_seed: int = 0) -> dict:
    """Two task-agnostic chaos indicators for the processor's step unitary
    U1^reps, plus Poisson/COE/CUE reference statistics for the level-spacing
    ratio at the same Hilbert-space dimension, so <r> can be READ relative to
    the known integrable/chaotic limits rather than in isolation."""
    U_step = msc.step_unitary_mixed(params.N_p, params.g, params.terms, params.couplings,
                                     params.paulis, params.reps, params.bias_z)
    U1 = msc.single_layer_unitary_mixed(params.N_p, params.g, params.terms, params.couplings,
                                         params.paulis, params.bias_z)
    op_ent = msc.operator_entanglement(U1, params.N_p)
    r_mean = msc.level_spacing_ratio(U_step)
    d = 2 ** params.N_p
    refs = msc.sample_reference_r_statistics(d, trials=n_ref_trials, seed=ref_seed)
    return {
        "kappa_processor": params.kappa_processor, "g": params.g, "J": params.J,
        "operator_entanglement": op_ent, "operator_entanglement_max": np.log(2 ** (params.N_p // 2)),
        "level_spacing_ratio": r_mean,
        "r_poisson_mean": refs["poisson"][0], "r_poisson_std": refs["poisson"][1],
        "r_coe_mean": refs["coe"][0], "r_coe_std": refs["coe"][1],
        "r_cue_mean": refs["cue"][0], "r_cue_std": refs["cue"][1],
    }


def eoc_scan(N_p: int, kappa_grid, term_seed: int = 0, disorder_seed: int = 0, reps: int = 2,
             G_MAX: float = 0.6, J_MAX: float = 0.6, n_ref_trials: int = 20) -> list:
    """Diagnostics across a kappa grid -- DQRC's OWN, independently derived
    processor EOC scan (never re-using the stored notebook-4 kappa=0.960
    value as if it were this processor's own crossover point; N_p here is
    typically smaller than notebook 4's N=6, so the crossover location need
    not coincide numerically -- that is the point of re-deriving it)."""
    out = []
    for kappa in kappa_grid:
        params = sample_processor_params(N_p, kappa, term_seed, disorder_seed, reps, G_MAX, J_MAX)
        out.append(processor_chaos_diagnostics(params, n_ref_trials=n_ref_trials))
    return out


def find_eoc_kappa(scan_results: list) -> float:
    """DQRC's own EOC-point estimate: the kappa whose <r> is closest to the
    midpoint between the Poisson and COE reference means (the standard
    'crossover' operational definition -- not a task-performance optimum,
    unlike the audit's finding about the repo's existing kappa=0.960)."""
    poisson = np.mean([r["r_poisson_mean"] for r in scan_results])
    coe = np.mean([r["r_coe_mean"] for r in scan_results])
    midpoint = 0.5 * (poisson + coe)
    diffs = [abs(r["level_spacing_ratio"] - midpoint) for r in scan_results]
    return scan_results[int(np.argmin(diffs))]["kappa_processor"]


# =============================================================================
# Standalone processor run (for Part 15 Section 5's "nonlinear IPC vs
# g_processor" plot) -- driven directly like a small monolithic reservoir.
# =============================================================================

def run_processor_standalone_gJ(N_p: int, g: float, J: float, T: int, reps: int = 2, max_weight: int = 3,
                                 term_seed: int = 0, disorder_seed: int = 0, input_seed: int = 0,
                                 method: str = "density_matrix", use_gpu: bool = False):
    """Part 2/8: standalone processor with g and J supplied INDEPENDENTLY --
    no `kappa_to_gJ` call, so (g,J) need not lie on the anti-diagonal line
    every kappa-based scan has been confined to
    (docs/DQRC_GJ_EOC_AUDIT.md Q7). Identical circuit-building/execution to
    `run_processor_standalone`, which is now a thin wrapper around this."""
    cfg = ReservoirConfig(N=N_p, g=0.0, reps=reps, input_qubit=0, seed=disorder_seed)
    u = random_input(T, seed=input_seed)
    labels, X, info, _ = msc.run_reservoir_mixed(cfg, u, g=g, J=J, reps=reps, term_seed=term_seed,
                                                  method=method, use_gpu=use_gpu, max_weight=max_weight)
    return labels, X, u, info


def run_processor_standalone(N_p: int, kappa_processor: float, T: int, reps: int = 2,
                              max_weight: int = 3, term_seed: int = 0, disorder_seed: int = 0,
                              input_seed: int = 0, method: str = "density_matrix",
                              G_MAX: float = 0.6, J_MAX: float = 0.6, use_gpu: bool = False):
    """The processor characterized in isolation, driven by its OWN input
    qubit (same recurrent encoding convention as the memory register and the
    monolithic baseline) -- lets Part 15 Section 5 plot nonlinear IPC purely
    as a function of `kappa_processor`, decoupled from any memory-register
    effects (those appear only once interface.py couples the two).
    UNCHANGED signature/behavior (backward compatible); now delegates to
    `run_processor_standalone_gJ` after the same `kappa_to_gJ` conversion."""
    g, J = msc.kappa_to_gJ(kappa_processor, G_MAX, J_MAX)
    return run_processor_standalone_gJ(N_p, g, J, T, reps=reps, max_weight=max_weight, term_seed=term_seed,
                                        disorder_seed=disorder_seed, input_seed=input_seed, method=method,
                                        use_gpu=use_gpu)
