"""
candidate_eval.py -- V2.1 orchestration layer fixing Defects 4, 7, 10
together, since they all revolve around the same object: ONE evaluation of
ONE (m,g,J) candidate point.

Defect 7 fix: `diagnose_ceiling` is called HERE, on THIS candidate's own
(X, train) -- never on stale variables left over from a previous cell (the
exact V2 bug: Part 4's ceiling diagnostic silently reused the standalone
processor's X/train instead of any actual DQRC candidate's).

Defect 4 fix: `evaluate_candidate` returns a `valid: bool` plus an
itemized `rejection_reasons: list[str]` BEFORE any ranking/selection
happens -- `select_candidates` (below) only ever ranks points that passed
every hard filter, via a multi-objective score, never by "max NL_instant"
alone.

Defect 10 fix: `evaluate_candidate_cached` is a single `@utils.cached`
function whose FULL argument list is the complete set of scientifically
relevant fields (circuit config, IPC config, gate thresholds, seeds,
schema version, provenance) -- nothing is read from a notebook global. The
`cached()` decorator hashes `repr(args)`, so ANY change to ANY of these
arguments produces a different cache key by construction.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field, asdict

import numpy as np
import qiskit
import qiskit_aer

from .causal_latency import causal_latency_profile
from .chaos_symmetry import symmetry_resolved_level_spacing
from .directional_dqrc import DirectionalConfig
from .feature_analysis import diagnose_feature_group
from .interfaces_advanced import memory_disturbance
from .ipc_decomposition import diagnose_ceiling
from .seeded_runner import run_directional_dqrc_seeded, matched_processor_params
from .utils import cached, ensure_repo_code_on_path
from .validation_utils import NestedSeeds, ControlRange, stencil_margin_ok

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit.quantum_info import DensityMatrix  # noqa: E402
from qrc_qiskit import chrono_split, make_simulator, random_input  # noqa: E402

CACHE_SCHEMA_VERSION = "v2.1.0"


def compute_provenance_tag() -> str:
    """A single, stable string bundling every 'external' fact Defect 10
    requires in the cache key: package versions, git commit, schema
    version. Computed once and passed EXPLICITLY into every cached call
    (never read from a global at cache-lookup time) so the cache key
    itself captures it."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
    return (f"schema={CACHE_SCHEMA_VERSION}|git={commit}|qiskit={qiskit.__version__}|"
            f"aer={qiskit_aer.__version__}|py={platform.python_version()}|precision=complex128")


@dataclass
class CandidatePointResult:
    m: float
    g: float
    J: float
    m_tilde: float
    g_tilde: float
    J_tilde: float
    M_bc: float
    NL_instant_bc: float
    NL_temporal_bc: float
    NL_legacy_bc: float
    M_legacy: float
    NL_instant_legacy: float
    NL_temporal_legacy: float
    NL_legacy_legacy: float
    ell_0: int
    NL_local_bc: float
    NL_local_legacy: float
    ceiling_fraction: float
    ceiling_contaminated: bool
    target_count_ceiling: float
    rank_ceiling: float
    sample_ceiling: float
    feature_rank: int
    effective_rank: float
    ridge_dof: float
    n_train: int
    D_M: float
    r_chaos: float
    r_chaos_symmetry_used: object
    stencil_ok: dict
    valid: bool
    rejection_reasons: list = field(default_factory=list)


def _stencil_check(m_tilde, g_tilde, J_tilde, h_m_tilde, h_g_tilde, h_J_tilde, k=2) -> dict:
    return {
        "m": stencil_margin_ok(m_tilde, h_m_tilde, k=k),
        "g": stencil_margin_ok(g_tilde, h_g_tilde, k=k),
        "J": stencil_margin_ok(J_tilde, h_J_tilde, k=k),
    }


def evaluate_candidate(
    m: float, g: float, J: float, seeds: NestedSeeds, *,
    N_M: int, N_P: int, theta: float, phi: float, ap_kind: str,
    m_range: ControlRange, g_range: ControlRange, J_range: ControlRange,
    h_m_tilde: float, h_g_tilde: float, h_J_tilde: float,
    T: int, washout: int, n_val: int, n_test: int, max_delay: int, max_degree: int,
    max_targets_per_degree: int, n_surrogates: int,
    nontriviality_threshold: float = 0.1, backaction_threshold: float = 0.1,
    ceiling_threshold: float = 0.95, min_sector_size: int = 8, stencil_k: int = 2,
    provenance_tag: str = "",
) -> CandidatePointResult:
    """ONE full candidate-point evaluation: circuit run, order-delay
    decomposition (continuous + legacy), candidate-SPECIFIC ceiling
    diagnostic (Defect 7), causal-latency-aligned NL_local, symmetry-
    resolved chaos diagnostic on the MATCHED Hamiltonian, back-action, and
    the stencil-margin/hard-validity-filter checks needed before this
    point may ever be ranked (Defect 4)."""
    reasons = []
    m_tilde, g_tilde, J_tilde = m_range.to_dimensionless(m), g_range.to_dimensionless(g), J_range.to_dimensionless(J)
    stencil_ok = _stencil_check(m_tilde, g_tilde, J_tilde, h_m_tilde, h_g_tilde, h_J_tilde, k=stencil_k)
    if not all(stencil_ok.values()):
        reasons.append("stencil_margin_violated")

    cfg = DirectionalConfig(memory_variant="protected_integrable", N_M=N_M, N_P=N_P, g_processor=g,
                             J_processor=J, epsilon_M=m, theta=theta, phi=phi, ap_kind=ap_kind)
    run = run_directional_dqrc_seeded(cfg, T=T, seeds=seeds, reset_period=None)
    gap = max_delay + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)

    latency = causal_latency_profile(run.u, run.X_combined, train, val, test, max_delay=max_delay,
                                      max_degree=max_degree, max_targets_per_degree=max_targets_per_degree,
                                      n_surrogates=n_surrogates, seed=seeds.reservoir_seed)
    decomp = latency.decomp
    ceil = diagnose_ceiling(decomp, run.X_combined, train, contamination_threshold=ceiling_threshold)
    if ceil.ceiling_contaminated:
        reasons.append("ceiling_contaminated")
    fdiag = diagnose_feature_group(run.X_combined[train])
    # sample-adequacy is checked against EFFECTIVE rank (participation-ratio-style, exp(spectral
    # entropy)), not the raw numerical rank `diagnose_ceiling` uses -- with hundreds of highly
    # correlated Pauli-expectation features, numerical rank is dominated by near-degenerate
    # directions carrying negligible real information, while effective rank tracks the TRUE number
    # of independent directions the ridge fit actually has to resolve (Phase 2's own explicit
    # distinction between reporting numerical rank, effective rank, AND ridge EDOF separately).
    if ceil.n_train < 2 * max(fdiag.effective_rank, 1.0):
        reasons.append("insufficient_sample_to_rank_ratio")

    params = matched_processor_params(cfg, seeds)
    U1 = msc.single_layer_unitary_mixed(params.N_p, params.g, params.terms, params.couplings,
                                         params.paulis, params.bias_z)
    chaos = symmetry_resolved_level_spacing(U1, params.N_p, min_sector_size=min_sector_size)
    if np.isnan(chaos.r_mean):
        reasons.append("chaos_diagnostic_not_evaluable")

    cfg_off = DirectionalConfig(memory_variant="protected_integrable", N_M=N_M, N_P=N_P, g_processor=g,
                                 J_processor=J, epsilon_M=m, theta=0.0, phi=0.0, ap_kind=ap_kind)
    D_M = _single_point_backaction(cfg, cfg_off, seeds)
    if D_M > backaction_threshold:
        reasons.append("backaction_above_threshold")

    if latency.NL_local_bc < nontriviality_threshold:
        reasons.append("nl_local_below_nontriviality_threshold")
    if not np.isfinite(decomp.M_long_bc) or not np.isfinite(latency.NL_local_bc):
        reasons.append("non_finite_metric")

    return CandidatePointResult(
        m=m, g=g, J=J, m_tilde=m_tilde, g_tilde=g_tilde, J_tilde=J_tilde,
        M_bc=decomp.M_long_bc, NL_instant_bc=decomp.NL_instant_bc, NL_temporal_bc=decomp.NL_temporal_bc,
        NL_legacy_bc=decomp.NL_legacy_bc, M_legacy=decomp.M_long, NL_instant_legacy=decomp.NL_instant,
        NL_temporal_legacy=decomp.NL_temporal, NL_legacy_legacy=decomp.NL_legacy,
        ell_0=latency.ell_0, NL_local_bc=latency.NL_local_bc, NL_local_legacy=latency.NL_local,
        ceiling_fraction=ceil.fraction_of_ceiling, ceiling_contaminated=ceil.ceiling_contaminated,
        target_count_ceiling=ceil.M_max_target_count, rank_ceiling=ceil.M_max_rank,
        sample_ceiling=ceil.M_max_sample, feature_rank=ceil.feature_rank,
        effective_rank=fdiag.effective_rank, ridge_dof=fdiag.ridge_dof, n_train=ceil.n_train,
        D_M=D_M, r_chaos=chaos.r_mean, r_chaos_symmetry_used=chaos.symmetry_used, stencil_ok=stencil_ok,
        valid=(len(reasons) == 0), rejection_reasons=reasons,
    )


evaluate_candidate_cached = cached("gj_v2_1")(evaluate_candidate)


def _single_point_backaction(cfg_on: DirectionalConfig, cfg_off: DirectionalConfig, seeds: NestedSeeds,
                              T_small: int = 12) -> float:
    """A single-timestep (final-state) D_M, reusing
    `interfaces_advanced.memory_disturbance` -- the cheaper check used
    inside the discovery/hard-filter loop; `backaction_trajectory.py`'s
    full D_M(t) is reserved for the (more expensive) confirmed-candidate
    stage."""
    from .reset_ablation import build_directional_circuit_with_reset
    u_small = random_input(T_small, seed=seeds.input_seed)
    qc_on, _, mem_qubits, _, _ = build_directional_circuit_with_reset(cfg_on, u_small, seeds, None)
    qc_off, _, _, _, _ = build_directional_circuit_with_reset(cfg_off, u_small, seeds, None)
    qc_on.save_density_matrix(label="rho")
    qc_off.save_density_matrix(label="rho")
    sim = make_simulator(method="density_matrix")
    all_qubits = list(range(qc_on.num_qubits))
    rho_on = DensityMatrix(np.asarray(sim.run(transpile(qc_on, sim, optimization_level=1), shots=1)
                                       .result().data(0)["rho"]))
    rho_off = DensityMatrix(np.asarray(sim.run(transpile(qc_off, sim, optimization_level=1), shots=1)
                                        .result().data(0)["rho"]))
    return memory_disturbance(rho_on, rho_off, mem_qubits, all_qubits)["trace_distance"]


def select_candidates(results: list, top_k: int = 5) -> list:
    """Defect 4 fix: ranks ONLY the points with `valid=True` (every hard
    filter passed), via a multi-objective score combining memory strength,
    local nonlinear strength, low back-action, chaos proximity to the
    known EOC band (~0.53 Poisson to ~0.6 COE/CUE -- scored by closeness
    to 0.55 as a simple proxy), and derivative-friendliness (favors points
    with more stencil margin, i.e. further from any boundary) -- NEVER a
    ranking by raw max(NL_instant) alone."""
    valid = [r for r in results if r.valid]
    if not valid:
        return []

    def _norm(vals):
        vals = np.asarray(vals, dtype=float)
        lo, hi = np.min(vals), np.max(vals)
        return (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

    nl_local = _norm([r.NL_local_bc for r in valid])
    m_strength = _norm([r.M_bc for r in valid])
    low_backaction = _norm([-r.D_M for r in valid])
    eoc_proximity = _norm([-abs(r.r_chaos - 0.55) for r in valid])
    margin = _norm([min(r.m_tilde, 1 - r.m_tilde, r.g_tilde, 1 - r.g_tilde, r.J_tilde, 1 - r.J_tilde)
                     for r in valid])

    score = 0.35 * nl_local + 0.2 * m_strength + 0.2 * low_backaction + 0.15 * eoc_proximity + 0.1 * margin
    order = np.argsort(-score)
    return [valid[i] for i in order[:top_k]]
