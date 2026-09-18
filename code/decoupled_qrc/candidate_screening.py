"""
candidate_screening.py -- V2.2 Phase 8: screens a candidate (m,g,J) using
the CORRECTED fixed-delay signed-capacity protocol (Phases 3/4), with
candidate-specific ceiling/chaos diagnostics (V2.1's own Defect-7 fix,
reused unchanged) and a real D_M(t) trajectory. This is EXPLORATORY
SCREENING, never confirmation (Rule 6) -- callers must label output
accordingly.

Because Phase 1's intervention test found `ell_causal=0` uniformly across
every feature group tested (the circuit genuinely is causal at the SAME
timestep it encodes u_t), `NL_0_signed` (the fixed tau=0 delay) is used
as Gate D's PRIMARY nonlinear-selectivity target throughout V2.2 -- not a
parameter-dependent peak delay, and not an assumed-but-unverified
"aligned" delay. Every other fixed delay is still computed and reported
(Phase 3's explicit "never collapse the tensor" rule), just not used as
the primary derivative target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backaction_trajectory import compute_backaction_trajectory
from .chaos_symmetry import symmetry_resolved_level_spacing
from .directional_dqrc import DirectionalConfig
from .ipc_decomposition import compute_ipc_decomposed, diagnose_ceiling, nl_tensor_by_fixed_delay
from .seeded_runner import run_directional_dqrc_seeded, matched_processor_params
from .validation_utils import NestedSeeds
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402
from qrc_qiskit import chrono_split  # noqa: E402


@dataclass
class ScreeningResult:
    m: float
    g: float
    J: float
    reservoir_idx: int
    input_idx: int
    M_signed: float
    M_legacy: float
    NL_by_delay_signed: dict            # {tau: signed}
    NL_by_delay_bc: dict                # {tau: clipped}
    NL_0_signed: float                  # PRIMARY (ell_causal=0, verified by Phase 1)
    ceiling_fraction: float
    ceiling_contaminated: bool
    feature_rank: int
    n_train: int
    D_M_mean: float
    D_M_max: float
    r_chaos: float
    r_chaos_symmetry_used: object
    valid: bool
    rejection_reasons: list = field(default_factory=list)


def screen_candidate(m: float, g: float, J: float, seeds: NestedSeeds, *, N_M: int, N_P: int, theta: float,
                      phi: float, ap_kind: str, T: int, washout: int, n_val: int, n_test: int, max_delay: int,
                      max_degree: int, max_targets_per_degree: int, n_surrogates: int,
                      nontriviality_threshold: float = 0.05, backaction_threshold: float = 0.1,
                      ceiling_threshold: float = 0.95, min_sector_size: int = 8, T_backaction: int = 12
                      ) -> ScreeningResult:
    reasons = []
    cfg = DirectionalConfig(memory_variant="protected_integrable", N_M=N_M, N_P=N_P, g_processor=g,
                             J_processor=J, epsilon_M=m, theta=theta, phi=phi, ap_kind=ap_kind)
    run = run_directional_dqrc_seeded(cfg, T=T, seeds=seeds, reset_period=None)
    gap = max_delay + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)

    decomp = compute_ipc_decomposed(run.u, run.X_combined, train, val, test, max_delay=max_delay,
                                     max_degree=max_degree, max_targets_per_degree=max_targets_per_degree,
                                     n_surrogates=n_surrogates, seed=seeds.reservoir_seed)
    nl_tensor = nl_tensor_by_fixed_delay(decomp.records, max_delay)
    nl_signed = {tau: entry["signed"] for tau, entry in nl_tensor.items()}
    nl_bc = {tau: entry["bc"] for tau, entry in nl_tensor.items()}
    NL_0_signed = nl_signed[0]

    ceil = diagnose_ceiling(decomp, run.X_combined, train, contamination_threshold=ceiling_threshold)
    if ceil.ceiling_contaminated:
        reasons.append("ceiling_contaminated")

    from .feature_analysis import diagnose_feature_group
    fdiag = diagnose_feature_group(run.X_combined[train])
    if ceil.n_train < 2 * max(fdiag.effective_rank, 1.0):
        reasons.append("insufficient_sample_to_rank_ratio")

    params = matched_processor_params(cfg, seeds)
    U1 = msc.single_layer_unitary_mixed(params.N_p, params.g, params.terms, params.couplings,
                                         params.paulis, params.bias_z)
    chaos = symmetry_resolved_level_spacing(U1, params.N_p, min_sector_size=min_sector_size)

    cfg_off = DirectionalConfig(memory_variant="protected_integrable", N_M=N_M, N_P=N_P, g_processor=g,
                                 J_processor=J, epsilon_M=m, theta=0.0, phi=0.0, ap_kind=ap_kind)
    traj = compute_backaction_trajectory(cfg, cfg_off, T_backaction, seeds)
    if traj.mean_trace_distance > backaction_threshold:
        reasons.append("backaction_above_threshold")

    if NL_0_signed < nontriviality_threshold:
        reasons.append("nl0_below_nontriviality_threshold")
    if not np.isfinite(NL_0_signed) or not np.isfinite(decomp.M_long_signed):
        reasons.append("non_finite_metric")

    return ScreeningResult(
        m=m, g=g, J=J, reservoir_idx=seeds.reservoir_idx, input_idx=seeds.input_idx,
        M_signed=decomp.M_long_signed, M_legacy=decomp.M_long, NL_by_delay_signed=nl_signed,
        NL_by_delay_bc=nl_bc, NL_0_signed=NL_0_signed, ceiling_fraction=ceil.fraction_of_ceiling,
        ceiling_contaminated=ceil.ceiling_contaminated, feature_rank=ceil.feature_rank, n_train=ceil.n_train,
        D_M_mean=traj.mean_trace_distance, D_M_max=traj.max_trace_distance, r_chaos=chaos.r_mean,
        r_chaos_symmetry_used=chaos.symmetry_used, valid=(len(reasons) == 0), rejection_reasons=reasons,
    )


def screening_score(results: list) -> list:
    """Ranks candidates by CORRECTED response selectivity proxies
    available at screening time (Phase 8's 'rank by corrected response
    selectivity, not static capacity alone'): nontrivial NL_0, low
    back-action, EOC proximity, and memory-vs-NL0 BALANCE (neither metric
    dominating the other by an extreme ratio is a necessary, not
    sufficient, precondition for a later orthogonal response -- a crude
    but transparent proxy used ONLY to order candidates for the expensive
    Phase 9 stencil, never as a gate decision by itself). Returns
    `results` sorted best-first, each augmented with `.score_components`
    and `.score` attributes for full transparency."""
    valid = [r for r in results if r.valid]
    if not valid:
        return []

    def norm(vals):
        vals = np.asarray(vals, dtype=float)
        lo, hi = np.min(vals), np.max(vals)
        return (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

    nl0 = norm([r.NL_0_signed for r in valid])
    low_backaction = norm([-r.D_M_mean for r in valid])
    eoc_proximity = norm([-abs(r.r_chaos - 0.55) for r in valid])
    balance = norm([-abs(np.log((abs(r.M_signed) + 1e-6) / (abs(r.NL_0_signed) + 1e-6))) for r in valid])

    scores = 0.4 * nl0 + 0.25 * low_backaction + 0.2 * eoc_proximity + 0.15 * balance
    for r, s, c1, c2, c3, c4 in zip(valid, scores, nl0, low_backaction, eoc_proximity, balance):
        r.score = float(s)
        r.score_components = {"nl0": float(c1), "low_backaction": float(c2), "eoc_proximity": float(c3),
                               "balance": float(c4)}
    order = np.argsort(-scores)
    return [valid[i] for i in order]
