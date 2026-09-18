"""
ipc_decomposition.py -- Part 3/4 of the V2 validation spec: separates the
scalar legacy NL (which mixes temporal memory and nonlinear transformation)
into instantaneous vs. temporal nonlinear capacity, and diagnoses whether
M is hitting a target-count/rank/finite-horizon ceiling.

Built entirely on `ipc.compute_ipc_detailed`'s per-profile
`ProfileCapacity` records -- no new capacity computation, only new
aggregation/diagnosis of numbers `ipc.py` already produces per target.

Definitions (exactly as specified):
    M_long        = sum_{tau>=0} C_{1,tau}                       (all degree-1 targets)
    NL_instant    = sum_{d=2}^{6} C_{d, delays=(0,)}              (single-pair, delay=0 only)
    NL_temporal   = sum_{d=2}^{6} sum_{profiles with max_delay>0} C_{d,profile}
    NL_legacy     = sum_{d=2}^{6} IPC_d                            (unchanged from ipc.compute_ipc)

A degree-d profile is "instantaneous" ONLY if it is the single-delay
profile L_d(v_t) (one pair, delay 0) -- any profile that touches a delay>0,
even combined with a delay-0 factor, requires genuine memory and is
counted as temporal (Part 3's explicit "do not misclassify cross-delay
nonlinear targets as instantaneous" requirement).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ipc import compute_ipc_detailed, ProfileCapacity, DEFAULT_ALPHAS


def _bias_corrected(r: "ProfileCapacity") -> float:
    """max(0, raw_capacity - null_mean) -- a CONTINUOUS, non-thresholded
    per-profile capacity (V2.1 Defect 5). Unlike `r.capacity` (which is
    exactly 0 below the `mean + z*std` significance cutoff and exactly
    `raw_capacity` above it -- a hard discontinuity that must never be
    finite-differenced), this varies smoothly with the underlying signal
    strength and is the PRIMARY metric V2.1 uses for response-surface
    derivatives. It can still legitimately go negative-then-clip (a raw
    score below its own null mean, e.g. pure noise) -- clipping at 0 keeps
    it non-negative like a real capacity without introducing a hard jump
    (the clip only bites when the true value is statistically
    indistinguishable from 0 anyway). V2.2 Phase 4: this remains available
    as a DESCRIPTIVE non-negative metric, but is no longer the primary
    derivative metric -- see `_signed` below."""
    return max(0.0, r.raw_capacity - r.null_mean)


def _signed(r: "ProfileCapacity") -> float:
    """raw_capacity - null_mean, UNCLIPPED. V2.2 Phase 4: `_bias_corrected`
    (clipped at 0) has a KINK in its derivative exactly where a target
    crosses its own null mean -- differentiating it near that crossing is
    still technically well-defined almost everywhere but the kink itself
    is a real non-smoothness a finite-difference stencil straddling it
    would silently mis-estimate. `_signed` has no such kink (it is exactly
    linear in `raw_capacity`) and is the PRIMARY metric V2.2 uses for every
    response-surface/derivative calculation; `_bias_corrected` (clipped)
    is kept only as a descriptive non-negative summary."""
    return r.raw_capacity - r.null_mean


@dataclass
class IPCDecomposition:
    records: list                  # list[ProfileCapacity]
    M_long: float
    NL_instant: float
    NL_temporal: float
    NL_legacy: float
    total_legacy: float
    by_degree_delay: dict          # {(degree, max_delay): capacity_sum} -- for the C_{d,tau} heatmap
    n_instant_profiles: dict       # {degree: n tested at delay=0}
    n_temporal_profiles: dict      # {degree: n tested with max_delay>0}
    max_delay_used: int
    n_train: int
    config: dict = field(default_factory=dict)
    # --- V2.1 Defect 5: continuous (non-thresholded) companions to every
    # legacy field above, PLUS the raw/null components they're built from.
    # Legacy fields (M_long, NL_instant, NL_temporal, NL_legacy) are
    # UNCHANGED -- these are ADDITIONS, not replacements.
    M_long_raw: float = 0.0
    M_long_null: float = 0.0
    M_long_bc: float = 0.0
    NL_instant_raw: float = 0.0
    NL_instant_null: float = 0.0
    NL_instant_bc: float = 0.0
    NL_temporal_raw: float = 0.0
    NL_temporal_null: float = 0.0
    NL_temporal_bc: float = 0.0
    NL_legacy_raw: float = 0.0
    NL_legacy_null: float = 0.0
    NL_legacy_bc: float = 0.0
    # --- V2.2 Phase 4: SIGNED (unclipped raw-null) companions -- the PRIMARY
    # derivative/response-surface metric from V2.2 onward. Never clipped, so
    # never has the "_bc" fields' kink at zero.
    M_long_signed: float = 0.0
    NL_instant_signed: float = 0.0
    NL_temporal_signed: float = 0.0
    NL_legacy_signed: float = 0.0


def compute_ipc_decomposed(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                            max_delay: int = 8, max_degree: int = 6, max_targets_per_degree: int = 25,
                            n_surrogates: int = 8, significance_z: float = 2.0, alphas=DEFAULT_ALPHAS,
                            seed: int = 0, always_include_single_delays: bool = False) -> IPCDecomposition:
    records, was_capped, n_tested = compute_ipc_detailed(
        u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
        max_targets_per_degree=max_targets_per_degree, n_surrogates=n_surrogates,
        significance_z=significance_z, alphas=alphas, seed=seed,
        always_include_single_delays=always_include_single_delays)

    M_long = sum(r.capacity for r in records if r.degree == 1)

    instant_records = [r for r in records if r.degree >= 2 and r.delays == (0,)]
    temporal_records = [r for r in records if r.degree >= 2 and r.max_delay > 0]
    legacy_records = [r for r in records if r.degree >= 2]
    m_records = [r for r in records if r.degree == 1]

    NL_instant = sum(r.capacity for r in instant_records)
    NL_temporal = sum(r.capacity for r in temporal_records)
    NL_legacy = sum(r.capacity for r in legacy_records)
    total_legacy = M_long + NL_legacy

    by_degree_delay = {}
    for r in records:
        key = (r.degree, r.max_delay)
        by_degree_delay[key] = by_degree_delay.get(key, 0.0) + r.capacity

    n_instant = {d: sum(1 for r in records if r.degree == d and r.delays == (0,)) for d in range(2, max_degree + 1)}
    n_temporal = {d: sum(1 for r in records if r.degree == d and r.max_delay > 0) for d in range(2, max_degree + 1)}

    return IPCDecomposition(
        records=records, M_long=M_long, NL_instant=NL_instant, NL_temporal=NL_temporal,
        NL_legacy=NL_legacy, total_legacy=total_legacy, by_degree_delay=by_degree_delay,
        n_instant_profiles=n_instant, n_temporal_profiles=n_temporal, max_delay_used=max_delay,
        n_train=len(train),
        config=dict(max_delay=max_delay, max_degree=max_degree, max_targets_per_degree=max_targets_per_degree,
                    n_surrogates=n_surrogates, significance_z=significance_z),
        M_long_raw=sum((r.raw_capacity for r in m_records), 0.0),
        M_long_null=sum((r.null_mean for r in m_records), 0.0),
        M_long_bc=sum((_bias_corrected(r) for r in m_records), 0.0),
        NL_instant_raw=sum((r.raw_capacity for r in instant_records), 0.0),
        NL_instant_null=sum((r.null_mean for r in instant_records), 0.0),
        NL_instant_bc=sum((_bias_corrected(r) for r in instant_records), 0.0),
        NL_temporal_raw=sum((r.raw_capacity for r in temporal_records), 0.0),
        NL_temporal_null=sum((r.null_mean for r in temporal_records), 0.0),
        NL_temporal_bc=sum((_bias_corrected(r) for r in temporal_records), 0.0),
        NL_legacy_raw=sum((r.raw_capacity for r in legacy_records), 0.0),
        NL_legacy_null=sum((r.null_mean for r in legacy_records), 0.0),
        NL_legacy_bc=sum((_bias_corrected(r) for r in legacy_records), 0.0),
        M_long_signed=sum((_signed(r) for r in m_records), 0.0),
        NL_instant_signed=sum((_signed(r) for r in instant_records), 0.0),
        NL_temporal_signed=sum((_signed(r) for r in temporal_records), 0.0),
        NL_legacy_signed=sum((_signed(r) for r in legacy_records), 0.0),
    )


def capacity_at_delay(records: list, delay: int, degree_min: int = 2, degree_max: int = None) -> dict:
    """Sum legacy/raw/null/bias-corrected/SIGNED capacity over records
    whose ONLY delay is exactly `delay` (single-pair profiles
    L_d(v_{t-delay})), for degree in [degree_min, degree_max] (degree_max
    unset = no upper bound). Used by `causal_latency.py`/`delay_concepts.py`
    to build FIXED-delay quantities such as NL_tau = capacity_at_delay(
    records, tau) -- V2.2 Phase 3's explicit rule: report every fixed
    delay separately, never collapse the tensor onto one parameter-
    dependent peak delay."""
    sel = [r for r in records if r.degree >= degree_min and (degree_max is None or r.degree <= degree_max)
           and r.delays == (delay,)]
    return {
        "delay": delay, "legacy": sum((r.capacity for r in sel), 0.0), "raw": sum((r.raw_capacity for r in sel), 0.0),
        "null": sum((r.null_mean for r in sel), 0.0), "bc": sum((_bias_corrected(r) for r in sel), 0.0),
        "signed": sum((_signed(r) for r in sel), 0.0), "n_profiles": len(sel),
    }


def nl_tensor_by_fixed_delay(records: list, max_delay: int, degree_min: int = 2, degree_max: int = None) -> dict:
    """V2.2 Phase 3: {tau: capacity_at_delay(...)} for EVERY tau in
    0..max_delay, all degree>=degree_min single-delay profiles pooled --
    the fixed-delay NL_tau tensor, computed once and reused everywhere
    (response derivatives, reset comparison, standalone-vs-DQRC
    retention) so no comparison ever silently substitutes a different
    delay's target for another's."""
    return {tau: capacity_at_delay(records, tau, degree_min=degree_min, degree_max=degree_max)
            for tau in range(max_delay + 1)}


def m_tensor_by_fixed_delay(records: list, max_delay: int) -> dict:
    """The degree-1-only analogue of `nl_tensor_by_fixed_delay` -- C_{1,tau}
    at every fixed delay tau, never collapsed to a single M_long sum when
    per-delay detail is what's needed (e.g. for ell_detect/ell_peak)."""
    return {tau: capacity_at_delay(records, tau, degree_min=1, degree_max=1) for tau in range(max_delay + 1)}


# =============================================================================
# Part 4 -- capacity-ceiling diagnostics
# =============================================================================

@dataclass
class CeilingDiagnostic:
    M_long: float
    n_degree1_targets_tested: int      # = max_delay + 1 (one per delay 0..max_delay)
    M_max_target_count: float          # target-count ceiling: at most 1.0 per target
    feature_rank: int
    M_max_rank: float                  # rank ceiling: at most `feature_rank` (can't resolve more
                                        # independent linear directions than the feature space has)
    n_train: int
    M_max_sample: float                # a ridge fit with n_train samples cannot exceed n_train
                                        # independent perfectly-resolved targets either
    M_max_effective: float             # min of the three ceilings above
    fraction_of_ceiling: float          # M_long / M_max_effective
    ceiling_contaminated: bool          # fraction_of_ceiling > 0.95


def diagnose_ceiling(decomp: IPCDecomposition, X: np.ndarray, train: np.ndarray,
                      contamination_threshold: float = 0.95) -> CeilingDiagnostic:
    """Part 4: is M_long saturating a target-count, feature-rank, or
    sample-size ceiling rather than reflecting genuine memory capacity
    heterogeneity? `feature_rank` uses `np.linalg.matrix_rank` on the
    TRAIN-block features (numerically stable SVD-based rank, not a naive
    determinant/inverse)."""
    n_targets = decomp.config["max_delay"] + 1
    feature_rank = int(np.linalg.matrix_rank(X[train] - X[train].mean(axis=0, keepdims=True)))
    n_train = len(train)

    M_max_target_count = float(n_targets)
    M_max_rank = float(feature_rank)
    M_max_sample = float(n_train)
    M_max_effective = min(M_max_target_count, M_max_rank, M_max_sample)
    fraction = decomp.M_long / M_max_effective if M_max_effective > 0 else float("nan")

    return CeilingDiagnostic(
        M_long=decomp.M_long, n_degree1_targets_tested=n_targets, M_max_target_count=M_max_target_count,
        feature_rank=feature_rank, M_max_rank=M_max_rank, n_train=n_train, M_max_sample=M_max_sample,
        M_max_effective=M_max_effective, fraction_of_ceiling=fraction,
        ceiling_contaminated=fraction > contamination_threshold,
    )


# =============================================================================
# Heatmap data helper
# =============================================================================

def to_heatmap(decomp: IPCDecomposition, max_degree: int, max_delay: int):
    """Dense (degree x delay) matrix from `by_degree_delay`'s sparse dict,
    for `imshow`-style plotting -- pure data reshaping, no new computation."""
    mat = np.zeros((max_degree, max_delay + 1))
    for (d, tau), cap in decomp.by_degree_delay.items():
        if d <= max_degree and tau <= max_delay:
            mat[d - 1, tau] = cap
    return mat
