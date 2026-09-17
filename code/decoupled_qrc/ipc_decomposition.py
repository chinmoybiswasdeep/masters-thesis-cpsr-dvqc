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


def compute_ipc_decomposed(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                            max_delay: int = 8, max_degree: int = 6, max_targets_per_degree: int = 25,
                            n_surrogates: int = 8, significance_z: float = 2.0, alphas=DEFAULT_ALPHAS,
                            seed: int = 0) -> IPCDecomposition:
    records, was_capped, n_tested = compute_ipc_detailed(
        u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
        max_targets_per_degree=max_targets_per_degree, n_surrogates=n_surrogates,
        significance_z=significance_z, alphas=alphas, seed=seed)

    M_long = sum(r.capacity for r in records if r.degree == 1)

    instant_records = [r for r in records if r.degree >= 2 and r.delays == (0,)]
    temporal_records = [r for r in records if r.degree >= 2 and r.max_delay > 0]
    NL_instant = sum(r.capacity for r in instant_records)
    NL_temporal = sum(r.capacity for r in temporal_records)
    NL_legacy = sum(r.capacity for r in records if r.degree >= 2)
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
    )


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
