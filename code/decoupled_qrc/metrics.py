"""
metrics.py -- Part 10 (Jacobian decoupling diagnostic) and Part 8/9
(Pareto-frontier construction/comparison, resource accounting).

M = IPC_1 (memory), NL = sum(IPC_2..6) (nonlinearity) throughout, computed
elsewhere (ipc.compute_ipc) and passed in here as plain numbers/point lists
-- this module has no reservoir-specific knowledge, only geometry/statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .utils import bootstrap_ci, bootstrap_diff_ci


# =============================================================================
# Part 10 -- Jacobian decoupling diagnostic
# =============================================================================

@dataclass
class JacobianResult:
    M0: float
    NL0: float
    dM_dm: float
    dM_dg: float
    dNL_dm: float
    dNL_dg: float
    cross_coupling: float
    decoupling_score: float


def jacobian_decoupling(MN_func: Callable[[float, float], tuple], m0: float, g0: float,
                         dm: float, dg: float, eps: float = 1e-9) -> JacobianResult:
    """Finite-difference Jacobian of (M, NL) w.r.t. (m, g) at (m0, g0), where
    `MN_func(m, g) -> (M, NL)` is any DQRC memory-control-parameter/processor-
    control-parameter -> (memory, nonlinearity) map (e.g. built from
    `ipc.compute_ipc` on a run at that (m,g)). Uses a one-sided (forward)
    difference -- 3 `MN_func` evaluations total ((m0,g0), (m0+dm,g0),
    (m0,g0+dg)) rather than 5 for a central difference, since `MN_func` is
    typically an expensive full reservoir+IPC computation and FAST_MODE
    budgets few evaluations; callers wanting a central-difference Jacobian
    can call this twice with dm/dg negated and average.

    `cross_coupling = (|dM/dg| + |dNL/dm|) / (|dM/dm| + |dNL/dg| + eps)` and
    `decoupling_score = 1 / (1 + cross_coupling)` are reported as engineering
    DIAGNOSTICS only, per the task spec -- never treated as a fundamental
    quantity. The raw Jacobian entries are the primary result.
    """
    M0, NL0 = MN_func(m0, g0)
    M_dm, NL_dm = MN_func(m0 + dm, g0)
    M_dg, NL_dg = MN_func(m0, g0 + dg)

    dM_dm = (M_dm - M0) / dm
    dNL_dm = (NL_dm - NL0) / dm
    dM_dg = (M_dg - M0) / dg
    dNL_dg = (NL_dg - NL0) / dg

    cross = (abs(dM_dg) + abs(dNL_dm)) / (abs(dM_dm) + abs(dNL_dg) + eps)
    score = 1.0 / (1.0 + cross)

    return JacobianResult(M0=M0, NL0=NL0, dM_dm=dM_dm, dM_dg=dM_dg, dNL_dm=dNL_dm, dNL_dg=dNL_dg,
                           cross_coupling=cross, decoupling_score=score)


# =============================================================================
# Part 8/9 -- Pareto frontier
# =============================================================================

def pareto_frontier(points: Sequence[tuple]) -> list:
    """Indices of the Pareto-optimal points (maximize BOTH M and NL) among
    `points` (each an (M, NL) pair). Standard sweep: sort by M descending,
    keep a point if its NL exceeds every higher-M point's NL seen so far."""
    order = sorted(range(len(points)), key=lambda i: (-points[i][0], -points[i][1]))
    frontier = []
    best_nl = -np.inf
    for i in order:
        if points[i][1] > best_nl:
            frontier.append(i)
            best_nl = points[i][1]
    return frontier


def is_dominated(point: tuple, frontier_points: Sequence[tuple]) -> bool:
    """True if `point` is weakly dominated by every axis and strictly by at
    least one -- i.e. some frontier point is at least as good in both M and
    NL and strictly better in one. A point NOT dominated by a baseline
    frontier is a candidate 'outside the conventional trade-off' point."""
    for fm, fnl in frontier_points:
        if fm >= point[0] and fnl >= point[1] and (fm > point[0] or fnl > point[1]):
            return True
    return False


def pareto_hypervolume(points: Sequence[tuple], ref: tuple = (0.0, 0.0)) -> float:
    """2D hypervolume indicator: area dominated by the Pareto frontier of
    `points`, relative to `ref` (standard multi-objective-optimization
    quality measure -- a single scalar summarizing 'how good is this whole
    frontier', usable for bootstrap comparison across seeds)."""
    idx = pareto_frontier(points)
    front = sorted([points[i] for i in idx], key=lambda p: p[0])
    if not front:
        return 0.0
    area = 0.0
    prev_m = ref[0]
    for m, nl in front:
        if nl <= ref[1]:
            prev_m = m
            continue
        area += max(0.0, m - prev_m) * (nl - ref[1])
        prev_m = m
    return float(area)


@dataclass
class FrontierComparison:
    hv_baseline: tuple   # (mean, ci_lo, ci_hi)
    hv_dqrc: tuple
    hv_diff: tuple        # (dqrc - baseline, ci_lo, ci_hi)
    n_dqrc_outside_baseline: int
    n_dqrc_total: int
    fraction_outside: float


def frontier_comparison(baseline_points_by_seed: Sequence[Sequence[tuple]],
                         dqrc_points_by_seed: Sequence[Sequence[tuple]],
                         n_boot: int = 1000, seed: int = 0, ref: tuple = (0.0, 0.0)) -> FrontierComparison:
    """Bootstrapped comparison of two architectures' (M, NL) point clouds
    across independent seeds (Part 8/9/13's statistical-significance
    requirement -- never a single-seed claim). Reports per-seed-pooled
    hypervolume with a bootstrap CI for each architecture and for their
    difference, plus how many DQRC points (pooled across all seeds) are NOT
    dominated by the baseline's OWN pooled Pareto frontier -- the direct,
    literal 'points outside the conventional trade-off frontier' count the
    task spec asks for."""
    hv_base = [pareto_hypervolume(pts, ref) for pts in baseline_points_by_seed]
    hv_dqrc = [pareto_hypervolume(pts, ref) for pts in dqrc_points_by_seed]

    hv_base_stats = bootstrap_ci(hv_base, n_boot=n_boot, seed=seed)
    hv_dqrc_stats = bootstrap_ci(hv_dqrc, n_boot=n_boot, seed=seed + 1)
    hv_diff_stats = bootstrap_diff_ci(hv_dqrc, hv_base, n_boot=n_boot, seed=seed + 2)

    pooled_baseline = [p for seed_pts in baseline_points_by_seed for p in seed_pts]
    pooled_dqrc = [p for seed_pts in dqrc_points_by_seed for p in seed_pts]
    baseline_frontier_pts = [pooled_baseline[i] for i in pareto_frontier(pooled_baseline)]
    n_outside = sum(1 for p in pooled_dqrc if not is_dominated(p, baseline_frontier_pts))

    return FrontierComparison(
        hv_baseline=hv_base_stats, hv_dqrc=hv_dqrc_stats, hv_diff=hv_diff_stats,
        n_dqrc_outside_baseline=n_outside, n_dqrc_total=len(pooled_dqrc),
        fraction_outside=n_outside / len(pooled_dqrc) if pooled_dqrc else 0.0,
    )
