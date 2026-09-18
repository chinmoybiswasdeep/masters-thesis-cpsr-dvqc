"""
v3_1_stats.py -- V3.1's statistical layer:

  * hierarchical bootstrap (reservoir seed outer, input seed inner), with
    between- and within-reservoir variance reported separately;
  * the CORRECTED response-geometry angle -- V3 declared the angle
    NOT EVALUABLE whenever the desired off-diagonal term was zero, which is
    exactly the success case. V3.1 projects onto the strongest processor
    control direction instead, so an exact off-diagonal zero is a PASS
    condition rather than an evaluation failure;
  * selectivity reported as a LOWER BOUND from a cross-sensitivity upper
    bound when a denominator is below resolution, never an epsilon-rescued
    enormous ratio;
  * Pareto frontier, hypervolume, and paired bootstrap hypervolume
    differences for the matched-resource comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


# =============================================================================
# Hierarchical bootstrap
# =============================================================================

@dataclass
class HierarchicalCI:
    mean: float
    ci_low: float
    ci_high: float
    between_reservoir_var: float
    within_reservoir_var: float
    n_reservoir: int
    n_input_per_reservoir: dict
    n_boot: int

    def as_dict(self) -> dict:
        return asdict(self)

    def excludes(self, value: float) -> bool:
        """True when `value` lies outside the interval -- the form used for
        'the CI excludes zero / excludes the negligible region'."""
        return not (self.ci_low <= value <= self.ci_high)


def hierarchical_bootstrap(nested, n_boot: int = 2000, alpha: float = 0.05,
                            seed: int = 0, statistic=np.mean) -> HierarchicalCI:
    """`nested` maps reservoir_idx -> list of per-input-seed values.

    Resampling is two-level: reservoir seeds are resampled WITH replacement,
    then within each drawn reservoir its input seeds are resampled with
    replacement. This propagates both variance components, unlike a flat
    bootstrap over pooled values which would understate between-reservoir
    variation."""
    keys = [k for k, v in nested.items() if len(v) > 0]
    if not keys:
        return HierarchicalCI(float("nan"), float("nan"), float("nan"), float("nan"),
                               float("nan"), 0, {}, n_boot)
    groups = [np.asarray(nested[k], dtype=float) for k in keys]
    group_means = np.array([g.mean() for g in groups])
    between = float(np.var(group_means, ddof=1)) if len(groups) > 1 else 0.0
    within = float(np.mean([np.var(g, ddof=1) if g.size > 1 else 0.0 for g in groups]))

    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot)
    n_groups = len(groups)
    for b in range(n_boot):
        drawn = rng.randint(0, n_groups, n_groups)
        vals = []
        for gi in drawn:
            g = groups[gi]
            vals.extend(g[rng.randint(0, g.size, g.size)])
        stats[b] = statistic(vals)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    pooled = np.concatenate(groups)
    return HierarchicalCI(mean=float(statistic(pooled)), ci_low=float(lo), ci_high=float(hi),
                           between_reservoir_var=between, within_reservoir_var=within,
                           n_reservoir=len(keys),
                           n_input_per_reservoir={str(k): len(nested[k]) for k in keys},
                           n_boot=n_boot)


# =============================================================================
# Response geometry -- corrected
# =============================================================================

@dataclass
class ResponseGeometry:
    v_m: tuple
    v_P: tuple
    angle_deg: float
    evaluable: bool
    reason: str
    q_hat: tuple

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("v_m", "v_P", "q_hat"):
            d[k] = list(getattr(self, k))
        return d


def response_angle(dM_dm: float, dNL_dm: float, dM_dg: float, dM_dJ: float,
                    dNL_dg: float, dNL_dJ: float, resolution: float = 1e-9) -> ResponseGeometry:
    """V3.1's corrected geometry.

        v_m   = (dM/dm,  dNL/dm)
        q_hat = grad_{g,J} NL / ||grad_{g,J} NL||        (strongest processor control direction)
        v_P   = (grad_{g,J} M . q_hat,  grad_{g,J} NL . q_hat)

    The angle between v_m and v_P is 90 degrees exactly when the memory
    control moves only M and the processor control moves only NL -- the
    IDEAL case. V3's formulation instead became NOT EVALUABLE precisely
    when the off-diagonal terms vanished, i.e. it could not recognise
    success. Here the angle is NOT EVALUABLE only if an ENTIRE response
    vector is unresolved."""
    grad_nl = np.array([dNL_dg, dNL_dJ], dtype=float)
    grad_m = np.array([dM_dg, dM_dJ], dtype=float)
    v_m = np.array([dM_dm, dNL_dm], dtype=float)

    nl_norm = float(np.linalg.norm(grad_nl))
    if nl_norm < resolution:
        return ResponseGeometry(tuple(v_m), (float("nan"), float("nan")), float("nan"), False,
                                 "processor control direction unresolved: ||grad_{g,J} NL|| below resolution",
                                 (float("nan"), float("nan")))
    q_hat = grad_nl / nl_norm
    v_P = np.array([float(grad_m @ q_hat), float(grad_nl @ q_hat)], dtype=float)

    if float(np.linalg.norm(v_m)) < resolution:
        return ResponseGeometry(tuple(v_m), tuple(v_P), float("nan"), False,
                                 "memory-control response vector unresolved: ||v_m|| below resolution",
                                 tuple(q_hat))
    cos_a = float(v_m @ v_P / (np.linalg.norm(v_m) * np.linalg.norm(v_P)))
    angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return ResponseGeometry(tuple(v_m), tuple(v_P), angle, True, "", tuple(q_hat))


# =============================================================================
# Selectivity with structural zeros
# =============================================================================

@dataclass
class SelectivityBound:
    numerator: float
    denominator_estimate: float
    denominator_upper_bound: float
    ratio: float
    ratio_lower_bound: float
    kind: str                 # "ratio" | "lower_bound" | "not_evaluable"
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def selectivity(numerator: float, denominator_estimate: float, denominator_upper_bound: float,
                 resolution: float = 1e-9) -> SelectivityBound:
    """When the cross-sensitivity denominator is resolved, report the ratio.
    When it is below resolution (the structural-zero case), report a LOWER
    BOUND `numerator / upper_bound` -- a defensible statement such as
    "selectivity exceeds 400" -- instead of dividing by epsilon and printing
    an arbitrarily enormous number."""
    if not np.isfinite(numerator) or not np.isfinite(denominator_upper_bound):
        return SelectivityBound(numerator, denominator_estimate, denominator_upper_bound,
                                 float("nan"), float("nan"), "not_evaluable", "non-finite input")
    if abs(numerator) < resolution:
        return SelectivityBound(numerator, denominator_estimate, denominator_upper_bound,
                                 float("nan"), float("nan"), "not_evaluable",
                                 "numerator below resolution: the diagonal response is itself "
                                 "unresolved, so selectivity is meaningless")
    if denominator_estimate > resolution:
        return SelectivityBound(numerator, denominator_estimate, denominator_upper_bound,
                                 float(numerator / denominator_estimate),
                                 float(numerator / max(denominator_upper_bound, resolution)),
                                 "ratio", "")
    if denominator_upper_bound <= resolution:
        return SelectivityBound(numerator, denominator_estimate, denominator_upper_bound,
                                 float("nan"), float("nan"), "not_evaluable",
                                 "cross-sensitivity upper bound is itself below resolution; "
                                 "report the structural isolation bound instead")
    return SelectivityBound(numerator, denominator_estimate, denominator_upper_bound,
                             float("nan"), float(numerator / denominator_upper_bound),
                             "lower_bound",
                             "denominator below resolution (structural zero): reporting a lower "
                             "bound from the cross-sensitivity upper bound")


# =============================================================================
# Pareto frontier and hypervolume
# =============================================================================

def pareto_front(points) -> list:
    """Indices of the non-dominated points, maximising BOTH coordinates
    (memory, instantaneous NL)."""
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    nondominated = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dominated = True
                break
        if not dominated:
            nondominated.append(i)
    return nondominated


def hypervolume(points, reference=(0.0, 0.0)) -> float:
    """2-D hypervolume (dominated area) above `reference`, maximising both
    coordinates. Computed exactly by sweeping the non-dominated set."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    keep = pts[np.all(pts > ref, axis=1)] if pts.ndim == 2 else np.empty((0, 2))
    if keep.size == 0:
        return 0.0
    idx = pareto_front(keep)
    front = keep[idx]
    front = front[np.argsort(-front[:, 0])]       # descending x
    area, prev_y = 0.0, ref[1]
    for x, y in front:
        if y > prev_y:
            area += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(area)


@dataclass
class HypervolumeComparison:
    hv_a: float
    hv_b: float
    delta: float
    ci_low: float
    ci_high: float
    favours_a: bool
    n_boot: int

    def as_dict(self) -> dict:
        return asdict(self)


def paired_hypervolume_bootstrap(nested_a, nested_b, reference=(0.0, 0.0), n_boot: int = 1000,
                                  alpha: float = 0.05, seed: int = 0) -> HypervolumeComparison:
    """`nested_a` / `nested_b` map reservoir_idx -> list of (M, NL) points.
    PAIRED at the reservoir level (the same drawn reservoir contributes to
    both arms), so the difference isolates the architecture rather than the
    seed draw. `favours_a` requires the CI to lie strictly above zero."""
    keys = sorted(set(nested_a) & set(nested_b))
    if not keys:
        return HypervolumeComparison(float("nan"), float("nan"), float("nan"),
                                      float("nan"), float("nan"), False, n_boot)
    hv_a = hypervolume([p for k in keys for p in nested_a[k]], reference)
    hv_b = hypervolume([p for k in keys for p in nested_b[k]], reference)

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        drawn = [keys[i] for i in rng.randint(0, len(keys), len(keys))]
        pa = [p for k in drawn for p in nested_a[k]]
        pb = [p for k in drawn for p in nested_b[k]]
        deltas[b] = hypervolume(pa, reference) - hypervolume(pb, reference)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return HypervolumeComparison(hv_a=hv_a, hv_b=hv_b, delta=hv_a - hv_b, ci_low=float(lo),
                                  ci_high=float(hi), favours_a=bool(lo > 0.0), n_boot=n_boot)


# =============================================================================
# Dynamic range (processor controllability criterion)
# =============================================================================

def dynamic_range(values, eps: float = 1e-9) -> dict:
    """E_NL = (Q_0.9 - Q_0.1) / (|median| + eps) -- how much the metric
    actually MOVES over the scanned control region, relative to its own
    scale. The preregistered discovery gate is E_NL > 0.20."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return {"E": float("nan"), "q10": float("nan"), "q90": float("nan"),
                "median": float("nan"), "n": int(v.size)}
    q10, q90 = np.percentile(v, [10, 90])
    med = float(np.median(v))
    return {"E": float((q90 - q10) / (abs(med) + eps)), "q10": float(q10), "q90": float(q90),
            "median": med, "n": int(v.size)}
