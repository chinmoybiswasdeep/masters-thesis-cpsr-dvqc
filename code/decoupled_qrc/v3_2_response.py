"""
v3_2_response.py -- dimensionless controls, five-point stencils, multi-estimator
derivatives, stability adjudication and the response Jacobian.

The statistics themselves are NOT reimplemented: `v3_1_stats` already carries
the repaired hierarchical bootstrap (two-level: reservoir seeds resampled,
then input seeds within each), the repaired `response_angle` (V3's version
became NOT EVALUABLE precisely in the success case) and `selectivity` (which
reports a defensible LOWER BOUND when the cross-sensitivity denominator is
unresolved instead of dividing by epsilon). This module supplies only what
V3.2 needs on top.

NORMALISED CONTROLS. Every derivative is taken with respect to
p~ = (p - p_min)/(p_max - p_min), so dM/dm~, dNL/dg~ and dNL/dJ~ are
commensurable and the Jacobian's entries can be compared at all.

INTERIOR MARGIN IS A HARD PRECONDITION. A symmetric five-point stencil at
p~* needs p~* +/- 2h to lie strictly inside [0, 1]. V3.1 selected m* = 0.95 at
the very top of m in [0.1, 0.95]: no symmetric stencil fits there, which is
why its derivative was unstable. `stencil_points` REFUSES to build a stencil
that would leave the domain rather than silently one-siding it.

FOUR ESTIMATORS, ONE VERDICT. Each derivative is computed by small central
difference, large central difference, the O(h^4) five-point rule and a
quadratic fit. `assess_stability` calls a derivative UNSTABLE if the
estimators disagree in sign, if the relative spread exceeds its threshold, if
the bootstrap CI contains zero, or if a single reservoir seed dominates. A
decisive ratio (Gate G) is never computed from an unstable derivative.

FROZEN vs RETRAINED READOUT. Both are reported: the frozen-readout derivative
(ridge alpha and target set fixed once at the centre, via
`frozen_protocol`-style reuse) isolates representation change, while the
retrained-readout derivative measures usable-task change. They answer
different questions and neither substitutes for the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .v3_1_stats import hierarchical_bootstrap, response_angle, selectivity


class CI:
    """Uniform (point, lo, hi) view over `v3_1_stats.HierarchicalCI`.

    That dataclass names its fields `mean` / `ci_low` / `ci_high` /
    `between_reservoir_var` / `within_reservoir_var` / `n_reservoir`. Adapting
    once here keeps the legacy naming from leaking into every call site, and
    keeps V3.2 from accidentally reading an attribute that does not exist.
    """

    __slots__ = ("point", "lo", "hi", "between_var", "within_var", "n_groups", "raw")

    def __init__(self, raw):
        self.raw = raw
        self.point = float(raw.mean)
        self.lo = float(raw.ci_low)
        self.hi = float(raw.ci_high)
        self.between_var = float(raw.between_reservoir_var)
        self.within_var = float(raw.within_reservoir_var)
        self.n_groups = int(raw.n_reservoir)

    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi) and not (self.lo <= 0.0 <= self.hi))

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "between_var": self.between_var, "within_var": self.within_var,
                "n_groups": self.n_groups}


def bootstrap_ci(nested, n_boot: int = 2000, seed: int = 0) -> CI:
    """Two-level bootstrap over {reservoir_idx: [per-input-seed values]}."""
    return CI(hierarchical_bootstrap(nested, n_boot=n_boot, seed=seed))


def normalize(p: float, lo: float, hi: float) -> float:
    """p -> p~ in [0, 1]."""
    if hi <= lo:
        raise ValueError(f"empty control range [{lo}, {hi}]")
    return (float(p) - lo) / (hi - lo)


def denormalize(p_tilde: float, lo: float, hi: float) -> float:
    return lo + float(p_tilde) * (hi - lo)


def interior_margin(p: float, lo: float, hi: float) -> float:
    """Distance from the nearer boundary, in normalised units."""
    pt = normalize(p, lo, hi)
    return float(min(pt, 1.0 - pt))


def stencil_points(p: float, lo: float, hi: float, h: float) -> dict:
    """The five-point stencil p~* + {-2h, -h, 0, +h, +2h}, in RAW units.

    Raises if the stencil would leave [0, 1] in normalised units -- a
    boundary-incompatible candidate must be rejected, not silently one-sided.
    """
    pt = normalize(p, lo, hi)
    if h <= 0:
        raise ValueError(f"step h must be positive, got {h}")
    if pt - 2 * h < -1e-12 or pt + 2 * h > 1 + 1e-12:
        raise ValueError(
            f"five-point stencil at p~={pt:.4f} with h={h} leaves [0,1] "
            f"(needs margin >= {2 * h}); candidate is boundary-incompatible")
    offsets = {"m2h": -2 * h, "m1h": -h, "c": 0.0, "p1h": h, "p2h": 2 * h}
    return {k: denormalize(pt + o, lo, hi) for k, o in offsets.items()}


def derivative_estimates(values: dict, h: float) -> dict:
    """Derivatives w.r.t. the NORMALISED control from a five-point stencil.

    `values` maps the `stencil_points` keys to the measured metric.
    """
    f_m2, f_m1, f_c, f_p1, f_p2 = (values["m2h"], values["m1h"], values["c"],
                                   values["p1h"], values["p2h"])
    est = {
        "central_small": (f_p1 - f_m1) / (2 * h),
        "central_large": (f_p2 - f_m2) / (4 * h),
        "five_point": (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h),
    }
    xs = np.array([-2 * h, -h, 0.0, h, 2 * h])
    ys = np.array([f_m2, f_m1, f_c, f_p1, f_p2])
    if np.all(np.isfinite(ys)):
        est["quadratic_fit"] = float(np.polyfit(xs, ys, 2)[1])
        est["linear_fit"] = float(np.polyfit(xs, ys, 1)[0])
    return {k: float(v) for k, v in est.items()}


@dataclass
class DerivativeVerdict:
    name: str
    estimates: dict
    point_estimate: float
    ci: CI = None
    stable: bool = False
    reasons: list = field(default_factory=list)
    rel_spread: float = float("nan")
    sign_agreement: bool = False
    seed_dominated: bool = False

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "ci"}
        d["ci"] = self.ci.as_dict() if self.ci is not None else None
        return d

    @property
    def upper_abs_bound(self) -> float:
        """Upper 95% bound on |derivative| -- the equivalence bound Gate F
        needs. 'Not significant' is NOT evidence of zero; this is."""
        if self.ci is None or not np.isfinite(self.ci.lo) or not np.isfinite(self.ci.hi):
            return float("nan")
        return float(max(abs(self.ci.lo), abs(self.ci.hi)))


def assess_stability(name: str, estimates: dict, nested_per_seed: dict = None, *,
                     rel_spread_max: float = 0.5, n_boot: int = 2000, seed: int = 0,
                     require_ci_excludes_zero: bool = True,
                     seed_dominance_max: float = 0.7) -> DerivativeVerdict:
    """Adjudicate one derivative.

    `nested_per_seed` maps reservoir_idx -> [per-input-seed derivative values]
    and drives the hierarchical bootstrap. Without it only the estimator-
    agreement checks can run, and the verdict says so.
    """
    vals = np.array([v for v in estimates.values() if np.isfinite(v)], dtype=float)
    reasons = []
    if vals.size == 0:
        return DerivativeVerdict(name=name, estimates=estimates, point_estimate=float("nan"),
                                 stable=False, reasons=["no finite estimator"])

    point = float(np.median(vals))
    signs = np.sign(vals[np.abs(vals) > 1e-12])
    sign_ok = bool(signs.size == 0 or np.all(signs == signs[0]))
    if not sign_ok:
        reasons.append("estimators disagree in sign")

    scale = max(abs(point), 1e-12)
    rel_spread = float((vals.max() - vals.min()) / scale)
    if rel_spread > rel_spread_max:
        reasons.append(f"relative spread {rel_spread:.3f} > {rel_spread_max}")

    ci, seed_dom = None, False
    if nested_per_seed:
        ci = bootstrap_ci(nested_per_seed, n_boot=n_boot, seed=seed)
        if require_ci_excludes_zero and np.isfinite(ci.lo) and np.isfinite(ci.hi):
            if ci.lo <= 0.0 <= ci.hi:
                reasons.append("bootstrap CI contains zero")
        means = {k: float(np.mean(v)) for k, v in nested_per_seed.items() if len(v)}
        if len(means) > 1:
            total = sum(abs(x) for x in means.values())
            if total > 0 and max(abs(x) for x in means.values()) / total > seed_dominance_max:
                seed_dom = True
                reasons.append("a single reservoir seed dominates the estimate")
        point = float(ci.point) if np.isfinite(ci.point) else point
    else:
        reasons.append("no per-seed data: CI not evaluated")

    return DerivativeVerdict(name=name, estimates=estimates, point_estimate=point, ci=ci,
                             stable=bool(not reasons), reasons=reasons,
                             rel_spread=rel_spread, sign_agreement=sign_ok,
                             seed_dominated=seed_dom)


@dataclass
class Jacobian:
    """The 2x3 normalised response matrix and its geometry."""

    dM_dm: DerivativeVerdict
    dM_dg: DerivativeVerdict
    dM_dJ: DerivativeVerdict
    dNL_dm: DerivativeVerdict
    dNL_dg: DerivativeVerdict
    dNL_dJ: DerivativeVerdict
    metric: str = "NL_0"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.dM_dm.point_estimate, self.dM_dg.point_estimate,
                          self.dM_dJ.point_estimate],
                         [self.dNL_dm.point_estimate, self.dNL_dg.point_estimate,
                          self.dNL_dJ.point_estimate]], dtype=float)

    def geometry(self) -> dict:
        J = self.matrix
        geo = response_angle(dM_dm=J[0, 0], dNL_dm=J[1, 0], dM_dg=J[0, 1], dM_dJ=J[0, 2],
                             dNL_dg=J[1, 1], dNL_dJ=J[1, 2])
        svals = np.linalg.svd(J, compute_uv=False)
        off = float(np.sqrt(J[0, 1] ** 2 + J[0, 2] ** 2 + J[1, 0] ** 2))
        return {"angle_deg": geo.angle_deg, "evaluable": bool(geo.evaluable),
                "reason": geo.reason,
                "singular_values": [float(s) for s in svals],
                "condition_number": float(svals[0] / svals[-1]) if svals[-1] > 1e-15 else float("inf"),
                "off_diagonal_norm": off, "frobenius_norm": float(np.linalg.norm(J))}

    def selectivity_ratios(self) -> dict:
        """R_M and R_NL0, computed ONLY from stable derivatives."""
        stable = {k: getattr(self, k).stable for k in
                  ("dM_dm", "dM_dg", "dM_dJ", "dNL_dm", "dNL_dg", "dNL_dJ")}
        J = self.matrix
        out = {"stable": stable}

        if stable["dM_dm"] and stable["dM_dg"] and stable["dM_dJ"]:
            cross = float(np.hypot(J[0, 1], J[0, 2]))
            ub = float(np.hypot(self.dM_dg.upper_abs_bound, self.dM_dJ.upper_abs_bound))
            out["R_M"] = selectivity(abs(J[0, 0]), cross, ub).as_dict()
        else:
            out["R_M"] = {"kind": "not_evaluable",
                          "note": "one or more component derivatives are unstable"}

        if stable["dNL_dg"] and stable["dNL_dJ"] and stable["dNL_dm"]:
            num = float(np.hypot(J[1, 1], J[1, 2]))
            out["R_NL0"] = selectivity(num, abs(J[1, 0]),
                                       self.dNL_dm.upper_abs_bound).as_dict()
        else:
            out["R_NL0"] = {"kind": "not_evaluable",
                            "note": "one or more component derivatives are unstable"}
        return out

    def as_dict(self) -> dict:
        return {"metric": self.metric,
                "matrix": self.matrix.tolist(),
                "derivatives": {k: getattr(self, k).as_dict() for k in
                                ("dM_dm", "dM_dg", "dM_dJ", "dNL_dm", "dNL_dg", "dNL_dJ")},
                "geometry": self.geometry(),
                "selectivity": self.selectivity_ratios()}


def equivalence_bound(verdict: DerivativeVerdict, reference: float, margin_fraction: float) -> dict:
    """Gate F: is |derivative| demonstrably SMALLER than margin*reference?

    Uses the upper confidence bound, never a failure to reject zero. A
    non-significant derivative with a wide CI fails this test, which is the
    correct behaviour.
    """
    ub = verdict.upper_abs_bound
    allowed = float(margin_fraction) * abs(float(reference))
    ok = bool(np.isfinite(ub) and ub < allowed)
    return {"name": verdict.name, "upper_abs_bound": ub, "reference": float(reference),
            "margin_fraction": float(margin_fraction), "allowed": allowed,
            "suppressed": ok,
            "reason": ("upper bound below the equivalence margin" if ok
                       else "upper bound not below the equivalence margin "
                            "(non-significance alone is not evidence of zero)")}
