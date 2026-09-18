"""
ceiling_audit.py -- V3.1's ceiling and tail diagnostics.

Motivation, verified directly from the V3 results: the V3 standalone
processor reported `NL0_raw = 2.000` with `max_degree = 3`, and the number
of single-delay tau=0 targets at that setting is exactly 2 (L2, L3), each
capped at capacity 1.0. The metric was therefore pinned EXACTLY at its
target-count ceiling. A saturated metric has zero gradient by
construction, so it cannot respond to (g,J) -- which is why V3 saw an
almost flat NL(g,J) surface and unstable (g,J) derivatives.

Measured consequence (same processor, same seeds, T=200):
    max_degree = 3 -> NL0_raw = 2.0000 (= ceiling), E_NL = 0.044
    max_degree = 5 -> NL0_raw = 4.0000 (= ceiling), E_NL = --
    max_degree = 8 -> NL0_raw = 5.0863 (< 7),      E_NL = 0.222

This module makes that failure impossible to miss: any metric whose raw
capacity sits at its ceiling is flagged `ceiling_contaminated`, and V3.1
refuses to differentiate such a metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .ipc_decomposition import capacity_at_delay, nl_tensor_by_fixed_delay


@dataclass
class CeilingReport:
    metric: str
    raw: float
    n_targets: int                 # target-count ceiling (each target caps at 1.0)
    numerical_rank: int
    effective_rank: float
    n_train: int
    target_count_ceiling: float
    rank_ceiling: float
    sample_ceiling: float
    effective_ceiling: float
    fraction_of_ceiling: float
    ceiling_contaminated: bool
    contamination_threshold: float
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def audit_ceiling(metric: str, raw: float, n_targets: int, numerical_rank: int,
                   effective_rank: float, n_train: int,
                   contamination_threshold: float = 0.95) -> CeilingReport:
    """Three independent ceilings; the binding one is the minimum.

    target-count : each IPC target contributes at most 1.0, so a sum over
                   `n_targets` targets cannot exceed `n_targets`.
    rank         : a ridge fit cannot resolve more independent directions
                   than the feature matrix has.
    sample       : nor more than the number of training samples.
    """
    tc = float(max(n_targets, 0))
    rc = float(max(numerical_rank, 0))
    sc = float(max(n_train, 0))
    eff = min(x for x in (tc, rc, sc) if x > 0) if any(x > 0 for x in (tc, rc, sc)) else 0.0
    frac = float(raw / eff) if eff > 0 else float("nan")
    contaminated = bool(np.isfinite(frac) and frac > contamination_threshold)
    reason = ""
    if contaminated:
        binding = min((tc, "target_count"), (rc, "numerical_rank"), (sc, "sample_count"),
                      key=lambda p: p[0])[1]
        reason = (f"raw {raw:.4f} is {frac:.1%} of the effective ceiling {eff:.1f} "
                  f"(binding: {binding}); the metric is saturated and MUST NOT be differentiated")
    return CeilingReport(metric=metric, raw=float(raw), n_targets=int(n_targets),
                          numerical_rank=int(numerical_rank), effective_rank=float(effective_rank),
                          n_train=int(n_train), target_count_ceiling=tc, rank_ceiling=rc,
                          sample_ceiling=sc, effective_ceiling=eff, fraction_of_ceiling=frac,
                          ceiling_contaminated=contaminated,
                          contamination_threshold=contamination_threshold, reason=reason)


def count_single_delay_targets(records: list, delay: int, degree_min: int = 2) -> int:
    """How many single-delay targets actually entered the metric at this
    delay -- the target-count ceiling for NL_{tau=delay}."""
    return sum(1 for r in records if r.degree >= degree_min and r.delays == (delay,))


def audit_nl0(records: list, numerical_rank: int, effective_rank: float, n_train: int,
               contamination_threshold: float = 0.95) -> CeilingReport:
    """Ceiling audit for the primary instantaneous-NL metric."""
    at0 = capacity_at_delay(records, 0, degree_min=2)
    return audit_ceiling("NL_0", at0["raw"], count_single_delay_targets(records, 0),
                          numerical_rank, effective_rank, n_train, contamination_threshold)


def audit_memory(records: list, numerical_rank: int, effective_rank: float, n_train: int,
                  max_delay: int, contamination_threshold: float = 0.95) -> CeilingReport:
    """Ceiling audit for linear memory: one degree-1 target per delay."""
    m_records = [r for r in records if r.degree == 1]
    raw = sum(r.raw_capacity for r in m_records)
    return audit_ceiling("M_long", raw, len(m_records), numerical_rank, effective_rank,
                          n_train, contamination_threshold)


@dataclass
class TailReport:
    kind: str                      # "order" or "delay"
    values: dict                   # {order or delay: capacity}
    tail_value: float
    peak_value: float
    tail_fraction: float
    declining: bool
    threshold: float
    recommendation: str

    def as_dict(self) -> dict:
        return asdict(self)


def order_tail(records: list, max_degree: int, delay: int = 0,
                tail_threshold: float = 0.10) -> TailReport:
    """C_{d,delay} as a function of polynomial order d. If the HIGHEST order
    still carries a substantial fraction of the peak, the order horizon is
    too low and the summed metric is truncating real capacity."""
    vals = {}
    for d in range(2, max_degree + 1):
        sel = [r for r in records if r.degree == d and r.delays == (delay,)]
        vals[d] = float(sum(max(0.0, r.raw_capacity - r.null_mean) for r in sel))
    if not vals:
        return TailReport("order", {}, 0.0, 0.0, float("nan"), False, tail_threshold,
                           "no degree>=2 single-delay targets present")
    peak = max(vals.values()) if vals else 0.0
    tail = vals[max(vals)]
    frac = float(tail / peak) if peak > 1e-12 else float("nan")
    declining = bool(np.isfinite(frac) and frac < tail_threshold)
    rec = ("order horizon adequate: the top order carries a small fraction of the peak"
           if declining else
           f"INCREASE max_degree: order {max(vals)} still carries {frac:.1%} of the peak")
    return TailReport("order", vals, tail, peak, frac, declining, tail_threshold, rec)


def delay_tail(records: list, max_delay: int, tail_threshold: float = 0.10) -> TailReport:
    """C_{1,tau} as a function of delay. If the LONGEST delay still carries
    substantial capacity, the delay horizon truncates real memory."""
    vals = {}
    for tau in range(max_delay + 1):
        sel = [r for r in records if r.degree == 1 and r.delays == (tau,)]
        vals[tau] = float(sum(max(0.0, r.raw_capacity - r.null_mean) for r in sel))
    if not vals:
        return TailReport("delay", {}, 0.0, 0.0, float("nan"), False, tail_threshold,
                           "no degree-1 single-delay targets present")
    peak = max(vals.values()) if vals else 0.0
    tail = vals[max(vals)]
    frac = float(tail / peak) if peak > 1e-12 else float("nan")
    declining = bool(np.isfinite(frac) and frac < tail_threshold)
    rec = ("delay horizon adequate: the memory tail has decayed"
           if declining else
           f"INCREASE max_delay: delay {max(vals)} still carries {frac:.1%} of the peak")
    return TailReport("delay", vals, tail, peak, frac, declining, tail_threshold, rec)


@dataclass
class NullReport:
    metric: str
    raw: float
    null_mean: float
    null_std: float
    null_ci: tuple
    null_fraction: float          # null_mean / raw
    acceptable: bool
    threshold: float
    empirical_p: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["null_ci"] = list(self.null_ci)
        return d


def audit_null(records: list, metric: str, selector, raw: float,
                fraction_threshold: float = 0.20) -> NullReport:
    """Null-bias diagnostic. `C_null / C_raw < 0.20` is the preregistered
    initial gate: a metric whose null baseline is a large fraction of its
    raw value is mostly measuring finite-sample bias."""
    sel = [r for r in records if selector(r)]
    if not sel:
        return NullReport(metric, float(raw), 0.0, 0.0, (0.0, 0.0), float("nan"), False,
                           fraction_threshold, float("nan"))
    null_mean = float(sum(r.null_mean for r in sel))
    null_stds = np.array([r.null_std for r in sel], dtype=float)
    null_std = float(np.sqrt(np.sum(null_stds ** 2)))      # independent targets
    ci = (null_mean - 1.96 * null_std, null_mean + 1.96 * null_std)
    frac = float(null_mean / raw) if abs(raw) > 1e-12 else float("nan")
    # empirical p: fraction of targets whose real score failed its own threshold
    n_fail = sum(1 for r in sel if r.raw_capacity <= r.threshold)
    p_emp = float(n_fail / len(sel))
    return NullReport(metric, float(raw), null_mean, null_std, ci, frac,
                       bool(np.isfinite(frac) and frac < fraction_threshold),
                       fraction_threshold, p_emp)


def benjamini_hochberg(p_values, alpha: float = 0.05) -> dict:
    """False-discovery correction for target-level significance claims."""
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return {"n": 0, "n_significant": 0, "threshold": float("nan"), "rejected": []}
    order = np.argsort(p)
    ranked = p[order]
    crit = alpha * (np.arange(1, n + 1) / n)
    below = np.where(ranked <= crit)[0]
    if below.size == 0:
        return {"n": int(n), "n_significant": 0, "threshold": 0.0, "rejected": []}
    k = int(below.max())
    thr = float(ranked[k])
    rejected = sorted(int(i) for i in order[: k + 1])
    return {"n": int(n), "n_significant": len(rejected), "threshold": thr, "rejected": rejected}
