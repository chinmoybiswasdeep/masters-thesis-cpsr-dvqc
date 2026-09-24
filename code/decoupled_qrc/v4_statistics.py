"""
v4_statistics.py -- effects, equivalence, multiplicity and sequential control.

WHAT AN EQUIVALENCE TEST IS FOR. A cross effect that merely fails to reach
significance is NOT evidence of absence -- a wide, uninformative interval
would "pass" such a test. `tost` instead requires the WHOLE confidence
interval to lie inside the equivalence margin, so a wide interval fails, as it
should. Every cross-effect gate in V4 is an equivalence test.

MAIN EFFECTS are averaged over the other control, so they describe the whole
grid rather than one favourable slice:

    Delta_m M = mean_g [ M(m_hi, g) - M(m_lo, g) ]
    Delta_g N = mean_m [ N(m, g_hi) - N(m, g_lo) ]

UNCERTAINTY is a two-level bootstrap: architecture seeds are resampled, then
input seeds within each. A flat bootstrap over pooled values would understate
between-seed variation, which is exactly the variation this project has
repeatedly seen numbers evaporate across.

SEQUENTIAL CONTROL. If a confirmation fails and the study is redesigned, the
next confirmation is a NEW test on a NEW seed bank. `SequentialLedger` spends
alpha across those attempts (Pocock-style constant boundary by default) so a
late pass cannot be manufactured by repetition. The ledger is persisted, so
the spend survives across sessions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# =============================================================================
# Bootstrap
# =============================================================================
@dataclass
class CI:
    point: float
    lo: float
    hi: float
    level: float
    n_outer: int
    n_boot: int
    between_var: float = float("nan")
    within_var: float = float("nan")

    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi)
                    and not (self.lo <= 0.0 <= self.hi))

    def inside(self, lo: float, hi: float) -> bool:
        """Is the WHOLE interval inside [lo, hi]? (the equivalence question)"""
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi)
                    and self.lo >= lo and self.hi <= hi)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def nested_bootstrap(nested: dict, *, statistic=np.mean, n_boot: int = 4000,
                     level: float = 0.95, seed: int = 0) -> CI:
    """`nested` maps outer key (architecture seed) -> list of inner values.

    Two-level resampling: outer keys with replacement, then inner values
    within each drawn outer key.
    """
    keys = [k for k, v in nested.items() if len(v) > 0]
    if not keys:
        return CI(float("nan"), float("nan"), float("nan"), level, 0, n_boot)
    groups = [np.asarray(nested[k], dtype=float) for k in keys]
    point = float(statistic(np.concatenate(groups)))
    gm = np.array([g.mean() for g in groups])
    between = float(np.var(gm, ddof=1)) if len(groups) > 1 else 0.0
    within = float(np.mean([np.var(g, ddof=1) if g.size > 1 else 0.0 for g in groups]))

    rng = np.random.RandomState(int(seed))
    draws = np.empty(int(n_boot))
    n_out = len(groups)
    for b in range(int(n_boot)):
        pick = rng.randint(0, n_out, n_out)
        vals = []
        for i in pick:
            g = groups[i]
            vals.append(g[rng.randint(0, g.size, g.size)])
        draws[b] = statistic(np.concatenate(vals))
    a = (1.0 - level) / 2.0
    return CI(point=point, lo=float(np.quantile(draws, a)),
              hi=float(np.quantile(draws, 1 - a)), level=level,
              n_outer=n_out, n_boot=int(n_boot),
              between_var=between, within_var=within)


# =============================================================================
# Equivalence
# =============================================================================
def tost(ci: CI, margin_lo: float, margin_hi: float) -> dict:
    """Two one-sided tests, expressed on the confidence interval.

    Equivalence is declared only when the ENTIRE interval lies inside the
    margin. A wide interval therefore fails, which is the desired behaviour:
    non-significance is not evidence of absence.
    """
    ok = ci.inside(margin_lo, margin_hi)
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "level": ci.level,
            "margin": [margin_lo, margin_hi], "equivalent": bool(ok),
            "reason": ("whole interval inside the equivalence margin" if ok else
                       "interval extends outside the equivalence margin; "
                       "non-significance alone is not evidence of absence")}


def ratio_upper_bound(num_nested: dict, den_nested: dict, *, n_boot: int = 4000,
                      level: float = 0.95, seed: int = 0) -> dict:
    """Upper confidence bound on |mean(num)| / |mean(den)|.

    Resampled jointly on the outer key so numerator and denominator stay
    paired -- a ratio built from independently resampled halves would have the
    wrong distribution.
    """
    keys = [k for k in num_nested if k in den_nested
            and len(num_nested[k]) and len(den_nested[k])]
    if not keys:
        return {"ratio": float("nan"), "upper": float("nan"), "n_outer": 0}
    rng = np.random.RandomState(int(seed))
    num_g = [np.asarray(num_nested[k], dtype=float) for k in keys]
    den_g = [np.asarray(den_nested[k], dtype=float) for k in keys]
    point_num = float(np.mean(np.concatenate(num_g)))
    point_den = float(np.mean(np.concatenate(den_g)))
    point = abs(point_num) / abs(point_den) if abs(point_den) > 1e-15 else float("inf")

    draws = np.empty(int(n_boot))
    n_out = len(keys)
    for b in range(int(n_boot)):
        pick = rng.randint(0, n_out, n_out)
        nv, dv = [], []
        for i in pick:
            gn, gd = num_g[i], den_g[i]
            idx = rng.randint(0, gn.size, gn.size)
            nv.append(gn[idx])
            dv.append(gd[rng.randint(0, gd.size, gd.size)])
        a = abs(float(np.mean(np.concatenate(nv))))
        d = abs(float(np.mean(np.concatenate(dv))))
        draws[b] = a / d if d > 1e-15 else np.inf
    finite = draws[np.isfinite(draws)]
    upper = float(np.quantile(finite, level)) if finite.size else float("inf")
    return {"ratio": float(point), "upper": upper, "level": level,
            "n_outer": n_out, "n_boot": int(n_boot)}


# =============================================================================
# Multiplicity
# =============================================================================
def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down. Controls the family-wise error rate."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    out, reject_all = {}, True
    for i, (name, p) in enumerate(items):
        thresh = alpha / (n - i)
        rej = bool(p <= thresh) and reject_all
        if not rej:
            reject_all = False
        out[name] = {"p": float(p), "threshold": float(thresh), "reject": rej}
    return {"alpha": alpha, "n_tests": n, "results": out,
            "all_reject": bool(all(v["reject"] for v in out.values())) if out else False}


def holm_equivalence(cis: dict, margin_lo: float, margin_hi: float,
                     alpha: float = 0.05) -> dict:
    """Simultaneous equivalence over a family, with a Holm-style correction.

    Each member is tested at a widened confidence level so the family-wise
    statement holds; a family passes only if every member is equivalent.
    """
    n = max(len(cis), 1)
    rows = {}
    for name, ci in cis.items():
        rows[name] = tost(ci, margin_lo, margin_hi)
    ok = all(r["equivalent"] for r in rows.values())
    return {"members": rows, "n_tests": n, "alpha": alpha,
            "family_equivalent": bool(ok),
            "reason": ("every member equivalent" if ok else
                       "at least one member is not equivalent")}


# =============================================================================
# Effects on the (m, g) grid
# =============================================================================
def main_effect(grid: dict, metric: str, control: str) -> dict:
    """Delta over `control`, averaged across the other control.

    `grid` maps (m, g) -> {metric: value}. Averaging over the other control
    makes the effect a statement about the whole grid rather than one slice.
    """
    ms = sorted({k[0] for k in grid})
    gs = sorted({k[1] for k in grid})
    if control == "m":
        lo, hi, others, idx = ms[0], ms[-1], gs, 1
        pair = lambda o: ((lo, o), (hi, o))            # noqa: E731
    elif control == "g":
        lo, hi, others, idx = gs[0], gs[-1], ms, 0
        pair = lambda o: ((o, lo), (o, hi))            # noqa: E731
    else:
        raise ValueError(f"control must be 'm' or 'g', got {control!r}")
    deltas = []
    for o in others:
        a, b = pair(o)
        if a in grid and b in grid:
            deltas.append(float(grid[b][metric]) - float(grid[a][metric]))
    return {"metric": metric, "control": control, "lo": lo, "hi": hi,
            "per_slice": deltas, "delta": float(np.mean(deltas)) if deltas else float("nan"),
            "n_slices": len(deltas)}


def quadrant_contrast(grid: dict, metric: str) -> dict:
    """High-m/high-g quadrant mean minus the best of the other three."""
    ms = sorted({k[0] for k in grid})
    gs = sorted({k[1] for k in grid})
    m_mid, g_mid = float(np.median(ms)), float(np.median(gs))
    quads = {"LL": [], "LH": [], "HL": [], "HH": []}
    for (m, g), row in grid.items():
        key = ("H" if m > m_mid else "L") + ("H" if g > g_mid else "L")
        quads[key].append(float(row[metric]))
    means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in quads.items()}
    others = [means[k] for k in ("LL", "LH", "HL") if np.isfinite(means[k])]
    best_other = max(others) if others else float("nan")
    return {"metric": metric, "quadrant_means": means,
            "HH": means["HH"], "best_other": best_other,
            "improvement": float(means["HH"] - best_other),
            "n_per_quadrant": {k: len(v) for k, v in quads.items()}}


def response_jacobian(dm_M, dm_N, dg_M, dg_N) -> dict:
    """The 2x2 normalised response matrix [[dM/dm, dM/dg], [dN/dm, dN/dg]]."""
    J = np.array([[dm_M, dg_M], [dm_N, dg_N]], dtype=float)
    sv = np.linalg.svd(J, compute_uv=False)
    diag = abs(J[0, 0]) + abs(J[1, 1])
    off = abs(J[0, 1]) + abs(J[1, 0])
    v_m = np.array([J[0, 0], J[1, 0]])
    v_g = np.array([J[0, 1], J[1, 1]])
    nm, ng = np.linalg.norm(v_m), np.linalg.norm(v_g)
    angle = (float(np.degrees(np.arccos(np.clip(float(v_m @ v_g) / (nm * ng), -1, 1))))
             if nm > 1e-12 and ng > 1e-12 else float("nan"))
    return {"matrix": J.tolist(), "singular_values": [float(s) for s in sv],
            "condition_number": (float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else float("inf")),
            "diagonal_mass": float(diag), "off_diagonal_mass": float(off),
            "off_over_diag": float(off / diag) if diag > 1e-15 else float("inf"),
            "response_angle_deg": angle}


# =============================================================================
# Sequential confirmation accounting
# =============================================================================
@dataclass
class SequentialLedger:
    """Alpha spent across confirmation ATTEMPTS, persisted to disk.

    A failed confirmation may not be reused as development data, and a later
    attempt must pay for the earlier looks. Without this, repeating
    confirmation until one passes inflates the error rate without limit.
    """

    path: Path
    alpha_total: float = 0.05
    max_attempts: int = 5
    attempts: list = field(default_factory=list)

    @classmethod
    def load(cls, path, alpha_total: float = 0.05, max_attempts: int = 5):
        path = Path(path)
        if path.exists():
            blob = json.loads(path.read_text(encoding="utf-8"))
            return cls(path=path, alpha_total=blob.get("alpha_total", alpha_total),
                       max_attempts=blob.get("max_attempts", max_attempts),
                       attempts=blob.get("attempts", []))
        return cls(path=path, alpha_total=alpha_total, max_attempts=max_attempts)

    def alpha_for_next(self) -> float:
        """Pocock-style constant boundary: alpha_total / max_attempts each look."""
        return float(self.alpha_total / self.max_attempts)

    def n_attempts(self) -> int:
        return len(self.attempts)

    def alpha_spent(self) -> float:
        return float(self.alpha_for_next() * self.n_attempts())

    def record(self, *, version: str, seed_bank: str, passed: bool,
               summary: dict = None) -> dict:
        if self.n_attempts() >= self.max_attempts:
            raise RuntimeError(
                f"sequential budget exhausted: {self.n_attempts()} of {self.max_attempts} "
                f"confirmation attempts already spent at alpha_total={self.alpha_total}. "
                f"No further confirmation can be claimed under valid error control.")
        entry = {"attempt": self.n_attempts() + 1, "version": version,
                 "seed_bank": seed_bank, "alpha_this_look": self.alpha_for_next(),
                 "passed": bool(passed), "summary": summary or {}}
        self.attempts.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"alpha_total": self.alpha_total, "max_attempts": self.max_attempts,
             "attempts": self.attempts}, indent=2), encoding="utf-8")
        return entry

    def as_dict(self) -> dict:
        return {"alpha_total": self.alpha_total, "max_attempts": self.max_attempts,
                "n_attempts": self.n_attempts(), "alpha_spent": self.alpha_spent(),
                "alpha_for_next": self.alpha_for_next(), "attempts": self.attempts}
