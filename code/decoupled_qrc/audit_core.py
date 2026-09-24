"""
audit_core.py -- per-point evaluation, factorial runner, statistics and gates for
the independent V4 audit (and, unchanged, for any later architecture).

An architecture enters through an ADAPTER exposing `run(u, m, g, seed)` that
returns the route-local feature blocks {'R', 'P', 'J'}. Everything downstream
is architecture-agnostic, so a redesigned architecture is judged by exactly
the same code as the one it replaces.

THREE READOUT ANALYSES are computed at every point, from one code path:
    ridge    -- the preregistered fixed penalty (V4's resource-constrained metric)
    ols_std  -- scale-invariant: standardised features, pseudo-inverse
    ols_raw  -- unregularised on centred raw features
Raw capacities are never clipped; clipped variants are carried alongside only
so the V4 definition can be reproduced for comparison.

UNCERTAINTY is a seed-level bootstrap: each seed is one (architecture, input)
pair and the whole grid is re-measured on it, so the unit of resampling is the
unit of replication.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audit_ipc as A

METHODS = ("ridge", "ols_std", "ols_raw")
READOUTS = {"R": METHODS, "P": METHODS, "J": METHODS,
            "ALL": ("ridge", "ols_std"), "CLS": ("ridge", "ols_std")}
CLASSES = ("x_P1P1", "x_P2P1", "x_P1P1P1", "C1_curNL_x_oldLin", "C2_oldNL",
           "C3_old_x_old", "C4_old3", "V4_NLONG", "SENTINEL_future")


# =============================================================================
# Adapter for the frozen V4 architecture
# =============================================================================
class V4Adapter:
    """Wraps the FROZEN V4 candidate. Nothing about V4 is changed here."""

    name = "V4"

    def __init__(self, frozen_path):
        from .v4_search import build
        blob = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        self.frozen_sha256 = blob["sha256"]
        self.candidate = blob["payload"]["architecture"]["candidate"]
        self.spec, self.cfg = build(self.candidate)
        self.alpha = float(self.cfg.alpha)

    def run(self, u, m, g, seed):
        from .v4_architecture import run_v4
        r = run_v4(self.spec, np.asarray(u, dtype=float), m=float(m), g=float(g),
                   seed=int(seed))
        return {"R": r.X_R, "P": r.X_P, "J": r.X_J,
                "labels_R": list(r.labels_R), "labels_P": list(r.labels_P),
                "labels_J": list(r.labels_J)}

    def describe(self) -> dict:
        return {"name": self.name, "frozen_sha256": self.frozen_sha256,
                "candidate": self.candidate, "alpha": self.alpha,
                "spec": self.spec.as_dict(), "resources": self.spec.resources()}


def classical_products(XR: np.ndarray, XP: np.ndarray) -> np.ndarray:
    """ALL pairwise products of the local R and P features, formed classically."""
    if XR.size == 0 or XP.size == 0:
        return np.empty((XR.shape[0], 0))
    return (XR[:, :, None] * XP[:, None, :]).reshape(XR.shape[0], -1)


# =============================================================================
# Point evaluation
# =============================================================================
@dataclass
class Context:
    T: int = 1600
    tau_max: int = 8
    tau_L: int = 2
    tau_S: int = 0
    n_null: int = 120
    alpha: float = 0.5
    washout: int = 60
    train_frac: float = 0.65
    gap: int = 12
    max_degree: int = 4
    dtype: str = "float64"          # readout precision (robustness: float32)
    lib: list = field(default=None, repr=False)

    def __post_init__(self):
        if self.lib is None:
            self.lib = A.build_library(tau_max=self.tau_max, tau_L=self.tau_L,
                                       max_degree=self.max_degree)

    @property
    def split(self) -> A.Split:
        return A.Split.standard(self.T, self.washout, self.train_frac, self.gap)

    def as_dict(self) -> dict:
        return {"T": self.T, "tau_max": self.tau_max, "tau_L": self.tau_L,
                "tau_S": self.tau_S, "n_null": self.n_null, "alpha": self.alpha,
                "washout": self.washout, "train_frac": self.train_frac, "gap": self.gap,
                "n_targets": len(self.lib), "pinv_cutoff": A.PINV_CUTOFF,
                "max_degree": self.max_degree, "dtype": self.dtype}


_Y_CACHE: dict = {}


def _targets(ctx: Context, input_seed: int):
    key = (int(input_seed), ctx.T, len(ctx.lib), ctx.tau_max, ctx.max_degree)
    if key not in _Y_CACHE:
        u = np.random.default_rng(int(input_seed)).uniform(-1.0, 1.0, ctx.T)
        Y = A.build_Y(u, ctx.lib)
        Nn = A.null_targets(ctx.T, ctx.n_null, seed=int(input_seed) + 7_000_003)
        if len(_Y_CACHE) > 64:
            _Y_CACHE.clear()
        _Y_CACHE[key] = (u, Y, Nn)
    return _Y_CACHE[key]


def summarise_scores(sc: np.ndarray, lib: list, n_null: int, ctx: Context) -> dict:
    """Compress one (readout, method) score vector into what the gates need."""
    real, null = sc[:-n_null], sc[-n_null:]
    names = [t.name for t in lib]
    cls = np.array([t.cls for t in lib])
    single = {}
    for d in range(1, ctx.max_degree + 1):
        for tau in range(ctx.tau_max + 1):
            single[f"d{d}_t{tau}"] = float(real[names.index(f"P{d}(t-{tau})")])
    cmeans = {}
    for c in CLASSES:
        v = real[cls == c]
        if v.size:
            cmeans[c] = float(np.nanmean(v))
            cmeans[c + "__clip"] = float(np.nanmean(np.clip(v, 0.0, 1.0)))
            cmeans[c + "__max"] = float(np.nanmax(v))
    sun = {}
    for i, t in enumerate(lib):
        if t.cls == "sunada":
            sun[t.name] = float(real[i])
    primary = [real[i] for i, t in enumerate(lib)
               if t.cls.startswith("single") or t.cls.startswith("x_")]
    return {"single": single, "class": cmeans, "sunada": sun,
            "null_q99": float(np.quantile(null, 0.99)),
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "saturated_fraction": float(np.mean(np.array(primary) > 0.995))}


def derived_metrics(block: dict, ctx: Context) -> dict:
    """M, N, N_long from a readout block. Unclipped primary, clipped for V4."""
    s = block["single"]
    Mv = [s[f"d1_t{t}"] for t in range(ctx.tau_L, ctx.tau_max + 1)]
    Nv = [s[f"d{d}_t{t}"] for d in (2, 3, 4) for t in range(ctx.tau_S + 1)]
    return {"M": float(np.mean(Mv)), "M_clip": float(np.mean(np.clip(Mv, 0, 1))),
            "N": float(np.mean(Nv)), "N_clip": float(np.mean(np.clip(Nv, 0, 1)))}


def evaluate_point(adapter, m: float, g: float, seed_pair, ctx: Context, *,
                   readouts=READOUTS, feature_hook=None) -> dict:
    """Every readout x analysis at one (m, g, seed). `feature_hook` lets the
    invalid-control and robustness stages perturb features without touching
    the architecture."""
    a_seed, i_seed = int(seed_pair[0]), int(seed_pair[1])
    u, Y, Nn = _targets(ctx, i_seed)
    t0 = time.perf_counter()
    f = adapter.run(u, m, g, a_seed)
    X = {"R": f["R"], "P": f["P"], "J": f["J"]}
    X["ALL"] = np.hstack([X[k] for k in ("R", "P", "J") if X[k].size])
    X["CLS"] = classical_products(X["R"], X["P"])
    if feature_hook is not None:
        X = feature_hook(X, u=u, m=m, g=g)
    sim_s = time.perf_counter() - t0
    split = ctx.split
    Yall = np.hstack([Y, Nn])
    blocks, counts, spectra = {}, {}, {}
    for ro, methods in readouts.items():
        if ro not in X:
            continue
        counts[ro] = int(X[ro].shape[1])
        spectra[ro] = {k: v for k, v in A.spectrum(X[ro], split).items()
                       if k != "singular_values"}
        for meth in methods:
            sc = A.score_all(X[ro], Yall, _lib_with_null(ctx), split, meth,
                             alpha=ctx.alpha, dtype=np.dtype(ctx.dtype))
            blk = summarise_scores(sc, ctx.lib, ctx.n_null, ctx)
            blk["metrics"] = derived_metrics(blk, ctx)
            blocks[f"{ro}|{meth}"] = blk
    return {"m": float(m), "g": float(g), "arch_seed": a_seed, "input_seed": i_seed,
            "blocks": blocks, "feature_counts": counts, "spectra": spectra,
            "sim_seconds": sim_s, "split": split.check()}


_LIB_NULL: dict = {}


def _lib_with_null(ctx: Context) -> list:
    key = (id(ctx.lib), ctx.n_null)
    if key not in _LIB_NULL:
        _LIB_NULL[key] = list(ctx.lib) + [A.T(f"null{i}", "NULL", ((0, 1),))
                                          for i in range(ctx.n_null)]
    return _LIB_NULL[key]


# =============================================================================
# Factorial runner with checkpointing
# =============================================================================
def run_factorial(adapter, *, m_values, g_values, seeds, ctx: Context,
                  checkpoint_path, tag: str, readouts=READOUTS, feature_hook=None,
                  verbose: bool = True) -> list:
    """Full grid over seeds. One JSONL row per point; resumes from checkpoint."""
    cp = Path(checkpoint_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["key"]] = r
            except Exception:
                continue
    rows = []
    total = len(seeds) * len(m_values) * len(g_values)
    t0 = time.time()
    for sp in seeds:
        for m in m_values:
            for g in g_values:
                key = f"{tag}|a{sp[0]}|i{sp[1]}|m{m}|g{g}"
                if key in done:
                    rows.append(done[key])
                    continue
                rec = evaluate_point(adapter, m, g, sp, ctx, readouts=readouts,
                                     feature_hook=feature_hook)
                rec["key"] = key
                rec["tag"] = tag
                with open(cp, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                rows.append(rec)
                if verbose and len(rows) % 25 == 0:
                    el = time.time() - t0
                    print(f"    [{tag}] {len(rows)}/{total} points  ({el:.0f}s)", flush=True)
    return rows


# =============================================================================
# Statistics
# =============================================================================
def boot(values, *, n_boot: int = 10000, seed: int = 0, stat=np.mean) -> dict:
    """Seed-level bootstrap of a statistic over per-seed values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"point": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    draws = stat(v[idx], axis=1)
    q = {f"q{p}": float(np.quantile(draws, p / 1000)) for p in (5, 10, 25, 50, 950, 975, 990, 995)}
    return {"point": float(stat(v)), "n": int(v.size), "sd": float(v.std(ddof=1))
            if v.size > 1 else 0.0, **q, "values": [float(x) for x in v]}


def boot_ratio(num, den, *, n_boot: int = 10000, seed: int = 0) -> dict:
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, num.size, size=(int(n_boot), num.size))
    r = np.abs(num[idx].mean(axis=1)) / np.maximum(np.abs(den[idx].mean(axis=1)), 1e-300)
    pt = abs(num.mean()) / max(abs(den.mean()), 1e-300)
    return {"ratio": float(pt), "q950": float(np.quantile(r, 0.95)),
            "q975": float(np.quantile(r, 0.975)), "q990": float(np.quantile(r, 0.99))}


def per_seed_grid(rows, key_fn) -> dict:
    """{seed_pair: {(m, g): value}}."""
    out = {}
    for r in rows:
        sp = (r["arch_seed"], r["input_seed"])
        out.setdefault(sp, {})[(r["m"], r["g"])] = key_fn(r)
    return out


def effect_per_seed(grid_by_seed: dict, control: str, lo=None, hi=None) -> list:
    """Delta over `control`, averaged across the other control, per seed."""
    vals = []
    for sp, grid in grid_by_seed.items():
        ms = sorted({k[0] for k in grid})
        gs = sorted({k[1] for k in grid})
        if control == "m":
            a, b = (lo if lo is not None else ms[0]), (hi if hi is not None else ms[-1])
            d = [grid[(b, g)] - grid[(a, g)] for g in gs if (a, g) in grid and (b, g) in grid]
        else:
            a, b = (lo if lo is not None else gs[0]), (hi if hi is not None else gs[-1])
            d = [grid[(m, b)] - grid[(m, a)] for m in ms if (m, a) in grid and (m, b) in grid]
        vals.append(float(np.mean(d)) if d else float("nan"))
    return vals


# =============================================================================
# Gates
# =============================================================================
@dataclass
class Levels:
    """CI quantiles. 'default' = the user's levels; 'sequential' = stricter
    levels implied by the repository's alpha-spending ledger (alpha 0.01/look)."""

    main_lo: str = "q25"         # lower bound of a two-sided 95% CI
    tost_lo: str = "q50"         # 90% two-sided interval
    tost_hi: str = "q950"
    ratio_hi: str = "q975"

    @classmethod
    def sequential(cls):
        return cls(main_lo="q10", tost_lo="q10", tost_hi="q990", ratio_hi="q990")


def metric_key(readout: str, method: str, name: str):
    return lambda r: r["blocks"][f"{readout}|{method}"]["metrics"][name]


def single_key(readout: str, method: str, d: int, tau: int):
    return lambda r: r["blocks"][f"{readout}|{method}"]["single"][f"d{d}_t{tau}"]


def class_key(readout: str, method: str, cls: str):
    return lambda r: r["blocks"][f"{readout}|{method}"]["class"][cls]


def gate_main(rows, readout, method, name, control, *, lv: Levels, thr=0.10,
              lo=None, hi=None, seed=0) -> dict:
    g = per_seed_grid(rows, metric_key(readout, method, name))
    b = boot(effect_per_seed(g, control, lo, hi), seed=seed)
    lb = b.get(lv.main_lo, float("nan"))
    ok = bool(np.isfinite(b["point"]) and b["point"] >= thr and lb > 0)
    return {"estimate": b["point"], "lower_bound": lb, "ci_level": lv.main_lo,
            "threshold": thr, "passed": ok, "n_seeds": b["n"], "sd": b.get("sd"),
            "per_seed": b.get("values")}


def gate_cross(rows, readout, method, name, control, main_readout, main_name,
               main_control, *, lv: Levels, margin=0.03, ratio_max=0.20, seed=0) -> dict:
    gc = per_seed_grid(rows, metric_key(readout, method, name))
    gm = per_seed_grid(rows, metric_key(main_readout, method, main_name))
    cross = effect_per_seed(gc, control)
    main = effect_per_seed(gm, main_control)
    b = boot(cross, seed=seed)
    lo, hi = b.get(lv.tost_lo, np.nan), b.get(lv.tost_hi, np.nan)
    rat = boot_ratio(cross, main, seed=seed + 1)
    rhi = rat.get(lv.ratio_hi, np.nan)
    eq = bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -margin and hi <= margin)
    ok = bool(eq and np.isfinite(rhi) and rhi < ratio_max)
    return {"estimate": b["point"], "interval": [lo, hi], "margin": [-margin, margin],
            "equivalent": eq, "ratio": rat["ratio"], "ratio_upper": rhi,
            "ratio_max": ratio_max, "passed": ok, "n_seeds": b["n"]}


def gate_profile_equivalence(rows, readout, method, members, control, *, lv: Levels,
                             margin=0.03, seed=0) -> dict:
    """Bonferroni-corrected simultaneous TOST over a family of capacities.

    Bonferroni is at least as strict as Holm; if every member passes it, every
    member passes Holm."""
    k = len(members)
    alpha = 0.05 if lv.tost_lo == "q50" else 0.01
    a_adj = alpha / k
    res, ok = {}, True
    for (d, tau) in members:
        g = per_seed_grid(rows, single_key(readout, method, d, tau))
        v = np.asarray(effect_per_seed(g, control), float)
        rng = np.random.default_rng(seed + d * 100 + tau)
        idx = rng.integers(0, v.size, size=(10000, v.size))
        draws = v[idx].mean(axis=1)
        lo, hi = float(np.quantile(draws, a_adj)), float(np.quantile(draws, 1 - a_adj))
        eq = bool(lo >= -margin and hi <= margin)
        ok &= eq
        res[f"d{d}_t{tau}"] = {"estimate": float(v.mean()), "interval": [lo, hi],
                               "equivalent": eq}
    return {"members": res, "n_members": k, "alpha_per_member": a_adj,
            "correction": "Bonferroni (at least as strict as Holm)", "passed": bool(ok)}


def quadrant_contrasts(rows, readout, method, cls, *, lows=(0.0, 0.25), highs=(0.75, 1.0),
                       lv: Levels, seed=0) -> dict:
    """HH minus each other quadrant, per seed, bootstrapped."""
    g = per_seed_grid(rows, class_key(readout, method, cls))
    q = {k: [] for k in ("LL", "LH", "HL", "HH")}
    for sp, grid in g.items():
        def mean_q(ms, gs):
            v = [grid[(m, gg)] for m in ms for gg in gs if (m, gg) in grid]
            return float(np.mean(v)) if v else float("nan")
        q["LL"].append(mean_q(lows, lows))
        q["LH"].append(mean_q(lows, highs))
        q["HL"].append(mean_q(highs, lows))
        q["HH"].append(mean_q(highs, highs))
    q = {k: np.asarray(v) for k, v in q.items()}
    out = {"quadrant_means": {k: float(np.nanmean(v)) for k, v in q.items()}}
    ok = True
    for other in ("LL", "LH", "HL"):
        # fixed offsets: hash(str) is randomised per process and would make
        # the bootstrap -- and therefore the gate -- non-reproducible
        b = boot(q["HH"] - q[other], seed=seed + {"LL": 11, "LH": 23, "HL": 37}[other])
        lb = b.get(lv.main_lo, np.nan)
        out[f"HH_minus_{other}"] = {"estimate": b["point"], "lower_bound": lb}
        ok &= bool(np.isfinite(lb) and lb > 0)
    out["HH_beats_all"] = bool(ok)
    return out


def saturation_and_sentinel(rows, *, readout="ALL", method="ridge") -> dict:
    sat = max(r["blocks"][f"{readout}|{method}"]["saturated_fraction"] for r in rows
              if f"{readout}|{method}" in r["blocks"])
    sen = [r["blocks"][f"{readout}|{method}"]["class"]["SENTINEL_future__max"]
           - r["blocks"][f"{readout}|{method}"]["null_q99"] for r in rows]
    return {"worst_saturated_fraction": float(sat), "saturation_ok": bool(sat <= 0.20),
            "sentinel_excess_max": float(np.max(sen)),
            "sentinel_ok": bool(np.max(sen) <= 0.02)}


def feature_count_invariance(rows) -> dict:
    counts = {}
    for r in rows:
        for ro, n in r["feature_counts"].items():
            counts.setdefault(ro, set()).add(n)
    return {"per_readout": {k: sorted(v) for k, v in counts.items()},
            "invariant": bool(all(len(v) == 1 for v in counts.values()))}
