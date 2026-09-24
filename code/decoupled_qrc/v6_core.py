"""
v6_core.py -- targets, point evaluation, statistics and gates for V6.

Built ON the V4 audit's IPC primitives (audit_ipc: Legendre targets, splits,
ridge / OLS readouts, unclipped capacity, nulls), which are imported and never
edited -- they are hashed in the V4 preregistration and the V5.4 freeze.

What V6 adds
------------
* Four combined nonlinear-memory classes, each required INDEPENDENTLY:
    V6_C1  P2(u_t) P1(u_{t-tau})           current nonlinear x old linear
    V6_C2  P2(u_{t-tau})                   delayed nonlinear
    V6_C3  P1(u_{t-t1}) P1(u_{t-t2})       old x old, distinct delays
    V6_C4  P3(u_{t-tau})                   delayed cubic
  delays 1..6 (C3: all 15 pairs 1 <= t1 < t2 <= 6).
* An unreachable-target sentinel: P5 of a single input at delay 0..3. Every
  V6 processing path sees at most 3 copies of any one input, so a degree-5
  function of a single input is structurally inaccessible.
* Routes R (memory), P (instantaneous nonlinear), J (joint R x P), Q
  (nonlinear memory); operational readout ALL = R+P+J+Q; classical baselines
  CLS_M (matched budget) and CLS_F (full degree-2 expansion of marginals).
* Bootstrap at arbitrary quantiles (Bonferroni over many contrasts needs
  quantiles far below the fixed per-mille set of audit_core.boot).
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audit_ipc as A

METHODS = ("ridge", "ols_std", "ols_raw")
V6_TAUS = tuple(range(1, 7))
V6_CLASSES = ("V6_C1", "V6_C2", "V6_C3", "V6_C4")
SENTINELS = ("SENTINEL_future", "SENTINEL_unreachable")
REPORT_CLASSES = V6_CLASSES + SENTINELS + ("x_P1P1", "x_P2P1", "C1_curNL_x_oldLin",
                                           "C2_oldNL", "C3_old_x_old", "C4_old3")


def v6_targets() -> list:
    L = []
    for tau in V6_TAUS:
        L.append(A.T(A._nm(((0, 2), (tau, 1))), "V6_C1", ((0, 2), (tau, 1))))
    for tau in V6_TAUS:
        L.append(A.T(A._nm(((tau, 2),)), "V6_C2", ((tau, 2),)))
    for t1, t2 in itertools.combinations(V6_TAUS, 2):
        L.append(A.T(A._nm(((t1, 1), (t2, 1))), "V6_C3", ((t1, 1), (t2, 1))))
    for tau in V6_TAUS:
        L.append(A.T(A._nm(((tau, 3),)), "V6_C4", ((tau, 3),)))
    for tau in range(0, 4):
        L.append(A.T(A._nm(((tau, 5),)), "SENTINEL_unreachable", ((tau, 5),)))
    return L


def v6_library(tau_max: int = 8, max_degree: int = 4) -> list:
    return A.build_library(tau_max=tau_max, max_degree=max_degree) + v6_targets()


@dataclass
class Ctx:
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
    dtype: str = "float64"
    lib: list = field(default=None, repr=False)

    def __post_init__(self):
        if self.lib is None:
            self.lib = v6_library(self.tau_max, self.max_degree)

    @property
    def split(self) -> A.Split:
        return A.Split.standard(self.T, self.washout, self.train_frac, self.gap)

    def as_dict(self) -> dict:
        return {"T": self.T, "tau_max": self.tau_max, "tau_L": self.tau_L, "tau_S": self.tau_S,
                "n_null": self.n_null, "alpha": self.alpha, "washout": self.washout,
                "train_frac": self.train_frac, "gap": self.gap, "max_degree": self.max_degree,
                "dtype": self.dtype, "n_targets": len(self.lib), "pinv_cutoff": A.PINV_CUTOFF}


_YC: dict = {}


def targets_for(ctx: Ctx, input_seed: int):
    key = (int(input_seed), ctx.T, len(ctx.lib), ctx.tau_max, ctx.max_degree, ctx.n_null)
    if key not in _YC:
        u = np.random.default_rng(int(input_seed)).uniform(-1.0, 1.0, ctx.T)
        Y = np.hstack([A.build_Y(u, ctx.lib),
                       A.null_targets(ctx.T, ctx.n_null, seed=int(input_seed) + 7_000_003)])
        if len(_YC) > 32:
            _YC.clear()
        _YC[key] = (u, Y)
    return _YC[key]


def _lib_null(ctx: Ctx) -> list:
    return list(ctx.lib) + [A.T(f"null{i}", "NULL", ((0, 1),)) for i in range(ctx.n_null)]


def summarise(sc: np.ndarray, ctx: Ctx) -> dict:
    lib, n = ctx.lib, ctx.n_null
    real, null = sc[:-n], sc[-n:]
    names = [t.name for t in lib]
    cls = np.array([t.cls for t in lib])
    single = {f"d{d}_t{tau}": float(real[names.index(f"P{d}(t-{tau})")])
              for d in range(1, ctx.max_degree + 1) for tau in range(ctx.tau_max + 1)}
    classes, members = {}, {}
    for c in REPORT_CLASSES:
        v = real[cls == c]
        if v.size:
            classes[c] = float(np.nanmean(v))
            classes[c + "__max"] = float(np.nanmax(v))
            if c in V6_CLASSES or c in SENTINELS:
                members[c] = [float(x) for x in v]
    primary = [real[i] for i, t in enumerate(lib)
               if t.cls.startswith("single") or t.cls.startswith("x_") or t.cls in V6_CLASSES]
    s = single
    return {"single": single, "class": classes, "members": members,
            "null_q99": float(np.quantile(null, 0.99)), "null_mean": float(null.mean()),
            "saturated_fraction": float(np.mean(np.array(primary) > 0.995)),
            "metrics": {"M": float(np.mean([s[f"d1_t{t}"] for t in range(ctx.tau_L, ctx.tau_max + 1)])),
                        "N": float(np.mean([s[f"d{d}_t{t}"] for d in (2, 3, 4)
                                            for t in range(ctx.tau_S + 1)]))}}


def classical_blocks(f: dict) -> dict:
    """Classical baselines from MEASURED MARGINALS only.

    CLS_M: same feature count as the quantum operational readout -- marginals
           plus exactly the products of marginals that correspond to each quantum
           joint observable (J_r <-> zR_r * fY ;  Q pair (a,b) <-> zQ_a * zQ_b).
    CLS_F: marginals plus EVERY same-time degree-2 product of marginals."""
    marg = np.hstack([f["R"], f["P"], f["PY"], f["Qz"]])
    prods = [f["R"] * f["PY"]]
    for a, b in f["Q_pairs_rails_idx"]:
        prods.append((f["Qall"][:, a] * f["Qall"][:, b])[:, None])
    cls_m = np.hstack([f["R"], f["P"], np.hstack(prods), f["Qz"]])
    iu = np.triu_indices(marg.shape[1])
    cls_f = np.hstack([marg, (marg[:, :, None] * marg[:, None, :])[:, iu[0], iu[1]]])
    return {"CLS_M": cls_m, "CLS_F": cls_f}


def evaluate_point(adapter, m: float, g: float, seed_pair, ctx: Ctx, *, readouts=None,
                   feature_hook=None, classical: bool = True) -> dict:
    a_seed, i_seed = int(seed_pair[0]), int(seed_pair[1])
    u, Y = targets_for(ctx, i_seed)
    t0 = time.perf_counter()
    f = adapter.run(u, m, g, a_seed)
    X = {"R": f["R"], "P": f["P"], "J": f["J"], "Q": f["Q"]}
    X["ALL"] = np.hstack([X[k] for k in ("R", "P", "J", "Q") if X[k].size])
    if classical and "PY" in f:
        X.update(classical_blocks(f))
    if feature_hook is not None:
        X = feature_hook(X, u=u, m=m, g=g)
    sim_s = time.perf_counter() - t0
    readouts = readouts or {k: METHODS for k in X}
    blocks, counts = {}, {}
    lib = _lib_null(ctx)
    for ro, meths in readouts.items():
        if ro not in X:
            continue
        counts[ro] = int(X[ro].shape[1])
        for meth in meths:
            sc = A.score_all(X[ro], Y, lib, ctx.split, meth, alpha=ctx.alpha,
                             dtype=np.dtype(ctx.dtype))
            blocks[f"{ro}|{meth}"] = summarise(sc, ctx)
    return {"m": float(m), "g": float(g), "arch_seed": a_seed, "input_seed": i_seed,
            "blocks": blocks, "feature_counts": counts, "sim_seconds": sim_s,
            "split": ctx.split.check()}


def run_factorial(adapter, *, m_values, g_values, seeds, ctx: Ctx, checkpoint_path, tag: str,
                  readouts=None, feature_hook=None, verbose=True, classical=True) -> list:
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
    rows, t0 = [], time.time()
    total = len(seeds) * len(m_values) * len(g_values)
    for sp in seeds:
        for m in m_values:
            for g in g_values:
                key = f"{tag}|a{sp[0]}|i{sp[1]}|m{m}|g{g}"
                if key in done:
                    rows.append(done[key])
                    continue
                rec = evaluate_point(adapter, m, g, sp, ctx, readouts=readouts,
                                     feature_hook=feature_hook, classical=classical)
                rec["key"], rec["tag"] = key, tag
                with open(cp, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                rows.append(rec)
                if verbose and len(rows) % 25 == 0:
                    print(f"    [{tag}] {len(rows)}/{total} points ({time.time() - t0:.0f}s)",
                          flush=True)
    return rows


# =============================================================================
# Statistics
# =============================================================================
def boot_q(values, qs, *, n_boot: int = 100_000, seed: int = 0) -> dict:
    """Seed-level bootstrap of the mean at arbitrary quantiles."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"point": float("nan"), "n": 0, "q": {}}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(n_boot)
    for s in range(0, n_boot, 20_000):                       # bounded memory
        e = min(n_boot, s + 20_000)
        draws[s:e] = v[rng.integers(0, v.size, size=(e - s, v.size))].mean(axis=1)
    return {"point": float(v.mean()), "n": int(v.size), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "q": {repr(float(q)): float(np.quantile(draws, q)) for q in qs}, "values": v.tolist()}


def grid_by_seed(rows, getter) -> dict:
    out = {}
    for r in rows:
        out.setdefault((r["arch_seed"], r["input_seed"]), {})[(r["m"], r["g"])] = getter(r)
    return out


def effect_per_seed(grid: dict, control: str, lo=None, hi=None) -> list:
    eff = []
    for _, cells in grid.items():
        ms = sorted({k[0] for k in cells}); gs = sorted({k[1] for k in cells})
        a, b = (lo if lo is not None else (ms[0] if control == "m" else gs[0]),
                hi if hi is not None else (ms[-1] if control == "m" else gs[-1]))
        others = gs if control == "m" else ms
        d = [cells[(b, o)] - cells[(a, o)] if control == "m" else cells[(o, b)] - cells[(o, a)]
             for o in others if ((b, o) if control == "m" else (o, b)) in cells]
        eff.append(float(np.mean(d)))
    return eff


def quadrant_values(rows, key, getter, lows, highs) -> dict:
    g = grid_by_seed([r for r in rows if key in r["blocks"]], lambda r: getter(r["blocks"][key]))
    q = {k: [] for k in ("LL", "LH", "HL", "HH")}
    for _, cells in g.items():
        def mean_q(ms, gs):
            v = [cells[(m, gg)] for m in ms for gg in gs if (m, gg) in cells]
            return float(np.mean(v)) if v else float("nan")
        q["LL"].append(mean_q(lows, lows)); q["LH"].append(mean_q(lows, highs))
        q["HL"].append(mean_q(highs, lows)); q["HH"].append(mean_q(highs, highs))
    return {k: np.asarray(v) for k, v in q.items()}
