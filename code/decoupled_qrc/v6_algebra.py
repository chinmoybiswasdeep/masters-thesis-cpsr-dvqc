"""
v6_algebra.py -- feature-span analysis done BEFORE any expensive simulation.

reachability(feature_fn)
    Long-sequence (T = 40 000) unregularised OLS capacity of every target of the
    four V6 classes and both sentinels from a feature map. With K features the
    finite-sample bias of an orthogonal target is ~K/T <= 0.002, so
        reachable      max member capacity >= 0.05
        unreachable    max member capacity <= 0.005
        indeterminate  otherwise (treated as NOT reachable)
exact_affinity(feature_fn)
    A map F of the input sequence is affine iff F(lam a + (1-lam) b) =
    lam F(a) + (1-lam) F(b) for all sequences a, b. Checked on random pairs; exact
    to rounding (tolerance 1e-10). Used for the encoder-leakage gate at g = 0.

V5.4 span theorem (proved in docs/V6_PROGRESS.md, verified numerically here)
    V5.4's operational features are z_r = sum_k A_rk u_{t-k} (k >= 1),
    f(u_t) = a1 u_t + a2 u_t^2 and z_r f(u_t). Every one is a polynomial in which
    each PAST input u_{t-k} (k >= 1) appears with degree <= 1, and no monomial
    contains two distinct past inputs. Under i.i.d. U[-1,1] inputs the
    Legendre products are orthonormal, so
        P2(u_{t-tau}), P3(u_{t-tau})       (degree >= 2 in one past input)
        P1(u_{t-t1}) P1(u_{t-t2}), t1 != t2 >= 1   (two past inputs)
    are orthogonal to the whole span: classes C2, C3, C4 have capacity EXACTLY 0.
"""
from __future__ import annotations

import numpy as np

from . import audit_ipc as A
from .v6_core import V6_CLASSES, v6_targets

REACH, UNREACH = 0.05, 0.005


def reachability(feature_fn, *, T: int = 40_000, seed: int = 990_001, washout: int = 200) -> dict:
    u = np.random.default_rng(seed).uniform(-1, 1, T)
    X = np.asarray(feature_fn(u), dtype=float)
    lib = v6_targets()
    Y = A.build_Y(u, lib)
    sp = A.Split(np.arange(washout, int(0.7 * T)), np.arange(int(0.7 * T) + 20, T))
    cap = A.capacity(A.predict(X, Y, sp, "ols_raw"), Y, sp)
    out = {}
    for c in V6_CLASSES + ("SENTINEL_unreachable",):
        v = np.array([cap[i] for i, t in enumerate(lib) if t.cls == c])
        mx = float(v.max())
        out[c] = {"mean": float(v.mean()), "max": mx, "n_targets": int(v.size),
                  "members": {t.name: float(cap[i]) for i, t in enumerate(lib) if t.cls == c},
                  "status": "reachable" if mx >= REACH else ("unreachable" if mx <= UNREACH
                                                             else "indeterminate")}
    out["n_features"] = int(X.shape[1])
    out["all_four_reachable"] = bool(all(out[c]["status"] == "reachable" for c in V6_CLASSES))
    out["sentinel_unreachable_null"] = bool(out["SENTINEL_unreachable"]["status"] == "unreachable")
    return out


def exact_affinity(feature_fn, *, T: int = 400, n_pairs: int = 5, seed: int = 990_002,
                   tol: float = 1e-10) -> dict:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_pairs):
        a, b = rng.uniform(-1, 1, T), rng.uniform(-1, 1, T)
        lam = float(rng.uniform(0.1, 0.9))
        Fa, Fb = np.asarray(feature_fn(a)), np.asarray(feature_fn(b))
        Fm = np.asarray(feature_fn(lam * a + (1 - lam) * b))
        worst = max(worst, float(np.abs(Fm - (lam * Fa + (1 - lam) * Fb)).max()))
    return {"max_superposition_violation": worst, "tolerance": tol, "affine": bool(worst <= tol)}


def v5_4_operational(spec, m: float = 1.0, g: float = 1.0):
    from .v5_architecture import V5Adapter
    ad = V5Adapter(spec)

    def fn(u):
        f = ad.run(u, m, g)
        return np.hstack([f["R"], f["P"], f["J"]])
    return fn


def v6_operational(adapter, m: float, g: float):
    def fn(u):
        f = adapter.run(u, m, g, 0)
        return np.hstack([f["R"], f["P"], f["J"], f["Q"]])
    return fn
