"""
audit_ipc.py -- an INDEPENDENT Information Processing Capacity engine.

Written for the adversarial audit of V4, deliberately NOT reusing `v4_ipc`, so
an error in V4's own estimator cannot silently reproduce itself here. It is
cross-checked against `v4_ipc.BoundedReadout` in `tests/test_audit.py`.

Differences from the V4 estimator, all deliberate:

  * RAW CAPACITIES ARE NOT CLIPPED. C = 1 - MSE/Var can be negative on held-out
    data (an overfit readout is worse than predicting the mean). Clipping hides
    that, and it also hides the finite-sample null bias. Clipping is applied
    only where a gate explicitly asks for it.
  * THREE READOUT ANALYSES from one code path, so they cannot drift apart:
        ridge    -- standardised features, fixed penalty (V4's definition)
        ols_std  -- standardised features, pseudo-inverse with a documented cutoff
        ols_raw  -- centred but UNstandardised features, same cutoff
    OLS is invariant to any invertible rescaling of the features, so ols_std
    and ols_raw may differ ONLY through the relative pseudo-inverse cutoff.
    Ridge is not scale invariant; that asymmetry is exactly what the audit
    probes.
  * EVERY target is evaluated in one vectorised solve per readout, so the
    target library cannot be quietly narrowed at one parameter point.
  * A FUTURE-INPUT SENTINEL (P_1(u_{t+k}), k > 0) is included. A causal
    reservoir fed i.i.d. input cannot predict it, so capacity above the null
    floor on the sentinel means leakage.

Targets: orthonormal Legendre P_d on U[-1,1], products over distinct delays;
Sunada targets sin(nu u_{t-tau})/nu with the exact nu -> 0 limit, scored by
corr^2 as in Sunada et al., Sci. Rep. 9, 19078 (2019).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as npleg

PINV_CUTOFF = 1e-10           # relative singular-value cutoff for the OLS analyses


def legendre(v, d: int) -> np.ndarray:
    """sqrt(2d+1) P_d(v): orthonormal under the U[-1,1] probability measure."""
    return np.sqrt(2 * d + 1) * npleg.legval(np.asarray(v, dtype=float), [0] * d + [1])


def shift(u: np.ndarray, tau: int) -> np.ndarray:
    """u_{t-tau}; tau < 0 gives the FUTURE input u_{t+|tau|} (sentinel only)."""
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    if tau == 0:
        out[:] = u
    elif tau > 0:
        out[tau:] = u[:-tau]
    else:
        k = -tau
        out[:-k] = u[k:]
    return out


@dataclass(frozen=True)
class T:
    """One target. `pairs` = ((delay, degree), ...); Sunada targets use nu."""

    name: str
    cls: str
    pairs: tuple = ()
    nu: float = None
    score: str = "cap"          # 'cap' = 1 - MSE/Var ;  'corr2' = Sunada score

    @property
    def max_delay(self) -> int:
        if self.nu is not None:
            return int(self.pairs[0][0])
        return max(t for t, _ in self.pairs)

    @property
    def degree_at_positive_delay(self) -> int:
        pos = [d for t, d in self.pairs if t > 0]
        return max(pos) if pos else 0

    def build(self, u: np.ndarray) -> np.ndarray:
        if self.nu is not None:
            tau = int(self.pairs[0][0])
            x = shift(u, tau)
            return x.copy() if self.nu == 0.0 else np.sin(self.nu * x) / self.nu
        y = np.ones(len(u))
        for tau, d in self.pairs:
            y = y * legendre(shift(u, tau), d)
        return y


def _nm(pairs):
    return "*".join(f"P{d}(t{'-' if t >= 0 else '+'}{abs(t)})" for t, d in sorted(pairs))


def build_library(*, tau_max: int = 8, max_degree: int = 4, tau_L: int = 2,
                  nlong_v4=(3, 5), sunada_nus=(0.0, 0.5, 1.0, 2.0, 4.0)) -> list:
    """The complete, preregistered target library. Identical at every point."""
    L = []
    taus = range(tau_max + 1)
    for d in range(1, max_degree + 1):
        for tau in taus:
            L.append(T(_nm(((tau, d),)), f"single_d{d}", ((tau, d),)))
    for t1, t2 in itertools.combinations(taus, 2):
        L.append(T(_nm(((t1, 1), (t2, 1))), "x_P1P1", ((t1, 1), (t2, 1))))
    for t1 in taus:
        for t2 in taus:
            if t1 != t2:
                L.append(T(_nm(((t1, 2), (t2, 1))), "x_P2P1", ((t1, 2), (t2, 1))))
    for t1, t2, t3 in itertools.combinations(taus, 3):
        L.append(T(_nm(((t1, 1), (t2, 1), (t3, 1))), "x_P1P1P1",
                   ((t1, 1), (t2, 1), (t3, 1))))
    # ---- the four combined nonlinear-memory classes (audit section 9) -------
    for d in (2, 3):
        for tau in range(tau_L + 1, tau_max + 1):
            L.append(T(_nm(((0, d), (tau, 1))), "C1_curNL_x_oldLin", ((0, d), (tau, 1))))
    for d in (2, 3, 4):
        for tau in range(1, tau_max + 1):
            L.append(T(_nm(((tau, d),)), "C2_oldNL", ((tau, d),)))
    for t1, t2 in itertools.combinations(range(1, tau_max + 1), 2):
        L.append(T(_nm(((t1, 1), (t2, 1))), "C3_old_x_old", ((t1, 1), (t2, 1))))
    for t1, t2, t3 in itertools.combinations(range(1, tau_max + 1), 3):
        L.append(T(_nm(((t1, 1), (t2, 1), (t3, 1))), "C4_old3",
                   ((t1, 1), (t2, 1), (t3, 1))))
    # ---- V4's own N_long library, to reproduce its number exactly -----------
    for d in (2, 3):
        for tau in range(nlong_v4[0], nlong_v4[1] + 1):
            L.append(T(_nm(((0, d), (tau, 1))), "V4_NLONG", ((0, d), (tau, 1))))
    # ---- Sunada ------------------------------------------------------------
    for nu in sunada_nus:
        for tau in taus:
            L.append(T(f"sunada_nu{nu}_t{tau}", "sunada", ((tau, 1),), nu=float(nu),
                       score="corr2"))
    # ---- causality sentinel ------------------------------------------------
    for k in (1, 2):
        L.append(T(_nm(((-k, 1),)), "SENTINEL_future", ((-k, 1),)))
    return L


def build_Y(u: np.ndarray, lib: list) -> np.ndarray:
    return np.column_stack([t.build(u) for t in lib])


def null_targets(T_len: int, n_null: int, seed: int) -> np.ndarray:
    """Independent targets: Legendre polynomials of an INDEPENDENT uniform
    sequence. They share the marginal statistics of real targets but carry no
    information about u, so their capacity distribution is the finite-sample
    null for the readout that produced it."""
    rng = np.random.default_rng(int(seed))
    cols = []
    for i in range(int(n_null)):
        v = rng.uniform(-1.0, 1.0, T_len)
        cols.append(legendre(v, 1 + (i % 4)))
    return np.column_stack(cols)


# =============================================================================
# Readouts
# =============================================================================
@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray

    @classmethod
    def standard(cls, T_len: int, washout: int = 60, train_frac: float = 0.65, gap: int = 12):
        tr = np.arange(washout, int(train_frac * T_len))
        te = np.arange(int(train_frac * T_len) + gap, T_len)
        return cls(tr, te)

    def check(self) -> dict:
        overlap = np.intersect1d(self.train, self.test)
        return {"disjoint": bool(overlap.size == 0), "n_train": int(len(self.train)),
                "n_test": int(len(self.test)), "overlap": int(overlap.size)}


def _prep(X, split: Split, standardize: bool):
    Xtr = X[split.train]
    mu = Xtr.mean(axis=0)
    if standardize:
        sd = Xtr.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
    else:
        sd = np.ones(X.shape[1])
    return (Xtr - mu) / sd, (X[split.test] - mu) / sd


def predict(X: np.ndarray, Y: np.ndarray, split: Split, method: str, *,
            alpha: float = None, cutoff: float = PINV_CUTOFF, dtype=np.float64) -> np.ndarray:
    """Held-out predictions for EVERY target column at once.

    method: 'ridge' (standardised, fixed alpha), 'ols_std', 'ols_raw'.
    """
    X = np.asarray(X, dtype=dtype)
    Y = np.asarray(Y, dtype=dtype)
    if X.shape[1] == 0:
        return np.tile(Y[split.train].mean(axis=0), (len(split.test), 1))
    standardize = method in ("ridge", "ols_std")
    Xtr, Xte = _prep(X, split, standardize)
    Ytr = Y[split.train]
    ybar = Ytr.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    if method == "ridge":
        if alpha is None or alpha < 0:
            raise ValueError("ridge needs alpha >= 0")
        filt = s / (s ** 2 + alpha) if alpha > 0 else np.where(s > cutoff * s.max(), 1 / s, 0.0)
    elif method in ("ols_std", "ols_raw"):
        keep = s > cutoff * (s.max() if s.size else 1.0)
        filt = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    else:
        raise ValueError(f"unknown method {method!r}")
    W = Vt.T @ (filt[:, None] * (U.T @ (Ytr - ybar)))
    return Xte @ W + ybar


def capacity(Yhat: np.ndarray, Y: np.ndarray, split: Split) -> np.ndarray:
    """C = 1 - MSE/Var on the held-out block. NOT CLIPPED."""
    Yte = Y[split.test]
    var = Yte.var(axis=0)
    mse = ((Yhat - Yte) ** 2).mean(axis=0)
    return np.where(var > 1e-12, 1.0 - mse / np.where(var > 1e-12, var, 1.0), np.nan)


def corr2(Yhat: np.ndarray, Y: np.ndarray, split: Split) -> np.ndarray:
    Yte = Y[split.test]
    a = Yhat - Yhat.mean(axis=0)
    b = Yte - Yte.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    return np.where(den > 1e-15, ((a * b).sum(axis=0) / np.where(den > 1e-15, den, 1)) ** 2,
                    0.0)


def score_all(X, Y, lib, split, method, **kw) -> np.ndarray:
    """Capacity (or corr^2 for Sunada targets) for every target in `lib`."""
    Yhat = predict(X, Y, split, method, **kw)
    cap = capacity(Yhat, Y, split)
    c2 = corr2(Yhat, Y, split)
    use_c2 = np.array([t.score == "corr2" for t in lib])
    return np.where(use_c2, c2, cap)


def spectrum(X: np.ndarray, split: Split, cutoff: float = PINV_CUTOFF) -> dict:
    """Singular values of the STANDARDISED training design."""
    if X.shape[1] == 0:
        return {"singular_values": [], "numerical_rank": 0, "effective_rank": 0.0}
    Xtr, _ = _prep(X, split, standardize=True)
    s = np.linalg.svd(Xtr, compute_uv=False)
    s2 = s ** 2
    return {"singular_values": [float(v) for v in s],
            "numerical_rank": int(np.sum(s > cutoff * s.max())),
            "effective_rank": float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0}


# =============================================================================
# Exact function-space analysis of a MEMORYLESS route (audit section 6, D)
# =============================================================================
def function_subspace(feature_fn, n_quad: int = 96, cutoff: float = 1e-10) -> dict:
    """Subspace spanned by the centred features of a memoryless map u -> f(u).

    Gauss-Legendre quadrature makes this EXACT for polynomial features, so it
    reports the infinite-data, noiseless capacity -- the quantity an
    unregularised readout converges to. No sampling, no seed.
    """
    x, w = np.polynomial.legendre.leggauss(int(n_quad))
    w = w / 2.0
    F = np.array([feature_fn(float(xi)) for xi in x], dtype=float)
    F = F - (w[:, None] * F).sum(axis=0)
    sw = np.sqrt(w)[:, None]
    A = sw * F
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > cutoff * max(s.max() if s.size else 0.0, 1e-300)
    Q = U[:, keep]
    proj = {}
    for d in range(1, 9):
        pw = np.sqrt(w) * legendre(x, d)
        proj[d] = float(np.clip((Q.T @ pw) @ (Q.T @ pw) / (pw @ pw), 0.0, 1.0))
    return {"Q": Q, "singular_values": [float(v) for v in s], "rank": int(keep.sum()),
            "degree_capacity": proj}


def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> dict:
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return {"angles_deg": [], "max_deg": float("nan"), "projection_distance": float("nan")}
    sv = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), -1.0, 1.0)
    ang = np.degrees(np.arccos(sv))
    P1, P2 = Q1 @ Q1.T, Q2 @ Q2.T
    return {"angles_deg": [float(a) for a in ang], "max_deg": float(ang.max()),
            "projection_distance": float(np.linalg.norm(P1 - P2, 2)),
            "same_dimension": bool(Q1.shape[1] == Q2.shape[1])}
