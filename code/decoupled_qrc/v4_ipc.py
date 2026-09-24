"""
v4_ipc.py -- degree-and-delay Information Processing Capacity for V4.

Method: Dambre et al., Sci. Rep. 2, 514 (2012), doi:10.1038/srep00514;
delay/degree decomposition following Llodra et al., Adv. Quantum Technol. 6,
2200100 (2023), doi:10.1002/qute.202200100.

One input stream u_t ~ U[-1,1]; orthonormal Legendre targets
    P_d(u_{t-tau})  and products over distinct delays;
linear trained readouts only; held-out capacity

    C_j = 1 - MSE(y_j, yhat_j) / Var(y_j)

THE BOUNDED READOUT IS PART OF THE MEASUREMENT, AND WHY
---------------------------------------------------------------------------
With an UNBOUNDED linear readout in exact arithmetic, capacity depends only on
the SPAN of the feature functions and not at all on coefficient magnitudes.
That makes every span-reachable capacity jump to 1.0 the instant a coupling is
switched on, and it was measured directly during development: with a
validation-selected ridge alpha, N(g) was 0.003 at g=0 and then pinned at
exactly 3.000 for EVERY g > 0, giving dN/dg = 0. No feature count and no
interaction strength could grade it.

V4 therefore measures capacity with a FIXED, preregistered ridge penalty
`alpha`. This is a bounded-norm readout: a component whose relative amplitude
is O(g) needs weights O(1/g) to extract, which the penalty forbids. It is a
RESOURCE CONSTRAINT -- identical at every (m, g), entered into the frozen
config, and counted in the resource table -- not a per-point tuned knob, and
not noise. `alpha` is never re-selected after results are seen.

This is explicitly NOT the forbidden move of using finite shots to manufacture
nonlinearity: the simulation is exact and noiseless everywhere, and the
readout bound applies equally to the memory route, the nonlinear route and
every baseline.

SATURATION IS A VALIDITY FAILURE, NOT A RESULT. If more than
`max_saturated_fraction` of the primary diagnostic targets exceed 0.995 the
metric is not measuring anything and `saturation_report` flags it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import legendre as npleg


def legendre_target(v: np.ndarray, degree: int) -> np.ndarray:
    """Orthonormal Legendre P_d on Uniform[-1,1]: sqrt(2d+1) * P_d."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    return np.sqrt(2 * degree + 1) * npleg.legval(np.asarray(v, dtype=float),
                                                  [0] * degree + [1])


@dataclass(frozen=True)
class Target:
    """A product of Legendre factors over DISTINCT delays.

    `pairs` is a canonical, sorted tuple of (delay, degree). Sorting makes the
    representation unique, so two constructions of the same target are the
    same object and cannot be double counted.
    """

    pairs: tuple

    def __post_init__(self):
        delays = [t for t, _ in self.pairs]
        if len(set(delays)) != len(delays):
            raise ValueError(f"delays must be distinct within a target: {self.pairs}")
        object.__setattr__(self, "pairs", tuple(sorted(self.pairs)))

    @property
    def total_degree(self) -> int:
        return sum(d for _, d in self.pairs)

    @property
    def max_delay(self) -> int:
        return max(t for t, _ in self.pairs)

    @property
    def min_delay(self) -> int:
        return min(t for t, _ in self.pairs)

    @property
    def n_factors(self) -> int:
        return len(self.pairs)

    @property
    def is_single_delay(self) -> bool:
        return len(self.pairs) == 1

    @property
    def degree_at_positive_delay(self) -> int:
        """Largest degree carried by a STRICTLY POSITIVE delay.

        >= 2 means a multilinear memory route cannot produce it -- the V4
        capability boundary (see `v4_encoder.reachable_monomial`).
        """
        pos = [d for t, d in self.pairs if t > 0]
        return max(pos) if pos else 0

    @property
    def reachable_by_product_architecture(self) -> bool:
        return self.degree_at_positive_delay < 2

    def key(self) -> str:
        return "*".join(f"P{d}(t-{t})" for t, d in self.pairs)

    def build(self, v: np.ndarray) -> np.ndarray:
        T = len(v)
        y = np.ones(T, dtype=float)
        for tau, d in self.pairs:
            shifted = np.zeros(T, dtype=float)
            if tau == 0:
                shifted[:] = v
            else:
                shifted[tau:] = v[:T - tau]
            y *= legendre_target(shifted, d)
        y[:self.max_delay] = 0.0
        return y


# =============================================================================
# Target libraries -- preregistered, built before any results are seen
# =============================================================================
def single_delay_targets(max_degree: int, max_delay: int) -> list:
    return [Target(((tau, d),)) for d in range(1, max_degree + 1)
            for tau in range(max_delay + 1)]


def cross_delay_targets(max_delay: int, *, seed: int = 0, max_per_family: int = 18) -> list:
    """P1*P1, P2*P1 and P1*P1*P1 over distinct delays.

    Enumeration is exhaustive and then SEEDED-subsampled if a family overflows,
    so the library is data-independent and reproducible.
    """
    rng = np.random.RandomState(int(seed))
    fams = {"P1P1": [], "P2P1": [], "P1P1P1": []}
    taus = list(range(max_delay + 1))
    for t1, t2 in itertools.combinations(taus, 2):
        fams["P1P1"].append(Target(((t1, 1), (t2, 1))))
    for t1 in taus:
        for t2 in taus:
            if t1 != t2:
                fams["P2P1"].append(Target(((t1, 2), (t2, 1))))
    for t1, t2, t3 in itertools.combinations(taus, 3):
        fams["P1P1P1"].append(Target(((t1, 1), (t2, 1), (t3, 1))))
    out = []
    for name, lst in fams.items():
        lst = sorted(set(lst), key=lambda t: t.pairs)
        if len(lst) > max_per_family:
            idx = rng.choice(len(lst), size=max_per_family, replace=False)
            lst = [lst[i] for i in sorted(idx)]
        out.extend(lst)
    return out


def nlong_targets(tau_L: int, max_delay: int, degrees=(2, 3), *,
                  reachable_only: bool = True, nlong_max_delay: int = None) -> list:
    """N_long library: a nonlinear factor at delay 0 times a linear factor at
    a delay beyond tau_L.

    With `reachable_only` (the default) every target satisfies the V4
    capability boundary: the degree >= 2 sits at delay 0, supplied by the
    processor route (control g), while the far delay is supplied by the memory
    route (control m). BOTH controls are therefore required, which is what
    gives the combined-capability gate its content.

    Setting `reachable_only=False` additionally returns single-delay
    high-degree targets at tau > 0. Those are PROVABLY unreachable by a
    multilinear memory route and are reported as a characterised capability
    boundary -- never placed inside a gate the architecture must pass.
    """
    hi = int(nlong_max_delay if nlong_max_delay is not None else max_delay)
    hi = min(hi, max_delay)
    out = [Target(((0, d), (tau, 1))) for d in degrees
           for tau in range(tau_L + 1, hi + 1)]
    if not reachable_only:
        out += [Target(((tau, d),)) for d in degrees
                for tau in range(tau_L + 1, hi + 1)]
    return out


# =============================================================================
# Bounded-norm ridge readout
# =============================================================================
class BoundedReadout:
    """One SVD of the standardised training design, reused for every target.

    `alpha` is FIXED. Standardisation uses train statistics only. The test
    block is touched once, after the weights are fully determined.
    """

    def __init__(self, X: np.ndarray, train, test, alpha: float):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, F), got {X.shape}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0 (a bounded readout), got {alpha}")
        self.train, self.test, self.alpha = np.asarray(train), np.asarray(test), float(alpha)
        mu = X[self.train].mean(axis=0)
        sd = X[self.train].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        self._Xtr = (X[self.train] - mu) / sd
        self._Xte = (X[self.test] - mu) / sd
        self._U, self._s, self._Vt = np.linalg.svd(self._Xtr, full_matrices=False)
        self.n_features = X.shape[1]

    @property
    def numerical_rank(self) -> int:
        return int(np.sum(self._s > 1e-10 * max(self._s[0], 1e-300)))

    def capacity(self, y: np.ndarray) -> float:
        """C = 1 - MSE/Var on the held-out test block, clipped to [0, 1]."""
        y = np.asarray(y, dtype=float)
        ytr = y[self.train]
        ybar = float(ytr.mean())
        filt = self._s / (self._s ** 2 + self.alpha)     # bounded-norm ridge filter
        w = self._Vt.T @ (filt * (self._U.T @ (ytr - ybar)))
        pred = self._Xte @ w + ybar
        yte = y[self.test]
        var = float(np.var(yte))
        if var < 1e-12:
            return 0.0
        return float(np.clip(1.0 - float(np.mean((pred - yte) ** 2)) / var, 0.0, 1.0))


# =============================================================================
# Null calibration -- preregistered BEFORE any architecture search
# =============================================================================
def null_threshold(readout: BoundedReadout, T: int, *, n_null: int = 200,
                   seed: int = 0, quantile: float = 0.99) -> dict:
    """Finite-sample null: capacities of INDEPENDENT random targets.

    Independent draws (not shuffles of a real target) keep the null free of any
    structure the true target shares with the features. The reported threshold
    is the `quantile` of that distribution and is fixed before the search.
    """
    rng = np.random.RandomState(int(seed))
    vals = np.empty(int(n_null))
    for i in range(int(n_null)):
        v_null = rng.uniform(-1.0, 1.0, T)
        d = 1 + (i % 4)
        vals[i] = readout.capacity(legendre_target(v_null, d))
    return {"threshold": float(np.quantile(vals, quantile)),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "max": float(vals.max()), "n_null": int(n_null),
            "quantile": float(quantile)}


def saturation_report(capacities, *, ceiling: float = 0.995,
                      max_saturated_fraction: float = 0.20) -> dict:
    """Automatic saturation failure (user requirement section 7)."""
    vals = [float(v) for v in capacities]
    n_sat = sum(1 for v in vals if v > ceiling)
    frac = (n_sat / len(vals)) if vals else 0.0
    return {"n_targets": len(vals), "n_saturated": n_sat,
            "saturated_fraction": float(frac), "ceiling": ceiling,
            "max_saturated_fraction": max_saturated_fraction,
            "saturated": bool(frac > max_saturated_fraction),
            "reason": ("ok" if frac <= max_saturated_fraction else
                       f"{n_sat}/{len(vals)} targets exceed {ceiling}: the metric is "
                       f"ceiling-pinned and is not measuring a response")}


# =============================================================================
# The V4 metric set
# =============================================================================
@dataclass
class IPCConfig:
    """Preregistered. Frozen before ARCHITECTURE_SEARCH; never edited after."""

    max_degree: int = 4
    max_delay: int = 8
    tau_L: int = 2                 # M sums delays >= tau_L
    tau_S: int = 0                 # N sums delays <= tau_S (short by design)
    degree_weights: tuple = (1.0, 1.0, 1.0)      # for d = 2, 3, 4
    alpha: float = 0.5             # FIXED bounded readout
    washout: int = 60
    train_frac: float = 0.65
    gap: int = 12
    nlong_max_delay: int = 5       # N_long delays stay inside the memory horizon
    n_null: int = 200
    null_quantile: float = 0.99
    cross_seed: int = 0
    max_per_family: int = 18

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def splits(self, T: int) -> tuple:
        tr_end = int(self.train_frac * T)
        train = np.arange(self.washout, tr_end)
        test = np.arange(tr_end + self.gap, T)
        if len(train) < 50 or len(test) < 50:
            raise ValueError(f"degenerate split at T={T}: train={len(train)} test={len(test)}")
        return train, test


@dataclass
class IPCResult:
    M: float
    N: float
    N_long: float
    M_raw: float
    N_raw: float
    N_long_raw: float
    delay_profile: dict
    degree_profile: dict
    cross_delay: dict
    cross_delay_families: dict
    H1: int
    saturation: dict
    null: dict
    n_features: dict
    numerical_rank: dict
    unreachable_probe: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _profile(readout, targets, v):
    return {t.key(): readout.capacity(t.build(v)) for t in targets}


def compute_metrics(run, cfg: IPCConfig, *, null_seed: int = 0,
                    include_unreachable_probe: bool = True) -> IPCResult:
    """M, N and N_long from a V4 run, each on its own route-restricted readout.

        M      from R-local observables ONLY  -> cannot move with g
        N      from P-local observables ONLY  -> cannot move with m
        N_long from the JOINT observables     -> needs both routes
    """
    v = np.asarray(run.u, dtype=float)
    T = len(v)
    train, test = cfg.splits(T)

    has_R = run.X_R.size > 0
    has_P = run.X_P.size > 0
    has_J = run.X_J.size > 0

    rd_R = BoundedReadout(run.X_R, train, test, cfg.alpha) if has_R else None
    rd_P = BoundedReadout(run.X_P, train, test, cfg.alpha) if has_P else None
    rd_J = BoundedReadout(run.X_J, train, test, cfg.alpha) if has_J else None

    # ---- M: linear memory at long delays, memory route only ----------------
    delay_profile, M_raw, H1 = {}, 0.0, -1
    if rd_R is not None:
        for tau in range(cfg.max_delay + 1):
            delay_profile[tau] = rd_R.capacity(Target(((tau, 1),)).build(v))
        M_raw = float(sum(delay_profile[t] for t in range(cfg.tau_L, cfg.max_delay + 1)))
        above = [t for t in sorted(delay_profile) if delay_profile[t] >= 0.5]
        H1 = int(max(above)) if above else -1
    Z_M = float(cfg.max_delay - cfg.tau_L + 1)

    # ---- N: local nonlinearity at short delays, processor route only -------
    degree_profile, N_raw = {}, 0.0
    if rd_P is not None:
        for i, d in enumerate((2, 3, 4)[:cfg.max_degree - 1]):
            w = cfg.degree_weights[i] if i < len(cfg.degree_weights) else 1.0
            for tau in range(cfg.tau_S + 1):
                c = rd_P.capacity(Target(((tau, d),)).build(v))
                degree_profile[f"d{d}_t{tau}"] = c
                N_raw += w * c
    Z_N = float(sum(cfg.degree_weights) * (cfg.tau_S + 1))

    # ---- N_long: joint route, reachable cross-delay targets ----------------
    nl_targets = nlong_targets(cfg.tau_L, cfg.max_delay, reachable_only=True,
                               nlong_max_delay=cfg.nlong_max_delay)
    cross = {}
    if rd_J is not None:
        cross = _profile(rd_J, nl_targets, v)
    N_long_raw = float(sum(cross.values()))
    Z_L = float(max(len(nl_targets), 1))

    # ---- the provably-unreachable class, reported never gated --------------
    probe = {}
    if include_unreachable_probe and rd_J is not None:
        unreachable = [t for t in nlong_targets(cfg.tau_L, cfg.max_delay,
                                                reachable_only=False,
                                                nlong_max_delay=cfg.nlong_max_delay)
                       if not t.reachable_by_product_architecture]
        probe = {"targets": [t.key() for t in unreachable],
                 "capacities": _profile(rd_J, unreachable, v),
                 "note": ("degree >= 2 at a strictly positive delay; a multilinear "
                          "memory route provably cannot produce these. Reported as a "
                          "capability boundary, never gated.")}

    # ---- required cross-delay IPC families (user section 4) ---------------
    # P1*P1, P2*P1 and P1*P1*P1 over distinct delays. These are part of the
    # PRIMARY diagnostic set, so they also enter the saturation audit -- a
    # metric judged on a handful of easy targets can look saturated purely
    # because the library was too small.
    xfam = {}
    if rd_J is not None:
        xfam = _profile(rd_J, cross_delay_targets(cfg.max_delay, seed=cfg.cross_seed,
                                                  max_per_family=cfg.max_per_family), v)

    # ---- validity audits ---------------------------------------------------
    primary = (list(delay_profile.values()) + list(degree_profile.values())
               + list(cross.values()) + list(xfam.values()))
    sat = saturation_report(primary)
    null = (null_threshold(rd_R if rd_R is not None else rd_P, T,
                           n_null=cfg.n_null, seed=null_seed,
                           quantile=cfg.null_quantile)
            if (rd_R is not None or rd_P is not None) else {})

    return IPCResult(
        M=float(M_raw / Z_M), N=float(N_raw / Z_N), N_long=float(N_long_raw / Z_L),
        M_raw=M_raw, N_raw=N_raw, N_long_raw=N_long_raw,
        delay_profile={int(k): float(x) for k, x in delay_profile.items()},
        degree_profile={k: float(x) for k, x in degree_profile.items()},
        cross_delay={k: float(x) for k, x in cross.items()},
        cross_delay_families={k: float(x) for k, x in xfam.items()},
        H1=H1, saturation=sat, null=null,
        n_features={"R": run.X_R.shape[1], "P": run.X_P.shape[1], "J": run.X_J.shape[1]},
        numerical_rank={"R": rd_R.numerical_rank if rd_R else 0,
                        "P": rd_P.numerical_rank if rd_P else 0,
                        "J": rd_J.numerical_rank if rd_J else 0},
        unreachable_probe=probe,
        config={**cfg.as_dict(), "Z_M": Z_M, "Z_N": Z_N, "Z_L": Z_L,
                "n_train": int(len(train)), "n_test": int(len(test)), "T": int(T)})
