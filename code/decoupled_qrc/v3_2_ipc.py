"""
v3_2_ipc.py -- the V3.2 IPC layer.

This is deliberately THIN. The repository's existing estimator is already
correct on the points that matter and is reused verbatim:

  * `ipc.legendre_target`  -- orthonormal L_d = sqrt(2d+1) P_d under the
    Uniform[-1,1] probability measure (the normalisation a Gauss-quadrature
    test in `tests/test_ipc.py` already pins).
  * `ipc.generate_profiles` -- delay/degree enumeration, de-duplicated by
    canonical (delay, degree) tuples, with SEEDED (data-independent)
    subsampling when a degree overflows its cap.
  * `ipc.build_target`, `ipc.to_v`, `ipc.ProfileCapacity`.
  * `ipc_decomposition` -- the classification predicate
    `degree >= 2 AND delays == (0,)` for instantaneous, so a cross-delay
    target with a tau=0 factor is NEVER counted as instantaneous.
  * `ceiling_audit` -- target-count / rank / sample ceilings.
  * `qrc_qiskit.chrono_split` -- guard-gapped chronological split.

WHAT V3.2 ADDS

1. SPEED. `compute_ipc_detailed` refits `sklearn.Ridge` from scratch for every
   (target, surrogate, alpha). At the confirmation-stage settings (199 nulls,
   ~50 targets, 7 alphas) that is ~70k solves per evaluation and would
   dominate the entire study. `RidgeBank` factorises the standardised train
   design ONCE by SVD and reuses it for every target, surrogate and alpha.
   `tests/test_v3_2_ipc.py::test_fast_scorer_matches_repo_estimator` asserts
   the fast path reproduces `ipc._capacity_score` to 1e-10, so this is an
   optimisation, not a redefinition.

2. tau-RESOLVED M_long. `ipc_decomposition.M_long` sums degree-1 capacity over
   ALL delays. The V3.2 primary memory metric is the LONG-delay part only,
   sum_{tau >= tau_min} C_{1,tau}, with tau_min fixed before confirmation and
   justified from the calibration-stage delay profile.

3. SAMPLE-SIZE ENFORCEMENT. n_train >= max(10 * r_eff, 500) is checked and
   reported, not assumed. V3.1 ran this analysis on n_train = 43 with a null
   fraction of ~0.50; that combination is now a hard validity failure.

4. FOUR NUMBERS, ALWAYS: raw, null, bias-corrected (max(raw-null, 0)) and the
   UNCHANGED legacy `capacity` (raw if significant else 0). The legacy metric
   is never silently redefined, so V3.1 comparisons stay meaningful.

5. ADAPTIVE DELAYS, and an explicit record of the structural degree bound: the
   V3.2 encoder makes every instantaneous feature a polynomial of degree at
   most N_P, so instantaneous capacity above degree N_P is STRUCTURALLY zero.
   Reporting it as an empirical null would be wrong, so it is labelled.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                      # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .ipc import (DEFAULT_ALPHAS, Profile, ProfileCapacity, build_target,
                  generate_profiles, to_v)
from .ceiling_audit import audit_ceiling

# --------------------------------------------------------------------------
# Fast, equivalence-tested ridge scoring
# --------------------------------------------------------------------------
class RidgeBank:
    """One SVD of the standardised training design, reused for every target.

    Reproduces `ipc._capacity_score` exactly: features standardised on train
    statistics only (population std, matching `StandardScaler`), ridge with an
    UNPENALISED intercept, alpha selected on validation MSE, squared Pearson
    correlation reported on the held-out test block and clipped to [0, 1].
    """

    def __init__(self, X: np.ndarray, train, val, test, alphas=DEFAULT_ALPHAS):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, n_features), got shape {X.shape}")
        self.train, self.val, self.test = np.asarray(train), np.asarray(val), np.asarray(test)
        self.alphas = tuple(float(a) for a in alphas)
        mu = X[self.train].mean(axis=0)
        sd = X[self.train].std(axis=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        self._Xtr = (X[self.train] - mu) / sd
        self._Xva = (X[self.val] - mu) / sd
        self._Xte = (X[self.test] - mu) / sd
        self._U, self._s, self._Vt = np.linalg.svd(self._Xtr, full_matrices=False)

    @property
    def numerical_rank(self) -> int:
        return int(np.sum(self._s > 1e-10 * max(self._s[0], 1e-300)))

    @property
    def effective_rank(self) -> float:
        """Participation ratio (sum s^2)^2 / sum s^4 -- a continuous stand-in
        for the number of directions the readout can actually use."""
        s2 = self._s ** 2
        denom = float(np.sum(s2 ** 2))
        return float(np.sum(s2) ** 2 / denom) if denom > 0 else 0.0

    def _weights(self, uty: np.ndarray, alpha: float) -> np.ndarray:
        return self._Vt.T @ ((self._s / (self._s ** 2 + alpha)) * uty)

    def score(self, y: np.ndarray, alpha: float = None) -> tuple:
        """Return (capacity, alpha_used). With `alpha` given, no validation
        search is performed -- that is the frozen-readout path."""
        y = np.asarray(y, dtype=float)
        ytr = y[self.train]
        ybar = float(ytr.mean())
        uty = self._U.T @ (ytr - ybar)

        if alpha is None:
            best_alpha, best_err, best_w = self.alphas[0], np.inf, None
            for a in self.alphas:
                w = self._weights(uty, a)
                err = float(np.mean((self._Xva @ w + ybar - y[self.val]) ** 2))
                if err < best_err:
                    best_err, best_alpha, best_w = err, a, w
            w, alpha_used = best_w, best_alpha
        else:
            w, alpha_used = self._weights(uty, float(alpha)), float(alpha)

        pred = self._Xte @ w + ybar
        y_test = y[self.test]
        cov = float(np.cov(pred, y_test)[0, 1]) ** 2
        denom = float(np.var(y_test) * np.var(pred)) + 1e-12
        return float(np.clip(cov / denom, 0.0, 1.0)), alpha_used


def profile_capacities(u: np.ndarray, X: np.ndarray, train, val, test, *,
                        max_delay: int = 8, max_degree: int = 5,
                        max_targets_per_degree: int = 30, n_surrogates: int = 99,
                        significance_z: float = 2.0, alphas=DEFAULT_ALPHAS,
                        seed: int = 0, null_seed: int = None,
                        always_include_single_delays: bool = True) -> tuple:
    """Per-target capacities with raw / null / bias-corrected / legacy values.

    Surrogates permute the TRAIN targets only, leaving X untouched -- the same
    null the repository already uses, kept so the numbers stay comparable.
    """
    v = to_v(np.asarray(u, dtype=float))
    profiles = generate_profiles(max_delay, max_degree, max_targets_per_degree, seed=seed,
                                 always_include_single_delays=always_include_single_delays)
    was_capped = profiles.pop("_was_capped")
    rng = np.random.RandomState(int(seed + 9999 if null_seed is None else null_seed))
    bank = RidgeBank(X, train, val, test, alphas)

    records = []
    for d in range(1, max_degree + 1):
        for prof in profiles[d]:
            y = build_target(prof, v)
            real, _ = bank.score(y)
            nulls = np.empty(n_surrogates)
            for s in range(n_surrogates):
                y_shuf = y.copy()
                y_shuf[bank.train] = y[bank.train][rng.permutation(len(bank.train))]
                nulls[s], _ = bank.score(y_shuf)
            thr = float(nulls.mean() + significance_z * nulls.std())
            delays = tuple(sorted(tau for tau, _ in prof.pairs))
            records.append(ProfileCapacity(
                degree=d, delays=delays, max_delay=max(delays), min_delay=min(delays),
                span=max(delays) - min(delays), n_delays=len(delays),
                raw_capacity=real, null_mean=float(nulls.mean()), null_std=float(nulls.std()),
                threshold=thr, significant=bool(real > thr),
                capacity=real if real > thr else 0.0))
    n_tested = {d: len(profiles[d]) for d in range(1, max_degree + 1)}
    return records, was_capped, n_tested, bank


def _sum(records, pred, attr) -> float:
    return float(sum(getattr(r, attr) for r in records if pred(r)))


def _bc(records, pred) -> float:
    return float(sum(max(r.raw_capacity - r.null_mean, 0.0) for r in records if pred(r)))


@dataclass
class IPCReport:
    """Everything the gates and the report need from one feature matrix."""

    M_long: float                  # sum_{tau >= tau_min} C_{1,tau}, bias-corrected
    M_long_raw: float
    M_long_null: float
    M_long_legacy: float
    NL_0: float                    # instantaneous, degree >= 2, delays == (0,), bias-corrected
    NL_0_raw: float
    NL_0_null: float
    NL_0_legacy: float
    NL_temporal: float
    NL_legacy_all: float
    delay_profile: dict            # {tau: bias-corrected C_{1,tau}}
    degree_profile: dict           # {degree: bias-corrected instantaneous C_d}
    null_fraction_NL0: float
    null_fraction_M: float
    numerical_rank: int
    effective_rank: float
    n_train: int
    n_val: int
    n_test: int
    sample_size_ok: bool
    sample_size_required: int
    ceiling_NL0: dict
    ceiling_M: dict
    tau_min: int
    max_structural_degree: int
    degrees_above_structural_bound: list
    was_capped: dict
    n_targets: dict
    records: list = field(repr=False, default_factory=list)
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "records"}
        return out


def compute_ipc_report(u: np.ndarray, X: np.ndarray, train, val, test, *,
                        tau_min: int = 2, max_delay: int = 8, max_degree: int = 5,
                        max_targets_per_degree: int = 30, n_surrogates: int = 99,
                        significance_z: float = 2.0, alphas=DEFAULT_ALPHAS,
                        seed: int = 0, null_seed: int = None,
                        max_structural_degree: int = None,
                        min_train_absolute: int = 500,
                        train_per_rank: int = 10) -> IPCReport:
    """The single IPC entry point used by calibration, discovery and confirmation."""
    records, was_capped, n_tested, bank = profile_capacities(
        u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
        max_targets_per_degree=max_targets_per_degree, n_surrogates=n_surrogates,
        significance_z=significance_z, alphas=alphas, seed=seed, null_seed=null_seed)

    is_lin = lambda r: r.degree == 1                                    # noqa: E731
    is_long = lambda r: r.degree == 1 and r.delays == (r.max_delay,) and r.max_delay >= tau_min  # noqa: E731
    is_inst = lambda r: r.degree >= 2 and r.delays == (0,)              # noqa: E731
    is_temp = lambda r: r.degree >= 2 and r.max_delay > 0               # noqa: E731
    is_leg = lambda r: r.degree >= 2                                    # noqa: E731

    delay_profile = {}
    for r in records:
        if r.degree == 1 and len(r.delays) == 1:
            delay_profile[int(r.max_delay)] = max(r.raw_capacity - r.null_mean, 0.0)
    degree_profile = {}
    for r in records:
        if is_inst(r):
            degree_profile[int(r.degree)] = degree_profile.get(int(r.degree), 0.0) \
                + max(r.raw_capacity - r.null_mean, 0.0)

    nl0_raw = _sum(records, is_inst, "raw_capacity")
    nl0_null = _sum(records, is_inst, "null_mean")
    m_raw = _sum(records, is_long, "raw_capacity")
    m_null = _sum(records, is_long, "null_mean")
    n_inst = sum(1 for r in records if is_inst(r))
    n_long = sum(1 for r in records if is_long(r))

    ceil_nl0 = audit_ceiling("NL_0", nl0_raw, max(n_inst, 1), bank.numerical_rank,
                             bank.effective_rank, len(bank.train))
    ceil_m = audit_ceiling("M_long", m_raw, max(n_long, 1), bank.numerical_rank,
                           bank.effective_rank, len(bank.train))

    required = int(max(train_per_rank * bank.effective_rank, min_train_absolute))
    if max_structural_degree is None:
        max_structural_degree = max_degree
    above = [d for d in sorted(degree_profile) if d > max_structural_degree]

    return IPCReport(
        M_long=_bc(records, is_long), M_long_raw=m_raw, M_long_null=m_null,
        M_long_legacy=_sum(records, is_lin, "capacity"),
        NL_0=_bc(records, is_inst), NL_0_raw=nl0_raw, NL_0_null=nl0_null,
        NL_0_legacy=_sum(records, is_inst, "capacity"),
        NL_temporal=_bc(records, is_temp), NL_legacy_all=_sum(records, is_leg, "capacity"),
        delay_profile=delay_profile, degree_profile=degree_profile,
        null_fraction_NL0=float(nl0_null / nl0_raw) if nl0_raw > 1e-12 else float("nan"),
        null_fraction_M=float(m_null / m_raw) if m_raw > 1e-12 else float("nan"),
        numerical_rank=bank.numerical_rank, effective_rank=bank.effective_rank,
        n_train=len(bank.train), n_val=len(bank.val), n_test=len(bank.test),
        sample_size_ok=bool(len(bank.train) >= required), sample_size_required=required,
        ceiling_NL0=ceil_nl0.as_dict() if hasattr(ceil_nl0, "as_dict") else vars(ceil_nl0),
        ceiling_M=ceil_m.as_dict() if hasattr(ceil_m, "as_dict") else vars(ceil_m),
        tau_min=int(tau_min), max_structural_degree=int(max_structural_degree),
        degrees_above_structural_bound=above, was_capped=was_capped, n_targets=n_tested,
        records=records,
        config=dict(max_delay=max_delay, max_degree=max_degree,
                    max_targets_per_degree=max_targets_per_degree,
                    n_surrogates=n_surrogates, significance_z=significance_z,
                    tau_min=tau_min, seed=seed, null_seed=null_seed,
                    alphas=list(alphas)))


def adaptive_max_delay(u: np.ndarray, X: np.ndarray, train, val, test, *,
                        start: int = 6, cap: int = 24, step: int = 4,
                        n_surrogates: int = 15, n_consecutive: int = 3,
                        alphas=DEFAULT_ALPHAS, seed: int = 0) -> dict:
    """Grow the delay window until the linear-memory tail dies.

    Stops when `n_consecutive` successive delays have bias-corrected
    C_{1,tau} below the null band (null_mean + 2*null_std), or when `cap` is
    reached -- in which case `reason='resource_cap'` is recorded so a
    truncated tail is never mistaken for a decayed one.
    """
    v = to_v(np.asarray(u, dtype=float))
    bank = RidgeBank(X, train, val, test, alphas)
    rng = np.random.RandomState(seed + 4242)
    profile, chosen, reason = {}, None, "resource_cap"

    tau = 0
    while tau <= cap:
        y = build_target(Profile(pairs=((tau, 1),)), v)
        real, _ = bank.score(y)
        nulls = np.empty(n_surrogates)
        for s in range(n_surrogates):
            ys = y.copy()
            ys[bank.train] = y[bank.train][rng.permutation(len(bank.train))]
            nulls[s], _ = bank.score(ys)
        band = float(nulls.mean() + 2.0 * nulls.std())
        profile[tau] = {"raw": real, "null_mean": float(nulls.mean()),
                        "band": band, "above_band": bool(real > band)}
        taus = sorted(profile)
        if len(taus) > n_consecutive and tau >= start:
            recent = [profile[t]["above_band"] for t in taus[-n_consecutive:]]
            if not any(recent):
                chosen, reason = tau, "tail_decayed"
                break
        tau += 1

    if chosen is None:
        chosen = min(cap, max(profile) if profile else start)
    return {"max_delay": int(chosen), "reason": reason, "profile": profile,
            "n_consecutive_required": n_consecutive, "cap": cap}
