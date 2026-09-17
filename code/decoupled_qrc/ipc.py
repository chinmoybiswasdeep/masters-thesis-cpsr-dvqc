"""
ipc.py -- Information Processing Capacity (Dambre et al. 2012 / Cindrak et
al. 2026 construction), built on orthonormal Legendre polynomial targets for
Uniform[-1,1] inputs.

    IPC_1        = linear memory      (degree-1 targets only)
    IPC_2..IPC_6 = nonlinear terms    (degree>=2 targets)
    Memory       = IPC_1
    Nonlinearity = sum_{d=2}^{6} IPC_d
    TotalIPC     = sum_{d=1}^{6} IPC_d

Target functions: for a *profile* -- a set of (delay, degree) pairs with
distinct delays and degree>=1 each -- the target is
    y_t = prod_{(tau, k) in profile} L_k(v_{t-tau}),   v = 2u - 1 in [-1,1]
with total degree = sum of the per-delay degrees. `u` is the reservoir's own
[0,1] input convention (Ry(pi*u)); `v` is only used inside this module to
define targets on the textbook Uniform[-1,1] domain.

Capacity per target: squared correlation between the target and its
ridge-regression reconstruction from reservoir features (same construction as
`qrc_qiskit.memory_capacity`'s per-lag score, generalized to arbitrary
polynomial-product targets), clipped to [0,1], alpha selected on a validation
block, reported on a held-out test block never touched during selection.

Finite-data significance filtering (Part 7): a target's capacity only counts
if it clears a null threshold estimated by re-fitting the SAME target against
`n_surrogates` independent random-input-order shuffles of the SAME reservoir
features (breaking any real feature-target relationship while preserving each
side's marginal statistics) -- a target whose real-data score does not exceed
`mean + z * std` of its own shuffled-null distribration is treated as noise
(capacity 0 for that target), not a real information-processing contribution.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import eval_legendre
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Orthonormal Legendre targets
# =============================================================================

def legendre_target(v: np.ndarray, degree: int) -> np.ndarray:
    """Orthonormal Legendre polynomial of `degree`, evaluated at v in
    [-1,1]: L_d(v) = sqrt(2d+1) * P_d(v), so that
    (1/2) * integral_{-1}^{1} L_d(v) L_e(v) dv = delta_{de}
    under the UNIFORM[-1,1] PROBABILITY measure (density 1/2 dv) -- the
    convention IPC theory requires (targets orthonormal w.r.t. the actual
    input distribution, not w.r.t. plain Lebesgue measure on [-1,1], which
    would instead need sqrt((2d+1)/2) and is the WRONG constant for this
    purpose). `degree=0` is the constant 1 -- callers building nonconstant
    targets never use it directly (see `generate_profiles`, which only ever
    assigns degree>=1)."""
    if degree < 0:
        raise ValueError(degree)
    return np.sqrt(2 * degree + 1) * eval_legendre(degree, v)


def to_v(u: np.ndarray) -> np.ndarray:
    """Map the reservoir's own [0,1] input convention to Legendre's native
    [-1,1] domain: v = 2u - 1."""
    return 2.0 * np.asarray(u, dtype=float) - 1.0


# =============================================================================
# Degree-profile enumeration (Part 7: degree 1..6, no duplicate targets)
# =============================================================================

def _integer_partitions(total: int, max_part: int):
    """All partitions of `total` into parts <= max_part, as sorted-descending
    tuples. Standard recursive partition generator."""
    def helper(remaining, cap):
        if remaining == 0:
            yield ()
            return
        for part in range(min(cap, remaining), 0, -1):
            for rest in helper(remaining - part, part):
                yield (part,) + rest
    yield from helper(total, min(max_part, total))


@dataclass(frozen=True)
class Profile:
    """One IPC target: an immutable, canonically-ordered tuple of
    (delay, degree) pairs sorted by delay -- the canonical form used for
    de-duplication (two profiles with the same set of pairs are the same
    target regardless of construction order)."""
    pairs: tuple  # tuple of (delay:int, degree:int), sorted by delay

    @property
    def total_degree(self) -> int:
        return sum(d for _, d in self.pairs)

    def __str__(self):
        return "*".join(f"L{d}(t-{tau})" for tau, d in self.pairs)


def generate_profiles(max_delay: int, max_degree: int, max_targets_per_degree: int,
                       seed: int = 0) -> dict:
    """All degree-1..max_degree profiles using delays in [0, max_delay],
    each delay used at most once per profile (Part 7's degree-2/3 examples:
    L2(u_{t-k}); L1(u_{t-k1})L1(u_{t-k2}); L3(...); L2(...)L1(...);
    L1L1L1(...); etc.). De-duplicated by construction (a profile is a
    canonical set of (delay,degree) pairs). If the exhaustive enumeration for
    a degree exceeds `max_targets_per_degree`, a SEEDED random subsample of
    that size is kept (reported via the returned `was_capped` flags) rather
    than silently truncating in enumeration order.

    Returns {degree: list[Profile]}.
    """
    delays = list(range(max_delay + 1))
    rng = np.random.RandomState(seed)
    out = {}
    was_capped = {}
    for d in range(1, max_degree + 1):
        profiles = set()
        for partition in _integer_partitions(d, d):
            k = len(partition)
            if k > len(delays):
                continue
            # group equal-valued parts so equal-degree delays are chosen as
            # an unordered combination (swapping which delay gets which
            # equal-degree slot is the same target).
            value_counts = []
            for val, grp in itertools.groupby(partition):
                value_counts.append((val, len(list(grp))))

            def assign(groups, available):
                if not groups:
                    yield ()
                    return
                val, cnt = groups[0]
                for combo in itertools.combinations(available, cnt):
                    remaining = [a for a in available if a not in combo]
                    for rest in assign(groups[1:], remaining):
                        yield tuple((delay, val) for delay in combo) + rest

            for pairs in assign(value_counts, delays):
                profiles.add(Profile(pairs=tuple(sorted(pairs))))

        profiles = list(profiles)
        capped = len(profiles) > max_targets_per_degree
        if capped:
            idx = rng.choice(len(profiles), size=max_targets_per_degree, replace=False)
            profiles = [profiles[i] for i in idx]
        out[d] = profiles
        was_capped[d] = capped
    out["_was_capped"] = was_capped
    return out


def build_target(profile: Profile, v: np.ndarray) -> np.ndarray:
    """y_t = prod_{(tau,k) in profile} L_k(v_{t-tau}); entries with t<tau for
    any (tau,k) in the profile are set to 0 (caller must washout by
    max(delay) before fitting -- see `compute_ipc`)."""
    T = len(v)
    y = np.ones(T)
    for tau, k in profile.pairs:
        shifted = np.zeros(T)
        if tau == 0:
            shifted[:] = v
        else:
            shifted[tau:] = v[:T - tau]
        y *= legendre_target(shifted, k)
    max_tau = max(tau for tau, _ in profile.pairs)
    y[:max_tau] = 0.0
    return y


# =============================================================================
# Per-target capacity (ridge reconstruction, val-selected alpha, held-out test)
# =============================================================================

DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _capacity_score(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
                     test: np.ndarray, alphas=DEFAULT_ALPHAS) -> float:
    scaler = StandardScaler().fit(X[train])
    Xtr, Xval, Xte = scaler.transform(X[train]), scaler.transform(X[val]), scaler.transform(X[test])
    best_alpha, best_val_err = alphas[0], np.inf
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr, y[train])
        pred = model.predict(Xval)
        err = np.mean((pred - y[val]) ** 2)
        if err < best_val_err:
            best_val_err, best_alpha = err, a
    model = Ridge(alpha=best_alpha).fit(Xtr, y[train])
    pred = model.predict(Xte)
    y_test = y[test]
    cov = np.cov(pred, y_test)[0, 1] ** 2
    denom = np.var(y_test) * np.var(pred) + 1e-12
    return float(np.clip(cov / denom, 0.0, 1.0))


@dataclass
class IPCResult:
    ipc_by_degree: dict           # {degree: float}  (sum of significant target capacities)
    memory: float                 # IPC_1
    nonlinearity: float           # sum(IPC_2..max_degree)
    total: float                  # sum(IPC_1..max_degree)
    n_targets_tested: dict        # {degree: int}
    n_targets_significant: dict   # {degree: int}
    was_capped: dict              # {degree: bool} -- profile enumeration was subsampled
    config: dict = field(default_factory=dict)


@dataclass
class ProfileCapacity:
    """One per-target record -- the raw material for the order-delay
    decomposition `ipc_decomposition.py` builds on top of. Nothing here
    changes what `compute_ipc` reports; it is the SAME per-profile
    real_score/threshold computation `compute_ipc` already did internally,
    just no longer thrown away after being summed into `ipc_by_degree`."""
    degree: int
    delays: tuple          # sorted tuple of delays used by this profile, e.g. (0,) or (2,5)
    max_delay: int          # max(delays) -- 0 means a purely instantaneous (no-memory) target
    min_delay: int
    span: int               # max_delay - min_delay
    n_delays: int            # len(delays) -- number of distinct delayed inputs used
    raw_capacity: float      # real_score, BEFORE significance filtering
    null_mean: float
    null_std: float
    threshold: float
    significant: bool
    capacity: float          # raw_capacity if significant else 0.0 -- what compute_ipc sums


def compute_ipc_detailed(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                          max_delay: int = 8, max_degree: int = 6, max_targets_per_degree: int = 25,
                          n_surrogates: int = 8, significance_z: float = 2.0, alphas=DEFAULT_ALPHAS,
                          seed: int = 0):
    """Same computation `compute_ipc` performs, but returns the full list of
    per-profile `ProfileCapacity` records instead of only the degree-summed
    total -- the raw material for order-delay (C_{d,tau}) decomposition.
    `compute_ipc` is now a thin aggregation wrapper around this function
    (Part 3 of the V2 validation spec); its own output is UNCHANGED bit-for-
    bit (verified by `tests/test_ipc_decomposition.py`), so nothing about
    the legacy M/NL definitions is silently altered."""
    v = to_v(u)
    profiles = generate_profiles(max_delay, max_degree, max_targets_per_degree, seed=seed)
    was_capped = profiles.pop("_was_capped")
    rng = np.random.RandomState(seed + 9999)

    records = []
    for d in range(1, max_degree + 1):
        for prof in profiles[d]:
            y = build_target(prof, v)
            real_score = _capacity_score(X, y, train, val, test, alphas)

            null_scores = np.empty(n_surrogates)
            for s in range(n_surrogates):
                perm = rng.permutation(len(train))
                y_shuf = y.copy()
                y_shuf[train] = y[train][perm]
                null_scores[s] = _capacity_score(X, y_shuf, train, val, test, alphas)
            threshold = null_scores.mean() + significance_z * null_scores.std()
            significant = real_score > threshold

            delays = tuple(sorted(tau for tau, _ in prof.pairs))
            records.append(ProfileCapacity(
                degree=d, delays=delays, max_delay=max(delays), min_delay=min(delays),
                span=max(delays) - min(delays), n_delays=len(delays),
                raw_capacity=real_score, null_mean=float(null_scores.mean()),
                null_std=float(null_scores.std()), threshold=float(threshold),
                significant=significant, capacity=real_score if significant else 0.0,
            ))

    n_tested = {d: len(profiles[d]) for d in range(1, max_degree + 1)}
    return records, was_capped, n_tested


def compute_ipc(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                 max_delay: int = 8, max_degree: int = 6, max_targets_per_degree: int = 25,
                 n_surrogates: int = 8, significance_z: float = 2.0, alphas=DEFAULT_ALPHAS,
                 seed: int = 0) -> IPCResult:
    """Compute IPC_1..IPC_max_degree for reservoir features `X` against input
    sequence `u` (reservoir's own [0,1] convention). `train`/`val`/`test`
    should come from `qrc_qiskit.chrono_split` with a guard gap >= max_delay,
    exactly as for every other task in this repo (no leakage across splits).
    UNCHANGED output vs. every prior pass in this project -- now implemented
    by aggregating `compute_ipc_detailed`'s per-profile records rather than
    summing inline, so the two never drift apart."""
    records, was_capped, n_tested = compute_ipc_detailed(
        u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
        max_targets_per_degree=max_targets_per_degree, n_surrogates=n_surrogates,
        significance_z=significance_z, alphas=alphas, seed=seed)

    ipc_by_degree, n_sig = {}, {}
    for d in range(1, max_degree + 1):
        d_records = [r for r in records if r.degree == d]
        ipc_by_degree[d] = sum(r.capacity for r in d_records)
        n_sig[d] = sum(1 for r in d_records if r.significant)

    memory = ipc_by_degree.get(1, 0.0)
    nonlinearity = sum(ipc_by_degree.get(d, 0.0) for d in range(2, max_degree + 1))
    total = memory + nonlinearity

    return IPCResult(
        ipc_by_degree=ipc_by_degree, memory=memory, nonlinearity=nonlinearity, total=total,
        n_targets_tested=n_tested, n_targets_significant=n_sig, was_capped=was_capped,
        config=dict(max_delay=max_delay, max_degree=max_degree,
                    max_targets_per_degree=max_targets_per_degree, n_surrogates=n_surrogates,
                    significance_z=significance_z, n_train=len(train), n_val=len(val), n_test=len(test)),
    )
