"""
frozen_protocol.py -- V2.2 Phase 5: a finite-difference derivative is only
valid if the READOUT DEFINITION (target list, ridge hyperparameter, null-
permutation draws, delay/degree conventions) is IDENTICAL at every
plus/minus stencil point -- only the underlying reservoir FEATURES may
differ (that is the whole point of the derivative). V2.1 re-derived the
target list, re-searched ridge alpha, and re-drew null permutations
independently at every stencil point, so a derivative could in principle
reflect a change in what was being measured, not just a change in the
physics.

`freeze_protocol` runs once at the CENTER point: generates the target
list, ONE-TIME-searches ridge alpha per target on that point's own
train/val split, and returns a `FrozenSpec` capturing every one of those
choices. `evaluate_frozen` then reuses the frozen target list, the frozen
per-target alpha (no search), and the SAME permutation-RNG seed (so the
sequence of surrogate draws is bit-identical) at every subsequent
(plus/minus) point -- only the feature matrix `X` (and hence the fitted
ridge COEFFICIENTS, not the model class or its hyperparameter) is allowed
to vary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .ipc import (Profile, generate_profiles, build_target, to_v, DEFAULT_ALPHAS,
                   _capacity_score_fixed_alpha, ProfileCapacity)


def _search_alpha(X, y, train, val, alphas):
    scaler = StandardScaler().fit(X[train])
    Xtr, Xval = scaler.transform(X[train]), scaler.transform(X[val])
    best_alpha, best_err = alphas[0], np.inf
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr, y[train])
        err = np.mean((model.predict(Xval) - y[val]) ** 2)
        if err < best_err:
            best_err, best_alpha = err, a
    return best_alpha


@dataclass
class FrozenSpec:
    profiles: dict              # {degree: [Profile, ...]} -- the EXACT target list, generated once
    alpha_by_key: dict          # {(degree, delays_tuple): alpha} -- frozen ridge hyperparameter
    n_surrogates: int
    seed: int
    max_delay: int
    max_degree: int
    significance_z: float = 2.0
    center_records: list = field(default_factory=list, repr=False)


def freeze_protocol(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                     max_delay: int = 8, max_degree: int = 6, max_targets_per_degree: int = 25,
                     n_surrogates: int = 19, seed: int = 0, alphas=DEFAULT_ALPHAS,
                     significance_z: float = 2.0,
                     always_include_single_delays: bool = False) -> FrozenSpec:
    """Run ONCE, at the candidate center. Generates the target list,
    searches ridge alpha per target using THIS point's own train/val
    split, and returns a `FrozenSpec` for `evaluate_frozen` to reuse
    unchanged at every other stencil point."""
    v = to_v(u)
    profiles_dict = generate_profiles(max_delay, max_degree, max_targets_per_degree, seed=seed,
                                       always_include_single_delays=always_include_single_delays)
    profiles_dict.pop("_was_capped")
    rng = np.random.RandomState(seed + 9999)

    alpha_by_key, records = {}, []
    for d in range(1, max_degree + 1):
        for prof in profiles_dict[d]:
            y = build_target(prof, v)
            alpha = _search_alpha(X, y, train, val, alphas)
            key = (d, tuple(sorted(tau for tau, _ in prof.pairs)))
            alpha_by_key[key] = alpha

            real_score = _capacity_score_fixed_alpha(X, y, train, val, test, alpha)
            null_scores = np.empty(n_surrogates)
            for s in range(n_surrogates):
                perm = rng.permutation(len(train))
                y_shuf = y.copy()
                y_shuf[train] = y[train][perm]
                null_scores[s] = _capacity_score_fixed_alpha(X, y_shuf, train, val, test, alpha)
            threshold = null_scores.mean() + significance_z * null_scores.std()
            significant = real_score > threshold
            delays = tuple(sorted(tau for tau, _ in prof.pairs))
            records.append(ProfileCapacity(
                degree=d, delays=delays, max_delay=max(delays), min_delay=min(delays),
                span=max(delays) - min(delays), n_delays=len(delays), raw_capacity=real_score,
                null_mean=float(null_scores.mean()), null_std=float(null_scores.std()),
                threshold=float(threshold), significant=significant,
                capacity=real_score if significant else 0.0,
            ))

    return FrozenSpec(profiles=profiles_dict, alpha_by_key=alpha_by_key, n_surrogates=n_surrogates,
                       seed=seed, max_delay=max_delay, max_degree=max_degree,
                       significance_z=significance_z, center_records=records)


def evaluate_frozen(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                     spec: FrozenSpec) -> list:
    """Reuses `spec.profiles` (identical target list, not regenerated),
    `spec.alpha_by_key` (no alpha search), and re-seeds the surrogate RNG
    from `spec.seed` so the SAME sequence of permutation draws is used --
    only the feature matrix `X` may legitimately differ from the center
    point. Returns a list of `ProfileCapacity` records, structurally
    identical to `ipc.compute_ipc_detailed`'s own output, so every
    `ipc_decomposition` aggregation helper (`capacity_at_delay`,
    `nl_tensor_by_fixed_delay`, etc.) works unchanged on frozen-protocol
    output too."""
    v = to_v(u)
    rng = np.random.RandomState(spec.seed + 9999)
    records = []
    for d in range(1, spec.max_degree + 1):
        for prof in spec.profiles[d]:
            y = build_target(prof, v)
            key = (d, tuple(sorted(tau for tau, _ in prof.pairs)))
            alpha = spec.alpha_by_key[key]

            real_score = _capacity_score_fixed_alpha(X, y, train, val, test, alpha)
            null_scores = np.empty(spec.n_surrogates)
            for s in range(spec.n_surrogates):
                perm = rng.permutation(len(train))
                y_shuf = y.copy()
                y_shuf[train] = y[train][perm]
                null_scores[s] = _capacity_score_fixed_alpha(X, y_shuf, train, val, test, alpha)
            threshold = null_scores.mean() + spec.significance_z * null_scores.std()
            significant = real_score > threshold
            delays = tuple(sorted(tau for tau, _ in prof.pairs))
            records.append(ProfileCapacity(
                degree=d, delays=delays, max_delay=max(delays), min_delay=min(delays),
                span=max(delays) - min(delays), n_delays=len(delays), raw_capacity=real_score,
                null_mean=float(null_scores.mean()), null_std=float(null_scores.std()),
                threshold=float(threshold), significant=significant,
                capacity=real_score if significant else 0.0,
            ))
    return records
