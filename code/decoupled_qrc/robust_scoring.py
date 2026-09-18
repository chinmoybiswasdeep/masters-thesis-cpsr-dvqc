"""
robust_scoring.py -- V3 fixes for two V2.2 scoring defects, both verified
in the saved V2.2 results:

  * `phase8_ranking` shows the single VALID candidate received
    `score = 0.000` with every component 0.0, because V2.2's min-max
    normalisation divides by (max - min) and returns all-zeros when only
    one candidate survives. A ranking that scores a genuinely valid
    candidate as exactly zero is meaningless.

  * `phase15_advancement_decision.checks['7_not_driven_by_one_seed']`
    returned True even though only ONE of the two response seeds actually
    showed nonzero NL -- the check only counted how many seeds were RUN,
    not how many showed the effect.

V3 replaces the min-max score with a LOWER-CONFIDENCE-BOUND score that is
well defined for a single candidate, and replaces the seed check with one
that counts seeds where the effect is actually PRESENT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LCBScore:
    mean: float
    std: float
    n: int
    lcb: float                  # mean - z * std / sqrt(n)
    z: float
    components: dict = field(default_factory=dict)


def lower_confidence_bound(values, z: float = 1.0) -> LCBScore:
    """Robustness-aware score: the lower confidence bound of the per-seed
    values, so a candidate that is excellent on one seed and zero on
    another scores WORSE than one that is consistently moderate. Well
    defined for n = 1 (where std = 0 and lcb = the single value), unlike a
    min-max normalisation across candidates."""
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return LCBScore(mean=float("nan"), std=float("nan"), n=0, lcb=float("nan"), z=z)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    lcb = mean - z * std / np.sqrt(vals.size)
    return LCBScore(mean=mean, std=std, n=int(vals.size), lcb=float(lcb), z=z)


def robust_candidate_score(per_seed_nl0, per_seed_memory, per_seed_backaction,
                            z: float = 1.0, backaction_weight: float = 0.25) -> LCBScore:
    """Ranks a candidate by the lower confidence bound of its
    instantaneous NL across seeds, mildly penalised by mean back-action
    and mildly rewarded for memory strength. NOT a min-max normalisation,
    so it never collapses to zero when only one candidate is in play."""
    nl = lower_confidence_bound(per_seed_nl0, z=z)
    mem = lower_confidence_bound(per_seed_memory, z=z)
    ba = lower_confidence_bound(per_seed_backaction, z=z)
    penalty = backaction_weight * (ba.mean if np.isfinite(ba.mean) else 0.0)
    score = nl.lcb - penalty
    return LCBScore(mean=nl.mean, std=nl.std, n=nl.n, lcb=float(score), z=z,
                     components={"nl0_lcb": nl.lcb, "nl0_mean": nl.mean, "nl0_std": nl.std,
                                  "memory_mean": mem.mean, "backaction_mean": ba.mean,
                                  "backaction_penalty": penalty})


@dataclass
class SeedSupport:
    n_seeds_run: int
    n_seeds_with_effect: int
    fraction_with_effect: float
    threshold: float
    driven_by_one_seed: bool
    sufficient: bool


def seed_support(per_seed_values, threshold: float = 0.05, min_fraction: float = 0.6) -> SeedSupport:
    """Counts how many seeds actually SHOW the effect (value > threshold),
    not merely how many were run. `driven_by_one_seed` is True when at
    most one seed carries the effect while more than one was run -- the
    exact V2.2 situation (1 of 2 response seeds showed NL, yet the old
    check passed)."""
    vals = np.asarray(list(per_seed_values), dtype=float)
    vals = vals[np.isfinite(vals)]
    n_run = int(vals.size)
    n_effect = int(np.sum(vals > threshold))
    frac = float(n_effect / n_run) if n_run else 0.0
    driven_by_one = bool(n_run > 1 and n_effect <= 1)
    return SeedSupport(n_seeds_run=n_run, n_seeds_with_effect=n_effect, fraction_with_effect=frac,
                        threshold=threshold, driven_by_one_seed=driven_by_one,
                        sufficient=bool(n_run >= 2 and frac >= min_fraction and not driven_by_one))
