"""Multiplicity-corrected paired inference used by every V8 gate."""

from __future__ import annotations

import numpy as np


def bonferroni_tail_alpha(alpha: float, comparisons: int) -> float:
    if not 0.0 < alpha < 1.0 or comparisons < 1:
        raise ValueError("invalid alpha or comparison count")
    return alpha / (2.0 * comparisons)


def paired_interval(values, *, alpha: float, comparisons: int, resamples: int, seed: int = 8080) -> dict:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError("paired samples must contain at least two finite values")
    rng = np.random.default_rng(seed)
    means = sample[rng.integers(0, sample.size, (resamples, sample.size))].mean(axis=1)
    tail = bonferroni_tail_alpha(alpha, comparisons)
    return {
        "estimate": float(sample.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "tail_alpha": tail,
        "n": int(sample.size),
    }


def simultaneous_intervals(matrix, *, alpha: float, comparisons: int, resamples: int, seed: int = 8080) -> list[dict]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("expected seeds by components")
    return [
        paired_interval(values[:, index], alpha=alpha, comparisons=comparisons, resamples=resamples, seed=seed + index)
        for index in range(values.shape[1])
    ]

