"""
driven_timescales.py -- V2.2 Phase 12: V2.1's timescale analysis used
ISOLATED memory/processor layer unitaries (no input driving, no collision
channel, no reset) -- convenient for a clean autocorrelation definition,
but not what the correlation structure of the ACTUAL driven, open DQRC
circuit looks like. This module computes the same kind of correlation
descriptor directly from a REAL `DirectionalRun`'s own feature time
series (input encoding + memory evolution + collision channel + processor
evolution + optional reset all included, exactly as executed), then
reuses `robust_timescales.compute_robust_timescale_descriptors`
(unchanged, still the correct "no invalid exponential claims" machinery)
on it.
"""
from __future__ import annotations

import numpy as np

from .directional_dqrc import DirectionalRun
from .robust_timescales import compute_robust_timescale_descriptors, RobustTimescaleDescriptors


def feature_series_autocorrelation(x: np.ndarray, max_lag: int):
    """Empirical (sample) autocorrelation of a single feature TIME SERIES
    `x` (length T, one column of X_mem/X_proc from a real driven run) --
    mean-subtracted, normalized so `corr[0]=1`. This is a standard
    time-series autocorrelation of the OBSERVED trajectory, not an
    infinite-temperature Heisenberg-picture operator autocorrelation (the
    isolated-layer approach V2.1 used) -- the two are different
    quantities; this one reflects the actual driven dynamics including
    the input's own randomness."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x ** 2))
    steps, corr = [], []
    for lag in range(max_lag + 1):
        steps.append(lag)
        if lag == 0:
            corr.append(1.0)
        elif denom < 1e-12:
            corr.append(0.0)
        else:
            corr.append(float(np.sum(x[:len(x) - lag] * x[lag:]) / denom))
    return steps, corr


def driven_system_timescales(run: DirectionalRun, max_lag: int, mem_col: int = 0, proc_col: int = 0) -> dict:
    """Computes driven-trajectory autocorrelation descriptors for one
    representative memory feature column and one representative processor
    feature column. Returns {'memory': RobustTimescaleDescriptors,
    'processor': RobustTimescaleDescriptors, 'memory_raw': (steps,corr),
    'processor_raw': (steps,corr)}."""
    steps_m, corr_m = feature_series_autocorrelation(run.X_mem[:, mem_col], max_lag)
    steps_p, corr_p = feature_series_autocorrelation(run.X_proc[:, proc_col], max_lag)
    desc_m = compute_robust_timescale_descriptors(steps_m, corr_m)
    desc_p = compute_robust_timescale_descriptors(steps_p, corr_p)
    return {"memory": desc_m, "processor": desc_p, "memory_raw": (steps_m, corr_m),
            "processor_raw": (steps_p, corr_p)}
