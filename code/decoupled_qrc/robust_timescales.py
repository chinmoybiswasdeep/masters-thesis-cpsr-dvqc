"""
robust_timescales.py -- V2.1 Phase 9: V2's `timescales.py` reported
`tau_P=11.2 (r2=0.18)` and `tau_M=inf (r2=nan)` -- both FAILED exponential
fits (an r2 of 0.18 is barely better than a flat line; tau=inf/nan is not
a timescale at all) that were nonetheless printed as if they were usable
numbers. For OSCILLATORY, finite-size, closed-quantum-system dynamics
(the memory/processor autocorrelation functions in this project are
generically NOT simple exponential decays -- they can ring, partially
revive, and plateau at a nonzero long-time value set by the finite Hilbert
space), a single exponential is frequently the wrong model entirely.

This module reports SEVERAL descriptors that make sense regardless of
whether the underlying decay is exponential, and only reports a fitted
exponential tau when a preregistered quality gate is actually met.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustTimescaleDescriptors:
    integrated_abs_time: float        # sum_t |C(t)| * dt  (dt=1 step) -- always defined, model-free
    first_1e_crossing: float          # smallest t where |C(t)| first drops below |C(0)|/e; NaN if it never does
    first_zero_crossing: float        # smallest t where C(t) first changes sign; NaN if it never does
    envelope_decay_rate: float        # slope of a robust (Huber-like) fit to log(running-max envelope); NaN if unfit
    long_time_plateau: float          # mean of |C(t)| over the LAST quarter of the window -- the "does it fully decay" check
    recurrence_amplitude: float       # max |C(t)| for t beyond first_zero_crossing (or beyond 20% of window if no zero crossing) -- revival strength
    exponential_tau: float            # NaN unless `fit_quality_ok` below
    exponential_r2: float
    exponential_fit_quality_ok: bool  # the preregistered gate: r2 >= min_r2 AND n_points_fit >= min_points
    identifiable: str                 # "EXPONENTIAL" | "NOT IDENTIFIABLE" -- the single label callers should branch on


def _first_1e_crossing(steps, corr):
    c0 = abs(corr[0]) + 1e-12
    for t, c in zip(steps, corr):
        if abs(c) < c0 / np.e:
            return float(t)
    return float("nan")


def _first_zero_crossing(steps, corr):
    for i in range(1, len(corr)):
        if corr[i - 1] * corr[i] < 0:
            # linear interpolation between the two bracketing steps
            t0, t1 = steps[i - 1], steps[i]
            c0, c1 = corr[i - 1], corr[i]
            frac = c0 / (c0 - c1) if (c0 - c1) != 0 else 0.0
            return float(t0 + frac * (t1 - t0))
    return float("nan")


def _envelope_decay_rate(steps, corr, min_corr=0.02):
    """Fits log(running-max |C|) vs t via ordinary least squares over the
    points where the running max is still above `min_corr` -- a crude but
    honest 'is the ENVELOPE decaying at all' rate, robust to individual
    oscillation zero-crossings (unlike fitting log|C(t)| directly, which
    blows up at every zero crossing)."""
    corr = np.asarray(corr, dtype=float)
    # env(t) = max(|C(tau)| for tau>=t) -- monotone non-increasing by construction, so oscillation
    # zero-crossings (which would blow up a direct log|C(t)| fit) never enter this envelope.
    env = np.array([np.max(np.abs(corr[i:])) for i in range(len(corr))])
    steps = np.asarray(steps, dtype=float)
    mask = env > min_corr
    if mask.sum() < 3:
        return float("nan")
    t_fit = steps[mask]
    y_fit = np.log(env[mask] + 1e-12)
    A = np.column_stack([t_fit, np.ones_like(t_fit)])
    beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    slope = float(beta[0])
    return slope if slope < 0 else float("nan")


def compute_robust_timescale_descriptors(steps, corr, min_r2: float = 0.8, min_points_fit: int = 6,
                                          min_corr: float = 0.02) -> RobustTimescaleDescriptors:
    steps = np.asarray(steps, dtype=float)
    corr = np.asarray(corr, dtype=float)
    dt = float(np.mean(np.diff(steps))) if len(steps) > 1 else 1.0

    integrated_abs_time = float(np.sum(np.abs(corr)) * dt)
    t_1e = _first_1e_crossing(steps, corr)
    t_zero = _first_zero_crossing(steps, corr)
    envelope_rate = _envelope_decay_rate(steps, corr, min_corr=min_corr)

    n = len(corr)
    tail = corr[max(0, n - max(1, n // 4)):]
    plateau = float(np.mean(np.abs(tail)))

    if not np.isnan(t_zero):
        beyond = corr[steps > t_zero]
    else:
        beyond = corr[steps > 0.2 * steps[-1]]
    recurrence = float(np.max(np.abs(beyond))) if len(beyond) > 0 else float("nan")

    # exponential fit, same construction as the (now-superseded) timescales.py, but
    # GATED: tau/r2 are only ever REPORTED (non-NaN) if the preregistered quality bar is met.
    mask = np.abs(corr) > min_corr
    exp_tau, exp_r2, ok = float("nan"), float("nan"), False
    if mask.sum() >= min_points_fit:
        t_fit = steps[mask]
        y_fit = np.log(np.abs(corr[mask]) + 1e-12)
        A = np.column_stack([t_fit, np.ones_like(t_fit)])
        beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
        slope, intercept = beta
        pred = A @ beta
        ss_res = np.sum((y_fit - pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        if slope < 0 and not np.isnan(r2) and r2 >= min_r2:
            exp_tau, exp_r2, ok = float(-1.0 / slope), r2, True
        elif not np.isnan(r2):
            exp_r2 = r2  # report the r2 even on failure, so callers can see HOW badly it failed

    return RobustTimescaleDescriptors(
        integrated_abs_time=integrated_abs_time, first_1e_crossing=t_1e, first_zero_crossing=t_zero,
        envelope_decay_rate=envelope_rate, long_time_plateau=plateau, recurrence_amplitude=recurrence,
        exponential_tau=exp_tau, exponential_r2=exp_r2, exponential_fit_quality_ok=ok,
        identifiable=("EXPONENTIAL" if ok else "NOT IDENTIFIABLE"),
    )
