"""
timescales.py -- Part 13 of the V2 validation spec: independent memory
(tau_M) and processor (tau_P) autocorrelation/mixing timescales, and the
hypothesis test tau_P << tau_M (NOT assumed -- Part 13 is explicit that
this is a hypothesis).

Both timescales use the SAME infinite-temperature Heisenberg-picture
operator-autocorrelation construction `directional_memory.z_autocorrelation`
already implements for the memory sector -- this module reuses that
function for tau_M and applies the identical construction to the
processor's own single-layer unitary for tau_P, rather than defining two
different, harder-to-compare notions of "timescale".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .directional_memory import z_autocorrelation
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402


def processor_z_autocorrelation(processor_params, n_steps: int, qubit: int = 0):
    """Identical construction to `directional_memory.z_autocorrelation`,
    applied to the PROCESSOR's own single-layer unitary
    (`mixed_syk_core.single_layer_unitary_mixed`, reused directly -- the
    same function `processor.processor_chaos_diagnostics` already calls)."""
    U1 = msc.single_layer_unitary_mixed(processor_params.N_p, processor_params.g, processor_params.terms,
                                         processor_params.couplings, processor_params.paulis,
                                         processor_params.bias_z)
    dim = 2 ** processor_params.N_p
    Z = np.array([[1.0]])
    for i in range(processor_params.N_p - 1, -1, -1):
        Z = np.kron(Z, np.array([[1, 0], [0, -1]]) if i == qubit else np.eye(2))
    Z0 = Z.astype(np.complex128)
    Zt = Z0.copy()
    corr = [1.0]
    for _ in range(n_steps):
        Zt = U1.conj().T @ Zt @ U1
        corr.append(float(np.real(np.trace(Zt @ Z0))) / dim / (np.real(np.trace(Z0 @ Z0)) / dim))
    return list(range(n_steps + 1)), corr


@dataclass
class TimescaleFit:
    tau: float
    r_squared: float
    n_points_fit: int
    corr: list
    steps: list
    fit_failed: bool


def fit_exponential_decay_timescale(steps, corr, min_corr: float = 0.02) -> TimescaleFit:
    """Fit |C(t)| ~ exp(-t/tau) via a linear least-squares fit of
    log|C(t)| vs t, using only points where |C(t)| stays above `min_corr`
    (avoids fitting pure numerical noise once the correlator has decayed
    to ~0 -- an explicit, reported goodness-of-fit / fitting-window
    choice, per Part 13's 'report uncertainty and goodness of fit'
    requirement, not a silently-chosen window)."""
    steps = np.asarray(steps, dtype=float)
    corr = np.asarray(corr, dtype=float)
    mask = np.abs(corr) > min_corr
    if mask.sum() < 3:
        return TimescaleFit(tau=float("nan"), r_squared=float("nan"), n_points_fit=int(mask.sum()),
                             corr=corr.tolist(), steps=steps.tolist(), fit_failed=True)
    t_fit = steps[mask]
    y_fit = np.log(np.abs(corr[mask]) + 1e-12)
    A = np.column_stack([t_fit, np.ones_like(t_fit)])
    beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    slope, intercept = beta
    pred = A @ beta
    ss_res = np.sum((y_fit - pred) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    tau = float(-1.0 / slope) if slope < 0 else float("inf")
    return TimescaleFit(tau=tau, r_squared=r_squared, n_points_fit=int(mask.sum()), corr=corr.tolist(),
                         steps=steps.tolist(), fit_failed=(slope >= 0))
