"""
causal_intervention.py -- V2.2 Phase 1: V2.1's `ell_0 = argmax_tau C_{1,tau}`
was called "causal latency", but it is the STRONGEST-MEMORY delay, not
necessarily the delay at which a change to u_t first becomes visible in
the features. This module replaces that inference with a direct
INTERVENTION: two input sequences, identical except at exactly one
timestep, run through the IDENTICAL Hamiltonian/disorder realization
(same `NestedSeeds.reservoir_seed`), and the resulting feature
trajectories compared step by step. The first step at which they diverge
by more than a numerically-justified tolerance is the physical causal
latency `ell_causal`.

This is a completely separate, structural question from "how much
capacity is there at each delay" (`ipc_decomposition`/`causal_latency.py`
answer that) -- Phase 2 keeps all three delay concepts (`ell_causal`,
`ell_detect`, `ell_peak`) distinct, never substituting one for another.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .directional_dqrc import DirectionalConfig
from .seeded_runner import run_directional_dqrc_with_explicit_input
from .validation_utils import NestedSeeds
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import random_input  # noqa: E402


@dataclass
class InterventionResult:
    perturb_t: int
    epsilon_numeric: float
    delta_by_group: dict        # {group_name: np.ndarray of len T, delta[t] = ||X_a[t]-X_b[t]||}
    ell_causal_by_group: dict   # {group_name: int or None}  -- None = never detected within T
    ell_causal: int             # min over groups that DID detect a response (or None if none did)
    timing_table: list


def estimate_numeric_noise_floor(cfg: DirectionalConfig, u: np.ndarray, seeds: NestedSeeds,
                                  reset_period=None) -> float:
    """Runs the SAME (u, seeds) circuit twice and returns the max feature
    difference observed -- this project's simulations are exact
    (`shots=1`, no sampling noise; `method='density_matrix'` is a
    deterministic linear-algebra computation), so two identical runs
    should differ only by floating-point round-off. This IS the
    'numerical precision' half of the tolerance Phase 1 asks for."""
    run1 = run_directional_dqrc_with_explicit_input(cfg, u, seeds, reset_period)
    run2 = run_directional_dqrc_with_explicit_input(cfg, u, seeds, reset_period)
    diffs = [np.max(np.abs(run1.X_mem - run2.X_mem)), np.max(np.abs(run1.X_proc - run2.X_proc)),
             np.max(np.abs(run1.X_cross - run2.X_cross))]
    return float(max(diffs))


def intervention_causal_latency(cfg: DirectionalConfig, T: int, seeds: NestedSeeds, perturb_t: int,
                                 reset_period=None, tolerance_multiplier: float = 100.0,
                                 min_tolerance: float = 1e-6, residualizer_train_frac: float = 0.5
                                 ) -> InterventionResult:
    """Builds u_a (the normal input sequence for this `seeds` object) and
    u_b (IDENTICAL to u_a except `u_b[perturb_t] = 1 - u_a[perturb_t]`,
    the maximal single-timestep perturbation available in the [0,1]
    encoding convention), runs both through the SAME Hamiltonian/disorder
    realization, and reports the step-by-step feature divergence for
    every feature group Phase 1 asks for (X_M, X_P, X_M+X_P, X_P_perp,
    cross). `ell_causal` per group is the smallest tau>=0 such that
    `delta[perturb_t+tau] > epsilon_numeric`; a numerically-justified
    tolerance is derived from an actual repeated-identical-run noise-floor
    measurement, not an arbitrary constant."""
    u_a = random_input(T, seed=seeds.input_seed)
    u_b = u_a.copy()
    u_b[perturb_t] = 1.0 - u_a[perturb_t]

    noise_floor = estimate_numeric_noise_floor(cfg, u_a, seeds, reset_period)
    epsilon_numeric = max(tolerance_multiplier * noise_floor, min_tolerance)

    run_a = run_directional_dqrc_with_explicit_input(cfg, u_a, seeds, reset_period)
    run_b = run_directional_dqrc_with_explicit_input(cfg, u_b, seeds, reset_period)

    from .feature_analysis import fit_residualizer
    n_train_fit = max(int(residualizer_train_frac * perturb_t), 5) if perturb_t >= 10 else None
    if n_train_fit is not None:
        train_idx = np.arange(n_train_fit)
        applier = fit_residualizer(run_a.X_proc, run_a.X_mem, train_idx)
        X_P_perp_a = applier(run_a.X_proc, run_a.X_mem)
        X_P_perp_b = applier(run_b.X_proc, run_b.X_mem)
    else:
        X_P_perp_a, X_P_perp_b = run_a.X_proc, run_b.X_proc

    groups = {
        "X_M": (run_a.X_mem, run_b.X_mem), "X_P": (run_a.X_proc, run_b.X_proc),
        "X_M+X_P": (run_a.X_combined, run_b.X_combined), "X_P_perp": (X_P_perp_a, X_P_perp_b),
        "cross": (run_a.X_cross, run_b.X_cross),
    }
    delta_by_group, ell_by_group = {}, {}
    for name, (Xa, Xb) in groups.items():
        delta = np.linalg.norm(Xa - Xb, axis=1)
        delta_by_group[name] = delta
        detected = None
        for tau in range(T - perturb_t):
            if delta[perturb_t + tau] > epsilon_numeric:
                detected = tau
                break
        ell_by_group[name] = detected

    detected_vals = [v for v in ell_by_group.values() if v is not None]
    ell_causal = min(detected_vals) if detected_vals else None

    from .causal_latency import circuit_timing_table
    timing_table = circuit_timing_table(cfg.memory_variant)

    return InterventionResult(perturb_t=perturb_t, epsilon_numeric=epsilon_numeric,
                               delta_by_group=delta_by_group, ell_causal_by_group=ell_by_group,
                               ell_causal=ell_causal, timing_table=timing_table)
