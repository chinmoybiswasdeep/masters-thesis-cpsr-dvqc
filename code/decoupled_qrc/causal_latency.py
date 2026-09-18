"""
causal_latency.py -- V2.1 Phase 1: the FIRST experiment run, before any
candidate is selected. The directional pipeline u_t -> M -> A -> P may
introduce real latency between when an input enters and when it is fully
"visible" through the processor's own readout, and V2's own metric
(NL_instant = capacity_at_delay(records, 0)) silently assumed that latency
is exactly 0. This module measures where the linear response actually
peaks (`ell_0`) and reports NL at BOTH tau=0 (literal) and tau=ell_0
(architecture-aligned local) -- never silently renaming one as the other.

Built entirely on `ipc_decomposition.compute_ipc_decomposed`'s existing
per-profile records; no new capacity computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge

from .ipc_decomposition import compute_ipc_decomposed, capacity_at_delay, IPCDecomposition


@dataclass
class CausalLatencyResult:
    ell_0: int                       # argmax_tau C_{1,tau} (bias-corrected) -- the diagnostic causal latency
    C1_by_delay: dict                # {tau: {'legacy':..,'raw':..,'null':..,'bc':..}}
    NL_literal: float                # sum_{d>=2} C_{d, tau=0}   (== decomp.NL_instant, kept under its own name)
    NL_literal_bc: float
    NL_local: float                  # sum_{d>=2} C_{d, tau=ell_0}  -- architecture-aligned, NOT renamed from tau=0
    NL_local_bc: float
    probe: dict                      # cross-correlation / MI-proxy / linear-probe-R2, per tau (see `feature_input_alignment`)
    decomp: IPCDecomposition = field(repr=False, default=None)


def causal_latency_profile(u: np.ndarray, X: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray,
                            max_delay: int = 8, max_degree: int = 4, max_targets_per_degree: int = 12,
                            n_surrogates: int = 19, seed: int = 0) -> CausalLatencyResult:
    """Runs ONE `compute_ipc_decomposed` pass (degree 1..max_degree, delay
    0..max_delay) and re-derives `ell_0` and both NL_literal/NL_local from
    its records -- the metric this validation actually differentiates and
    reports going forward is `NL_local` (architecture-aligned), never
    `NL_literal` alone, per Phase 1's explicit instruction."""
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
                                     max_targets_per_degree=max_targets_per_degree, n_surrogates=n_surrogates,
                                     seed=seed)
    c1_by_delay = {tau: capacity_at_delay(decomp.records, tau, degree_min=1) for tau in range(max_delay + 1)}
    # ell_0 via the CONTINUOUS bias-corrected metric (Defect 5: never differentiate/select on the
    # hard-thresholded legacy metric) -- ties broken toward the smallest delay (most physically
    # plausible causal latency, and the M-only case where all delays score ~0 defaults to tau=0).
    ell_0 = max(range(max_delay + 1), key=lambda tau: (c1_by_delay[tau]["bc"], -tau))

    at0 = capacity_at_delay(decomp.records, 0, degree_min=2)
    at_ell0 = capacity_at_delay(decomp.records, ell_0, degree_min=2)
    probe = feature_input_alignment(u, X, train, test, max_delay)

    return CausalLatencyResult(
        ell_0=ell_0, C1_by_delay=c1_by_delay, NL_literal=at0["legacy"], NL_literal_bc=at0["bc"],
        NL_local=at_ell0["legacy"], NL_local_bc=at_ell0["bc"], probe=probe, decomp=decomp,
    )


def feature_input_alignment(u: np.ndarray, X: np.ndarray, train: np.ndarray, test: np.ndarray,
                             max_delay: int, ridge_alpha: float = 1.0) -> dict:
    """For tau=0..max_delay: a linear probe (Ridge, fit on train, scored on
    held-out test) of raw u_{t-tau} FROM the feature matrix X -- distinct
    from (and simpler than) the Legendre-target IPC machinery, reported
    per Phase 1's explicit request for 'feature/input cross-correlations;
    mutual-information proxy...; linear probe accuracy'. The MI figure
    reported here is an explicit GAUSSIAN-APPROXIMATION proxy
    (-0.5*log(1-R^2), the exact mutual information between two jointly
    Gaussian variables with that R^2), not a real nonparametric MI
    estimate -- labeled as such, never presented as literal MI."""
    T = len(u)
    out = {}
    for tau in range(max_delay + 1):
        target = np.zeros(T)
        if tau == 0:
            target[:] = u
        else:
            target[tau:] = u[:T - tau]
        valid_train = train[train >= tau]
        if len(valid_train) < 5 or len(test) < 5:
            out[tau] = {"r2": float("nan"), "pearson_r": float("nan"), "mi_proxy_nats": float("nan")}
            continue
        model = Ridge(alpha=ridge_alpha).fit(X[valid_train], target[valid_train])
        pred = model.predict(X[test])
        y_true = target[test]
        ss_res = np.sum((pred - y_true) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)
        r2_clipped = float(np.clip(r2, 0.0, 1.0 - 1e-9))
        pearson = float(np.corrcoef(pred, y_true)[0, 1]) if np.std(pred) > 1e-12 else 0.0
        mi_proxy = float(-0.5 * np.log(1.0 - r2_clipped))
        out[tau] = {"r2": r2, "pearson_r": pearson, "mi_proxy_nats": mi_proxy}
    return out


def circuit_timing_table(memory_variant: str) -> list:
    """Phase 1's required explicit per-timestep timing table -- a STATIC
    description of `directional_dqrc.build_directional_circuit`'s own
    instruction order (verified by reading that function directly; kept
    here as a single source of truth rather than re-derived by hand in
    every notebook/report). Each row: (stage, description, causal
    dependency). The instruction order inside the `for t, u_t in
    enumerate(u_seq)` loop is, exactly: (1) mem_step(u_t) [memory encodes
    u_t and evolves], (2) apply_collision_step [fresh ancilla mediates
    M->P], (3) processor layer, (4) save_expectation_value reads for step
    t. So X_t (the features saved at step t) CAN and DOES causally depend
    on u_t (encoded at the START of the same step, before any readout) --
    consistent with ell_0=0 being physically plausible, while ell_0>0
    would indicate the *processor's own* nonlinear response lags behind
    the memory's encoding of u_t by ell_0 extra steps before it becomes
    linearly visible in X_t, not that the circuit itself is a-causal."""
    return [
        {"step": 1, "stage": "input encoding", "description": f"u_t encoded into the memory register "
         f"(memory_variant={memory_variant!r})", "depends_on_u_t": True},
        {"step": 2, "stage": "memory evolution", "description": "memory register's own (shift or "
         "protected-integrable) internal dynamics applied for this timestep", "depends_on_u_t": True},
        {"step": 3, "stage": "memory-ancilla interaction", "description": "fresh ancilla A reset to |0>, "
         "then QND-in-Z_M collision coupling (theta,phi) applied between the tapped memory qubit and A",
         "depends_on_u_t": True},
        {"step": 4, "stage": "ancilla-processor interaction", "description": "the SAME collision step "
         "couples A into the processor's entry qubit (ap_kind exchange coupling)", "depends_on_u_t": True},
        {"step": 5, "stage": "processor layer", "description": "one EOC processor layer (mixed SYK2/SYK4, "
         "params g,J) applied to the full processor register", "depends_on_u_t": True},
        {"step": 6, "stage": "feature readout", "description": "save_expectation_value for every memory/"
         "processor/cross operator, LABELED with this timestep t -- taken AFTER steps 1-5 above, so X_t "
         "can and does causally depend on u_t", "depends_on_u_t": True},
    ]
