"""
fixed_delay_response.py -- V2.2 Phases 5/9/10: the 5-point response
analysis, built entirely on the FROZEN readout protocol
(`frozen_protocol.py`) so every stencil point differentiates the SAME
target list, ridge hyperparameter, and null-permutation draws -- only the
physics (the reservoir features) legitimately varies. Compares four
derivative-estimation methods (small central difference, large central
difference, five-point stencil, local quadratic fit) and classifies each
partial derivative STABLE / UNSTABLE / NOT EVALUABLE (Phase 10).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .directional_dqrc import DirectionalConfig
from .frozen_protocol import freeze_protocol, evaluate_frozen
from .ipc_decomposition import nl_tensor_by_fixed_delay, capacity_at_delay
from .seeded_runner import run_directional_dqrc_seeded
from .validation_utils import ControlRange, NestedSeeds, is_unstable
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import chrono_split  # noqa: E402

AXES = ("m", "g", "J")


def _build_cfg(N_M, N_P, theta, phi, ap_kind, m, g, J):
    return DirectionalConfig(memory_variant="protected_integrable", N_M=N_M, N_P=N_P, g_processor=g,
                              J_processor=J, epsilon_M=m, theta=theta, phi=phi, ap_kind=ap_kind)


@dataclass
class StencilPointValue:
    axis: str
    offset_tilde: float          # 0.0 for center, else in {-2h,-h,h,2h}
    M_signed: float
    NL_signed_by_delay: dict     # {tau: signed}


@dataclass
class OneSeedStencil:
    seeds: NestedSeeds
    center: StencilPointValue
    per_axis: dict               # {axis: [StencilPointValue for -2h,-h,h,2h]}


def run_stencil_for_seed(N_M: int, N_P: int, theta: float, phi: float, ap_kind: str, m0: float, g0: float,
                          J0: float, m_range: ControlRange, g_range: ControlRange, J_range: ControlRange,
                          h_small: float, h_large: float, seeds: NestedSeeds, T: int, washout: int, n_val: int,
                          n_test: int, max_delay: int, max_degree: int, max_targets_per_degree: int,
                          n_surrogates: int) -> OneSeedStencil:
    """Runs the center point, FREEZES the readout protocol on it, then
    evaluates the 4 non-center offsets on EACH axis using
    `evaluate_frozen` -- 13 circuit simulations total for one seed (1
    center + 4 offsets x 3 axes)."""
    gap = max_delay + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)

    def run_at(m, g, J):
        cfg = _build_cfg(N_M, N_P, theta, phi, ap_kind, m, g, J)
        return run_directional_dqrc_seeded(cfg, T=T, seeds=seeds, reset_period=None)

    center_run = run_at(m0, g0, J0)
    spec = freeze_protocol(center_run.u, center_run.X_combined, train, val, test, max_delay=max_delay,
                            max_degree=max_degree, max_targets_per_degree=max_targets_per_degree,
                            n_surrogates=n_surrogates, seed=seeds.reservoir_seed)
    nl_center = nl_tensor_by_fixed_delay(spec.center_records, max_delay)
    M_center = capacity_at_delay(spec.center_records, delay=0, degree_min=1, degree_max=1)
    m_records_center = [r for r in spec.center_records if r.degree == 1]
    M_signed_center = sum(r.raw_capacity - r.null_mean for r in m_records_center)
    center_pt = StencilPointValue(axis="center", offset_tilde=0.0, M_signed=M_signed_center,
                                   NL_signed_by_delay={tau: e["signed"] for tau, e in nl_center.items()})

    m0_t, g0_t, J0_t = m_range.to_dimensionless(m0), g_range.to_dimensionless(g0), J_range.to_dimensionless(J0)
    per_axis = {}
    for axis, rng_obj, base_t in (("m", m_range, m0_t), ("g", g_range, g0_t), ("J", J_range, J0_t)):
        points = []
        for h in (-h_large, -h_small, h_small, h_large):
            p_t = base_t + h
            p_raw = rng_obj.from_dimensionless(p_t)
            m_eval = p_raw if axis == "m" else m0
            g_eval = p_raw if axis == "g" else g0
            J_eval = p_raw if axis == "J" else J0
            run = run_at(m_eval, g_eval, J_eval)
            records = evaluate_frozen(run.u, run.X_combined, train, val, test, spec)
            nl_tau = nl_tensor_by_fixed_delay(records, max_delay)
            m_records = [r for r in records if r.degree == 1]
            M_signed = sum(r.raw_capacity - r.null_mean for r in m_records)
            points.append(StencilPointValue(axis=axis, offset_tilde=h, M_signed=M_signed,
                                             NL_signed_by_delay={tau: e["signed"] for tau, e in nl_tau.items()}))
        per_axis[axis] = points

    return OneSeedStencil(seeds=seeds, center=center_pt, per_axis=per_axis)


def _derivative_methods(values_by_offset: dict, h_small: float, h_large: float) -> dict:
    """`values_by_offset` = {-2h:.., -h:.., 0:.., h:.., 2h:..} (using the
    ACTUAL h_small/h_large as keys -2*h_large etc. -- see caller). Returns
    {'small_cd':.., 'large_cd':.., 'five_point':.., 'quadratic':..}."""
    f_m2h, f_mh, f_0, f_ph, f_p2h = (values_by_offset[-h_large], values_by_offset[-h_small],
                                      values_by_offset[0.0], values_by_offset[h_small], values_by_offset[h_large])
    small_cd = (f_ph - f_mh) / (2 * h_small)
    large_cd = (f_p2h - f_m2h) / (2 * h_large)
    # classic 5-point stencil (assumes h_large == 2*h_small, i.e. a UNIFORM step h=h_small)
    five_point = (-f_p2h + 8 * f_ph - 8 * f_mh + f_m2h) / (12 * h_small) if abs(h_large - 2 * h_small) < 1e-9 \
        else float("nan")
    xs = np.array([-h_large, -h_small, 0.0, h_small, h_large])
    ys = np.array([f_m2h, f_mh, f_0, f_ph, f_p2h])
    coeffs = np.polyfit(xs, ys, deg=2)  # [c2, c1, c0]
    quadratic = float(coeffs[1])
    return {"small_cd": float(small_cd), "large_cd": float(large_cd), "five_point": float(five_point),
            "quadratic": quadratic}


@dataclass
class DerivativeEstimate:
    axis: str
    target: str                  # 'M' or f'NL_tau{tau}'
    methods: dict                 # {'small_cd':..,'large_cd':..,'five_point':..,'quadratic':..}
    across_seed_means: list       # per-seed mean-across-methods estimate (for cross-seed stability)
    stability: dict               # is_unstable()'s own dict, evaluated across METHODS (one seed) or SEEDS


def aggregate_stencils(stencils: list, target_delay: int) -> dict:
    """`stencils` = list[OneSeedStencil] (one per exploratory seed). For
    each axis, computes the 4 derivative-method estimates PER SEED (for M
    and for NL at `target_delay`), then classifies stability both within-
    seed (across the 4 methods) and across-seed (using each seed's
    'five_point' estimate, the most accurate single method)."""
    per_axis_target = {}
    for axis in AXES:
        for target_name, extractor in (("M", lambda p: p.M_signed),
                                        (f"NL_tau{target_delay}", lambda p: p.NL_signed_by_delay[target_delay])):
            per_seed_methods = []
            for stencil in stencils:
                pts = stencil.per_axis[axis]
                h_large = abs(pts[0].offset_tilde)
                h_small = abs(pts[1].offset_tilde)
                values_by_offset = {-h_large: extractor(pts[0]), -h_small: extractor(pts[1]),
                                     0.0: extractor(stencil.center), h_small: extractor(pts[2]),
                                     h_large: extractor(pts[3])}
                per_seed_methods.append(_derivative_methods(values_by_offset, h_small, h_large))

            within_seed_stability = [is_unstable([m["small_cd"], m["large_cd"], m["five_point"], m["quadratic"]])
                                      for m in per_seed_methods if not np.isnan(m["five_point"])]
            across_seed_estimates = [m["five_point"] for m in per_seed_methods if not np.isnan(m["five_point"])]
            across_seed_stability = is_unstable(across_seed_estimates) if len(across_seed_estimates) > 1 else None

            per_axis_target[(axis, target_name)] = DerivativeEstimate(
                axis=axis, target=target_name, methods=per_seed_methods[0] if per_seed_methods else {},
                across_seed_means=across_seed_estimates,
                stability={"within_seed": within_seed_stability, "across_seed": across_seed_stability},
            )
    return per_axis_target
