"""
v3_2_gates.py -- the preregistered acceptance gates A..L.

Thresholds are fixed here, BEFORE any discovery or confirmation data exists,
and are written into the frozen artifact. Changing a threshold after seeing
confirmation results is the one thing this whole iteration exists to prevent.

THREE OUTCOMES, NOT TWO. Every gate returns PASS, FAIL or NOT_EVALUABLE.
V2.2 established why this matters: an unstable derivative cannot support a
decisive ratio claim in EITHER direction, so reporting it as FAIL would
overstate the evidence exactly as reporting it as PASS would.

'NOT SIGNIFICANT' IS NOT 'ZERO'. Gate F is an equivalence test against an
upper confidence bound, never a failure to reject zero.

E_NL REUSES THE REPOSITORY'S OWN CONVENTION, `v3_1_stats.dynamic_range`:
E = (Q_0.9 - Q_0.1) / (|median| + eps), preregistered threshold 0.20 -- the
same definition and the same number V3.1 was measured against (and failed, at
0.0739).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .v3_1_stats import dynamic_range

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"

THRESHOLDS = {
    "A_isolation_atol": 1e-10,
    "A_control_detect_min": 1e-6,
    "B_dm_atol": 1e-10,
    "B_cache_atol": 1e-12,
    "B_train_per_rank": 10,
    "B_train_absolute": 500,
    "C_null_fraction_max": 0.20,
    "C_ceiling_frac_max": 0.95,
    "D_M_long_min": 1.0,
    "D_rel_spread_max": 0.5,
    "D_min_sign_agreement": 3,
    "E_NL0_min": 0.5,
    "E_dynamic_range_min": 0.20,
    "E_f_enc_max": 0.5,
    "F_equivalence_margin": 0.25,
    "G_selectivity_min": 2.0,
    "H_angle_lo": 70.0,
    "H_angle_hi": 110.0,
    "I_retention_min": 0.7,
    "J_min_seeds_passing": 4,
    "J_n_seeds": 5,
    "K_min_neighbours_passing": 4,
    "K_n_neighbours": 6,
    "L_disturbance_max": 0.05,
}


@dataclass
class GateResult:
    name: str
    status: str
    values: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "passed": self.passed,
                "values": self.values, "thresholds": self.thresholds, "reason": self.reason}


def _res(name, ok, values, thresholds, reason_pass="", reason_fail=""):
    return GateResult(name=name, status=PASS if ok else FAIL, values=values,
                      thresholds=thresholds, reason=reason_pass if ok else reason_fail)


def gate_A_structural_isolation(iso_report: dict, injected_control: dict,
                                 serial_control: dict = None) -> GateResult:
    """X_P must not depend on m at lambda=0, and the controls must DETECT a
    planted dependency -- otherwise the test proves nothing."""
    atol = THRESHOLDS["A_isolation_atol"]
    detect = THRESHOLDS["A_control_detect_min"]
    vals = {"max_dXP_dm": iso_report.get("max_dXP_dm"),
            "max_dXM_dg": iso_report.get("max_dXM_dg"),
            "max_dXM_dJ": iso_report.get("max_dXM_dJ"),
            "injected_control_max_dXP_dm": injected_control.get("max_dXP_dm"),
            "injected_control_passed": injected_control.get("passed")}
    ok_iso = bool(iso_report.get("passed"))
    ok_ctl = bool((injected_control.get("max_dXP_dm") or 0.0) > detect
                  and not injected_control.get("passed"))
    if serial_control is not None:
        vals["serial_control_max_dXP_dm"] = serial_control.get("max_dXP_dm")
        ok_ctl = ok_ctl and bool((serial_control.get("max_dXP_dm") or 0.0) > detect)
    return _res("A_structural_isolation", ok_iso and ok_ctl, vals,
                {"atol": atol, "control_detect_min": detect},
                "routes isolated and the falsifiability controls detect a planted dependency",
                "isolation violated, or a negative control failed to detect its planted dependency")


def gate_B_numerical_validity(*, tests_passed: bool, ipc_report, dm_audit: dict,
                               cache_equivalence: float = None) -> GateResult:
    atol, catol = THRESHOLDS["B_dm_atol"], THRESHOLDS["B_cache_atol"]
    vals = {"tests_passed": bool(tests_passed),
            "n_train": ipc_report.n_train, "required_n_train": ipc_report.sample_size_required,
            "sample_size_ok": ipc_report.sample_size_ok,
            "effective_rank": ipc_report.effective_rank,
            "numerical_rank": ipc_report.numerical_rank,
            "trace_dev": dm_audit.get("trace_dev"),
            "hermiticity_dev": dm_audit.get("hermiticity_dev"),
            "min_eigenvalue": dm_audit.get("min_eigenvalue"),
            "cache_equivalence_max_abs_diff": cache_equivalence}
    ok = bool(tests_passed and ipc_report.sample_size_ok
              and (dm_audit.get("trace_dev", 1) <= atol)
              and (dm_audit.get("hermiticity_dev", 1) <= atol)
              and (dm_audit.get("min_eigenvalue", -1) >= -atol)
              and (cache_equivalence is None or cache_equivalence <= catol))
    return _res("B_numerical_validity", ok, vals,
                {"dm_atol": atol, "cache_atol": catol,
                 "train_per_rank": THRESHOLDS["B_train_per_rank"],
                 "train_absolute": THRESHOLDS["B_train_absolute"]},
                "numerics, sample size and cache equivalence all valid",
                "a numerical-validity precondition failed")


def gate_C_ipc_validity(ipc_report, tail_decayed: bool = None) -> GateResult:
    nmax, cmax = THRESHOLDS["C_null_fraction_max"], THRESHOLDS["C_ceiling_frac_max"]
    ceil_nl0 = ipc_report.ceiling_NL0 or {}
    ceil_m = ipc_report.ceiling_M or {}
    # `ceiling_audit.CeilingReport.as_dict()` emits `fraction_of_ceiling` and
    # `ceiling_contaminated`. Reading a key it does not emit silently returns
    # None, and a None fraction passes every comparison -- which made this
    # guard unreachable against the exact defect V3.1 was diagnosed with.
    frac_nl0 = ceil_nl0.get("fraction_of_ceiling")
    frac_m = ceil_m.get("fraction_of_ceiling")
    flagged = bool(ceil_nl0.get("ceiling_contaminated") or ceil_m.get("ceiling_contaminated"))
    vals = {"null_fraction_NL0": ipc_report.null_fraction_NL0,
            "null_fraction_M": ipc_report.null_fraction_M,
            "ceiling_frac_NL0": frac_nl0, "ceiling_frac_M": frac_m,
            "ceiling_contaminated_flag": flagged,
            "tail_decayed": tail_decayed,
            "degrees_above_structural_bound": ipc_report.degrees_above_structural_bound,
            "was_capped": ipc_report.was_capped}

    def _nf_ok(x):
        return bool(x is not None and np.isfinite(x) and x < nmax)

    def _cf_ok(x):
        # A MISSING fraction is not evidence of safety: refuse it.
        return bool(x is not None and np.isfinite(x) and x <= cmax)

    if not np.isfinite(ipc_report.null_fraction_NL0) and not np.isfinite(ipc_report.null_fraction_M):
        return GateResult("C_ipc_validity", NOT_EVALUABLE, vals,
                          {"null_fraction_max": nmax, "ceiling_frac_max": cmax},
                          "both raw capacities are ~0, so the null fraction is undefined")
    ok = (_nf_ok(ipc_report.null_fraction_NL0) and _nf_ok(ipc_report.null_fraction_M)
          and _cf_ok(frac_nl0) and _cf_ok(frac_m) and not flagged
          and (tail_decayed is not False))
    return _res("C_ipc_validity", ok, vals,
                {"null_fraction_max": nmax, "ceiling_frac_max": cmax},
                "null bias, ceilings and tail all within preregistered limits",
                "null fraction too high, capacity ceiling-contaminated, or tail not decayed")


def gate_D_memory_controllability(M_long: float, dM_dm_verdict) -> GateResult:
    thr = THRESHOLDS
    est = list(dM_dm_verdict.estimates.values())
    signs = [np.sign(v) for v in est if np.isfinite(v) and abs(v) > 1e-12]
    agree = int(max((signs.count(s) for s in set(signs)), default=0))
    ci = dM_dm_verdict.ci
    vals = {"M_long": float(M_long), "dM_dm": dM_dm_verdict.point_estimate,
            "estimators": dM_dm_verdict.estimates, "sign_agreement_count": agree,
            "rel_spread": dM_dm_verdict.rel_spread, "stable": dM_dm_verdict.stable,
            "ci": ([ci.lo, ci.hi] if ci is not None else None),
            "reasons": dM_dm_verdict.reasons}
    if ci is None:
        return GateResult("D_memory_controllability", NOT_EVALUABLE, vals, dict(thr),
                          "no per-seed bootstrap available for dM_long/dm~")
    ci_excludes_zero = bool(np.isfinite(ci.lo) and np.isfinite(ci.hi) and not (ci.lo <= 0 <= ci.hi))
    ok = bool(M_long >= thr["D_M_long_min"] and ci_excludes_zero
              and agree >= thr["D_min_sign_agreement"]
              and dM_dm_verdict.rel_spread <= thr["D_rel_spread_max"])
    return _res("D_memory_controllability", ok, vals,
                {k: thr[k] for k in ("D_M_long_min", "D_rel_spread_max", "D_min_sign_agreement")},
                "m produces a nontrivial, stable, nonzero long-delay memory response",
                "memory response trivial, unstable, or its CI contains zero")


def gate_E_processor_controllability(*, NL_0: float, nl0_values_over_grid,
                                      delta_interaction_ci, f_enc: float,
                                      dNL_dg_verdict=None, dNL_dJ_verdict=None) -> GateResult:
    """`delta_interaction_ci` must be a paired CI on NL_0(g,J) - NL_0(0,0)."""
    thr = THRESHOLDS
    dr = dynamic_range(list(nl0_values_over_grid))
    lo = getattr(delta_interaction_ci, "lo", None)
    hi = getattr(delta_interaction_ci, "hi", None)
    point = getattr(delta_interaction_ci, "point", None)
    resp_stable = [v.stable for v in (dNL_dg_verdict, dNL_dJ_verdict) if v is not None]
    vals = {"NL_0": float(NL_0), "E_NL": dr["E"], "E_NL_median": dr["median"],
            "E_NL_q10": dr["q10"], "E_NL_q90": dr["q90"], "n_grid": dr["n"],
            "delta_interaction": point, "delta_interaction_ci": [lo, hi],
            "f_enc": float(f_enc),
            "processor_response_stable": (all(resp_stable) if resp_stable else None)}
    if lo is None or not np.isfinite(lo):
        return GateResult("E_processor_controllability", NOT_EVALUABLE, vals, dict(thr),
                          "no paired confidence interval on the interaction-generated NL_0")
    delta_ok = bool(lo > 0.0)
    ok = bool(NL_0 >= thr["E_NL0_min"] and np.isfinite(dr["E"])
              and dr["E"] > thr["E_dynamic_range_min"] and delta_ok
              and f_enc < thr["E_f_enc_max"]
              and (all(resp_stable) if resp_stable else True))
    return _res("E_processor_controllability", ok, vals,
                {k: thr[k] for k in ("E_NL0_min", "E_dynamic_range_min", "E_f_enc_max")},
                "interaction-generated nonlinearity with real dynamic range and a low encoder fraction",
                "nonlinearity trivial, ceiling-flat, encoder-dominated, or not interaction-generated")


def gate_F_cross_suppression(bounds: list) -> GateResult:
    """`bounds` are `v3_2_response.equivalence_bound` dicts.

    Entries flagged `structural=True` (an off-diagonal that is zero by
    construction) are reported as STRUCTURAL and do not count as empirical
    evidence of suppression.

    NOTE: at lambda > 0 NONE of the three off-diagonals is structural. The
    planning assumption that dM_long/dg~ and dM_long/dJ~ were zero by
    construction was refuted by direct measurement -- tracing the processor
    away after a joint unitary leaves (g,J)-dependent back-action on the
    memory (see `v3_2_architecture`). They are flagged structural only for the
    lambda = 0 control architecture.
    """
    margin = THRESHOLDS["F_equivalence_margin"]
    empirical = [b for b in bounds if not b.get("structural")]
    structural = [b for b in bounds if b.get("structural")]
    vals = {"bounds": bounds, "n_empirical": len(empirical), "n_structural": len(structural)}
    if not empirical:
        return GateResult("F_cross_suppression", NOT_EVALUABLE, vals,
                          {"equivalence_margin": margin},
                          "every off-diagonal is structurally zero: there is no empirical "
                          "suppression claim to make (a zero by construction is not evidence)")
    unresolved = [b["name"] for b in empirical if not np.isfinite(b.get("upper_abs_bound", np.nan))]
    if unresolved:
        return GateResult("F_cross_suppression", NOT_EVALUABLE, vals,
                          {"equivalence_margin": margin},
                          f"no usable upper confidence bound for {unresolved}")
    ok = all(b["suppressed"] for b in empirical)
    return _res("F_cross_suppression", ok, vals, {"equivalence_margin": margin},
                "every empirical off-diagonal is bounded below its equivalence margin",
                "an off-diagonal upper confidence bound exceeds its equivalence margin")


def gate_G_selectivity(selectivity_report: dict) -> GateResult:
    thr = THRESHOLDS["G_selectivity_min"]
    rm, rnl = selectivity_report.get("R_M", {}), selectivity_report.get("R_NL0", {})
    vals = {"R_M": rm, "R_NL0": rnl, "stable": selectivity_report.get("stable")}
    if rm.get("kind") == "not_evaluable" or rnl.get("kind") == "not_evaluable":
        return GateResult("G_selectivity", NOT_EVALUABLE, vals, {"min_ratio": thr},
                          "a component derivative is unstable: a decisive ratio cannot be "
                          "computed from it in either direction")
    def _val(d):
        """A resolved ratio, else the defensible lower bound `v3_1_stats`
        reports when the cross-sensitivity denominator is below resolution."""
        for k in ("ratio", "ratio_lower_bound"):
            v = d.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)
        return float("nan")
    v_m, v_nl = _val(rm), _val(rnl)
    vals["R_M_value"], vals["R_NL0_value"] = v_m, v_nl
    if not (np.isfinite(v_m) and np.isfinite(v_nl)):
        return GateResult("G_selectivity", NOT_EVALUABLE, vals, {"min_ratio": thr},
                          "selectivity ratios are not finite")
    return _res("G_selectivity", bool(v_m > thr and v_nl > thr), vals, {"min_ratio": thr},
                "both selectivity ratios exceed the preregistered minimum",
                "a selectivity ratio is below the preregistered minimum")


def gate_H_response_geometry(geometry: dict) -> GateResult:
    lo, hi = THRESHOLDS["H_angle_lo"], THRESHOLDS["H_angle_hi"]
    ang = geometry.get("angle_deg")
    vals = dict(geometry)
    if geometry.get("evaluable") is False or ang is None or not np.isfinite(ang):
        return GateResult("H_response_geometry", NOT_EVALUABLE, vals,
                          {"angle_lo": lo, "angle_hi": hi},
                          f"response angle not evaluable ({geometry.get('reason')})")
    return _res("H_response_geometry", bool(lo <= ang <= hi), vals,
                {"angle_lo": lo, "angle_hi": hi},
                "response vectors are close to orthogonal",
                "response vectors are not close to orthogonal")


def gate_I_functional_retention(eta_M_ci, eta_NL_ci, resources_matched: bool) -> GateResult:
    thr = THRESHOLDS["I_retention_min"]
    vals = {"eta_M": getattr(eta_M_ci, "point", None), "eta_M_ci": [getattr(eta_M_ci, "lo", None),
                                                                    getattr(eta_M_ci, "hi", None)],
            "eta_NL0": getattr(eta_NL_ci, "point", None),
            "eta_NL0_ci": [getattr(eta_NL_ci, "lo", None), getattr(eta_NL_ci, "hi", None)],
            "resources_matched": bool(resources_matched)}
    if not resources_matched:
        return GateResult("I_functional_retention", NOT_EVALUABLE, vals, {"min_retention": thr},
                          "baseline resources are not matched: a retention comparison "
                          "between unmatched systems is not interpretable")
    pm, pn = vals["eta_M"], vals["eta_NL0"]
    if pm is None or pn is None or not (np.isfinite(pm) and np.isfinite(pn)):
        return GateResult("I_functional_retention", NOT_EVALUABLE, vals, {"min_retention": thr},
                          "retention ratios not finite")
    return _res("I_functional_retention", bool(pm >= thr and pn >= thr), vals,
                {"min_retention": thr},
                "both modules retain their standalone capability in the combined system",
                "a module loses too much capability in the combined system")


def gate_J_seed_robustness(per_seed_pass: dict) -> GateResult:
    need, n = THRESHOLDS["J_min_seeds_passing"], THRESHOLDS["J_n_seeds"]
    n_pass = sum(1 for v in per_seed_pass.values() if v)
    vals = {"per_seed_pass": {str(k): bool(v) for k, v in per_seed_pass.items()},
            "n_passing": n_pass, "n_seeds": len(per_seed_pass)}
    if len(per_seed_pass) < n:
        return GateResult("J_seed_robustness", NOT_EVALUABLE, vals,
                          {"min_passing": need, "n_seeds": n},
                          f"only {len(per_seed_pass)} reservoir seeds available, {n} required")
    return _res("J_seed_robustness", bool(n_pass >= need), vals,
                {"min_passing": need, "n_seeds": n},
                "decisive gates hold across reservoir seeds",
                "decisive gates do not hold across enough reservoir seeds")


def gate_K_neighbourhood(centre_pass: bool, neighbour_pass: dict) -> GateResult:
    need, n = THRESHOLDS["K_min_neighbours_passing"], THRESHOLDS["K_n_neighbours"]
    n_pass = sum(1 for v in neighbour_pass.values() if v)
    vals = {"centre_pass": bool(centre_pass), "n_neighbours_passing": n_pass,
            "n_neighbours": len(neighbour_pass),
            "neighbour_pass": {str(k): bool(v) for k, v in neighbour_pass.items()}}
    if len(neighbour_pass) < n:
        return GateResult("K_neighbourhood", NOT_EVALUABLE, vals,
                          {"min_neighbours": need, "n_neighbours": n},
                          f"only {len(neighbour_pass)} neighbours evaluated, {n} required")
    return _res("K_neighbourhood", bool(centre_pass and n_pass >= need), vals,
                {"min_neighbours": need, "n_neighbours": n},
                "the operating region, not a single optimised point, passes",
                "the centre or too many neighbours fail: an isolated point is not enough")


def gate_L_back_action(back_action: dict) -> GateResult:
    thr = THRESHOLDS["L_disturbance_max"]
    obs = back_action.get("max_observable_disturbance")
    td = back_action.get("max_trace_distance")
    vals = dict(back_action)
    if obs is None or td is None:
        return GateResult("L_back_action", NOT_EVALUABLE, vals, {"max_disturbance": thr},
                          "back-action was not measured")
    return _res("L_back_action", bool(obs <= thr and td <= thr), vals,
                {"max_disturbance": thr},
                "memory disturbance from the M->P channel is within the preregistered bound",
                "the M->P channel disturbs the memory beyond the preregistered bound")


def summarise(gates) -> dict:
    """Gate table plus the highest claim level the EVIDENCE supports."""
    table = {g.name: g.as_dict() for g in gates}
    status = {g.name: g.status for g in gates}

    def ok(prefix):
        return any(k.startswith(prefix) and v == PASS for k, v in status.items())

    level, why = 0, "implementation or calibration incomplete"
    if ok("A_") and ok("B_"):
        level, why = 1, "structural route isolation holds with falsifiable controls"
    if level >= 1 and ok("C_") and ok("D_") and ok("E_"):
        level, why = 2, "both modules have independent diagonal functional controls"
    if level >= 2 and ok("F_"):
        level, why = 3, "local off-diagonal response is suppressed by equivalence bounds"
    if level >= 3 and ok("J_") and ok("K_"):
        level, why = 4, "separation is robust across seeds and a finite neighbourhood"
    return {"table": table, "status": status, "claim_level": level, "claim_reason": why,
            "n_pass": sum(1 for v in status.values() if v == PASS),
            "n_fail": sum(1 for v in status.values() if v == FAIL),
            "n_not_evaluable": sum(1 for v in status.values() if v == NOT_EVALUABLE)}
