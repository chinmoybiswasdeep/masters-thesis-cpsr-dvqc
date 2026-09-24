"""
v4_experiment.py -- grid evaluation and the mandatory gates.

Every threshold here is preregistered and frozen before ARCHITECTURE_SEARCH.
None is ever relaxed after results are seen; if a gate fails the architecture
is redesigned instead.

Gate outcomes are PASS / FAIL / NOT_EVALUABLE. "Not evaluable" is a real third
outcome: an effect whose interval is too wide to decide cannot support a claim
in either direction, and reporting it as FAIL would overstate the evidence as
badly as reporting it as PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .v4_architecture import V4Spec, route_independence, run_v4, verify_product_factorisation
from .v4_ipc import IPCConfig, compute_metrics
from .v4_nonlinear_memory import nm_summary, nonlinear_memory_curves
from .v4_statistics import (CI, holm_equivalence, main_effect, nested_bootstrap,
                            quadrant_contrast, ratio_upper_bound, response_jacobian, tost)

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"

THRESHOLDS = {
    "main_effect_min": 0.10,
    "equivalence_margin": 0.03,
    "cross_ratio_max": 0.20,
    "main_ci_level": 0.95,
    "equivalence_ci_level": 0.90,
    "combined_ci_level": 0.95,
    "saturation_ceiling": 0.995,
    "saturation_max_fraction": 0.20,
    "route_isolation_atol": 1e-12,
    "control_detect_min": 1e-6,
    "encoder_null_slack": 3.0,        # encoder-only N must sit below slack * null
    # freeze-time safety margins (development must beat these, not merely the gates)
    "freeze_main_multiple": 2.0,
    "freeze_cross_fraction": 0.5,
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


def _res(name, ok, values, thr, ok_msg, bad_msg):
    return GateResult(name, PASS if ok else FAIL, values, thr, ok_msg if ok else bad_msg)


# =============================================================================
# Grid evaluation
# =============================================================================
def evaluate_grid(spec: V4Spec, cfg: IPCConfig, *, m_values, g_values,
                  arch_seeds, input_seeds, T: int, with_nm: bool = False,
                  progress=None) -> dict:
    """Full m x g grid over architecture seeds x input seeds.

    The SAME input sequence is used at every (m, g) within an input seed
    (common random numbers), so a difference between grid points is the
    control and not a resampling artifact.
    """
    rows = []
    for a_i, a_seed in enumerate(arch_seeds):
        for i_i, i_seed in enumerate(input_seeds):
            u = np.random.default_rng(int(i_seed)).uniform(-1.0, 1.0, int(T))
            for m in m_values:
                for g in g_values:
                    run = run_v4(spec, u, m=float(m), g=float(g), seed=int(a_seed),
                                 audit_state=True)
                    res = compute_metrics(run, cfg, null_seed=int(i_seed))
                    row = {"arch_seed": int(a_seed), "input_seed": int(i_seed),
                           "m": float(m), "g": float(g),
                           "M": res.M, "N": res.N, "N_long": res.N_long,
                           "H1": res.H1,
                           "saturated_fraction": res.saturation["saturated_fraction"],
                           "saturated": res.saturation["saturated"],
                           "null_threshold": res.null.get("threshold", float("nan")),
                           "degree_profile": res.degree_profile,
                           "delay_profile": res.delay_profile,
                           "cross_delay_mean": (float(np.mean(list(res.cross_delay.values())))
                                                if res.cross_delay else float("nan")),
                           "cross_family_mean": (float(np.mean(list(
                               res.cross_delay_families.values())))
                               if res.cross_delay_families else float("nan")),
                           "unreachable_max": (float(np.max(list(
                               res.unreachable_probe["capacities"].values())))
                               if res.unreachable_probe.get("capacities") else 0.0),
                           "dm_audit_ok": bool(run.dm_audit.get("ok", True))}
                    if with_nm:
                        nm = nonlinear_memory_curves(run, cfg, feature_set="all")
                        row["nm_summary"] = nm_summary(nm)
                        row["nm_curves"] = nm["curves"]
                    rows.append(row)
                    if progress:
                        progress(row)
    return {"rows": rows, "m_values": [float(x) for x in m_values],
            "g_values": [float(x) for x in g_values],
            "arch_seeds": [int(s) for s in arch_seeds],
            "input_seeds": [int(s) for s in input_seeds], "T": int(T)}


def _grid_by_seed(rows, arch_seed, input_seed) -> dict:
    return {(r["m"], r["g"]): r for r in rows
            if r["arch_seed"] == arch_seed and r["input_seed"] == input_seed}


def nested_effects(grid_out: dict, metric: str, control: str) -> dict:
    """{arch_seed: [effect for each input seed]} -- the bootstrap's input."""
    nested = {}
    for a in grid_out["arch_seeds"]:
        vals = []
        for i in grid_out["input_seeds"]:
            sub = _grid_by_seed(grid_out["rows"], a, i)
            if sub:
                vals.append(main_effect(sub, metric, control)["delta"])
        nested[a] = vals
    return nested


def mean_grid(grid_out: dict) -> dict:
    """(m, g) -> metric means across every seed."""
    out = {}
    for m in grid_out["m_values"]:
        for g in grid_out["g_values"]:
            sel = [r for r in grid_out["rows"] if r["m"] == m and r["g"] == g]
            if sel:
                out[(m, g)] = {k: float(np.mean([r[k] for r in sel]))
                               for k in ("M", "N", "N_long", "cross_delay_mean",
                                         "cross_family_mean", "saturated_fraction")}
    return out


# =============================================================================
# Mandatory gates
# =============================================================================
def gate_main_effect(name: str, nested: dict, *, seed: int = 0) -> GateResult:
    thr = THRESHOLDS["main_effect_min"]
    ci = nested_bootstrap(nested, level=THRESHOLDS["main_ci_level"], seed=seed)
    vals = {"delta": ci.point, "ci": [ci.lo, ci.hi], "level": ci.level,
            "n_outer": ci.n_outer, "between_var": ci.between_var,
            "within_var": ci.within_var}
    if not np.isfinite(ci.point) or ci.n_outer == 0:
        return GateResult(name, NOT_EVALUABLE, vals, {"min": thr}, "no usable effect samples")
    ok = bool(ci.point >= thr and np.isfinite(ci.lo) and ci.lo > 0.0)
    return _res(name, ok, vals, {"min": thr, "ci_level": THRESHOLDS["main_ci_level"]},
                "main effect exceeds the threshold with a CI above zero",
                "main effect below threshold, or its lower CI bound does not exclude zero")


def gate_cross_effect(name: str, cross_nested: dict, main_nested: dict, *,
                      seed: int = 0) -> GateResult:
    """Equivalence on the cross effect PLUS an upper bound on the ratio."""
    marg = THRESHOLDS["equivalence_margin"]
    rmax = THRESHOLDS["cross_ratio_max"]
    ci = nested_bootstrap(cross_nested, level=THRESHOLDS["equivalence_ci_level"], seed=seed)
    eq = tost(ci, -marg, +marg)
    ratio = ratio_upper_bound(cross_nested, main_nested,
                              level=THRESHOLDS["main_ci_level"], seed=seed + 1)
    vals = {"cross_delta": ci.point, "ci": [ci.lo, ci.hi], "level": ci.level,
            "equivalent": eq["equivalent"], "ratio": ratio["ratio"],
            "ratio_upper": ratio["upper"], "n_outer": ci.n_outer}
    if not np.isfinite(ci.point) or ci.n_outer == 0:
        return GateResult(name, NOT_EVALUABLE, vals, {"margin": marg, "ratio_max": rmax},
                          "no usable cross-effect samples")
    ratio_ok = bool(np.isfinite(ratio["upper"]) and ratio["upper"] < rmax)
    ok = bool(eq["equivalent"] and ratio_ok)
    return _res(name, ok, vals,
                {"margin": [-marg, marg], "ratio_max": rmax,
                 "ci_level": THRESHOLDS["equivalence_ci_level"]},
                "cross effect equivalent to zero and ratio bounded",
                ("cross effect not equivalent to zero" if not eq["equivalent"]
                 else "ratio upper bound not below its limit"))


def gate_degree_profile(grid_out: dict, *, control: str, seed: int = 0) -> GateResult:
    """During the m sweep the local degree-2/3/4 capacities must not move;
    during the g sweep the linear-memory curve must not move. Holm-corrected."""
    marg = THRESHOLDS["equivalence_margin"]
    rows = grid_out["rows"]
    keys = (sorted(rows[0]["degree_profile"]) if control == "m"
            else sorted(rows[0]["delay_profile"]))
    field_name = "degree_profile" if control == "m" else "delay_profile"
    cis = {}
    for k in keys:
        nested = {}
        for a in grid_out["arch_seeds"]:
            vals = []
            for i in grid_out["input_seeds"]:
                sub = _grid_by_seed(rows, a, i)
                flat = {kk: {"v": vv[field_name][k]} for kk, vv in sub.items()}
                vals.append(main_effect(flat, "v", control)["delta"])
            nested[a] = vals
        cis[str(k)] = nested_bootstrap(nested, level=THRESHOLDS["equivalence_ci_level"],
                                       seed=seed)
    fam = holm_equivalence(cis, -marg, +marg)
    vals = {"members": {k: {"delta": v["point"], "ci": [v["lo"], v["hi"]]}
                        for k, v in fam["members"].items()},
            "n_tests": fam["n_tests"], "control": control, "field": field_name}
    return _res(f"degree_profile_equivalence_{control}", fam["family_equivalent"], vals,
                {"margin": [-marg, marg]},
                "every member of the family is equivalent to zero after correction",
                "at least one member of the family moved beyond the equivalence margin")


def gate_combined(grid_out: dict, *, seed: int = 0) -> GateResult:
    """High-m/high-g must beat every other quadrant, with a CI above zero."""
    metrics = ("N_long", "cross_delay_mean", "cross_family_mean")
    nested = {mt: {} for mt in metrics}
    point = {}
    for mt in metrics:
        for a in grid_out["arch_seeds"]:
            vals = []
            for i in grid_out["input_seeds"]:
                sub = _grid_by_seed(grid_out["rows"], a, i)
                if sub:
                    vals.append(quadrant_contrast(sub, mt)["improvement"])
            nested[mt][a] = vals
        point[mt] = quadrant_contrast(mean_grid(grid_out), mt)
    cis = {mt: nested_bootstrap(nested[mt], level=THRESHOLDS["combined_ci_level"],
                                seed=seed) for mt in metrics}
    vals = {mt: {"improvement": cis[mt].point, "ci": [cis[mt].lo, cis[mt].hi],
                 "quadrant_means": point[mt]["quadrant_means"]} for mt in metrics}
    usable = [mt for mt in metrics if np.isfinite(cis[mt].point)]
    if not usable:
        return GateResult("combined_capability", NOT_EVALUABLE, vals, {},
                          "no usable quadrant contrasts")
    ok = all(np.isfinite(cis[mt].lo) and cis[mt].lo > 0.0 for mt in usable)
    return _res("combined_capability", ok, vals,
                {"ci_level": THRESHOLDS["combined_ci_level"], "requires": "lower bound > 0"},
                "high-m/high-g beats every other quadrant with a CI above zero",
                "the high-m/high-g advantage is not established above zero")


def gate_controls(spec: V4Spec, cfg: IPCConfig, *, u_probe, arch_seed: int,
                  grid_out: dict, encoder_only_N: float, null_threshold: float) -> GateResult:
    """Route isolation, factorisation, negative controls, saturation, resources."""
    iso = route_independence(spec, u_probe, m=0.6, g=0.5, seed=arch_seed,
                             atol=THRESHOLDS["route_isolation_atol"])
    fac = verify_product_factorisation(spec, u_probe[:12], m=0.6, g=0.5, seed=arch_seed)
    cont = route_independence(V4Spec(architecture="contaminated", memory=spec.memory,
                                     processor=spec.processor, n_joint=spec.n_joint,
                                     contamination=0.4),
                              u_probe, m=0.6, g=0.5, seed=arch_seed)
    ser = route_independence(V4Spec(architecture="serial", memory=spec.memory,
                                    processor=spec.processor, n_joint=spec.n_joint),
                             u_probe, m=0.6, g=0.5, seed=arch_seed)
    worst_sat = max(r["saturated_fraction"] for r in grid_out["rows"])
    unreachable_max = max(r["unreachable_max"] for r in grid_out["rows"])
    dm_ok = all(r["dm_audit_ok"] for r in grid_out["rows"])

    detect = THRESHOLDS["control_detect_min"]
    vals = {"max_dXR_dg": iso["max_dXR_dg"], "max_dXP_dm": iso["max_dXP_dm"],
            "factorisation_dev": fac["max_abs_deviation"],
            "contaminated_detected": bool(cont["max_dXP_dm"] > detect),
            "contaminated_max_dXP_dm": cont["max_dXP_dm"],
            "serial_detected": bool(ser["max_dXP_dm"] > detect),
            "serial_max_dXP_dm": ser["max_dXP_dm"],
            "worst_saturated_fraction": worst_sat,
            "encoder_only_N": float(encoder_only_N),
            "null_threshold": float(null_threshold),
            "unreachable_max_capacity": unreachable_max,
            "density_matrix_ok": dm_ok}
    enc_ok = bool(encoder_only_N <= THRESHOLDS["encoder_null_slack"] * max(null_threshold, 1e-6))
    ok = bool(iso["passed"] and fac["factorises"]
              and vals["contaminated_detected"] and vals["serial_detected"]
              and worst_sat <= THRESHOLDS["saturation_max_fraction"]
              and enc_ok and dm_ok)
    reasons = []
    if not iso["passed"]:
        reasons.append("route isolation violated")
    if not fac["factorises"]:
        reasons.append("product factorisation failed")
    if not vals["contaminated_detected"]:
        reasons.append("contaminated control NOT detected -- the test has no power")
    if not vals["serial_detected"]:
        reasons.append("serial control NOT detected -- the test has no power")
    if worst_sat > THRESHOLDS["saturation_max_fraction"]:
        reasons.append(f"saturation {worst_sat:.3f} above limit")
    if not enc_ok:
        reasons.append("encoder-only nonlinear capacity above the null floor")
    if not dm_ok:
        reasons.append("an invalid density matrix was produced")
    return _res("controls", ok, vals,
                {"atol": THRESHOLDS["route_isolation_atol"], "detect_min": detect,
                 "saturation_max": THRESHOLDS["saturation_max_fraction"]},
                "isolation exact, negative controls detected, no saturation or leakage",
                "; ".join(reasons))


def evaluate_all_gates(grid_out: dict, spec: V4Spec, cfg: IPCConfig, *,
                       u_probe, encoder_only_N: float, seed: int = 0) -> dict:
    """The full mandatory gate table plus the response Jacobian."""
    n_mM = nested_effects(grid_out, "M", "m")
    n_gN = nested_effects(grid_out, "N", "g")
    n_mN = nested_effects(grid_out, "N", "m")
    n_gM = nested_effects(grid_out, "M", "g")

    gates = [
        gate_main_effect("memory_main_effect", n_mM, seed=seed),
        gate_main_effect("nonlinear_main_effect", n_gN, seed=seed + 1),
        gate_cross_effect("cross_m_to_N", n_mN, n_mM, seed=seed + 2),
        gate_cross_effect("cross_g_to_M", n_gM, n_gN, seed=seed + 3),
        gate_degree_profile(grid_out, control="m", seed=seed + 4),
        gate_degree_profile(grid_out, control="g", seed=seed + 5),
        gate_combined(grid_out, seed=seed + 6),
        gate_controls(spec, cfg, u_probe=u_probe,
                      arch_seed=grid_out["arch_seeds"][0], grid_out=grid_out,
                      encoder_only_N=encoder_only_N,
                      null_threshold=float(np.median([r["null_threshold"]
                                                      for r in grid_out["rows"]]))),
    ]
    jac = response_jacobian(
        dm_M=nested_bootstrap(n_mM, seed=seed).point,
        dm_N=nested_bootstrap(n_mN, seed=seed).point,
        dg_M=nested_bootstrap(n_gM, seed=seed).point,
        dg_N=nested_bootstrap(n_gN, seed=seed).point)
    table = {g.name: g.as_dict() for g in gates}
    return {"gates": table, "jacobian": jac,
            "all_passed": all(g.passed for g in gates),
            "n_pass": sum(1 for g in gates if g.status == PASS),
            "n_fail": sum(1 for g in gates if g.status == FAIL),
            "n_not_evaluable": sum(1 for g in gates if g.status == NOT_EVALUABLE),
            "thresholds": dict(THRESHOLDS)}


def freeze_margin_check(gate_table: dict) -> dict:
    """Development must beat the gates by a MARGIN before anything is frozen.

    Main effects at >= 2x the threshold and cross effects at <= half the
    equivalence margin. A candidate that merely scrapes past the gates on
    development data will not survive an untouched confirmation.
    """
    need_main = THRESHOLDS["main_effect_min"] * THRESHOLDS["freeze_main_multiple"]
    need_cross = THRESHOLDS["equivalence_margin"] * THRESHOLDS["freeze_cross_fraction"]
    checks, ok = {}, True
    for name in ("memory_main_effect", "nonlinear_main_effect"):
        v = gate_table["gates"].get(name, {}).get("values", {}).get("delta", float("nan"))
        good = bool(np.isfinite(v) and abs(v) >= need_main)
        checks[name] = {"value": v, "required": need_main, "ok": good}
        ok &= good
    for name in ("cross_m_to_N", "cross_g_to_M"):
        vv = gate_table["gates"].get(name, {}).get("values", {})
        ci = vv.get("ci", [float("nan"), float("nan")])
        worst = max(abs(ci[0]), abs(ci[1])) if all(np.isfinite(c) for c in ci) else float("nan")
        good = bool(np.isfinite(worst) and worst <= need_cross)
        checks[name] = {"worst_ci_bound": worst, "required_max": need_cross, "ok": good}
        ok &= good
    return {"checks": checks, "margin_ok": bool(ok),
            "required_main": need_main, "required_cross": need_cross}
