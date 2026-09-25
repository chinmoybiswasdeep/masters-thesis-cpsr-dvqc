"""Complete executable V8 gate engine; missing evidence always fails."""

from __future__ import annotations

import numpy as np

from .v8_metrics import COMBINED_FAMILIES
from .v8_statistics import paired_interval, simultaneous_intervals


EFFECTIVE_MANDATORY_COMPARISONS = 5000


def _all_inside(intervals, margin):
    return all(item["lower"] > -margin and item["upper"] < margin for item in intervals)


def _all_lower(intervals, minimum):
    return all(item["lower"] > minimum for item in intervals)


def _gate(passed, **details):
    return {"passed": bool(passed), **details}


def evaluate_gates(evidence: dict, protocol: dict) -> dict:
    """Evaluate every registered gate from seed-level persisted evidence."""
    t = protocol["thresholds"]
    stats = protocol["statistics"]
    alpha, resamples = stats["familywise_alpha"], stats["bootstrap_resamples"]
    expected_seeds = evidence.get("expected_seed_count")
    observed_seeds = evidence.get("observed_seed_count")
    gates = {}

    def intervals(name, width=None):
        values = np.asarray(evidence[name], dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if width is not None and values.shape[1] != width:
            raise ValueError(f"{name} must have {width} components")
        return simultaneous_intervals(
            values,
            alpha=alpha,
            comparisons=max(stats["mandatory_comparisons"], EFFECTIVE_MANDATORY_COMPARISONS),
            resamples=resamples,
        )

    try:
        memory_main = intervals("memory_main")
        nonlinear_main = intervals("nonlinearity_main")
        gates["memory_main"] = _gate(
            _all_lower(memory_main, t["main_effect_lower_bound_min"])
            and all(x["estimate"] >= t["main_effect_min"] for x in memory_main), intervals=memory_main
        )
        gates["nonlinearity_main"] = _gate(
            _all_lower(nonlinear_main, t["main_effect_lower_bound_min"])
            and all(x["estimate"] >= t["main_effect_min"] for x in nonlinear_main), intervals=nonlinear_main
        )
        cross_mn, cross_nm = intervals("memory_to_nonlinearity"), intervals("nonlinearity_to_memory")
        gates["memory_to_nonlinearity_equivalence"] = _gate(_all_inside(cross_mn, t["cross_margin"]), intervals=cross_mn)
        gates["nonlinearity_to_memory_equivalence"] = _gate(_all_inside(cross_nm, t["cross_margin"]), intervals=cross_nm)
        degree, memory_curve = intervals("degree_profile_differences", 3), intervals("memory_curve_differences", 12)
        gates["degree_profile_preservation"] = _gate(_all_inside(degree, t["profile_margin"]), intervals=degree)
        gates["memory_curve_preservation"] = _gate(_all_inside(memory_curve, t["profile_margin"]), intervals=memory_curve)

        structural = evidence["structural_invariance"]
        gates["feature_structural_invariance"] = _gate(
            structural["memory_max_abs"] <= t["feature_invariance_tolerance"]
            and structural["nonlinearity_max_abs"] <= t["feature_invariance_tolerance"], **structural
        )
        gates["scale_invariant_separation"] = _gate(bool(evidence["scale_invariant_pass"]))
        geometry = evidence["geometry"]
        gates["nonlinear_geometry"] = _gate(
            geometry["cka_low_high_g"] <= t["nonlinear_geometry_cka_max"]
            and abs(geometry["amplitude_control_cka"] - 1.0) <= t["feature_invariance_tolerance"], **geometry
        )

        per_delay_ok, hh_ok = True, True
        per_delay_details, hh_details = {}, {}
        readout_count = len(protocol["readouts"])
        for family in protocol["task_families"]:
            caps = intervals("per_delay:" + family, 12 * readout_count)
            estimates = [x["estimate"] for x in caps]
            delay_passes = [x >= t["per_delay_capacity_min"] for x in estimates]
            tails = [12 * readout + delay for readout in range(readout_count) for delay in range(8, 12)]
            family_ok = (
                sum(delay_passes) / len(delay_passes) >= t["delay_pass_fraction_min"]
                and all(estimates[index] >= t["tail_capacity_min"] for index in tails)
            )
            per_delay_ok &= family_ok
            per_delay_details[family] = {"intervals": caps, "passed": family_ok}
            if family in COMBINED_FAMILIES:
                advantages = intervals("hh_advantage:" + family, 12 * readout_count)
                family_hh = all(
                    advantages[index]["estimate"] >= t["hh_advantage_min"]
                    and advantages[index]["lower"] > 0.0
                    for index in (12 * readout + delay for readout in range(readout_count) for delay in range(8, 12))
                )
                hh_ok &= family_hh
                hh_details[family] = {"intervals": advantages, "passed": family_hh}
        gates["per_delay_requirements"] = _gate(per_delay_ok, families=per_delay_details)
        gates["combined_hh"] = _gate(hh_ok, families=hh_details)

        heldout = intervals("heldout_capacities", 5 * readout_count)
        gates["heldout_targets"] = _gate(
            all(item["estimate"] >= t["per_delay_capacity_min"] and item["lower"] > 0.0 for item in heldout),
            intervals=heldout,
        )

        gates["encoder_leakage"] = _gate(evidence["encoder_leakage_max"] <= t["null_capacity_upper"])
        gates["train_test_leakage"] = _gate(bool(evidence["split_disjoint"] and evidence["no_future_features"]))
        gates["saturation"] = _gate(evidence["saturated_fraction"] <= t["saturated_fraction_max"])
        surface = evidence["response_surface"]
        gates["response_surface"] = _gate(
            surface["complete_points"] == 25
            and surface["memory_ordered"] and surface["nonlinear_ordered"]
            and surface["memory_interior"] and surface["nonlinear_interior"],
            **surface,
        )
        gates["conditioning"] = _gate(bool(evidence["conditioning_pass"]))
        gates["learning_curve"] = _gate(bool(evidence["learning_curve_pass"]))
        nulls = evidence["negative_controls"]
        gates["negative_controls"] = _gate(all(value <= t["null_capacity_upper"] for value in nulls.values()), values=nulls)
        gates["finite_shot_robustness"] = _gate(bool(evidence["finite_shot_pass"]))
        gates["fake_backend_robustness"] = _gate(evidence["noise_effect_retention"] >= t["noise_effect_retention_min"] and evidence["noise_hh_ordered"])
        gates["precision_robustness"] = _gate(bool(evidence["precision_pass"]))
        gates["classical_baselines"] = _gate(bool(evidence["baseline_complete"]), quantum_advantage=bool(evidence["quantum_advantage"]))
        gates["seed_completeness"] = _gate(expected_seeds == observed_seeds, expected=expected_seeds, observed=observed_seeds)
        gates["target_completeness"] = _gate(set(evidence["target_names"]) == set(protocol["required_targets"]))
        gates["internal_validation"] = _gate(bool(evidence["internal_validation_pass"]))
        gates["confirmation"] = _gate(bool(evidence["confirmation_pass"]))
    except (KeyError, TypeError, ValueError) as error:
        gates["evidence_complete"] = _gate(False, error=str(error))

    gates["all_passed"] = all(item["passed"] for name, item in gates.items() if name != "confirmation")
    if evidence.get("stage") == "confirmation":
        gates["all_passed"] &= gates.get("confirmation", {"passed": False})["passed"]
    return gates
