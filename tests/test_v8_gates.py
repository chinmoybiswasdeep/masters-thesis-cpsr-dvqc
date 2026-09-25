import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from decoupled_qrc.v8_gates import evaluate_gates
from decoupled_qrc.v8_metrics import COMBINED_FAMILIES
from decoupled_qrc.v8_protocol import load_protocol
from decoupled_qrc.v8_statistics import bonferroni_tail_alpha


def valid_evidence():
    protocol = copy.deepcopy(load_protocol())
    protocol["statistics"]["bootstrap_resamples"] = 1000
    n = protocol["seed_banks"]["development"]["count"]
    evidence = {
        "stage": "development", "expected_seed_count": n, "observed_seed_count": n,
        "memory_main": np.full((n, 8), 0.25), "nonlinearity_main": np.full((n, 8), 0.25),
        "memory_to_nonlinearity": np.zeros((n, 8)), "nonlinearity_to_memory": np.zeros((n, 8)),
        "degree_profile_differences": np.zeros((n, 3)), "memory_curve_differences": np.zeros((n, 12)),
        "structural_invariance": {"memory_max_abs": 1e-12, "nonlinearity_max_abs": 1e-12},
        "scale_invariant_pass": True,
        "geometry": {"cka_low_high_g": 0.8, "amplitude_control_cka": 1.0},
        "encoder_leakage_max": 0.0, "split_disjoint": True, "no_future_features": True,
        "saturated_fraction": 0.0,
        "negative_controls": {"future": 0.0, "random": 0.0, "permuted": 0.0, "destroyed": 0.0},
        "finite_shot_pass": True, "noise_effect_retention": 0.8, "noise_hh_ordered": True,
        "precision_pass": True, "baseline_complete": True, "quantum_advantage": False,
        "target_names": protocol["required_targets"], "internal_validation_pass": True,
        "confirmation_pass": False,
    }
    for family in protocol["task_families"]:
        evidence["per_delay:" + family] = np.full((n, 12), 0.20)
        if family in COMBINED_FAMILIES:
            evidence["hh_advantage:" + family] = np.full((n, 12), 0.08)
    return protocol, evidence


def test_all_gates_pass_clear_valid_data():
    protocol, evidence = valid_evidence()
    result = evaluate_gates(evidence, protocol)
    assert result["all_passed"], result


def test_rejects_weak_main_and_coupled_response():
    protocol, evidence = valid_evidence()
    evidence["memory_main"][:] = 0.05
    assert not evaluate_gates(evidence, protocol)["memory_main"]["passed"]
    protocol, evidence = valid_evidence()
    evidence["memory_to_nonlinearity"][:] = 0.08
    assert not evaluate_gates(evidence, protocol)["memory_to_nonlinearity_equivalence"]["passed"]


def test_rejects_hh_tail_failure_hidden_by_average():
    protocol, evidence = valid_evidence()
    family = "current_quadratic_x_delayed_linear"
    evidence["hh_advantage:" + family][:, 11] = -0.01
    assert not evaluate_gates(evidence, protocol)["combined_hh"]["passed"]


def test_delay_12_is_required_and_average_cannot_hide_it():
    protocol, evidence = valid_evidence()
    family = protocol["task_families"][0]
    evidence["per_delay:" + family][:, 11] = 0.0
    assert not evaluate_gates(evidence, protocol)["per_delay_requirements"]["passed"]


def test_rejects_saturation_leakage_and_amplitude_only_geometry():
    protocol, evidence = valid_evidence()
    evidence["saturated_fraction"] = 0.5
    evidence["encoder_leakage_max"] = 0.2
    evidence["geometry"]["cka_low_high_g"] = 1.0
    result = evaluate_gates(evidence, protocol)
    assert not result["saturation"]["passed"]
    assert not result["encoder_leakage"]["passed"]
    assert not result["nonlinear_geometry"]["passed"]


def test_rejects_missing_seed_target_and_missing_evidence():
    protocol, evidence = valid_evidence()
    evidence["observed_seed_count"] -= 1
    evidence["target_names"] = evidence["target_names"][:-1]
    result = evaluate_gates(evidence, protocol)
    assert not result["seed_completeness"]["passed"]
    assert not result["target_completeness"]["passed"]
    del evidence["finite_shot_pass"]
    assert not evaluate_gates(evidence, protocol)["all_passed"]


def test_two_sided_multiplicity_correction_is_not_uncorrected():
    assert bonferroni_tail_alpha(0.01, 400) == 0.01 / 800
    assert bonferroni_tail_alpha(0.01, 400) < 0.01 / 2
