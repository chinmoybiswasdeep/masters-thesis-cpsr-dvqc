"""V3.2 readout: the shot model is validated, not assumed; resources are counted."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_readout import (  # noqa: E402
    ShotBudget, account_resources, add_shot_noise, group_settings, parse_label,
    resources_match, sample_shot_noise_exact, shot_std, verify_shot_noise_model)


def test_parse_label_round_trips_single_and_two_body():
    assert parse_label("X0") == {0: "X"}
    assert parse_label("Z1X10") == {1: "Z", 10: "X"}
    with pytest.raises(ValueError):
        parse_label("Q0")


def test_local_pauli_readout_needs_three_settings():
    """X_i, Y_i, Z_i do not commute: one shot budget cannot estimate them all."""
    labels = [f"{p}{q}" for q in range(5) for p in ("X", "Y", "Z")]
    settings = group_settings(labels)
    assert len(settings) == 3
    assert sum(len(s["labels"]) for s in settings) == len(labels)


def test_settings_never_assign_two_bases_to_one_qubit():
    labels = [f"{p1}{q1}{p2}{q2}" for q1, q2 in [(0, 1), (1, 2)]
              for p1 in "XYZ" for p2 in "XYZ"]
    for s in group_settings(labels):
        for lab in s["labels"]:
            for q, p in parse_label(lab).items():
                assert s["bases"][q] == p


def test_shot_std_matches_the_binomial_variance():
    for p in (-0.8, 0.0, 0.5, 0.99):
        assert abs(shot_std(np.array([p]), 1000)[0] - np.sqrt((1 - p ** 2) / 1000)) < 1e-12


def test_gaussian_shot_model_agrees_with_true_binomial_sampling():
    """This is what makes the analytic shot model an approximation that has
    been CHECKED rather than assumed."""
    rows = verify_shot_noise_model(shots=2000, n_rep=20000, seed=5)
    for p, r in rows.items():
        assert r["mean_abs_diff"] < 5e-3, (p, r)
        assert r["var_rel_diff"] < 0.12, (p, r)


def test_noiseless_budget_is_a_passthrough():
    F = np.random.default_rng(0).uniform(-1, 1, (10, 4))
    assert np.allclose(add_shot_noise(F, ShotBudget(None), 1), F)
    assert np.allclose(sample_shot_noise_exact(F, ShotBudget(None), 1), F)


def test_shot_noise_is_reproducible_and_seed_dependent():
    F = np.random.default_rng(0).uniform(-1, 1, (20, 5))
    a = add_shot_noise(F, ShotBudget(5000), 11)
    b = add_shot_noise(F, ShotBudget(5000), 11)
    c = add_shot_noise(F, ShotBudget(5000), 12)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_more_shots_means_less_noise():
    F = np.zeros((4000, 1))
    lo = add_shot_noise(F, ShotBudget(100), 3).std()
    hi = add_shot_noise(F, ShotBudget(10000), 3).std()
    assert hi < lo


def test_resource_row_counts_settings_not_just_shots():
    labels_P = [f"{p}{q}" for q in range(5) for p in ("X", "Y", "Z")]
    row = account_resources(name="processor_only", memory_qubits=0, processor_qubits=5,
                            input_copies=5, ancillas=0, labels_M=[], labels_P=labels_P,
                            budget=ShotBudget(10000))
    assert row["measurement_settings"] == 3
    assert row["shots_per_timestep"] == 30000        # 3 settings x 10000, not 10000
    assert row["readout_parameters"] == len(labels_P) + 1


def test_resources_match_detects_a_mismatch():
    labels = [f"{p}0" for p in "XYZ"]
    a = account_resources(name="a", memory_qubits=0, processor_qubits=1, input_copies=1,
                          ancillas=0, labels_M=[], labels_P=labels, budget=ShotBudget(1000))
    b = account_resources(name="b", memory_qubits=0, processor_qubits=1, input_copies=4,
                          ancillas=0, labels_M=[], labels_P=labels, budget=ShotBudget(1000))
    rep = resources_match([a, b])
    assert not rep["all_matched"]
    assert not rep["input_copies"]["matched"]
    assert resources_match([a, a])["all_matched"]
