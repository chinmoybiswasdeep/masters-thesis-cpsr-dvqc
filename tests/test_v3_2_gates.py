"""V3.2 acceptance gates: each rejection corresponds to a V3.1 failure mode."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_2_gates import (  # noqa: E402
    FAIL, NOT_EVALUABLE, PASS, THRESHOLDS, gate_A_structural_isolation,
    gate_B_numerical_validity, gate_C_ipc_validity, gate_D_memory_controllability,
    gate_E_processor_controllability, gate_F_cross_suppression, gate_G_selectivity,
    gate_H_response_geometry, gate_I_functional_retention, gate_J_seed_robustness,
    gate_K_neighbourhood, gate_L_back_action, summarise)
from decoupled_qrc.v3_2_response import assess_stability  # noqa: E402


def _ceiling(raw, n_targets=4):
    """Build the fixture from the REAL audit so a key rename cannot make the
    guard unreachable while the test still passes. The previous fixture
    hand-typed {"frac": ...}, a key `ceiling_audit` does not emit."""
    from decoupled_qrc.ceiling_audit import audit_ceiling
    a = audit_ceiling("m", raw=raw, n_targets=n_targets, numerical_rank=15,
                      effective_rank=1.5, n_train=644)
    return a.as_dict() if hasattr(a, "as_dict") else vars(a)


def _ipc(**kw):
    base = dict(null_fraction_NL0=0.02, null_fraction_M=0.02,
                ceiling_NL0=_ceiling(1.6), ceiling_M=_ceiling(1.6),
                degrees_above_structural_bound=[], was_capped={},
                n_train=800, sample_size_required=500, sample_size_ok=True,
                effective_rank=30.0, numerical_rank=40)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _ci(point, lo, hi):
    return types.SimpleNamespace(point=point, lo=lo, hi=hi)


# --------------------------------------------------------------------------
# Gate A
# --------------------------------------------------------------------------
def test_gate_A_requires_the_negative_control_to_detect():
    iso = {"passed": True, "max_dXP_dm": 0.0, "max_dXM_dg": 0.0, "max_dXM_dJ": 0.0}
    good = gate_A_structural_isolation(iso, {"passed": False, "max_dXP_dm": 1e-2})
    assert good.status == PASS
    # a control that does NOT detect its own planted dependency proves nothing
    blind = gate_A_structural_isolation(iso, {"passed": True, "max_dXP_dm": 0.0})
    assert blind.status == FAIL


def test_gate_A_fails_when_isolation_is_violated():
    iso = {"passed": False, "max_dXP_dm": 1e-3, "max_dXM_dg": 0.0, "max_dXM_dJ": 0.0}
    assert gate_A_structural_isolation(iso, {"passed": False, "max_dXP_dm": 1e-2}).status == FAIL


# --------------------------------------------------------------------------
# Gate B / C -- the V3.1 statistical failures
# --------------------------------------------------------------------------
def test_gate_B_rejects_insufficient_samples():
    """V3.1 ran this analysis on n_train = 43."""
    bad = _ipc(n_train=43, sample_size_ok=False)
    dm = {"trace_dev": 0.0, "hermiticity_dev": 0.0, "min_eigenvalue": 0.0}
    assert gate_B_numerical_validity(tests_passed=True, ipc_report=bad, dm_audit=dm).status == FAIL


def test_gate_B_rejects_a_bad_density_matrix():
    dm = {"trace_dev": 1e-3, "hermiticity_dev": 0.0, "min_eigenvalue": 0.0}
    assert gate_B_numerical_validity(tests_passed=True, ipc_report=_ipc(), dm_audit=dm).status == FAIL
    dm_neg = {"trace_dev": 0.0, "hermiticity_dev": 0.0, "min_eigenvalue": -1e-3}
    assert gate_B_numerical_validity(tests_passed=True, ipc_report=_ipc(),
                                     dm_audit=dm_neg).status == FAIL


def test_gate_B_rejects_a_cache_mismatch():
    dm = {"trace_dev": 0.0, "hermiticity_dev": 0.0, "min_eigenvalue": 0.0}
    assert gate_B_numerical_validity(tests_passed=True, ipc_report=_ipc(), dm_audit=dm,
                                     cache_equivalence=1e-6).status == FAIL


def test_gate_C_rejects_an_excessive_null_fraction():
    """V3.1 reported null fractions of 0.495 and 0.511."""
    assert gate_C_ipc_validity(_ipc(null_fraction_NL0=0.495)).status == FAIL


def test_gate_C_rejects_ceiling_contamination():
    """Against a REAL contaminated audit (fraction_of_ceiling 0.975, flagged)."""
    contaminated = _ceiling(3.9)
    assert contaminated["fraction_of_ceiling"] > 0.95 and contaminated["ceiling_contaminated"]
    assert gate_C_ipc_validity(_ipc(ceiling_NL0=contaminated)).status == FAIL


def test_gate_C_refuses_a_missing_ceiling_fraction():
    """A key the audit does not emit returns None; None must NOT pass."""
    assert gate_C_ipc_validity(_ipc(ceiling_NL0={"frac": 0.5})).status == FAIL


def test_gate_C_rejects_an_undecayed_tail():
    assert gate_C_ipc_validity(_ipc(), tail_decayed=False).status == FAIL


def test_gate_C_is_not_evaluable_when_capacity_is_zero():
    g = gate_C_ipc_validity(_ipc(null_fraction_NL0=float("nan"), null_fraction_M=float("nan")))
    assert g.status == NOT_EVALUABLE


# --------------------------------------------------------------------------
# Gate D / E -- diagonal controllability
# --------------------------------------------------------------------------
def _stable_verdict(value=2.0):
    nested = {k: [value + 0.02 * k, value - 0.01 * k] for k in range(3)}
    return assess_stability("dM_dm", {"a": value, "b": value * 1.01, "c": value * 0.99},
                            nested, n_boot=400)


def test_gate_D_passes_a_clean_memory_response():
    assert gate_D_memory_controllability(3.0, _stable_verdict()).status == PASS


def test_gate_D_rejects_trivial_memory():
    assert gate_D_memory_controllability(0.2, _stable_verdict()).status == FAIL


def test_gate_D_rejects_an_unstable_derivative():
    unstable = assess_stability("dM_dm", {"a": 1.0, "b": -1.2, "c": 0.8},
                                {0: [1.0, -1.0], 1: [-0.9, 1.1]}, n_boot=400)
    assert gate_D_memory_controllability(3.0, unstable).status == FAIL


def test_gate_D_is_not_evaluable_without_a_bootstrap():
    v = assess_stability("dM_dm", {"a": 2.0, "b": 2.01})
    assert gate_D_memory_controllability(3.0, v).status == NOT_EVALUABLE


def test_gate_E_rejects_a_flat_dynamic_range():
    """V3.1's processor sat at E_NL = 0.0739 against a 0.20 gate."""
    flat = [2.0, 2.001, 2.002, 1.999, 2.0]
    g = gate_E_processor_controllability(NL_0=2.0, nl0_values_over_grid=flat,
                                         delta_interaction_ci=_ci(1.5, 1.0, 2.0), f_enc=0.0)
    assert g.status == FAIL
    assert g.values["E_NL"] < THRESHOLDS["E_dynamic_range_min"]


def test_gate_E_rejects_an_encoder_dominated_processor():
    """V3.1: NL_0(encoding only) = 0.7201 vs NL_0(full) = 0.6632."""
    grid = [0.0, 1.0, 2.0, 2.5, 3.0]
    g = gate_E_processor_controllability(NL_0=2.0, nl0_values_over_grid=grid,
                                         delta_interaction_ci=_ci(1.5, 1.0, 2.0),
                                         f_enc=1.09)
    assert g.status == FAIL


def test_gate_E_rejects_a_delta_interaction_ci_containing_zero():
    grid = [0.0, 1.0, 2.0, 2.5, 3.0]
    g = gate_E_processor_controllability(NL_0=2.0, nl0_values_over_grid=grid,
                                         delta_interaction_ci=_ci(0.1, -0.4, 0.6), f_enc=0.0)
    assert g.status == FAIL


def test_gate_E_rejects_trivial_capacity():
    grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    g = gate_E_processor_controllability(NL_0=0.1, nl0_values_over_grid=grid,
                                         delta_interaction_ci=_ci(0.05, 0.01, 0.1), f_enc=0.0)
    assert g.status == FAIL


def test_gate_E_passes_a_healthy_processor():
    grid = [0.0, 1.0, 1.8, 2.4, 2.7, 2.3]
    g = gate_E_processor_controllability(NL_0=2.4, nl0_values_over_grid=grid,
                                         delta_interaction_ci=_ci(2.4, 1.9, 2.8), f_enc=0.0)
    assert g.status == PASS


# --------------------------------------------------------------------------
# Gate F -- equivalence, not non-significance
# --------------------------------------------------------------------------
def test_gate_F_is_not_evaluable_when_every_off_diagonal_is_structural():
    """A zero by construction is not evidence of suppression."""
    bounds = [{"name": "dNL_dm", "structural": True, "upper_abs_bound": 0.0, "suppressed": True}]
    assert gate_F_cross_suppression(bounds).status == NOT_EVALUABLE


def test_gate_F_fails_on_an_unsuppressed_empirical_off_diagonal():
    bounds = [{"name": "dNL_dm", "upper_abs_bound": 0.9, "suppressed": False}]
    assert gate_F_cross_suppression(bounds).status == FAIL


def test_gate_F_is_not_evaluable_without_a_usable_bound():
    bounds = [{"name": "dNL_dm", "upper_abs_bound": float("nan"), "suppressed": False}]
    assert gate_F_cross_suppression(bounds).status == NOT_EVALUABLE


def test_gate_F_passes_when_bounded_below_the_margin():
    bounds = [{"name": "dNL_dm", "upper_abs_bound": 0.05, "suppressed": True}]
    assert gate_F_cross_suppression(bounds).status == PASS


# --------------------------------------------------------------------------
# Gates G, H, I, J, K, L
# --------------------------------------------------------------------------
def test_gate_G_is_not_evaluable_from_unstable_components():
    rep = {"R_M": {"kind": "not_evaluable"}, "R_NL0": {"kind": "ratio", "ratio": 5.0}}
    assert gate_G_selectivity(rep).status == NOT_EVALUABLE


def test_gate_G_uses_the_lower_bound_when_the_denominator_is_unresolved():
    rep = {"R_M": {"kind": "lower_bound", "ratio": float("nan"), "ratio_lower_bound": 400.0},
           "R_NL0": {"kind": "ratio", "ratio": 5.0}}
    assert gate_G_selectivity(rep).status == PASS


def test_gate_G_fails_a_low_ratio():
    rep = {"R_M": {"kind": "ratio", "ratio": 1.2}, "R_NL0": {"kind": "ratio", "ratio": 5.0}}
    assert gate_G_selectivity(rep).status == FAIL


def test_gate_H_accepts_near_orthogonal_and_rejects_aligned():
    assert gate_H_response_geometry({"angle_deg": 88.0, "evaluable": True}).status == PASS
    assert gate_H_response_geometry({"angle_deg": 20.0, "evaluable": True}).status == FAIL
    assert gate_H_response_geometry({"angle_deg": float("nan"), "evaluable": False,
                                     "reason": "degenerate"}).status == NOT_EVALUABLE


def test_gate_I_is_not_evaluable_on_unmatched_resources():
    g = gate_I_functional_retention(_ci(0.9, 0.8, 1.0), _ci(0.9, 0.8, 1.0),
                                    resources_matched=False)
    assert g.status == NOT_EVALUABLE


def test_gate_I_passes_and_fails_on_retention():
    assert gate_I_functional_retention(_ci(0.9, 0.8, 1.0), _ci(0.85, 0.7, 0.95),
                                       resources_matched=True).status == PASS
    assert gate_I_functional_retention(_ci(0.9, 0.8, 1.0), _ci(0.4, 0.3, 0.5),
                                       resources_matched=True).status == FAIL


def test_gate_J_requires_enough_seeds():
    assert gate_J_seed_robustness({0: True, 1: True}).status == NOT_EVALUABLE
    assert gate_J_seed_robustness({i: True for i in range(5)}).status == PASS
    mostly = {0: True, 1: True, 2: True, 3: False, 4: False}
    assert gate_J_seed_robustness(mostly).status == FAIL


def test_gate_K_requires_the_centre_and_a_majority_of_neighbours():
    nb = {i: True for i in range(6)}
    assert gate_K_neighbourhood(True, nb).status == PASS
    assert gate_K_neighbourhood(False, nb).status == FAIL
    assert gate_K_neighbourhood(True, {i: i < 2 for i in range(6)}).status == FAIL
    assert gate_K_neighbourhood(True, {0: True}).status == NOT_EVALUABLE


def test_gate_L_bounds_back_action():
    assert gate_L_back_action({"max_observable_disturbance": 0.01,
                               "max_trace_distance": 0.02}).status == PASS
    assert gate_L_back_action({"max_observable_disturbance": 0.4,
                               "max_trace_distance": 0.3}).status == FAIL
    assert gate_L_back_action({}).status == NOT_EVALUABLE


# --------------------------------------------------------------------------
# Claim ladder
# --------------------------------------------------------------------------
def test_claim_ladder_stops_at_level_one_without_diagonal_control():
    iso = {"passed": True, "max_dXP_dm": 0.0, "max_dXM_dg": 0.0, "max_dXM_dJ": 0.0}
    dm = {"trace_dev": 0.0, "hermiticity_dev": 0.0, "min_eigenvalue": 0.0}
    gates = [gate_A_structural_isolation(iso, {"passed": False, "max_dXP_dm": 1e-2}),
             gate_B_numerical_validity(tests_passed=True, ipc_report=_ipc(), dm_audit=dm),
             gate_C_ipc_validity(_ipc()),
             gate_D_memory_controllability(0.2, _stable_verdict()),          # trivial memory
             gate_E_processor_controllability(NL_0=2.4, nl0_values_over_grid=[2.0] * 5,
                                              delta_interaction_ci=_ci(1.0, 0.5, 1.5),
                                              f_enc=0.0)]
    s = summarise(gates)
    assert s["claim_level"] == 1
    assert "structural" in s["claim_reason"]


def test_claim_ladder_reaches_level_two_with_diagonal_control():
    iso = {"passed": True, "max_dXP_dm": 0.0, "max_dXM_dg": 0.0, "max_dXM_dJ": 0.0}
    dm = {"trace_dev": 0.0, "hermiticity_dev": 0.0, "min_eigenvalue": 0.0}
    gates = [gate_A_structural_isolation(iso, {"passed": False, "max_dXP_dm": 1e-2}),
             gate_B_numerical_validity(tests_passed=True, ipc_report=_ipc(), dm_audit=dm),
             gate_C_ipc_validity(_ipc()),
             gate_D_memory_controllability(3.0, _stable_verdict()),
             gate_E_processor_controllability(NL_0=2.4,
                                              nl0_values_over_grid=[0.0, 1.0, 1.8, 2.4, 2.7, 2.3],
                                              delta_interaction_ci=_ci(2.4, 1.9, 2.8), f_enc=0.0)]
    assert summarise(gates)["claim_level"] == 2


def test_empty_gate_set_is_level_zero():
    assert summarise([])["claim_level"] == 0
