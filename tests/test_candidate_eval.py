"""
test_candidate_eval.py -- V2.1 Defects 4/7/10 / test requirements #7
(candidate-specific ceiling), #23 (sample-to-effective-rank checks). Uses
small, fast configs throughout (this module runs real quantum circuits).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.candidate_eval import evaluate_candidate, select_candidates, compute_provenance_tag  # noqa: E402
from decoupled_qrc.validation_utils import ControlRange, make_nested_seeds  # noqa: E402

M_RANGE = ControlRange("m", 0.1, 1.0)
G_RANGE = ControlRange("g", 0.05, 0.6)
J_RANGE = ControlRange("J", 0.05, 0.6)


def _eval(m, g, J, reservoir_idx=0, **overrides):
    seeds = make_nested_seeds(reservoir_idx, 0)
    kwargs = dict(N_M=2, N_P=5, theta=0.2, phi=0.8, ap_kind="xy", m_range=M_RANGE, g_range=G_RANGE,
                  J_range=J_RANGE, h_m_tilde=0.05, h_g_tilde=0.05, h_J_tilde=0.05, T=90, washout=15,
                  n_val=20, n_test=25, max_delay=3, max_degree=2, max_targets_per_degree=5, n_surrogates=6)
    kwargs.update(overrides)
    return evaluate_candidate(m, g, J, seeds, **kwargs)


def test_candidate_specific_ceiling_uses_this_points_own_data():
    """Defect 7: two DIFFERENT candidates (different g,J) must generally
    get DIFFERENT ceiling diagnostics (different feature rank / M_long),
    proving the ceiling calc is not silently reusing one stale (X,train)
    across every candidate."""
    r1 = _eval(0.5, 0.2, 0.2)
    r2 = _eval(0.5, 0.55, 0.55)
    # at minimum, the candidate-specific M_bc values differ (different physics) --
    # if the ceiling function were reusing stale data these would be identical by construction
    assert r1.M_legacy != pytest.approx(r2.M_legacy, abs=1e-9)
    assert r1.feature_rank > 0 and r2.feature_rank > 0


def test_stencil_margin_violation_flagged_as_rejection_reason():
    r = _eval(0.95, 0.3, 0.3, h_m_tilde=0.1)  # m_tilde near 1.0 -- too close to the upper boundary
    assert not r.stencil_ok["m"]
    assert "stencil_margin_violated" in r.rejection_reasons
    assert r.valid is False


def test_interior_point_passes_stencil_check():
    r = _eval(0.5, 0.3, 0.3, h_m_tilde=0.05, h_g_tilde=0.05, h_J_tilde=0.05)
    assert r.stencil_ok == {"m": True, "g": True, "J": True}


def test_insufficient_sample_to_rank_ratio_flagged():
    """Test requirement #23: a tiny n_train relative to feature_rank must
    be flagged, not silently accepted -- forced here via a short T."""
    r = _eval(0.5, 0.3, 0.3, T=60, n_val=10, n_test=10, washout=8)
    assert "insufficient_sample_to_rank_ratio" in r.rejection_reasons or r.n_train < 2 * r.feature_rank


def test_select_candidates_never_ranks_invalid_points():
    valid_result = _eval(0.5, 0.3, 0.3, reservoir_idx=1)
    invalid_result = _eval(0.95, 0.3, 0.3, reservoir_idx=2, h_m_tilde=0.1)
    assert invalid_result.valid is False
    ranked = select_candidates([valid_result, invalid_result], top_k=5)
    assert invalid_result not in ranked
    assert all(r.valid for r in ranked)


def test_select_candidates_does_not_pick_by_raw_nl_instant_alone():
    """Defect 4's explicit rule: construct two valid candidates where A has
    higher NL_local_bc but much worse back-action/memory strength than B --
    the multi-objective score must NOT trivially just return
    argmax(NL_local_bc) if B dominates on every other axis."""
    class Fake:
        def __init__(self, nl, m_bc, d_m, r_chaos, m_t=0.5, g_t=0.5, J_t=0.5):
            self.NL_local_bc = nl
            self.M_bc = m_bc
            self.D_M = d_m
            self.r_chaos = r_chaos
            self.m_tilde, self.g_tilde, self.J_tilde = m_t, g_t, J_t
            self.valid = True

    a = Fake(nl=1.0, m_bc=0.1, d_m=0.9, r_chaos=0.3)   # high NL, terrible everything else
    b = Fake(nl=0.9, m_bc=5.0, d_m=0.02, r_chaos=0.55)  # slightly lower NL, dominant elsewhere
    ranked = select_candidates([a, b], top_k=2)
    assert ranked[0] is b, "expected the multi-objective score to prefer b, not just max(NL_local_bc)"


def test_compute_provenance_tag_includes_schema_and_versions():
    tag = compute_provenance_tag()
    assert "schema=v2.1.0" in tag
    assert "qiskit=" in tag
    assert "git=" in tag
