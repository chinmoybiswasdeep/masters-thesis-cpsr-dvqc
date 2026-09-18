"""
test_candidate_screening.py -- V2.2 Phase 8 / test requirement #12:
candidate-specific Gate A diagnostics reused correctly in the corrected
fixed-delay screening pipeline, plus screening-score ranking behavior.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.candidate_screening import screen_candidate, screening_score  # noqa: E402
from decoupled_qrc.validation_utils import make_nested_seeds  # noqa: E402


def _screen(m, g, J, reservoir_idx=0, **overrides):
    seeds = make_nested_seeds(reservoir_idx, 0)
    kwargs = dict(N_M=2, N_P=5, theta=0.2, phi=0.8, ap_kind="xy", T=90, washout=15, n_val=20, n_test=25,
                  max_delay=8, max_degree=2, max_targets_per_degree=4, n_surrogates=8)
    kwargs.update(overrides)
    return screen_candidate(m, g, J, seeds, **kwargs)


def test_screening_reports_full_fixed_delay_tensor_not_collapsed():
    r = _screen(0.5, 0.3, 0.3)
    assert set(r.NL_by_delay_signed.keys()) == set(range(9))
    assert r.NL_0_signed == pytest.approx(r.NL_by_delay_signed[0], abs=1e-12)


def test_screening_uses_candidate_specific_ceiling():
    r1 = _screen(0.5, 0.2, 0.2, reservoir_idx=0)
    r2 = _screen(0.5, 0.55, 0.55, reservoir_idx=0)
    assert r1.M_legacy != pytest.approx(r2.M_legacy, abs=1e-9)


def test_screening_score_ranks_only_valid_candidates():
    r_valid = _screen(0.5, 0.3, 0.3, reservoir_idx=1)
    r_invalid = _screen(0.99, 0.01, 0.01, reservoir_idx=2, T=60, n_val=10, n_test=10, washout=8)
    ranked = screening_score([r_valid, r_invalid])
    assert all(r.valid for r in ranked)
    assert r_invalid not in ranked


def test_screening_score_components_recorded_for_transparency():
    r1 = _screen(0.5, 0.3, 0.3, reservoir_idx=3)
    r2 = _screen(0.4, 0.4, 0.2, reservoir_idx=4)
    ranked = screening_score([r1, r2])
    for r in ranked:
        assert hasattr(r, "score")
        assert hasattr(r, "score_components")
        assert set(r.score_components.keys()) == {"nl0", "low_backaction", "eoc_proximity", "balance"}
