"""
test_run_stages.py -- V2.1 Defect 2 / test requirements #13, #14: genuine
FULL-mode seed-loop counts and held-out confirmation execution, verified
with a cheap monkeypatched fake `evaluate_fn` (no real quantum circuits).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.run_stages import discovery_stage, confirmation_stage, neighborhood_stage  # noqa: E402
from decoupled_qrc.validation_utils import ControlRange  # noqa: E402

M_RANGE = ControlRange("m", 0.1, 1.0)
G_RANGE = ControlRange("g", 0.05, 0.6)
J_RANGE = ControlRange("J", 0.05, 0.6)


class _FakeResult:
    def __init__(self, m, g, J, seeds, valid=True):
        self.m, self.g, self.J, self.seeds = m, g, J, seeds
        self.valid = valid


def _cheap_evaluate_fn(calls_log, valid_fn=lambda m, g, J, seeds: True):
    def fn(m, g, J, seeds):
        calls_log.append((m, g, J, seeds.reservoir_idx, seeds.input_idx))
        return _FakeResult(m, g, J, seeds, valid=valid_fn(m, g, J, seeds))
    return fn


def test_discovery_stage_visits_exactly_n_points_with_unique_discovery_seeds():
    calls = []
    outcome = discovery_stage(12, M_RANGE, G_RANGE, J_RANGE, _cheap_evaluate_fn(calls), sampling_seed=0)
    assert outcome.n_sampled == 12
    assert len(calls) == 12
    assert len(outcome.reservoir_idxs_used) == 12
    assert all(idx < 0 for idx in outcome.reservoir_idxs_used), "discovery must only ever use negative reservoir_idx"


def test_discovery_stage_records_both_valid_and_invalid_points():
    calls = []
    outcome = discovery_stage(10, M_RANGE, G_RANGE, J_RANGE,
                               _cheap_evaluate_fn(calls, valid_fn=lambda m, g, J, s: g > 0.3), sampling_seed=1)
    assert outcome.n_sampled == 10
    assert 0 < outcome.n_valid < 10, "expected a mix of valid/invalid points for this fake filter"
    assert len(outcome.all_results) == 10  # ALL points kept, not just the valid ones


def test_confirmation_stage_visits_every_reservoir_x_input_combination():
    """Test requirement #13: the required >=5 reservoir seeds x >=3 input
    seeds FULL-mode grid must be visited EXACTLY -- 15 calls, not fewer."""
    calls = []
    candidate = {"m": 0.5, "g": 0.3, "J": 0.3}
    outcome = confirmation_stage(candidate, n_reservoir_seeds=5, n_input_seeds=3,
                                  evaluate_fn=_cheap_evaluate_fn(calls))
    assert len(calls) == 15
    assert len(outcome.per_seed_results) == 15
    reservoir_idxs = {c[3] for c in calls}
    input_idxs = {c[4] for c in calls}
    assert reservoir_idxs == {0, 1, 2, 3, 4}
    assert input_idxs == {0, 1, 2}
    assert all(idx >= 0 for idx in reservoir_idxs), "confirmation must only ever use non-negative reservoir_idx"


def test_confirmation_stage_candidate_point_never_changes():
    calls = []
    candidate = {"m": 0.5, "g": 0.3, "J": 0.3}
    outcome = confirmation_stage(candidate, n_reservoir_seeds=2, n_input_seeds=2,
                                  evaluate_fn=_cheap_evaluate_fn(calls))
    for m, g, J, r_idx, i_idx in calls:
        assert (m, g, J) == (0.5, 0.3, 0.3), "confirmation must evaluate the SAME frozen candidate every time"
    assert outcome.candidate == candidate


def test_confirmation_stage_rejects_overlap_with_discovery_indices():
    """Test requirement #14 (held-out confirmation execution): confirmation
    must REFUSE to run if a discovery reservoir_idx sneaks into its own
    grid -- proving disjointness is actually enforced, not just achieved
    by accident through the negative/non-negative convention."""
    calls = []
    candidate = {"m": 0.5, "g": 0.3, "J": 0.3}
    with pytest.raises(ValueError, match="also used during discovery"):
        confirmation_stage(candidate, n_reservoir_seeds=3, n_input_seeds=1,
                            evaluate_fn=_cheap_evaluate_fn(calls), discovery_reservoir_idxs={1})


def test_discovery_then_confirmation_never_naturally_overlap():
    """End-to-end: discovery's own reservoir_idxs_used (always negative)
    passed as `discovery_reservoir_idxs` to confirmation must NEVER trigger
    the overlap check under the normal negative/non-negative convention."""
    calls_disc, calls_conf = [], []
    disc = discovery_stage(20, M_RANGE, G_RANGE, J_RANGE, _cheap_evaluate_fn(calls_disc), sampling_seed=2)
    candidate = {"m": 0.5, "g": 0.3, "J": 0.3}
    conf = confirmation_stage(candidate, n_reservoir_seeds=5, n_input_seeds=3,
                               evaluate_fn=_cheap_evaluate_fn(calls_conf),
                               discovery_reservoir_idxs=disc.reservoir_idxs_used)
    assert len(conf.per_seed_results) == 15
    assert disc.reservoir_idxs_used.isdisjoint(conf.reservoir_idxs_used)


def test_neighborhood_stage_visits_center_plus_stencil_offsets_on_every_axis():
    calls = []
    from decoupled_qrc.validation_utils import make_nested_seeds
    candidate = {"m": 0.5, "g": 0.3, "J": 0.3}
    seeds = make_nested_seeds(0, 0)
    outcome = neighborhood_stage(candidate, M_RANGE, G_RANGE, J_RANGE, h_tilde=0.05,
                                  evaluate_fn=_cheap_evaluate_fn(calls), seeds=seeds)
    # center + up to 4 nonzero offsets (-2h,-h,h,2h) per axis x 3 axes = up to 1 + 12 = 13
    assert outcome.results[0].m == pytest.approx(0.5)
    assert len(outcome.results) == len(outcome.offsets)
    assert len(outcome.results) >= 1 + 3 * 2  # at minimum some perturbations must have landed in-domain
    # every call used the SAME seeds object (common random numbers across the neighborhood)
    reservoir_idxs_used = {c[3] for c in calls}
    assert reservoir_idxs_used == {seeds.reservoir_idx}


def test_neighborhood_stage_skips_out_of_domain_offsets():
    """A candidate near the boundary with a large h_tilde must skip
    offsets that would fall outside [0,1], not silently evaluate garbage."""
    calls = []
    from decoupled_qrc.validation_utils import make_nested_seeds
    candidate = {"m": 0.12, "g": 0.3, "J": 0.3}  # m_tilde close to 0
    seeds = make_nested_seeds(0, 0)
    outcome = neighborhood_stage(candidate, M_RANGE, G_RANGE, J_RANGE, h_tilde=0.2,
                                  evaluate_fn=_cheap_evaluate_fn(calls), seeds=seeds)
    for m, g, J, r_idx, i_idx in calls:
        m_tilde = M_RANGE.to_dimensionless(m)
        assert -1e-9 <= m_tilde <= 1.0 + 1e-9
