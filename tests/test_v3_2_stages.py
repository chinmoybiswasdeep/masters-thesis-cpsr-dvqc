"""V3.2 stage machine: every guard here corresponds to a V3.1 defect."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_1_workflow import (  # noqa: E402
    FrozenConfigError, IllegalTransitionError, SeedStageError)
from decoupled_qrc.v3_2_stages import (  # noqa: E402
    CalibrationNotPassed, Checkpoint, RunStage, StageMachine, StageSeedBroker, StageViolation,
    assert_seeds_disjoint_nonempty, verify_confirmation_complete)
from decoupled_qrc.v3_seeds import confirmation_seeds, discovery_seeds  # noqa: E402


# --------------------------------------------------------------------------
# Concern 11: a SMOKE run froze a candidate
# --------------------------------------------------------------------------
def test_smoke_cannot_freeze(tmp_path):
    sm = StageMachine(stage=RunStage.SMOKE)
    with pytest.raises(IllegalTransitionError):
        sm.freeze({"a": 1}, tmp_path / "frozen.json")


def test_calibration_cannot_freeze(tmp_path):
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    with pytest.raises(IllegalTransitionError):
        sm.freeze({"a": 1}, tmp_path / "frozen.json")


def test_freeze_is_refused_when_a_hard_gate_failed(tmp_path):
    """V3.1 froze with the processor gate at E_NL = 0.0739 against 0.20."""
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": True})
    sm.advance(RunStage.DISCOVERY)
    with pytest.raises(StageViolation):
        sm.freeze({"a": 1}, tmp_path / "frozen.json",
                  gates_passed={"E_processor_controllability": False})


def test_freeze_succeeds_only_with_calibration_and_passing_gates(tmp_path):
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": True})
    sm.advance(RunStage.DISCOVERY)
    out = sm.freeze({"a": 1}, tmp_path / "frozen.json", gates_passed={"E": True})
    assert "sha256" in out and (tmp_path / "frozen.json").exists()


# --------------------------------------------------------------------------
# Concern 16 / claim inflation: SMOKE printed "claim level 3"
# --------------------------------------------------------------------------
def test_smoke_and_calibration_cannot_claim_above_level_zero():
    sm = StageMachine(stage=RunStage.SMOKE)
    assert sm.max_claim_level() == 0
    with pytest.raises(StageViolation):
        sm.assert_claim_allowed(3)
    sm.advance(RunStage.CALIBRATION)
    with pytest.raises(StageViolation):
        sm.assert_claim_allowed(1)


def test_discovery_caps_the_claim_at_level_one():
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": True})
    sm.advance(RunStage.DISCOVERY)
    sm.assert_claim_allowed(1)
    with pytest.raises(StageViolation):
        sm.assert_claim_allowed(2)


# --------------------------------------------------------------------------
# Calibration failure must block discovery
# --------------------------------------------------------------------------
def test_calibration_failure_blocks_discovery():
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": False, "why": "no controllable m"})
    with pytest.raises(CalibrationNotPassed):
        sm.advance(RunStage.DISCOVERY)


def test_illegal_transitions_are_refused():
    sm = StageMachine(stage=RunStage.SMOKE)
    with pytest.raises(IllegalTransitionError):
        sm.advance(RunStage.CONFIRMATION)
    with pytest.raises(IllegalTransitionError):
        sm.advance(RunStage.DISCOVERY)


def test_report_is_terminal():
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.REPORT)
    with pytest.raises(IllegalTransitionError):
        sm.advance(RunStage.CALIBRATION)


# --------------------------------------------------------------------------
# Concern 13: confirmation must LOAD, never recreate
# --------------------------------------------------------------------------
def _frozen_machine(tmp_path, payload=None):
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": True})
    sm.advance(RunStage.DISCOVERY)
    sm.freeze(payload or {"m": 0.4, "g": 0.6}, tmp_path / "frozen.json",
              gates_passed={"E": True})
    return sm


def test_confirmation_requires_a_frozen_artifact():
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": True})
    sm.advance(RunStage.DISCOVERY)
    with pytest.raises(StageViolation):
        sm.advance(RunStage.CONFIRMATION)


def test_confirmation_loads_the_frozen_artifact(tmp_path):
    sm = _frozen_machine(tmp_path)
    sm.advance(RunStage.CONFIRMATION)
    loaded = sm.load_frozen_discovery()
    assert loaded["payload"] == {"m": 0.4, "g": 0.6}
    assert len(loaded["sha256"]) == 64


def test_a_tampered_frozen_artifact_is_rejected(tmp_path):
    sm = _frozen_machine(tmp_path)
    blob = json.loads((tmp_path / "frozen.json").read_text())
    blob["payload"]["m"] = 0.95                      # retune after freezing
    (tmp_path / "frozen.json").write_text(json.dumps(blob))
    with pytest.raises(FrozenConfigError):
        sm.load_frozen_discovery()


def test_confirmation_config_drift_is_rejected(tmp_path):
    sm = _frozen_machine(tmp_path)
    sm.advance(RunStage.CONFIRMATION)
    sm.assert_confirmation_matches_frozen({"m": 0.4, "g": 0.6})
    with pytest.raises(FrozenConfigError):
        sm.assert_confirmation_matches_frozen({"m": 0.42, "g": 0.6})


# --------------------------------------------------------------------------
# Concern 14: a disjointness proof over an empty set is vacuous
# --------------------------------------------------------------------------
def test_legacy_disjointness_proof_passes_vacuously_on_an_empty_set():
    """Regression pin for V3.1 concern 14. The V1-V3.1 helper is deliberately
    NOT modified (this repository is append-only), so its vacuous behaviour is
    documented here instead."""
    from decoupled_qrc.v3_seeds import assert_disjoint
    assert_disjoint([discovery_seeds(k) for k in range(5)], [])     # does not raise


def test_v3_2_disjointness_proof_rejects_an_empty_confirmation_set():
    with pytest.raises(SeedStageError):
        assert_seeds_disjoint_nonempty([discovery_seeds(k) for k in range(5)], [])


def test_v3_2_disjointness_proof_rejects_an_empty_discovery_set():
    with pytest.raises(SeedStageError):
        assert_seeds_disjoint_nonempty([], [confirmation_seeds(k) for k in range(3)])


def test_v3_2_disjointness_proof_checks_every_stream():
    rep = assert_seeds_disjoint_nonempty([discovery_seeds(k) for k in range(6)],
                                         [confirmation_seeds(k) for k in range(5)])
    assert rep["n_discovery"] == 6 and rep["n_confirmation"] == 5
    for name, entry in rep["streams"].items():
        assert entry["overlap"] == [], name


def test_planted_overlap_is_detected():
    with pytest.raises(SeedStageError):
        assert_seeds_disjoint_nonempty([discovery_seeds(0)], [discovery_seeds(0)])


def test_discovery_stage_cannot_obtain_confirmation_seeds():
    b = StageSeedBroker()
    s = b.for_stage(RunStage.DISCOVERY, 0)
    assert s.reservoir_idx < 0                       # discovery family
    assert b.issued["confirmation"] == []
    with pytest.raises(SeedStageError):
        b.proof()                                    # empty confirmation set -> vacuous
    c = b.for_stage(RunStage.CONFIRMATION, 0)
    assert c.reservoir_idx >= 0
    assert b.proof()["n_confirmation"] == 1


def test_calibration_uses_the_discovery_family_never_the_holdout():
    b = StageSeedBroker()
    s = b.for_stage(RunStage.CALIBRATION, 2)
    assert s.reservoir_idx < 0
    assert b.issued["confirmation"] == []


# --------------------------------------------------------------------------
# Concern 12: confirmation completeness must be DERIVED
# --------------------------------------------------------------------------
def test_placeholder_confirmation_fails_completeness():
    """V3.1's confirmation_results.csv held only 'no rows produced at this stage'."""
    rep = verify_confirmation_complete([], ["r1", "r2"], config_hash="abc")
    assert not rep["complete"] and rep["missing"] == ["r1", "r2"]


def test_no_expected_rows_is_not_complete():
    rep = verify_confirmation_complete([], [], config_hash="abc")
    assert not rep["complete"]


def test_complete_confirmation_is_recognised():
    rows = [{"row_key": "r1", "config_hash": "abc", "M_long": 1.0, "NL_0": 2.0},
            {"row_key": "r2", "config_hash": "abc", "M_long": 1.1, "NL_0": 2.1}]
    rep = verify_confirmation_complete(rows, ["r1", "r2"], required_metrics=("M_long", "NL_0"),
                                       config_hash="abc")
    assert rep["complete"] and rep["n_rows"] == 2


def test_duplicate_rows_fail():
    rows = [{"row_key": "r1", "config_hash": "abc"}, {"row_key": "r1", "config_hash": "abc"}]
    assert not verify_confirmation_complete(rows, ["r1"], config_hash="abc")["complete"]


def test_wrong_config_hash_fails():
    rows = [{"row_key": "r1", "config_hash": "WRONG"}]
    rep = verify_confirmation_complete(rows, ["r1"], config_hash="abc")
    assert not rep["complete"] and rep["rows_with_wrong_config_hash"] == ["r1"]


def test_non_finite_metric_fails():
    rows = [{"row_key": "r1", "config_hash": "abc", "M_long": float("nan")}]
    rep = verify_confirmation_complete(rows, ["r1"], required_metrics=("M_long",),
                                       config_hash="abc")
    assert not rep["complete"] and rep["non_finite_metrics"]


# --------------------------------------------------------------------------
# Resumability
# --------------------------------------------------------------------------
def test_checkpoint_resumes_without_duplicating(tmp_path):
    cp = Checkpoint(tmp_path / "rows.jsonl")
    assert cp.done_keys() == set()
    cp.append("r1", {"value": 1})
    cp.append("r2", {"value": 2})
    assert cp.done_keys() == {"r1", "r2"}
    cp2 = Checkpoint(tmp_path / "rows.jsonl")
    todo = [k for k in ("r1", "r2", "r3") if k not in cp2.done_keys()]
    assert todo == ["r3"]
    assert len(cp2.rows()) == 2


def test_checkpoint_tolerates_a_torn_final_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    cp = Checkpoint(path)
    cp.append("r1", {"value": 1})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"row_key": "r2", "val')          # interrupted mid-write
    assert Checkpoint(path).done_keys() == {"r1"}


# --------------------------------------------------------------------------
# Bypasses found by the independent adversarial review -- each was REAL and
# each is pinned closed here. Without these, the V3.1 failure modes were
# reachable in four lines.
# --------------------------------------------------------------------------
def test_machine_cannot_be_constructed_past_smoke():
    """Constructing directly at a later stage skipped every prerequisite."""
    with pytest.raises(StageViolation):
        StageMachine(stage=RunStage.DISCOVERY, calibration_passed=True)
    with pytest.raises(StageViolation):
        StageMachine(stage=RunStage.CONFIRMATION)
    StageMachine(stage=RunStage.SMOKE)                       # the only legal entry


def test_calibration_verdict_is_derived_not_asserted():
    """A caller could previously assert passed=True over an empty report."""
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    with pytest.raises(StageViolation):
        sm.record_calibration({}, passed=True)               # no verdict in the report
    with pytest.raises(StageViolation):
        sm.record_calibration({"passed": False}, passed=True)  # contradicts the report
    sm.record_calibration({"passed": True})
    assert sm.calibration_passed is True


def test_freeze_requires_explicit_gate_verdicts(tmp_path):
    """`gates_passed=None` used to mean 'nothing failed'."""
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm.record_calibration({"passed": True})
    sm.advance(RunStage.DISCOVERY)
    with pytest.raises(StageViolation):
        sm.freeze({"m": 0.95}, tmp_path / "f.json")           # no gates supplied
    with pytest.raises(StageViolation):
        sm.freeze({"m": 0.95}, tmp_path / "f.json", gates_passed={})
    assert sm.freeze({"m": 0.4}, tmp_path / "f.json", gates_passed={"E": True})["sha256"]


def test_claim_cap_follows_stages_completed_not_current_stage():
    """SMOKE -> REPORT used to grant claim level 5."""
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.REPORT)
    assert sm.max_claim_level() == 0
    with pytest.raises(StageViolation):
        sm.assert_claim_allowed(1)

    sm2 = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    sm2.record_calibration({"passed": True})
    sm2.advance(RunStage.DISCOVERY).advance(RunStage.REPORT)
    assert sm2.max_claim_level() == 1                         # earned by DISCOVERY, kept in REPORT
    with pytest.raises(StageViolation):
        sm2.assert_claim_allowed(2)
