"""
test_v3_1_workflow.py -- V3.1 stage control, seed separation, freeze
immutability and the expensive-run guard. Several of these tests would
FAIL against the V3 notebook, which is the point: they lock in the
repairs.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v3_1_workflow import (  # noqa: E402
    Workflow, WorkflowState, SeedBroker, SeedStageError, IllegalTransitionError,
    NotAuthorizedError, FrozenConfigError, freeze_config, load_frozen_config,
    assert_unmodified, seed_row)
from decoupled_qrc.v3_seeds import discovery_seeds, confirmation_seeds  # noqa: E402


# --------------------------------------------------------------------------
# Defect 1 -- seed separation
# --------------------------------------------------------------------------

def test_broker_refuses_discovery_seed_during_confirmation():
    """This is the V3 bug in test form: V3 called `discovery_seeds(...)` in
    every stage. The broker makes that impossible."""
    b = SeedBroker()
    with pytest.raises(SeedStageError):
        b.discovery(0, stage="confirmation")
    with pytest.raises(SeedStageError):
        b.confirmation(0, stage="discovery")


def test_for_stage_issues_the_right_family():
    b = SeedBroker()
    d = b.for_stage("discovery", 0)
    c = b.for_stage("confirmation", 0)
    assert d.reservoir_idx < 0, "discovery seeds must use negative reservoir_idx"
    assert c.reservoir_idx >= 0, "confirmation seeds must use non-negative reservoir_idx"


def test_all_eight_seed_streams_are_disjoint_between_stages():
    """Not merely `reservoir_idx`: EVERY stream must be disjoint."""
    b = SeedBroker()
    for k in range(6):
        for i in range(3):
            b.for_stage("discovery", k, i)
    for k in range(5):
        for i in range(3):
            b.for_stage("confirmation", k, i)
    report = b.assert_disjoint()
    for name, entry in report.items():
        if name == "reservoir_idx":
            continue
        assert entry["overlap"] == [], f"stream {name} overlapped: {entry['overlap']}"


def test_assert_disjoint_raises_on_a_planted_overlap():
    """The check must be able to FAIL -- otherwise it proves nothing."""
    b = SeedBroker()
    b.for_stage("discovery", 0)
    b.issued["confirmation"].append(discovery_seeds(0))   # plant the exact V3 mistake
    with pytest.raises(SeedStageError):
        b.assert_disjoint()


def test_every_result_row_carries_stage_and_all_seeds():
    row = seed_row(confirmation_seeds(2, 1), "confirmation")
    assert row["seed_stage"] == "confirmation"
    for stream in ("hamiltonian", "disorder", "input_sequence", "regression_split",
                   "null_surrogate", "projection", "bootstrap", "shot_noise"):
        assert stream in row


# --------------------------------------------------------------------------
# Defect 2 -- expensive-run guard
# --------------------------------------------------------------------------

def test_unauthorized_expensive_stage_never_calls_the_worker():
    """V3 computed `RUN_WORK` and never used it. Here an unauthorized
    expensive stage must not invoke the simulator function at all."""
    wf = Workflow(run_mode="DISCOVERY", authorized=False)
    calls = []

    def expensive():
        calls.append(1)
        return "should not happen"

    res = wf.run_stage("discovery", expensive, plan={"n_sims": 100})
    assert res.status == "PLAN_ONLY"
    assert calls == [], "no simulator call may occur when authorization is False"
    assert res.plan["n_sims"] == 100


def test_authorized_expensive_stage_runs():
    wf = Workflow(run_mode="DISCOVERY", authorized=True)
    calls = []
    res = wf.run_stage("discovery", lambda: calls.append(1) or "done", plan={})
    assert res.status == "COMPLETED" and calls == [1]


def test_cheap_stage_runs_without_authorization():
    wf = Workflow(run_mode="SMOKE", authorized=False)
    res = wf.run_stage("software_tests", lambda: "ok", plan={})
    assert res.status == "COMPLETED"


def test_guard_expensive_raises_when_unauthorized():
    wf = Workflow(run_mode="CONFIRMATION", authorized=False)
    with pytest.raises(NotAuthorizedError):
        wf.guard_expensive("confirmation")
    Workflow(run_mode="CONFIRMATION", authorized=True).guard_expensive("confirmation")


# --------------------------------------------------------------------------
# Defect 3 -- workflow state machine
# --------------------------------------------------------------------------

def test_legal_workflow_path():
    wf = Workflow(run_mode="CONFIRMATION", authorized=True)
    for state in (WorkflowState.TESTED, WorkflowState.DISCOVERY_COMPLETE, WorkflowState.FROZEN,
                  WorkflowState.CONFIRMATION_COMPLETE, WorkflowState.REPORTED):
        wf.transition(state)
    assert wf.state is WorkflowState.REPORTED
    assert len(wf.history) == 5


def test_confirmation_cannot_run_before_freeze():
    wf = Workflow(run_mode="CONFIRMATION", authorized=True)
    wf.transition(WorkflowState.TESTED)
    wf.transition(WorkflowState.DISCOVERY_COMPLETE)
    with pytest.raises(IllegalTransitionError):
        wf.transition(WorkflowState.CONFIRMATION_COMPLETE)


def test_cannot_skip_testing():
    wf = Workflow(run_mode="SMOKE", authorized=True)
    with pytest.raises(IllegalTransitionError):
        wf.transition(WorkflowState.DISCOVERY_COMPLETE)


def test_require_state_rejects_wrong_stage():
    wf = Workflow(run_mode="SMOKE", authorized=True)
    with pytest.raises(IllegalTransitionError):
        wf.require_state(WorkflowState.FROZEN)


# --------------------------------------------------------------------------
# Freeze immutability
# --------------------------------------------------------------------------

def test_frozen_config_roundtrip_and_hash(tmp_path):
    payload = {"candidate": {"m": 0.7, "g": 0.8, "J": 0.6}, "thresholds": {"E_NL": 0.2}}
    path = tmp_path / "frozen.json"
    fc = freeze_config(payload, path)
    assert fc.verify()
    loaded = load_frozen_config(path)
    assert loaded.sha256 == fc.sha256


def test_tampered_frozen_config_is_detected(tmp_path):
    path = tmp_path / "frozen.json"
    fc = freeze_config({"candidate": {"g": 0.8}}, path)
    blob = json.loads(path.read_text())
    blob["payload"]["candidate"]["g"] = 0.9        # tamper after freezing
    path.write_text(json.dumps(blob))
    with pytest.raises(FrozenConfigError):
        load_frozen_config(path)


def test_confirmation_cannot_modify_frozen_gates(tmp_path):
    """Gate thresholds are part of the frozen payload, so changing them
    after the freeze is detected before confirmation does any work."""
    payload = {"thresholds": {"eta_min": 0.7, "E_NL": 0.20}}
    fc = freeze_config(payload, tmp_path / "f.json")
    assert_unmodified(fc, payload)                                  # unchanged: fine
    with pytest.raises(FrozenConfigError):
        assert_unmodified(fc, {"thresholds": {"eta_min": 0.5, "E_NL": 0.20}})


def test_workflow_summary_is_serializable():
    wf = Workflow(run_mode="SMOKE", authorized=True)
    wf.transition(WorkflowState.TESTED)
    wf.run_stage("smoke", lambda: 1, plan={"n": 1})
    json.loads(json.dumps(wf.summary(), default=str))
