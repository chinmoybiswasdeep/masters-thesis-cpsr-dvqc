"""V4 artifacts: freezing, confirmation integrity, checkpoints, claim enforcement."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v4_artifacts import (  # noqa: E402
    CLAIM_SENTENCE, REQUIRED_FROZEN_COMPONENTS, Checkpoint, ClaimGuard, FrozenViolation,
    assert_matches_frozen, freeze, load_frozen, sha256_of, verify_complete)
from decoupled_qrc.v4_search import (  # noqa: E402
    CONFIRMATION_ARCH, CONFIRMATION_INPUT, DEVELOPMENT_ARCH, DEVELOPMENT_INPUT,
    SeedBankViolation, assert_confirmation_only, assert_development_only)


def _payload(**over):
    p = {k: {"x": 1} for k in REQUIRED_FROZEN_COMPONENTS}
    p.update(over)
    return p


# --------------------------------------------------------------------------
# Freezing
# --------------------------------------------------------------------------
def test_freeze_requires_every_component(tmp_path):
    incomplete = {k: {"x": 1} for k in REQUIRED_FROZEN_COMPONENTS[:-1]}
    with pytest.raises(FrozenViolation):
        freeze(incomplete, tmp_path / "f.json")
    fc = freeze(_payload(), tmp_path / "f.json")
    assert len(fc.sha256) == 64


def test_a_frozen_artifact_edited_afterwards_fails_to_load(tmp_path):
    p = tmp_path / "f.json"
    freeze(_payload(thresholds={"main_effect_min": 0.10}), p)
    blob = json.loads(p.read_text())
    blob["payload"]["thresholds"]["main_effect_min"] = 0.01      # weaken after freezing
    p.write_text(json.dumps(blob))
    with pytest.raises(FrozenViolation):
        load_frozen(p)


def test_frozen_round_trip_preserves_the_hash(tmp_path):
    p = tmp_path / "f.json"
    fc = freeze(_payload(grid={"m": [0, 1]}), p)
    assert load_frozen(p).sha256 == fc.sha256


def test_component_drift_is_detected(tmp_path):
    fc = freeze(_payload(grid={"m_values": [0, 1]}), tmp_path / "f.json")
    assert_matches_frozen(fc, {"m_values": [0, 1]}, "grid")
    with pytest.raises(FrozenViolation):
        assert_matches_frozen(fc, {"m_values": [0, 0.5, 1]}, "grid")


def test_hash_is_order_independent():
    assert sha256_of({"a": 1, "b": 2}) == sha256_of({"b": 2, "a": 1})
    assert sha256_of({"a": 1}) != sha256_of({"a": 2})


# --------------------------------------------------------------------------
# Seed banks
# --------------------------------------------------------------------------
def test_seed_banks_are_disjoint():
    assert set(DEVELOPMENT_ARCH).isdisjoint(CONFIRMATION_ARCH)
    assert set(DEVELOPMENT_INPUT).isdisjoint(CONFIRMATION_INPUT)


def test_confirmation_seeds_cannot_reach_development_code():
    assert_development_only(DEVELOPMENT_ARCH[:2], DEVELOPMENT_INPUT[:2])
    with pytest.raises(SeedBankViolation):
        assert_development_only([CONFIRMATION_ARCH[0]], DEVELOPMENT_INPUT[:1])
    with pytest.raises(SeedBankViolation):
        assert_development_only(DEVELOPMENT_ARCH[:1], [CONFIRMATION_INPUT[0]])


def test_development_seeds_cannot_reach_the_confirmation_run():
    assert_confirmation_only(CONFIRMATION_ARCH[:2], CONFIRMATION_INPUT[:2])
    with pytest.raises(SeedBankViolation):
        assert_confirmation_only([DEVELOPMENT_ARCH[0]], CONFIRMATION_INPUT[:1])


# --------------------------------------------------------------------------
# Confirmation integrity
# --------------------------------------------------------------------------
def test_placeholder_confirmation_is_not_complete():
    assert verify_complete([], ["r1", "r2"], config_hash="abc")["complete"] is False
    assert verify_complete([], [], config_hash="abc")["complete"] is False


def test_complete_confirmation_is_recognised():
    rows = [{"row_key": "r1", "config_hash": "abc", "M": 0.4, "N": 0.9},
            {"row_key": "r2", "config_hash": "abc", "M": 0.5, "N": 0.8}]
    rep = verify_complete(rows, ["r1", "r2"], required_fields=("M", "N"), config_hash="abc")
    assert rep["complete"] is True


def test_wrong_hash_duplicate_and_nonfinite_rows_all_fail():
    assert not verify_complete([{"row_key": "r1", "config_hash": "WRONG"}],
                               ["r1"], config_hash="abc")["complete"]
    dup = [{"row_key": "r1", "config_hash": "a"}, {"row_key": "r1", "config_hash": "a"}]
    assert not verify_complete(dup, ["r1"], config_hash="a")["complete"]
    nan = [{"row_key": "r1", "config_hash": "a", "M": float("nan")}]
    assert not verify_complete(nan, ["r1"], required_fields=("M",), config_hash="a")["complete"]


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def test_checkpoint_restart_is_reproducible(tmp_path):
    cp = Checkpoint(tmp_path / "rows.jsonl")
    cp.append("r1", {"v": 1})
    cp.append("r2", {"v": 2})
    again = Checkpoint(tmp_path / "rows.jsonl")
    assert again.done_keys() == {"r1", "r2"}
    todo = [k for k in ("r1", "r2", "r3") if k not in again.done_keys()]
    assert todo == ["r3"]
    assert [r["v"] for r in again.rows()] == [1, 2]


def test_checkpoint_survives_a_torn_final_line(tmp_path):
    p = tmp_path / "rows.jsonl"
    cp = Checkpoint(p)
    cp.append("r1", {"v": 1})
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"row_key": "r2", "v')
    assert Checkpoint(p).done_keys() == {"r1"}


# --------------------------------------------------------------------------
# Claim enforcement -- the sentence cannot be produced any other way
# --------------------------------------------------------------------------
def _good_gates():
    return {f"g{i}": {"passed": True, "status": "PASS"} for i in range(8)}


def test_claim_requires_every_gate_to_pass():
    g = _good_gates()
    g["g3"] = {"passed": False, "status": "FAIL"}
    out = ClaimGuard(gates=g, confirmation_complete=True, frozen_hash="a",
                     run_config_hash="a").render()
    assert out != CLAIM_SENTENCE and "g3" in out


def test_claim_requires_a_complete_confirmation():
    out = ClaimGuard(gates=_good_gates(), confirmation_complete=False,
                     frozen_hash="a", run_config_hash="a").render()
    assert out != CLAIM_SENTENCE and "not complete" in out


def test_claim_requires_matching_frozen_provenance():
    out = ClaimGuard(gates=_good_gates(), confirmation_complete=True,
                     frozen_hash="a", run_config_hash="b").render()
    assert out != CLAIM_SENTENCE and "does not match" in out
    out2 = ClaimGuard(gates=_good_gates(), confirmation_complete=True).render()
    assert out2 != CLAIM_SENTENCE and "provenance" in out2


def test_claim_requires_valid_sequential_control():
    out = ClaimGuard(gates=_good_gates(), confirmation_complete=True, frozen_hash="a",
                     run_config_hash="a", sequential_valid=False).render()
    assert out != CLAIM_SENTENCE and "sequential" in out


def test_empty_gate_set_cannot_claim():
    assert ClaimGuard(gates={}, confirmation_complete=True, frozen_hash="a",
                      run_config_hash="a").render() != CLAIM_SENTENCE


def test_claim_is_produced_only_when_everything_holds():
    guard = ClaimGuard(gates=_good_gates(), confirmation_complete=True,
                       frozen_hash="a", run_config_hash="a", sequential_valid=True)
    assert guard.blockers() == []
    assert guard.render() == CLAIM_SENTENCE
