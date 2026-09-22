"""
v3_2_stages.py -- the V3.2 stage machine and frozen-artifact protocol.

    RUN_STAGE in {SMOKE, CALIBRATION, DISCOVERY, CONFIRMATION, REPORT}

Each guard here exists because V3.1 violated exactly that rule:

  * SMOKE CANNOT FREEZE. V3.1's cell 9 printed
    `FROZEN CONFIG sha256 = 50ff8685...` and `workflow state -> FROZEN`
    while cell 0 held `RUN_MODE = "SMOKE"`, and it did so with the processor
    gate failing (best_E_NL = 0.0739 < 0.20). `StageMachine.freeze` refuses
    unless the stage is DISCOVERY, calibration has passed, and every hard
    gate the caller declares has passed.

  * SMOKE CANNOT CLAIM. V3.1's summary printed "Highest supported claim
    level: 3" from a SMOKE run whose confirmation had never executed.
    `max_claim_level` caps SMOKE and CALIBRATION at 0.

  * CALIBRATION FAILURE BLOCKS DISCOVERY -- the transition simply does not
    exist unless `calibration_passed` is recorded.

  * CONFIRMATION LOADS, NEVER RECREATES. `load_frozen_discovery` is the only
    way into CONFIRMATION, it verifies the sha256, and `confirmation_context`
    hands back a read-only view. V3.1's machine constrained TRANSITIONS but
    not DATA PROVENANCE, so nothing stopped a confirmation cell from
    recomputing what it should have loaded.

  * A DISJOINTNESS PROOF OVER AN EMPTY SET MUST FAIL.
    `v3_seeds.assert_disjoint` and `v3_1_workflow.SeedBroker.assert_disjoint`
    are set intersections: with V3.1's confirmation set empty (its
    `confirmation_results.csv` holds only "no rows produced at this stage"),
    the proof passed vacuously and proved nothing.
    `assert_seeds_disjoint_nonempty` requires BOTH sets to be non-empty.
    The V1-V3.1 functions are deliberately NOT modified -- this repository is
    append-only and their behaviour must stay reproducible -- so
    `tests/test_v3_2_stages.py` pins the legacy vacuous pass as a regression
    test alongside the strict V3.2 behaviour.

  * EVERY EXPENSIVE PHASE IS RESUMABLE. `Checkpoint` writes one row at a
    time and `confirmation_complete` is DERIVED from an expected-row-count
    check, never assigned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .v3_1_workflow import (FrozenConfigError, IllegalTransitionError, SeedStageError,
                            assert_unmodified, freeze_config, load_frozen_config)
from .v3_seeds import confirmation_seeds, discovery_seeds

SEED_STREAMS = ("hamiltonian", "disorder", "input_sequence", "regression_split",
                "null_surrogate", "projection", "bootstrap", "shot_noise")


class RunStage(str, Enum):
    SMOKE = "SMOKE"
    CALIBRATION = "CALIBRATION"
    DISCOVERY = "DISCOVERY"
    CONFIRMATION = "CONFIRMATION"
    REPORT = "REPORT"


LEGAL_STAGE_ORDER = {
    RunStage.SMOKE: {RunStage.CALIBRATION, RunStage.REPORT},
    RunStage.CALIBRATION: {RunStage.DISCOVERY, RunStage.REPORT},
    RunStage.DISCOVERY: {RunStage.CONFIRMATION, RunStage.REPORT},
    RunStage.CONFIRMATION: {RunStage.REPORT},
    RunStage.REPORT: set(),
}

# The highest claim level a stage may report, regardless of what the numbers
# look like. REPORT is NOT a constant 5: it inherits the cap earned by the
# stages actually completed (see `StageMachine.max_claim_level`). Treating
# REPORT as 5 let a SMOKE-only run reach REPORT and claim level 5 -- V3.1
# concern 16 reproduced exactly.
MAX_CLAIM_LEVEL = {RunStage.SMOKE: 0, RunStage.CALIBRATION: 0, RunStage.DISCOVERY: 1,
                   RunStage.CONFIRMATION: 5, RunStage.REPORT: 0}


class CalibrationNotPassed(RuntimeError):
    pass


class StageViolation(RuntimeError):
    pass


def assert_seeds_disjoint_nonempty(discovery_bundles, confirmation_bundles) -> dict:
    """Strict V3.2 disjointness proof: non-empty on BOTH sides, on ALL streams.

    A set intersection over an empty set proves nothing; V3.1 shipped exactly
    that. This raises instead.
    """
    d_list, c_list = list(discovery_bundles), list(confirmation_bundles)
    if not d_list:
        raise SeedStageError("discovery seed set is EMPTY -- a disjointness proof over an "
                             "empty set is vacuous and proves nothing")
    if not c_list:
        raise SeedStageError("confirmation seed set is EMPTY -- a disjointness proof over an "
                             "empty set is vacuous and proves nothing (this is V3.1 concern 14)")
    report = {"n_discovery": len(d_list), "n_confirmation": len(c_list), "streams": {}}
    for name in SEED_STREAMS:
        d = {getattr(s, name) for s in d_list}
        c = {getattr(s, name) for s in c_list}
        overlap = sorted(d & c)
        report["streams"][name] = {"n_discovery": len(d), "n_confirmation": len(c),
                                   "overlap": overlap}
        if overlap:
            raise SeedStageError(f"seed stream {name!r} overlaps between stages: {overlap}")
    idx_d = {s.reservoir_idx for s in d_list}
    idx_c = {s.reservoir_idx for s in c_list}
    if idx_d & idx_c:
        raise SeedStageError(f"reservoir_idx overlap: {sorted(idx_d & idx_c)}")
    report["reservoir_idx"] = {"discovery": sorted(idx_d), "confirmation": sorted(idx_c),
                              "overlap": []}
    return report


class StageSeedBroker:
    """Issues seeds and refuses to cross stages. Records what it issued so the
    disjointness proof is over the seeds ACTUALLY USED."""

    def __init__(self):
        self.issued = {"discovery": [], "confirmation": []}

    def for_stage(self, stage, k: int, input_idx: int = 0):
        stage = RunStage(stage)
        if stage is RunStage.DISCOVERY:
            s = discovery_seeds(k, input_idx)
            self.issued["discovery"].append(s)
            return s
        if stage is RunStage.CONFIRMATION:
            s = confirmation_seeds(k, input_idx)
            self.issued["confirmation"].append(s)
            return s
        if stage in (RunStage.SMOKE, RunStage.CALIBRATION):
            # calibration and smoke are software/prerequisite stages: they use
            # the DISCOVERY family so they can never touch a held-out seed
            s = discovery_seeds(k, input_idx)
            self.issued["discovery"].append(s)
            return s
        raise SeedStageError(f"stage {stage} does not issue seeds")

    def proof(self) -> dict:
        return assert_seeds_disjoint_nonempty(self.issued["discovery"],
                                              self.issued["confirmation"])


@dataclass
class StageMachine:
    """The guard rail. `stage` only moves along `LEGAL_STAGE_ORDER`."""

    stage: RunStage = RunStage.SMOKE
    calibration_passed: bool = False
    calibration_report: dict = field(default_factory=dict)
    frozen_path: Path = None
    history: list = field(default_factory=list)
    _allow_direct: bool = False        # tests only; never set in a scientific run

    def __post_init__(self):
        self.stage = RunStage(self.stage)
        if self.stage is not RunStage.SMOKE and not self._allow_direct:
            raise StageViolation(
                f"a StageMachine may only be CONSTRUCTED at SMOKE, not at {self.stage}. "
                f"Constructing directly at a later stage skips every prerequisite the "
                f"machine exists to enforce; use .advance() instead.")
        self.history.append(str(self.stage))

    def advance(self, target) -> "StageMachine":
        target = RunStage(target)
        if target not in LEGAL_STAGE_ORDER[self.stage]:
            raise IllegalTransitionError(
                f"illegal stage transition {self.stage} -> {target}; "
                f"legal: {sorted(str(s) for s in LEGAL_STAGE_ORDER[self.stage])}")
        if target is RunStage.DISCOVERY and not self.calibration_passed:
            raise CalibrationNotPassed(
                "CALIBRATION has not passed: the diagonal-control prerequisites "
                "(controllable memory response to m, interaction-generated NL_0) must be "
                "established before any discovery sweep is run")
        if target is RunStage.CONFIRMATION and self.frozen_path is None:
            raise StageViolation("CONFIRMATION requires a frozen discovery artifact; "
                                 "call freeze() first and confirmation will LOAD it")
        self.stage = target
        self.history.append(str(target))
        return self

    def record_calibration(self, report: dict, passed: bool = None) -> None:
        """Record the calibration outcome. `passed` is DERIVED from the report.

        Accepting the caller's word for it let a caller assert a pass over an
        empty report -- the machine then had no way to know the prerequisites
        had never been evaluated. A supplied `passed` is cross-checked and a
        mismatch raises.
        """
        if self.stage is not RunStage.CALIBRATION:
            raise StageViolation(f"calibration results can only be recorded in CALIBRATION, "
                                 f"not {self.stage}")
        if "passed" not in report:
            raise StageViolation(
                "the calibration report carries no 'passed' verdict; the machine will not "
                "take the caller's word that the prerequisites held")
        derived = bool(report["passed"])
        if passed is not None and bool(passed) != derived:
            raise StageViolation(
                f"caller claimed calibration passed={bool(passed)} but the report says "
                f"{derived}; refusing to record a verdict the evidence does not support")
        self.calibration_report = dict(report)
        self.calibration_passed = derived

    def max_claim_level(self) -> int:
        """The cap EARNED by the stages actually completed.

        A claim level is a statement about evidence, so it is derived from the
        run's history rather than from whichever stage happens to be current.
        Reaching REPORT does not earn anything by itself.
        """
        done = set(self.history)
        if str(RunStage.CONFIRMATION) in done:
            return MAX_CLAIM_LEVEL[RunStage.CONFIRMATION]
        if str(RunStage.DISCOVERY) in done:
            return MAX_CLAIM_LEVEL[RunStage.DISCOVERY]
        return 0

    def assert_claim_allowed(self, level: int) -> None:
        cap = self.max_claim_level()
        if int(level) > cap:
            raise StageViolation(
                f"stage {self.stage} may report at most claim level {cap}, not {level}. "
                f"V3.1 printed 'Highest supported claim level: 3' from a SMOKE run whose "
                f"confirmation had never executed; that is what this guard prevents.")

    def freeze(self, payload: dict, path, *, gates_passed: dict = None) -> dict:
        """Freeze the discovery artifact. Refused outside DISCOVERY, refused
        without calibration, refused if any declared hard gate failed."""
        if self.stage is not RunStage.DISCOVERY:
            raise IllegalTransitionError(
                f"freezing is only permitted in DISCOVERY, not {self.stage}. "
                f"V3.1 froze a candidate while RUN_MODE was 'SMOKE'.")
        if not self.calibration_passed:
            raise CalibrationNotPassed("cannot freeze without a passing CALIBRATION stage")
        if not gates_passed:
            raise StageViolation(
                "freeze() requires the hard-gate verdicts. Defaulting to 'no gates supplied "
                "means no gate failed' is how V3.1 froze a candidate whose processor gate "
                "read E_NL = 0.0739 against a 0.20 threshold.")
        failed = [k for k, v in dict(gates_passed).items() if not v]
        if failed:
            raise StageViolation(
                f"refusing to freeze: hard gate(s) failed {failed}. V3.1 froze its candidate "
                f"with the processor gate at E_NL = 0.0739 against a 0.20 threshold.")
        fc = freeze_config(payload, path)
        self.frozen_path = Path(path)
        return fc.as_dict() if hasattr(fc, "as_dict") else {"sha256": fc.sha256}

    def load_frozen_discovery(self, path=None) -> dict:
        """The ONLY entry point into confirmation's configuration.

        Returns a plain dict copy of the frozen payload; the caller cannot
        reach any discovery routine through it.
        """
        p = Path(path) if path is not None else self.frozen_path
        if p is None:
            raise StageViolation("no frozen artifact to load")
        fc = load_frozen_config(p)        # verifies its own sha256, raises if modified
        return {"payload": json.loads(json.dumps(fc.payload)), "sha256": fc.sha256,
                "created_at": fc.created_at, "path": str(p)}

    def assert_confirmation_matches_frozen(self, current_payload: dict) -> None:
        """Refuse a confirmation run whose config drifted from the freeze."""
        loaded = self.load_frozen_discovery()
        fc = load_frozen_config(loaded["path"])
        assert_unmodified(fc, current_payload)


# =============================================================================
# Resumable checkpointing + derived completeness
# =============================================================================
class Checkpoint:
    """Append-only JSONL row store so an interrupted run resumes at one row."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_keys(self) -> set:
        if not self.path.exists():
            return set()
        keys = set()
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line)["row_key"])
                except Exception:              # pragma: no cover -- tolerate a torn last line
                    continue
        return keys

    def append(self, row_key: str, row: dict) -> None:
        payload = dict(row)
        payload["row_key"] = str(row_key)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str, sort_keys=True) + "\n")

    def rows(self) -> list:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:          # pragma: no cover
                        continue
        return out


def verify_confirmation_complete(rows, expected_row_keys, *, required_metrics=(),
                                 config_hash: str = None) -> dict:
    """DERIVE the confirmation-complete flag. Never assign it.

    Checks: every expected row present, no duplicates, no unexpected rows,
    every row carries the frozen config hash, and every required metric is
    finite. A placeholder or empty result set fails all of these.
    """
    import math

    rows = list(rows)
    expected = list(dict.fromkeys(str(k) for k in expected_row_keys))
    seen = [str(r.get("row_key")) for r in rows]
    seen_set, expected_set = set(seen), set(expected)

    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    missing = sorted(expected_set - seen_set)
    unexpected = sorted(seen_set - expected_set)

    bad_hash, non_finite = [], []
    for r in rows:
        if config_hash is not None and str(r.get("config_hash")) != str(config_hash):
            bad_hash.append(r.get("row_key"))
        for metric in required_metrics:
            v = r.get(metric)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                non_finite.append((r.get("row_key"), metric))

    complete = bool(expected and not missing and not duplicates and not unexpected
                    and not bad_hash and not non_finite)
    return {"complete": complete, "n_expected": len(expected), "n_rows": len(rows),
            "missing": missing, "duplicates": duplicates, "unexpected": unexpected,
            "rows_with_wrong_config_hash": bad_hash,
            "non_finite_metrics": [list(x) for x in non_finite],
            "reason": ("ok" if complete else
                       "; ".join(filter(None, [
                           f"{len(missing)} missing" if missing else "",
                           f"{len(duplicates)} duplicated" if duplicates else "",
                           f"{len(unexpected)} unexpected" if unexpected else "",
                           f"{len(bad_hash)} wrong config hash" if bad_hash else "",
                           f"{len(non_finite)} non-finite metrics" if non_finite else "",
                           "no expected rows declared" if not expected else ""])))}
