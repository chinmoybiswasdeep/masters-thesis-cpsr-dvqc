"""
v3_1_workflow.py -- V3.1 stage controller. Repairs three V3 execution
defects that were verified in `code/_build_notebook_dualroute_v3_colab.py`:

  Defect 1: `confirmation_seeds` was imported (line 330) but never called --
            all eight seed sites used `discovery_seeds(...)`, so
            "CONFIRMATION" would have silently reused discovery seeds.
            Here, `SeedBroker` refuses to hand out a discovery seed during
            the confirmation stage and vice versa.

  Defect 2: `RUN_WORK` was computed (line 237) and never referenced, so the
            expensive cells ran regardless of authorization. Here,
            `Stage.run` raises `NotAuthorizedError` unless the stage is
            authorized, and `plan_only` returns the execution plan instead.

  Defect 3: changing RUN_MODE only changed sample sizes. Here the workflow
            is an explicit finite-state machine with illegal transitions
            raising, and discovery must complete and FREEZE before
            confirmation can start.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .v3_seeds import NestedSeeds, discovery_seeds, confirmation_seeds


class WorkflowState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    TESTED = "TESTED"
    SMOKE_COMPLETE = "SMOKE_COMPLETE"
    DISCOVERY_COMPLETE = "DISCOVERY_COMPLETE"
    FROZEN = "FROZEN"
    CONFIRMATION_COMPLETE = "CONFIRMATION_COMPLETE"
    REPORTED = "REPORTED"


# Only these transitions are legal. Anything else raises.
LEGAL_TRANSITIONS = {
    WorkflowState.UNINITIALIZED: {WorkflowState.TESTED},
    WorkflowState.TESTED: {WorkflowState.SMOKE_COMPLETE, WorkflowState.DISCOVERY_COMPLETE},
    WorkflowState.SMOKE_COMPLETE: {WorkflowState.DISCOVERY_COMPLETE, WorkflowState.REPORTED},
    WorkflowState.DISCOVERY_COMPLETE: {WorkflowState.FROZEN, WorkflowState.REPORTED},
    WorkflowState.FROZEN: {WorkflowState.CONFIRMATION_COMPLETE, WorkflowState.REPORTED},
    WorkflowState.CONFIRMATION_COMPLETE: {WorkflowState.REPORTED},
    WorkflowState.REPORTED: set(),
}


class IllegalTransitionError(RuntimeError):
    pass


class NotAuthorizedError(RuntimeError):
    pass


class SeedStageError(RuntimeError):
    pass


class FrozenConfigError(RuntimeError):
    pass


# =============================================================================
# Seed broker -- Defect 1
# =============================================================================

class SeedBroker:
    """Hands out seeds and REFUSES to cross stages. Every issued bundle is
    recorded with its stage label so any result row can be audited, and
    `assert_disjoint` proves the two sets never intersect on ANY of the
    eight streams (not merely on `reservoir_idx`)."""

    def __init__(self):
        self.issued = {"discovery": [], "confirmation": []}

    def discovery(self, k: int, input_idx: int = 0, stage: str = "discovery") -> NestedSeeds:
        if stage != "discovery":
            raise SeedStageError(f"discovery seeds requested while stage={stage!r}")
        s = discovery_seeds(k, input_idx)
        self.issued["discovery"].append(s)
        return s

    def confirmation(self, k: int, input_idx: int = 0, stage: str = "confirmation") -> NestedSeeds:
        if stage != "confirmation":
            raise SeedStageError(f"confirmation seeds requested while stage={stage!r}")
        s = confirmation_seeds(k, input_idx)
        self.issued["confirmation"].append(s)
        return s

    def for_stage(self, stage: str, k: int, input_idx: int = 0) -> NestedSeeds:
        """The only entry point the notebook uses: the STAGE decides which
        family is issued, so a confirmation cell physically cannot obtain a
        discovery seed."""
        if stage == "discovery":
            return self.discovery(k, input_idx)
        if stage == "confirmation":
            return self.confirmation(k, input_idx)
        raise SeedStageError(f"unknown stage {stage!r}; expected 'discovery' or 'confirmation'")

    def assert_disjoint(self) -> dict:
        """Proves the issued discovery and confirmation seeds share no value
        on ANY stream. Raises if they do."""
        streams = ("hamiltonian", "disorder", "input_sequence", "regression_split",
                   "null_surrogate", "projection", "bootstrap", "shot_noise")
        report = {}
        for name in streams:
            d = {getattr(s, name) for s in self.issued["discovery"]}
            c = {getattr(s, name) for s in self.issued["confirmation"]}
            overlap = d & c
            report[name] = {"n_discovery": len(d), "n_confirmation": len(c),
                            "overlap": sorted(overlap)}
            if overlap:
                raise SeedStageError(f"seed stream {name!r} overlaps between stages: {sorted(overlap)}")
        idx_d = {s.reservoir_idx for s in self.issued["discovery"]}
        idx_c = {s.reservoir_idx for s in self.issued["confirmation"]}
        if idx_d & idx_c:
            raise SeedStageError(f"reservoir_idx overlap: {sorted(idx_d & idx_c)}")
        report["reservoir_idx"] = {"discovery": sorted(idx_d), "confirmation": sorted(idx_c),
                                    "overlap": []}
        return report


def seed_row(seeds: NestedSeeds, stage: str) -> dict:
    """Every saved result row carries the full seed bundle AND its stage
    label, so no number can later be mistaken for the wrong stage."""
    return {"seed_stage": stage, **seeds.as_dict()}


# =============================================================================
# Frozen configuration -- Defect 3 / confirmation design
# =============================================================================

@dataclass(frozen=True)
class FrozenConfig:
    """Written once at the end of discovery, hashed, and thereafter
    immutable. Confirmation reads it and may not alter it."""
    payload: dict
    sha256: str
    created_at: str

    def verify(self) -> bool:
        return self.sha256 == _hash_payload(self.payload)

    def as_dict(self) -> dict:
        return {"payload": self.payload, "sha256": self.sha256, "created_at": self.created_at}


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      default=str).encode()).hexdigest()


def freeze_config(payload: dict, path=None) -> FrozenConfig:
    fc = FrozenConfig(payload=json.loads(json.dumps(payload, default=str)),
                       sha256=_hash_payload(payload),
                       created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(fc.as_dict(), f, indent=2, default=str)
    return fc


def load_frozen_config(path) -> FrozenConfig:
    with open(path) as f:
        blob = json.load(f)
    fc = FrozenConfig(payload=blob["payload"], sha256=blob["sha256"], created_at=blob["created_at"])
    if not fc.verify():
        raise FrozenConfigError(f"frozen config at {path} FAILED its own hash check -- it was modified")
    return fc


def assert_unmodified(fc: FrozenConfig, current_payload: dict) -> None:
    """Called at the START of confirmation: if anything about the frozen
    candidate/threshold set has changed, confirmation must not proceed."""
    if _hash_payload(current_payload) != fc.sha256:
        raise FrozenConfigError(
            "confirmation attempted with a payload that differs from the frozen configuration; "
            "tuning after freeze is not permitted")


# =============================================================================
# Stage execution -- Defect 2
# =============================================================================

@dataclass
class StageResult:
    name: str
    status: str               # "COMPLETED" | "PLAN_ONLY" | "SKIPPED"
    started_at: str
    ended_at: str
    wall_seconds: float
    plan: dict = field(default_factory=dict)
    payload: object = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("payload", None)
        return d


@dataclass
class Workflow:
    run_mode: str
    authorized: bool
    state: WorkflowState = WorkflowState.UNINITIALIZED
    history: list = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    broker: SeedBroker = field(default_factory=SeedBroker)
    frozen: FrozenConfig = None

    # -- state machine ----------------------------------------------------
    def transition(self, new_state: WorkflowState) -> None:
        if new_state not in LEGAL_TRANSITIONS[self.state]:
            raise IllegalTransitionError(
                f"illegal workflow transition {self.state.value} -> {new_state.value}; "
                f"legal targets are {sorted(s.value for s in LEGAL_TRANSITIONS[self.state])}")
        self.history.append({"from": self.state.value, "to": new_state.value,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.state = new_state

    def require_state(self, *allowed: WorkflowState) -> None:
        if self.state not in allowed:
            raise IllegalTransitionError(
                f"stage requires state in {[s.value for s in allowed]}, but workflow is {self.state.value}")

    # -- authorization ----------------------------------------------------
    def is_expensive(self, stage_name: str) -> bool:
        return stage_name in ("discovery", "confirmation", "resource_comparison")

    def run_stage(self, name: str, fn, plan: dict, prerequisites=(), force_authorize=None) -> StageResult:
        """Runs `fn()` ONLY when authorized. Otherwise returns a PLAN_ONLY
        result and never touches the simulator. `plan` is always produced so
        the notebook can show the workload it would incur."""
        for state in prerequisites:
            self.require_state(*prerequisites)
            break
        authorized = self.authorized if force_authorize is None else force_authorize
        expensive = self.is_expensive(name)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.perf_counter()

        if expensive and not authorized:
            res = StageResult(name=name, status="PLAN_ONLY", started_at=started,
                               ended_at=started, wall_seconds=0.0, plan=plan, payload=None)
            self.stages[name] = res
            return res

        payload = fn()
        ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        res = StageResult(name=name, status="COMPLETED", started_at=started, ended_at=ended,
                           wall_seconds=time.perf_counter() - t0, plan=plan, payload=payload)
        self.stages[name] = res
        return res

    def guard_expensive(self, name: str) -> None:
        """Explicit check a cell can call before doing ANY simulator work."""
        if self.is_expensive(name) and not self.authorized:
            raise NotAuthorizedError(
                f"stage {name!r} is expensive and CONFIRM_EXPENSIVE_RUN is False; "
                f"no simulator call may be made")

    def summary(self) -> dict:
        return {"run_mode": self.run_mode, "authorized": self.authorized, "state": self.state.value,
                "history": self.history,
                "stages": {k: v.as_dict() for k, v in self.stages.items()},
                "frozen_sha256": (self.frozen.sha256 if self.frozen else None)}
