"""
run_v3_2_stage.py -- the V3.2 stage runner.

    python run_v3_2_stage.py --stage SMOKE
    python run_v3_2_stage.py --stage CALIBRATION

The notebook (`DQRC_DualRoute_Decoupling_V3_2_Colab.ipynb`) calls exactly these
functions, so the notebook and a local run cannot diverge. Every stage writes
its artifacts under `results/dqrc_dual_route_v3_2/` and is resumable.

The stage machine enforces the ordering; this script only supplies the
settings and the I/O. In particular it cannot freeze a candidate from SMOKE
or CALIBRATION, and it cannot enter DISCOVERY unless CALIBRATION passed.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decoupled_qrc import v3_2_memory as mem                      # noqa: E402
from decoupled_qrc import v3_2_processor as proc                  # noqa: E402
from decoupled_qrc.v3_2_architecture import (                     # noqa: E402
    ArchitectureSpec, check_structural_isolation, evaluate_point, resource_row,
    run_architecture)
from decoupled_qrc.v3_2_cache import environment_fingerprint      # noqa: E402
from decoupled_qrc.v3_2_calibration import (                      # noqa: E402
    CalibrationSettings, run_calibration)
from decoupled_qrc.v3_2_gates import (                            # noqa: E402
    THRESHOLDS, gate_A_structural_isolation, gate_L_back_action, summarise)
from decoupled_qrc.v3_2_readout import ShotBudget                 # noqa: E402
from decoupled_qrc.v3_2_stages import RunStage, StageMachine, StageSeedBroker  # noqa: E402

RESULTS = HERE.parent / "results" / "dqrc_dual_route_v3_2"


def _write(name: str, payload) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonify, sort_keys=True)
    print(f"  wrote {path.relative_to(HERE.parent)}")
    return path


def _jsonify(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if k != "records"}
    return str(o)


def run_manifest(stage: str, started: float, extra: dict = None) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(HERE),
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(HERE),
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"stage": stage, "git_commit": commit, "worktree_dirty": dirty,
            "environment": environment_fingerprint(),
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "cpu_count": __import__("os").cpu_count()},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - started, 2),
            "thresholds": THRESHOLDS, **(extra or {})}


# =============================================================================
# SMOKE -- software validation only
# =============================================================================
def stage_smoke() -> dict:
    """Tiny end-to-end exercise. Proves the machinery runs and that the guards
    hold; makes NO scientific claim and CANNOT freeze anything."""
    t0 = time.time()
    sm = StageMachine(stage=RunStage.SMOKE)
    broker = StageSeedBroker()
    seeds = broker.for_stage(RunStage.SMOKE, 0, 0)

    mspec = mem.MemorySpec(mechanism="fractional_swap", L=3)
    pspec = proc.ProcessorSpec(N_P=3, dt=0.5, R=2)
    u = np.random.default_rng(int(seeds.input_sequence)).uniform(0, 1, 360)

    weak = ArchitectureSpec(architecture="dual_route_weak", memory=mspec, processor=pspec, lam=0.3)
    current = ArchitectureSpec(architecture="dual_route_current", memory=mspec, processor=pspec)

    iso = check_structural_isolation(current, u, m=0.4, g=0.6, J=0.5,
                                     disorder_seed=int(seeds.disorder),
                                     hamiltonian_seed=int(seeds.hamiltonian))
    injected = check_structural_isolation(current, u, m=0.4, g=0.6, J=0.5,
                                          disorder_seed=int(seeds.disorder),
                                          hamiltonian_seed=int(seeds.hamiltonian),
                                          m_leak_into_processor=0.3)
    gate_a = gate_A_structural_isolation(iso, injected)

    ev = evaluate_point(weak, m=0.4, g=0.6, J=0.5, seeds=seeds, budget=ShotBudget(5000),
                        T=360, washout=30, n_val=80, n_test=90, tau_min=2, max_delay=4,
                        max_degree=3, max_targets_per_degree=8, n_surrogates=5,
                        measure_back_action=True)
    run = run_architecture(weak, u, m=0.4, g=0.6, J=0.5, disorder_seed=int(seeds.disorder),
                           hamiltonian_seed=int(seeds.hamiltonian), measure_back_action=True)
    gate_l = gate_L_back_action(run.back_action)

    guards = _check_guards(sm)
    summary = summarise([gate_a, gate_l])
    sm.assert_claim_allowed(0)                      # SMOKE may claim nothing

    out = {"stage": "SMOKE", "purpose": "software validation only -- no scientific claim",
           "structural_isolation": iso, "injected_dependency_control": injected,
           "gate_A": gate_a.as_dict(), "gate_L": gate_l.as_dict(),
           "evaluation": {"M_long": ev.get("M_long"), "NL_0": ev.get("NL_0"),
                          "splits": ev["splits"], "dm_audit": ev["dm_audit"],
                          "back_action": ev["back_action"]},
           "resources": resource_row(weak, run, ShotBudget(5000)),
           "guards": guards,
           "gate_summary_note": ("claim level is capped at 0 in SMOKE regardless of the "
                                 "numbers above"),
           "max_claim_level_allowed": sm.max_claim_level(),
           "claim_level_reported": 0,
           "gate_table": summary["table"]}
    _write("smoke_results.json", out)
    _write("run_manifest_smoke.json", run_manifest("SMOKE", t0))
    return out


def _check_guards(sm: StageMachine) -> dict:
    """Actively demonstrate that the V3.1 failure modes are now impossible."""
    from decoupled_qrc.v3_1_workflow import IllegalTransitionError, SeedStageError
    from decoupled_qrc.v3_2_stages import (CalibrationNotPassed, StageViolation,
                                           assert_seeds_disjoint_nonempty,
                                           verify_confirmation_complete)
    from decoupled_qrc.v3_seeds import discovery_seeds
    guards = {}

    try:
        sm.freeze({"x": 1}, RESULTS / "_should_not_exist.json")
        guards["smoke_cannot_freeze"] = False
    except IllegalTransitionError:
        guards["smoke_cannot_freeze"] = True

    try:
        sm.assert_claim_allowed(3)
        guards["smoke_cannot_claim_level_3"] = False
    except StageViolation:
        guards["smoke_cannot_claim_level_3"] = True

    probe = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    probe.record_calibration({"passed": False, "why": "probe"})
    try:
        probe.advance(RunStage.DISCOVERY)
        guards["failed_calibration_blocks_discovery"] = False
    except CalibrationNotPassed:
        guards["failed_calibration_blocks_discovery"] = True

    try:
        assert_seeds_disjoint_nonempty([discovery_seeds(0)], [])
        guards["empty_confirmation_seed_proof_rejected"] = False
    except SeedStageError:
        guards["empty_confirmation_seed_proof_rejected"] = True

    guards["placeholder_confirmation_rejected"] = not verify_confirmation_complete(
        [], ["r1"], config_hash="abc")["complete"]
    return guards


# =============================================================================
# CALIBRATION -- the diagonal-control prerequisites
# =============================================================================
def stage_calibration(settings: CalibrationSettings = None) -> dict:
    t0 = time.time()
    settings = settings or CalibrationSettings()
    sm = StageMachine(stage=RunStage.SMOKE).advance(RunStage.CALIBRATION)
    broker = StageSeedBroker()

    pspec = proc.ProcessorSpec(N_P=5, dt=0.5, R=2)
    mechanisms = [mem.MemorySpec(mechanism="fractional_swap", L=4),
                  mem.MemorySpec(mechanism="leaky_collision", L=4)]

    report = run_calibration(pspec, mechanisms, settings, broker, verbose=True)
    sm.record_calibration(report)          # verdict is DERIVED from the report
    sm.assert_claim_allowed(0)                      # CALIBRATION may claim nothing

    report["stage"] = "CALIBRATION"
    report["max_claim_level_allowed"] = sm.max_claim_level()
    report["can_proceed_to_discovery"] = bool(report["passed"])
    _write("calibration_results.json", report)
    _write("run_manifest_calibration.json", run_manifest("CALIBRATION", t0))
    stale = RESULTS / "failure_diagnostics.json"
    if not report["passed"]:
        _write("failure_diagnostics.json",
               {"stage": "CALIBRATION", "passed": False,
                "diagnosis": report["diagnosis"], "failures": report["failures"],
                "consequence": ("DISCOVERY was NOT run. Per the preregistered failure "
                                "discipline the study stops at the earliest failed "
                                "prerequisite; no candidate is frozen and no claim above "
                                "Level 1 is made.")})
    elif stale.exists():
        # A failure report left by an earlier (e.g. reduced-settings) run would
        # misrepresent this one. Artifacts must describe the run that wrote them.
        stale.unlink()
        print("  removed stale failure_diagnostics.json from a previous failing run")
    return report


# =============================================================================
# REPORT -- figures and tables from saved artifacts only
# =============================================================================
def stage_report() -> dict:
    """Regenerate every figure the saved artifacts support.

    This stage NEVER simulates. A figure whose inputs do not exist is recorded
    as skipped with a reason rather than back-filled from another stage.
    """
    t0 = time.time()
    from decoupled_qrc.v3_2_report import build_report
    out = build_report(RESULTS, stage="REPORT")
    print(f"  figures rendered: {out['n_rendered']}, skipped: {out['n_skipped']}")
    _write("run_manifest_report.json", run_manifest("REPORT", t0,
                                                    {"figures": out["n_rendered"],
                                                     "figures_skipped": out["n_skipped"]}))
    return out


def main_stage(stage: str) -> dict:
    """DISCOVERY / CONFIRMATION are gated behind a passing CALIBRATION.

    They are deliberately not implemented until CALIBRATION passes: building an
    expensive sweep on top of an unverified prerequisite is exactly what V3.1
    did. The stage machine would refuse the transition in any case.
    """
    cal_path = RESULTS / "calibration_results.json"
    if not cal_path.exists():
        raise SystemExit("CALIBRATION has not been run; DISCOVERY cannot start")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if not cal.get("passed"):
        raise SystemExit("CALIBRATION did not pass -- DISCOVERY is NOT run. "
                         f"diagnosis: {cal.get('diagnosis')}")
    raise SystemExit(
        f"{stage} is not implemented in this iteration. CALIBRATION passed, so the next "
        f"step is to build out the discovery sweep against the frozen calibration outputs.")


def main():
    ap = argparse.ArgumentParser(description="V3.2 stage runner")
    ap.add_argument("--stage", required=True,
                    choices=["SMOKE", "CALIBRATION", "DISCOVERY", "CONFIRMATION", "REPORT"])
    ap.add_argument("--fast", action="store_true",
                    help="reduced calibration settings for a quick check (NOT a scientific run)")
    args = ap.parse_args()

    if args.stage == "SMOKE":
        out = stage_smoke()
        print(f"\nSMOKE complete. guards={out['guards']}")
        return
    if args.stage == "CALIBRATION":
        settings = CalibrationSettings()
        if args.fast:
            settings = CalibrationSettings(T=600, n_val=120, n_test=140, n_surrogates=9,
                                           max_targets_per_degree=8, n_reservoir_seeds=2,
                                           n_input_seeds=1,
                                           m_grid=(0.25, 0.35, 0.45, 0.55, 0.65),
                                           g_grid=(0.0, 0.4, 0.8), J_grid=(0.0, 0.4, 0.8))
        out = stage_calibration(settings)
        print(f"\nCALIBRATION passed={out['passed']}")
        return
    if args.stage == "REPORT":
        out = stage_report()
        print(f"\nREPORT complete: {out['n_rendered']} figures, {out['n_skipped']} skipped")
        return
    main_stage(args.stage)


if __name__ == "__main__":
    main()
