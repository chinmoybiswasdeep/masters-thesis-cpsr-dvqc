"""
run_v4_stage.py -- the V4 stage runner.

    python run_v4_stage.py --stage AUDIT
    python run_v4_stage.py --stage SMOKE
    python run_v4_stage.py --stage CALIBRATION
    python run_v4_stage.py --stage ARCHITECTURE_SEARCH
    python run_v4_stage.py --stage STRESS_TEST
    python run_v4_stage.py --stage FREEZE
    python run_v4_stage.py --stage CONFIRMATION
    python run_v4_stage.py --stage REPORT

Stage order is enforced: each stage refuses to run unless its prerequisite
artifact exists. CONFIRMATION additionally refuses unless the frozen artifact
re-hashes correctly and the seed bank it is handed is disjoint from every seed
the search ever saw.

The notebook calls these same functions, so a Colab run and a local run cannot
diverge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decoupled_qrc.v4_architecture import (MemorySpec, ProcessorSpec, V4Spec,  # noqa: E402
                                           route_independence, run_v4,
                                           verify_product_factorisation)
from decoupled_qrc.v4_artifacts import (CLAIM_SENTENCE, Checkpoint, ClaimGuard,  # noqa: E402
                                        FrozenViolation, environment, freeze,
                                        load_frozen, sha256_of, verify_complete)
from decoupled_qrc.v4_encoder import audit_encoder_state, multilinearity_report  # noqa: E402
from decoupled_qrc.v4_experiment import (THRESHOLDS, evaluate_all_gates,  # noqa: E402
                                         evaluate_grid, freeze_margin_check, mean_grid)
from decoupled_qrc.v4_ipc import IPCConfig, compute_metrics  # noqa: E402
from decoupled_qrc.v4_nonlinear_memory import nm_summary, nonlinear_memory_curves  # noqa: E402
from decoupled_qrc.v4_search import (CONFIRMATION_ARCH, CONFIRMATION_INPUT,  # noqa: E402
                                     DEVELOPMENT_ARCH, DEVELOPMENT_INPUT,
                                     ObjectiveWeights, SearchSpace,
                                     assert_confirmation_only, build, random_search,
                                     stress_test)
from decoupled_qrc.v4_statistics import SequentialLedger  # noqa: E402

RESULTS = HERE.parent / "results" / "v4"
FROZEN_PATH = RESULTS / "frozen_config.json"
LEDGER_PATH = RESULTS / "sequential_ledger.json"

# Preregistered grid and sizes, fixed before the search.
DEV_M = [0.0, 0.5, 1.0]
DEV_G = [0.0, 0.5, 1.0]
CONF_M = [0.0, 0.25, 0.5, 0.75, 1.0]
CONF_G = [0.0, 0.25, 0.5, 0.75, 1.0]
DEV_T = 1200
CONF_T = 1600
VERSION = "v4.0"


def _write(name: str, payload) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    from decoupled_qrc.v4_artifacts import _jsonable
    p.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {p.relative_to(HERE.parent)}")
    return p


def _load(name: str):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"missing prerequisite artifact {p.name}; run the earlier stage first")
    return json.loads(p.read_text(encoding="utf-8"))


def _manifest(stage: str, t0: float, extra: dict = None) -> dict:
    return {"stage": stage, "version": VERSION, "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - t0, 2),
            "thresholds": dict(THRESHOLDS), **(extra or {})}


# =============================================================================
def stage_audit() -> dict:
    """AUDIT -- the encoder claims, checked symbolically/numerically."""
    t0 = time.time()
    enc = {n: audit_encoder_state(n) for n in (1, 2, 3, 4)}
    spec = V4Spec()
    from decoupled_qrc.v4_architecture import MemoryRoute

    def feat(us):
        r = MemoryRoute(spec.memory, 0.8, 3)
        f = None
        for u in us:
            f = r.step(float(u))
        return float(f[0])

    multi = multilinearity_report(feat, 3, base=[0.3, -0.5, 0.15])
    out = {"stage": "AUDIT",
           "encoder_polynomial_expansion": enc,
           "memory_route_multilinearity": multi,
           "finding": (
               "((I+uZ)/2)^(x)n is NOT globally linear: the n-body observable carries "
               "degrees up to n. It is affine only at the LOCAL readout, and the n copies "
               "are counted as a nonlinear resource."),
           "capability_boundary": (
               "affine single-copy injection + CPTP chain => the memory state is "
               "MULTILINEAR in the past inputs, so any target with degree >= 2 at a "
               "strictly positive delay is unreachable. N_long is therefore built from "
               "targets whose degree sits at delay 0.")}
    _write("audit.json", out)
    _write("manifest_audit.json", _manifest("AUDIT", t0))
    print(f"  encoder globally affine? "
          f"{[enc[n]['state_is_globally_affine'] for n in (1,2,3,4)]}")
    print(f"  memory route multilinear? {multi['multilinear']} "
          f"(max degree per input {multi['max_degree_per_input']})")
    return out


def stage_smoke() -> dict:
    """SMOKE -- tiny end-to-end run; software validation only, no claim."""
    t0 = time.time()
    spec = V4Spec(memory=MemorySpec(L_R=3), processor=ProcessorSpec(N_P=3), n_joint=16)
    cfg = IPCConfig(alpha=0.5, max_delay=4, tau_L=2, tau_S=0, nlong_max_delay=4, n_null=20)
    u = np.random.default_rng(int(DEVELOPMENT_INPUT[0])).uniform(-1, 1, 500)
    run = run_v4(spec, u, m=0.6, g=0.5, seed=int(DEVELOPMENT_ARCH[0]), audit_state=True)
    res = compute_metrics(run, cfg)
    iso = route_independence(spec, u[:40], m=0.6, g=0.5, seed=int(DEVELOPMENT_ARCH[0]))
    fac = verify_product_factorisation(spec, u[:10], m=0.6, g=0.5,
                                       seed=int(DEVELOPMENT_ARCH[0]))
    guard = ClaimGuard(gates={}, confirmation_complete=False)
    out = {"stage": "SMOKE", "M": res.M, "N": res.N, "N_long": res.N_long,
           "route_independence": iso, "factorisation": fac,
           "dm_audit": run.dm_audit, "resources": spec.resources(),
           "saturation": res.saturation, "null": res.null,
           "claim": guard.render()}
    _write("smoke.json", out)
    _write("manifest_smoke.json", _manifest("SMOKE", t0))
    print(f"  M={res.M:.4f} N={res.N:.4f} N_long={res.N_long:.4f}  "
          f"isolation_exact={iso['passed']} factorises={fac['factorises']}")
    print(f"  claim: {out['claim']}")
    return out


def stage_calibration() -> dict:
    """CALIBRATION -- null threshold and saturation, fixed BEFORE the search."""
    t0 = time.time()
    spec, cfg = build(_default_candidate())
    rows = []
    for m in DEV_M:
        for g in DEV_G:
            u = np.random.default_rng(int(DEVELOPMENT_INPUT[0])).uniform(-1, 1, DEV_T)
            res = compute_metrics(run_v4(spec, u, m=m, g=g,
                                         seed=int(DEVELOPMENT_ARCH[0])), cfg)
            rows.append({"m": m, "g": g, "M": res.M, "N": res.N, "N_long": res.N_long,
                         "null_threshold": res.null["threshold"],
                         "saturated_fraction": res.saturation["saturated_fraction"],
                         "n_primary_targets": res.saturation["n_targets"]})
    worst_sat = max(r["saturated_fraction"] for r in rows)
    null_med = float(np.median([r["null_threshold"] for r in rows]))
    passed = bool(worst_sat <= THRESHOLDS["saturation_max_fraction"])
    out = {"stage": "CALIBRATION", "rows": rows, "worst_saturated_fraction": worst_sat,
           "null_threshold_median": null_med, "passed": passed,
           "note": ("the null procedure and the saturation limit are preregistered here "
                    "and are not revisited after results are seen")}
    _write("calibration.json", out)
    _write("manifest_calibration.json", _manifest("CALIBRATION", t0))
    print(f"  worst saturated fraction {worst_sat:.3f} (limit "
          f"{THRESHOLDS['saturation_max_fraction']}) -> passed={passed}")
    print(f"  null threshold (median) {null_med:.5f}")
    return out


def _default_candidate() -> dict:
    """The hand-designed starting point the search must beat or match."""
    return {"L_R": 4, "hop": 0.7, "tau_mix": 1.0, "m_max": 0.9,
            "retention": "amplitude_damping", "pair_readout": True,
            "N_P": 4, "depth": 2, "dt": 0.6, "g_scale": 0.5,
            "n_joint": 72, "alpha": 0.5}


def stage_search(n_candidates: int = 24) -> dict:
    """ARCHITECTURE_SEARCH -- development seeds only."""
    t0 = time.time()
    cal = _load("calibration.json")
    if not cal.get("passed"):
        raise SystemExit("CALIBRATION did not pass; the metric is not valid to search on")
    arch = list(DEVELOPMENT_ARCH[:2])
    inp = list(DEVELOPMENT_INPUT[:2])
    ck = Checkpoint(RESULTS / "search_history.jsonl")
    print(f"  searching {n_candidates} candidates on arch={arch} input={inp}")
    out = random_search(n_candidates=n_candidates, seed=20260923,
                        m_values=DEV_M, g_values=DEV_G, arch_seeds=arch,
                        input_seeds=inp, T=DEV_T, space=SearchSpace(),
                        weights=ObjectiveWeights(), checkpoint=ck)
    # the hand-designed default competes on the same footing
    spec, cfg = build(_default_candidate())
    grid = evaluate_grid(spec, cfg, m_values=DEV_M, g_values=DEV_G,
                         arch_seeds=arch, input_seeds=inp, T=DEV_T)
    from decoupled_qrc.v4_search import objective, summarise_candidate
    enc = float(np.mean([r["N"] for r in grid["rows"] if r["g"] == 0.0]))
    summ = summarise_candidate(grid, enc)
    base = {"candidate": _default_candidate(), "summary": summ,
            "score": objective(summ, ObjectiveWeights()), "error": None,
            "row_key": "hand_designed"}
    ranked = sorted(out["ranked"] + [base], key=lambda h: h["score"])
    out["ranked"] = ranked
    out["best"] = ranked[0]
    out["hand_designed"] = base
    out["stage"] = "ARCHITECTURE_SEARCH"
    _write("search.json", out)
    _write("manifest_search.json", _manifest("ARCHITECTURE_SEARCH", t0,
                                             {"n_candidates": n_candidates}))
    b = out["best"]
    print(f"  best = {b['row_key']} score={b['score']:+.4f}")
    print(f"  params: {b['candidate']}")
    return out


def stage_stress() -> dict:
    """STRESS_TEST -- extra seeds and parameter perturbations."""
    t0 = time.time()
    best = _load("search.json")["best"]["candidate"]
    arch = list(DEVELOPMENT_ARCH[2:5])          # seeds the search did NOT use
    inp = list(DEVELOPMENT_INPUT[2:5])
    print(f"  stress on unseen development seeds arch={arch} input={inp}")
    out = stress_test(best, m_values=DEV_M, g_values=DEV_G, arch_seeds=arch,
                      input_seeds=inp, T=DEV_T)
    out["stage"] = "STRESS_TEST"
    out["candidate"] = best
    out["seeds"] = {"arch": arch, "input": inp}
    _write("stress_test.json", out)
    _write("manifest_stress.json", _manifest("STRESS_TEST", t0))
    print(f"  {out['n_all_passed']}/{out['n_variants']} variants pass every gate; "
          f"robust={out['robust']}")
    return out


def stage_freeze() -> dict:
    """FREEZE -- lock everything, hash it, and stop touching it."""
    t0 = time.time()
    stress = _load("stress_test.json")
    if not stress.get("robust"):
        raise SystemExit("refusing to FREEZE: the candidate is not robust under stress test")
    cand = stress["candidate"]
    spec, cfg = build(cand)
    payload = {
        "version": VERSION,
        "architecture": {"candidate": cand, "spec": spec.as_dict()},
        "grid": {"m_values": CONF_M, "g_values": CONF_G, "T": CONF_T},
        "metrics": {"M": "mean_{tau>=tau_L} C_{1,tau} from R-local observables",
                    "N": "weighted mean_{d=2..4, tau<=tau_S} C_{d,tau} from P-local observables",
                    "N_long": "mean over joint-observable cross-delay targets beyond tau_L",
                    "capacity": "C = 1 - MSE/Var on the held-out block"},
        "target_libraries": {"single_delay": "P_d(u_{t-tau}), d=1..4, tau=0..8",
                             "cross_delay": "P1P1, P2P1, P1P1P1 over distinct delays",
                             "nlong": "P_d(u_t) * P_1(u_{t-tau}), d in {2,3}, tau in 3..5",
                             "sunada": "sin(nu u_{t-tau})/nu, nu in {0,0.5,1,2,4}"},
        "thresholds": dict(THRESHOLDS),
        "preprocessing": {"standardisation": "train statistics only",
                          "washout": cfg.washout, "train_frac": cfg.train_frac,
                          "gap": cfg.gap},
        "observables": {"R": spec.memory.observables()[0],
                        "P": spec.processor.observables()[0],
                        "n_joint": spec.n_joint},
        "resources": spec.resources(),
        "statistics": {"bootstrap": "two-level, arch seeds then input seeds",
                       "n_boot": 4000, "main_ci": THRESHOLDS["main_ci_level"],
                       "equivalence_ci": THRESHOLDS["equivalence_ci_level"],
                       "multiplicity": "Holm", "equivalence": "TOST on the CI"},
        "ipc_config": cfg.as_dict(),
        "seed_bank": {"development_arch": list(DEVELOPMENT_ARCH),
                      "development_input": list(DEVELOPMENT_INPUT),
                      "confirmation_arch": list(CONFIRMATION_ARCH[:5]),
                      "confirmation_input": list(CONFIRMATION_INPUT[:3]),
                      "note": "confirmation seeds are untouched until CONFIRMATION"},
    }
    fc = freeze(payload, FROZEN_PATH)
    _write("manifest_freeze.json", _manifest("FREEZE", t0, {"sha256": fc.sha256}))
    print(f"  frozen sha256 = {fc.sha256}")
    print(f"  components: {sorted(payload)}")
    return {"stage": "FREEZE", **fc.as_dict()}


def stage_confirmation() -> dict:
    """CONFIRMATION -- one untouched run on the frozen configuration."""
    t0 = time.time()
    fc = load_frozen(FROZEN_PATH)                       # re-hashes; raises if edited
    cand = fc.payload["architecture"]["candidate"]
    spec, cfg = build(cand)

    arch = [int(s) for s in fc.payload["seed_bank"]["confirmation_arch"]]
    inp = [int(s) for s in fc.payload["seed_bank"]["confirmation_input"]]
    assert_confirmation_only(arch, inp)                 # refuses a development seed

    ledger = SequentialLedger.load(LEDGER_PATH)
    alpha_this = ledger.alpha_for_next()
    print(f"  frozen sha256 {fc.sha256[:16]}...  attempt {ledger.n_attempts()+1}"
          f"/{ledger.max_attempts}  alpha_this_look={alpha_this}")
    print(f"  confirmation seeds arch={arch} input={inp} (never seen by the search)")

    ck = Checkpoint(RESULTS / "confirmation_rows.jsonl")
    done = ck.done_keys()
    expected = [f"a{a}_i{i}_m{m}_g{g}" for a in arch for i in inp
                for m in CONF_M for g in CONF_G]
    print(f"  grid {len(CONF_M)}x{len(CONF_G)} x {len(arch)} arch x {len(inp)} input "
          f"= {len(expected)} rows ({len(done)} already done)")

    def progress(row):
        key = f"a{row['arch_seed']}_i{row['input_seed']}_m{row['m']}_g{row['g']}"
        if key not in done:
            ck.append(key, {**row, "config_hash": fc.sha256})

    grid = evaluate_grid(spec, cfg, m_values=CONF_M, g_values=CONF_G,
                         arch_seeds=arch, input_seeds=inp, T=int(fc.payload["grid"]["T"]),
                         with_nm=True, progress=progress)
    completeness = verify_complete(ck.rows(), expected,
                                   required_fields=("M", "N", "N_long"),
                                   config_hash=fc.sha256)

    u_probe = np.random.default_rng(int(inp[0])).uniform(-1, 1, 40)
    enc_N = float(np.mean([r["N"] for r in grid["rows"] if r["g"] == 0.0]))
    gt = evaluate_all_gates(grid, spec, cfg, u_probe=u_probe, encoder_only_N=enc_N, seed=99)

    guard = ClaimGuard(gates=gt["gates"], confirmation_complete=completeness["complete"],
                       frozen_hash=fc.sha256, run_config_hash=fc.sha256,
                       sequential_valid=(ledger.n_attempts() < ledger.max_attempts))
    ledger.record(version=VERSION, seed_bank=f"arch={arch},input={inp}",
                  passed=bool(gt["all_passed"] and completeness["complete"]),
                  summary={"n_pass": gt["n_pass"], "n_fail": gt["n_fail"]})

    nm = [r["nm_summary"] for r in grid["rows"] if "nm_summary" in r]
    out = {"stage": "CONFIRMATION", "frozen_sha256": fc.sha256,
           "seeds": {"arch": arch, "input": inp},
           "grid_means": {f"m{m}_g{g}": v for (m, g), v in mean_grid(grid).items()},
           "gates": gt["gates"], "jacobian": gt["jacobian"],
           "all_passed": gt["all_passed"], "n_pass": gt["n_pass"],
           "n_fail": gt["n_fail"], "n_not_evaluable": gt["n_not_evaluable"],
           "completeness": completeness,
           "nonlinear_memory": {
               "mean_reachable": float(np.mean([x["mean_reachable"] for x in nm])) if nm else None,
               "mean_boundary": float(np.mean([x["mean_boundary"] for x in nm])) if nm else None,
               "max_boundary": float(np.max([x["max_boundary"] for x in nm])) if nm else None},
           "sequential": ledger.as_dict(),
           "claim": guard.render(), "claim_detail": guard.as_dict()}
    _write("confirmation.json", out)
    _write("manifest_confirmation.json", _manifest("CONFIRMATION", t0))
    print(f"\n  gates: {gt['n_pass']} pass / {gt['n_fail']} fail / "
          f"{gt['n_not_evaluable']} not-evaluable   complete={completeness['complete']}")
    print(f"  CLAIM: {out['claim']}")
    return out


def stage_report() -> dict:
    """REPORT -- assemble from saved artifacts only; never recompute."""
    t0 = time.time()
    from decoupled_qrc.v4_report import build_report
    out = build_report(RESULTS)
    _write("manifest_report.json", _manifest("REPORT", t0,
                                             {"figures": out.get("n_rendered")}))
    print(f"  report.md + report.json written; figures {out.get('n_rendered')} "
          f"({out.get('n_skipped')} skipped)")
    return out


STAGES = {"AUDIT": stage_audit, "SMOKE": stage_smoke, "CALIBRATION": stage_calibration,
          "ARCHITECTURE_SEARCH": stage_search, "STRESS_TEST": stage_stress,
          "FREEZE": stage_freeze, "CONFIRMATION": stage_confirmation,
          "REPORT": stage_report}


def main():
    ap = argparse.ArgumentParser(description="V4 stage runner")
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--n-candidates", type=int, default=24)
    args = ap.parse_args()
    print(f"=== V4 {args.stage} ===")
    if args.stage == "ARCHITECTURE_SEARCH":
        STAGES[args.stage](n_candidates=args.n_candidates)
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
