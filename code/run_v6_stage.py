"""
run_v6_stage.py -- V6: general memory / nonlinearity / nonlinear-memory separation.

    python run_v6_stage.py --stage STEP1                 # lock + reproduce the V5.4 baseline
    python run_v6_stage.py --stage PREREG_GATES          # write + hash the V6 gate plan
    python run_v6_stage.py --stage ALGEBRA               # feature-span proofs (V5.4 + candidates)
    python run_v6_stage.py --stage DEVGATES --version V6.0
    python run_v6_stage.py --stage FREEZE   --version V6.0
    python run_v6_stage.py --stage CONFIRM  --version V6.0
    python run_v6_stage.py --stage REPORT   --version V6.0

Nothing from V4 / V5 is modified. The V4 audit's IPC primitives (audit_ipc) and
point evaluator (audit_core) are imported, never edited: both are hashed in the
V4 preregistration and the V5.4 freeze.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_v4_audit as W                              # noqa: E402
from decoupled_qrc import audit_core as K            # noqa: E402

OUT = ROOT / "results" / "v6"


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=W._js), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def _leaves(o, prefix=""):
    """Flatten every numeric leaf of a row's blocks (for bitwise comparison)."""
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _leaves(o[k], f"{prefix}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _leaves(v, f"{prefix}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield prefix, float(o)


def compare_rows(a: list, b: list) -> dict:
    ka = {r["key"]: r for r in a}
    kb = {r["key"]: r for r in b}
    common = sorted(set(ka) & set(kb))
    worst, n_exact, n_leaf, where = 0.0, 0, 0, None
    for k in common:
        la = dict(_leaves(ka[k]["blocks"]))
        lb = dict(_leaves(kb[k]["blocks"]))
        for name, va in la.items():
            vb = lb.get(name, np.nan)
            n_leaf += 1
            if va == vb or (np.isnan(va) and np.isnan(vb)):
                n_exact += 1
                continue
            d = abs(va - vb)
            if not d <= worst:
                worst, where = d, (k, name)
    return {"rows_a": len(a), "rows_b": len(b), "rows_compared": len(common),
            "numeric_leaves": n_leaf, "bit_identical_leaves": n_exact,
            "max_abs_diff": worst, "argmax": where}


# =============================================================================
# STEP 1 -- reproduce and lock the V5.4 baseline
# =============================================================================
def stage_step1():
    import run_v5_stage as S
    t0 = time.time()
    out = OUT / "step1"
    res = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip()}
    # 1. frozen-file and amendment history
    pre = W.verify_prereg()
    res["v4_audit_prereg_verified"] = pre["sha256"]
    S.configure("V5.4")
    blob = S.verify_frozen()
    res["v5_4_frozen_verified"] = blob["sha256"]
    log = subprocess.run(["git", "log", "--format=%h %cI %s", "--", "results/v5_4/frozen_v5_4.json",
                          "results/v5_4/amendment_01.json", "results/v5_4/gates.json"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    res["v5_4_history"] = log[::-1]
    am = json.loads((S.OUT / "amendment_01.json").read_text(encoding="utf-8"))
    res["v5_4_amendment_scope"] = am["scope"]

    # 2. independent regeneration of the 500 confirmation rows
    p = blob["payload"]
    ad = S.V.V5Adapter(S.spec_of(p["spec"]), alpha=p["ridge_alpha"])
    ctx = K.Context(alpha=p["ridge_alpha"])
    rows = K.run_factorial(ad, m_values=S.GRID, g_values=S.GRID, seeds=S.CONF_SEEDS, ctx=ctx,
                           checkpoint_path=out / "v5_4_repro_rows.jsonl", tag="conf")
    stored = W.load_rows(S.OUT / "conf_rows.jsonl")
    res["row_comparison"] = compare_rows(stored, rows)

    # 3. gates recomputed from the regenerated rows, compared with the stored verdicts
    st = json.loads((S.OUT / "structure_confirmation.json").read_text(encoding="utf-8"))
    s6 = json.loads((S.OUT / "section6_confirmation.json").read_text(encoding="utf-8"))
    g_new = W.all_gates(rows, K.Levels.sequential(), s6=s6, st=st, lows=(0.0, 0.25),
                        highs=(0.75, 1.0))
    g_old = json.loads((S.OUT / "gates.json").read_text(encoding="utf-8"))["gates_sequential"]
    res["gate_verdicts_reproduced"] = {k: {"stored": g_old[k]["passed"], "now": g_new[k]["passed"]}
                                       for k in g_old if k.startswith("C")}
    est = lambda g: {  # noqa: E731
        "dM_ridge": g["C2_resource_constrained"]["main_m_M"]["estimate"],
        "dN_ridge": g["C2_resource_constrained"]["main_g_N"]["estimate"],
        "interior_m_M": g["C3_intrinsic"]["interior_m_M"]["estimate"],
        "interior_g_N": g["C3_intrinsic"]["interior_g_N"]["estimate"],
        "C1_HH": g["C4_combined"]["classes"]["C1_curNL_x_oldLin"]["ridge"]["quadrant_means"]["HH"]}
    res["principal_estimates"] = {"stored": est(g_old), "now": est(g_new)}
    res["all_verdicts_match"] = all(v["stored"] == v["now"]
                                    for v in res["gate_verdicts_reproduced"].values())

    # 4. BLAS-thread dependence: two rows under 1 thread vs the default pool
    res["thread_dependence"] = {}
    for nt in ("1", "4"):
        env = {**os.environ, "OPENBLAS_NUM_THREADS": nt, "OMP_NUM_THREADS": nt}
        cp = out / f"v5_4_repro_threads{nt}.jsonl"
        code = ("import sys; sys.path.insert(0, '.'); import run_v5_stage as S; "
                "from decoupled_qrc import audit_core as K; S.configure('V5.4'); "
                "b = S.verify_frozen()['payload']; "
                "ad = S.V.V5Adapter(S.spec_of(b['spec']), alpha=b['ridge_alpha']); "
                f"K.run_factorial(ad, m_values=(1.0,), g_values=(0.0, 1.0), seeds=S.CONF_SEEDS[:2], "
                f"ctx=K.Context(alpha=b['ridge_alpha']), checkpoint_path=r'{cp}', tag='conf', "
                "verbose=False)")
        subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, check=True)
        res["thread_dependence"][f"threads_{nt}"] = compare_rows(stored, W.load_rows(cp))
    res["runtime_s"] = round(time.time() - t0, 1)
    write(out / "v5_4_reproduction.json", res)
    rc = res["row_comparison"]
    print(f"  rows {rc['rows_compared']}/500  bit-identical leaves {rc['bit_identical_leaves']}/"
          f"{rc['numeric_leaves']}  max|diff| {rc['max_abs_diff']:.3e}")
    print(f"  verdicts reproduced: {res['all_verdicts_match']}  "
          f"threads: " + ", ".join(f"{k} max|diff| {v['max_abs_diff']:.2e}"
                                   for k, v in res["thread_dependence"].items()))


# =============================================================================
# PREREG_GATES -- written and hashed BEFORE any V6 development experiment
# =============================================================================
DEV_SEEDS = [(100000 + k, 105000 + k) for k in range(10)]


def conf_seeds(attempt: int) -> list:
    base = 110000 + 10000 * (attempt - 1)
    return [(base + j, base + 5000 + j) for j in range(20)]


PREREG = OUT / "preregistered_v6_gates.json"
PREREG_SOURCES = ["code/decoupled_qrc/v6_core.py", "code/decoupled_qrc/v6_gates.py",
                  "code/decoupled_qrc/audit_ipc.py"]


def used_seeds() -> dict:
    import run_v5_stage as S
    return {"v4_original": W.USED_BEFORE, "v4_audit_dev": W.DEV_SEEDS, "v4_audit_conf": W.CONF_SEEDS,
            "v5_dev": S.DEV_SEEDS, "v5_4_conf": S.CONF_SEEDS, "v6_dev": DEV_SEEDS,
            "v6_algebra": [990001, 990002]}


def stage_prereg_gates():
    from decoupled_qrc import v6_gates as G6
    if PREREG.exists():
        raise SystemExit("V6 gate preregistration already exists; it is never rewritten")
    plan = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "every V6.x candidate; written before any V6 development experiment",
        "data": {"input": "u_t ~ i.i.d. U[-1,1], one sequence per seed", "T": 1600,
                 "split": "washout 60, 65% train, gap 12, rest test (chronological)",
                 "grid": {"m": [0, .25, .5, .75, 1], "g": [0, .25, .5, .75, 1]},
                 "quadrants": {"L": list(G6.LOWS), "H": list(G6.HIGHS)},
                 "readouts": {"ridge": "standardised, alpha = 0.5 (V4/V5 value, never tuned)",
                              "ols_std": "standardised, pseudo-inverse, rel. cutoff 1e-10",
                              "ols_raw": "centred raw, pseudo-inverse, rel. cutoff 1e-10"},
                 "capacity": "C = 1 - MSE/Var on held-out block, NOT clipped",
                 "null": "120 independent-sequence Legendre targets per point; 99th pct"},
        "metrics": {"M": "R-local, mean_{tau=2..8} C[P1(u_{t-tau})]",
                    "N": "P-local, mean_{d=2,3,4} C[P_d(u_t)]",
                    "classes": {"V6_C1": "P2(u_t) P1(u_{t-tau}), tau 1..6",
                                "V6_C2": "P2(u_{t-tau}), tau 1..6",
                                "V6_C3": "P1(u_{t-t1}) P1(u_{t-t2}), 1 <= t1 < t2 <= 6 (15)",
                                "V6_C4": "P3(u_{t-tau}), tau 1..6",
                                "class_score": "mean member capacity on ALL = R+P+J+Q"},
                    "sentinels": {"future": "P1(u_{t+1}), P1(u_{t+2})",
                                  "unreachable": "P5(u_{t-tau}), tau 0..3 (> 3 copies of one input)"}},
        "gates": {
            "01/02 main effects": "m->M and g->N >= 0.10, bootstrap LB(alpha) > 0, and >= 90% of "
                                  "seeds individually >= 0.10; ridge AND ols_std AND ols_raw",
            "03/04 cross effects": "TOST interval [alpha, 1-alpha] inside +-0.03, every readout",
            "05 ratio": "|cross|/main upper bound (1-alpha) < 0.20, both directions, every readout",
            "06 degree profile": "C[P_d(u_t)], d = 2,3,4, under m: TOST, Bonferroni alpha/3",
            "07 memory curve": "C[P1(u_{t-tau})], tau 0..8, under g: TOST, Bonferroni alpha/9",
            "08-10 separation": "01-07 all hold for raw OLS, standardised OLS, fixed ridge",
            "11 interior": "M(1)-M(0.25) and N(1)-N(0.25) as 01/02, every readout; P section-6 "
                           "verdict HOLDS",
            "12 combined": "EVERY class C1..C4 and EVERY readout: (1) best quadrant > null99 + 0.02; "
                           "(2) HH > LL, LH, HL; (3) each HH contrast LB > 0 at alpha/36 "
                           "(Bonferroni over 4 classes x 3 contrasts x 3 readouts); (4) all "
                           "readouts; (5) <= 20% of members > 0.995 in every quadrant and class "
                           "mean <= 0.95; (6) HH - max(other) > 0 in >= 90% of seeds; (7) "
                           "sentinels at null",
            "13 HH best": "HH point estimate best AND contrasts pass, every class and readout",
            "14 feature counts": "identical at every (m, g), every readout",
            "15 encoder leakage": "at g = 0 every operational feature is EXACTLY affine in the "
                                  "input history (superposition test, tol 1e-10) and no nonlinear "
                                  "class exceeds null99 + 0.02 at g = 0",
            "16 saturation": "per metric (M, N): <= 20% of constituents > 0.995 at every control "
                             "value and metric <= 0.95; pooled <= 20%; per class as in 12(5)",
            "17/18 sentinels": "future and unreachable: max member - null99 <= 0.02, every row",
            "19 isolation": "dX_R/dg = dX_P/dm = 0 exactly over 100 random draws + full-range; "
                            "g->R, m->P, serial contaminations detected; M bit-identical over g "
                            "and N over m",
            "20 invalid controls": "target leakage, feature-count change, train-as-test detected",
            "21 seeds": "as in 01/02 and 12(6)",
            "22-25 robustness": "longer T, half/double training, float32, perturbed controls, "
                                "+-10% continuous parameters, ridge 0.05/5, delay range 12, "
                                "degree 6, 1000 and 10000 shots, FakeTorino-informed noise: every "
                                "main and interior(0.5->1) effect >= 0.10, cross exactly 0, every "
                                "class/readout HH - max(other) > 0, no metric saturation "
                                "(noiseless conditions); run on dev AND on confirmation seeds",
            "26 unit tests": "complete suite passes"},
        "thresholds": G6.THRESH,
        "dev_freeze_margins": G6.DEV_MARGIN,
        "error_control": {
            "ledger_before_v6": [{"look": 1, "what": "V4 original confirmation", "alpha": 0.01},
                                 {"look": 2, "what": "V4 independent-audit confirmation", "alpha": 0.01},
                                 {"look": 3, "what": "V5.4 confirmation", "alpha": 0.01}],
            "v6_attempt_k_alpha": "0.01 / 2^(k-1)  (0.01, 0.005, 0.0025, ...; sum <= 0.02; "
                                  "cumulative over V4, V5 and V6 <= 0.05)",
            "bootstrap": "100 000 seed-level resamples"},
        "seeds": {"v6_dev": DEV_SEEDS, "confirmation_attempt_k": "(110000 + 10000(k-1) + j, "
                  "115000 + 10000(k-1) + j), j < 20", "previously_used": used_seeds()},
        "quantum_specific_claim": {
            "status": "reported separately; NOT part of the V6 mandatory objective",
            "pass_rule": "(a) ALL beats BOTH classical baselines (CLS_M matched budget: marginals + "
                         "the product of marginals for each quantum joint observable; CLS_F: "
                         "marginals + every same-time degree-2 product) on the HH mean of the four "
                         "class scores with LB(alpha) > 0, AND (b) the mechanism is not "
                         "classically simulable: a register that stays diagonal in a fixed "
                         "product basis is a classical Markov chain and FAILS (b) by definition",
            "fail_statement": "General memory-nonlinearity separation succeeds, but the mechanism "
                              "is classically reproducible."},
        "implementation_of_record": {f: W.sha_file(ROOT / f) for f in PREREG_SOURCES},
        "amendment_rule": "implementation bugs may be fixed before a confirmation via a hashed "
                          "amendment in results/v6/amendments/; criteria and thresholds never",
    }
    write(PREREG, plan)
    h = W.sha_file(PREREG)
    PREREG.with_suffix(".sha256").write_text(h + "\n", encoding="utf-8")
    print(f"  V6 gate preregistration sha256 {h}")


STAGES = {"STEP1": stage_step1, "PREREG_GATES": stage_prereg_gates}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--version", default=None)
    a = ap.parse_args()
    print(f"=== V6 {a.stage} {a.version or ''} ===")
    STAGES[a.stage]() if a.version is None else STAGES[a.stage](a.version)
