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
import itertools
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
            "v6_sentinel_calibration": [s for b in sorted(set(CAL_BLOCK.values())) for s in cal_seeds(b)],
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


# =============================================================================
# Candidate registry (append-only) and progress
# =============================================================================
REGISTRY = OUT / "candidate_registry.jsonl"
PROGRESS = ROOT / "results" / "v6_progress.json"


def v6_versions() -> dict:
    """Every V6 candidate, specified BEFORE it is evaluated. Never edited after use."""
    from decoupled_qrc.v6_architecture import V6Spec
    return {
        "V6.0": {"spec": V6Spec(),
                 "rationale": "R = V5.4 memory (L16, stride 2, p 0.7); P = 3-copy processor, one "
                              "observable, degrees 1-3 mixed by g (theta 0.38 pi, chi 0.30 pi, "
                              "phi pi/3); J = z^R x Y-channel (0 at g = 0) for C1; Q = second "
                              "random-SWAP register (L 12, all rails read, p 0.7) written with the "
                              "processor output (C2, C4) and read by 10 g-rotated pair "
                              "correlators (C3)."},
        # V6.0 rejected at PROBE: dN 0.027 (P1 part of u^3 dominates); P2 content of the Q
        # write non-monotone in g. V6.1 = exact-quadrature redesign (no data used):
        "V6.1": {"spec": V6Spec(theta_max=0.12 * np.pi, chi_max=-0.30 * np.pi, phi=np.pi / 4,
                                joint="rotated", thetaJ_max=0.40 * np.pi, phiJ=np.pi / 4,
                                thetaW_max=0.14 * np.pi, chiW_max=-0.24 * np.pi, phiW=np.pi / 4),
                 "rationale": "P: chi_max < 0 cancels the P1 part of u^3 at g = 1 (design scan "
                              "results/v6/algebra/V6.1_processor_design_scan.json; rule: best "
                              "worst-case interior N among robust designs with C3(1) >= 0.25) -> "
                              "exact C1/C2/C3 at g=1 = 0.00/0.60/0.33. J: g-rotated joint readout "
                              "of R rail x P output (affine at g = 0, product weight sin(theta_J)). "
                              "Q writer: separate processor keeping a linear part for old x old "
                              "products (only robust design in V6.1_writer_design_scan.json: "
                              "C1/C2/C3 = 0.40/0.47/0.13 at g=1, C2 and C3 monotone in g)."},
        # V6.1 rejected at PROBE: C3 HH 0.013 < HL 0.082 -- products of the nonlinear write
        # lose their P1P1 part as g grows (structural). V6.2 takes old x old from g-rotated
        # pair readouts on the LINEAR register R (clean u_i u_j), drops the Q-register pairs.
        "V6.2": {"spec": V6Spec(theta_max=0.12 * np.pi, chi_max=-0.30 * np.pi, phi=np.pi / 4,
                                joint="rotated", thetaJ_max=0.40 * np.pi, phiJ=np.pi / 4,
                                thetaW_max=0.14 * np.pi, chiW_max=-0.24 * np.pi, phiW=np.pi / 4,
                                q_pairs=(),
                                r_pairs=tuple(itertools.combinations(range(1, 7), 2)) + ((3, 7), (5, 7)),
                                thetaRP_max=0.40 * np.pi, phiRP=np.pi / 4),
                 "rationale": "V6.1 + C3 from 17 g-rotated pair readouts of R rails (all pairs "
                              "of rails 1..6 plus (3,7), (5,7); chosen by long-sequence algebra "
                              "on the algebra seed: C3 mean 0.81 at (1,1), 0.61 at (.75,.75)). "
                              "M still uses ONLY R single-site Z; pair settings are extra "
                              "measurement settings on the same register, assigned to route Q."},
        # V6.2 rejected at DEVGATES (freeze margins): (i) C4 capacity FELL with g (writer cubic
        # share saturates early; algebra 0.33 -> 0.21); (ii) under exact expectations OLS
        # recovers any nonzero product coefficient, so C2/C3 are switched on at g = 0+
        # (perturbed grid g = 0.03: C3 HH - HL = -0.001). Long-sequence algebra (no dev data):
        # a finite shot budget grades every class monotonically in g
        # (S = 1e4: C3 0.007 / 0.28 / 0.45 / 0.57 at g = .03 / .25 / .5 / .97).
        "V6.3": {"spec": V6Spec(theta_max=0.12 * np.pi, chi_max=-0.30 * np.pi, phi=np.pi / 4,
                                joint="rotated", thetaJ_max=0.40 * np.pi, phiJ=np.pi / 4,
                                thetaW_max=0.14 * np.pi, chiW_max=-0.24 * np.pi, phiW=np.pi / 4,
                                chiW_power=2, q_pairs=(),
                                r_pairs=tuple(itertools.combinations(range(1, 7), 2)) + ((3, 7), (5, 7)),
                                thetaRP_max=0.40 * np.pi, phiRP=np.pi / 4, shots=10000),
                 "rationale": "V6.2 + writer cubic angle chi_W = g^2 chi_W_max (C4 grows with g) + "
                              "a declared readout budget of 10 000 shots per setting, identical at "
                              "every (m, g), for every feature and for the classical baselines. "
                              "DISCLOSED: with exact expectations (infinite shots) C2 and C3 are "
                              "switched on by any g > 0; their gradation in g is a finite-statistics "
                              "effect. The primary N control is intrinsic (single-feature "
                              "composition). The exact-expectation perturbed-controls grid is run "
                              "and reported as a non-gating diagnostic."},
        # V6.3 rejected at DEVGATES: gate 17 future sentinel 0.0257 > 0.02 (one row) -- chance
        # fits of 46 full-rank noisy features; 500-row dev calibration: 2/500 rows exceed.
        # V6.4 cuts the operational readout to 22 features (calibration 0/500 at K 27; K 22
        # chosen by long-sequence algebra, design Y) and keeps every other V6.3 choice.
        "V6.4": {"spec": V6Spec(L_R=12, theta_max=0.12 * np.pi, chi_max=-0.30 * np.pi, phi=np.pi / 4,
                                joint="rotated", thetaJ_max=0.40 * np.pi, phiJ=np.pi / 4,
                                J_rails=(2, 4, 6),
                                thetaW_max=0.14 * np.pi, chiW_max=-0.24 * np.pi, phiW=np.pi / 4,
                                chiW_power=2, L_Q=6, q_pairs=(),
                                r_pairs=((1, 2), (1, 3), (2, 4), (3, 5), (2, 5), (4, 6)),
                                thetaRP_max=0.40 * np.pi, phiRP=np.pi / 4, shots=10000),
                 "rationale": "V6.3 with the operational readout reduced from 46 to 22 features "
                              "(R: L 12 stride 2 -> 6; P 1; J on rails 2,4,6 -> 3; Q register L 6 "
                              "all read -> 6; 6 R pairs) so that OLS chance fits stay inside the "
                              "preregistered sentinel margin at confirmation size. Long-sequence "
                              "algebra at (1, .97): C1 .15, C2 .55, C3 .20, C4 .16; all ~0 at "
                              "g = .03. New dev freeze margin: sentinel calibration on 500 extra dev "
                              "rows, no exceedance and max <= 0.016. Same disclosed limitation as "
                              "V6.3 (C2/C3 gradation is a finite-shot effect)."},
        # V6.4 rejected at SENTCAL: max excess 0.0168 > 0.016 margin (no row > 0.02).
        "V6.5": {"spec": V6Spec(L_R=12, theta_max=0.12 * np.pi, chi_max=-0.30 * np.pi, phi=np.pi / 4,
                                joint="rotated", thetaJ_max=0.40 * np.pi, phiJ=np.pi / 4,
                                J_rails=(2, 4),
                                thetaW_max=0.14 * np.pi, chiW_max=-0.24 * np.pi, phiW=np.pi / 4,
                                chiW_power=2, L_Q=5, q_pairs=(),
                                r_pairs=((1, 2), (1, 3), (2, 4), (3, 5)),
                                thetaRP_max=0.40 * np.pi, phiRP=np.pi / 4, shots=10000),
                 "rationale": "V6.4 reduced to 18 operational features (R 6, P 1, J rails 2,4 -> 2, "
                              "Q register L 5 -> 5, 4 R pairs); chance-fit tail ~ sqrt(K) -> expected "
                              "max ~0.015. Algebra (T 12000, S 1e4): M(.25) .12, M(1) .50; classes "
                              "at (1,.97) C1 .13 C2 .49 C3 .15 C4 .15, at g .03 all <= .006. "
                              "Calibrated on a FRESH dev block (block 1)."},
    }


def registry() -> list:
    return W.load_rows(REGISTRY)


def register(version: str, event: str, **info):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "version": version,
           "event": event, **info}
    with open(REGISTRY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=W._js) + "\n")


def progress(version: str, stage: str, status: str, diagnosis: str, next_command: str):
    p = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {"history": []}
    p["current"] = {"version": version, "stage": stage, "status": status, "diagnosis": diagnosis,
                    "next_command": next_command,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p["history"].append(p["current"])
    PROGRESS.write_text(json.dumps(p, indent=2), encoding="utf-8")


def ensure_specified(version: str):
    from decoupled_qrc.v6_architecture import V6Adapter
    if any(r["version"] == version and r["event"] == "specified" for r in registry()):
        return
    v = v6_versions()[version]
    ad = V6Adapter(v["spec"])
    register(version, "specified", spec=v["spec"].as_dict(), rationale=v["rationale"],
             resources=ad.describe()["resources"],
             source_sha256={f: W.sha_file(ROOT / f) for f in V6_SOURCES})


V6_SOURCES = ["code/decoupled_qrc/v6_architecture.py", "code/decoupled_qrc/v6_core.py",
              "code/decoupled_qrc/v6_gates.py", "code/decoupled_qrc/v6_algebra.py",
              "code/decoupled_qrc/v5_architecture.py", "code/decoupled_qrc/audit_ipc.py",
              "code/decoupled_qrc/audit_checks.py", "code/run_v6_stage.py"]


# =============================================================================
# ALGEBRA -- feature-span reachability before any expensive simulation
# =============================================================================
def stage_algebra(version: str):
    from decoupled_qrc import v6_algebra as ALG
    from decoupled_qrc.v6_architecture import V6Adapter
    out = OUT / "algebra"
    if version == "V5.4":
        import run_v5_stage as S
        S.configure("V5.4")
        spec = S.spec_of(S.verify_frozen()["payload"]["spec"])
        res = {"version": "V5.4", "reachability_m1_g1": ALG.reachability(ALG.v5_4_operational(spec)),
               "theorem": "see v6_algebra module docstring: every V5.4 feature is linear in each "
                          "past input and contains at most one past input -> C2, C3, C4 have "
                          "capacity exactly 0"}
        write(out / "V5.4.json", res)
        for c in ("V6_C1", "V6_C2", "V6_C3", "V6_C4"):
            r = res["reachability_m1_g1"][c]
            print(f"  V5.4 {c}: {r['status']:13s} max {r['max']:+.4f}")
        return
    ensure_specified(version)
    ad = V6Adapter(v6_versions()[version]["spec"])
    res = {"version": version,
           "reachability_m1_g1": ALG.reachability(ALG.v6_operational(ad, 1.0, 1.0)),
           "affine_at_g0": {f"m{m}": ALG.exact_affinity(ALG.v6_operational(ad, m, 0.0))
                            for m in (0.25, 0.5, 1.0)},
           "affine_at_g1": ALG.exact_affinity(ALG.v6_operational(ad, 1.0, 1.0)),
           "reachability_m0_g1": ALG.reachability(ALG.v6_operational(ad, 0.0, 1.0)),
           "reachability_m1_g0": ALG.reachability(ALG.v6_operational(ad, 1.0, 0.0))}
    r = res["reachability_m1_g1"]
    ok = bool(r["all_four_reachable"] and r["sentinel_unreachable_null"]
              and all(v["affine"] for v in res["affine_at_g0"].values())
              and not res["affine_at_g1"]["affine"])
    res["passed"] = ok
    write(out / f"{version}.json", res)
    for c in ("V6_C1", "V6_C2", "V6_C3", "V6_C4", "SENTINEL_unreachable"):
        print(f"  {version} {c:22s} m1g1 {r[c]['status']:13s} max {r[c]['max']:+.4f} | "
              f"m0g1 {res['reachability_m0_g1'][c]['max']:+.4f} | m1g0 {res['reachability_m1_g0'][c]['max']:+.4f}")
    print(f"  affine at g=0: {[v['affine'] for v in res['affine_at_g0'].values()]}  "
          f"at g=1: {res['affine_at_g1']['affine']}  -> ALGEBRA {'PASS' if ok else 'FAIL'}")
    register(version, "algebra", passed=ok,
             summary={c: r[c]["status"] for c in ("V6_C1", "V6_C2", "V6_C3", "V6_C4",
                                                  "SENTINEL_unreachable")})
    if not ok:
        register(version, "rejected", stage="ALGEBRA", reason="required class unreachable or "
                 "g = 0 not affine / g = 1 affine")


# =============================================================================
# Development
# =============================================================================
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ROBUST_GRID = (0.0, 0.5, 1.0)
NOISE_READOUTS = {k: ("ridge", "ols_std", "ols_raw") for k in ("R", "P", "J", "Q", "ALL")}


def vdir(version: str) -> Path:
    return OUT / version.replace(".", "_")


def make_adapter(spec, kind=None, shots=None):
    """The physical architecture: exact expectations, or S-shot estimates if the spec
    declares a shot budget (shots overrides it for robustness conditions)."""
    from decoupled_qrc.v6_architecture import V6Adapter, V6ShotAdapter
    S = spec.shots if shots is None else shots
    base = V6Adapter(spec, kind=kind)
    return V6ShotAdapter(base, S) if S else base


def adapter_for(version: str, kind=None):
    return make_adapter(v6_versions()[version]["spec"], kind=kind)


def structure(version: str, seed: int) -> dict:
    from decoupled_qrc import audit_checks as C
    make = lambda kind: adapter_for(version, kind)          # noqa: E731
    return {k or "frozen": C.structural_dependency_adapter(make, n_draws=100, seed=seed, kind=k)
            for k in (None, "g_into_R", "m_into_P", "serial")}


def invalid_controls(version: str, ad, ctx, st, seeds) -> dict:
    from decoupled_qrc import audit_checks as C
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc.v6_gates import sentinels
    out = {k: {"detected": not st[k]["isolated"], "detector": "structural dependency test"}
           for k in ("g_into_R", "m_into_P", "serial")}
    rows = K6.run_factorial(ad, m_values=(0.0, 1.0), g_values=(0.0, 1.0), seeds=seeds, ctx=ctx,
                            checkpoint_path=vdir(version) / "invalid_leak_targets.jsonl",
                            tag="leak", feature_hook=C.leak_hook("leak_targets"), verbose=False,
                            classical=False, readouts=NOISE_READOUTS)
    c2 = max(r["blocks"]["R|ridge"]["class"]["V6_C2__max"] - r["blocks"]["R|ridge"]["null_q99"]
             for r in rows)
    sen = sentinels(rows)
    out["leak_targets"] = {"detected": bool(c2 > 0.02 or not sen["SENTINEL_future"]["ok"]),
                           "detector": "R-only delayed-nonlinear class + future sentinel",
                           "evidence": {"R_C2_excess": c2, "future": sen["SENTINEL_future"]}}
    rows = K6.run_factorial(ad, m_values=(0.0, 1.0), g_values=(0.0, 1.0), seeds=seeds, ctx=ctx,
                            checkpoint_path=vdir(version) / "invalid_feature_count.jsonl",
                            tag="fc", feature_hook=C.leak_hook("feature_count_changes"),
                            verbose=False, classical=False, readouts=NOISE_READOUTS)
    from decoupled_qrc.v6_gates import feature_count_invariance
    out["feature_count_changes"] = {"detected": not feature_count_invariance(rows)["invariant"],
                                    "detector": "feature-count invariance"}
    bad = type(ctx.split)(ctx.split.train, ctx.split.train)
    out["reuse_train_as_test"] = {"detected": not bad.check()["disjoint"],
                                  "detector": "split integrity"}
    out["all_detected"] = bool(all(v["detected"] for v in out.values()))
    return out


def robustness_conditions(version: str) -> dict:
    """name -> (adapter, ctx, grid, noiseless)"""
    from decoupled_qrc import audit_checks as C
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc.v6_architecture import V6Adapter, V6ShotAdapter
    spec = v6_versions()[version]["spec"]
    ad = make_adapter(spec)
    G = ROBUST_GRID
    nl = not spec.shots          # saturation is only required of noiseless conditions
    c = {"baseline": (ad, K6.Ctx(), G, nl),
         "longer_T3200": (ad, K6.Ctx(T=3200), G, nl),
         "half_training": (ad, K6.Ctx(T=1110, train_frac=550 / 1110), G, nl),
         "double_training": (ad, K6.Ctx(T=2580, train_frac=2020 / 2580), G, nl),
         "float32_readout": (ad, K6.Ctx(dtype="float32"), G, nl),
         "perturbed_controls": (ad, K6.Ctx(), (0.03, 0.47, 0.97), nl),
         "ridge_0.05": (ad, K6.Ctx(alpha=0.05), G, nl),
         "ridge_5.0": (ad, K6.Ctx(alpha=5.0), G, nl),
         "delay_range_12": (ad, K6.Ctx(tau_max=12), G, nl),
         "degree_6": (ad, K6.Ctx(max_degree=6), G, nl),
         "shots_1000": (make_adapter(spec, shots=1000), K6.Ctx(), G, False),
         "shots_10000": (make_adapter(spec, shots=10000), K6.Ctx(), G, False)}
    if spec.shots:
        c["shots_100000"] = (make_adapter(spec, shots=100000), K6.Ctx(), G, False)
    fields = ["pR_max", "theta_max", "chi_max", "phi", "pQ_max", "thetaQ_max", "phiQ"]
    if spec.joint == "rotated":
        fields += ["thetaJ_max", "phiJ"]
    fields += [f for f in ("thetaW_max", "chiW_max", "phiW") if getattr(spec, f) is not None]
    for field in fields:
        for f in (0.9, 1.1):
            val = getattr(spec, field) * f
            if field in ("pR_max", "pQ_max"):
                val = min(val, 1.0)
            c[f"{field}_x{f}"] = (ad.with_spec(**{field: val}), K6.Ctx(), G, not spec.shots)
    ibm = C.ibm_noise_parameters("FakeTorino")
    if ibm.get("available"):
        e1, e2, ro = ibm["median_sx_error"], ibm["median_2q_error"], ibm["median_readout_error"]
        noisy = V6Adapter(spec, depol_R=min(0.5, 4 * e1 + 3 * e2), depol_Q=min(0.5, 4 * e1 + 3 * e2),
                          depol_P=min(0.5, 3 * e1 + 6 * e2))
        c["ibm_noise_FakeTorino"] = (V6ShotAdapter(noisy, 4096, readout_error=ro), K6.Ctx(), G, False)
    return c


DIAGNOSTICS = ("exact_expectations_perturbed_controls",)


def diagnostic_conditions(version: str) -> dict:
    """Reported, NOT gating: the infinite-shot limit at the perturbed control grid."""
    from decoupled_qrc import v6_core as K6
    spec = v6_versions()[version]["spec"]
    if not spec.shots:
        return {}
    return {"exact_expectations_perturbed_controls":
            (make_adapter(spec, shots=0), K6.Ctx(), (0.03, 0.47, 0.97), True)}


def run_robustness(version: str, seeds, tag: str, dev: bool) -> dict:
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc.v6_gates import robustness_ok
    out = {}
    conds = {**robustness_conditions(version), **diagnostic_conditions(version)}
    for name, (adp, ctx, grid, noiseless) in conds.items():
        rows = K6.run_factorial(adp, m_values=grid, g_values=grid, seeds=seeds, ctx=ctx,
                                checkpoint_path=vdir(version) / f"{tag}_robust_{name}.jsonl",
                                tag=name, readouts=NOISE_READOUTS, classical=False, verbose=False)
        r = robustness_ok(rows, ctx, grid, noiseless=noiseless, dev=dev)
        out[name] = r
        print(f"      {name:22s} worst effect {r['worst_effect']:+.3f}  worst HH adv "
              f"{r['worst_hh_advantage']:+.3f}  iso {r['isolated']}  sat {r['saturation']['ok']}  "
              f"-> {'ok' if r['passed'] else 'FAIL'}", flush=True)
    return out


def tests_passed() -> bool:
    meta = OUT / "step1" / "full_tests_meta.txt"
    return meta.exists() and "exit_code=0" in meta.read_text()


def stage_probe(version: str):
    """3 dev seeds, grid (0, 0.25, 1): a fast look before the full grid."""
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc.v6_gates import combined_class, metric_saturation, robustness_ok
    ensure_specified(version)
    ad = adapter_for(version)
    ctx = K6.Ctx()
    rows = K6.run_factorial(ad, m_values=(0.0, 0.25, 1.0), g_values=(0.0, 0.25, 1.0),
                            seeds=DEV_SEEDS[:3], ctx=ctx, checkpoint_path=vdir(version) / "probe.jsonl",
                            tag="probe", verbose=False)
    r = robustness_ok(rows, ctx, (0.0, 0.25, 1.0), noiseless=True, dev=True)
    print(f"  effects: " + ", ".join(f"{k}={v:.3f}" for k, v in r["effects"].items()))
    for cls in ("V6_C1", "V6_C2", "V6_C3", "V6_C4"):
        for meth in ("ridge", "ols_std", "ols_raw"):
            c = combined_class(rows, cls, meth, 0.01, lows=(0.0, 0.25), highs=(1.0,))
            q = c["quadrant_means"]
            print(f"  {cls} {meth:8s} LL {q['LL']:+.3f} LH {q['LH']:+.3f} HL {q['HL']:+.3f} "
                  f"HH {q['HH']:+.3f} null99 {c['null_q99']:.3f} sat {c['constituent_saturated_fraction']:.2f}")
    print(f"  metric saturation ok: {metric_saturation(rows, ctx)['ok']}")


def stage_devgates(version: str):
    from decoupled_qrc import audit_checks as C
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc import v6_gates as G6
    ensure_specified(version)
    t0 = time.time()
    d = vdir(version)
    alg = json.loads((OUT / "algebra" / f"{version}.json").read_text(encoding="utf-8"))
    if not alg["passed"]:
        raise SystemExit(f"{version} failed ALGEBRA; it cannot enter development")
    ad = adapter_for(version)
    ctx = K6.Ctx()
    st = structure(version, 11)
    write(d / "structure.json", st)
    s6 = C.intrinsic_nonlinearity(ad, seeds=DEV_SEEDS[:5])
    write(d / "section6.json", s6)
    print(f"  section 6: {s6['verdict']}")
    rows = K6.run_factorial(ad, m_values=GRID, g_values=GRID, seeds=DEV_SEEDS, ctx=ctx,
                            checkpoint_path=d / "dev_rows.jsonl", tag="dev")
    inv = invalid_controls(version, ad, ctx, st, DEV_SEEDS[:3])
    write(d / "invalid_controls.json", inv)
    enc = {"passed": all(v["affine"] for v in alg["affine_at_g0"].values()),
           "affine_at_g0": alg["affine_at_g0"]}
    g = G6.all_gates(rows, ctx, G6.alpha_for_attempt(1), st=st, s6=s6, enc_exact=enc, inv=inv,
                     tests_passed=tests_passed())
    print("  robustness (5 dev seeds):")
    rob = run_robustness(version, DEV_SEEDS[:5], "dev", dev=True)
    write(d / "robustness_dev.json", rob)
    margins = {}
    for meth in K6.METHODS:
        for k in ("m_M", "g_N", "interior_m_M", "interior_g_N"):
            margins[f"{k}|{meth}"] = g["mains"][meth][k]["estimate"] >= G6.DEV_MARGIN["effect_min"]
        for cls in K6.V6_CLASSES:
            c = g["combined"][cls][meth]
            margins[f"{cls}|{meth}|support"] = c["best"] >= c["null_q99"] + G6.DEV_MARGIN["support_margin"]
            margins[f"{cls}|{meth}|hh_adv"] = c["HH_advantage"] >= G6.DEV_MARGIN["hh_adv_min"]
    margins["robustness_all"] = all(v["passed"] for k, v in rob.items() if k not in DIAGNOSTICS)
    if version not in ("V6.0", "V6.1", "V6.2", "V6.3"):      # margin introduced with V6.4
        cal = vdir(version) / "sentinel_calibration.json"
        if not cal.exists():
            raise SystemExit("run --stage SENTCAL first")
        c = json.loads(cal.read_text(encoding="utf-8"))
        margins["sentinel_calibration"] = bool(
            c["rows_exceeding_0.02"] <= SENTCAL_MARGIN["rows_exceeding_0.02"]
            and c["max_excess"] <= SENTCAL_MARGIN["max_excess"])
    crit = {"gates_all": g["passed_all"], "margins_all": all(margins.values())}
    crit["passed"] = bool(crit["gates_all"] and crit["margins_all"])
    write(d / "dev_gates.json", {"gates": g, "margins": margins, "freeze_criterion": crit,
                                 "runtime_s": round(time.time() - t0, 1)})
    for k, v in g["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    bad = [k for k, v in margins.items() if not v]
    print(f"  margins failing: {bad if bad else 'none'}")
    print(f"  FREEZE CRITERION: {'MET' if crit['passed'] else 'NOT MET'}")
    register(version, "devgates", freeze_criterion=crit,
             failing_gates=[k for k, v in g["gates"].items() if not v], failing_margins=bad)
    progress(version, "DEVGATES", "freeze criterion met" if crit["passed"] else "freeze criterion NOT met",
             "see dev_gates.json", f"python run_v6_stage.py --stage "
             f"{'FREEZE' if crit['passed'] else 'DIAGNOSE'} --version {version}")


# =============================================================================
# TESTS / FREEZE / CONFIRM
# =============================================================================
CONF_ROBUST_SEEDS = 10          # robustness on the first 10 confirmation seeds


def stage_tests(version: str):
    """Full suite, logged per version (gate 26)."""
    d = vdir(version)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    (d / "full_test_log.txt").write_text(
        f"COMMAND: python -m pytest tests/ -q\nexit_code={p.returncode} runtime_s="
        f"{time.time() - t0:.0f}\n--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}",
        encoding="utf-8")
    print(f"  exit {p.returncode}: {p.stdout.strip().splitlines()[-1]}")


def _tests_ok(version: str) -> bool:
    f = vdir(version) / "full_test_log.txt"
    return f.exists() and "exit_code=0" in f.read_text(encoding="utf-8")


def frozen_attempts() -> list:
    return [r for r in registry() if r["event"] == "frozen"]


def stage_freeze(version: str):
    d = vdir(version)
    dg = json.loads((d / "dev_gates.json").read_text(encoding="utf-8"))
    if not dg["freeze_criterion"]["passed"]:
        raise SystemExit("dev freeze criterion NOT met")
    if not _tests_ok(version):
        raise SystemExit("full test suite not passing for this version; run --stage TESTS")
    if any(r["version"] == version for r in frozen_attempts()):
        raise SystemExit(f"{version} already frozen; a redesign needs a new version id")
    k = len(frozen_attempts()) + 1
    seeds = conf_seeds(k)
    used = set()
    for v in used_seeds().values():
        for x in (v.values() if isinstance(v, dict) else v):
            for s in (x if isinstance(x, (list, tuple)) else [x]):
                used |= set(s) if isinstance(s, (list, tuple)) else {s}
    for r in frozen_attempts():
        used |= {s for p in r["confirmation_seeds"] for s in p}
    overlap = used & {s for p in seeds for s in p}
    if overlap:
        raise SystemExit(f"confirmation bank overlaps used seeds: {sorted(overlap)[:5]}")
    from decoupled_qrc.v6_gates import alpha_for_attempt
    payload = {"version": version, "attempt": k, "alpha": alpha_for_attempt(k),
               "spec": v6_versions()[version]["spec"].as_dict(),
               "rationale": v6_versions()[version]["rationale"],
               "confirmation_seeds": seeds, "robustness_seeds": seeds[:CONF_ROBUST_SEEDS],
               "n_previously_used_seeds": len(used),
               "prereg_sha256": W.sha_file(PREREG),
               "source_sha256": {f: W.sha_file(ROOT / f) for f in V6_SOURCES},
               "dev_gates_sha256": W.sha_file(d / "dev_gates.json"),
               "test_log_sha256": W.sha_file(d / "full_test_log.txt"),
               "alpha_ledger": {"before_v6": [0.01, 0.01, 0.01],
                                "v6_attempts": [alpha_for_attempt(i) for i in range(1, k + 1)],
                                "cumulative": 0.03 + sum(alpha_for_attempt(i) for i in range(1, k + 1))},
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    blob = {"payload": payload, "sha256": W.sha_obj(payload)}
    write(d / "frozen.json", blob)
    register(version, "frozen", attempt=k, alpha=payload["alpha"], confirmation_seeds=seeds,
             frozen_sha256=blob["sha256"])
    progress(version, "FREEZE", f"frozen, attempt {k}, alpha {payload['alpha']}", "",
             f"python run_v6_stage.py --stage CONFIRM --version {version}")
    print(f"  frozen {version} attempt {k} alpha {payload['alpha']} sha256 {blob['sha256']}")


def verify_frozen(version: str) -> dict:
    d = vdir(version)
    blob = json.loads((d / "frozen.json").read_text(encoding="utf-8"))
    if W.sha_obj(blob["payload"]) != blob["sha256"]:
        raise SystemExit("frozen artifact modified")
    if W.sha_file(PREREG) != blob["payload"]["prereg_sha256"]:
        raise SystemExit("V6 preregistration modified")
    amended = {}
    for a in sorted((d / "amendments").glob("amendment_*.json")) if (d / "amendments").exists() else []:
        if W.sha_file(a) != a.with_suffix(".sha256").read_text().strip():
            raise SystemExit(f"amendment {a.name} modified")
        for f, ch in json.loads(a.read_text(encoding="utf-8"))["source_changes"].items():
            amended.setdefault(f, set()).add((ch["from"], ch["to"]))
    for f, want in blob["payload"]["source_sha256"].items():
        now = W.sha_file(ROOT / f)
        if now != want and (want, now) not in amended.get(f, set()):
            raise SystemExit(f"source {f} changed since freeze; refusing")
    return blob


def quantum_specific(rows, alpha) -> dict:
    """Separate claim: ALL vs both classical baselines on the HH mean of the class scores,
    plus the classical-simulability rule."""
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc.v6_gates import HIGHS
    out = {}
    for base in ("CLS_M", "CLS_F"):
        diffs = []
        for (a, i), cells in K6.grid_by_seed(rows, lambda r: r).items():
            hh = [c for (m, g), c in cells.items() if m in HIGHS and g in HIGHS]
            q = np.mean([np.mean([r["blocks"]["ALL|ridge"]["class"][c] for c in K6.V6_CLASSES]) for r in hh])
            cl = np.mean([np.mean([r["blocks"][f"{base}|ridge"]["class"][c] for c in K6.V6_CLASSES]) for r in hh])
            diffs.append(q - cl)
        b = K6.boot_q(diffs, [alpha], seed=404)
        out[base] = {"estimate": b["point"], "lower_bound": b["q"][repr(float(alpha))]}
    beats = all(v["lower_bound"] > 0 for v in out.values())
    out["classically_simulable"] = ("yes: R and Q registers stay diagonal in the Z product basis "
                                    "(classical Markov chains over bitstrings); P is reset each step "
                                    "and read through a polynomial of u")
    out["passed"] = False if out["classically_simulable"].startswith("yes") else beats
    out["beats_both_baselines"] = beats
    out["statement"] = ("QUANTUM-SPECIFIC MECHANISM" if out["passed"] else
                        "General memory-nonlinearity separation succeeds, but the mechanism is "
                        "classically reproducible.")
    return out


def stage_confirm(version: str):
    from decoupled_qrc import audit_checks as C
    from decoupled_qrc import v6_core as K6
    from decoupled_qrc import v6_gates as G6
    blob = verify_frozen(version)
    d = vdir(version)
    if (d / "gates.json").exists():
        raise SystemExit("confirmation already ran for this version")
    p = blob["payload"]
    t0 = time.time()
    ad = adapter_for(version)
    ctx = K6.Ctx()
    seeds = [tuple(s) for s in p["confirmation_seeds"]]
    st = structure(version, 31)
    write(d / "structure_confirmation.json", st)
    s6 = C.intrinsic_nonlinearity(ad, seeds=seeds[:5])
    write(d / "section6_confirmation.json", s6)
    rows = K6.run_factorial(ad, m_values=GRID, g_values=GRID, seeds=seeds, ctx=ctx,
                            checkpoint_path=d / "conf_rows.jsonl", tag="conf")
    inv = invalid_controls(version, ad, ctx, st, seeds[:3])
    write(d / "invalid_controls_confirmation.json", inv)
    alg = json.loads((OUT / "algebra" / f"{version}.json").read_text(encoding="utf-8"))
    enc = {"passed": all(v["affine"] for v in alg["affine_at_g0"].values()),
           "affine_at_g0": alg["affine_at_g0"]}
    g = G6.all_gates(rows, ctx, p["alpha"], st=st, s6=s6, enc_exact=enc, inv=inv,
                     tests_passed=_tests_ok(version))
    print("  robustness on confirmation seeds:")
    rob = run_robustness(version, [tuple(s) for s in p["robustness_seeds"]], "conf", dev=False)
    write(d / "robustness_confirmation.json", rob)
    g["gates"]["22_25_robustness"] = all(v["passed"] for k, v in rob.items() if k not in DIAGNOSTICS)
    g["passed_all"] = bool(all(g["gates"].values()))
    qs = quantum_specific(rows, p["alpha"])
    result = {"frozen_sha256": blob["sha256"], "attempt": p["attempt"], "alpha": p["alpha"],
              "n_rows": len(rows), "gates": g, "robustness": rob, "quantum_specific": qs,
              "success": g["passed_all"], "runtime_s": round(time.time() - t0, 1)}
    write(d / "gates.json", result)
    for k, v in g["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  SUCCESS (all mandatory gates): {g['passed_all']}")
    print(f"  quantum-specific: {qs['statement']}")
    register(version, "confirmed" if g["passed_all"] else "confirmation_failed",
             failing=[k for k, v in g["gates"].items() if not v])
    progress(version, "CONFIRM", "SUCCESS" if g["passed_all"] else "confirmation FAILED",
             "", f"python run_v6_stage.py --stage REPORT --version {version}")


def stage_report(version: str):
    from decoupled_qrc.v6_report import build
    build(vdir(version), version)


# =============================================================================
# SENTCAL -- development-only sentinel calibration at confirmation size
# =============================================================================
CAL_BLOCK = {"V6.4": 0, "V6.5": 1}   # a FRESH 500-row dev block per version (no reuse)


def cal_seeds(block: int) -> list:
    if not 0 <= block <= 8:
        raise ValueError("calibration blocks 0..8 only (keeps arch / input ranges disjoint)")
    return [(100100 + 500 * block + k, 105100 + 500 * block + k) for k in range(500)]


CAL_SEEDS = cal_seeds(0)
SENTCAL_MARGIN = {"rows_exceeding_0.02": 0, "max_excess": 0.016}   # V6.4 dev freeze margin (set
# before V6.4 was calibrated): no exceedance in 500 dev rows AND 20% headroom below 0.02


def sentinel_calibration(spec, tag: str, n: int = 500, block: int = 0) -> dict:
    """Chance rate of the preregistered future / unreachable sentinels (max member -
    null99 > 0.02) over n development rows at grid points cycled deterministically."""
    from decoupled_qrc import audit_ipc as A
    from decoupled_qrc import v6_core as K6
    ctx = K6.Ctx()
    keep = [t for t in ctx.lib if t.cls in K6.SENTINELS]
    lib = keep + [A.T(f"null{i}", "NULL", ((0, 1),)) for i in range(ctx.n_null)]
    cls = np.array([t.cls for t in lib])
    ad = make_adapter(spec)
    cp = OUT / "sentcal" / f"{tag}.jsonl"
    done = {json.loads(l)["i"]: json.loads(l) for l in cp.read_text().splitlines()} if cp.exists() else {}
    cp.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, sp in enumerate(cal_seeds(block)[:n]):
        if i in done:
            recs.append(done[i]); continue
        m, g = GRID[i % 5], GRID[(i // 5) % 5]
        u = np.random.default_rng(sp[1]).uniform(-1, 1, ctx.T)
        Y = np.hstack([A.build_Y(u, keep), A.null_targets(ctx.T, ctx.n_null, seed=sp[1] + 7_000_003)])
        f = ad.run(u, m, g, sp[0])
        X = np.hstack([f["R"], f["P"], f["J"], f["Q"]])
        rec = {"i": i, "m": m, "g": g, "K": int(X.shape[1])}
        for meth in K6.METHODS:
            c = A.score_all(X, Y, lib, ctx.split, meth, alpha=ctx.alpha)
            q99 = float(np.quantile(c[cls == "NULL"], 0.99))
            for s in K6.SENTINELS:
                rec[f"{s}|{meth}"] = float(c[cls == s].max() - q99)
        with open(cp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        recs.append(rec)
    ex = np.array([[r[k] for k in r if "|" in k] for r in recs])
    return {"n_rows": len(recs), "K": recs[0]["K"], "block": block, "max_excess": float(ex.max()),
            "rows_exceeding_0.02": int(np.sum(ex.max(axis=1) > 0.02)),
            "q99_row_max_excess": float(np.quantile(ex.max(axis=1), 0.99))}


def stage_sentcal(version: str):
    ensure_specified(version)
    r = sentinel_calibration(v6_versions()[version]["spec"], version, block=CAL_BLOCK[version])
    write(vdir(version) / "sentinel_calibration.json", r)
    print(f"  {version}: K {r['K']}  rows {r['n_rows']}  max excess {r['max_excess']:+.4f}  "
          f"rows > 0.02: {r['rows_exceeding_0.02']}  margin {SENTCAL_MARGIN}")


STAGES = {"STEP1": stage_step1, "PREREG_GATES": stage_prereg_gates, "ALGEBRA": stage_algebra,
          "SENTCAL": stage_sentcal,
          "PROBE": stage_probe, "DEVGATES": stage_devgates, "TESTS": stage_tests,
          "FREEZE": stage_freeze, "CONFIRM": stage_confirm, "REPORT": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--version", default=None)
    a = ap.parse_args()
    print(f"=== V6 {a.stage} {a.version or ''} ===")
    STAGES[a.stage]() if a.version is None else STAGES[a.stage](a.version)
