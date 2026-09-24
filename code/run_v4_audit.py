"""
run_v4_audit.py -- adversarial, reproducible audit of the FROZEN V4 architecture.

    python run_v4_audit.py --stage ENV
    python run_v4_audit.py --stage DEV          # development seeds only
    python run_v4_audit.py --stage PREREG       # freeze + hash the audit plan
    python run_v4_audit.py --stage ARTIFACTS    # inspect existing results/v4 (after PREREG)
    python run_v4_audit.py --stage CONFIRM      # one run on a FRESH seed bank
    python run_v4_audit.py --stage REPORT

Order is enforced. CONFIRM refuses unless the preregistration exists, re-hashes,
and the audit source still matches the hash recorded in it -- so the gates
cannot be edited after the confirmation data exist.

Nothing in V4 is modified. The architecture is loaded from its own frozen
artifact (results/v4/frozen_config.json) and evaluated by an independent IPC
engine (decoupled_qrc/audit_ipc.py) that is cross-checked against V4's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from decoupled_qrc import audit_checks as C          # noqa: E402
from decoupled_qrc import audit_core as K            # noqa: E402

OUT = ROOT / "results" / "v4" / "independent_audit"
FROZEN_V4 = ROOT / "results" / "v4" / "frozen_config.json"
PREREG = OUT / "preregistered_audit.json"

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ROBUST_GRID = (0.0, 0.5, 1.0)
DEV_SEEDS = [(20000 + k, 30000 + k) for k in range(10)]
CONF_SEEDS = [(70000 + k, 80000 + k) for k in range(20)]
USED_BEFORE = {  # every seed any earlier stage of this project is known to have used
    "v4_dev_arch": list(range(100, 140)), "v4_dev_input": list(range(1000, 1040)),
    "v4_conf_arch": list(range(900, 940)), "v4_conf_input": list(range(9000, 9040)),
    "probes": [3, 5, 7, 11, 12, 77, 900, 9000, 31337, 424242],
}

AUDIT_SOURCES = ["code/decoupled_qrc/audit_ipc.py", "code/decoupled_qrc/audit_core.py",
                 "code/decoupled_qrc/audit_checks.py", "code/decoupled_qrc/audit_report.py",
                 "code/run_v4_audit.py"]


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_js),
                            encoding="utf-8")
    print(f"  wrote {(OUT / name).relative_to(ROOT)}")


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_rows(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# =============================================================================
# ENV
# =============================================================================
def stage_env():
    import importlib
    t0 = time.time()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    versions = {}
    for m in ("numpy", "scipy", "sklearn", "matplotlib", "qiskit", "qiskit_aer",
              "qiskit_ibm_runtime", "pytest"):
        try:
            versions[m] = importlib.import_module(m).__version__
        except Exception as exc:
            versions[m] = f"unavailable ({type(exc).__name__})"
    try:
        gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except Exception:
        gpu = "none"
    v4_files = sorted([str(p.relative_to(ROOT)) for p in (HERE / "decoupled_qrc").glob("v4_*.py")]
                      + ["code/run_v4_stage.py", "code/DQRC_DualRoute_Decoupling_V4_Colab.ipynb",
                         "code/_build_notebook_dualroute_v4_colab.py"]
                      + [str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_v4_*.py")])
    # the notebook: driver or implementation?
    nb = json.loads((HERE / "DQRC_DualRoute_Decoupling_V4_Colab.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "\n".join("".join(c["source"]) for c in code_cells)
    imports = sorted({line.split()[1].split(".")[0] if line.startswith("import ")
                      else line.split()[1] for line in src.splitlines()
                      if line.strip().startswith(("import ", "from "))})
    import re
    local_mods = sorted(set(re.findall(r"(?:from|import)\s+(decoupled_qrc\.\w+|run_v4_stage)", src)))
    exists = {m: (HERE / (m.replace(".", "/") + ".py")).exists() for m in local_mods}
    defs = len(re.findall(r"^\s*(def|class)\s", src, flags=re.M))
    executed = any(c.get("execution_count") is not None or c.get("outputs") for c in code_cells)
    out = {"git_commit": git("rev-parse", "HEAD"),
           "working_tree_dirty": bool(git("status", "--porcelain")),
           "dirty_entries": git("status", "--porcelain").splitlines(),
           "python": sys.version, "platform": platform.platform(),
           "processor": platform.processor(), "cpu_count": os.cpu_count(), "gpu": gpu,
           "package_versions": versions,
           "v4_source_sha256": {f: sha_file(ROOT / f) for f in v4_files},
           "v4_frozen_config_sha256_declared": json.loads(FROZEN_V4.read_text())["sha256"],
           "notebook": {"n_cells": len(nb["cells"]), "n_code_cells": len(code_cells),
                        "function_or_class_definitions": defs,
                        "has_executed_outputs": executed,
                        "local_imports": exists,
                        "all_local_imports_exist": all(exists.values()),
                        "classification": ("DRIVER ONLY: wires run_v4_stage; contains no "
                                           "implementation and no results" if defs == 0 and
                                           not executed else "contains implementation or results")},
           "seed_banks": {"audit_dev": DEV_SEEDS, "audit_confirmation": CONF_SEEDS,
                          "used_before": USED_BEFORE},
           "runtime_s": round(time.time() - t0, 2)}
    used = set(sum(USED_BEFORE.values(), []))
    fresh = {s for sp in CONF_SEEDS for s in sp}
    devs = {s for sp in DEV_SEEDS for s in sp}
    out["confirmation_bank_fresh"] = bool(not (fresh & used) and not (fresh & devs))
    write("env.json", out)
    print(f"  commit {out['git_commit'][:12]} dirty={out['working_tree_dirty']} "
          f"notebook={out['notebook']['classification'][:24]} "
          f"imports_exist={out['notebook']['all_local_imports_exist']} "
          f"fresh_bank={out['confirmation_bank_fresh']}")
    return out


# =============================================================================
# DEV -- development seeds only; nothing here may touch the confirmation bank
# =============================================================================
def stage_dev():
    t0 = time.time()
    ad = K.V4Adapter(FROZEN_V4)
    ctx = K.Context(alpha=ad.alpha)
    print("  [1] encoder algebra + encoder-only capacity")
    enc = {"symbolic": [C.symbolic_encoder_expansion(n) for n in range(1, 6)],
           "numeric": [C.numeric_encoder_expansion(n) for n in range(1, 6)],
           "encoder_only_capacity": C.encoder_only_capacity(ad)}
    write("encoder.json", enc)

    print("  [2] structural dependency + contaminated controls (100 draws each)")
    st = {k or "frozen": C.structural_dependency(ad.spec, n_draws=100, seed=11, kind=k)
          for k in (None, "g_into_R", "m_into_P", "serial")}
    st["indirect_inspection"] = C.indirect_dependency_inspection()
    write("structure.json", st)

    print("  [3] section 6: intrinsic nonlinear control vs ridge artifact")
    s6 = C.intrinsic_nonlinearity(ad, seeds=DEV_SEEDS[:5])
    write("section6.json", s6)

    print("  [4] dev factorial: 10 seeds x 5x5")
    rows = K.run_factorial(ad, m_values=GRID, g_values=GRID, seeds=DEV_SEEDS, ctx=ctx,
                           checkpoint_path=OUT / "dev_rows.jsonl", tag="dev")
    print(f"      {len(rows)} rows")

    print("  [5] invalid controls")
    inv = invalid_controls(ad, ctx)
    write("invalid_controls.json", inv)

    print("  [6] classical baseline at finite shots + Sunada finite-shot (dev)")
    write("classical_dev.json", classical_finite_shots(ad, ctx, DEV_SEEDS[:5]))

    print("  [7] robustness conditions")
    write("robustness.json", robustness(ad))

    dev_gates = all_gates(rows, K.Levels(), s6=s6, st=st, lows=(0.0, 0.25), highs=(0.75, 1.0))
    write("dev_gates.json", dev_gates)
    print(f"  DEV done in {time.time() - t0:.0f}s")


def invalid_controls(ad, ctx) -> dict:
    """Every deliberately invalid control must be detected."""
    out = {}
    st = {k: C.structural_dependency(ad.spec, n_draws=100, seed=21, kind=k)
          for k in ("g_into_R", "m_into_P", "serial")}
    for k, v in st.items():
        out[k] = {"detected": not v["isolated"], "detector": "structural dependency test",
                  "evidence": {x: v[x] for x in ("max_dXR_dg", "max_dXP_dm")}}
    small = [(20000 + k, 30000 + k) for k in range(3)]
    for kind, detector in (("leak_targets", "unreachable-class + future-sentinel"),
                           ("feature_count_changes", "feature-count invariance")):
        rows = K.run_factorial(ad, m_values=(0.0, 1.0), g_values=(0.0, 1.0), seeds=small,
                               ctx=ctx, checkpoint_path=OUT / f"invalid_{kind}.jsonl",
                               tag=kind, feature_hook=C.leak_hook(kind), verbose=False)
        if kind == "leak_targets":
            c2 = max(r["blocks"]["R|ridge"]["class"]["C2_oldNL__max"]
                     - r["blocks"]["R|ridge"]["null_q99"] for r in rows)
            sen = K.saturation_and_sentinel(rows)
            out[kind] = {"detected": bool(c2 > 0.02 or not sen["sentinel_ok"]),
                         "detector": detector,
                         "evidence": {"unreachable_class_excess": c2,
                                      "sentinel_excess": sen["sentinel_excess_max"]}}
        else:
            fc = K.feature_count_invariance(rows)
            out[kind] = {"detected": not fc["invariant"], "detector": detector,
                         "evidence": fc}
    # reuse training data as test
    bad = type(ctx.split)(ctx.split.train, ctx.split.train)
    chk = bad.check()
    u = np.random.default_rng(30000).uniform(-1, 1, ctx.T)
    X = ad.run(u, 0.5, 0.5, 20000)
    Xall = np.hstack([X["R"], X["P"], X["J"]])
    nul = K.A.null_targets(ctx.T, 60, seed=99)
    cap_leak = K.A.capacity(K.A.predict(Xall, nul, bad, "ols_std"), nul, bad)
    cap_ok = K.A.capacity(K.A.predict(Xall, nul, ctx.split, "ols_std"), nul, ctx.split)
    out["reuse_train_as_test"] = {
        "detected": bool(not chk["disjoint"]), "detector": "split integrity",
        "evidence": {"split_check": chk,
                     "null_capacity_mean_when_train_is_test": float(np.mean(cap_leak)),
                     "null_capacity_mean_proper_split": float(np.mean(cap_ok))}}
    out["all_detected"] = bool(all(v["detected"] for v in out.values() if isinstance(v, dict)))
    return out


def classical_finite_shots(ad, ctx, seeds, shots=(100, 1000, 10000)) -> dict:
    """Quantum joint estimator vs classical product-of-marginals, same shots."""
    from decoupled_qrc.v4_architecture import _joint_pairs
    lib = ctx.lib
    c1 = np.array([t.cls == "C1_curNL_x_oldLin" for t in lib])
    sun = np.array([t.cls == "sunada" for t in lib])
    out = {"shots": {}, "exact": {}}
    for sp in seeds:
        u, Y, _ = K._targets(ctx, sp[1])
        for (m, g) in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)):
            f = ad.run(u, m, g, sp[0])
            pairs = _joint_pairs(f["R"].shape[1], f["P"].shape[1], ad.spec.n_joint, sp[0])
            XJ = f["J"]
            XC = np.column_stack([f["R"][:, a] * f["P"][:, b] for a, b in pairs])
            ex = out["exact"].setdefault(f"m{m}_g{g}", [])
            sj = K.A.score_all(XJ, Y, lib, ctx.split, "ridge", alpha=ctx.alpha)
            sc = K.A.score_all(XC, Y, lib, ctx.split, "ridge", alpha=ctx.alpha)
            ex.append({"J_C1": float(np.mean(sj[c1])), "CLS_same_pairs_C1": float(np.mean(sc[c1])),
                       "max_abs_feature_diff": float(np.abs(XJ - XC).max())})
            for S in shots:
                rng = np.random.default_rng(sp[0] * 7 + int(S))
                q, cl = C.joint_estimators(f["R"], f["P"], pairs, S, rng)
                sq = K.A.score_all(q, Y, lib, ctx.split, "ridge", alpha=ctx.alpha)
                scl = K.A.score_all(cl, Y, lib, ctx.split, "ridge", alpha=ctx.alpha)
                out["shots"].setdefault(f"S{S}_m{m}_g{g}", []).append(
                    {"quantum_joint_C1": float(np.mean(sq[c1])),
                     "classical_product_C1": float(np.mean(scl[c1])),
                     "quantum_joint_sunada": float(np.mean(sq[sun])),
                     "classical_product_sunada": float(np.mean(scl[sun])),
                     "feature_rmse_quantum": float(np.sqrt(np.mean((q - XJ) ** 2))),
                     "feature_rmse_classical": float(np.sqrt(np.mean((cl - XJ) ** 2)))})
    labR, labP = ad.spec.memory.observables()[0], ad.spec.processor.observables()[0]
    from decoupled_qrc.v4_architecture import _joint_pairs as jp
    pairs = jp(len(labR), len(labP), ad.spec.n_joint, 0)
    labJ = [f"J:{labR[a].split(':')[1]}{labP[b].split(':')[1]}" for a, b in pairs]
    out["resources"] = {
        "quantum_joint": {"observables": len(labJ),
                          "measurement_settings": C.measurement_settings(
                              [f"X:{l.split(':')[1]}" for l in labJ])},
        "classical_product": {"observables_measured": len(labR) + len(labP),
                              "measurement_settings_parallel": max(
                                  C.measurement_settings(labR), C.measurement_settings(labP)),
                              "note": "R and P are measured in parallel on separate registers; "
                                      "products are formed in post-processing"},
        "energy": "not available: no hardware run"}
    return out


def robustness(ad) -> dict:
    """Principal conclusions re-derived under each perturbation (dev seeds)."""
    seeds = [(20000 + k, 30000 + k) for k in range(5)]
    base = dict(alpha=ad.alpha)
    conds = {
        "baseline": (ad, K.Context(**base), ROBUST_GRID),
        "longer_T3200": (ad, K.Context(T=3200, **base), ROBUST_GRID),
        "half_training": (ad, K.Context(T=1110, train_frac=550 / 1110, **base), ROBUST_GRID),
        "double_training": (ad, K.Context(T=2580, train_frac=2020 / 2580, **base), ROBUST_GRID),
        "float32_readout": (ad, K.Context(dtype="float32", **base), ROBUST_GRID),
        "perturbed_controls": (ad, K.Context(**base), (0.03, 0.47, 0.97)),
        "ridge_0.05": (ad, K.Context(alpha=0.05), ROBUST_GRID),
        "ridge_5.0": (ad, K.Context(alpha=5.0), ROBUST_GRID),
        "delay_range_12": (ad, K.Context(tau_max=12, **base), ROBUST_GRID),
        "degree_6": (ad, K.Context(max_degree=6, **base), ROBUST_GRID),
        "shots_1000": (C.finite_shot_adapter(ad, 1000), K.Context(**base), ROBUST_GRID),
        "shots_10000": (C.finite_shot_adapter(ad, 10000), K.Context(**base), ROBUST_GRID),
    }
    ibm = C.ibm_noise_parameters("FakeTorino")
    if ibm.get("available"):
        conds["ibm_noise_FakeTorino"] = (C.ibm_noisy_adapter(ad, ibm), K.Context(**base),
                                          ROBUST_GRID)
    out = {"ibm_noise_parameters": ibm, "conditions": {}}
    for name, (adp, ctx, grid) in conds.items():
        rows = K.run_factorial(adp, m_values=grid, g_values=grid, seeds=seeds, ctx=ctx,
                               checkpoint_path=OUT / f"robust_{name}.jsonl", tag=name,
                               verbose=False)
        out["conditions"][name] = principal(rows, grid, ctx)
        pc = out["conditions"][name]
        print(f"      {name:22s} dM={pc['dM_ridge']:+.3f} dN={pc['dN_ridge']:+.3f} "
              f"dN_int_ols={pc['dN_interior_ols']:+.3f} isoM={pc['M_spread_over_g']:.1e} "
              f"isoN={pc['N_spread_over_m']:.1e} C1_HHadv={pc['C1_HH_minus_best']:+.3f}")
    # drop-one-observable on corners only
    drops = {}
    corner = (0.0, 1.0)
    for ro, n in (("R", len(ad.spec.memory.observables()[0])),
                  ("P", len(ad.spec.processor.observables()[0]))):
        for j in range(n):
            def hook(X, *, u, m, g, ro=ro, j=j):
                X = dict(X)
                X[ro] = np.delete(X[ro], j, axis=1)
                return X
            rows = K.run_factorial(ad, m_values=corner, g_values=corner, seeds=seeds[:3],
                                   ctx=K.Context(**base), checkpoint_path=OUT /
                                   f"robust_drop_{ro}{j}.jsonl", tag=f"drop_{ro}{j}",
                                   readouts={"R": ("ridge",), "P": ("ridge",)},
                                   feature_hook=hook, verbose=False)
            gM = K.per_seed_grid(rows, K.metric_key("R", "ridge", "M"))
            gN = K.per_seed_grid(rows, K.metric_key("P", "ridge", "N"))
            drops[f"{ro}{j}"] = {"dM": float(np.mean(K.effect_per_seed(gM, "m"))),
                                 "dN": float(np.mean(K.effect_per_seed(gN, "g")))}
    out["drop_one_observable"] = drops
    return out


def principal(rows, grid, ctx) -> dict:
    lo, hi = grid[0], grid[-1]
    gM = K.per_seed_grid(rows, K.metric_key("R", "ridge", "M"))
    gN = K.per_seed_grid(rows, K.metric_key("P", "ridge", "N"))
    gNo = K.per_seed_grid(rows, K.metric_key("P", "ols_std", "N"))
    gMo = K.per_seed_grid(rows, K.metric_key("R", "ols_std", "M"))
    mli = C.metric_level_independence(rows)
    qc = K.quadrant_contrasts(rows, "ALL", "ridge", "C1_curNL_x_oldLin",
                              lows=(lo,), highs=(hi,), lv=K.Levels())
    qm = qc["quadrant_means"]
    return {"dM_ridge": float(np.mean(K.effect_per_seed(gM, "m"))),
            "dN_ridge": float(np.mean(K.effect_per_seed(gN, "g"))),
            "dM_ols": float(np.mean(K.effect_per_seed(gMo, "m"))),
            "dN_ols": float(np.mean(K.effect_per_seed(gNo, "g"))),
            "dN_interior_ols": float(np.mean(K.effect_per_seed(gNo, "g", lo=grid[1], hi=hi))),
            "dM_interior_ols": float(np.mean(K.effect_per_seed(gMo, "m", lo=grid[1], hi=hi))),
            "M_spread_over_g": max(v["max_spread_M_over_g"] for v in mli.values()),
            "N_spread_over_m": max(v["max_spread_N_over_m"] for v in mli.values()),
            "C1_quadrants": qm,
            "C1_HH_minus_best": qm["HH"] - max(qm["LL"], qm["LH"], qm["HL"]),
            "saturation": K.saturation_and_sentinel(rows),
            "feature_counts": K.feature_count_invariance(rows)}


# =============================================================================
# Gates -- one function, used identically on dev and confirmation rows
# =============================================================================
def all_gates(rows, lv, *, s6, st, lows, highs, seed=0) -> dict:
    g = {}
    mli = C.metric_level_independence(rows)
    g["C1_structural"] = {
        "frozen_isolated": st["frozen"]["isolated"],
        "contaminated_detected": {k: not st[k]["isolated"]
                                  for k in ("g_into_R", "m_into_P", "serial")},
        "metric_level_independence": mli}
    g["C1_structural"]["passed"] = bool(
        st["frozen"]["isolated"] and all(g["C1_structural"]["contaminated_detected"].values())
        and all(v["max_spread_M_over_g"] == 0 and v["max_spread_N_over_m"] == 0
                for v in mli.values()))

    def separation(meth):
        out = {"main_m_M": K.gate_main(rows, "R", meth, "M", "m", lv=lv, seed=seed + 1),
               "main_g_N": K.gate_main(rows, "P", meth, "N", "g", lv=lv, seed=seed + 2),
               "cross_m_N": K.gate_cross(rows, "P", meth, "N", "m", "R", "M", "m", lv=lv,
                                         seed=seed + 3),
               "cross_g_M": K.gate_cross(rows, "R", meth, "M", "g", "P", "N", "g", lv=lv,
                                         seed=seed + 4),
               "degree_profile_m": K.gate_profile_equivalence(
                   rows, "P", meth, [(d, 0) for d in (2, 3, 4)], "m", lv=lv, seed=seed + 5),
               "memory_curve_g": K.gate_profile_equivalence(
                   rows, "R", meth, [(1, t) for t in range(9)], "g", lv=lv, seed=seed + 6)}
        out["passed"] = bool(all(v["passed"] for v in out.values()))
        return out

    sat = K.saturation_and_sentinel(rows)
    fc = K.feature_count_invariance(rows)
    g["C2_resource_constrained"] = separation("ridge")
    g["C2_resource_constrained"].update({"saturation_sentinel": sat, "feature_counts": fc})
    g["C2_resource_constrained"]["passed"] = bool(
        g["C2_resource_constrained"]["passed"] and sat["saturation_ok"] and sat["sentinel_ok"]
        and fc["invariant"])

    c3 = {"ols_std": separation("ols_std"), "ols_raw": separation("ols_raw")}
    c3["interior_m_M"] = K.gate_main(rows, "R", "ols_std", "M", "m", lv=lv, lo=0.25, hi=1.0,
                                     seed=seed + 7)
    c3["interior_g_N"] = K.gate_main(rows, "P", "ols_std", "N", "g", lv=lv, lo=0.25, hi=1.0,
                                     seed=seed + 8)
    nsat = []
    for r in rows:
        if r["g"] > 0:
            s = r["blocks"]["P|ols_std"]["single"]
            nsat += [s[f"d{d}_t0"] > 0.995 for d in (2, 3, 4)]
    c3["N_target_saturation_fraction_g_positive"] = float(np.mean(nsat)) if nsat else float("nan")
    c3["section6_verdict"] = s6["verdict"]
    c3["passed"] = bool(c3["ols_std"]["passed"] and c3["ols_raw"]["passed"]
                        and c3["interior_m_M"]["passed"] and c3["interior_g_N"]["passed"]
                        and s6["verdict"].endswith("HOLDS")
                        and c3["N_target_saturation_fraction_g_positive"] <= 0.20)
    g["C3_intrinsic"] = c3

    c4 = {"lows": list(lows), "highs": list(highs), "classes": {}}
    null99 = float(np.mean([r["blocks"]["ALL|ridge"]["null_q99"] for r in rows]))
    for cls in ("C1_curNL_x_oldLin", "C2_oldNL", "C3_old_x_old", "C4_old3"):
        q = K.quadrant_contrasts(rows, "ALL", "ridge", cls, lows=lows, highs=highs, lv=lv,
                                 seed=seed + 9)
        qo = K.quadrant_contrasts(rows, "ALL", "ols_std", cls, lows=lows, highs=highs, lv=lv,
                                  seed=seed + 10)
        best = max(q["quadrant_means"].values())
        supported = bool(best > null99 + 0.02)
        c4["classes"][cls] = {"supported": supported, "best_quadrant_mean": best,
                              "ridge": q, "ols_std": qo,
                              "HH_beats_all": q["HH_beats_all"]}
    sup = [k for k, v in c4["classes"].items() if v["supported"]]
    c4["null_q99_ALL_ridge"] = null99
    c4["supported_classes"] = sup
    c4["unsupported_classes"] = [k for k in c4["classes"] if k not in sup]
    c4["passed"] = bool(sup and all(c4["classes"][k]["HH_beats_all"] for k in sup))
    g["C4_combined"] = c4

    c5 = {}
    gJ = K.per_seed_grid(rows, K.class_key("J", "ridge", "C1_curNL_x_oldLin"))
    gC = K.per_seed_grid(rows, K.class_key("CLS", "ridge", "C1_curNL_x_oldLin"))
    diff = [np.mean([gJ[s][k] - gC[s][k] for k in gJ[s]]) for s in gJ]
    b = K.boot(diff, seed=seed + 11)
    c5["quantum_joint_minus_classical_allpairs_C1"] = {"estimate": b["point"],
                                                       "lower_bound": b.get(lv.main_lo)}
    c5["passed"] = bool(np.isfinite(b.get(lv.main_lo, np.nan)) and b[lv.main_lo] > 0)
    c5["statement"] = ("QUANTUM-SPECIFIC ADVANTAGE" if c5["passed"]
                       else "COMBINED MECHANISM IS CLASSICALLY REPRODUCIBLE")
    g["C5_quantum_specific"] = c5
    return g


def overall(g) -> str:
    p = {k: g[k]["passed"] for k in g if k.startswith("C")}
    if all(p.values()):
        return "V4 WORKS AS CLAIMED"
    if not p["C1_structural"]:
        return "V4 DOES NOT DEMONSTRATE MEMORY-NONLINEARITY SEPARATION"
    if p["C2_resource_constrained"] and not p["C3_intrinsic"]:
        return "V4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED DEFINITION"
    if not p["C2_resource_constrained"]:
        return "V4 PROVIDES MODULAR ISOLATION BUT NOT COMPUTATIONAL SEPARATION"
    return "V4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED DEFINITION"


# =============================================================================
# PREREG
# =============================================================================
def stage_prereg():
    ad = K.V4Adapter(FROZEN_V4)
    ctx = K.Context(alpha=ad.alpha)
    env = json.loads((OUT / "env.json").read_text())
    plan = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis_under_test": "V4 as frozen at sha256 " + ad.frozen_sha256,
        "source_sha256": {"v4": env["v4_source_sha256"],
                          "audit": {f: sha_file(ROOT / f) for f in AUDIT_SOURCES}},
        "git_commit": env["git_commit"],
        "architecture": ad.describe(),
        "controls": {"m": "normalised memory control in [0,1]; retention = m * m_max",
                     "g": "normalised nonlinear control in [0,1]; interaction = g * g_scale"},
        "grid": {"m": list(GRID), "g": list(GRID)},
        "seeds": {"development": DEV_SEEDS, "confirmation": CONF_SEEDS,
                  "previously_used": USED_BEFORE,
                  "confirmation_bank_fresh": env["confirmation_bank_fresh"]},
        "data": {**ctx.as_dict(), "input": "u_t ~ U[-1,1] i.i.d., one sequence per seed",
                 "split": ctx.split.check(), "validation_set": "none (fixed ridge; OLS)"},
        "observables": {"R": ad.spec.memory.observables()[0],
                        "P": ad.spec.processor.observables()[0],
                        "J": f"{ad.spec.n_joint} fixed O_R (x) O_P pairs",
                        "CLS": "all pairwise classical products of R and P locals"},
        "target_library": {c: sum(1 for t in ctx.lib if t.cls == c)
                           for c in sorted({t.cls for t in ctx.lib})},
        "readouts": {"ridge": f"standardised, alpha={ad.alpha} (V4's preregistered penalty)",
                     "ols_std": "standardised, pseudo-inverse, relative cutoff 1e-10",
                     "ols_raw": "centred raw, pseudo-inverse, relative cutoff 1e-10",
                     "standardisation": "train-block mean/std of the SAME readout; sd<1e-12 -> 1"},
        "capacity": "C = 1 - MSE/Var on the held-out block, NOT clipped",
        "null": f"{ctx.n_null} independent-sequence Legendre targets per point; threshold = 99th pct",
        "statistics": {"unit": "seed (one architecture seed + one input seed)",
                       "bootstrap": "10000 resamples of seeds",
                       "levels_reported": ["user default", "sequential"],
                       "levels_for_verdicts": "SEQUENTIAL (stricter): repository ledger spends "
                                              "alpha=0.01 per look -> main LB at 1st pct, "
                                              "TOST 98% interval, ratio UB 99th pct",
                       "profile_equivalence": "Bonferroni over members (>= Holm strictness)"},
        "margins": {"main_effect_min": 0.10, "equivalence": [-0.03, 0.03],
                    "cross_ratio_max": 0.20, "saturation_max_fraction": 0.20,
                    "saturation_ceiling": 0.995, "sentinel_excess_max": 0.02,
                    "supported_class_margin_over_null": 0.02},
        "claims": {
            "1_structural": "frozen V4 isolated to exactly 0 over 100 random input/state draws "
                            "(dX_R/dg, dX_P/dm, full-range changes); all 3 contaminated controls "
                            "detected; M bit-identical across g and N across m for all analyses",
            "2_resource_constrained": "ridge: main effects >=0.10 LB>0; cross TOST in +-0.03 and "
                                      "ratio UB<0.20 both directions; degree-profile (m) and "
                                      "memory-curve (g) equivalence; no saturation; sentinel ok; "
                                      "feature counts invariant",
            "3_intrinsic": "claim-2 separation gates pass under BOTH ols_std and ols_raw; interior "
                           "effects M(1)-M(0.25) and N(1)-N(0.25) >=0.10 LB>0 under ols_std; "
                           "section-6 verdict HOLDS; N-target saturation over g>0 <= 0.20",
            "4_combined": "operational readout (R+P+J), ridge. A class is SUPPORTED if its best "
                          "quadrant mean exceeds the null 99th pct + 0.02. PASS iff >=1 supported "
                          "class and HH beats LL, LH, HL with LB>0 on EVERY supported class. "
                          "Quadrants: L={0,0.25}, H={0.75,1}.",
            "5_quantum_specific": "quantum joint features beat the equally resourced classical "
                                  "product composition on C1 with LB>0"},
        "overall_mapping": {"all five pass": "V4 WORKS AS CLAIMED",
                            "1 fails": "V4 DOES NOT DEMONSTRATE MEMORY-NONLINEARITY SEPARATION",
                            "1 passes, 2 fails": "V4 PROVIDES MODULAR ISOLATION BUT NOT "
                                                 "COMPUTATIONAL SEPARATION",
                            "1,2 pass, 3 fails": "V4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED "
                                                 "DEFINITION"},
        "disclosure": ("The auditor saw V4's earlier confirmation numbers in a previous session. "
                       "All gates above are the brief's own values (or stricter repository "
                       "values); none was chosen with reference to those numbers. "
                       "audit_checks.py gained two V4-inert hooks (section 6 takes the processor "
                       "map from adapter.processor_fn when present; structural_dependency_adapter) "
                       "after the DEV process had loaded it; ARTIFACTS regenerates V4 section 6 "
                       "from the current source and records the difference."),
    }
    PREREG.write_text(json.dumps(plan, indent=2, sort_keys=True, default=_js), encoding="utf-8")
    h = sha_file(PREREG)
    (OUT / "preregistered_audit.sha256").write_text(h + "\n", encoding="utf-8")
    print(f"  preregistration frozen: sha256 {h}")


def verify_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit("no preregistration; run --stage PREREG first")
    h = sha_file(PREREG)
    stored = (OUT / "preregistered_audit.sha256").read_text().strip()
    if h != stored:
        raise SystemExit(f"preregistration was MODIFIED after freezing ({stored[:12]} -> {h[:12]})")
    plan = json.loads(PREREG.read_text(encoding="utf-8"))
    for f, want in plan["source_sha256"]["audit"].items():
        if sha_file(ROOT / f) != want:
            raise SystemExit(f"audit source {f} changed since preregistration; refusing")
    for f, want in plan["source_sha256"]["v4"].items():
        if sha_file(ROOT / f) != want:
            raise SystemExit(f"V4 source {f} changed since preregistration; refusing")
    return {"sha256": h, "plan": plan}


# =============================================================================
# ARTIFACTS -- only after PREREG
# =============================================================================
def stage_artifacts():
    verify_prereg()
    from decoupled_qrc.v4_artifacts import ClaimGuard, load_frozen, sha256_of
    ad = K.V4Adapter(FROZEN_V4)
    base = ROOT / "results" / "v4"
    out = {}
    fc = load_frozen(FROZEN_V4)
    out["frozen_config"] = {"rehash_ok": True, "sha256": fc.sha256}
    rows = [json.loads(l) for l in (base / "confirmation_rows.jsonl").read_text().splitlines() if l.strip()]
    out["confirmation_rows"] = {
        "n_rows": len(rows), "n_expected": 5 * 5 * 5 * 3,
        "all_carry_frozen_hash": all(r.get("config_hash") == fc.sha256 for r in rows),
        "seeds_arch": sorted({r["arch_seed"] for r in rows}),
        "seeds_input": sorted({r["input_seed"] for r in rows})}
    # regenerate a sample of stored rows from CURRENT source
    from decoupled_qrc.v4_ipc import compute_metrics
    from decoupled_qrc.v4_architecture import run_v4
    rng = np.random.default_rng(0)
    sample = [rows[i] for i in rng.choice(len(rows), 6, replace=False)]
    regen = []
    for r in sample:
        u = np.random.default_rng(int(r["input_seed"])).uniform(-1, 1, 1600)
        res = compute_metrics(run_v4(ad.spec, u, m=r["m"], g=r["g"], seed=int(r["arch_seed"])),
                              ad.cfg, null_seed=int(r["input_seed"]))
        regen.append({"key": r["row_key"], "dM": abs(res.M - r["M"]), "dN": abs(res.N - r["N"]),
                      "dNL": abs(res.N_long - r["N_long"])})
    out["regeneration_from_current_source"] = {
        "rows": regen, "max_abs_diff": max(max(x["dM"], x["dN"], x["dNL"]) for x in regen)}
    search = json.loads((base / "search.json").read_text())
    dev_a = set(search.get("arch_seeds", [])); dev_i = set(search.get("input_seeds", []))
    stress = json.loads((base / "stress_test.json").read_text())
    dev_a |= set(stress.get("seeds", {}).get("arch", [])); dev_i |= set(stress.get("seeds", {}).get("input", []))
    out["discovery_confirmation_disjoint"] = bool(
        not (dev_a & set(out["confirmation_rows"]["seeds_arch"]))
        and not (dev_i & set(out["confirmation_rows"]["seeds_input"])))
    import inspect
    conf = json.loads((base / "confirmation.json").read_text())
    # can a claim be produced without a completed confirmation?
    g = ClaimGuard(gates=conf["gates"], confirmation_complete=False,
                   frozen_hash=fc.sha256, run_config_hash=fc.sha256)
    src = (HERE / "run_v4_stage.py").read_text()
    out["claim_guard"] = {
        "refuses_without_complete_confirmation": g.render() != "MEMORY-NONLINEARITY SEPARATION DEMONSTRATED",
        "provenance_check_is_tautological": "frozen_hash=fc.sha256, run_config_hash=fc.sha256" in src,
        "report_recomputes_claim": "ClaimGuard" in (HERE / "decoupled_qrc" / "v4_report.py").read_text(),
        "confirmation_json_hashed": False,
        "sequential_alpha_used_in_gate_CIs": "alpha_for_next" in inspect.getsource(
            __import__("decoupled_qrc.v4_experiment", fromlist=["x"])),
    }
    # computed, not asserted: compare the two stored columns row by row
    nl_vs_cdm = max(abs(r["N_long"] - r["cross_delay_mean"]) for r in rows)
    from decoupled_qrc import v4_statistics, v4_experiment
    holm_src = inspect.getsource(v4_statistics.holm_equivalence)
    out["gate_implementation_findings"] = {
        "combined_gate_max_abs(N_long - cross_delay_mean)": nl_vs_cdm,
        "combined_gate_two_of_three_metrics_identical": bool(nl_vs_cdm < 1e-12),
        "holm_equivalence_adjusts_interval_level": bool("level" in holm_src.split("rows = {}")[1]),
        "gate_ci_levels": {k: v4_experiment.THRESHOLDS[k] for k in
                           ("main_ci_level", "equivalence_ci_level", "combined_ci_level")},
        "ledger_alpha_per_look": 0.01,
        "v4_capacity_clipped_to_0_1": "np.clip(1.0 -" in (HERE / "decoupled_qrc" / "v4_ipc.py").read_text(),
    }
    out["manual_edit_detectable"] = {"frozen_config": "yes (re-hash on load)",
                                     "confirmation_rows": "only via regeneration (no per-row hash)",
                                     "confirmation.json / report.*": "no (unhashed)"}
    s6_dev = json.loads((OUT / "section6.json").read_text())
    s6_now = C.intrinsic_nonlinearity(ad, seeds=DEV_SEEDS[:5])
    out["section6_regenerated_from_current_source"] = {
        "same_verdict": s6_now["verdict"] == s6_dev["verdict"],
        "max_abs_diff_ols_and_ridge_curves": max(
            abs(float(s6_now[c][k]) - float(s6_dev[c][str(k)] if str(k) in s6_dev[c] else s6_dev[c][k]))
            for c in ("ols_raw_curve", "ols_std_curve") for k in s6_now[c])}
    write("artifact_audit.json", out)
    print(f"  regeneration max |diff| = {out['regeneration_from_current_source']['max_abs_diff']:.2e}; "
          f"rows {out['confirmation_rows']['n_rows']}/{out['confirmation_rows']['n_expected']}; "
          f"disjoint={out['discovery_confirmation_disjoint']}; "
          f"tautological provenance={out['claim_guard']['provenance_check_is_tautological']}")


# =============================================================================
# CONFIRM
# =============================================================================
def stage_confirm():
    pre = verify_prereg()
    t0 = time.time()
    ad = K.V4Adapter(FROZEN_V4)
    ctx = K.Context(alpha=ad.alpha)
    print(f"  prereg {pre['sha256'][:12]} verified; confirmation bank {CONF_SEEDS[0]}..{CONF_SEEDS[-1]}")
    rows = K.run_factorial(ad, m_values=GRID, g_values=GRID, seeds=CONF_SEEDS, ctx=ctx,
                           checkpoint_path=OUT / "conf_rows.jsonl", tag="conf")
    s6 = json.loads((OUT / "section6.json").read_text())
    st = json.loads((OUT / "structure.json").read_text())
    s6c = C.intrinsic_nonlinearity(ad, seeds=CONF_SEEDS[:5])
    write("section6_confirmation.json", s6c)
    gates_seq = all_gates(rows, K.Levels.sequential(), s6=s6c, st=st, lows=(0.0, 0.25),
                          highs=(0.75, 1.0))
    gates_def = all_gates(rows, K.Levels(), s6=s6c, st=st, lows=(0.0, 0.25), highs=(0.75, 1.0))
    cl = classical_finite_shots(ad, ctx, CONF_SEEDS[:8])
    write("classical_confirmation.json", cl)
    result = {"prereg_sha256": pre["sha256"], "n_rows": len(rows),
              "gates_sequential": gates_seq, "gates_default_levels": gates_def,
              "overall": overall(gates_seq), "overall_default_levels": overall(gates_def),
              "runtime_s": round(time.time() - t0, 1)}
    write("gates.json", result)
    for k in ("C1_structural", "C2_resource_constrained", "C3_intrinsic", "C4_combined",
              "C5_quantum_specific"):
        print(f"  {k:26s} {'PASS' if gates_seq[k]['passed'] else 'FAIL'}")
    print(f"  OVERALL: {result['overall']}")


def stage_report():
    from decoupled_qrc.audit_report import build
    build(OUT)


STAGES = {"ENV": stage_env, "DEV": stage_dev, "PREREG": stage_prereg,
          "ARTIFACTS": stage_artifacts, "CONFIRM": stage_confirm, "REPORT": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    a = ap.parse_args()
    print(f"=== V4 AUDIT {a.stage} ===")
    STAGES[a.stage]()
