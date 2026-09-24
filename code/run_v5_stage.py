"""
run_v5_stage.py -- V5 design iteration, judged by the SAME gates as the V4 audit.

    python run_v5_stage.py --stage SEARCH [--version V5.1]   # development seeds only
    python run_v5_stage.py --stage DEVGATES   # full dev gates + controls + robustness
    python run_v5_stage.py --stage FREEZE     # hash spec + sources + confirmation plan
    python run_v5_stage.py --stage CONFIRM    # ONE run on a fresh seed bank
    python run_v5_stage.py --stage REPORT

Gate code is imported unchanged from run_v4_audit (all_gates, principal).
FREEZE refuses unless every dev gate passed with the preregistered margins;
CONFIRM refuses unless the freeze re-hashes, sources are unchanged, and no
earlier confirmation of this version exists.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_v4_audit as W                              # noqa: E402
from decoupled_qrc import audit_checks as C          # noqa: E402
from decoupled_qrc import audit_core as K            # noqa: E402
from decoupled_qrc import v5_architecture as V       # noqa: E402

# Append-only lineage. V5 failed its freeze criterion on dev data (memory
# interior effect 0.070 < 0.10 under the delay-range-12 robustness condition:
# L = 6 cannot reach delays 7..12). V5.1 searches longer registers and puts
# that robustness condition INSIDE the search. The processor is V5's dev choice.
VERSIONS = {
    "V5": {"out": "v5", "robust_in_search": False,
           "space": {"L": (4, 6, 8), "p_max": (0.6, 0.75, 0.9, 1.0),
                     "theta_max_over_pi": (0.40, 0.42, 0.44), "phi_over_pi": (0.25, 1 / 3)}},
    "V5.1": {"out": "v5_1", "robust_in_search": True,
             "space": {"L": (8, 10, 12, 14), "p_max": (0.6, 0.75, 0.9),
                       "theta_max_over_pi": (0.44,), "phi_over_pi": (1 / 3,)}},
    # V5.1 met its freeze criterion but its MEMORY METRIC SATURATES: at m = 1,
    # 6/7 of M's constituent targets (d1, tau 2..8) exceed 0.995 and M_ols = 1.000.
    # The pooled saturation check (3.1%) hid it. V5.2 applies the audit's own
    # definition (ceiling 0.995, fraction <= 0.20) PER METRIC at every grid point,
    # plus metric headroom <= 0.95, in the search AND the freeze criterion.
    # Noiseless OLS inverts short SWAP transport exactly, so only blurred, short
    # registers are candidates.
    "V5.2": {"out": "v5_2", "robust_in_search": True, "strict_saturation": True,
             "space": {"L": (5, 6, 7, 8), "p_max": (0.8, 0.85, 0.9, 0.95),
                       "theta_max_over_pi": (0.44,), "phi_over_pi": (1 / 3,)}},
    # V5.2 found NO candidate: in noiseless OLS a random-SWAP register either
    # saturates (L >= 7: OLS inverts the transport for early delays) or cannot
    # reach tau 9..12 (interior effect 0.07-0.09 at tau_max 12). V5.3 reads only
    # every read_stride-th rail of a LONG chain: long reach, and total linear
    # capacity <= #read rails spread over more delays than rails -> no ceiling.
    # Dev probe (3 seeds): L16, stride 2, p_max 0.8 -> sat 0, M_max 0.55,
    # interior(0.5->1) 0.16 (tau 8) / 0.17 (tau 12). Criteria unchanged from V5.2.
    "V5.3": {"out": "v5_3", "robust_in_search": True, "strict_saturation": True,
             "space": {"L": (12, 16, 20), "p_max": (0.6, 0.65, 0.7, 0.75, 0.8),
                       "theta_max_over_pi": (0.44,), "phi_over_pi": (1 / 3,),
                       "read_stride": (2,)}},
    # V5.3: memory side works (no saturation, interior 0.155-0.168 at tau 8/12),
    # but EVERY finalist failed stress at theta_max x1.1: 0.484 pi makes the
    # processor feature almost pure u^2 (C2 ~ 0.997), saturating N. theta_max was
    # chosen in V5 before any saturation criterion existed; V5.4 re-searches the
    # processor jointly. Criteria unchanged.
    "V5.4": {"out": "v5_4", "robust_in_search": True, "strict_saturation": True,
             "space": {"L": (12, 16), "p_max": (0.65, 0.7, 0.75),
                       "theta_max_over_pi": (0.38, 0.40, 0.42), "phi_over_pi": (0.25, 1 / 3),
                       "read_stride": (2,)}},
}
SAT_CEILING, SAT_FRACTION, METRIC_HEADROOM = 0.995, 0.20, 0.95
ROBUST_SEARCH_MIN = 0.15     # V5.1: worst effect under the in-search robustness conditions


def configure(version: str):
    global VERSION, OUT, FROZEN, SEARCH_SPACE, ROBUST_IN_SEARCH, STRICT_SATURATION
    v = VERSIONS[version]
    STRICT_SATURATION = bool(v.get("strict_saturation", False))
    VERSION, OUT = version, ROOT / "results" / v["out"]
    FROZEN = OUT / f"frozen_{v['out']}.json"
    SEARCH_SPACE, ROBUST_IN_SEARCH = v["space"], v["robust_in_search"]


configure("V5.4")
GRID = W.GRID
SEARCH_GRID = (0.0, 0.25, 1.0)
DEV_SEEDS = [(40000 + k, 45000 + k) for k in range(10)]
CONF_SEEDS = [(90000 + k, 95000 + k) for k in range(20)]
SOURCES = ["code/decoupled_qrc/v5_architecture.py", "code/run_v5_stage.py",
           *W.AUDIT_SOURCES]
ALPHA = 0.5            # V4's preregistered ridge penalty, NOT tuned for V5

STRESS = (("p_max", 0.9), ("p_max", 1.1), ("theta_max", 0.9), ("theta_max", 1.1),
          ("phi", 0.9), ("phi", 1.1))
MARGIN = {"effect_min": 0.20,        # every main AND interior effect, 2x the gate
          "c1_adv_min": 0.05}        # HH - best other quadrant on C1
ALPHA_LEDGER = [
    {"look": 1, "what": "V4 original confirmation (results/v4)", "alpha": 0.01},
    {"look": 2, "what": "V4 independent-audit confirmation (results/v4/independent_audit)",
     "alpha": 0.01},
    {"look": 3, "what": "V5.x confirmation: the first V5-lineage version to pass its dev freeze "
                        "criterion (V5 did not; it never looked at confirmation data)",
     "alpha": 0.01},
]
SEARCH_READOUTS = {"R": K.METHODS, "P": K.METHODS, "ALL": ("ridge",)}


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=W._js),
                            encoding="utf-8")
    print(f"  wrote {(OUT / name).relative_to(ROOT)}")


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def spec_of(c: dict) -> V.V5Spec:
    return V.V5Spec(L=int(c["L"]), p_max=float(c["p_max"]),
                    theta_max=float(c["theta_max"]), phi=float(c["phi"]),
                    read_stride=int(c.get("read_stride", 1)))


def cand_dict(s: V.V5Spec) -> dict:
    d = {"L": s.L, "p_max": s.p_max, "theta_max": s.theta_max, "phi": s.phi}
    if s.read_stride != 1:                  # keeps V5-V5.2 artifacts byte-identical
        d["read_stride"] = s.read_stride
    return d


def worst_effect(pc: dict) -> float:
    return min(pc["dM_ridge"], pc["dM_ols"], pc["dM_interior_ols"],
               pc["dN_ridge"], pc["dN_ols"], pc["dN_interior_ols"])


def metric_saturation(rows, ctx) -> dict:
    """Per-metric saturation at every grid point, seed means, every analysis."""
    out, ok = {}, True
    for meth in K.METHODS:
        for name, ro, ctrl, keys in (
                ("M", "R", "m", [f"d1_t{t}" for t in range(ctx.tau_L, ctx.tau_max + 1)]),
                ("N", "P", "g", [f"d{d}_t{t}" for d in (2, 3, 4) for t in range(ctx.tau_S + 1)])):
            blk = f"{ro}|{meth}"
            if not rows or blk not in rows[0]["blocks"]:
                continue
            frac, top = 0.0, -np.inf
            for v in sorted({r[ctrl] for r in rows}):
                sel = [r for r in rows if r[ctrl] == v]
                caps = np.array([np.mean([r["blocks"][blk]["single"][k] for r in sel]) for k in keys])
                frac, top = max(frac, float(np.mean(caps > SAT_CEILING))), max(top, float(caps.mean()))
            good = bool(frac <= SAT_FRACTION and top <= METRIC_HEADROOM)
            out[f"{name}|{meth}"] = {"worst_constituent_saturated_fraction": frac,
                                      "max_metric_value": top, "ok": good}
            ok &= good
    out["ok"] = bool(ok)
    return out


def feasible(pc: dict) -> bool:
    if STRICT_SATURATION and not pc["metric_saturation"]["ok"]:
        return False
    return bool(worst_effect(pc) >= MARGIN["effect_min"]
                and pc["C1_HH_minus_best"] >= MARGIN["c1_adv_min"]
                and pc["M_spread_over_g"] == 0 and pc["N_spread_over_m"] == 0
                and pc["saturation"]["saturation_ok"] and pc["saturation"]["sentinel_ok"])


def evaluate(spec, seeds, grid, tag, ctx=None, readouts=SEARCH_READOUTS, adapter=None):
    ad = adapter or V.V5Adapter(spec, alpha=ALPHA)
    ctx = ctx or K.Context(alpha=ALPHA)
    rows = K.run_factorial(ad, m_values=grid, g_values=grid, seeds=seeds, ctx=ctx,
                           checkpoint_path=OUT / "search" / f"{tag}.jsonl", tag=tag,
                           readouts=readouts, verbose=False)
    pc = W.principal(rows, grid, ctx)
    pc["metric_saturation"] = metric_saturation(rows, ctx)
    return pc


# =============================================================================
# SEARCH -- dev seeds only; rule fixed above before this ran
# =============================================================================
def stage_search():
    t0 = time.time()
    seeds = DEV_SEEDS[:5]
    res = []
    for L, pm, th, ph, *rest in itertools.product(*SEARCH_SPACE.values()):
        st = rest[0] if rest else 1
        s = V.V5Spec(L=L, p_max=pm, theta_max=th * np.pi, phi=ph * np.pi, read_stride=st)
        tag = f"L{L}_p{pm}_t{th}_f{ph:.4f}" + (f"_s{st}" if rest else "")
        pc = evaluate(s, seeds, SEARCH_GRID, tag)
        rec = {"candidate": cand_dict(s), "tag": tag, "nominal": pc,
               "worst_effect": worst_effect(pc), "feasible_nominal": feasible(pc)}
        if ROBUST_IN_SEARCH and rec["feasible_nominal"]:
            rec["in_search_robustness"] = {}
            for name, ctx in (("interior_0.5_tau8", K.Context(alpha=ALPHA)),
                              ("interior_0.5_tau12", K.Context(tau_max=12, alpha=ALPHA))):
                rp = evaluate(s, seeds, W.ROBUST_GRID, f"{tag}_{name}", ctx=ctx)
                rec["in_search_robustness"][name] = {"worst_effect": worst_effect(rp),
                                                     "C1_HH_minus_best": rp["C1_HH_minus_best"],
                                                     "saturation_ok": rp["metric_saturation"]["ok"]}
            rec["feasible_nominal"] = bool(all(
                v["worst_effect"] >= ROBUST_SEARCH_MIN and v["C1_HH_minus_best"] > 0
                and (v["saturation_ok"] or not STRICT_SATURATION)
                for v in rec["in_search_robustness"].values()))
            rec["worst_effect"] = min([rec["worst_effect"]] + [
                v["worst_effect"] for v in rec["in_search_robustness"].values()])
        res.append(rec)
        print(f"    {tag:28s} worst={rec['worst_effect']:+.3f} C1adv={pc['C1_HH_minus_best']:+.3f} "
              f"feasible={rec['feasible_nominal']}", flush=True)
    # stress the 6 best feasible candidates at +-10% on every continuous parameter
    top = sorted([r for r in res if r["feasible_nominal"]],
                 key=lambda r: (-r["worst_effect"], r["candidate"]["L"]))[:6]
    for r in top:
        s = spec_of(r["candidate"])
        r["stress"] = {}
        for field, f in STRESS:
            val = getattr(s, field) * f
            if field == "p_max":
                val = min(val, 1.0)
            if field == "theta_max":
                val = min(val, 0.499 * np.pi)
            sp = V.V5Spec(**{**cand_dict(s), field: val})
            pc = evaluate(sp, seeds, SEARCH_GRID, f"{r['tag']}_{field}{f}")
            r["stress"][f"{field}x{f}"] = {"worst_effect": worst_effect(pc),
                                           "feasible": feasible(pc),
                                           "C1_HH_minus_best": pc["C1_HH_minus_best"]}
        r["worst_case_effect"] = min([r["worst_effect"]] +
                                     [v["worst_effect"] for v in r["stress"].values()])
        r["robust_feasible"] = bool(all(v["feasible"] for v in r["stress"].values()))
        print(f"    stress {r['tag']:28s} worst-case={r['worst_case_effect']:+.3f} "
              f"robust={r['robust_feasible']}", flush=True)
    ok = [r for r in top if r["robust_feasible"]]
    chosen = max(ok, key=lambda r: (r["worst_case_effect"], -r["candidate"]["L"])) if ok else None
    write("search.json", {"version": VERSION, "robust_in_search": ROBUST_IN_SEARCH,
                          "robust_search_min": ROBUST_SEARCH_MIN, "seeds": seeds, "grid": SEARCH_GRID, "space": SEARCH_SPACE,
                          "stress": STRESS, "margins": MARGIN, "ridge_alpha": ALPHA,
                          "rule": "among nominal-feasible, stress the 6 with the largest worst "
                                  "effect; choose the robust-feasible one with the largest "
                                  "worst-case effect; tie -> smaller L; remaining ties -> "
                                  "enumeration order (smallest p_max first)",
                          "results": res, "chosen": chosen and chosen["candidate"],
                          "runtime_s": round(time.time() - t0, 1)})
    print(f"  chosen: {chosen and chosen['candidate']}")


# =============================================================================
# DEVGATES -- full gates on dev seeds, controls, robustness
# =============================================================================
def chosen_adapter(kind=None) -> V.V5Adapter:
    return V.V5Adapter(spec_of(read("search.json")["chosen"]), alpha=ALPHA, kind=kind)


def structure(seed) -> dict:
    make = lambda kind: chosen_adapter(kind)        # noqa: E731
    st = {k or "frozen": C.structural_dependency_adapter(make, n_draws=100, seed=seed, kind=k)
          for k in (None, "g_into_R", "m_into_P", "serial")}
    st["indirect_inspection"] = [
        {"channel": "shared seeds", "finding": "V5 has no disorder; the seed selects only "
                                               "the input sequence", "risk": "none"},
        {"channel": "shared normalization / preprocessing / caches / feature selection",
         "finding": "none: R and P are computed by separate functions from u only", "risk": "none"},
        {"channel": "joint features", "finding": "J = z_r * f: both controls by design, never "
                                                 "used for M or N", "risk": "none for M/N"}]
    return st


def invalid_controls(ad, ctx) -> dict:
    out = {}
    st = structure(21)
    for k in ("g_into_R", "m_into_P", "serial"):
        v = st[k]
        out[k] = {"detected": not v["isolated"], "detector": "structural dependency test",
                  "evidence": {x: v[x] for x in ("max_dXR_dg", "max_dXP_dm")}}
    for kind, detector in (("leak_targets", "unreachable-class + future-sentinel"),
                           ("feature_count_changes", "feature-count invariance")):
        rows = K.run_factorial(ad, m_values=(0.0, 1.0), g_values=(0.0, 1.0),
                               seeds=DEV_SEEDS[:3], ctx=ctx,
                               checkpoint_path=OUT / f"invalid_{kind}.jsonl", tag=kind,
                               feature_hook=C.leak_hook(kind), verbose=False)
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
            out[kind] = {"detected": not fc["invariant"], "detector": detector, "evidence": fc}
    bad = type(ctx.split)(ctx.split.train, ctx.split.train)
    out["reuse_train_as_test"] = {"detected": bool(not bad.check()["disjoint"]),
                                  "detector": "split integrity", "evidence": bad.check()}
    out["all_detected"] = bool(all(v["detected"] for v in out.values() if isinstance(v, dict)))
    return out


def robustness(ad) -> dict:
    seeds = DEV_SEEDS[:5]
    G = W.ROBUST_GRID
    b = dict(alpha=ALPHA)
    conds = {
        "baseline": (ad, K.Context(**b), G),
        "longer_T3200": (ad, K.Context(T=3200, **b), G),
        "half_training": (ad, K.Context(T=1110, train_frac=550 / 1110, **b), G),
        "double_training": (ad, K.Context(T=2580, train_frac=2020 / 2580, **b), G),
        "float32_readout": (ad, K.Context(dtype="float32", **b), G),
        "perturbed_controls": (ad, K.Context(**b), (0.03, 0.47, 0.97)),
        "ridge_0.05": (ad, K.Context(alpha=0.05), G),
        "ridge_5.0": (ad, K.Context(alpha=5.0), G),
        "delay_range_12": (ad, K.Context(tau_max=12, **b), G),
        "degree_6": (ad, K.Context(max_degree=6, **b), G),
        "shots_1000": (V.V5ShotAdapter(ad, 1000), K.Context(**b), G),
        "shots_10000": (V.V5ShotAdapter(ad, 10000), K.Context(**b), G),
    }
    ibm = C.ibm_noise_parameters("FakeTorino")
    if ibm.get("available"):
        conds["ibm_noise_FakeTorino"] = (V.ibm_informed(ad, ibm), K.Context(**b), G)
    out = {"ibm_noise_parameters": ibm, "conditions": {}}
    for name, (adp, ctx, grid) in conds.items():
        rows = K.run_factorial(adp, m_values=grid, g_values=grid, seeds=seeds, ctx=ctx,
                               checkpoint_path=OUT / f"robust_{name}.jsonl", tag=name,
                               verbose=False)
        pc = W.principal(rows, grid, ctx)
        pc["metric_saturation"] = metric_saturation(rows, ctx)
        pc["passes_gate_thresholds"] = bool(worst_effect(pc) >= 0.10 and pc["C1_HH_minus_best"] > 0
                                            and pc["M_spread_over_g"] == 0
                                            and pc["N_spread_over_m"] == 0)
        out["conditions"][name] = pc
        print(f"      {name:22s} worst={worst_effect(pc):+.3f} dM={pc['dM_ridge']:+.3f} "
              f"dN={pc['dN_ridge']:+.3f} dN_int_ols={pc['dN_interior_ols']:+.3f} "
              f"C1adv={pc['C1_HH_minus_best']:+.3f} ok={pc['passes_gate_thresholds']}",
              flush=True)
    drops = {}
    for j in range(len(ad.spec.read_rails)):     # read rails only (stride-aware)
        def hook(X, *, u, m, g, j=j):
            X = dict(X)
            X["R"] = np.delete(X["R"], j, axis=1)
            return X
        rows = K.run_factorial(ad, m_values=(0.0, 1.0), g_values=(0.0, 1.0), seeds=seeds[:3],
                               ctx=K.Context(**b), checkpoint_path=OUT / f"robust_drop_R{j}.jsonl",
                               tag=f"drop_R{j}", readouts={"R": ("ridge",)}, feature_hook=hook,
                               verbose=False)
        gM = K.per_seed_grid(rows, K.metric_key("R", "ridge", "M"))
        drops[f"R{j}"] = {"dM": float(np.mean(K.effect_per_seed(gM, "m")))}
    drops["P0"] = "not applicable: P has ONE observable by design (dropping it removes N)"
    out["drop_one_observable"] = drops
    return out


def dev_pass(g: dict, rob: dict, inv: dict, sat: dict = None) -> dict:
    """The freeze criterion. Every item must hold."""
    c2, c3 = g["C2_resource_constrained"], g["C3_intrinsic"]
    effects = {"main_m_M_ridge": c2["main_m_M"]["estimate"],
               "main_g_N_ridge": c2["main_g_N"]["estimate"],
               "main_m_M_ols_std": c3["ols_std"]["main_m_M"]["estimate"],
               "main_g_N_ols_std": c3["ols_std"]["main_g_N"]["estimate"],
               "main_m_M_ols_raw": c3["ols_raw"]["main_m_M"]["estimate"],
               "main_g_N_ols_raw": c3["ols_raw"]["main_g_N"]["estimate"],
               "interior_m_M": c3["interior_m_M"]["estimate"],
               "interior_g_N": c3["interior_g_N"]["estimate"]}
    items = {"C1_structural": g["C1_structural"]["passed"],
             "C2_resource_constrained": c2["passed"], "C3_intrinsic": c3["passed"],
             "C4_combined": g["C4_combined"]["passed"],
             "effects_with_margin": all(v >= MARGIN["effect_min"] for v in effects.values()),
             "invalid_controls_all_detected": inv["all_detected"],
             "robustness_all_conditions": all(v["passes_gate_thresholds"]
                                              for v in rob["conditions"].values())}
    if STRICT_SATURATION:
        items["no_metric_saturation_dev"] = bool(sat["ok"])
        items["no_metric_saturation_robustness"] = all(
            v["metric_saturation"]["ok"] for k, v in rob["conditions"].items()
            if not k.startswith(("shots", "ibm")))
    return {"items": items, "effects": effects, "passed": bool(all(items.values()))}


def stage_devgates():
    t0 = time.time()
    ad = chosen_adapter()
    ctx = K.Context(alpha=ALPHA)
    print(f"  candidate {ad.describe()['spec']}")
    enc = V.encoder_only_capacity(ad.spec)
    write("encoder.json", enc)
    st = structure(11)
    write("structure.json", st)
    s6 = C.intrinsic_nonlinearity(ad, seeds=DEV_SEEDS[:5])
    write("section6.json", s6)
    print(f"  section 6: {s6['verdict']}")
    rows = K.run_factorial(ad, m_values=GRID, g_values=GRID, seeds=DEV_SEEDS, ctx=ctx,
                           checkpoint_path=OUT / "dev_rows.jsonl", tag="dev")
    inv = invalid_controls(ad, ctx)
    write("invalid_controls.json", inv)
    rob = robustness(ad)
    write("robustness.json", rob)
    g = W.all_gates(rows, K.Levels.sequential(), s6=s6, st=st, lows=(0.0, 0.25),
                    highs=(0.75, 1.0))
    sat = metric_saturation(rows, ctx)
    dp = dev_pass(g, rob, inv, sat)
    write("dev_gates.json", {"gates_sequential": g, "freeze_criterion": dp,
                             "metric_saturation": sat,
                             "encoder_operational_nonlinear_capacity":
                                 enc["3_operational_all"]["nonlinear_capacity"],
                             "runtime_s": round(time.time() - t0, 1)})
    for k, v in dp["items"].items():
        print(f"  {k:34s} {'PASS' if v else 'FAIL'}")
    print(f"  effects: " + ", ".join(f"{k}={v:.3f}" for k, v in dp["effects"].items()))
    print(f"  FREEZE CRITERION: {'MET' if dp['passed'] else 'NOT MET'}")


# =============================================================================
# FREEZE / CONFIRM
# =============================================================================
def stage_freeze():
    dg = read("dev_gates.json")
    if not dg["freeze_criterion"]["passed"]:
        raise SystemExit("dev freeze criterion NOT met; redesign on dev data (new version id)")
    used = set()
    for a, i in W.DEV_SEEDS + W.CONF_SEEDS + DEV_SEEDS:
        used |= {a, i}
    for v in W.USED_BEFORE.values():
        used |= set(v)
    conf = {s for p in CONF_SEEDS for s in p}
    if used & conf:
        raise SystemExit(f"confirmation bank overlaps used seeds: {sorted(used & conf)[:5]}")
    ad = chosen_adapter()
    payload = {"version": VERSION, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "spec": cand_dict(ad.spec), "describe": ad.describe(), "ridge_alpha": ALPHA,
               "context": K.Context(alpha=ALPHA).as_dict(), "grid": list(GRID),
               "confirmation_seeds": CONF_SEEDS, "previously_used_seed_count": len(used),
               "source_sha256": {f: W.sha_file(ROOT / f) for f in SOURCES},
               "gate_levels_for_verdict": "sequential (alpha 0.01 per look)",
               "alpha_ledger": ALPHA_LEDGER,
               "gates": "run_v4_audit.all_gates, unchanged (claims 1-5)",
               "success_definition": "claims 1-4 pass on confirmation at sequential levels; "
                                     "claim 5 reported, not required",
               "dev_results_sha256": W.sha_file(OUT / "dev_gates.json"),
               "lineage_not_frozen": {
                   "V5": "freeze criterion not met: memory interior effect 0.070 < 0.10 under "
                         "delay range 12 (L = 6 cannot reach tau 7..12)",
                   "V5.1": "met its criterion but memory metric saturated (6/7 constituents "
                           "> 0.995 at m = 1, M_ols = 1.000); strict per-metric saturation "
                           "criterion added",
                   "V5.2": "search found no candidate (saturation vs reach trade-off)",
                   "V5.3": "every finalist failed theta_max x1.1 stress (N saturated)"},
               "disclosure": "all V5.x searches reused the same DEVELOPMENT seeds "
                             "(40000.., 45000..); no V5.x version has seen any confirmation "
                             "seed; every criterion change across V5.x made the criteria "
                             "stricter, none weaker"}
    blob = {"payload": payload, "sha256": W.sha_obj(payload)}
    FROZEN.write_text(json.dumps(blob, indent=2, sort_keys=True, default=W._js), encoding="utf-8")
    print(f"  frozen {VERSION}: sha256 {blob['sha256']}")


def verify_frozen() -> dict:
    blob = json.loads(FROZEN.read_text(encoding="utf-8"))
    if W.sha_obj(blob["payload"]) != blob["sha256"]:
        raise SystemExit("frozen artifact modified after freezing")
    # A source may differ from the freeze ONLY through a recorded, hashed
    # amendment that names the exact from -> to hashes (implementation bug fixes).
    amended = {}
    for a in sorted(OUT.glob("amendment_*.json")):
        if W.sha_file(a) != a.with_suffix(".sha256").read_text().strip():
            raise SystemExit(f"amendment {a.name} modified after it was recorded")
        for f, ch in json.loads(a.read_text(encoding="utf-8"))["source_changes"].items():
            amended.setdefault(f, set()).add((ch["from"], ch["to"]))
    for f, want in blob["payload"]["source_sha256"].items():
        now = W.sha_file(ROOT / f)
        if now != want and (want, now) not in amended.get(f, set()):
            raise SystemExit(f"source {f} changed since freeze; refusing")
    return blob


def stage_confirm():
    blob = verify_frozen()
    if (OUT / "gates.json").exists():
        raise SystemExit("confirmation already ran for this version; archive it and bump the version")
    t0 = time.time()
    p = blob["payload"]
    ad = V.V5Adapter(spec_of(p["spec"]), alpha=p["ridge_alpha"])
    ctx = K.Context(alpha=p["ridge_alpha"])
    rows = K.run_factorial(ad, m_values=GRID, g_values=GRID, seeds=CONF_SEEDS, ctx=ctx,
                           checkpoint_path=OUT / "conf_rows.jsonl", tag="conf")
    s6 = C.intrinsic_nonlinearity(ad, seeds=CONF_SEEDS[:5])
    write("section6_confirmation.json", s6)
    st = structure(31)
    write("structure_confirmation.json", st)
    gs = W.all_gates(rows, K.Levels.sequential(), s6=s6, st=st, lows=(0.0, 0.25), highs=(0.75, 1.0))
    gd = W.all_gates(rows, K.Levels(), s6=s6, st=st, lows=(0.0, 0.25), highs=(0.75, 1.0))
    cls = classical_baseline(ad, ctx, CONF_SEEDS[:8])
    write("classical_confirmation.json", cls)
    ov = W.overall(gs).replace("V4", VERSION)
    success = all(gs[k]["passed"] for k in ("C1_structural", "C2_resource_constrained",
                                            "C3_intrinsic", "C4_combined"))
    write("gates.json", {"frozen_sha256": blob["sha256"], "n_rows": len(rows),
                         "gates_sequential": gs, "gates_default_levels": gd,
                         "overall": ov, "overall_default_levels": W.overall(gd).replace("V4", VERSION),
                         "success_claims_1_to_4": success,
                         "runtime_s": round(time.time() - t0, 1)})
    for k in ("C1_structural", "C2_resource_constrained", "C3_intrinsic", "C4_combined",
              "C5_quantum_specific"):
        print(f"  {k:26s} {'PASS' if gs[k]['passed'] else 'FAIL'}")
    print(f"  OVERALL: {ov}; success (claims 1-4) = {success}")


def classical_baseline(ad, ctx, seeds, shots=(100, 1000, 10000)) -> dict:
    """Quantum per-shot joint estimator vs classical product of marginals, same shots.
    Noiselessly J == R*P exactly (asserted), so any difference is estimator variance."""
    lib = ctx.lib
    c1 = np.array([t.cls == "C1_curNL_x_oldLin" for t in lib])
    sun = np.array([t.cls == "sunada" for t in lib])
    out = {"exact": [], "shots": {}}
    for sp in seeds:
        u, Y, _ = K._targets(ctx, sp[1])
        f = ad.run(u, 1.0, 1.0, sp[0])
        XC = f["R"] * f["P"]
        out["exact"].append({"max_abs_J_minus_classical": float(np.abs(f["J"] - XC).max())})
        for S in shots:
            rng = np.random.default_rng(sp[0] * 7 + int(S))
            q, c = C.joint_estimators(f["R"], f["P"], [(r, 0) for r in range(len(ad.spec.read_rails))], S, rng)
            sq = K.A.score_all(q, Y, lib, ctx.split, "ridge", alpha=ctx.alpha)
            sc = K.A.score_all(c, Y, lib, ctx.split, "ridge", alpha=ctx.alpha)
            out["shots"].setdefault(f"S{S}", []).append(
                {"quantum_joint_C1": float(np.mean(sq[c1])), "classical_product_C1": float(np.mean(sc[c1])),
                 "quantum_joint_sunada": float(np.mean(sq[sun])),
                 "classical_product_sunada": float(np.mean(sc[sun]))})
    L = len(ad.spec.read_rails)                 # observables actually read
    out["resources"] = {
        "quantum_joint": {"observables": L, "measurement_settings": 1,
                          "circuits_per_step": 1, "qubits": ad.spec.resources()["qubits"]},
        "classical_product": {"observables_measured": L + 1, "measurement_settings": 1,
                              "circuits_per_step": 1, "note": "same shots; products in post-processing"},
        "energy": "not available: no hardware run"}
    return out


# =============================================================================
# REPORT
# =============================================================================
def stage_report():
    from decoupled_qrc.v5_report import build
    build(OUT)


STAGES = {"SEARCH": stage_search, "DEVGATES": stage_devgates, "FREEZE": stage_freeze,
          "CONFIRM": stage_confirm, "REPORT": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--version", default="V5.4", choices=sorted(VERSIONS))
    a = ap.parse_args()
    configure(a.version)
    print(f"=== {VERSION} {a.stage} ===")
    STAGES[a.stage]()
