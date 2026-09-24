"""
v6_gates.py -- the preregistered V6 gates, one function for dev and confirmation.

Gate numbers follow the mandatory list of the V6 brief. Thresholds are the V5
thresholds (minimums) plus the new all-four-classes combined gate. Every CI level
is derived from the per-look alpha allotted by the cumulative alpha ledger:
    main-effect lower bound      quantile alpha
    TOST interval                [alpha, 1 - alpha]
    cross/main ratio upper bound quantile 1 - alpha
    profile TOST (k members)     Bonferroni alpha / k
    class HH contrasts           Bonferroni alpha / (4 classes x 3 contrasts x 3 readouts)
"""
from __future__ import annotations

import numpy as np

from .v6_core import METHODS, V6_CLASSES, boot_q, effect_per_seed, grid_by_seed, quadrant_values

THRESH = {"main_min": 0.10, "equivalence": 0.03, "ratio_max": 0.20, "support_margin": 0.02,
          "sat_ceiling": 0.995, "sat_fraction": 0.20, "metric_headroom": 0.95,
          "sentinel_margin": 0.02, "seed_consistency": 0.90}
DEV_MARGIN = {"effect_min": 0.20, "support_margin": 0.04, "hh_adv_min": 0.05,
              "robust_effect_min": 0.15, "robust_hh_adv_min": 0.02}
LOWS, HIGHS = (0.0, 0.25), (0.75, 1.0)
N_CONTRASTS = len(V6_CLASSES) * 3 * len(METHODS)


def alpha_for_attempt(k: int) -> float:
    """V6 confirmation attempt k (1-based): 0.01, 0.005, 0.0025, ... (sum <= 0.02)."""
    return 0.01 / 2 ** (k - 1)


def _metric(ro, meth, name):
    return lambda r: r["blocks"][f"{ro}|{meth}"]["metrics"][name]


def _single(ro, meth, d, tau):
    return lambda r: r["blocks"][f"{ro}|{meth}"]["single"][f"d{d}_t{tau}"]


def main_effect(rows, ro, meth, name, control, alpha, *, lo=None, hi=None, seed=0):
    eff = effect_per_seed(grid_by_seed(rows, _metric(ro, meth, name)), control, lo, hi)
    b = boot_q(eff, [alpha], seed=seed)
    lb = b["q"][repr(float(alpha))]
    frac = float(np.mean(np.asarray(eff) >= THRESH["main_min"]))
    return {"estimate": b["point"], "lower_bound": lb, "alpha": alpha, "n_seeds": b["n"],
            "per_seed": eff, "seed_fraction_above_min": frac,
            "passed": bool(b["point"] >= THRESH["main_min"] and lb > 0
                           and frac >= THRESH["seed_consistency"])}


def cross_effect(rows, meth, alpha, *, seed=0):
    """m -> N and g -> M: TOST inside +-0.03 and ratio UB < 0.20, both directions."""
    out = {}
    for name, (ro_c, n_c, ctl), (ro_m, n_m, ctl_m) in (
            ("m_to_N", ("P", "N", "m"), ("R", "M", "m")),
            ("g_to_M", ("R", "M", "g"), ("P", "N", "g"))):
        cross = np.asarray(effect_per_seed(grid_by_seed(rows, _metric(ro_c, meth, n_c)), ctl))
        main = np.asarray(effect_per_seed(grid_by_seed(rows, _metric(ro_m, meth, n_m)), ctl_m))
        b = boot_q(cross, [alpha, 1 - alpha], seed=seed)
        lo, hi = b["q"][repr(float(alpha))], b["q"][repr(float(1 - alpha))]
        rng = np.random.default_rng(seed + 1)
        idx = rng.integers(0, len(cross), size=(100_000, len(cross)))
        ratio = np.abs(cross[idx].mean(1)) / np.maximum(np.abs(main[idx].mean(1)), 1e-12)
        rhi = float(np.quantile(ratio, 1 - alpha))
        eq = bool(lo >= -THRESH["equivalence"] and hi <= THRESH["equivalence"])
        out[name] = {"estimate": b["point"], "interval": [lo, hi], "equivalent": eq,
                     "ratio_upper": rhi, "passed": bool(eq and rhi < THRESH["ratio_max"])}
    return out


def profile_equivalence(rows, ro, meth, members, control, alpha, *, seed=0):
    k = len(members)
    res, ok = {}, True
    for (d, tau) in members:
        v = effect_per_seed(grid_by_seed(rows, _single(ro, meth, d, tau)), control)
        b = boot_q(v, [alpha / k, 1 - alpha / k], seed=seed + 10 * d + tau)
        lo, hi = b["q"][repr(float(alpha / k))], b["q"][repr(float(1 - alpha / k))]
        eq = bool(lo >= -THRESH["equivalence"] and hi <= THRESH["equivalence"])
        ok &= eq
        res[f"d{d}_t{tau}"] = {"estimate": b["point"], "interval": [lo, hi], "equivalent": eq}
    return {"members": res, "alpha_per_member": alpha / k, "passed": bool(ok)}


def metric_saturation(rows, ctx) -> dict:
    out, ok = {}, True
    for meth in METHODS:
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
                frac = max(frac, float(np.mean(caps > THRESH["sat_ceiling"])))
                top = max(top, float(caps.mean()))
            good = bool(frac <= THRESH["sat_fraction"] and top <= THRESH["metric_headroom"])
            out[f"{name}|{meth}"] = {"worst_constituent_saturated_fraction": frac,
                                      "max_metric_value": top, "ok": good}
            ok &= good
    out["ok"] = bool(ok)
    return out


def combined_class(rows, cls, meth, alpha, *, lows=LOWS, highs=HIGHS, seed=0) -> dict:
    """The seven per-class criteria of the V6 combined gate, for one readout."""
    key = f"ALL|{meth}"
    q = quadrant_values(rows, key, lambda b: b["class"][cls], lows, highs)
    qm = {k: float(np.nanmean(v)) for k, v in q.items()}
    null99 = float(np.mean([r["blocks"][key]["null_q99"] for r in rows if key in r["blocks"]]))
    best = max(qm.values())
    supported = bool(best > null99 + THRESH["support_margin"])
    a_c = alpha / N_CONTRASTS
    contrasts, beats = {}, True
    for j, other in enumerate(("LL", "LH", "HL")):
        b = boot_q(q["HH"] - q[other], [a_c], seed=seed + 97 * j)
        lb = b["q"][repr(float(a_c))]
        contrasts[f"HH_minus_{other}"] = {"estimate": b["point"], "lower_bound": lb}
        beats &= bool(np.isfinite(lb) and lb > 0)
    per_seed_adv = q["HH"] - np.maximum(np.maximum(q["LL"], q["LH"]), q["HL"])
    consistency = float(np.mean(per_seed_adv > 0))
    # saturation: constituent targets in every quadrant, seed means
    sat_frac, top = 0.0, -np.inf
    for ms, gs in ((lows, lows), (lows, highs), (highs, lows), (highs, highs)):
        sel = [r for r in rows if r["m"] in ms and r["g"] in gs and key in r["blocks"]]
        if not sel:
            continue
        mem = np.mean([r["blocks"][key]["members"][cls] for r in sel], axis=0)
        sat_frac = max(sat_frac, float(np.mean(mem > THRESH["sat_ceiling"])))
        top = max(top, float(mem.mean()))
    not_sat = bool(sat_frac <= THRESH["sat_fraction"] and top <= THRESH["metric_headroom"])
    hh_adv = qm["HH"] - max(qm["LL"], qm["LH"], qm["HL"])
    return {"quadrant_means": qm, "null_q99": null99, "best": best, "supported": supported,
            "contrasts": contrasts, "alpha_per_contrast": a_c, "HH_beats_all": beats,
            "HH_best_point": bool(hh_adv > 0), "HH_advantage": hh_adv,
            "seed_consistency": consistency, "constituent_saturated_fraction": sat_frac,
            "class_max_mean": top, "not_saturated": not_sat,
            "passed": bool(supported and beats and hh_adv > 0 and not_sat
                           and consistency >= THRESH["seed_consistency"])}


def sentinels(rows) -> dict:
    out, ok = {}, True
    for cls in ("SENTINEL_future", "SENTINEL_unreachable"):
        worst = -np.inf
        for r in rows:
            for meth in METHODS:
                b = r["blocks"].get(f"ALL|{meth}")
                if b:
                    worst = max(worst, b["class"][cls + "__max"] - b["null_q99"])
        good = bool(worst <= THRESH["sentinel_margin"])
        out[cls] = {"max_excess_over_null99": float(worst), "ok": good}
        ok &= good
    out["ok"] = bool(ok)
    return out


def metric_level_independence(rows) -> dict:
    worst = {}
    for meth in METHODS:
        byM, byN = {}, {}
        for r in rows:
            k = (r["arch_seed"], r["input_seed"])
            byM.setdefault((k, r["m"]), []).append(r["blocks"][f"R|{meth}"]["metrics"]["M"])
            byN.setdefault((k, r["g"]), []).append(r["blocks"][f"P|{meth}"]["metrics"]["N"])
        worst[meth] = {"M_spread_over_g": max(max(v) - min(v) for v in byM.values()),
                       "N_spread_over_m": max(max(v) - min(v) for v in byN.values())}
    return worst


def feature_count_invariance(rows) -> dict:
    counts = {}
    for r in rows:
        for ro, n in r["feature_counts"].items():
            counts.setdefault(ro, set()).add(n)
    return {"per_readout": {k: sorted(v) for k, v in counts.items()},
            "invariant": bool(all(len(v) == 1 for v in counts.values()))}


def encoder_leakage(rows, exact: dict) -> dict:
    """(a) exact: at g = 0 every ALL feature is affine in the input history
    (checked by the caller on the architecture); (b) empirical, as preregistered:
    at g = 0 no nonlinear CLASS (class score = mean member capacity, the
    preregistered class score) exceeds null99 + 0.02 on any readout.

    Amendment 01 (results/v6/amendments/gates_amendment_01.json): the original
    code used the MAX member over each class and also scanned x_P1P1 / x_P2P1,
    which is stricter than the preregistered text; that statistic is kept below
    as a reported, non-gating diagnostic."""
    worst_cls, worst_max = -np.inf, -np.inf
    for r in rows:
        if r["g"] != 0.0:
            continue
        for meth in METHODS:
            b = r["blocks"][f"ALL|{meth}"]
            for c in V6_CLASSES:
                worst_cls = max(worst_cls, b["class"][c] - b["null_q99"])
            for c in V6_CLASSES + ("x_P1P1", "x_P2P1"):
                worst_max = max(worst_max, b["class"][c + "__max"] - b["null_q99"])
    emp = bool(worst_cls <= THRESH["sentinel_margin"])
    return {"exact": exact, "empirical_class_score_max_excess_at_g0": float(worst_cls),
            "diagnostic_max_member_excess_at_g0": float(worst_max),
            "passed": bool(exact.get("passed", False) and emp)}


def all_gates(rows, ctx, alpha, *, st, s6, enc_exact, inv, tests_passed, seed=0) -> dict:
    G = {}
    mains = {}
    for meth in METHODS:
        mains[meth] = {
            "m_M": main_effect(rows, "R", meth, "M", "m", alpha, seed=seed + 1),
            "g_N": main_effect(rows, "P", meth, "N", "g", alpha, seed=seed + 2),
            "interior_m_M": main_effect(rows, "R", meth, "M", "m", alpha, lo=0.25, hi=1.0,
                                        seed=seed + 3),
            "interior_g_N": main_effect(rows, "P", meth, "N", "g", alpha, lo=0.25, hi=1.0,
                                        seed=seed + 4),
            "cross": cross_effect(rows, meth, alpha, seed=seed + 5),
            "degree_profile_m": profile_equivalence(rows, "P", meth, [(d, 0) for d in (2, 3, 4)],
                                                    "m", alpha, seed=seed + 6),
            "memory_curve_g": profile_equivalence(rows, "R", meth, [(1, t) for t in range(9)],
                                                  "g", alpha, seed=seed + 7)}
    G["01_main_m_M"] = all(mains[m]["m_M"]["passed"] for m in METHODS)
    G["02_main_g_N"] = all(mains[m]["g_N"]["passed"] for m in METHODS)
    G["03_cross_m_N_equivalent"] = all(mains[m]["cross"]["m_to_N"]["equivalent"] for m in METHODS)
    G["04_cross_g_M_equivalent"] = all(mains[m]["cross"]["g_to_M"]["equivalent"] for m in METHODS)
    G["05_cross_main_ratio"] = all(mains[m]["cross"][k]["ratio_upper"] < THRESH["ratio_max"]
                                   for m in METHODS for k in ("m_to_N", "g_to_M"))
    G["06_degree_profile_under_m"] = all(mains[m]["degree_profile_m"]["passed"] for m in METHODS)
    G["07_memory_curve_under_g"] = all(mains[m]["memory_curve_g"]["passed"] for m in METHODS)
    for i, meth in ((8, "ols_raw"), (9, "ols_std"), (10, "ridge")):
        v = mains[meth]
        G[f"{i:02d}_separation_{meth}"] = bool(
            v["m_M"]["passed"] and v["g_N"]["passed"] and v["cross"]["m_to_N"]["passed"]
            and v["cross"]["g_to_M"]["passed"] and v["degree_profile_m"]["passed"]
            and v["memory_curve_g"]["passed"])
    G["11_interior_effects"] = all(mains[m][k]["passed"] for m in METHODS
                                   for k in ("interior_m_M", "interior_g_N"))
    G["11b_section6_intrinsic_P"] = bool(s6["verdict"].endswith("HOLDS"))
    comb = {cls: {meth: combined_class(rows, cls, meth, alpha, seed=seed + 11 + 3 * i)
                  for meth in METHODS} for i, cls in enumerate(V6_CLASSES)}
    G["12_combined_C1_C4_all_pass"] = all(comb[c][m]["passed"] for c in V6_CLASSES for m in METHODS)
    G["13_HH_best_every_class"] = all(comb[c][m]["HH_best_point"] and comb[c][m]["HH_beats_all"]
                                      for c in V6_CLASSES for m in METHODS)
    fc = feature_count_invariance(rows)
    G["14_feature_counts_invariant"] = fc["invariant"]
    enc = encoder_leakage(rows, enc_exact)
    G["15_no_encoder_leakage"] = enc["passed"]
    sat = metric_saturation(rows, ctx)
    pooled = max(r["blocks"][f"ALL|{m}"]["saturated_fraction"] for r in rows for m in METHODS)
    G["16_no_saturation"] = bool(sat["ok"] and pooled <= THRESH["sat_fraction"]
                                 and all(comb[c][m]["not_saturated"] for c in V6_CLASSES
                                         for m in METHODS))
    sen = sentinels(rows)
    G["17_future_sentinel_null"] = sen["SENTINEL_future"]["ok"]
    G["18_unreachable_sentinel_null"] = sen["SENTINEL_unreachable"]["ok"]
    mli = metric_level_independence(rows)
    G["19_isolation_and_contaminated_controls"] = bool(
        st["frozen"]["isolated"] and all(not st[k]["isolated"] for k in ("g_into_R", "m_into_P", "serial"))
        and all(v["M_spread_over_g"] == 0 and v["N_spread_over_m"] == 0 for v in mli.values()))
    G["20_invalid_controls_detected"] = bool(inv["all_detected"])
    G["21_seed_reproducibility"] = bool(
        all(mains[m][k]["seed_fraction_above_min"] >= THRESH["seed_consistency"]
            for m in METHODS for k in ("m_M", "g_N"))
        and all(comb[c][m]["seed_consistency"] >= THRESH["seed_consistency"]
                for c in V6_CLASSES for m in METHODS))
    G["26_unit_tests_pass"] = bool(tests_passed)
    return {"alpha": alpha, "gates": G, "passed_all": bool(all(G.values())),
            "mains": mains, "combined": comb, "saturation": sat, "pooled_saturated_fraction": pooled,
            "sentinels": sen, "encoder_leakage": enc, "feature_counts": fc,
            "metric_level_independence": mli, "thresholds": THRESH}


def robustness_ok(rows, ctx, grid, *, noiseless: bool, dev: bool) -> dict:
    """Items 22-25 for one perturbed condition (point estimates on dev seeds)."""
    lo, mid, hi = grid[0], grid[len(grid) // 2], grid[-1]
    eff = {}
    for meth in METHODS:
        for nm, ro, name, ctl in (("dM", "R", "M", "m"), ("dN", "P", "N", "g")):
            gr = grid_by_seed(rows, _metric(ro, meth, name))
            eff[f"{nm}_{meth}"] = float(np.mean(effect_per_seed(gr, ctl)))
            eff[f"{nm}_interior_{meth}"] = float(np.mean(effect_per_seed(gr, ctl, lo=mid, hi=hi)))
    worst = min(eff.values())
    adv = {}
    for cls in V6_CLASSES:
        for meth in METHODS:
            q = quadrant_values(rows, f"ALL|{meth}", lambda b, c=cls: b["class"][c], (lo,), (hi,))
            qm = {k: float(np.nanmean(v)) for k, v in q.items()}
            adv[f"{cls}|{meth}"] = qm["HH"] - max(qm["LL"], qm["LH"], qm["HL"])
    mli = metric_level_independence(rows)
    iso = all(v["M_spread_over_g"] == 0 and v["N_spread_over_m"] == 0 for v in mli.values())
    sat = metric_saturation(rows, ctx) if noiseless else {"ok": True, "note": "not required under noise"}
    need_e = DEV_MARGIN["robust_effect_min"] if dev else THRESH["main_min"]
    need_a = DEV_MARGIN["robust_hh_adv_min"] if dev else 0.0
    return {"effects": eff, "worst_effect": worst, "hh_advantage": adv,
            "worst_hh_advantage": min(adv.values()), "isolated": iso, "saturation": sat,
            "passed": bool(worst >= need_e and min(adv.values()) > need_a and iso and sat["ok"])}
