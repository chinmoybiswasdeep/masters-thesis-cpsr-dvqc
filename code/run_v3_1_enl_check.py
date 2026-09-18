"""
run_v3_1_enl_check.py -- the V3.1 decision gate, run BEFORE committing to a
full discovery sweep.

V3.1 established that V3's near-flat NL(g,J) surface was substantially a
CEILING ARTIFACT: at max_degree=3 the metric sat at exactly 100% of its
2-target ceiling, and raising the order to 8 lifted the relative dynamic
range E_NL from 0.044 to 0.222 on a single seed.

A single seed is not evidence. This script asks the one question that
decides whether a full (20-hour) discovery sweep is worth running:

    does E_NL > 0.20 hold ACROSS independent reservoir seeds,
    at max_degree = 8, in the unsaturated regime?

Estimator settings are IDENTICAL to the notebook's DISCOVERY stage (T=420,
max_delay=8, max_degree=8, max_targets_per_degree=30, n_surrogates=49), so
the answer is directly comparable to what a full sweep would produce. Only
the (g,J) grid is coarser -- a resource decision, recorded here, not a
change to the scientific protocol.

Seeds come from the DISCOVERY family only (negative reservoir_idx), so
nothing here can contaminate a later held-out confirmation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoupled_qrc.ceiling_audit import audit_nl0
from decoupled_qrc.feature_analysis import diagnose_feature_group
from decoupled_qrc.ipc_decomposition import compute_ipc_decomposed, nl_tensor_by_fixed_delay
from decoupled_qrc.nonlinear_processor import build_tap_buffer
from decoupled_qrc.processor_variants import ProcVariantConfig, run_variant
from decoupled_qrc.v3_1_stats import dynamic_range, hierarchical_bootstrap
from decoupled_qrc.v3_analysis import analyse_feature_group
from decoupled_qrc.v3_seeds import discovery_seeds
from qrc_qiskit import chrono_split, random_input

# ---- DISCOVERY-grade estimator settings (unchanged from the notebook) ----
T, WASHOUT, N_VAL, N_TEST = 420, 40, 80, 100
MAX_DELAY, MAX_DEGREE, MAX_TPD, N_SUR = 8, 8, 30, 49
N_P, R = 4, 3
E_NL_GATE = 0.20

# ---- resource-bounded scan design (recorded, not a protocol change) ----
VARIANTS = ["P0", "P1"]
G_GRID = [0.30, 0.80, 1.30]
J_GRID = [0.20, 0.65, 1.10]
N_RESERVOIR = 3
N_INPUT = 1

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "dqrc_dual_route_v3_1", "e_nl_cross_seed_check.json")


def main():
    train, val, test = chrono_split(T, WASHOUT, N_VAL, N_TEST, MAX_DELAY + 1)
    total = len(VARIANTS) * len(G_GRID) * len(J_GRID) * N_RESERVOIR * N_INPUT
    print(f"E_NL cross-seed check: {total} evaluations "
          f"(train={len(train)} val={len(val)} test={len(test)})", flush=True)
    print(f"estimator: max_degree={MAX_DEGREE} max_delay={MAX_DELAY} "
          f"n_surrogates={N_SUR} T={T}\n", flush=True)

    rows, done, t_start = [], 0, time.perf_counter()
    for variant in VARIANTS:
        for r_idx in range(N_RESERVOIR):
            for i_idx in range(N_INPUT):
                seeds = discovery_seeds(300 + r_idx, i_idx)
                u = random_input(T, seed=seeds.input_sequence)
                for g in G_GRID:
                    for J in J_GRID:
                        cfg = ProcVariantConfig(variant=variant, N_P=N_P, g=g, J=J,
                                                 R=R, n_taps=1)
                        X = run_variant(cfg, build_tap_buffer(u, 1), seeds).X_P
                        # ONE decomposition per point: NL_0 and the ceiling audit are both
                        # derived from the same records (computing it twice doubled the cost
                        # for identical numbers).
                        decomp = compute_ipc_decomposed(
                            u, X, train, val, test, max_delay=MAX_DELAY, max_degree=MAX_DEGREE,
                            max_targets_per_degree=MAX_TPD, n_surrogates=N_SUR,
                            seed=seeds.null_surrogate, always_include_single_delays=True)
                        nl0 = nl_tensor_by_fixed_delay(decomp.records, MAX_DELAY)[0]
                        fd = diagnose_feature_group(X[train])
                        ceil = audit_nl0(decomp.records, fd.numerical_rank,
                                          fd.effective_rank, len(train))
                        rows.append({
                            "variant": variant, "reservoir_idx": seeds.reservoir_idx,
                            "input_idx": seeds.input_idx, "seed_stage": "discovery",
                            "g": g, "J": J,
                            "NL0_signed": nl0["signed"], "NL0_raw": nl0["raw"],
                            "NL0_null": nl0["null"], "NL0_legacy": nl0["legacy"],
                            "fraction_of_ceiling": ceil.fraction_of_ceiling,
                            "ceiling_contaminated": bool(ceil.ceiling_contaminated),
                            "n_targets": ceil.n_targets,
                            "effective_rank": fd.effective_rank,
                            "null_fraction": (nl0["null"] / nl0["raw"]
                                               if nl0["raw"] > 1e-12 else float("nan")),
                        })
                        done += 1
                        el = time.perf_counter() - t_start
                        print(f"[{done:3d}/{total}] {variant} seed={seeds.reservoir_idx} "
                              f"g={g:.2f} J={J:.2f} NL0={nl0['signed']:+.4f} "
                              f"ceil={ceil.fraction_of_ceiling:.2f} "
                              f"({el/60:.1f}m elapsed, ~{(total-done)*el/done/60:.1f}m left)",
                              flush=True)

    # ---- E_NL per (variant, seed): the cross-seed question ----
    summary = {}
    for variant in VARIANTS:
        per_seed, nested = {}, {}
        for r_idx in range(N_RESERVOIR):
            ridx = discovery_seeds(300 + r_idx, 0).reservoir_idx
            vals = [r["NL0_signed"] for r in rows
                    if r["variant"] == variant and r["reservoir_idx"] == ridx]
            if not vals:
                continue
            dr = dynamic_range(vals)
            per_seed[str(ridx)] = {**dr, "n_points": len(vals),
                                    "passes_gate": bool(np.isfinite(dr["E"]) and dr["E"] > E_NL_GATE)}
            nested[r_idx] = vals
        e_values = [v["E"] for v in per_seed.values() if np.isfinite(v["E"])]
        n_pass = sum(1 for v in per_seed.values() if v["passes_gate"])
        ci = hierarchical_bootstrap(nested, n_boot=1000, seed=0) if nested else None
        contaminated = sum(1 for r in rows if r["variant"] == variant and r["ceiling_contaminated"])
        summary[variant] = {
            "per_seed": per_seed,
            "E_NL_values": e_values,
            "E_NL_median": float(np.median(e_values)) if e_values else float("nan"),
            "E_NL_min": float(np.min(e_values)) if e_values else float("nan"),
            "E_NL_max": float(np.max(e_values)) if e_values else float("nan"),
            "n_seeds_passing_gate": n_pass,
            "n_seeds": len(per_seed),
            "gate_holds_across_seeds": bool(n_pass == len(per_seed) and len(per_seed) > 0),
            "n_ceiling_contaminated": contaminated,
            "NL0_hierarchical_ci": ci.as_dict() if ci else None,
        }

    verdict = {
        "question": "does E_NL > 0.20 hold across independent reservoir seeds at max_degree=8?",
        "gate": E_NL_GATE,
        "any_variant_passes_all_seeds": any(v["gate_holds_across_seeds"] for v in summary.values()),
        "best_variant": max(summary, key=lambda k: summary[k]["E_NL_median"]) if summary else None,
    }

    payload = {
        "purpose": "V3.1 decision gate before committing to a full discovery sweep",
        "estimator": {"T": T, "washout": WASHOUT, "n_val": N_VAL, "n_test": N_TEST,
                      "max_delay": MAX_DELAY, "max_degree": MAX_DEGREE,
                      "max_targets_per_degree": MAX_TPD, "n_surrogates": N_SUR,
                      "N_P": N_P, "R": R},
        "scan": {"variants": VARIANTS, "g_grid": G_GRID, "J_grid": J_GRID,
                 "n_reservoir": N_RESERVOIR, "n_input": N_INPUT,
                 "note": "coarser (g,J) grid than a full sweep -- a recorded resource "
                         "decision, not a change to the scientific protocol"},
        "seed_family": "discovery only (negative reservoir_idx)",
        "rows": rows, "summary": summary, "verdict": verdict,
        "wall_seconds": time.perf_counter() - t_start,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n" + "=" * 72)
    for variant, s in summary.items():
        print(f"{variant}: E_NL per seed = "
              f"{[round(v, 3) for v in s['E_NL_values']]}  median={s['E_NL_median']:.3f}  "
              f"gate held on {s['n_seeds_passing_gate']}/{s['n_seeds']} seeds  "
              f"(ceiling-contaminated evaluations: {s['n_ceiling_contaminated']})")
    print(f"\nVERDICT: any variant passing on ALL seeds = {verdict['any_variant_passes_all_seeds']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
