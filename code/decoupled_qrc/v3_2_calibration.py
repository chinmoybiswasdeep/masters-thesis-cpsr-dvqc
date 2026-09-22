"""
v3_2_calibration.py -- the CALIBRATION stage: verify the DIAGONAL functional
prerequisites before spending anything on discovery.

The failure discipline this stage implements: if the encoder still dominates
the nonlinearity, if (g,J) cannot generate capacity, or if m cannot control
retention, then discovery is pointless and the study stops here with a
precise diagnosis. V3.1 ran a full pipeline on top of two architectural
defects; this stage exists so that cannot happen again.

PROCESSOR PREREQUISITES
  P1 Hamiltonian audit: Hermiticity, [H_2,H_4] = 0, and genuine noncommutation
     between the mixing and interaction layers.
  P2 Null linearity: NL_0(g=0, J=0) = 0 for the fixed local readout (exact,
     by quadrature) -- so f_enc = 0 by construction.
  P3 Interaction-generated capacity: Delta NL_interaction = NL_0(g,J) - NL_0(0,0)
     with a PAIRED confidence interval excluding zero, computed under common
     random numbers.
  P4 Ablation distinctness: Hamiltonian, unitary, state and feature differences
     reported separately; any feature-identical pair NOT explained by the
     proven Z-diagonal symmetry is a defect that blocks discovery.
  P5 Dynamic range: E_NL > 0.20 over the (g,J) grid, using the repository's
     own `v3_1_stats.dynamic_range`.
  P6 Shot budget: choose S (from a preregistered ladder) so the response
     surface is NOT saturated. In a noiseless simulation instantaneous
     capacity is pinned at its ceiling and has no derivative at all, so this
     step is what makes the primary metric well-posed.

MEMORY PREREQUISITES
  M1 Channel spectrum: the second-largest eigenvalue modulus of the per-step
     map, and the retention time it implies, as a function of m.
  M2 Delay profile over 9-13 values of m under common random numbers.
  M3 A STABLE-SLOPE INTERIOR WINDOW: consecutive grid intervals whose
     dM_long/dm agrees in sign across reservoir seeds, away from both
     boundaries and away from any ceiling-saturated region.
  M4 Candidate selection by slope stability and interior margin -- NEVER by
     maximum capacity. V3.1 selected m* = 0.95 by maximising M and landed on
     the boundary of its own range inside the saturated region.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import v3_2_memory as mem
from . import v3_2_processor as proc
from .v3_1_stats import dynamic_range
from .v3_2_architecture import ArchitectureSpec, evaluate_point
from .v3_2_gates import THRESHOLDS
from .v3_2_readout import ShotBudget
from .v3_2_response import bootstrap_ci, interior_margin, normalize

SHOT_LADDER = (1000, 3000, 10000, 30000, 100000)


@dataclass
class CalibrationSettings:
    """Preregistered calibration settings -- fixed before any data is seen."""

    T: int = 1200
    washout: int = 60
    n_val: int = 220
    n_test: int = 260
    tau_min: int = 2
    max_delay: int = 8
    max_degree: int = 5
    max_targets_per_degree: int = 20
    n_surrogates: int = 39
    n_reservoir_seeds: int = 3
    n_input_seeds: int = 2
    m_grid: tuple = (0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85)
    g_grid: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    J_grid: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    stencil_h: float = 0.08
    shots: int = 10000

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# =============================================================================
# Processor calibration
# =============================================================================
def calibrate_processor(pspec: proc.ProcessorSpec, settings: CalibrationSettings,
                         broker, *, stage="CALIBRATION", verbose: bool = True) -> dict:
    """P1-P6. Returns a report with a `passed` verdict and named failures."""
    out = {"stage": str(stage), "settings": settings.as_dict(),
           "processor": pspec.as_dict(), "failures": []}

    seed_bundles = [[broker.for_stage(stage, k, i) for i in range(settings.n_input_seeds)]
                    for k in range(settings.n_reservoir_seeds)]

    # ---- P1 Hamiltonian audit -------------------------------------------
    hams = [proc.build_hamiltonians(pspec, int(b[0].disorder)) for b in seed_bundles]
    audits = [h.audit() for h in hams]
    out["P1_hamiltonian_audit"] = audits
    herm_ok = all(max(a["herm_H_mix"], a["herm_H_2"], a["herm_H_4"]) <= 1e-12 for a in audits)
    comm_ok = all(a["comm_H2_H4"] <= 1e-12 for a in audits)
    noncomm_ok = all(a["comm_Hmix_H2"] > 1e-6 and a["comm_Hmix_H4"] > 1e-6 for a in audits)
    out["P1_passed"] = bool(herm_ok and comm_ok and noncomm_ok)
    if not out["P1_passed"]:
        out["failures"].append("P1: Hamiltonian audit failed (hermiticity, [H2,H4]=0 or noncommutation)")

    # ---- P2 exact null linearity (quadrature, no sampling) ---------------
    null_rows = []
    for h in hams:
        U0 = proc.processor_unitary(pspec, h, 0.0, 0.0, ablation="full")
        e0 = proc.exact_instantaneous_capacities(pspec, U0)
        null_rows.append({"NL_0_exact": e0["NL_0_exact"], "feature_rank": e0["feature_rank"],
                          "capacities": e0["capacities"]})
    out["P2_null_linearity"] = null_rows
    out["P2_passed"] = bool(all(r["NL_0_exact"] <= 1e-9 for r in null_rows))
    if not out["P2_passed"]:
        out["failures"].append("P2: the null processor is NOT exactly linear -- the encoder "
                               "is generating nonlinearity before (g,J) acts (the V3.1 defect)")

    # ---- P4 ablation distinctness ---------------------------------------
    g_mid = float(np.median([g for g in settings.g_grid if g > 0]))
    J_mid = float(np.median([J for J in settings.J_grid if J > 0]))
    dist = proc.ablation_distinctness(pspec, hams[0], g_mid, J_mid,
                                      hamiltonian_seed=int(seed_bundles[0][0].hamiltonian))
    out["P4_ablation_distinctness"] = dist
    out["P4_passed"] = bool(not dist["unexplained_duplicates"])
    if not out["P4_passed"]:
        out["failures"].append(f"P4: unexplained identical ablations {dist['unexplained_duplicates']} "
                               f"-- possible Hamiltonian/unitary/feature/cache aliasing")

    # ---- P6 shot-budget selection, then P5 dynamic range, then P3 --------
    pspec_only = ArchitectureSpec(architecture="processor_only", processor=pspec)

    def nl0_at(g, J, bundle, budget):
        r = evaluate_point(pspec_only, m=0.5, g=g, J=J, seeds=bundle, budget=budget,
                           T=settings.T, washout=settings.washout, n_val=settings.n_val,
                           n_test=settings.n_test, tau_min=settings.tau_min,
                           max_delay=settings.max_delay, max_degree=settings.max_degree,
                           max_targets_per_degree=settings.max_targets_per_degree,
                           n_surrogates=settings.n_surrogates)
        return r

    ladder_rows, chosen_shots = [], None
    # EVERY reservoir seed, not one: picking the shot budget from a single
    # bundle would violate this project's own "never trust one seed" rule.
    probe_bundles = [b for bundles in seed_bundles for b in bundles]
    probe_pts = [(0.0, 0.0), (0.2, 0.2), (float(settings.g_grid[-1]), float(settings.J_grid[-1]))]
    ceiling = float(settings.max_degree - 1)              # one unit per degree 2..max_degree
    for S in SHOT_LADDER:
        budget = ShotBudget(S)
        raws, bcs = [], []
        for g, J in probe_pts:
            rs = [nl0_at(g, J, b, budget) for b in probe_bundles]
            raws.append(float(np.median([r["P"].NL_0_raw for r in rs])))
            bcs.append(float(np.median([r["NL_0"] for r in rs])))
        # Saturation is a property of the RAW capacity against its target-count
        # ceiling. Testing the BIAS-CORRECTED value instead made this branch
        # mathematically unreachable (raw <= ceiling, and raw - null is always
        # below 0.97*ceiling), so the ladder always returned its first entry.
        saturated = bool(raws[1] >= 0.97 * ceiling and raws[2] >= 0.97 * ceiling)
        ladder_rows.append({"shots": S, "NL_0_null_bc": bcs[0], "NL_0_low_bc": bcs[1],
                            "NL_0_high_bc": bcs[2], "NL_0_low_raw": raws[1],
                            "NL_0_high_raw": raws[2], "ceiling": ceiling,
                            "raw_fraction_of_ceiling": raws[2] / ceiling,
                            "saturated": saturated, "n_bundles": len(probe_bundles)})
    # Prefer the LARGEST unsaturated budget: more shots means more information,
    # and "first unsaturated" degenerates to "noisiest" whenever nothing
    # saturates. Ties are impossible since the ladder is strictly increasing.
    unsaturated = [r["shots"] for r in ladder_rows if not r["saturated"]]
    chosen_shots = max(unsaturated) if unsaturated else None
    out["P6_shot_ladder"] = ladder_rows
    out["P6_chosen_shots"] = chosen_shots if chosen_shots is not None else SHOT_LADDER[0]
    out["P6_passed"] = bool(chosen_shots is not None)
    if not out["P6_passed"]:
        out["failures"].append("P6: every shot budget on the ladder leaves the response surface "
                               "saturated -- NL_0 has no dynamic range to differentiate")
    budget = ShotBudget(out["P6_chosen_shots"])

    # ---- P5 dynamic range over the (g,J) grid ---------------------------
    surface, nl0_grid = [], []
    for g in settings.g_grid:
        for J in settings.J_grid:
            vals = []
            for b in seed_bundles:
                for bundle in b:
                    vals.append(nl0_at(g, J, bundle, budget)["NL_0"])
            med = float(np.median(vals))
            surface.append({"g": float(g), "J": float(J), "NL_0_median": med,
                            "NL_0_values": [float(v) for v in vals]})
            nl0_grid.append(med)
    out["P5_surface"] = surface
    dr = dynamic_range(nl0_grid)
    out["P5_dynamic_range"] = dr
    out["P5_passed"] = bool(np.isfinite(dr["E"]) and dr["E"] > THRESHOLDS["E_dynamic_range_min"])
    if not out["P5_passed"]:
        out["failures"].append(f"P5: E_NL = {dr['E']:.4f} does not exceed "
                               f"{THRESHOLDS['E_dynamic_range_min']} -- (g,J) do not move NL_0 enough")

    # ---- P3 paired interaction-generated capacity (common random numbers)
    best = max(surface, key=lambda r: r["NL_0_median"])
    interior = [r for r in surface if 0 < r["g"] < max(settings.g_grid)
                and 0 < r["J"] < max(settings.J_grid)]
    operating = max(interior, key=lambda r: r["NL_0_median"]) if interior else best
    nested_delta, f_enc_vals = {}, []
    for k, bundles in enumerate(seed_bundles):
        diffs = []
        for bundle in bundles:
            on = nl0_at(operating["g"], operating["J"], bundle, budget)["NL_0"]
            off = nl0_at(0.0, 0.0, bundle, budget)["NL_0"]       # SAME seeds: paired
            diffs.append(float(on - off))
            f_enc_vals.append(float(off / (on + 1e-12)))
        nested_delta[k] = diffs
    ci = bootstrap_ci(nested_delta, n_boot=2000, seed=17)
    out["P3_delta_interaction"] = {"operating_point": {"g": operating["g"], "J": operating["J"]},
                                   "nested": {str(k): v for k, v in nested_delta.items()},
                                   "point": ci.point, "lo": ci.lo, "hi": ci.hi}
    out["P3_f_enc"] = {"values": f_enc_vals, "median": float(np.median(f_enc_vals))}
    out["P3_passed"] = bool(np.isfinite(ci.lo) and ci.lo > 0.0
                            and out["P3_f_enc"]["median"] < THRESHOLDS["E_f_enc_max"])
    if not out["P3_passed"]:
        out["failures"].append("P3: interaction-generated NL_0 is not positive with a paired CI "
                               "excluding zero, or the encoder fraction is too high")

    out["passed"] = bool(out["P1_passed"] and out["P2_passed"] and out["P3_passed"]
                         and out["P4_passed"] and out["P5_passed"] and out["P6_passed"])
    out["chosen_operating_point"] = {"g": operating["g"], "J": operating["J"],
                                     "shots": out["P6_chosen_shots"]}
    if verbose:
        print(f"[processor calibration] passed={out['passed']} "
              f"E_NL={dr['E']:.4f} shots={out['P6_chosen_shots']} "
              f"f_enc={out['P3_f_enc']['median']:.4f} "
              f"delta_CI=[{ci.lo:.4f}, {ci.hi:.4f}]")
        for f in out["failures"]:
            print(f"   FAILURE: {f}")
    return out


# =============================================================================
# Memory calibration
# =============================================================================
def calibrate_memory(mspec: mem.MemorySpec, settings: CalibrationSettings, broker,
                     budget: ShotBudget, *, stage="CALIBRATION", verbose: bool = True) -> dict:
    """M1-M4 for ONE mechanism."""
    lo, hi = mspec.control_range
    out = {"mechanism": mspec.mechanism, "memory": mspec.as_dict(),
           "control_range": [lo, hi], "failures": []}
    seed_bundles = [[broker.for_stage(stage, k, i) for i in range(settings.n_input_seeds)]
                    for k in range(settings.n_reservoir_seeds)]
    arch = ArchitectureSpec(architecture="memory_only", memory=mspec)

    # ---- M1 spectrum -----------------------------------------------------
    spectra = []
    for m in settings.m_grid:
        s = mem.channel_spectrum(mspec, m, int(seed_bundles[0][0].disorder))
        spectra.append({"m": float(m), **s})
    out["M1_spectrum"] = spectra

    # ---- M2 delay profiles under common random numbers -------------------
    rows = []
    for m in settings.m_grid:
        per_seed = {}
        for k, bundles in enumerate(seed_bundles):
            vals = []
            for bundle in bundles:
                r = evaluate_point(arch, m=m, g=0.0, J=0.0, seeds=bundle, budget=budget,
                                   T=settings.T, washout=settings.washout, n_val=settings.n_val,
                                   n_test=settings.n_test, tau_min=settings.tau_min,
                                   max_delay=settings.max_delay, max_degree=settings.max_degree,
                                   max_targets_per_degree=settings.max_targets_per_degree,
                                   n_surrogates=settings.n_surrogates)
                desc = mem.delay_profile_descriptors(r["M"].delay_profile, settings.tau_min)
                vals.append({"M_long": r["M_long"], **desc,
                             "null_fraction": r["M"].null_fraction_M,
                             "ceiling": r["M"].ceiling_M,
                             "sample_size_ok": r["M"].sample_size_ok})
            per_seed[k] = vals
        flat = [v["M_long"] for vs in per_seed.values() for v in vs]
        rows.append({"m": float(m), "M_long_median": float(np.median(flat)),
                     "M_long_by_seed": {str(k): [v["M_long"] for v in vs]
                                        for k, vs in per_seed.items()},
                     "descriptors": {str(k): vs for k, vs in per_seed.items()},
                     "interior_margin": interior_margin(m, lo, hi),
                     "m_tilde": normalize(m, lo, hi)})
    out["M2_profile"] = rows

    # ---- M3 stable-slope interior window ---------------------------------
    grid = [r["m"] for r in rows]
    slopes = []
    for i in range(len(grid) - 1):
        dm_tilde = normalize(grid[i + 1], lo, hi) - normalize(grid[i], lo, hi)
        per_seed = {}
        for k in rows[i]["M_long_by_seed"]:
            a = np.array(rows[i]["M_long_by_seed"][k], dtype=float)
            b = np.array(rows[i + 1]["M_long_by_seed"][k], dtype=float)
            n = min(a.size, b.size)
            per_seed[k] = list((b[:n] - a[:n]) / dm_tilde)
        allv = [v for vs in per_seed.values() for v in vs]
        seed_means = [float(np.mean(v)) for v in per_seed.values() if len(v)]
        slopes.append({"m_lo": grid[i], "m_hi": grid[i + 1],
                       "m_mid": 0.5 * (grid[i] + grid[i + 1]),
                       "slope_median": float(np.median(allv)),
                       "slope_by_seed": per_seed,
                       "sign_consistent_across_seeds": bool(
                           len(seed_means) > 0 and
                           all(np.sign(s) == np.sign(seed_means[0]) for s in seed_means)
                           and abs(np.median(allv)) > 1e-9)})
    out["M3_slopes"] = slopes

    # longest run of sign-consistent, same-sign intervals
    best_run, run = (0, None), []
    for idx, s in enumerate(slopes):
        if s["sign_consistent_across_seeds"] and (
                not run or np.sign(s["slope_median"]) == np.sign(slopes[run[0]]["slope_median"])):
            run.append(idx)
        else:
            run = [idx] if s["sign_consistent_across_seeds"] else []
        if len(run) > best_run[0]:
            best_run = (len(run), list(run))
    out["M3_stable_window"] = ({"n_intervals": best_run[0],
                                "m_lo": slopes[best_run[1][0]]["m_lo"],
                                "m_hi": slopes[best_run[1][-1]]["m_hi"],
                                "slope_median": float(np.median(
                                    [slopes[i]["slope_median"] for i in best_run[1]]))}
                               if best_run[1] else None)

    # ---- M4 candidate selection: slope stability + interior margin -------
    h = settings.stencil_h
    candidates = []
    for r in rows:
        mt = r["m_tilde"]
        reasons = []
        if mt - 2 * h < 0 or mt + 2 * h > 1:
            reasons.append("five-point stencil would leave the domain")
        descs = [v for vs in r["descriptors"].values() for v in vs]
        if any(not d["sample_size_ok"] for d in descs):
            reasons.append("insufficient training samples")
        # `fraction_of_ceiling` / `ceiling_contaminated` are the keys
        # `ceiling_audit` actually emits; a missing value is refused rather
        # than treated as safe.
        ceil_fracs = [d["ceiling"].get("fraction_of_ceiling") for d in descs]
        if any(c is None or not np.isfinite(c) for c in ceil_fracs):
            reasons.append("ceiling fraction unavailable")
        elif any(c > THRESHOLDS["C_ceiling_frac_max"] for c in ceil_fracs):
            reasons.append("delay profile is ceiling-contaminated")
        if any(d["ceiling"].get("ceiling_contaminated") for d in descs):
            reasons.append("delay profile flagged ceiling-contaminated")
        nfs = [d["null_fraction"] for d in descs if np.isfinite(d["null_fraction"])]
        if nfs and float(np.median(nfs)) >= THRESHOLDS["C_null_fraction_max"]:
            reasons.append("null fraction above the preregistered limit")
        if r["M_long_median"] < THRESHOLDS["D_M_long_min"]:
            reasons.append("M_long below the nontriviality threshold")
        local = [s for s in slopes if s["m_lo"] <= r["m"] <= s["m_hi"]]
        stable_local = bool(local and all(s["sign_consistent_across_seeds"] for s in local))
        if not stable_local:
            reasons.append("local slope is not sign-consistent across reservoir seeds")
        slope_vals = [v for s in local for vs in s["slope_by_seed"].values() for v in vs]
        slope_med = float(np.median(slope_vals)) if slope_vals else float("nan")
        slope_se = (float(np.std(slope_vals, ddof=1) / max(np.sqrt(len(slope_vals)), 1))
                    if len(slope_vals) > 1 else float("nan"))
        snr = abs(slope_med) / slope_se if (slope_se and np.isfinite(slope_se) and slope_se > 0) else float("nan")
        candidates.append({"m": r["m"], "m_tilde": mt, "valid": not reasons, "reasons": reasons,
                           "M_long_median": r["M_long_median"],
                           "interior_margin": r["interior_margin"],
                           "slope_median": slope_med, "slope_se": slope_se, "slope_snr": snr,
                           "null_fraction_median": (float(np.median(nfs)) if nfs else float("nan"))})

    valid = [c for c in candidates if c["valid"] and np.isfinite(c["slope_snr"])]

    def _z(vals):
        v = np.asarray(vals, dtype=float)
        sd = v.std()
        return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)

    if valid:
        z_snr = _z([c["slope_snr"] for c in valid])
        z_margin = _z([c["interior_margin"] for c in valid])
        z_nontrivial = _z([c["M_long_median"] for c in valid])
        z_null = _z([0.0 if not np.isfinite(c["null_fraction_median"])
                     else c["null_fraction_median"] for c in valid])
        for c, a, b, d, e in zip(valid, z_snr, z_margin, z_nontrivial, z_null):
            # PREREGISTERED weighting: slope stability dominates, capacity is a
            # minor term. Ranking by max capacity is what put V3.1 on the boundary.
            c["score"] = float(1.0 * a + 0.8 * b + 0.3 * d - 0.5 * e)
        valid.sort(key=lambda c: c["score"], reverse=True)

    out["M4_candidates"] = candidates
    out["M4_selected"] = valid[0] if valid else None
    out["passed"] = bool(valid)
    if not valid:
        out["failures"].append(
            f"M4: no value of m for mechanism {mspec.mechanism!r} passes the hard filters "
            f"(interior stencil margin, non-ceiling profile, stable cross-seed slope, "
            f"nontrivial M_long). The diagonal memory-control prerequisite FAILED.")
    if verbose:
        sel = out["M4_selected"]
        print(f"[memory calibration: {mspec.mechanism}] passed={out['passed']} "
              + (f"m*={sel['m']:.3f} slope={sel['slope_median']:.2f} "
                 f"snr={sel['slope_snr']:.2f} margin={sel['interior_margin']:.3f} "
                 f"M_long={sel['M_long_median']:.3f}" if sel else "no valid candidate"))
        for f in out["failures"]:
            print(f"   FAILURE: {f}")
    return out


def run_calibration(pspec: proc.ProcessorSpec, mechanisms, settings: CalibrationSettings,
                    broker, *, verbose: bool = True) -> dict:
    """Full CALIBRATION stage. `passed` is True only if the processor AND at
    least one memory mechanism clear their prerequisites."""
    report = {"settings": settings.as_dict(), "failures": []}
    report["processor"] = calibrate_processor(pspec, settings, broker, verbose=verbose)
    budget = ShotBudget(report["processor"]["P6_chosen_shots"])
    report["shot_budget"] = budget.as_dict()

    report["memory"] = {}
    for mspec in mechanisms:
        report["memory"][mspec.mechanism] = calibrate_memory(mspec, settings, broker, budget,
                                                             verbose=verbose)

    passing = [k for k, v in report["memory"].items() if v["passed"]]
    report["memory_mechanisms_passing"] = passing
    report["selected_mechanism"] = passing[0] if passing else None
    report["failures"] = list(report["processor"]["failures"]) + \
        [f"{k}: {f}" for k, v in report["memory"].items() for f in v["failures"]]
    report["passed"] = bool(report["processor"]["passed"] and passing)
    report["diagnosis"] = _diagnose(report)
    if verbose:
        print(f"\n=== CALIBRATION passed={report['passed']} ===")
        print(f"    {report['diagnosis']}")
    return report


def _diagnose(report: dict) -> str:
    """A precise failure diagnosis, or the go-ahead."""
    if report["passed"]:
        return (f"both diagonal prerequisites hold; selected memory mechanism "
                f"{report['selected_mechanism']!r}, shots={report['shot_budget']['shots_per_setting']}. "
                f"DISCOVERY may proceed.")
    p = report["processor"]
    bits = []
    if not p["passed"]:
        if not p.get("P2_passed", True):
            bits.append("the encoder still generates the nonlinearity (P2)")
        if not p.get("P3_passed", True):
            bits.append("(g,J) do not generate nonlinear capacity with a paired CI excluding zero (P3)")
        if not p.get("P5_passed", True):
            bits.append(f"the processor dynamic range E_NL={p['P5_dynamic_range']['E']:.4f} "
                        f"is below its gate (P5)")
        if not p.get("P4_passed", True):
            bits.append("ablations produce unexplained identical features (P4)")
        if not p.get("P6_passed", True):
            bits.append("no shot budget leaves the surface unsaturated (P6)")
    if not report["memory_mechanisms_passing"]:
        bits.append("no tested memory mechanism provides controllable retention (M4)")
    return ("CALIBRATION FAILED -- " + "; ".join(bits) +
            ". Per the failure discipline, DISCOVERY is NOT run and the study reports "
            "Outcome B (structural isolation but inadequate diagonal control) unless the "
            "processor itself is invalid, in which case Outcome D applies.")
