"""
v4_search.py -- reproducible constrained architecture search.

SEED BANKS ARE DISJOINT BY CONSTRUCTION. The search may only ever see
DEVELOPMENT seeds; `CONFIRMATION_ARCH`/`CONFIRMATION_INPUT` are never returned
by any function in this module, and `assert_development_only` raises if a
confirmation seed reaches the search. Confirmation data is not development
data, and a failed confirmation may not be recycled as one.

WHAT THE SEARCH MAY VARY: register sizes, the retention channel and its cap,
memory mixing strength and time, processor interaction strength and depth,
observable selection, joint-observable count, and the readout regularisation.

WHAT IT MAY NEVER VARY: the statistical thresholds, the equivalence margins,
the target definitions, and the resource accounting -- those are frozen before
the search starts. Tuning any of them against the search objective would make
the eventual gate pass meaningless.

RANKING IS BY WORST CASE, NOT MEAN. Candidates are scored on the WORST
development seed, because this project has repeatedly watched single-seed
dynamic ranges evaporate across seeds. A candidate that is excellent on
average and poor on one seed is not a candidate.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from .v4_architecture import MemorySpec, ProcessorSpec, V4Spec
from .v4_experiment import (THRESHOLDS, evaluate_all_gates, evaluate_grid,
                            freeze_margin_check, nested_effects)
from .v4_ipc import IPCConfig
from .v4_statistics import nested_bootstrap, quadrant_contrast

# Disjoint seed banks. The confirmation bank is untouched by anything here.
DEVELOPMENT_ARCH = tuple(range(100, 140))
DEVELOPMENT_INPUT = tuple(range(1000, 1040))
CONFIRMATION_ARCH = tuple(range(900, 940))
CONFIRMATION_INPUT = tuple(range(9000, 9040))


class SeedBankViolation(RuntimeError):
    pass


def assert_development_only(arch_seeds, input_seeds) -> None:
    bad_a = sorted(set(arch_seeds) & set(CONFIRMATION_ARCH))
    bad_i = sorted(set(input_seeds) & set(CONFIRMATION_INPUT))
    if bad_a or bad_i:
        raise SeedBankViolation(
            f"confirmation seeds reached development code: arch={bad_a} input={bad_i}. "
            f"The confirmation bank must stay untouched until the frozen run.")


def assert_confirmation_only(arch_seeds, input_seeds) -> None:
    bad_a = sorted(set(arch_seeds) & set(DEVELOPMENT_ARCH))
    bad_i = sorted(set(input_seeds) & set(DEVELOPMENT_INPUT))
    if bad_a or bad_i:
        raise SeedBankViolation(
            f"development seeds reached the confirmation run: arch={bad_a} input={bad_i}. "
            f"Confirmation must be measured on data the search never saw.")


# =============================================================================
# Search space
# =============================================================================
@dataclass
class SearchSpace:
    L_R: tuple = (3, 4)
    hop: tuple = (0.5, 0.7, 0.9)
    tau_mix: tuple = (0.8, 1.0, 1.3)
    m_max: tuple = (0.85, 0.9, 0.95)
    retention: tuple = ("amplitude_damping", "depolarizing")
    pair_readout: tuple = (True, False)
    N_P: tuple = (3, 4)
    depth: tuple = (1, 2, 3)
    dt: tuple = (0.4, 0.6, 0.8)
    g_scale: tuple = (0.25, 0.5, 0.8)
    n_joint: tuple = (36, 72)
    alpha: tuple = (0.2, 0.5, 1.0)

    def sample(self, rng) -> dict:
        return {k: (v[rng.integers(len(v))] if not isinstance(v[0], bool)
                    else bool(v[rng.integers(len(v))]))
                for k, v in asdict(self).items()}

    def as_dict(self) -> dict:
        return {k: list(v) for k, v in asdict(self).items()}


def build(cand: dict) -> tuple:
    """(V4Spec, IPCConfig) from a candidate dict."""
    spec = V4Spec(
        memory=MemorySpec(L_R=int(cand["L_R"]), hop=float(cand["hop"]),
                          tau_mix=float(cand["tau_mix"]), m_max=float(cand["m_max"]),
                          retention=str(cand["retention"]),
                          pair_readout=bool(cand["pair_readout"])),
        processor=ProcessorSpec(N_P=int(cand["N_P"]), depth=int(cand["depth"]),
                                dt=float(cand["dt"]), g_scale=float(cand["g_scale"])),
        n_joint=int(cand["n_joint"]))
    cfg = IPCConfig(alpha=float(cand["alpha"]), max_delay=8, tau_L=2, tau_S=0,
                    nlong_max_delay=5, n_null=40)
    return spec, cfg


# =============================================================================
# Objective
# =============================================================================
@dataclass
class ObjectiveWeights:
    lam_cross_N: float = 12.0        # penalise |Delta_m N|
    lam_cross_M: float = 12.0        # penalise |Delta_g M|
    lam_sat: float = 6.0             # penalise metric saturation
    lam_leak: float = 6.0            # penalise encoder / route leakage
    lam_comb: float = 2.0            # reward the high-m/high-g advantage

    def as_dict(self) -> dict:
        return asdict(self)


def objective(summary: dict, w: ObjectiveWeights) -> float:
    """L = -dM - dN + lam1|dN/dm| + lam2|dM/dg| + lam3 S_sat + lam4 S_leak - lam5 dHH.

    Lower is better. Saturation and leakage enter as PENALTIES so the search
    cannot buy a large effect by pinning a metric at its ceiling.
    """
    s_sat = max(0.0, summary["worst_saturated_fraction"]
                - THRESHOLDS["saturation_max_fraction"])
    s_leak = max(0.0, summary["encoder_only_N"] - summary["null_threshold"])
    return float(
        -summary["worst_dM_dm"]
        - summary["worst_dN_dg"]
        + w.lam_cross_N * abs(summary["worst_abs_dN_dm"])
        + w.lam_cross_M * abs(summary["worst_abs_dM_dg"])
        + w.lam_sat * s_sat
        + w.lam_leak * s_leak
        - w.lam_comb * summary["worst_combined"])


def summarise_candidate(grid_out: dict, encoder_only_N: float) -> dict:
    """Worst-case-over-seeds summary. Never the mean."""
    def worst(metric, control, mode):
        nested = nested_effects(grid_out, metric, control)
        flat = [v for vals in nested.values() for v in vals]
        if not flat:
            return float("nan")
        return float(min(flat) if mode == "min" else max(abs(x) for x in flat))

    comb = []
    for a in grid_out["arch_seeds"]:
        for i in grid_out["input_seeds"]:
            sub = {(r["m"], r["g"]): r for r in grid_out["rows"]
                   if r["arch_seed"] == a and r["input_seed"] == i}
            if sub:
                comb.append(quadrant_contrast(sub, "N_long")["improvement"])
    return {"worst_dM_dm": worst("M", "m", "min"),
            "worst_dN_dg": worst("N", "g", "min"),
            "worst_abs_dN_dm": worst("N", "m", "absmax"),
            "worst_abs_dM_dg": worst("M", "g", "absmax"),
            "worst_combined": float(min(comb)) if comb else float("nan"),
            "worst_saturated_fraction": float(max(r["saturated_fraction"]
                                                  for r in grid_out["rows"])),
            "encoder_only_N": float(encoder_only_N),
            "null_threshold": float(np.median([r["null_threshold"]
                                               for r in grid_out["rows"]])),
            "max_unreachable": float(max(r["unreachable_max"] for r in grid_out["rows"]))}


# =============================================================================
# Search driver
# =============================================================================
def random_search(*, n_candidates: int, seed: int, m_values, g_values,
                  arch_seeds, input_seeds, T: int, space: SearchSpace = None,
                  weights: ObjectiveWeights = None, checkpoint=None,
                  verbose: bool = True) -> dict:
    """Seeded random search. Every candidate is cached and retained."""
    assert_development_only(arch_seeds, input_seeds)
    space = space or SearchSpace()
    weights = weights or ObjectiveWeights()
    rng = np.random.default_rng(int(seed))

    done = checkpoint.done_keys() if checkpoint is not None else set()
    history = list(checkpoint.rows()) if checkpoint is not None else []

    for i in range(int(n_candidates)):
        key = f"cand{i:04d}"
        if key in done:
            continue
        cand = space.sample(rng)
        try:
            spec, cfg = build(cand)
            grid = evaluate_grid(spec, cfg, m_values=m_values, g_values=g_values,
                                 arch_seeds=arch_seeds, input_seeds=input_seeds, T=T)
            enc = float(np.mean([r["N"] for r in grid["rows"] if r["g"] == min(g_values)]))
            summ = summarise_candidate(grid, enc)
            score = objective(summ, weights)
            row = {"candidate": cand, "summary": summ, "score": score, "error": None}
        except Exception as exc:                       # a candidate may be infeasible
            row = {"candidate": cand, "summary": None, "score": float("inf"),
                   "error": f"{type(exc).__name__}: {exc}"}
        if checkpoint is not None:
            checkpoint.append(key, row)
        history.append({**row, "row_key": key})
        if verbose:
            s = row["summary"]
            msg = (f"dM={s['worst_dM_dm']:+.3f} dN={s['worst_dN_dg']:+.3f} "
                   f"xN={s['worst_abs_dN_dm']:.1e} xM={s['worst_abs_dM_dg']:.1e} "
                   f"comb={s['worst_combined']:+.3f} sat={s['worst_saturated_fraction']:.2f}"
                   if s else row["error"])
            print(f"  [{key}] score={row['score']:+8.4f}  {msg}")

    ok = [h for h in history if h.get("summary") is not None]
    ok.sort(key=lambda h: h["score"])
    return {"history": history, "n_evaluated": len(history), "n_feasible": len(ok),
            "best": ok[0] if ok else None, "ranked": ok,
            "space": space.as_dict(), "weights": weights.as_dict(),
            "seed": int(seed), "m_values": list(m_values), "g_values": list(g_values),
            "arch_seeds": list(arch_seeds), "input_seeds": list(input_seeds), "T": int(T)}


def stress_test(cand: dict, *, m_values, g_values, arch_seeds, input_seeds, T: int,
                perturbations=(0.9, 1.1), verbose: bool = True) -> dict:
    """Re-evaluate the winner under extra seeds and small parameter perturbations.

    A candidate whose passing depends on an exact parameter value is not
    robust, and the confirmation run will not reproduce it.
    """
    assert_development_only(arch_seeds, input_seeds)
    rows = []
    variants = [("nominal", dict(cand))]
    for knob in ("hop", "tau_mix", "dt", "g_scale", "alpha", "m_max"):
        for f in perturbations:
            v = dict(cand)
            v[knob] = float(cand[knob]) * f
            if knob == "m_max":
                v[knob] = float(min(v[knob], 0.97))
            variants.append((f"{knob}x{f}", v))

    for name, v in variants:
        spec, cfg = build(v)
        grid = evaluate_grid(spec, cfg, m_values=m_values, g_values=g_values,
                             arch_seeds=arch_seeds, input_seeds=input_seeds, T=T)
        enc = float(np.mean([r["N"] for r in grid["rows"] if r["g"] == min(g_values)]))
        u_probe = np.random.default_rng(1).uniform(-1, 1, 40)
        gt = evaluate_all_gates(grid, spec, cfg, u_probe=u_probe, encoder_only_N=enc, seed=17)
        fm = freeze_margin_check(gt)
        rows.append({"variant": name, "params": v, "all_passed": gt["all_passed"],
                     "margin_ok": fm["margin_ok"], "n_pass": gt["n_pass"],
                     "n_fail": gt["n_fail"],
                     "dM": gt["gates"]["memory_main_effect"]["values"]["delta"],
                     "dN": gt["gates"]["nonlinear_main_effect"]["values"]["delta"],
                     "combined": gt["gates"]["combined_capability"]["values"]
                                   ["N_long"]["improvement"]})
        if verbose:
            print(f"  [{name:<14}] pass={rows[-1]['all_passed']} margin={rows[-1]['margin_ok']} "
                  f"dM={rows[-1]['dM']:+.3f} dN={rows[-1]['dN']:+.3f} "
                  f"comb={rows[-1]['combined']:+.3f}")
    n_ok = sum(1 for r in rows if r["all_passed"])
    return {"rows": rows, "n_variants": len(rows), "n_all_passed": n_ok,
            "robust": bool(n_ok == len(rows)),
            "n_margin_ok": sum(1 for r in rows if r["margin_ok"])}
