"""
v4_report.py -- figures and the final report, assembled ONLY from saved artifacts.

Nothing here simulates. Every figure reads a JSON written by a completed stage
and carries a provenance stamp (stage, seeds, uncertainty convention, source
file, frozen hash). A figure whose inputs are absent is SKIPPED and recorded as
skipped, never back-filled from a different stage.

Palette: categorical slots in fixed order (validated all-pairs for normal
vision, deuteranopia and tritanopia); a single-hue light-to-dark ramp for
magnitude, never a rainbow; the reserved status palette for gate verdicts,
always with a text verdict beside the colour so status is never colour alone.
The aqua slot is below 3:1 on the light surface, so every series also carries a
direct label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281")
STATUS = {"PASS": "#0ca30c", "FAIL": "#d03b3b", "NOT_EVALUABLE": "#fab219"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 5})
    return plt


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ)


def _stamp(fig, *, stage, source, extra=""):
    fig.text(0.005, -0.04, f"stage: {stage}  |  source: {source}  |  {extra}",
             fontsize=6.2, color=INK_MUTED, va="top", transform=fig.transFigure)


def _save(fig, outdir, name, manifest, title, source):
    outdir.mkdir(parents=True, exist_ok=True)
    png, pdf = outdir / f"{name}.png", outdir / f"{name}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    manifest.append({"name": name, "title": title, "png": png.name, "pdf": pdf.name,
                     "source": source})


def _skip(manifest, name, title, reason):
    manifest.append({"name": name, "title": title, "skipped": True, "reason": reason})


def _surface(conf, metric):
    gm = conf["grid_means"]
    ms = sorted({float(k.split("_")[0][1:]) for k in gm})
    gs = sorted({float(k.split("_")[1][1:]) for k in gm})
    Z = np.full((len(gs), len(ms)), np.nan)
    for k, v in gm.items():
        m = float(k.split("_")[0][1:]); g = float(k.split("_")[1][1:])
        Z[gs.index(g), ms.index(m)] = v[metric]
    return ms, gs, Z


def fig_surfaces(conf, outdir, manifest, stamp):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    for ax, metric, title in zip(axes, ("M", "N", "N_long"),
                                 ("M(m, g)  linear memory", "N(m, g)  local nonlinearity",
                                  r"$N_{\rm long}(m,g)$  combined")):
        ms, gs, Z = _surface(conf, metric)
        ax.grid(False)
        dm0 = (max(ms) - min(ms)) / (2 * max(len(ms) - 1, 1))
        dg0 = (max(gs) - min(gs)) / (2 * max(len(gs) - 1, 1))
        im = ax.imshow(Z, origin="lower", cmap=_cmap(), aspect="auto",
                       extent=[min(ms) - dm0, max(ms) + dm0,
                               min(gs) - dg0, max(gs) + dg0])
        rng = np.nanmax(Z) - np.nanmin(Z)
        for i, g in enumerate(gs):
            for j, m in enumerate(ms):
                if np.isfinite(Z[i, j]):
                    rel = (Z[i, j] - np.nanmin(Z)) / max(rng, 1e-12)
                    ax.text(m, g, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                            color="#ffffff" if rel > 0.55 else INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cb.outline.set_edgecolor(GRID)
        # pad the limits so the edge-cell labels are not clipped by the colorbar
        dm = (max(ms) - min(ms)) / (2 * max(len(ms) - 1, 1))
        dg = (max(gs) - min(gs)) / (2 * max(len(gs) - 1, 1))
        ax.set_xlim(min(ms) - dm, max(ms) + dm)
        ax.set_ylim(min(gs) - dg, max(gs) + dg)
        ax.set_xlabel("m  (memory control)")
        ax.set_ylabel("g  (nonlinearity control)")
        ax.set_title(title, loc="left", color=INK)
    fig.suptitle("V4 response surfaces: m moves only M, g moves only N, both are needed for N_long",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.05)
    fig.tight_layout()
    _stamp(fig, stage="CONFIRMATION", source="confirmation.json", extra=stamp)
    _save(fig, outdir, "03_response_surfaces", manifest,
          "M, N and N_long response surfaces", "confirmation.json")


def fig_effects(conf, outdir, manifest, stamp):
    plt = _plt()
    g = conf["gates"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
    # main effects
    ax = axes[0]
    names = [("memory_main_effect", r"$\Delta_m M$"), ("nonlinear_main_effect", r"$\Delta_g N$")]
    for i, (k, lab) in enumerate(names):
        v = g[k]["values"]
        ax.errorbar(v["delta"], i, xerr=[[v["delta"] - v["ci"][0]], [v["ci"][1] - v["delta"]]],
                    fmt="o", color=SERIES[i], capsize=4)
        ax.text(v["delta"], i + 0.18, f"{v['delta']:.3f}", ha="center", fontsize=8,
                color=SERIES[i])
    thr = conf["gates"]["memory_main_effect"]["thresholds"]["min"]
    ax.axvline(thr, color=STATUS["PASS"], ls="--", lw=1.4)
    ax.text(thr, 1.45, f"  threshold {thr}", color=STATUS["PASS"], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([lab for _, lab in names])
    ax.set_xlim(0, None)
    ax.set_xlabel("main effect (95% CI)")
    ax.set_title("Main effects clear the threshold", loc="left", color=INK)
    # cross effects vs equivalence margin
    ax = axes[1]
    cross = [("cross_m_to_N", r"$\Delta_m N$"), ("cross_g_to_M", r"$\Delta_g M$")]
    marg = g["cross_m_to_N"]["thresholds"]["margin"]
    ax.axvspan(marg[0], marg[1], color=SERIES[2], alpha=0.14)
    ax.text(0, 1.45, "equivalence margin", ha="center", fontsize=8, color=SERIES[2])
    for i, (k, lab) in enumerate(cross):
        v = g[k]["values"]
        ax.errorbar(v["cross_delta"], i,
                    xerr=[[max(v["cross_delta"] - v["ci"][0], 0)],
                          [max(v["ci"][1] - v["cross_delta"], 0)]],
                    fmt="o", color=SERIES[i], capsize=4)
        ax.text(v["cross_delta"], i + 0.18, f"{v['cross_delta']:.2e}", ha="center",
                fontsize=8, color=SERIES[i])
    ax.axvline(0, color=INK_MUTED, lw=1.0)
    ax.set_yticks(range(len(cross)))
    ax.set_yticklabels([lab for _, lab in cross])
    ax.set_xlim(marg[0] * 1.6, marg[1] * 1.6)
    ax.set_xlabel("cross effect (90% CI) vs equivalence margin")
    ax.set_title("Cross effects are inside the margin", loc="left", color=INK)
    fig.tight_layout()
    _stamp(fig, stage="CONFIRMATION", source="confirmation.json", extra=stamp)
    _save(fig, outdir, "04_main_and_cross_effects", manifest,
          "Main effects and cross-effect equivalence", "confirmation.json")


def fig_jacobian(conf, outdir, manifest, stamp):
    plt = _plt()
    J = np.array(conf["jacobian"]["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.grid(False)
    im = ax.imshow(np.abs(J), cmap=_cmap(), vmin=0)
    for i in range(2):
        for j in range(2):
            rel = abs(J[i, j]) / max(np.abs(J).max(), 1e-12)
            ax.text(j, i, f"{J[i, j]:.4f}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if rel > 0.55 else INK, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$\partial/\partial m$", r"$\partial/\partial g$"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([r"$M$", r"$N$"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.outline.set_edgecolor(GRID)
    ax.set_title(f"Response Jacobian\noff/diag = {conf['jacobian']['off_over_diag']:.3g},"
                 f"  angle = {conf['jacobian']['response_angle_deg']:.1f}°",
                 loc="left", color=INK)
    fig.tight_layout()
    _stamp(fig, stage="CONFIRMATION", source="confirmation.json", extra=stamp)
    _save(fig, outdir, "05_response_jacobian", manifest, "Response Jacobian", "confirmation.json")


def fig_gates(conf, outdir, manifest, stamp):
    plt = _plt()
    gates = conf["gates"]
    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(gates) + 1.3))
    ax.set_axis_off(); ax.grid(False)
    for i, (name, g) in enumerate(gates.items()):
        y = len(gates) - i
        st = g["status"]
        ax.add_patch(plt.Rectangle((0, y - 0.34), 0.028, 0.68,
                                   facecolor=STATUS.get(st, INK_MUTED), edgecolor="none"))
        ax.text(0.05, y, name.replace("_", " "), fontsize=9, va="center", color=INK)
        ax.text(0.52, y, st, fontsize=8.6, va="center", fontweight="bold",
                color=STATUS.get(st, INK_MUTED))
        ax.text(0.68, y, (g.get("reason") or "")[:66], fontsize=7, va="center", color=INK_2)
    ax.set_xlim(0, 1.7); ax.set_ylim(0.2, len(gates) + 1.1)
    ax.set_title("Mandatory confirmation gates", loc="left", color=INK)
    _stamp(fig, stage="CONFIRMATION", source="confirmation.json", extra=stamp)
    _save(fig, outdir, "09_gate_decisions", manifest, "Gate decision table", "confirmation.json")


def fig_search(search, outdir, manifest):
    plt = _plt()
    hist = [h for h in search.get("history", []) if h.get("summary")]
    if not hist:
        return _skip(manifest, "07_search_history", "Architecture-search history",
                     "no feasible candidates recorded")
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    sc = [h["score"] for h in hist]
    dm = [h["summary"]["worst_dM_dm"] for h in hist]
    dn = [h["summary"]["worst_dN_dg"] for h in hist]
    ax.scatter(dm, dn, c=sc, cmap=_cmap(), s=46, edgecolor=INK_MUTED, linewidth=0.4)
    best = min(hist, key=lambda h: h["score"])
    ax.scatter([best["summary"]["worst_dM_dm"]], [best["summary"]["worst_dN_dg"]],
               s=190, facecolor="none", edgecolor=STATUS["PASS"], linewidth=2.2)
    ax.annotate("selected", (best["summary"]["worst_dM_dm"], best["summary"]["worst_dN_dg"]),
                textcoords="offset points", xytext=(10, -12), color=STATUS["PASS"], fontsize=8)
    ax.axvline(0.10, color=INK_MUTED, ls="--", lw=1.0)
    ax.axhline(0.10, color=INK_MUTED, ls="--", lw=1.0)
    ax.text(0.105, ax.get_ylim()[0], " gate 0.10", fontsize=7.5, color=INK_MUTED)
    ax.set_xlabel(r"worst-seed $\Delta_m M$")
    ax.set_ylabel(r"worst-seed $\Delta_g N$")
    ax.set_title(f"Architecture search: {len(hist)} feasible candidates, ranked on the WORST seed",
                 loc="left", color=INK)
    fig.tight_layout()
    _stamp(fig, stage="ARCHITECTURE_SEARCH", source="search.json",
           extra="development seeds only")
    _save(fig, outdir, "07_search_history", manifest, "Architecture-search history",
          "search.json")


def fig_nm(conf, outdir, manifest, stamp):
    nm = conf.get("nonlinear_memory") or {}
    if nm.get("mean_reachable") is None:
        return _skip(manifest, "08_nonlinear_memory", "Nonlinear-memory benchmark",
                     "no Sunada curves in the confirmation artifact")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    labels = ["reachable cells\n(nu=0, or tau=0)", "boundary cells\n(nu>0 and tau>0)"]
    vals = [nm["mean_reachable"], nm["mean_boundary"]]
    ax.bar(labels, vals, color=[SERIES[0], SERIES[1]], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK_2)
    ax.set_ylabel(r"mean $NM_\nu(\tau) = {\rm corr}^2$")
    ax.set_ylim(0, max(max(vals) * 1.35, 0.1))
    ax.set_title("Sunada nonlinear-memory benchmark: the capability boundary, measured",
                 loc="left", color=INK)
    ax.text(0, -0.22, "boundary cells are provably unreachable (degree >= 2 at a positive "
                      "delay); they are reported, never gated",
            transform=ax.transAxes, fontsize=7.2, color=INK_MUTED)
    fig.tight_layout()
    _stamp(fig, stage="CONFIRMATION", source="confirmation.json", extra=stamp)
    _save(fig, outdir, "08_nonlinear_memory", manifest, "Nonlinear-memory benchmark",
          "confirmation.json")


# =============================================================================
def build_report(results_dir) -> dict:
    results_dir = Path(results_dir)
    figdir = results_dir / "figures"
    manifest = []

    def load(name):
        p = results_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    audit, smoke, cal = load("audit.json"), load("smoke.json"), load("calibration.json")
    search, stress = load("search.json"), load("stress_test.json")
    conf = load("confirmation.json")
    frozen = load("frozen_config.json")

    stamp = f"frozen {str((frozen or {}).get('sha256', 'n/a'))[:12]}"
    if conf:
        fig_surfaces(conf, figdir, manifest, stamp)
        fig_effects(conf, figdir, manifest, stamp)
        fig_jacobian(conf, figdir, manifest, stamp)
        fig_gates(conf, figdir, manifest, stamp)
        fig_nm(conf, figdir, manifest, stamp)
    else:
        for n, t in [("03_response_surfaces", "Response surfaces"),
                     ("04_main_and_cross_effects", "Main and cross effects"),
                     ("05_response_jacobian", "Response Jacobian"),
                     ("09_gate_decisions", "Gate decisions"),
                     ("08_nonlinear_memory", "Nonlinear-memory benchmark")]:
            _skip(manifest, n, t, "CONFIRMATION has not been run")
    if search:
        fig_search(search, figdir, manifest)
    else:
        _skip(manifest, "07_search_history", "Architecture-search history",
              "ARCHITECTURE_SEARCH has not been run")

    report = {"version": "v4.0",
              "claim": (conf or {}).get("claim", "SEPARATION NOT DEMONSTRATED -- "
                                                 "confirmation has not been run"),
              "audit": audit, "smoke": smoke, "calibration": cal,
              "search_best": (search or {}).get("best"),
              "search_n": (search or {}).get("n_evaluated"),
              "stress": {k: stress[k] for k in ("n_variants", "n_all_passed", "robust")}
                        if stress else None,
              "frozen_sha256": (frozen or {}).get("sha256"),
              "confirmation": conf,
              "figures": manifest,
              "n_rendered": sum(1 for m in manifest if not m.get("skipped")),
              "n_skipped": sum(1 for m in manifest if m.get("skipped"))}
    (results_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (results_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(r: dict) -> str:
    conf = r.get("confirmation") or {}
    gates = conf.get("gates", {})
    L = []
    A = L.append
    A("# DQRC V4 — Memory / Nonlinearity Separation\n")
    A(f"**{r['claim']}**\n")
    A(f"Frozen configuration: `{r.get('frozen_sha256')}`\n")

    A("\n## Gate table\n")
    A("| gate | status | key numbers |")
    A("|---|---|---|")
    for name, g in gates.items():
        v = g.get("values", {})
        if "delta" in v:
            num = f"Δ = {v['delta']:+.4f}, 95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
        elif "cross_delta" in v:
            num = (f"Δ = {v['cross_delta']:+.3e}, 90% CI "
                   f"[{v['ci'][0]:+.3e}, {v['ci'][1]:+.3e}], ratio ≤ {v['ratio_upper']:.4f}")
        elif name == "combined_capability":
            nl = v.get("N_long", {})
            num = (f"N_long improvement {nl.get('improvement', float('nan')):+.4f}, "
                   f"95% CI [{nl.get('ci', [float('nan')]*2)[0]:+.4f}, "
                   f"{nl.get('ci', [float('nan')]*2)[1]:+.4f}]")
        elif name == "controls":
            num = (f"isolation {v.get('max_dXR_dg', float('nan')):.1e}/"
                   f"{v.get('max_dXP_dm', float('nan')):.1e}, "
                   f"contaminated detected {v.get('contaminated_detected')}, "
                   f"saturation {v.get('worst_saturated_fraction', float('nan')):.3f}")
        else:
            num = "equivalence family, Holm-corrected"
        A(f"| {name.replace('_', ' ')} | **{g.get('status')}** | {num} |")

    if conf.get("jacobian"):
        J = conf["jacobian"]
        A("\n## Response Jacobian\n")
        A("```")
        A("            d/dm        d/dg")
        A(f"  M     {J['matrix'][0][0]:+10.6f}  {J['matrix'][0][1]:+10.6f}")
        A(f"  N     {J['matrix'][1][0]:+10.6f}  {J['matrix'][1][1]:+10.6f}")
        A("```")
        A(f"\noff-diagonal / diagonal mass = `{J['off_over_diag']:.4g}`; "
          f"response angle = `{J['response_angle_deg']:.2f}°`\n")

    if r.get("audit"):
        A("\n## Encoder audit\n")
        for n, a in sorted(r["audit"]["encoder_polynomial_expansion"].items()):
            A(f"- n={n}: local readout degrees {a['local_readout']['degrees_present']}, "
              f"global readout degrees {a['global_readout']['degrees_present']}, "
              f"globally affine = **{a['state_is_globally_affine']}**")
        A(f"\n{r['audit']['finding']}\n")
        A(f"\n**Capability boundary.** {r['audit']['capability_boundary']}\n")

    if r.get("stress"):
        A("\n## Stress test\n")
        A(f"{r['stress']['n_all_passed']}/{r['stress']['n_variants']} perturbed variants "
          f"pass every gate on development seeds the search never used; "
          f"robust = **{r['stress']['robust']}**\n")

    if conf.get("completeness"):
        c = conf["completeness"]
        A("\n## Confirmation integrity\n")
        A(f"- rows {c['n_rows']}/{c['n_expected']}, missing {len(c['missing'])}, "
          f"duplicates {len(c['duplicates'])}, wrong hash {len(c['wrong_config_hash'])}")
        A(f"- complete = **{c['complete']}**")
        s = conf.get("sequential", {})
        A(f"- sequential attempts {s.get('n_attempts')}/{s.get('max_attempts')}, "
          f"alpha spent {s.get('alpha_spent')}")
        A(f"- confirmation seeds: {conf.get('seeds')}\n")

    if conf.get("nonlinear_memory"):
        nm = conf["nonlinear_memory"]
        A("\n## Nonlinear-memory benchmark (Sunada)\n")
        A(f"- mean over REACHABLE cells: {nm.get('mean_reachable')}")
        A(f"- mean over BOUNDARY cells (provably unreachable): {nm.get('mean_boundary')}")
        A(f"- max over boundary cells: {nm.get('max_boundary')}\n")

    A("\n## Figures\n")
    for m in r["figures"]:
        A(f"- {'SKIPPED — ' + m['reason'] if m.get('skipped') else m['png']}: {m['title']}")
    return "\n".join(L) + "\n"
