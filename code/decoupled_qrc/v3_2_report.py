"""
v3_2_report.py -- figures and tables, regenerated ONLY from saved, validated
results.

No figure in this module runs a simulation. Every one reads a JSON artifact
written by a completed stage, and every one is stamped with the stage, the seed
counts, the uncertainty convention, the data source and the configuration
checksum, so a figure can always be traced back to the run that produced it.
A figure whose inputs do not exist is SKIPPED and recorded as skipped in the
manifest, never silently omitted and never back-filled from a different stage.

PALETTE. Categorical slots are used in fixed order and never cycled; the three
used here validate on all pairs for normal vision and for deuteranopia and
tritanopia. The aqua slot sits below 3:1 against the light surface, so every
series additionally carries a visible direct label rather than relying on hue.
Magnitude (the (g,J) response surface) uses a single-hue light-to-dark blue
ramp -- never a rainbow. Gate verdicts use the reserved status palette and
always ship a text verdict beside the colour, so status is never colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- design-system parameters (see the data-visualisation reference palette) --
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # fixed order, never cycled
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281")
STATUS = {"PASS": "#0ca30c", "FAIL": "#d03b3b", "NOT_EVALUABLE": "#fab219"}
STATUS_MARK = {"PASS": "PASS", "FAIL": "FAIL", "NOT_EVALUABLE": "NOT EVALUABLE"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
    })
    return plt


def _stamp(fig, *, stage, source, config_hash, n_res=None, n_inp=None, uncertainty=None):
    """Provenance line every figure must carry."""
    bits = [f"stage: {stage}"]
    if n_res is not None:
        bits.append(f"{n_res} reservoir x {n_inp} input seeds")
    if uncertainty:
        bits.append(uncertainty)
    bits.append(f"source: {source}")
    bits.append(f"config {str(config_hash)[:12]}")
    fig.text(0.005, -0.035, "  |  ".join(bits), fontsize=6.2, color=INK_MUTED,
             va="top", transform=fig.transFigure)


def _save(fig, outdir: Path, name: str, manifest: list, title: str, source: str, note: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)
    png, pdf = outdir / f"{name}.png", outdir / f"{name}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")                # vector
    import matplotlib.pyplot as plt
    plt.close(fig)
    manifest.append({"name": name, "title": title, "png": png.name, "pdf": pdf.name,
                     "data_source": source, "note": note})
    return png


def _skip(manifest: list, name: str, title: str, reason: str):
    manifest.append({"name": name, "title": title, "png": None, "pdf": None,
                     "data_source": None, "skipped": True, "reason": reason})


# =============================================================================
# Figures
# =============================================================================
def fig_architecture(outdir, manifest, cfg_hash, stage):
    """F01 -- architecture and causal-control diagram."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_axis_off()
    ax.grid(False)

    def box(x, y, w, h, label, sub, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=SURFACE, edgecolor=color, lw=2,
                                   joinstyle="round"))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", fontsize=10,
                color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=7.5, color=INK_2)

    box(0.06, 0.55, 0.26, 0.34, "memory  M(m)", "L rails, never reset", SERIES[0])
    box(0.06, 0.08, 0.26, 0.34, "processor  P(g, J)", "reset each step", SERIES[1])
    ax.annotate("", xy=(0.06, 0.72), xytext=(0.005, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.6))
    ax.annotate("", xy=(0.06, 0.25), xytext=(0.005, 0.25),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.6))
    ax.text(0.0, 0.49, "u(t)", fontsize=9, color=INK, fontweight="bold")
    ax.annotate("", xy=(0.19, 0.42), xytext=(0.19, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=SERIES[2], lw=2.2))
    ax.text(0.205, 0.485, r"$\lambda$  (M$\rightarrow$P channel)", fontsize=8, color=SERIES[2])

    ax.annotate("", xy=(0.46, 0.72), xytext=(0.32, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.6))
    ax.annotate("", xy=(0.46, 0.25), xytext=(0.32, 0.25),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.6))
    ax.text(0.47, 0.70, r"$X_M \rightarrow M_{\mathrm{long}} = \sum_{\tau\geq\tau_{\min}} C_{1,\tau}$",
            fontsize=9, color=INK)
    ax.text(0.47, 0.23, r"$X_P \rightarrow NL_0$  (degree $\geq 2$, $\tau = 0$)",
            fontsize=9, color=INK)
    ax.text(0.47, 0.50,
            "module-specific readouts:\ncombined-feature readouts are secondary diagnostics only",
            fontsize=7.4, color=INK_MUTED, va="center")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("V3.2 dual-route architecture and causal controls", loc="left", color=INK)
    _stamp(fig, stage=stage, source="architecture spec", config_hash=cfg_hash)
    return _save(fig, outdir, "01_architecture_causal_controls", manifest,
                 "Architecture and causal-control diagram", "architecture spec")


def fig_memory_profiles(cal, outdir, manifest, cfg_hash, stage):
    """F02 -- delay profiles C_{1,tau} across m, per mechanism."""
    plt = _mpl()
    mechs = list(cal.get("memory", {}))
    if not mechs:
        return _skip(manifest, "02_memory_delay_profiles", "Memory delay profiles",
                     "no memory calibration in the saved artifact")
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.6 * len(mechs), 3.3), squeeze=False,
                             sharey=True)
    n_res = cal["settings"]["n_reservoir_seeds"]
    n_inp = cal["settings"]["n_input_seeds"]
    for ax, mech in zip(axes[0], mechs):
        rows = cal["memory"][mech]["M2_profile"]
        chosen = [rows[0], rows[len(rows) // 2], rows[-1]]
        for row, colour in zip(chosen, SERIES):
            descs = [d for ds in row["descriptors"].values() for d in ds]
            # the per-seed delay profile is summarised by M_long/M_short/centroid;
            # plot the median M_long contribution by delay bucket
            ax.plot([0, 1, 2], [np.median([d["M_short"] for d in descs]),
                                np.median([d["M_long"] for d in descs]),
                                np.median([d["delay_centroid"] for d in descs])],
                    color=colour, marker="o", label=f"m={row['m']:.2f}")
            ax.text(2.04, np.median([d["delay_centroid"] for d in descs]),
                    f"m={row['m']:.2f}", color=colour, fontsize=7.5, va="center")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([r"$M_{\rm short}$", r"$M_{\rm long}$", "delay centroid"])
        ax.set_title(mech.replace("_", " "), loc="left", color=INK)
        ax.set_xlim(-0.2, 2.7)
    axes[0][0].set_ylabel("capacity / delay (bias-corrected)")
    fig.suptitle("Memory delay structure across the control m", x=0.02, ha="left",
                 color=INK, y=1.04)
    fig.tight_layout()
    _stamp(fig, stage=stage, source="calibration_results.json", config_hash=cfg_hash,
           n_res=n_res, n_inp=n_inp, uncertainty="median across seeds")
    return _save(fig, outdir, "02_memory_delay_profiles", manifest,
                 "Memory delay structure across m", "calibration_results.json")


def fig_memory_selection(cal, outdir, manifest, cfg_hash, stage):
    """F03 -- slope, interior margin and which candidates are valid."""
    plt = _mpl()
    mechs = list(cal.get("memory", {}))
    if not mechs:
        return _skip(manifest, "03_memory_slope_selection", "Memory slope and selection",
                     "no memory calibration in the saved artifact")
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.8 * len(mechs), 3.4), squeeze=False,
                             sharey=True)   # same measure -> one scale, never independent axes
    for ax, mech in zip(axes[0], mechs):
        cands = cal["memory"][mech]["M4_candidates"]
        m = [c["m"] for c in cands]
        slope = [c["slope_median"] for c in cands]
        valid = [c["valid"] for c in cands]
        ax.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
        ax.plot(m, slope, color=SERIES[0], marker="o", zorder=3,
                label=r"$\partial M_{\rm long}/\partial \tilde m$")
        ok = [(a, b) for a, b, v in zip(m, slope, valid) if v]
        if ok:
            ax.scatter([a for a, _ in ok], [b for _, b in ok], s=110, facecolor="none",
                       edgecolor=STATUS["PASS"], lw=2, zorder=4, label="passes hard filters")
        sel = cal["memory"][mech].get("M4_selected")
        if sel:
            ax.axvline(sel["m"], color=SERIES[2], lw=1.6, ls="--", zorder=2)
            ax.text(sel["m"], ax.get_ylim()[1], f"  selected m*={sel['m']:.2f}",
                    color=SERIES[2], fontsize=8, va="top")
        win = cal["memory"][mech].get("M3_stable_window")
        if win:
            ax.axvspan(win["m_lo"], win["m_hi"], color=SERIES[0], alpha=0.07, zorder=0)
            ax.text((win["m_lo"] + win["m_hi"]) / 2, ax.get_ylim()[0],
                    "sign-consistent window", color=SERIES[0], fontsize=7.5,
                    ha="center", va="bottom")
        ax.set_xlabel("m")
        ax.set_title(mech.replace("_", " "), loc="left", color=INK)
    axes[0][0].set_ylabel(r"$\partial M_{\rm long}/\partial \tilde m$  (normalised control)")
    axes[0][0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Memory candidate selection: slope stability and interior margin, not maximum capacity",
                 x=0.02, ha="left", color=INK, fontsize=10, y=1.06)
    fig.tight_layout()
    _stamp(fig, stage=stage, source="calibration_results.json", config_hash=cfg_hash,
           n_res=cal["settings"]["n_reservoir_seeds"], n_inp=cal["settings"]["n_input_seeds"],
           uncertainty="median slope across seeds")
    return _save(fig, outdir, "03_memory_slope_selection", manifest,
                 "Memory slope and interior-candidate selection", "calibration_results.json")


def fig_ablations(cal, outdir, manifest, cfg_hash, stage):
    """F04 -- exact instantaneous capacity per processor ablation."""
    plt = _mpl()
    dist = cal.get("processor", {}).get("P4_ablation_distinctness")
    if not dist:
        return _skip(manifest, "04_processor_ablations", "Processor ablation capacities",
                     "no ablation ladder in the saved artifact")
    rows = dist["rows"]
    names = list(rows)
    vals = [rows[n]["NL_0_exact"] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colours = [SERIES[2] if v < 1e-9 else SERIES[0] for v in vals]
    ax.barh(names, vals, color=colours, height=0.62)
    for n, v in zip(names, vals):
        ax.text(v + 0.05, n, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel(r"exact instantaneous nonlinear capacity  $NL_0 = \sum_{d\geq2} C_d$")
    ax.set_xlim(0, max(vals) * 1.22 + 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Processor ablation ladder (noiseless quadrature, infinite-data limit)",
                 loc="left", color=INK, pad=20)
    ax.text(0.0, 1.03, "a zero-length bar is EXACTLY zero: with no interaction the processor is a "
                       "product of single-qubit channels, so every feature stays affine in u",
            transform=ax.transAxes, fontsize=7.4, color=INK_MUTED, va="bottom")
    _stamp(fig, stage=stage, source="calibration_results.json", config_hash=cfg_hash,
           uncertainty="exact (no sampling)")
    return _save(fig, outdir, "04_processor_ablations", manifest,
                 "Processor ablation capacities", "calibration_results.json")


def fig_distinctness(cal, outdir, manifest, cfg_hash, stage):
    """F05 -- Hamiltonian / unitary / state / feature differences per pair."""
    plt = _mpl()
    dist = cal.get("processor", {}).get("P4_ablation_distinctness")
    if not dist:
        return _skip(manifest, "05_ablation_distinctness", "Ablation distinctness",
                     "no ablation ladder in the saved artifact")
    pairs = [p for p in dist["pairs"] if p["dH"] is not None]
    labels = [f"{p['a']}\n{p['b']}" for p in pairs]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.52 * len(pairs)), 3.6))
    x = np.arange(len(pairs))
    for k, (key, lab, colour) in enumerate([("dU", "unitary", SERIES[0]),
                                            ("state_trace_distance", "state", SERIES[1]),
                                            ("dFeatures", "features", SERIES[2])]):
        y = np.array([max(p[key], 1e-18) for p in pairs], dtype=float)
        ax.plot(x, y, marker="o", color=colour, lw=1.6, ls="none" if key == "dFeatures" else "-",
                label=lab)
        ax.text(x[-1] + 0.15, y[-1], lab, color=colour, fontsize=8, va="center")
    ax.axhline(1e-10, color=INK_MUTED, lw=1.0, ls=":")
    ax.text(0, 1.3e-10, "feature-identity threshold", fontsize=7.2, color=INK_MUTED)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.2, rotation=90)
    ax.set_ylabel("difference (log scale)")
    ax.set_xlim(-0.5, len(pairs) + 1.4)
    ax.set_title("Ablation distinctness: different unitaries can share features by proven symmetry",
                 loc="left", color=INK)
    ax.legend(fontsize=7.5, loc="lower left")
    _stamp(fig, stage=stage, source="calibration_results.json", config_hash=cfg_hash,
           uncertainty="exact (no sampling)")
    return _save(fig, outdir, "05_ablation_distinctness", manifest,
                 "Hamiltonian/unitary/state/feature distinctness", "calibration_results.json")


def fig_response_surface(cal, outdir, manifest, cfg_hash, stage):
    """F08 -- NL_0 over the (g, J) grid. Magnitude -> single-hue ramp."""
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    surf = cal.get("processor", {}).get("P5_surface")
    if not surf:
        return _skip(manifest, "08_processor_response_surface", "Processor response surface",
                     "no (g,J) surface in the saved artifact")
    gs = sorted({r["g"] for r in surf})
    Js = sorted({r["J"] for r in surf})
    Z = np.full((len(Js), len(gs)), np.nan)
    for r in surf:
        Z[Js.index(r["J"]), gs.index(r["g"])] = r["NL_0_median"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    ax.grid(False)
    im = ax.imshow(Z, origin="lower", cmap=cmap, aspect="auto",
                   extent=[min(gs), max(gs), min(Js), max(Js)])
    for i, J in enumerate(Js):
        for j, g in enumerate(gs):
            if np.isfinite(Z[i, j]):
                rel = (Z[i, j] - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
                ax.text(g, J, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                        color="#ffffff" if rel > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(r"$NL_0$ (bias-corrected)", fontsize=8)
    cb.outline.set_edgecolor(GRID)
    ax.set_xlabel("g  (two-body)")
    ax.set_ylabel("J  (four-body)")
    ax.set_title("Instantaneous nonlinear capacity over the processor controls",
                 loc="left", color=INK)
    _stamp(fig, stage=stage, source="calibration_results.json", config_hash=cfg_hash,
           n_res=cal["settings"]["n_reservoir_seeds"], n_inp=cal["settings"]["n_input_seeds"],
           uncertainty="median across seeds")
    return _save(fig, outdir, "08_processor_response_surface", manifest,
                 "Processor response surface over (g,J)", "calibration_results.json")


def fig_interaction_gain(cal, outdir, manifest, cfg_hash, stage):
    """F09 -- paired interaction-generated capacity and the encoder fraction."""
    plt = _mpl()
    p3 = cal.get("processor", {}).get("P3_delta_interaction")
    if not p3:
        return _skip(manifest, "09_interaction_generated_nl", "Interaction-generated capacity",
                     "no paired interaction test in the saved artifact")
    nested = p3["nested"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    keys = sorted(nested)
    for i, k in enumerate(keys):
        vals = nested[k]
        ax.scatter([i] * len(vals), vals, color=SERIES[0], s=42, zorder=3)
    ax.axhline(0, color=INK_MUTED, lw=1.2)
    ax.axhspan(p3["lo"], p3["hi"], color=SERIES[0], alpha=0.12, zorder=0)
    ax.axhline(p3["point"], color=SERIES[0], lw=2, zorder=2)
    ax.text(len(keys) - 0.4, p3["point"],
            f"  paired mean {p3['point']:.3f}\n  95% CI [{p3['lo']:.3f}, {p3['hi']:.3f}]",
            fontsize=8, color=SERIES[0], va="center")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"reservoir {k}" for k in keys], fontsize=8)
    ax.set_ylabel(r"$\Delta NL_{\rm interaction} = NL_0(g,J) - NL_0(0,0)$")
    ax.set_xlim(-0.5, len(keys) + 0.9)
    ax.set_title("Interaction-generated nonlinearity (paired, common random numbers)",
                 loc="left", color=INK)
    fe = cal["processor"].get("P3_f_enc", {})
    if fe:
        ax.text(0.0, -0.20, f"encoder-only fraction f_enc median = {fe['median']:.4f} "
                            f"(gate: < 0.5).  V3.1: f_enc ~ 1.09",
                transform=ax.transAxes, fontsize=7.6, color=INK_MUTED)
    _stamp(fig, stage=stage, source="calibration_results.json", config_hash=cfg_hash,
           n_res=len(keys), n_inp=cal["settings"]["n_input_seeds"],
           uncertainty="hierarchical bootstrap 95% CI")
    return _save(fig, outdir, "09_interaction_generated_nl", manifest,
                 "Interaction-generated nonlinear capacity", "calibration_results.json")


def fig_gate_chart(gates: dict, outdir, manifest, cfg_hash, stage, claim_level=None):
    """F17 -- acceptance-gate decision chart. Status colour + text verdict."""
    plt = _mpl()
    if not gates:
        return _skip(manifest, "17_gate_decision_chart", "Acceptance-gate decision chart",
                     "no gate table in the saved artifact")
    names = list(gates)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(names) + 1.5))
    ax.set_axis_off()
    ax.grid(False)
    for i, name in enumerate(names):
        g = gates[name]
        status = g.get("status", "NOT_EVALUABLE")
        y = len(names) - i
        ax.add_patch(plt.Rectangle((0.0, y - 0.34), 0.03, 0.68,
                                   facecolor=STATUS.get(status, INK_MUTED), edgecolor="none"))
        ax.text(0.05, y, name.replace("_", " "), fontsize=9, va="center", color=INK)
        ax.text(0.52, y, STATUS_MARK.get(status, status), fontsize=8.6, va="center",
                color=STATUS.get(status, INK_MUTED), fontweight="bold")
        ax.text(0.67, y, (g.get("reason") or "")[:74], fontsize=7, va="center", color=INK_2)
    ax.set_xlim(0, 1.75)
    ax.set_ylim(0.2, len(names) + 1.1)
    title = "Acceptance gates"
    if claim_level is not None:
        title += f"   -- highest supported claim level: {claim_level}"
    ax.set_title(title, loc="left", color=INK)
    _stamp(fig, stage=stage, source="gate table", config_hash=cfg_hash)
    return _save(fig, outdir, "17_gate_decision_chart", manifest,
                 "Acceptance-gate decision chart", "gate table")


# =============================================================================
# Driver
# =============================================================================
def build_report(results_dir, *, stage: str) -> dict:
    """Regenerate every figure the SAVED artifacts support. Nothing is computed."""
    results_dir = Path(results_dir)
    figdir = results_dir / "figures"
    manifest: list = []

    cal_path = results_dir / "calibration_results.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
    man_path = results_dir / "run_manifest_calibration.json"
    cfg_hash = "n/a"
    if man_path.exists():
        cfg_hash = json.loads(man_path.read_text(encoding="utf-8")).get("git_commit", "n/a")

    fig_architecture(figdir, manifest, cfg_hash, stage)
    if cal:
        fig_memory_profiles(cal, figdir, manifest, cfg_hash, "CALIBRATION")
        fig_memory_selection(cal, figdir, manifest, cfg_hash, "CALIBRATION")
        fig_ablations(cal, figdir, manifest, cfg_hash, "CALIBRATION")
        fig_distinctness(cal, figdir, manifest, cfg_hash, "CALIBRATION")
        fig_response_surface(cal, figdir, manifest, cfg_hash, "CALIBRATION")
        fig_interaction_gain(cal, figdir, manifest, cfg_hash, "CALIBRATION")
    else:
        for n, t in [("02_memory_delay_profiles", "Memory delay profiles"),
                     ("03_memory_slope_selection", "Memory slope and selection"),
                     ("04_processor_ablations", "Processor ablation capacities"),
                     ("05_ablation_distinctness", "Ablation distinctness"),
                     ("08_processor_response_surface", "Processor response surface"),
                     ("09_interaction_generated_nl", "Interaction-generated capacity")]:
            _skip(manifest, n, t, "CALIBRATION has not been run")

    smoke_path = results_dir / "smoke_results.json"
    gates = {}
    if smoke_path.exists():
        gates = json.loads(smoke_path.read_text(encoding="utf-8")).get("gate_table", {})
    fig_gate_chart(gates, figdir, manifest, cfg_hash, stage, claim_level=None)

    # Figures that require stages which have not run are recorded as skipped,
    # never back-filled from a different stage.
    for n, t in [("06_raw_null_bias_corrected_ipc", "Raw / null / bias-corrected IPC"),
                 ("07_order_delay_heatmap", "Order-delay IPC heatmap"),
                 ("10_full_normalised_jacobian", "Full normalised Jacobian"),
                 ("11_derivative_confidence_intervals", "Intended and unintended derivatives"),
                 ("12_response_vector_geometry", "Response-vector geometry"),
                 ("13_per_seed_selectivity", "Per-seed selectivity ratios"),
                 ("14_module_retention", "Module-retention comparison"),
                 ("15_centre_neighbourhood_map", "Centre-and-neighbourhood pass map"),
                 ("16_matched_resource_pareto", "Matched-resource Pareto")]:
        if not any(m["name"] == n for m in manifest):
            _skip(manifest, n, t, "requires DISCOVERY/CONFIRMATION, which did not run")

    out = {"stage": stage, "figures": manifest,
           "n_rendered": sum(1 for m in manifest if not m.get("skipped")),
           "n_skipped": sum(1 for m in manifest if m.get("skipped")),
           "palette": {"categorical": list(SERIES), "sequential": list(SEQ),
                       "status": STATUS,
                       "validation": ("all-pairs CVD deltaE 9.2 (deutan) / normal-vision 24.0, "
                                      "light surface; aqua slot is sub-3:1 vs surface so every "
                                      "series carries a visible direct label")}}
    (results_dir / "figure_manifest.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out
