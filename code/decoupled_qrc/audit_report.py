"""
audit_report.py -- CSVs, figures and the written report of the V4 audit.

Reads ONLY saved artifacts in results/v4/independent_audit; recomputes nothing.
Palette: fixed categorical slots (validated all-pairs), a single-hue ramp for
ordered magnitudes (ridge penalty, heat maps), the status palette for verdicts
with a text label beside every colour.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281")
STATUS = {"PASS": "#0ca30c", "FAIL": "#d03b3b"}
CLAIMS = [("C1_structural", "STRUCTURAL ROUTE ISOLATION"),
          ("C2_resource_constrained", "RESOURCE-CONSTRAINED M–NL SEPARATION"),
          ("C3_intrinsic", "INTRINSIC M–NL SEPARATION"),
          ("C4_combined", "COMBINED NONLINEAR MEMORY"),
          ("C5_quantum_specific", "QUANTUM-SPECIFIC MECHANISM")]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5})
    return plt


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", SEQ)


def _rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if p.exists() else []


def _load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, out, name, figs, stamp):
    fig.text(0.005, -0.03, stamp, fontsize=6.2, color=MUTED, va="top", transform=fig.transFigure)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    figs.append(f"figures/{name}.png")


# =============================================================================
def write_csvs(out: Path, banks: dict):
    with open(out / "raw_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bank", "arch_seed", "input_seed", "m", "g", "readout", "method", "M", "M_clip",
                    "N", "N_clip", "C1_curNL_x_oldLin", "C2_oldNL", "C3_old_x_old", "C4_old3",
                    "V4_NLONG", "null_q99", "saturated_fraction", "n_features"])
        for bank, rows in banks.items():
            for r in rows:
                for key, b in r["blocks"].items():
                    ro, meth = key.split("|")
                    mt, c = b["metrics"], b["class"]
                    w.writerow([bank, r["arch_seed"], r["input_seed"], r["m"], r["g"], ro, meth,
                                mt["M"], mt["M_clip"], mt["N"], mt["N_clip"],
                                c.get("C1_curNL_x_oldLin"), c.get("C2_oldNL"),
                                c.get("C3_old_x_old"), c.get("C4_old3"), c.get("V4_NLONG"),
                                b["null_q99"], b["saturated_fraction"],
                                r["feature_counts"].get(ro)])
    with open(out / "ipc_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bank", "arch_seed", "input_seed", "m", "g", "readout", "method", "degree",
                    "delay", "capacity_unclipped"])
        for bank, rows in banks.items():
            for r in rows:
                for key, b in r["blocks"].items():
                    ro, meth = key.split("|")
                    for k, v in b["single"].items():
                        d, t = k[1:].split("_t")
                        w.writerow([bank, r["arch_seed"], r["input_seed"], r["m"], r["g"], ro,
                                    meth, int(d), int(t), v])
    with open(out / "response_surfaces.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bank", "m", "g", "readout", "method", "metric", "mean", "sd", "n_seeds"])
        for bank, rows in banks.items():
            cells = {}
            for r in rows:
                for key, b in r["blocks"].items():
                    for met in ("M", "N"):
                        cells.setdefault((r["m"], r["g"], key, met), []).append(b["metrics"][met])
                    cells.setdefault((r["m"], r["g"], key, "N_long_C1"), []).append(
                        b["class"]["C1_curNL_x_oldLin"])
            for (m, g, key, met), v in sorted(cells.items()):
                ro, meth = key.split("|")
                w.writerow([bank, m, g, ro, meth, met, float(np.mean(v)), float(np.std(v, ddof=1))
                            if len(v) > 1 else 0.0, len(v)])


def surface(rows, key, getter):
    ms = sorted({r["m"] for r in rows}); gs = sorted({r["g"] for r in rows})
    Z = np.full((len(gs), len(ms)), np.nan)
    for i, g in enumerate(gs):
        for j, m in enumerate(ms):
            v = [getter(r["blocks"][key]) for r in rows if r["m"] == m and r["g"] == g]
            Z[i, j] = np.mean(v) if v else np.nan
    return ms, gs, Z


def figures(out: Path, conf, s6, gates, clas, rob, stamp):
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    figs = []
    plt = _plt()
    # 1 response surfaces
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    spec = [("R", "M", lambda b: b["metrics"]["M"], "M (memory, R-local)"),
            ("P", "N", lambda b: b["metrics"]["N"], "N (local nonlinear, P-local)"),
            ("ALL", "C1", lambda b: b["class"]["C1_curNL_x_oldLin"], "N_long (C1, operational)")]
    for r_i, meth in enumerate(("ridge", "ols_std")):
        for c_i, (ro, _, get, title) in enumerate(spec):
            ax = axes[r_i, c_i]; ax.grid(False)
            ms, gs, Z = surface(conf, f"{ro}|{meth}", get)
            im = ax.imshow(Z, origin="lower", cmap=_cmap(), aspect="auto",
                           extent=[-0.125, 1.125, -0.125, 1.125])
            for i, g in enumerate(gs):
                for j, m in enumerate(ms):
                    rel = (Z[i, j] - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
                    ax.text(m, g, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if rel > 0.55 else INK)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_edgecolor(GRID)
            ax.set_xlabel("m"); ax.set_ylabel("g")
            ax.set_title(f"{title} — {meth}", loc="left")
    fig.suptitle("Confirmation response surfaces (20 fresh seeds): fixed ridge vs scale-invariant OLS",
                 x=0.02, ha="left", fontsize=11, y=1.01)
    fig.tight_layout(); _save(fig, figdir, "fig01_response_surfaces", figs, stamp)

    # 2 section 6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    gvals = sorted(float(g) for g in s6["ols_raw_curve"])
    xs = [max(g, 1e-7) for g in gvals]
    ax = axes[0]
    lams = list(s6["ridge_curves"])
    cols = SEQ[1:] + ("#0d366b",)
    for i, lam in enumerate(lams):
        cur = s6["ridge_curves"][lam]
        ax.plot(xs, [cur[str(g)] if str(g) in cur else cur[g] for g in gvals], color=cols[i % len(cols)],
                lw=1.4, marker="o", ms=3)
        ax.text(xs[-1] * 1.08, (cur.get(str(gvals[-1])) if str(gvals[-1]) in cur else cur[gvals[-1]]),
                f"λ={lam}", fontsize=6.5, color=INK_2, va="center")
    ols = s6["ols_std_curve"]
    ax.plot(xs, [ols[str(g)] if str(g) in ols else ols[g] for g in gvals], color=SERIES[1], lw=2.6,
            label="OLS (standardised = raw)")
    ax.set_xscale("log"); ax.set_xlabel("g (log scale; g=0 plotted at 1e-7)")
    ax.set_ylabel("N (mean C_d, d=2..4, τ=0)")
    ax.set_title("Only the ridge penalty grades N(g); OLS is a step at g≈0", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    ax = axes[1]
    geo = s6["geometry"]
    ks = sorted(geo, key=float)
    ax.plot([max(float(k), 1e-7) for k in ks], [max(geo[k]["projection_distance_to_g1"], 1e-16) for k in ks],
            color=SERIES[0], marker="o")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("g"); ax.set_ylabel("‖P(g) − P(g=1)‖₂ (exact)")
    ax.set_title("The accessible nonlinear subspace is identical for all g ≳ 1e-4", loc="left")
    fig.tight_layout(); _save(fig, figdir, "fig02_section6_ridge_artifact", figs, stamp)

    # 3 IPC heat maps at the HH corner
    hh = [r for r in conf if r["m"] == 1.0 and r["g"] == 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, key in zip(axes, ("R|ridge", "R|ols_std", "P|ridge", "ALL|ridge")):
        ax.grid(False)
        H = np.array([[np.mean([r["blocks"][key]["single"][f"d{d}_t{t}"] for r in hh])
                       for t in range(9)] for d in range(1, 5)])
        im = ax.imshow(np.clip(H, 0, 1), cmap=_cmap(), vmin=0, vmax=1, aspect="auto", origin="lower",
                       extent=[-0.5, 8.5, 0.5, 4.5])
        for d in range(4):
            for t in range(9):
                ax.text(t, d + 1, f"{H[d, t]:.2f}", ha="center", va="center", fontsize=5.8,
                        color="#ffffff" if H[d, t] > 0.55 else INK)
        ax.set_xlabel("delay τ"); ax.set_ylabel("degree d"); ax.set_title(f"C(d,τ) {key}", loc="left")
    fig.colorbar(im, ax=axes, fraction=0.02).outline.set_edgecolor(GRID)
    _save(fig, figdir, "fig03_ipc_heatmaps_HH", figs, stamp)

    # 4 combined classes
    c4 = gates["gates_sequential"]["C4_combined"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    classes = list(c4["classes"])
    quads = ("LL", "LH", "HL", "HH")
    qcol = (MUTED, SERIES[1], SERIES[0], SERIES[2])
    for i, cls in enumerate(classes):
        qm = c4["classes"][cls]["ridge"]["quadrant_means"]
        for j, q in enumerate(quads):
            ax.bar(i + (j - 1.5) * 0.19, qm[q], width=0.17, color=qcol[j],
                   label=q if i == 0 else None)
        tag = ("supported" if c4["classes"][cls]["supported"] else "unsupported") + \
              (" · HH wins" if c4["classes"][cls]["HH_beats_all"] else " · HH does not win")
        ax.text(i, max(qm.values()) + 0.01, tag, ha="center", fontsize=7, color=INK_2)
    ax.axhline(c4["null_q99_ALL_ridge"] + 0.02, color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([c.split("_", 1)[1] for c in classes])
    ax.set_ylabel("class-mean capacity (unclipped)")
    ax.set_title("Combined nonlinear memory by target class, operational readout, ridge", loc="left")
    ax.legend(ncol=4, fontsize=7.5, loc="upper right")
    fig.tight_layout(); _save(fig, figdir, "fig04_combined_classes", figs, stamp)

    # 5 classical baseline at finite shots
    if clas:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        Ss = sorted({int(k.split("_")[0][1:]) for k in clas["shots"]})
        for lab, key, col in (("quantum joint estimator", "quantum_joint_C1", SERIES[0]),
                              ("classical product of marginals", "classical_product_C1", SERIES[1])):
            ys = [np.mean([x[key] for x in clas["shots"][f"S{S}_m1.0_g1.0"]]) for S in Ss]
            ax.plot(Ss, ys, marker="o", color=col, label=lab)
            ax.text(Ss[-1] * 1.1, ys[-1], lab.split()[0], fontsize=7.5, color=col, va="center")
        ex = np.mean([x["J_C1"] for x in clas["exact"]["m1.0_g1.0"]])
        ax.axhline(ex, color=MUTED, ls="--", lw=1); ax.text(Ss[0], ex, " exact (both identical)",
                                                            fontsize=7.5, color=MUTED, va="bottom")
        ax.set_xscale("log"); ax.set_xlabel("shots per setting"); ax.set_ylabel("C1 capacity at HH")
        ax.set_title("Same shot budget: the classical composition is never worse", loc="left")
        ax.legend(fontsize=7.5)
        fig.tight_layout(); _save(fig, figdir, "fig05_classical_baseline", figs, stamp)

    # 6 Sunada
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
    nus = (0.0, 0.5, 1.0, 2.0, 4.0)
    for ax, nu in zip(axes, nus):
        for k, (key, col) in enumerate((("R|ridge", SERIES[0]), ("P|ridge", SERIES[1]),
                                        ("J|ridge", SERIES[2]), ("CLS|ridge", MUTED))):
            ys = [np.mean([r["blocks"][key]["sunada"][f"sunada_nu{nu}_t{t}"] for r in hh])
                  for t in range(9)]
            ax.plot(range(9), ys, color=col, marker="o", ms=3, label=key.split("|")[0])
        ax.set_title(f"ν = {nu}", loc="left"); ax.set_xlabel("τ")
    axes[0].set_ylabel("NM_ν(τ) = corr²"); axes[-1].legend(fontsize=7)
    fig.suptitle("Sunada nonlinear-memory curves at the HH corner, by feature set", x=0.02,
                 ha="left", fontsize=11, y=1.04)
    fig.tight_layout(); _save(fig, figdir, "fig06_sunada", figs, stamp)

    # 7 robustness
    if rob:
        conds = list(rob["conditions"])
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(conds) + 1.4))
        for i, c in enumerate(conds):
            pc = rob["conditions"][c]
            for j, (k, col) in enumerate((("dM_ridge", SERIES[0]), ("dN_ridge", SERIES[1]),
                                          ("dN_interior_ols", SERIES[2]))):
                ax.plot(pc[k], i, "o", color=col, label=k if i == 0 else None)
        ax.axvline(0.10, color=MUTED, ls="--", lw=1)
        ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=7.5)
        ax.set_xlabel("effect (seed mean)"); ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title("Robustness: effects under every perturbation (dashed = 0.10 gate)", loc="left")
        fig.tight_layout(); _save(fig, figdir, "fig07_robustness", figs, stamp)

    # 8 verdicts
    gs = gates["gates_sequential"]
    fig, ax = plt.subplots(figsize=(8, 2.6)); ax.set_axis_off()
    for i, (k, lab) in enumerate(CLAIMS):
        st = "PASS" if gs[k]["passed"] else "FAIL"
        y = len(CLAIMS) - i
        ax.add_patch(plt.Rectangle((0, y - 0.35), 0.025, 0.7, color=STATUS[st]))
        ax.text(0.04, y, lab, va="center", fontsize=9)
        ax.text(0.78, y, st, va="center", fontsize=9, fontweight="bold", color=STATUS[st])
    ax.set_xlim(0, 1); ax.set_ylim(0.3, len(CLAIMS) + 0.8)
    ax.set_title(gates["overall"], loc="left")
    _save(fig, figdir, "fig08_verdicts", figs, stamp)
    return figs


# =============================================================================
def build(out: Path):
    out = Path(out)
    conf = _rows(out / "conf_rows.jsonl")
    dev = _rows(out / "dev_rows.jsonl")
    gates = _load(out / "gates.json")
    s6 = _load(out / "section6_confirmation.json") or _load(out / "section6.json")
    clas = _load(out / "classical_confirmation.json")
    rob = _load(out / "robustness.json")
    env, enc = _load(out / "env.json"), _load(out / "encoder.json")
    st, inv = _load(out / "structure.json"), _load(out / "invalid_controls.json")
    art = _load(out / "artifact_audit.json")
    pre_h = (out / "preregistered_audit.sha256").read_text().strip()
    stamp = f"V4 independent audit | prereg {pre_h[:12]} | commit {env['git_commit'][:10]}"
    write_csvs(out, {"development": dev, "confirmation": conf})
    figs = figures(out, conf, s6, gates, clas, rob, stamp)

    # test log
    meta = (out / "test_meta.txt").read_text()
    log = (out / "test_stdout.txt").read_text(encoding="utf-8", errors="replace")
    (out / "test_log.txt").write_text(
        "COMMAND: python -m pytest tests/ -q -rA --durations=15\n" + meta + "\n--- STDOUT ---\n"
        + log + "\n--- STDERR ---\n" + (out / "test_stderr.txt").read_text() , encoding="utf-8")

    gs = gates["gates_sequential"]
    rep = {"verdicts": {lab: ("PASS" if gs[k]["passed"] else "FAIL") for k, lab in CLAIMS},
           "overall": gates["overall"], "gates": gates, "env": env, "encoder": enc,
           "structure": st, "section6": s6, "invalid_controls": inv, "robustness": rob,
           "classical": clas, "artifacts": art, "figures": figs}
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(rep, pre_h), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "audit_manifest.json")
    man = {"prereg_sha256": pre_h, "git_commit": env["git_commit"],
           "files": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in files}}
    (out / "audit_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  report.md, report.json, 3 CSVs, {len(figs)} figures, manifest of {len(files)} files")


def _f(x, n=4):
    try:
        return f"{float(x):+.{n}f}"
    except Exception:
        return str(x)


def markdown(rep, pre_h) -> str:
    g = rep["gates"]["gates_sequential"]
    L = []; A = L.append
    A("# Independent audit of the frozen V4 architecture\n")
    A(f"Preregistration sha256 `{pre_h}` · git `{rep['env']['git_commit']}` · "
      f"confirmation: 20 fresh seeds, 5×5 grid, T=1600, exact noiseless simulation.\n")
    A("CI levels for verdicts are the SEQUENTIAL (stricter) ones: main-effect lower bound = 1st "
      "percentile, TOST interval = 1st–99th percentile, ratio bound = 99th percentile.\n")
    A("## Gate table\n")
    A("| claim | metric | threshold | estimate | CI / bound | result | evidence |")
    A("|---|---|---|---|---|---|---|")
    def row(claim, metric, thr, est, ci, ok, ev):
        A(f"| {claim} | {metric} | {thr} | {est} | {ci} | **{'PASS' if ok else 'FAIL'}** | {ev} |")
    c1 = g["C1_structural"]
    row("1", "frozen dX_R/dg, dX_P/dm, full-range (100 draws)", "= 0", "0 (all four)",
        "exact", c1["frozen_isolated"], "structure.json")
    row("1", "contaminated controls detected", "3/3", str(sum(c1["contaminated_detected"].values())) + "/3",
        "—", all(c1["contaminated_detected"].values()), "structure.json")
    for tag, key in (("2", "C2_resource_constrained"), ("3", ("C3_intrinsic", "ols_std")),
                     ("3", ("C3_intrinsic", "ols_raw"))):
        blk = g[key] if isinstance(key, str) else g[key[0]][key[1]]
        lab = "ridge" if tag == "2" else key[1]
        for name, gg in (("ΔmM", blk["main_m_M"]), ("ΔgN", blk["main_g_N"])):
            row(tag, f"{name} ({lab})", "≥0.10, LB>0", _f(gg["estimate"]), f"LB {_f(gg['lower_bound'])}",
                gg["passed"], "gates.json")
        for name, gg in (("ΔmN", blk["cross_m_N"]), ("ΔgM", blk["cross_g_M"])):
            row(tag, f"{name} ({lab})", "TOST ⊂ ±0.03, ratio<0.20", _f(gg["estimate"], 5),
                f"[{_f(gg['interval'][0], 5)}, {_f(gg['interval'][1], 5)}], ratio≤{_f(gg['ratio_upper'], 4)}",
                gg["passed"], "gates.json")
        row(tag, f"degree profile (m sweep, {lab})", "Bonferroni TOST", "—", "—",
            blk["degree_profile_m"]["passed"], "gates.json")
        row(tag, f"memory curve (g sweep, {lab})", "Bonferroni TOST", "—", "—",
            blk["memory_curve_g"]["passed"], "gates.json")
    c3 = g["C3_intrinsic"]
    for name, gg in (("interior M(1)−M(0.25), ols_std", c3["interior_m_M"]),
                     ("interior N(1)−N(0.25), ols_std", c3["interior_g_N"])):
        row("3", name, "≥0.10, LB>0", _f(gg["estimate"]), f"LB {_f(gg['lower_bound'])}", gg["passed"],
            "gates.json")
    row("3", "section-6 verdict", "HOLDS", c3["section6_verdict"], "—",
        c3["section6_verdict"].endswith("HOLDS"), "section6_confirmation.json")
    row("3", "N-target saturation over g>0 (ols_std)", "≤0.20",
        _f(c3["N_target_saturation_fraction_g_positive"], 3), "—",
        c3["N_target_saturation_fraction_g_positive"] <= 0.20, "gates.json")
    c4 = g["C4_combined"]
    for cls, v in c4["classes"].items():
        q = v["ridge"]
        worst = min(q[f"HH_minus_{o}"]["lower_bound"] for o in ("LL", "LH", "HL"))
        row("4", f"{cls} ({'supported' if v['supported'] else 'unsupported'})",
            "HH beats all, LB>0 (if supported)", _f(q["quadrant_means"]["HH"]),
            f"min LB {_f(worst)}", (v["HH_beats_all"] or not v["supported"]), "gates.json")
    c5 = g["C5_quantum_specific"]["quantum_joint_minus_classical_allpairs_C1"]
    row("5", "quantum joint − classical products (C1)", "LB>0", _f(c5["estimate"]),
        f"LB {_f(c5['lower_bound'])}", g["C5_quantum_specific"]["passed"],
        "gates.json, classical_confirmation.json")
    A("")
    A("## Verdicts\n")
    for k, lab in CLAIMS:
        A(f"- `{lab}: {'PASS' if g[k]['passed'] else 'FAIL'}`")
    A(f"\n**Overall: `{rep['overall']}`**\n")
    A(f"Supported combined classes: {c4['supported_classes']}; unsupported: {c4['unsupported_classes']}.\n")
    A(f"Section 6: **{c3['section6_verdict']}**.\n")
    A(f"Claim 5: **{g['C5_quantum_specific']['statement']}**.\n")
    A("## Figures\n")
    for f in rep["figures"]:
        A(f"- `{f}`")
    return "\n".join(L) + "\n"
