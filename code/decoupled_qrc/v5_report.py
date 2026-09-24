"""
v5_report.py -- CSVs, figures and report for V5. Reads saved artifacts only.

Gate table and CSV writers are the V4 audit's (audit_report), unchanged, so the
two versions are judged and tabulated identically.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .audit_report import (CLAIMS, GRID, INK, INK_2, MUTED, SEQ, SERIES, STATUS, _cmap, _load,
                           _plt, _rows, _save, markdown, surface, write_csvs)


def figures(out: Path, conf, s6, gates, clas, rob, stamp) -> list:
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    figs, plt = [], _plt()

    # 1 response surfaces: every panel on the SAME (m, g) axes
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    spec = [("R", lambda b: b["metrics"]["M"], "M (memory, R-local)"),
            ("P", lambda b: b["metrics"]["N"], "N (local nonlinear, P-local)"),
            ("ALL", lambda b: b["class"]["C1_curNL_x_oldLin"], "C1 (operational)")]
    for r_i, meth in enumerate(("ridge", "ols_std")):
        for c_i, (ro, get, title) in enumerate(spec):
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
            ax.set_xlabel("m"); ax.set_ylabel("g"); ax.set_title(f"{title} — {meth}", loc="left")
    fig.suptitle("Confirmation response surfaces (20 fresh seeds): M moves only with m, "
                 "N only with g", x=0.02, ha="left", fontsize=11, y=1.01)
    fig.tight_layout(); _save(fig, figdir, "fig01_response_surfaces", figs, stamp)

    # 2 section 6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    gvals = sorted(float(g) for g in s6["ols_raw_curve"])
    xs = [max(g, 1e-7) for g in gvals]
    get = lambda cur, g: cur[str(g)] if str(g) in cur else cur[g]  # noqa: E731
    ax = axes[0]
    for i, lam in enumerate(s6["ridge_curves"]):
        ax.plot(xs, [get(s6["ridge_curves"][lam], g) for g in gvals], color=SEQ[1 + i % 6],
                lw=1.2, marker="o", ms=2.5)
    ax.plot(xs, [get(s6["ols_std_curve"], g) for g in gvals], color=SERIES[1], lw=2.6,
            label="OLS standardised")
    ax.plot(xs, [get(s6["ols_raw_curve"], g) for g in gvals], color=SERIES[0], lw=1.4, ls="--",
            label="OLS raw")
    geo = s6["geometry"]
    ax.plot(xs, [geo[f"{g:g}"]["exact_N"] for g in gvals], color=INK, lw=0, marker="x",
            label="exact (quadrature)")
    ax.set_xscale("log"); ax.set_xlabel("g (log; g=0 at 1e-7)"); ax.set_ylabel("N")
    ax.set_title("N(g) is continuous under every readout (blue ramp = ridge λ)", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    ks = sorted(geo, key=float)
    ax.plot([max(float(k), 1e-7) for k in ks],
            [max(geo[k]["max_principal_angle_to_g1_deg"], 1e-9) for k in ks],
            color=SERIES[0], marker="o")
    ax.set_xscale("log"); ax.set_xlabel("g"); ax.set_ylabel("principal angle to P(g=1) [deg]")
    ax.set_title("g rotates the feature, not only its amplitude", loc="left")
    fig.tight_layout(); _save(fig, figdir, "fig02_section6", figs, stamp)

    # 3 combined classes
    c4 = gates["gates_sequential"]["C4_combined"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    classes = list(c4["classes"])
    qcol = (MUTED, SERIES[1], SERIES[0], SERIES[2])
    for i, cls in enumerate(classes):
        qm = c4["classes"][cls]["ridge"]["quadrant_means"]
        for j, q in enumerate(("LL", "LH", "HL", "HH")):
            ax.bar(i + (j - 1.5) * 0.19, qm[q], width=0.17, color=qcol[j], label=q if i == 0 else None)
        tag = ("supported" if c4["classes"][cls]["supported"] else "unsupported") + \
              (" · HH wins" if c4["classes"][cls]["HH_beats_all"] else "")
        ax.text(i, max(qm.values()) + 0.01, tag, ha="center", fontsize=7, color=INK_2)
    ax.axhline(c4["null_q99_ALL_ridge"] + 0.02, color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([c.split("_", 1)[1] for c in classes])
    ax.set_ylabel("class-mean capacity (unclipped)")
    ax.set_title("Combined nonlinear memory by class: only current-NL × old-linear is reachable",
                 loc="left")
    ax.legend(ncol=4, fontsize=7.5, loc="upper right")
    fig.tight_layout(); _save(fig, figdir, "fig03_combined_classes", figs, stamp)

    # 4 classical baseline
    if clas:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        Ss = sorted(int(k[1:]) for k in clas["shots"])
        for lab, key, col in (("quantum per-shot product", "quantum_joint_C1", SERIES[0]),
                              ("classical product of marginals", "classical_product_C1", SERIES[1])):
            ys = [np.mean([x[key] for x in clas["shots"][f"S{S}"]]) for S in Ss]
            ax.plot(Ss, ys, marker="o", color=col, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("shots"); ax.set_ylabel("C1 capacity at HH")
        ax.set_title("Same shots: the classical composition matches the joint readout", loc="left")
        ax.legend(fontsize=7.5)
        fig.tight_layout(); _save(fig, figdir, "fig04_classical_baseline", figs, stamp)

    # 5 robustness
    if rob:
        conds = list(rob["conditions"])
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(conds) + 1.4))
        for i, c in enumerate(conds):
            pc = rob["conditions"][c]
            for k, col in (("dM_ridge", SERIES[0]), ("dN_ridge", SERIES[1]),
                           ("dN_interior_ols", SERIES[2])):
                ax.plot(pc[k], i, "o", color=col, label=k if i == 0 else None)
        ax.axvline(0.10, color=MUTED, ls="--", lw=1)
        ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=7.5)
        ax.set_xlabel("effect (dev-seed mean)"); ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title("Robustness on dev seeds (dashed = 0.10 gate)", loc="left")
        fig.tight_layout(); _save(fig, figdir, "fig05_robustness", figs, stamp)

    # 6 verdicts
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
    _save(fig, figdir, "fig06_verdicts", figs, stamp)
    return figs


def build(out: Path):
    out = Path(out)
    conf, dev = _rows(out / "conf_rows.jsonl"), _rows(out / "dev_rows.jsonl")
    gates = _load(out / "gates.json")
    frozen = _load(next(out.glob("frozen_*.json")))
    ver = frozen["payload"]["version"]
    s6 = _load(out / "section6_confirmation.json")
    clas, rob = _load(out / "classical_confirmation.json"), _load(out / "robustness.json")
    root = out.parent.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                            text=True).stdout.strip()
    stamp = f"{ver} | frozen {frozen['sha256'][:12]} | commit {commit[:10]}"
    write_csvs(out, {"development": dev, "confirmation": conf})
    figs = figures(out, conf, s6, gates, clas, rob, stamp)
    gs = gates["gates_sequential"]
    rep = {"verdicts": {lab: ("PASS" if gs[k]["passed"] else "FAIL") for k, lab in CLAIMS},
           "overall": gates["overall"], "success_claims_1_to_4": gates["success_claims_1_to_4"],
           "gates": gates, "env": {"git_commit": commit}, "frozen": frozen,
           "encoder": _load(out / "encoder.json"), "search": _load(out / "search.json"),
           "dev_gates": _load(out / "dev_gates.json"), "section6": s6,
           "invalid_controls": _load(out / "invalid_controls.json"), "robustness": rob,
           "classical": clas, "figures": figs}
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = markdown(rep, frozen["sha256"])
    md = md.replace("# Independent audit of the frozen V4 architecture",
                    f"# {ver} — random-SWAP memory × two-copy processor, confirmation") \
           .replace("Preregistration sha256", f"Frozen {ver} sha256")
    passed = {k: gs[k]["passed"] for k, _ in CLAIMS}
    if all(v for k, v in passed.items() if k != "C5_quantum_specific") and not passed["C5_quantum_specific"]:
        note = (f"\n## Reading of the overall label\n\nThe label `{gates['overall']}` comes from the "
                "V4 audit's `overall()` mapping, which has only the brief's four outcomes. The pattern "
                "*claims 1-4 PASS, claim 5 FAIL* matches none of them and falls through to the last "
                "branch. That label is **inaccurate here: claim 3 (intrinsic separation) PASSED**. "
                "Correct reading: memory and nonlinearity are separately and intrinsically "
                "controllable (claims 1-3), combined nonlinear memory is highest at high-m/high-g for "
                "the one supported class (claim 4), and the combining mechanism is classically "
                "reproducible (claim 5). It is not `WORKS AS CLAIMED` in the brief's sense because "
                "claim 5 fails.\n")
        md = md.replace("## Figures", note + "\n## Figures", 1)
        rep["overall_reading"] = "claims 1-4 PASS; claim 5 FAIL (classically reproducible)"
        (out / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(md, encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"
                   and "search" not in p.parts)
    man = {"frozen_sha256": frozen["sha256"], "git_commit": commit,
           "files": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in files}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  report.md, report.json, 3 CSVs, {len(figs)} figures, manifest of {len(files)} files")
