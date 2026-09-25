"""
v6_report.py -- figures, tables and report for a V6 version. Reads saved artifacts only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .audit_report import GRID, INK, INK_2, MUTED, SERIES, STATUS, _cmap, _plt, _rows, _save
from .v6_core import METHODS, V6_CLASSES

CLASS_LABEL = {"V6_C1": "C1  P2(u_t)·P1(u_{t-τ})", "V6_C2": "C2  P2(u_{t-τ})",
               "V6_C3": "C3  P1(u_{t-τ1})·P1(u_{t-τ2})", "V6_C4": "C4  P3(u_{t-τ})"}


def _surface(rows, key, get):
    ms = sorted({r["m"] for r in rows}); gs = sorted({r["g"] for r in rows})
    Z = np.array([[np.mean([get(r["blocks"][key]) for r in rows if r["m"] == m and r["g"] == g])
                   for m in ms] for g in gs])
    return ms, gs, Z


def _heat(ax, fig, ms, gs, Z, title):
    ax.grid(False)
    im = ax.imshow(Z, origin="lower", cmap=_cmap(), aspect="auto",
                   extent=[-0.125, 1.125, -0.125, 1.125])
    lo, hi = np.nanmin(Z), np.nanmax(Z)
    for i, g in enumerate(gs):
        for j, m in enumerate(ms):
            rel = (Z[i, j] - lo) / max(hi - lo, 1e-12)
            ax.text(m, g, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                    color="#ffffff" if rel > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_edgecolor(GRID)
    ax.set_xlabel("m"); ax.set_ylabel("g"); ax.set_title(title, loc="left", fontsize=9)


def figures(d: Path, rows, gates, rob, stamp) -> list:
    fd = d / "figures"; fd.mkdir(exist_ok=True)
    figs, plt = [], _plt()
    # 1 primary metrics, three readouts
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for j, meth in enumerate(METHODS):
        _heat(axes[0, j], fig, *_surface(rows, f"R|{meth}", lambda b: b["metrics"]["M"]), f"M (R only) — {meth}")
        _heat(axes[1, j], fig, *_surface(rows, f"P|{meth}", lambda b: b["metrics"]["N"]), f"N (P only) — {meth}")
    fig.suptitle("Primary metrics: M moves only with m, N only with g, under every readout",
                 x=0.02, ha="left", fontsize=11, y=1.01)
    fig.tight_layout(); _save(fig, fd, "fig01_primary_surfaces", figs, stamp)
    # 2 class surfaces (ALL), three readouts
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for i, meth in enumerate(METHODS):
        for j, cls in enumerate(V6_CLASSES):
            _heat(axes[i, j], fig, *_surface(rows, f"ALL|{meth}", lambda b, c=cls: b["class"][c]),
                  f"{CLASS_LABEL[cls]} — {meth}")
    fig.suptitle("Combined nonlinear-memory classes on the operational readout (class-mean capacity)",
                 x=0.02, ha="left", fontsize=11, y=1.01)
    fig.tight_layout(); _save(fig, fd, "fig02_class_surfaces", figs, stamp)
    # 3 quadrant bars
    comb = gates["gates"]["combined"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    qcol = (MUTED, SERIES[1], SERIES[0], SERIES[2])
    for ax, meth in zip(axes, METHODS):
        for i, cls in enumerate(V6_CLASSES):
            c = comb[cls][meth]
            for j, q in enumerate(("LL", "LH", "HL", "HH")):
                ax.bar(i + (j - 1.5) * 0.19, c["quadrant_means"][q], width=0.17, color=qcol[j],
                       label=q if i == 0 else None)
            ax.text(i, max(c["quadrant_means"].values()) + 0.02, "PASS" if c["passed"] else "FAIL",
                    ha="center", fontsize=7.5, color=STATUS["PASS" if c["passed"] else "FAIL"],
                    fontweight="bold")
        ax.axhline(np.mean([comb[c][meth]["null_q99"] for c in V6_CLASSES]) + 0.02, color=MUTED,
                   ls="--", lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(["C1", "C2", "C3", "C4"])
        top = max(max(comb[c][meth]["quadrant_means"].values()) for c in V6_CLASSES)
        ax.set_ylim(top=top * 1.25)
        ax.set_title(f"quadrant means — {meth}", loc="left", pad=10)
    axes[0].set_ylabel("class-mean capacity (unclipped)"); axes[0].legend(ncol=4, fontsize=7.5)
    fig.tight_layout(); _save(fig, fd, "fig03_class_quadrants", figs, stamp)
    # 4 degree-delay IPC at HH
    hh = [r for r in rows if r["m"] == 1.0 and r["g"] == 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, key in zip(axes, ("R|ols_std", "P|ols_std", "Q|ols_std", "ALL|ols_std")):
        ax.grid(False)
        H = np.array([[np.mean([r["blocks"][key]["single"][f"d{dd}_t{t}"] for r in hh]) for t in range(9)]
                      for dd in range(1, 5)])
        im = ax.imshow(np.clip(H, 0, 1), cmap=_cmap(), vmin=0, vmax=1, aspect="auto", origin="lower",
                       extent=[-0.5, 8.5, 0.5, 4.5])
        for a in range(4):
            for t in range(9):
                ax.text(t, a + 1, f"{H[a, t]:.2f}", ha="center", va="center", fontsize=5.6,
                        color="#ffffff" if H[a, t] > 0.55 else INK)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_xlabel("delay τ"); ax.set_ylabel("degree d"); ax.set_title(f"C(d,τ) {key}", loc="left")
    fig.colorbar(im, ax=axes, fraction=0.02).outline.set_edgecolor(GRID)
    _save(fig, fd, "fig04_degree_delay_HH", figs, stamp)
    # 5 robustness
    if rob:
        names = list(rob)
        fig, ax = plt.subplots(figsize=(9, 0.3 * len(names) + 1.4))
        for i, n in enumerate(names):
            ax.plot(rob[n]["worst_effect"], i, "o", color=SERIES[0], label="worst main/interior effect" if i == 0 else None)
            ax.plot(rob[n]["worst_hh_advantage"], i, "s", color=SERIES[1], label="worst class HH advantage" if i == 0 else None)
        ax.axvline(0.10, color=MUTED, ls="--", lw=1); ax.axvline(0.0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title("Robustness (confirmation seeds): every point must stay right of its line", loc="left")
        fig.tight_layout(); _save(fig, fd, "fig05_robustness", figs, stamp)
    # 6 classical baseline
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, (key, col) in enumerate((("ALL", SERIES[0]), ("CLS_M", SERIES[1]), ("CLS_F", MUTED))):
        v = [np.mean([r["blocks"][f"{key}|ridge"]["class"][c] for r in hh]) for c in V6_CLASSES]
        ax.bar(np.arange(4) + (k - 1) * 0.26, v, width=0.24, color=col, label=key)
    ax.set_xticks(range(4)); ax.set_xticklabels(["C1", "C2", "C3", "C4"])
    ax.set_ylabel("class capacity at (m,g)=(1,1), ridge"); ax.legend(fontsize=7.5)
    ax.set_title("Quantum operational readout vs classical products of marginals", loc="left")
    fig.tight_layout(); _save(fig, fd, "fig06_classical_baseline", figs, stamp)
    return figs


def _f(x, n=3):
    try:
        return f"{float(x):+.{n}f}"
    except Exception:
        return str(x)


def build(d: Path, version: str):
    d = Path(d)
    gates = json.loads((d / "gates.json").read_text(encoding="utf-8"))
    frozen = json.loads((d / "frozen.json").read_text(encoding="utf-8"))
    rows = _rows(d / "conf_rows.jsonl")
    rob = gates.get("robustness")
    root = d.parent.parent.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    stamp = f"{version} | frozen {frozen['sha256'][:12]} | attempt {gates['attempt']} alpha {gates['alpha']} | commit {commit[:10]}"
    figs = figures(d, rows, gates, rob, stamp)
    g = gates["gates"]
    L = []; A = L.append
    A(f"# {version} — confirmation report\n")
    A(f"Frozen sha256 `{frozen['sha256']}` · attempt {gates['attempt']} · per-look α = {gates['alpha']} · "
      f"{gates['n_rows']} rows (20 untouched seeds × 5×5) · commit `{commit[:10]}`\n")
    A(f"**All mandatory gates: {'PASS' if gates['success'] else 'FAIL'}**\n")
    A("| gate | result |\n|---|---|")
    for k, v in g["gates"].items():
        A(f"| {k} | **{'PASS' if v else 'FAIL'}** |")
    A("\n## Main and cross effects\n")
    A("| readout | ΔmM (LB) | ΔgN (LB) | interior M (LB) | interior N (LB) | m→N TOST | g→M TOST | ratio UB m→N / g→M |")
    A("|---|---|---|---|---|---|---|---|")
    for meth in METHODS:
        x = g["mains"][meth]
        c = x["cross"]
        A(f"| {meth} | {_f(x['m_M']['estimate'])} ({_f(x['m_M']['lower_bound'])}) | "
          f"{_f(x['g_N']['estimate'])} ({_f(x['g_N']['lower_bound'])}) | "
          f"{_f(x['interior_m_M']['estimate'])} ({_f(x['interior_m_M']['lower_bound'])}) | "
          f"{_f(x['interior_g_N']['estimate'])} ({_f(x['interior_g_N']['lower_bound'])}) | "
          f"[{_f(c['m_to_N']['interval'][0], 4)}, {_f(c['m_to_N']['interval'][1], 4)}] | "
          f"[{_f(c['g_to_M']['interval'][0], 4)}, {_f(c['g_to_M']['interval'][1], 4)}] | "
          f"{_f(c['m_to_N']['ratio_upper'], 4)} / {_f(c['g_to_M']['ratio_upper'], 4)} |")
    A("\n## Combined nonlinear-memory classes (operational readout ALL)\n")
    A("| class | readout | LL | LH | HL | HH | min HH-contrast LB | null99 | seeds HH-best | sat. frac | result |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for cls in V6_CLASSES:
        for meth in METHODS:
            c = g["combined"][cls][meth]; q = c["quadrant_means"]
            lb = min(v["lower_bound"] for v in c["contrasts"].values())
            A(f"| {CLASS_LABEL[cls]} | {meth} | {_f(q['LL'])} | {_f(q['LH'])} | {_f(q['HL'])} | {_f(q['HH'])} | "
              f"{_f(lb)} | {_f(c['null_q99'])} | {c['seed_consistency']:.2f} | "
              f"{c['constituent_saturated_fraction']:.2f} | **{'PASS' if c['passed'] else 'FAIL'}** |")
    A("\n## Saturation and sentinels\n")
    A(f"- per-metric saturation: {json.dumps({k: v for k, v in g['saturation'].items()})}")
    A(f"- pooled saturated fraction (max over rows): {g['pooled_saturated_fraction']:.3f}")
    A(f"- sentinels: {json.dumps(g['sentinels'])}")
    A(f"- encoder leakage: {json.dumps(g['encoder_leakage'])}")
    A("\n## Quantum-specific claim (separate)\n")
    qs = gates["quantum_specific"]
    A(f"- ALL − CLS_M (HH, ridge): {_f(qs['CLS_M']['estimate'])} (LB {_f(qs['CLS_M']['lower_bound'])})")
    A(f"- ALL − CLS_F (HH, ridge): {_f(qs['CLS_F']['estimate'])} (LB {_f(qs['CLS_F']['lower_bound'])})")
    A(f"- classical simulability: {qs['classically_simulable']}")
    A(f"- **{qs['statement']}**\n")
    if rob:
        A("## Robustness (confirmation seeds)\n\n| condition | worst effect | worst HH adv | isolated | result |\n|---|---|---|---|---|")
        for n, v in rob.items():
            A(f"| {n} | {_f(v['worst_effect'])} | {_f(v['worst_hh_advantage'])} | {v['isolated']} | "
              f"**{'PASS' if v['passed'] else 'FAIL'}** |")
    A("\n## Figures\n")
    for f in figs:
        A(f"- `{f}`")
    (d / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name != "manifest.json")
    man = {"frozen_sha256": frozen["sha256"], "git_commit": commit,
           "files": {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (d / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  report.md, {len(figs)} figures, manifest of {len(files)} files")
