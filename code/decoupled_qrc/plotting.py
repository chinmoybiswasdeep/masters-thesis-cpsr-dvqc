"""
plotting.py -- matplotlib helpers matching the Part 15 notebook plot list.
Every function returns (fig, interpretation) -- the notebook prints
`interpretation` immediately below the figure, per the task spec's "print an
interpretation immediately below every major figure" instruction. No
function here computes anything scientific; they only visualize numbers
produced by the other modules.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_tradeoff_vs_g(kappa_grid, memory_vals, nl_vals, total_vals, title="Monolithic reservoir: memory-NL trade-off vs kappa"):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(kappa_grid, memory_vals, "o-", color="tab:blue", label="Memory (IPC$_1$)")
    ax1.plot(kappa_grid, nl_vals, "s-", color="tab:red", label="Nonlinearity ($\\sum$IPC$_{\\geq2}$)")
    ax1.plot(kappa_grid, total_vals, "^--", color="tab:gray", label="Total IPC", alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\kappa$ (SYK4-like $\leftrightarrow$ SYK2-like)")
    ax1.set_ylabel("Capacity")
    ax1.set_title(title)
    ax1.legend()
    fig.tight_layout()

    memory_vals = np.asarray(memory_vals)
    nl_vals = np.asarray(nl_vals)
    corr = np.corrcoef(memory_vals, nl_vals)[0, 1] if len(memory_vals) > 2 else float("nan")
    interp = (f"Correlation(Memory, Nonlinearity) across the kappa scan = {corr:.3f}. "
              f"A negative correlation is the signature of the conventional memory-nonlinearity "
              f"trade-off (Cindrak et al. 2026): more of one comes at the cost of the other on this "
              f"single, shared reservoir.")
    return fig, interp


def plot_pareto_comparison(baseline_points, dqrc_points, baseline_frontier_idx, title="Memory-Nonlinearity Pareto frontier"):
    fig, ax = plt.subplots(figsize=(6, 5))
    bm = np.array([p[0] for p in baseline_points])
    bn = np.array([p[1] for p in baseline_points])
    dm = np.array([p[0] for p in dqrc_points])
    dn = np.array([p[1] for p in dqrc_points])
    ax.scatter(bm, bn, color="tab:gray", alpha=0.5, label="Monolithic baseline (all points)")
    fm = bm[baseline_frontier_idx]
    fn = bn[baseline_frontier_idx]
    order = np.argsort(fm)
    ax.plot(fm[order], fn[order], "-", color="black", lw=2, label="Baseline Pareto frontier")
    ax.scatter(dm, dn, color="tab:red", marker="^", label="DQRC points")
    ax.set_xlabel("Memory (IPC$_1$)")
    ax.set_ylabel("Nonlinearity ($\\sum$IPC$_{\\geq2}$)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    from .metrics import is_dominated
    frontier_pts = list(zip(fm, fn))
    n_outside = sum(1 for p in zip(dm, dn) if not is_dominated(p, frontier_pts))
    interp = (f"{n_outside}/{len(dqrc_points)} DQRC points are NOT dominated by the baseline's own "
              f"Pareto frontier (i.e. lie outside the conventional trade-off region). This alone is "
              f"not a statistically validated claim -- see the bootstrapped frontier_comparison result "
              f"for a seed-aware verdict.")
    return fig, interp


def plot_memory_capacity_curve(k_values, C, memory_lifetime_val, title="Delay-resolved memory capacity C(k)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, C, "o-")
    ax.axvline(memory_lifetime_val, color="tab:red", linestyle="--", label=f"lifetime~{memory_lifetime_val:.1f}")
    ax.set_xlabel("delay k")
    ax.set_ylabel("C(k)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    interp = f"Memory capacity drops below threshold near k={memory_lifetime_val:.1f}."
    return fig, interp


def plot_chaos_diagnostics_vs_kappa(scan_results, found_eoc_kappa, title="Processor chaos diagnostics vs kappa_processor"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    kappas = [r["kappa_processor"] for r in scan_results]
    r_vals = [r["level_spacing_ratio"] for r in scan_results]
    op_ent = [r["operator_entanglement"] for r in scan_results]
    poisson = scan_results[0]["r_poisson_mean"]
    coe = scan_results[0]["r_coe_mean"]

    ax1.plot(kappas, r_vals, "o-", label="<r> (processor)")
    ax1.axhline(poisson, color="tab:green", ls="--", label="Poisson (integrable)")
    ax1.axhline(coe, color="tab:red", ls="--", label="COE (chaotic)")
    ax1.axvline(found_eoc_kappa, color="black", ls=":", label=f"found EOC kappa={found_eoc_kappa:.3f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("kappa_processor")
    ax1.set_ylabel("<r>")
    ax1.legend(fontsize=8)

    ax2.plot(kappas, op_ent, "s-", color="tab:purple")
    ax2.set_xscale("log")
    ax2.set_xlabel("kappa_processor")
    ax2.set_ylabel("operator entanglement entropy")
    fig.suptitle(title)
    fig.tight_layout()
    interp = (f"Independently-derived EOC kappa_processor={found_eoc_kappa:.3f} (crossover midpoint between "
              f"Poisson and COE reference <r>), NOT the repo's stored kappa=0.960 (that value was selected "
              f"by NARMA2 task performance on different data -- docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md sections 3/6).")
    return fig, interp


def plot_M_NL_vs_g_processor(kappas, M_vals, NL_vals, title="DQRC: M and NL vs g_processor (decoupling signature)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kappas, M_vals, "o-", color="tab:blue", label="M (memory)")
    ax.plot(kappas, NL_vals, "s-", color="tab:red", label="NL (nonlinearity)")
    ax.set_xscale("log")
    ax.set_xlabel("kappa_processor")
    ax.set_ylabel("Capacity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    M_vals = np.asarray(M_vals)
    spread = float(np.std(M_vals) / (np.mean(M_vals) + 1e-9))
    interp = (f"Relative spread of M across the kappa_processor scan = {spread:.3f}. The desired decoupling "
              f"signature is M approximately CONSTANT while NL rises near the processor's own EOC point -- "
              f"a small spread supports that; a large spread (M itself trading off against g_processor) would "
              f"indicate the memory and processor subsystems are NOT well decoupled in this configuration.")
    return fig, interp


def plot_M_NL_vs_memory_resource(n_mem_values, M_vals, NL_vals, title="DQRC: M and NL vs memory resource N_M"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_mem_values, M_vals, "o-", color="tab:blue", label="M (memory)")
    ax.plot(n_mem_values, NL_vals, "s-", color="tab:red", label="NL (nonlinearity)")
    ax.set_xlabel("N_M (memory qubits)")
    ax.set_ylabel("Capacity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    interp = "M should rise with N_M (more memory resource -> more memory); NL rising too would indicate memory-resource growth leaks into nonlinear capacity, undermining the decoupling claim."
    return fig, interp


def plot_jacobian_heatmap(jac_result, title="Decoupling Jacobian"):
    fig, ax = plt.subplots(figsize=(4, 4))
    mat = np.array([[jac_result.dM_dm, jac_result.dM_dg], [jac_result.dNL_dm, jac_result.dNL_dg]])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-np.max(np.abs(mat)), vmax=np.max(np.abs(mat)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["m (memory ctrl)", "g (processor ctrl)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["dM", "dNL"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im)
    ax.set_title(title)
    fig.tight_layout()
    interp = (f"cross_coupling={jac_result.cross_coupling:.3f}, decoupling_score={jac_result.decoupling_score:.3f} "
              f"(an engineering diagnostic, not a fundamental quantity -- see the raw Jacobian entries above for "
              f"the primary result).")
    return fig, interp


def plot_shadow_convergence(sweep, title="Classical shadow vs exact Pauli readout: RMSE vs shots"):
    fig, ax = plt.subplots(figsize=(6, 4))
    n_shots = [p.n_snapshots for p in sweep]
    rmse = [p.rmse for p in sweep]
    lo = [p.rmse_ci_lo for p in sweep]
    hi = [p.rmse_ci_hi for p in sweep]
    ax.plot(n_shots, rmse, "o-")
    ax.fill_between(n_shots, lo, hi, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shadow snapshots")
    ax.set_ylabel("RMSE(shadow, exact)")
    ax.set_title(title)
    fig.tight_layout()
    interp = (f"RMSE at {n_shots[0]} shots = {rmse[0]:.3f}; at {n_shots[-1]} shots = {rmse[-1]:.3f}. "
              f"Classical-shadow variance is expected to shrink roughly as 1/sqrt(K) -- a shot budget "
              f"that never reaches exact-readout-competitive error at this system size is the SAME honest "
              f"finding this repo's own notebook 5 self-audit already reported at larger N.")
    return fig, interp
