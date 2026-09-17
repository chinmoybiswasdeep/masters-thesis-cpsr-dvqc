# DQRC results (FAST_MODE)

Generated from `code/DQRC_Memory_Nonlinearity_Decoupling.ipynb`'s actual
executed FAST_MODE run (`results/dqrc/fast_mode_results.json`,
`results/dqrc/*.png`, `results/dqrc/*.csv`, `results/dqrc/config.json`),
0 error cells, `nbconvert --execute` verified. See `docs/DQRC_THEORY.md` for
the equations and `docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md` for what the
existing codebase already established.

**Scale of this run**: N_total = 6 physical qubits (fixed across every
architecture, Part 9), T=250 trajectory length, IPC max_delay=6/max_degree=6
with 12 targets/degree and 5 surrogates, n_seeds=2, ~300 bootstrap
resamples. This is deliberately small — see "Caveats" below and the
PUBLICATION_MODE plan for what changes at full scale.

## Direct answers (Part 20)

**1. Does the existing finalized monolithic reservoir reproduce a
memory-NL trade-off?**
Partially, and weakly, at this scale. Section 2's kappa scan
(`results/dqrc/section2_6_kappa_scans.csv`) gives M rising monotonically with
kappa (4.61 -> 4.66 -> 5.55 -> 6.70) while NL is non-monotonic (2.64 -> 3.19
-> 3.11 -> 2.47) — NL peaks in the middle of the scan and falls at both ends,
which is *consistent* with a trade-off at the SYK2-like end (kappa=20: M
highest, NL lowest) but not a clean monotonic anti-correlation across the
whole range. A single-seed, 4-point scan cannot establish this rigorously;
PUBLICATION_MODE's denser, multi-seed scan is needed before treating this as
confirmed either way.

**2. Does putting the monolithic reservoir at EOC remove the trade-off?**
No evidence either way from this run — `monolithic_qrc` (kappa=1.0) and
`monolithic_qrc_at_eoc` (kappa=0.960, the repo's stored point) give nearly
identical M (5.346 vs 5.344) and NL (5.246 vs 5.410) in the Section 11
ablation table, because 0.960 and 1.0 are very close on this kappa scale.
This comparison does not actually stress-test the EOC point distinctly from
a nearby non-EOC point; a wider kappa separation is needed in
PUBLICATION_MODE to answer this properly.

**3. Does quantum-memory/EOC-processor separation reduce cross-coupling?**
The Section 8 Jacobian at (m=lambda_im=0.3, g=kappa_processor=20, DQRC's
found EOC point) gives:

    dM/dm=10.07   dM/dg=0.10
    dNL/dm=10.73  dNL/dg=0.04
    cross_coupling=1.071, decoupling_score=0.483

`dM/dg` and `dNL/dg` are indeed much smaller than `dM/dm` and `dNL/dm` —
but `dNL/dm` is LARGE (10.73), comparable to `dM/dm` (10.07), which is
exactly what the cross_coupling formula (`|dM/dg|+|dNL/dm|` in the
numerator) penalizes: the processor's chaos knob is nicely decoupled from
memory, but the memory-coupling knob `lambda_im` strongly affects
nonlinearity too. cross_coupling=1.07 (not small) says this specific
2-parameter Jacobian is NOT strongly decoupled at this operating point —
an honest, unforced finding at a single evaluation point (3 `MN_func` calls,
one seed); not yet a general claim.

**4. Does DQRC generate points outside the baseline M-vs-NL Pareto frontier?**
**No**, at this FAST_MODE scale. Section 9's bootstrapped hypervolume
comparison (n_seeds=2, 300 bootstrap resamples):

    hv_baseline = 22.74  [20.47, 25.01]
    hv_dqrc     =  3.56  [ 3.41,  3.72]
    hv_diff (dqrc - baseline) = -19.18  [-21.61, -16.75]

The CI excludes zero **in the baseline's favor**. 0/8 pooled DQRC points lie
outside the baseline's own frontier. **Verdict: C — no convincing frontier
expansion** (the notebook's own printed verdict, generated mechanically from
the CI, not hand-picked). This is reported plainly rather than searching for
a more favorable configuration to report instead (Part 16 item 10).

**5. Is the result [(4), a lack of expansion) still present with matched
qubit count / matched feature count / finite shots / multiple seeds?**
- Matched qubit count: yes, already enforced — every architecture in
  Sections 2-13 uses N_total=6 (`ResourceUsage.n_qubits_physical`, verified
  by `tests/test_resource_accounting.py`).
- Matched feature count: yes, the gap persists. Section 13's equal-feature
  (51 features each) comparison at one representative kappa gives
  monolithic M=6.52/NL=2.88 vs DQRC M=4.41/NL=1.07 — DQRC still trails on
  both axes with the SAME feature budget, so the Section 9 finding is not an
  artifact of the baseline simply exposing more classical readout features.
- Finite shots: Section 14 only checked shadow-readout FIDELITY (RMSE=0.063
  at 2000 snapshots vs. exact, on one representative state) — it did **not**
  re-run the full IPC/task comparison under shadow features end-to-end (that
  would need per-timestep shadows across the whole T=250 trajectory, out of
  the FAST_MODE time budget — see question 9 below for the same honest
  scoping call applied to IDQNN).
- Multiple seeds: n_seeds=2 only (FAST_MODE). This is the single biggest
  caveat on the "no expansion" finding — 2 seeds bounds the CI but does not
  give it real statistical power. PUBLICATION_MODE's n_seeds=10 is needed
  before this verdict should be treated as robust rather than preliminary.

**6. How much improvement comes from memory separation / EOC / stitching /
shadows?**
None of these showed improvement in this run — the whole DQRC family
underperformed the monolithic baseline (Section 11):

| ablation | M | NL | qubits |
|---|---|---|---|
| monolithic_qrc | 5.346 +- 0.260 | 5.246 +- 0.599 | 6 |
| monolithic_qrc_at_eoc | 5.344 +- 0.258 | 5.410 +- 0.719 | 6 |
| memory_only | 3.030 +- 0.123 | 0.389 +- 0.025 | 6 |
| processor_only_at_eoc | 5.310 +- 0.415 | 4.288 +- 0.445 | 5 |
| dqrc_no_stitch | 3.974 +- 0.436 | 0.677 +- 0.389 | 6 |
| dqrc_with_stitch | 4.415 +- 0.038 | 1.073 +- 0.246 | 6 (+1 ancilla) |
| dqrc_randomized_processor | 4.207 +- 0.012 | 0.752 +- 0.061 | 6 |
| classical_delay_control | 7.000 +- 0.000 | 0.569 +- 0.029 | 0 |

Two things stand out. First, `processor_only_at_eoc` (5 qubits, no separate
memory register at all) nearly MATCHES the 6-qubit monolithic baseline's M
and gets most of its NL — a single, undivided processor register is doing
almost all the useful work, with one fewer qubit than the full DQRC split.
Second, stitching **does** help within the DQRC family (NL: 0.677 -> 1.073,
+~58%) — consistent with its purpose of restoring cross-block coupling — but
not nearly enough to close the gap to the monolithic baseline.
`dqrc_randomized_processor` (kappa~1e-6, deliberately far from any EOC) is
barely different from `dqrc_no_stitch`, suggesting the bottleneck at this
scale is architectural (small, weakly-coupled subsystems) rather than being
specifically about whether the processor sits at its own EOC point.
The `classical_delay_control`'s M=7.0 (exactly the maximum possible — 7
degree-1 targets at delays 0..6, each perfectly reconstructed from a raw
delay tap) is a useful sanity check that the IPC pipeline itself is correct,
not a claim about DQRC.

**7. Is a classical delay line equally effective?**
For pure memory, the classical delay line's M=7.0 exceeds every quantum
architecture tested (best quantum M was monolithic's 5.346) — a raw delay
tap perfectly reconstructs a linear target by construction, so this is
expected and not itself informative about quantum advantage. Its NL=0.569 is
higher than `memory_only`'s NL=0.389 but lower than every architecture that
includes a processor. **For nonlinearity, no — the classical delay line
clearly does not substitute for a nonlinear quantum processor** (NL=0.569
vs. monolithic's 5.246 or even DQRC's own 0.677-1.073).

**8. Is there evidence of an intrinsically quantum benefit, or only an
architectural decoupling benefit?**
Neither is demonstrated by this run — DQRC did not outperform the
monolithic baseline on any axis at this scale, so there is no benefit (of
either kind) yet to attribute. The one positive internal signal is that
stitching improves DQRC's own NL relative to no-stitch, which is at least
evidence that ADDING coupling helps within the DQRC family — but this is a
comparison within DQRC variants, not against the monolithic baseline or
against a classical control.

**9. Does the IDQNN experiment provide anything beyond the DQRC
architecture?**
Not evaluated — genuine IDQNN was not implemented in this pass. Per Part 5's
own escape hatch, `experimental/idqnn_memory_prototype.py` implements only
the honestly-named `ideal_depth_compressed_control` (matrix-power-once +
one noise application), explicitly NOT called IDQNN, and documents what a
real IDQNN claim would additionally require. No IDQNN claim is made
anywhere in this project. Likewise, `stitching.genuine_local_inversion_demonstrator`
(Part 4's "if feasible" local-inversion demonstrator) is deliberately left
unimplemented (raises `NotImplementedError` with a description of what it
would need) rather than shipping a superficial stand-in — both were the
first things cut under the FAST_MODE time budget, per the project plan.

**10. What is the strongest defensible conclusion from the data?**
At N_total=6, T=250, with a single (weak, single-tap, RZZ) interface
coupling and n_seeds=2: **DQRC does not expand, and in fact contracts, the
monolithic reservoir's memory-nonlinearity Pareto frontier under matched
qubit and feature-count constraints.** The clearest interpretation from the
ablations is that splitting a small qubit budget into weakly-coupled
subsystems costs more usable Hilbert-space/feature richness than the
architectural separation currently buys back — `processor_only_at_eoc`
alone, with one FEWER qubit than the full DQRC split, comes closer to the
monolithic baseline than any DQRC configuration does. This does not falsify
the underlying hypothesis in general (interface coupling strength, tap
count, and N_M/N_P allocation were none of them tuned, and n_seeds=2 gives
very wide uncertainty) — it is a preliminary, honestly-reported negative
result at small scale that should not be spun as either "DQRC works" or
"the hypothesis is false."

## Caveats (do not treat this run as final)

- **n_seeds=2**: far below the >=10-seed bar Part 13 sets for a real claim.
  Every CI here is correspondingly wide/unreliable.
- **Interface coupling not tuned**: `lambda_mp=0.3`, `n_taps=1` were fixed,
  reasonable-looking defaults, never scanned. A stronger or multi-tap
  interface could plausibly change the Section 9 verdict; that scan was not
  run under the FAST_MODE budget.
- **N_M/N_P split not scanned for the Pareto comparison itself** (only for
  the separate Section 7 memory-resource plot, at fixed EOC processor kappa).
- **Mackey-Glass/Lorenz tasks not run** (Section 12) — the generators exist
  (`tasks.py`) and are unit-exercised, but wiring an external chaotic driving
  sequence into `baseline.monolithic_qrc`/`experiments.run_dqrc` (which
  currently always draw their own `random_input`) is a small, undone
  mechanical step.
- **Shadow readout not run through the full IPC pipeline** — only a
  readout-fidelity check (Section 14), not a shadow-feature-based redo of
  Sections 6-9.

## Next steps (PUBLICATION_MODE)

1. Re-run with `utils.PUBLICATION_MODE = True` (n_seeds=10, denser scans,
   larger max_targets_per_degree/n_surrogates) — expect substantially longer
   wall-clock time; not attempted in this pass.
2. Scan `lambda_mp`, `n_taps`, and the N_M/N_P split as additional axes of
   the Pareto comparison, rather than holding them fixed.
3. Wire Mackey-Glass/Lorenz sequences into the reservoir builders as actual
   driving input (currently only `random_input`-driven tasks run).
4. If a genuine advantage does appear at larger N_total or with a tuned
   interface, re-run the finite-shot/noise-robustness check (Section 14)
   through the FULL IPC pipeline, not just a single-state fidelity check.
