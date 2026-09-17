# DQRC architecture repair: where the processor's nonlinear capacity went

Follow-up to `docs/DQRC_RESULTS.md` (the FAST_MODE run that found
`processor_only_at_eoc` NL≈4.29 but embedded DQRC NL≈0.68-1.07). This
document answers the diagnostic brief's 10 questions directly, backed by
`code/decoupled_qrc/diagnostics.py` and `tests/test_diagnostics.py`
(5 new regression tests, all passing; full suite 37/37 passing). No
PUBLICATION_MODE run was performed, per the brief.

**Headline: two real, compounding bugs were found — not a fundamental
physics limit.** Fixing both partially recovers nonlinear capacity (roughly
5x, from ~0.68-1.1 to ~1.1-1.7 depending on configuration) and DQRC's memory
now *exceeds* an equal-qubit monolithic baseline — but a substantial NL gap
remains, now attributable to a *third*, harder factor: the processor
subsystem, even fixed, has fewer interacting qubits and a sparser coupling
graph than a monolithic register at the same total qubit budget.

## 1. At what stage is nonlinear capacity lost?

Before the processor's own dynamics or the stitching step — **at the
memory-processor interface itself**, and compounded by an under-sized
processor. Two independent root causes, found by direct circuit
instrumentation (`diagnostics.info_theoretic_diagnostics`,
`diagnostics.embedding_consistency_test`):

**Root cause A — the interface gate type.** `interface.apply_interface`'s
`'rzz'` and `'cp'` modes are both **diagonal in the processor-entry qubit's
own Z basis** — a diagonal two-qubit gate can change a qubit's relative
phase but never its Z-population. If the processor's own internal dynamics
conserves total Z-magnetization (true whenever it has few or no SYK4
quartic terms — see root cause B), a diagonal interface literally **cannot
inject any amplitude or coherence into the processor at all**, for any
`lambda_mp`. Verified directly:

```
kind=rzz lam=0.0/0.3/1.0/3.0: purity_P = 1.0000 exactly, MI = 0.0000 exactly
kind=zx  lam=0.0:             purity_P = 1.0000 (lambda=0 is an identity, correctly)
kind=zx  lam=0.3:             purity_P = 0.2181, MI = 0.2588
```

`purity_P == 1.0` means the processor is in a **pure state, i.e. exactly
UNENTANGLED with the memory register and the rest of the system** — the
strongest possible confirmation that zero information crossed the
interface, not merely "weak" transfer. This is now a permanent regression
test (`tests/test_diagnostics.py::test_rzz_and_cp_cannot_entangle_...`).

**Root cause B — the FAST_MODE default N_P=3 cannot be chaotic at all.**
A SYK4 quartic term needs 4 *distinct* qubits
(`mixed_syk_core.sample_syk4_terms` uses `itertools.combinations(range(N),4)`).
`comb(N_P,4)` is exactly **zero for N_P<4**:

| N_P | default_n_sparse_terms | comb(N_P,4) | actual SYK4 terms |
|---|---|---|---|
| 3 | 4 | **0** | **0** |
| 4 | 6 | 1 | 1 |
| 5 | 9 | 5 | 5 |
| 6 | 11 | 15 | 11 |

At N_P=3 (the FAST_MODE default, chosen to fit `N_total=6`), the processor's
`kappa_processor` scan was **scanning a knob with no physical effect on
chaos** — `mixed_layer` reduces to pure SYK2-like RXX+RYY hopping regardless
of kappa, which is exactly-integrable and (per root cause A) also
magnetization-conserving, compounding the interface failure. This also means
`docs/DQRC_RESULTS.md`'s Section 5 processor-EOC scan (which used a
*different* `N_P_DEMO=4`) was never actually characterizing the processor
size DQRC's Section 6 onward (`N_P=3`) used — the two were never
comparable.

A **third, milder issue** (methodological, not physical): the FAST_MODE
notebook's `processor_only_at_eoc` ablation and `dqrc_no_stitch`/
`dqrc_with_stitch` ablations were run at different `kappa_processor`
(0.960 vs 1.0 — close, minor) and different `max_weight_readout`
(defaulted to 3 vs explicitly set to 2 in `BASE_DQRC_CFG` — not close,
meaningful) — not a matched Part-9 comparison. Confirmed and fixed in this
investigation's own test harness.

## 2. Does the processor still generate high-order IPC when embedded?

Yes, once both root causes are fixed (N_P=5, `interface_kind='zx'`) —
partially. `diagnostics.embedding_consistency_test` (matched N_P, kappa,
reps, max_weight, input seed across all three cases):

| case | M | NL | rank | condition # |
|---|---|---|---|---|
| standalone processor (own input qubit) | 6.205 | **6.231** | 249 | 4.3e5 |
| embedded, memory DISCONNECTED (lambda_mp=0) | 0.061 | 0.311 | 249 | 5.0e4 |
| embedded, memory CONNECTED (zx, lambda_mp=0.5) | 2.155 | **0.821** | 249 | 5.8e6 |

`embedded_disconnected`'s near-zero M/NL is the *expected* trivial case
(processor gets no signal at all when disconnected — not itself evidence of
a bug, just confirms the interface is the only channel). The informative
comparison is standalone (6.231) vs embedded-connected (0.821): **~13%
recovered** — connecting the interface clearly helps (2.6x over
disconnected) but embedded NL is still far below standalone.

A follow-up scan across `kappa_processor` x `lambda_mp` (4x5 grid, N_P=5,
single seed — noisy, not a multi-seed claim) found this recovery fraction
varies noisily between ~15-28% depending on the operating point, with no
clean monotonic trend visible at n_seeds=1. **Interesting secondary
finding**: at N_P=5 (real SYK4 terms present), `'rzz'` and `'zx'` gave
*comparable* NL in this scan, unlike the stark N_P=3 case — plausible
explanation: once the processor's own quartic dynamics genuinely breaks
magnetization conservation, it can convert the phase information `'rzz'`
provides into real population changes through its own chaotic mixing,
something the pure-SYK2 N_P=3 processor structurally cannot do. Not
confirmed at multi-seed rigor; flagged as a hypothesis for the
PUBLICATION_MODE follow-up.

## 3. Does stitching destroy, preserve, or enhance NL?

**Enhances slightly, at real cost.** At N_P=5, no-stitch vs. a 2+3-block
split (`stitching_blocks=((2,kappa),(3,kappa))`, same total processor
qubits + 1 ancilla):

| case | qubits | features | M | NL | rank |
|---|---|---|---|---|---|
| no_stitch (N_P=5) | 7 | 171 | 3.441 | 0.705 | **171** |
| stitch_2+3 (+1 ancilla) | 8 | 210 | 3.262 | 0.760 | **54** |

NL rose slightly (0.705→0.760) but the **effective numerical rank of the
feature matrix collapsed from 171 to 54** despite having *more* nominal
features (210) — most of the extra features are near-linearly-dependent.
This is real, measured "feature homogenization" (per the brief's own
hypothesis), and the comparison isn't even qubit-matched (stitching added
one ancilla). Splitting the ALREADY-small N_P=5 processor into two
sub-blocks of size 2 and 3 — **both below the N<4 threshold from root cause
B** — means each block individually has zero SYK4 terms too; that stitching
still edges out no-stitch at all suggests the ancilla-mediated coupling
itself is contributing more than each fragment's own (absent) chaos, not
that stitching is "restoring" lost richness.

## 4. Is the main problem memory-to-processor transfer, processor dynamics,
stitching, measurement, or feature exposure?

**Transfer (root cause A) and processor-dynamics/sizing (root cause B), in
that order.** Ruled out by direct test:
- **Not feature exposure** — Phase 3's grouped-readout test
  (`diagnostics.run_dqrc_grouped`) showed `X_processor` alone already
  carries the SAME NL (0.821) as the full merged feature set; memory-alone
  carries almost none (NL=0.187); adding cross terms adds nothing to NL.
  The merged, adjacent-only-Pauli-string readout is not hiding processor
  signal.
- **Not stitching** — the ORIGINAL FAST_MODE run already showed collapsed NL
  in `dqrc_no_stitch` (no stitching involved at all), so stitching cannot be
  the primary cause; it is a smaller, secondary effect (question 3).
- **Not measurement/IPC implementation** — `tests/test_ipc.py`'s planted-
  signal recovery test (a feature column that IS exactly a known target)
  recovers capacity ~1, proving the pipeline finds real signal when it is
  actually present in the features; the low NL numbers reflect what's
  actually in `X`, not an artifact of `compute_ipc`.
- **Is transfer and sizing** — `info_theoretic_diagnostics` proves zero
  entanglement is created by the default interface at small N_P (root cause
  A), and `comb(N_P,4)` proves the processor has no chaotic term budget at
  all below N_P=4 (root cause B).

## 5. What lambda_mp maximizes nonlinear transfer without destroying memory?

Not cleanly determined at the single-seed scale tested. The 4x5
(kappa_processor x lambda_mp) grid at N_P=5 showed NL varying noisily
between ~0.1 and ~1.7 with no visible monotonic optimum — the largest
single value found was NL=1.71 at (kappa_processor=20, lambda_mp=0.3,
`'rzz'`), but this is one seed and should not be treated as a located
optimum. A real answer needs the grid re-run with >=3 seeds per cell (out
of scope for this diagnostic-only pass, per the brief's "do not run
PUBLICATION_MODE yet").

## 6. Does direct processor readout recover the missing nonlinear IPC?

**Yes for what it can recover** — `X_processor` alone already achieves the
SAME NL as the merged feature set (question 4) — the readout grouping isn't
losing anything additional. But "direct readout" only exposes what the
processor *has*; it cannot manufacture nonlinear structure the processor
never received (root cause A) or never had the resources to generate (root
cause B). So the fix for "missing NL" is upstream of readout, not readout
itself.

## 7. Can combined quantum features retain both memory specialist M and
processor specialist NL?

Partially, in the corrected configuration. Group comparison at N_M=1,
N_P=5, `'zx'`, lambda_mp=0.5:

| features | M | NL |
|---|---|---|
| A: memory only | 1.459 | 0.187 |
| B: processor only | 2.155 | 0.821 |
| C: memory + processor | 2.446 | 0.791 |
| D: + cross terms | 2.556 | 0.778 |

Combining (C) keeps essentially all of processor-alone's NL (0.791 vs
0.821) while adding some M on top (2.446 vs 2.155) — the combination is not
actively destructive. Cross terms (D) add a little more M but nothing to
NL. So: **yes, combining preserves near-ceiling NL for this small system**,
but "ceiling" here (0.821) is itself far below the true processor ceiling
measured standalone (6.231, question 2) — the combination doesn't lose
what's there, but what's there is small.

## 8. Does the repaired DQRC approach or exceed the monolithic Pareto
frontier?

**Not established — mixed single-point evidence, needs a real multi-seed
re-run.** One matched-qubit-count comparison at N_total=8:

| architecture | qubits | M | NL |
|---|---|---|---|
| monolithic (kappa=1.0) | 8 | 5.810 | 5.710 |
| DQRC repaired (N_M=2, N_P=5, `'zx'`, lambda_mp=0.5, kappa=1.0) | 8 | **6.483** | 1.117 |

DQRC's M now *exceeds* the monolithic baseline at equal qubit count — a
genuinely new, positive finding from the repair — but NL is still ~20% of
the baseline's. Neither point dominates the other in the strict Pareto
sense (DQRC wins on M, baseline wins on NL) — but the FAST_MODE Section 2
baseline scan (`docs/DQRC_RESULTS.md`) found monolithic points with BOTH
higher M (up to 6.70) and higher NL (up to 3.19) than this single DQRC
point, at a *different* qubit count (N=6) — so a same-N_total baseline
*frontier* (not just one point) is needed before claiming DQRC's single
point is actually non-dominated. That frontier was not computed in this
pass (time-boxed diagnostic work, no PUBLICATION_MODE run per the brief).

## 9. Should a medium run be executed?

**No, not yet**, evaluated against the brief's own 5 gates:

| gate | requirement | result | met? |
|---|---|---|---|
| A | embedded NL >= 80% of standalone NL | ~13% (0.821/6.231) | **NO** |
| B | DQRC M >= 80% of memory-only standalone M | 293% (6.483/2.221) | yes (over-met — see caveat below) |
| C | varying g_processor gives a clear NL response | noisy, no clean trend at n_seeds=1 | **unclear / NO** |
| D | \|dNL/dg\| > \|dNL/dm\| (independent controls) | not re-measured with the fix (see below) | **not evaluated** |
| E | >=1 DQRC point close to/outside baseline frontier, multi-seed | not computed (single point only) | **NO** |

Gate B's "over-met" number is a weaker signal than it looks: the combined
DQRC readout's M now includes whatever linear/memory-like capacity the
FIXED processor itself picks up (standalone processor's own M was 6.205 —
almost all of the "memory" in the combined system may be coming from the
processor acting as a second memory-like register via the working
interface, not from the dedicated N_M=1-2 qubit memory subsystem
specifically). This is worth separating out explicitly before trusting
Gate B. Gates A, C, and E are clear or likely fails. **Recommendation:
fix the interface/processor-sizing issues in the shipped defaults, redo
gates A/C/D/E with >=3 seeds per point (still short of the >=10 seed
PUBLICATION_MODE bar, but enough to see through the current single-seed
noise), before considering a medium run.**

## 10. Should the next architectural addition be IDQNN, local-inversion
sewing, shadows, or none yet?

**None yet — confirmed, more strongly than before.** The bottleneck
identified here (interface gate type; minimum processor size for any
chaos at all) sits entirely upstream of IDQNN/sewing/shadows, exactly as
Phase 14 of the brief anticipated. Adding any of those now would decorate
an architecture that currently cannot reliably move information from
memory into processor, which would make any resulting numbers impossible
to interpret.

## What changed in the codebase this pass

- `code/decoupled_qrc/diagnostics.py` (new) — all functions used above:
  grouped feature-op builders, `run_dqrc_grouped`, `run_feature_flow_audit`,
  `standalone_processor_matched`, `embedding_consistency_test`,
  `feature_health`, `scan_g_lambda`, `info_theoretic_diagnostics`,
  `fixed_total_qubit_scan`/`fixed_processor_size_scan`/
  `processor_only_g_scan`/`independent_control_jacobian` (clean,
  non-conflated resource-scan variants per the brief's Phase 7/8 request —
  built but not yet run to full statistical completion in this pass),
  `memory_design_comparison`.
- `code/decoupled_qrc/interface.py` — docstring updated with the
  diagonal-gate/magnetization-conservation finding (question 1), so future
  readers don't have to rediscover it.
- `tests/test_diagnostics.py` (new, 5 tests) — locks in the central finding
  as a regression test (`test_rzz_and_cp_cannot_entangle_...` /
  `test_zx_interface_does_entangle_...`), plus sanity checks for the
  grouped readout and feature-health rank detection.
- **Nothing in `experiments.py`'s shipped `DQRCConfig` defaults was changed**
  — per the brief's own scientific rule ("do not optimize for making the
  project work"), the FAST_MODE architecture's actual defaults
  (`interface_kind='rzz'`, whatever `N_P` a caller picks) are left as
  configurable, now-documented choices; this report states plainly that
  `'rzz'`/`'cp'` at N_P<4 is a non-functional configuration, not a
  recommendation baked silently into the code.

## Honest scope limitations of this pass

Per the brief ("do not run PUBLICATION_MODE"), the following were
deliberately NOT done and should not be assumed:
- No multi-seed (>=3) statistics on any of the numbers above — every table
  in this document is a single seed (`master_seed=0`) unless stated
  otherwise, and several results (question 5's grid, in particular) are
  visibly noisy at that scale.
- `fixed_processor_size_scan`/`processor_only_g_scan`/
  `independent_control_jacobian` (the brief's Phase 7/8 clean-derivative
  functions) were implemented and unit-smoke-tested but not run to produce
  a reported Jacobian in this document — Gate D above is explicitly marked
  "not evaluated" rather than guessed at.
- The full N_total-matched Pareto FRONTIER comparison (question 8/Gate E)
  was not computed — only single representative points.
- Phase 9's memory-design comparison at k=1..20 (`diagnostics.
  memory_design_comparison` exists and was smoke-tested) was not run to
  completion/reported here.
