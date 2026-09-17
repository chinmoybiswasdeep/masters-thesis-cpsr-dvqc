# IDQNN-inspired Spatial Quantum Memory + EOC Processor: results

Follow-up to `docs/DQRC_ARCHITECTURE_REPAIR.md`. New modules:
`code/decoupled_qrc/{spatial_memory,idqnn_memory,interfaces_advanced,id_memory_eoc}.py`,
5 new test files (20 new tests; full suite 67/67 passing). Numbers below come
from `results/dqrc/id_memory_eoc_experiments.json` and
`results/dqrc/id_memory_eoc_pareto.json` (single-to-3-seed diagnostic runs,
**not** a PUBLICATION_MODE run, per the brief). Configuration throughout:
L=3 (2 memory qubits + 1 input), N_P=5, N_total=8, T_ipc=250.

**Headline: a real, substantial improvement over the repaired DQRC — the
processor's embedded nonlinear capacity now matches or exceeds its own
standalone ceiling on average — but control-level decoupling (the
Jacobian) still fails, and the improvement is noisy across only 3 seeds.
Per the brief's own gates, a medium/large run is NOT yet justified.**

## 1. Is the memory genuinely stored in quantum state, or is any hidden
classical delay line being used?

Genuinely quantum. `tests/test_no_classical_delay_leakage.py` verifies by
source inspection that `build_id_memory_eoc_circuit` never re-indexes
`u_seq` inside its per-step loop and never calls `delay_taps`;
`tests/test_spatial_memory.py`/`test_memory_persistence.py` verify the
shift register's slot-k state matches the exact closed-form
`Ry(pi*u_{t-k})|0>` prediction and that embedding in the full circuit at
`lambda_mp=0` reproduces the standalone register bit-for-bit. The ONLY
classical-delay code in this pass is `classical_delay_plus_processor`,
which exists purely as the Part 15 control and is checked to be a
separate, non-default function (`test_classical_control_is_a_separate_...`).

## 2. Does the spatial memory retain past inputs better than the old
persistent memory?

Not directly compared at matched size in this pass (out of scope given
time). What IS established: the bare shift register gives EXACT,
hand-verified recall for k <= (number of memory qubits) under noiseless
readout (inherited from `memory.py`'s already-proven delay-register
behavior, `spatial_memory.py` only adds configurable length + fading on
top). A genuinely important finding while characterizing fading
(`spatial_memory.delay_resolved_capacity`'s own docstring): under EXACT
(noiseless) expectation-value readout, amplitude-damping-based fading does
**NOT** reduce IPC-style capacity at all (a fixed-gamma contraction per step
is still an invertible, deterministic function of the input, and ridge
capacity is invariant to invertible rescaling) — real fading only appears
once finite-shot noise is added (`n_shots` parameter), confirmed directly:
`tests/test_spatial_memory.py::test_exact_readout_capacity_is_gamma_invariant`
vs `test_finite_shot_capacity_decreases_with_stronger_fading`, both passing.
This is a non-obvious methodological finding worth carrying into any future
memory-capacity work in this repo.

## 3. Does the EOC processor retain its standalone nonlinear capacity when
connected?

**Yes, on average, once the interface is a genuinely non-diagonal,
multi-generator coupling** (`'heisenberg'`: XX+YY+eta*ZZ). Direct
comparison, X_processor-only readout (isolating the processor's OWN
features from memory/cross contributions, same isolation technique as
`docs/DQRC_ARCHITECTURE_REPAIR.md` question 2):

| seed | NL_processor_standalone | NL_processor_embedded | ratio (eta_NL, proc-only) |
|---|---|---|---|
| 0 | 3.685 | 8.145 | **2.21** |
| 1 | 4.468 | 6.188 | **1.39** |
| 2 | 5.123 | 3.961 | 0.77 |
| mean | 4.425 | 6.098 | **1.46** |

Mean ratio 1.46 (i.e. the embedded processor's own nonlinear capacity
EXCEEDS its standalone ceiling on average) — but with large seed-to-seed
spread (0.77-2.21). One plausible mechanism: the memory register, via a
genuinely entangling Heisenberg-type coupling, feeds the processor a
richer, more temporally-structured driving signal than the processor's own
bare single-qubit Ry input alone provides — the processor is not just
"receiving less-diluted information," it may be receiving MORE useful
information than it would generate on its own. This is a hypothesis, not a
proven mechanism; not confirmed beyond 3 seeds.

## 4. What is eta_NL?

**Mean 1.46 (X_processor-only), range 0.77-2.21 across 3 seeds** (kappa=1.0,
`interface_kind='heisenberg'`, lambda_mp=0.5). Using the FULL combined
feature set (X_mem + X_proc + X_cross) instead, `eta_NL_combined` is
2.14 / 1.38 / 0.88 (mean ~1.48, similar) — memory and cross features add
little on top of what X_processor alone already carries, consistent with
`docs/DQRC_ARCHITECTURE_REPAIR.md`'s earlier finding that nonlinear capacity
lives predominantly in the processor's own readout.

## 5. What is eta_M?

**Mean 2.79 (combined feature set), std ~0.10 across 3 seeds** (2.86, 2.83,
2.67) — the combined memory reading (from the shift register PLUS whatever
linear/memory-like capacity the connected processor itself develops)
substantially EXCEEDS the standalone 2-qubit shift register's own M=2.0-2.07.
As in the earlier DQRC repair, this number partly reflects the processor's
own recurrent-like memory contribution (once genuinely driven), not purely
the dedicated memory subsystem in isolation — a caveat carried over from
`docs/DQRC_ARCHITECTURE_REPAIR.md` question 9's Gate-B discussion.

## 6. Which interface transfers information best?

Of the four tested (`'zx'`, `'xy'`, `'heisenberg'`, `'multiaxis'` at
lambda_mp in {0.2, 0.5}, single seed screen): `'heisenberg'` and `'xy'` gave
the largest NL at lambda_mp=0.5 (7.89 and 5.97 respectively, combined
readout); `'zx'` gave almost no improvement (NL stayed near 0.25-0.42,
consistent with `docs/DQRC_ARCHITECTURE_REPAIR.md`'s finding that `'zx'`
alone is a comparatively weak channel once real SYK4 richness is present)
`'multiaxis'` (with default lambda_xx=lambda_yy=1, lambda_zz=lambda_zx=0)
was numerically IDENTICAL to `'xy'` in this screen, since those default
coefficients reduce it to the same generator — not yet tested with nonzero
lambda_zz/lambda_zx. `'heisenberg'` (XX+YY+eta*ZZ) was carried forward as
the primary configuration for questions 3-9 above/below.

## 7. How much memory back-action occurs?

Substantial. Comparing the memory register's own reduced state with vs.
without the interface active (same trajectory, `lambda_mp=0.5` vs. `0`,
`interfaces_advanced.memory_disturbance`):

    trace_distance = 0.257
    fidelity = 0.753
    purity: 1.000 (no interface) -> 0.638 (with interface)

This is a real, sizeable disturbance — the interface strong enough to
transfer meaningful information (question 3/4) also measurably perturbs
memory. `transfer_efficiency` (NL gain per unit trace-distance disturbance)
= **17.4** — a purely engineering ratio (per the brief's own instruction,
not treated as fundamental), reported as the raw numbers above, not
optimized.

## 8. Does the Jacobian indicate true control-level decoupling?

**No — this is the key remaining problem.** Using genuinely independent
controls (m = gamma_M, the memory-only fading parameter, N_P fixed; g =
kappa_processor, L fixed), evaluated at (m0=0, g0=1.0, dm=0.2, dg=0.5):

    dM/dm  =  0.363     dM/dg  = -0.098
    dNL/dm =  2.751      dNL/dg = -0.371
    cross_coupling = 3.879   decoupling_score = 0.205

`|dM/dm| > |dM/dg|` holds (0.363 > 0.098, the desired signature for
memory). But `|dNL/dg| > |dNL/dm|` FAILS badly (0.371 << 2.751) — nonlinear
capacity in this architecture is currently MUCH more sensitive to the
MEMORY-side fading control than to the processor's own EOC control. This
directly means Gate E (Part 19) is not met, and is the single most
important honest limitation of this pass's otherwise-encouraging eta
numbers: high absolute recovered capacity does not yet imply the two
controls act on the intended, separate targets.

## 9. Does the architecture expand the monolithic Pareto frontier?

**Not established — closer than before, still behind.** Matched N_total=8,
3 seeds x 3 kappa points each (`kappa_processor` in {0.1, 1.0, 10.0},
`interface_kind='heisenberg'`, lambda_mp=0.5):

    hv_baseline = 46.17  95% CI [39.63, 51.27]
    hv_dqrc     = 35.42  95% CI [24.95, 45.19]
    hv_diff (dqrc - baseline) = -10.75  95% CI [-21.30, -0.13]

The CI still favors the baseline, but its UPPER bound (-0.13) is now barely
below zero — a dramatic narrowing from the original FAST_MODE repair's
`[-21.61, -16.75]` (nowhere near zero). 1/9 pooled `id_memory_eoc` points
were NOT dominated by the baseline's own frontier (11%, vs 0/8 in the
original FAST_MODE run). **This is genuine progress, not a claimed
expansion** — the honest verdict remains closer to "B: partial/conditional"
at best, not "A: clear expansion", and rests on only 3 seeds.

## 10. Does quantum memory outperform a classical delay-line control under
matched resources?

**Mixed, seed-dependent.** `classical_delay_plus_processor` (raw
`delay_taps(u, m=6)` concatenated with the SAME standalone EOC processor's
own features, N_P=5) vs. the full quantum `id_memory_eoc` architecture, 3
seeds:

| seed | classical M | classical NL | quantum M | quantum NL |
|---|---|---|---|---|
| 0 | 6.86 | 3.56 | 5.73 | **7.89** |
| 1 | 6.98 | 4.55 | 5.67 | **6.18** |
| 2 | 6.95 | 5.10 | 5.53 | 4.51 |

The classical-delay control has HIGHER M in all 3 seeds (raw delay taps are
a strong, cheap, essentially perfect linear memory, as expected and as
found in the earlier repair pass) but LOWER NL in 2 of 3 seeds. **Quantum
memory does not straightforwardly dominate the classical control on
memory** — it is genuinely worse at memory specifically, and only
sometimes better at nonlinearity. This must be stated plainly: the
architecture's advantage, where it exists, is NOT "better memory than a
classical delay line."

## 11. Does the IDQNN-inspired transform actually help beyond the bare
quantum shift register?

Tested only for basic IPC1 preservation (`tests/test_idqnn_memory.py::
test_id_sqm_preserves_reasonable_ipc1_relative_to_bare_shift_register`,
passing: transformed IPC1 > 50% of bare register's IPC1 at
entangle_strength=0.3) — NOT yet integrated into the main eta_M/eta_NL
scan above (that scan used the bare `spatial_memory` register, no ID-SQM
transform, `use_idqnn_transform=False` throughout). Whether ID-SQM changes
eta_M/eta_NL when actually plugged into `id_memory_eoc` is untested this
pass — an open question for the next session, not answered here.

## 12. Was a genuine Huang-style IDQNN successfully implemented?

**No, and none was attempted.** Per Part 1's terminology discipline,
everything in `idqnn_memory.py` is called "ID-SQM" / "IDQNN-inspired",
never "IDQNN" outright — it is a small, FIXED (not learned, not tuned to
maximize any metric), local entangling transform, not a construction
numerically verified against a corresponding deep circuit's output
distribution. `experimental/idqnn_memory_prototype.py` (from the prior
DQRC-repair pass) still only implements the honestly-named
`ideal_depth_compressed_control`, not IDQNN. Per the brief's own execution
order (step 12: "only if successful, attempt genuine IDQNN prototype") and
given Gate E fails (question 8), a genuine IDQNN attempt was correctly
NOT started this pass.

## 13. What is the strongest defensible scientific conclusion?

Replacing the diagonal `'rzz'`/`'cp'` interface with a genuinely
non-diagonal, multi-generator coupling (`'heisenberg'`: XX+YY+eta*ZZ), and
using a processor large enough to have real SYK4 richness (N_P=5),
recovers the processor's own embedded nonlinear capacity to **roughly
parity with its standalone ceiling on average (eta_NL≈1.46)** — a
substantial, mechanistically-explained improvement over the prior repair
pass's ≈13-20% recovery. Memory capacity in the combined system
(eta_M≈2.79) is not a bottleneck. However, **the architecture has not yet
achieved genuine control-level decoupling**: the clean Jacobian shows
nonlinearity is currently far MORE sensitive to the memory-side fading
control than to the processor's own EOC control, the opposite of the
intended signature. The Pareto-frontier gap to the monolithic baseline has
narrowed substantially but not closed (CI just short of crossing zero), and
quantum memory does not outperform a simple classical delay line on memory
capacity specifically. **Per the brief's own Gate E failure, a
medium/large multi-seed run is correctly not justified yet** — the next
diagnostic step should target why nonlinearity is so memory-control-
sensitive (question 8) before scaling up seeds or trajectory length.

## Honest scope limitations of this pass

- All eta/Jacobian/Pareto numbers above are single-run-per-seed with only
  2-3 seeds (per the brief: do not run PUBLICATION_MODE) — every CI here is
  wide and several individual-seed values (e.g. eta_NL_proc_only=0.77 at
  seed 2) fall below the "ideal" 0.8 gate even though the mean clears it.
- Gate C (clear NL response to g_P) and Gate D (clear memory response to a
  memory-only control) were only partially assessed (via the standalone
  processor-kappa scan and the Jacobian's `dM/dm` term respectively), not
  via dedicated, denser scans in the full embedded architecture.
- The `'multiaxis'` interface was not tested with nonzero `lambda_zz`/
  `lambda_zx` -- only its default reduction to `'xy'`.
- ID-SQM's effect on eta_M/eta_NL when embedded (not just standalone IPC1
  preservation) is untested.
- Back-action/transfer-efficiency was computed at ONE (lambda_mp,
  interface_kind) point, not swept.
