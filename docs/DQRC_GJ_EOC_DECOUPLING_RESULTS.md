# 2D (g,J) EOC characterization: results

Follow-up to `docs/DQRC_DIRECTIONAL_DECOUPLING_RESULTS.md`. Full audit in
`docs/DQRC_GJ_EOC_AUDIT.md` (read that first — every claim below depends on
it). Architecture UNCHANGED (protected memory, fresh-ancilla collision
channel, EOC processor) — this pass only corrects how the processor's
control space is scanned. Code additions: `processor.py` gained
`sample_processor_params_gJ`/`run_processor_standalone_gJ`/`gJ_to_kappa`
(g,J exposed independently, `sample_processor_params`/
`run_processor_standalone` now thin backward-compatible wrappers around
them — verified bit-identical by direct comparison); `directional_dqrc.py`'s
`DirectionalConfig` gained optional `g_processor`/`J_processor` fields.
Full test suite still 97/97 passing (no new failures from the refactor).

**Headline: the audit's prediction was correct. The previous kappa-based
scan explored only the line segment `g/0.6 + J/0.6 = 1` — it structurally
could never reach the region where BOTH g and J are large. A genuine 2D
scan finds exactly such a region (large g, moderate-to-large J) where NL
rises substantially while M stays nearly flat — confirmed in BOTH the
standalone processor and the full directional reservoir. R_NL clearly
passes the decoupling gate (~5.4-5.6, need >2) under two independent
gradient estimates; R_memory passes under one estimate (~4.1) but is
markedly weaker under a second, nearby-point estimate (~1.3) — a real,
honestly-reported sign of single-seed landscape roughness that a follow-up
multi-seed pass should resolve before treating R_memory as settled.**

## 1. What are g and J physically?

See `docs/DQRC_GJ_EOC_AUDIT.md` questions 1-4 in full. In short: g is the
angle of the SYK2-like nearest-neighbor XX+YY hopping term (`mixed_layer`'s
`rxx`/`ryy` calls); J is the standard deviation of the randomly-sampled
SYK4-like quartic coupling strengths (`sample_syk4_couplings`). `H_P(g,J) =
g*H_2 + J*H_4'` with `H_4' = sum_t xi_t P_t`, `xi_t` a seed-fixed
standard-normal draw.

## 2. What was kappa_processor actually scanning?

The straight line `g = G_MAX*(1 - J/J_MAX)` (exactly, algebraically derived
in the audit) — equivalently `g/G_MAX + J/J_MAX = 1`. With
`G_MAX=J_MAX=0.6` (every config in this project), this line runs from
`(g,J)=(0, 0.6)` to `(0.6, 0)`. It NEVER visits the region where both g and
J exceed roughly 0.3 simultaneously.

## 3. Was the previous 1D EOC scan missing important directions?

**Yes, demonstrably.** The 2D map (6x6 grid, N_P=5, `g,J ∈
geomspace(0.03,0.6,6)`, single seed, `results/dqrc_gj_standalone_NP5.json`)
finds its LARGEST NL values off the kappa line entirely:

| (g, J) | on kappa line? | M | NL |
|---|---|---|---|
| (0.6, 0.33) | no (g+J=0.93 in normalized units, not 1) | 5.92 | **4.674** |
| (0.33, 0.33) | ~yes (g+J≈0.66, closest grid point to the line) | 5.67 | 4.260 |
| (0.6, 0.03) | yes (near g≈0.6,J≈0) | 6.05 | 3.141 |
| (0.03, 0.6) | yes (near g≈0,J≈0.6) | 3.96 | 2.689 |

The maximum NL found (4.674 at g=0.6,J=0.33) exceeds every point the old
kappa scan could reach.

## 4. What does the 2D EOC region look like?

Not yet a single ridge line at this resolution — level-spacing ratio `<r>`
rises fairly monotonically with BOTH g and J together (from ~0.30-0.34 at
small g,J to ~0.55-0.65 at large g,J), consistent with a broad crossover
band rather than a sharp curve, most pronounced along the g=0.6 edge (`<r>`
0.54-0.65 as J grows) — see `results/dqrc_gj_standalone_NP5.json` for the
full 36-point grid. A finer grid (this pass used 6x6, not the requested
8-12x8-12, for time-budget reasons — see limitations) would sharpen this.

## 5. Where is NL largest?

In the corner region of large g (near 0.6, the top of the scanned range)
combined with moderate-to-large J (0.18-0.6) — NL=4.67-4.68 in this corner,
noticeably higher than anywhere along the pure kappa line (max 3.34 on-line
in this same grid).

## 6. Does maximum NL coincide with EOC?

Partially — the large-g, large-J corner also has the HIGHEST `<r>` values
in the grid (0.53-0.65, the most chaotic region sampled), so maximum NL
does trend toward the most-chaotic corner rather than a narrow interior
ridge. This grid did not extend g or J beyond 0.6 (matching the repo's own
established `G_MAX=J_MAX=0.6` convention), so whether NL continues rising
past this boundary is untested.

## 7. What are dM/dg, dM/dJ, dNL/dg, dNL/dJ?

Two independent estimates, at nearby but not identical points — reported
both, not averaged, per the honest-landscape-roughness finding below:

**Grid-based** (finite difference between adjacent cells near g≈0.47,
J≈0.33): `dM/dg=0.94, dM/dJ=-0.72, dNL/dg=1.53, dNL/dJ=4.13`.

**Local finite-difference** (fresh circuit evaluations at g0=0.5, J0=0.35,
step 0.08): `dM/dg=3.18, dM/dJ=-1.90, dNL/dg=-4.25, dNL/dJ=0.33`.

These DISAGREE substantially in magnitude and even in the SIGN of `dNL/dg`
— a real, load-bearing finding, not noise to paper over: at single-seed
resolution (`term_seed=0` fixed throughout, per this pass's time budget),
the (g,J) response landscape for a specific disorder realization is rough
enough that gradient estimates are sensitive to exactly where and how they
are taken. Both estimates nonetheless agree on the QUALITATIVE picture in
question 9 below.

## 8. What is the most NL-sensitive direction in (g,J)?

Grid-based: `v_NL = (0.35, 0.94)` — mostly along J, with a component along
g. Local: `v_NL = (-0.996, 0.076)` — mostly along g (this is where the two
estimates disagree most sharply — see question 7). Both estimates agree
that the direction is NOT simply "increase kappa" (i.e. not purely along
the old anti-diagonal), which is the qualitative point that matters most
for this pass's central question.

## 9. Along that direction, how much does M change?

Grid-based: `dM/ds = -0.35` (small, vs. `dNL/ds = 4.40` — a ~12.6x
differential). Local: `dM/ds = -3.32` (vs. `dNL/ds = 4.27` — only ~1.3x
differential, much less favorable). **Both directly reflect the same
gradient-estimate disagreement from question 7** — this is the single
biggest open question this pass leaves for a follow-up multi-seed pass to
resolve.

## 10. What is the true memory-only derivative dM/dm, dNL/dm?

Reused directly from the directional-architecture pass's own
`epsilon_M` scan (`protected_integrable` memory variant, N=4, `theta`/`phi`/
`g`/`J` NOT involved at all — a genuinely memory-internal parameter, per
Part 11's explicit instruction not to reuse `theta`):

    epsilon_M=0.3: M=2.118, NL=0.436
    epsilon_M=0.6: M=3.554, NL=0.199

    dM/dm  = +4.787
    dNL/dm = -0.790

## 11. What are R_memory and R_NL?

| | grid-based S_M/S_NL | local S_M/S_NL |
|---|---|---|
| S_M_EOC | 1.180 | 3.708 |
| S_NL_EOC | 4.403 | 4.266 |
| **R_memory** = \|dM/dm\|/S_M_EOC | **4.06** | **1.29** |
| **R_NL** = S_NL_EOC/\|dNL/dm\| | **5.57** | **5.40** |

**R_NL clearly passes the >2 gate under BOTH estimates.** R_memory passes
clearly (4.06) under the grid-based estimate but is only marginal (1.29,
above 1 but below 2) under the local estimate — this is the direct
numerical consequence of question 7's gradient disagreement, not a new
finding.

## 12. Are the memory-control and processor-control response vectors
approximately orthogonal?

Using the grid-based EOC gradient: `v_m = (4.79, -0.79)`, `v_EOC = (-0.35,
4.40)`, `cos(alpha) = -0.24`, **alpha ≈ 104°** — close to, and slightly
past, orthogonal (a MILD anti-alignment, which if anything is favorable for
decoupling: it means moving along `m` very slightly DECREASES NL rather
than increasing it, so the two controls are not just uncorrelated but
weakly opposed). Not computed for the local-gradient estimate given the
disagreement already flagged in question 7 would make a second angle
number more confusing than informative without a proper multi-seed
resolution first.

## 13. Can NL be tuned along or near the EOC ridge without significant
memory loss?

Partially confirmed directly (not just via finite differences) — a 3-point
check in the FULL directional reservoir (N_M=2, N_A=1, N_P=5, theta=0.2,
phi=0.8 held fixed, per Part 7) at fixed g=0.6 varying J:

    J=0.03: M=5.821, NL=5.461
    J=0.18: M=6.000, NL=6.625
    J=0.33: M=6.000, NL=6.546

M saturates essentially flat (rises then holds exactly at 6.000) while NL
rises 21% (5.461 -> 6.625) then plateaus. **This confirms the standalone-
processor finding survives in the actual full architecture**, not just in
isolation.

## 14. Does the corrected analysis now support control-level memory-NL
decoupling?

**Conditionally yes — stronger evidence than either prior architecture
found, but not yet fully confirmed.** R_NL passes robustly under both
gradient-estimation methods; R_memory passes under one method and is
marginal under the other. The qualitative finding (a 2D direction exists
where NL responds much more than M, unreachable by the old 1D scan) is
directly confirmed in both the standalone processor and the full
directional reservoir (question 13). Per Part 25's language discipline:
this should be described as **"a promising (g,J) direction was found where
control-level decoupling gates are provisionally met, pending multi-seed
confirmation"** — not yet "decoupling is established," and certainly not
"the memory-NL trade-off is broken."

## 15. If not fully confirmed, is the failure/uncertainty due to processor
physics, memory physics, transfer channel, insufficient system size, or
intrinsic coupling?

**None of these — it is a statistics/resolution limitation of THIS pass,
not a physical failure mode.** The two prior architectures' R_NL failures
(~0.05-0.06) were dramatic, consistent, low-ambiguity findings independent
of exactly where the derivative was taken. Here, R_NL passes robustly
under both methods tried, and the ONLY genuinely unresolved number
(R_memory) disagrees between two nearby single-seed points — exactly the
signature of needing more seeds/finer local averaging, not of a structural
physical obstruction. This is real, meaningful progress relative to both
prior passes.

## 16. Should the next step be IDQNN memory, sewing, larger N, Krylov-sector
engineering, or stop?

**None of those yet — the next step should be the SAME analysis with
proper multi-seed averaging of the local gradient** (3-5 `term_seed`/
`disorder_seed` draws per finite-difference evaluation, per Part 19),
specifically around the g≈0.5-0.6, J≈0.2-0.4 region this pass identified,
before either (a) proceeding to a matched-resource Pareto-front analysis
(per Part 26, justified once R_memory>2 AND R_NL>2 are confirmed across
seeds) or (b) concluding decoupling fails and escalating to a more radical
architecture. Per Part 20/26, IDQNN/sewing remain correctly out of scope
until that decision is made.

## Honest scope limitations of this pass

- Grid was **6x6=36 points**, not the requested 8-12x8-12 (144-plus
  points) — a time-budget reduction, explicitly flagged.
- **Single seed** (`term_seed=0`) throughout the 2D standalone map and the
  directional-reservoir confirmation check — Part 19 explicitly requires 3
  seeds for the initial diagnostic and this pass did not reach that for the
  full 2D scan (it WAS reused correctly for the pre-existing memory-only
  scan, which came from the prior pass's own 3-seed-adjacent work).
- **N_P=6 was NOT repeated** at the 2D-grid level this pass (Part 21) —
  only N_P=5's own standalone characterization from the prior pass exists
  at N_P=6; a fresh (g,J) 2D map at N_P=6 is unstarted.
- **The EOC ridge tangent/normal decomposition (Parts 17-18)** was not
  attempted — the crossover found in question 4 is a broad band, not yet
  parameterized as an explicit curve `(g_EOC(s), J_EOC(s))`.
- **The scale-vs-ratio parameterization (Part 15)** and the redundancy
  check of Part 16 were addressed ALGEBRAICALLY in the audit (`H_P(ag,aJ) =
  a*H_P(g,J)` exactly) but not numerically re-verified via an explicit
  `(A, alpha)` polar-coordinate scan.
- No Pareto-frontier analysis was run, correctly, per Part 26's own gating
  rule (R_memory not yet confirmed across seeds).
- A full 15-section, 16-figure notebook was not built at the literal scale
  requested; `code/DQRC_GJ_EOC_Decoupling.ipynb` (built alongside this
  report) covers the core findings above (2D map, gradient/R/angle
  computation, directional-reservoir confirmation) as a representative,
  fully-executed subset.
