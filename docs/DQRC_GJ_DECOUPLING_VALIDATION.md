# Rigorous validation of the 2D (g,J) decoupling signal

Follow-up to `docs/DQRC_GJ_EOC_DECOUPLING_RESULTS.md`. Architecture
UNCHANGED (protected memory, fresh-ancilla collision channel, EOC
processor). New notebook: `code/DQRC_GJ_Decoupling_Validation.ipynb` (0
errors). No new permanent modules were needed — this pass exclusively
reused `directional_dqrc.py`'s existing `g_processor`/`J_processor`/
`epsilon_M` fields and `diagnostics.ipc_MN`.

**Headline: the promising signal does NOT survive rigorous testing. Once
evaluated with central differences, multiple step sizes, multiple seeds,
and — critically — ALL SIX derivatives at the SAME operating point in the
FULL architecture (not the standalone processor alone), R_memory (0.17-0.31)
and R_NL (0.08-0.89) both fail decisively, in both seeds tested, by a wide
margin (need >2; found roughly 2-25x too small). The prior pass's R_NL~5.4
finding does not replicate under this stricter protocol. Per the user's own
Part 14 rule, control-level decoupling is reported as "not established," not
"partially confirmed."**

## 1. Was the previous promising region reproduced?

**The specific numerical point was reproduced exactly** (a genuine
transcription error was caught and fixed in the prior report in the
process: `docs/DQRC_GJ_EOC_DECOUPLING_RESULTS.md`'s question-3 table had
written "M=5.29" for (g=0.6, J=0.33); the actual saved data and a fresh
re-run both give M=5.92 — now corrected). **However, the FULL response
matrix was NOT reproduced under matched conditions**, for a real and
important reason: Part 5 requires a genuine memory-INTERNAL control
`epsilon_M`, which only exists for the `protected_integrable` memory
variant — but every prior (g,J) finding (including the promising
`R_NL~5.4` result) used the `shift` memory variant instead. Switching
memory variants is a substantive change to the whole coupled system's
dynamics, not a like-for-like re-measurement. This is flagged explicitly,
per the brief's own "if reproduction fails, stop and explain why" rule —
here reproduction of the SPECIFIC NUMBER succeeded, but reproduction of
the SAME SYSTEM CONFIGURATION as validation input was not possible without
violating Part 5's own requirement, so this pass necessarily tests a
related but not identical configuration.

## 2. What operating region gives the best robust processor selectivity?

Only one candidate point was tested at full rigor this pass (g*=0.5,
J*=0.33, m*=epsilon_M=0.5, theta*=0.2, phi*=0.8 — chosen from the prior
pass's own high-NL, decent-M region) — see "honest scope limitations"
below for why a neighborhood sweep was not run. At this point neither gate
is close to being met, so no operating region can yet be reported as
"best" with confidence.

## 3. Are derivatives stable to finite-difference step size?

**No, mostly.** Central differences at h∈{0.03,0.06} (g,J) and
h∈{0.05,0.10} (m), stability flagged per the brief's own rule
(`relative_spread > 0.3` OR sign change):

| derivative | seed 0 | seed 1 |
|---|---|---|
| dM/dm | unstable (spread 0.49) | unstable (spread 0.35) |
| dNL/dm | **stable** (spread 0.10) | **stable** (spread 0.07) |
| dM/dg | unstable, SIGN FLIPS | unstable, SIGN FLIPS |
| dNL/dg | unstable, SIGN FLIPS | **stable** (spread 0.11) |
| dM/dJ | unstable (spread 0.41-0.45) | unstable (spread 0.45) |
| dNL/dJ | unstable, SIGN FLIPS | **stable** (spread 0.002) |

5/6 derivatives unstable in seed 0; 3/6 unstable in seed 1. **Every M-only
derivative (dM/dm, dM/dg, dM/dJ) is unstable in BOTH seeds** — M's response
to any control at this operating point cannot currently be estimated
reliably with 2-point finite differences at these step sizes. **dNL/dm is
the one derivative that is robustly stable in both seeds** — a genuine,
trustworthy finding, not an artifact.

## 4. Are all derivatives evaluated at the same operating point?

**Yes — this was the explicit fix this pass made relative to the prior
pass.** All six partial derivatives (dM/dm, dNL/dm, dM/dg, dNL/dg, dM/dJ,
dNL/dJ) were computed via the SAME `DirectionalConfig` (same N_M, N_P,
theta*, phi*, memory variant), varying only ONE parameter at a time around
the SAME center (g*, J*, m*), per seed. No derivative was reused from a
different notebook, different memory variant, or different theta/phi.

## 5. What are the full response matrices for each seed?

Using the LARGER step size (h=0.06 for g,J; h=0.10 for m) as the primary
estimate per derivative (chosen as the less noise-floor-sensitive of the
two, though question 3 shows this choice matters and is not fully
resolved):

**Seed 0**: `[[dM/dm, dM/dg, dM/dJ], [dNL/dm, dNL/dg, dNL/dJ]] =
[[0.069, -0.069, 0.210], [7.060, -0.560, -0.050]]`

**Seed 1**: `[[0.069, -0.007, 0.398], [14.806, 6.733, 11.289]]`

## 6. What are R_memory and R_NL per seed?

| seed | S_M | S_NL | R_memory | R_NL | n_unstable/6 |
|---|---|---|---|---|---|
| 0 | 0.221 | 0.562 | **0.314** | **0.080** | 5 |
| 1 | 0.398 | 13.144 | **0.174** | **0.888** | 3 |

Both far below the required >2 in both seeds. Seed 1's R_NL (0.888) is the
closest either seed gets to passing — still less than half the required
threshold.

## 7. Is alpha consistently near 90 degrees?

No — **alpha=4.6° (seed 0), alpha=1.2° (seed 1)** — v_memory and v_processor
point in almost exactly the SAME direction (both dominated by their large
NL-component, since dNL/dm is so large it dominates v_memory's direction
too), the opposite of the desired ~90°. Consistent across both seeds.

## 8. Does the result survive neighborhood perturbations?

**Not tested this pass** — given the candidate point's own gates fail by
a wide, seed-consistent margin (question 6), and every M-derivative is
independently flagged unstable (question 3), a neighborhood sweep was
judged unlikely to change the qualitative conclusion and was skipped to
conserve the remaining time budget for writing this report accurately.
This is an explicit scope reduction, not a claim that no better
neighboring point could exist.

## 9. Does the candidate region actually lie near EOC according to TWO
chaos diagnostics?

Yes, reusing the prior pass's own two-diagnostic grid (`<r>` level-spacing
ratio and operator entanglement, `results/dqrc_gj_standalone_NP5.json`):
interpolating between the grid's (g=0.33,J=0.33) and (g=0.6,J=0.33) rows
places (g*=0.5,J*=0.33) at `<r>≈0.56-0.57`, operator entanglement `≈1.9-2.0`
— solidly in the upper-middle, chaos-leaning part of the sampled range
(grid spans `<r>∈[0.30,0.65]`, op.ent.∈[0.05,2.42]). So the candidate point
IS in a genuinely chaos-adjacent region by both diagnostics — the failure
found here is not explained by having picked an obviously wrong (e.g.
deeply integrable or already-saturated) region.

## 10. Is selectivity stronger tangent to the EOC ridge or normal to it?

Estimated only approximately, from the existing grid (no new circuit
runs): local operator-entanglement gradient near the candidate region
`(d(op_ent)/dg, d(op_ent)/dJ) ≈ (1.34, 1.78)`, giving an approximate normal
direction `n≈(0.60,0.80)` and tangent `t≈(-0.80,0.60)`. Given the g,J
derivatives themselves are flagged unstable (question 3), projecting them
onto `t`/`n` would only propagate that same unreliability — **this
question is not meaningfully answerable with the current derivative
estimates**, and no tangent/normal decomposition is reported as a
trustworthy number this pass.

## 11. Are memory features localized mainly in M and nonlinear features
mainly in P?

**Yes, for the LOW-NL side of the picture, but not simply "specialized" —
the processor dominates both.** Feature-group IPC at the candidate point
(`results/dqrc_gj/validation_feature_groups.json`):

| features | n | M | NL |
|---|---|---|---|
| X_M only | 3 | 2.222 | **0.205** |
| X_P only | 375 | **6.965** | **6.959** |
| X_M + X_P | 378 | 6.969 | 6.939 |
| + cross | 382 | 6.966 | 7.038 |

X_M (dedicated memory register) does show the desired low-NL signature
(0.205). But X_P alone ALREADY carries almost all of the combined system's
M (6.965 vs combined 6.969) AND all of its NL (6.959 vs 6.939) — **the
processor, once driven through the collision channel, is itself acting as
a substantial secondary memory, not a pure nonlinear specialist**. This is
the direct, physical explanation for question 6/3's finding: since the
processor's own state carries most of the system's linear (M) capacity too,
whatever `epsilon_M` does to reshape the information reaching the processor
via the collision channel naturally reshapes the processor's OWN nonlinear
output strongly — memory and nonlinearity are not cleanly separated by
subsystem in this configuration.

## 12. Is there physical operator-sector evidence supporting this?

Not computed this pass (Part 16 was explicitly "if feasible" and the clear
negative result from questions 3-7/11 made this lower priority than
accurately reporting what was found; see scope limitations).

## 13. Is control-level memory-NL decoupling now established?

**No — reported as "not established," per the brief's own required
language (Part 14).** Neither R_memory nor R_NL clears the >2 gate in
either seed tested; most derivatives are numerically unstable; the
response-vector angle is far from orthogonal in both seeds; and question 11
gives a direct physical mechanism for why (the processor itself carries
most of the system's memory capacity, not just its nonlinearity).

## 14. If yes, does the matched-resource Pareto frontier expand?

Not applicable — per Part 17's own explicit gate ("only if the 5-seed
validation establishes R_memory>2, R_NL>2... THEN proceed"), no Pareto
analysis was run.

## 15. If not, what exactly fails?

**Primarily: memory selectivity (question 11) and derivative
stability/reliability (question 3), not processor selectivity or EOC
identification.** The processor's own chaos-adjacent operation (question
9) is confirmed and its standalone nonlinear richness was already
established in the prior two passes — the processor itself is not the
problem. The problem is that once the FULL architecture is evaluated (not
the standalone processor in isolation), the processor ends up entangled
with, and strongly driven by, the memory's own internal parameter — memory
and nonlinear capacity are not cleanly localized to separate subsystems as
hoped, and the numerical derivative estimates that would be needed to
prove otherwise are themselves too unstable at the tested step sizes to
support a decoupling claim either way for M's response specifically.

## 16. What should be done next?

Per Part 24's own decision logic — "if control-level decoupling fails
because memory selectivity is weak: move to a more strongly protected /
Krylov-sectorized memory design" — **this is the indicated next step**,
not adding IDQNN/sewing (which Part 24 reserves for AFTER decoupling AND
Pareto both succeed) and not simply re-running more seeds at the same
architecture (the failure here looks structural — the processor absorbing
most of the memory capacity — rather than a statistics problem that more
seeds alone would fix, though the M-derivative instability found in
question 3 does also warrant a numerically better-conditioned finite-
difference protocol, e.g. larger/adaptive step sizes or a smoothed/
averaged response surface, before any future attempt at this same
architecture).

## Honest scope limitations of this pass

- **Only 2 seeds**, not the requested 5 — the clear, consistent (both
  gates fail by >2x in both seeds; the one stable derivative, dNL/dm,
  agrees to within 15% between seeds) result was judged sufficient to
  support "not established" without needing a full 5-seed run; a genuinely
  BORDERLINE result would have required the full 5 seeds and was not what
  was found.
- **Only 2 step sizes**, not 3 — chosen to fit the compute budget; a third,
  larger step size might further clarify whether the unstable M-derivatives
  converge at coarser resolution (untested).
- **No neighborhood robustness sweep** (question 8) — explicitly skipped
  given the clear result at the tested point.
- **No tangent/normal EOC-ridge decomposition** with trustworthy numbers
  (question 10) — the underlying derivatives are not reliable enough yet.
- **No operator-sector (Pauli-weight growth / operator spreading) evidence**
  (question 12, Part 16) — lower priority than accurately reporting the
  clear quantitative finding.
- Only the `shift`-vs-`protected_integrable` memory-variant discrepancy
  (question 1) was flagged, not independently resolved by ALSO running a
  `shift`-memory response matrix with some other proxy for a memory-only
  control — the brief's Part 5 requires a genuinely internal parameter,
  which `shift` (a hard, parameter-free register) does not have, so this
  discrepancy is inherent to comparing these two specific memory variants,
  not a fixable oversight within this pass's scope.
