# Directional M -> A -> P collision-channel architecture: results

Follow-up to `docs/IDQNN_MEMORY_EOC_RESULTS.md`. New modules:
`code/decoupled_qrc/{directional_memory,collision_interface,directional_processor,
directional_dqrc,directional_diagnostics}.py`, 5 new test files (35 new tests;
full suite 97/97 passing). Old notebooks/modules untouched, as instructed.
Configuration: N_M=2, N_A=1, N_P=5 (N_total=8, Part 9's primary diagnostic
budget), T_ipc=250, single-to-3-seed diagnostic runs only (not
PUBLICATION_MODE, per the brief).

**Headline: the collision channel genuinely achieves its designed physical
goal (measurably lower memory back-action than direct coupling, at
comparable or higher nonlinear capacity) — but it does NOT yet achieve
control-level decoupling. Nonlinearity remains far more sensitive to the
memory-side extraction control (theta) than to the processor's own EOC
control (kappa_processor), the same core problem found in the two prior
passes, now confirmed to persist even with a genuinely low-back-action
channel.** A real, load-bearing bug was found and fixed along the way: Part
4's literal QND generator (Z_M Z_A) cannot transfer any information to a
freshly-reset ancilla at all (both diagonal in the ancilla's own basis) —
the exact same physics as the earlier `'rzz'` interface failure, now
appearing at the M->A link. Fixed by using Z_M X_A instead (still exactly
QND for Z_M, verified by direct commutator computation).

## 1. Does the protected memory subsystem behave primarily as memory
rather than a nonlinear processor?

Partially confirmed, with an important caveat about measurement
reliability. The shift-register variant (N_M=2: 1 input-receiving qubit + 1
storage qubit) gives exact, hand-verifiable delay-1 recall (C(1)=1.0) and
low nonlinear IPC (NL_standalone=0.177). However, **a genuine, newly-found
limitation of `ipc.py`'s significance-filtering method was discovered while
computing this**: with only 3 readout features (a single qubit's X/Y/Z),
the shuffle-based surrogate null distribution becomes unreliable — some
surrogates (shuffled, i.e. FAKE targets) spuriously score capacity as high
as 1.0, causing the filter to reject even a genuinely perfect real signal
(IPC1 was computed as 0.000 despite C(1)=1.0 exactly). Verified this is
specific to very small feature counts: at 132 features (the scale used
throughout every other experiment in this project), the same test gives a
well-separated real score (0.98) vs. null (mean 0.077, max 0.39) — the
filter works correctly at normal scale. For this report, standalone memory
M is therefore reported via the raw `sum(C(k))` diagnostic (1.03-1.07
across 3 seeds) rather than the (here, unreliable) filtered IPC1 — a
substitution made necessary by, and explicitly flagged because of, this
newly-discovered edge case. This is a real methodological finding worth
carrying forward: **`ipc.py`'s permutation-test significance filter should
not be trusted at feature counts below roughly 10-20** without either more
surrogates or a different null construction.

## 2. Does the EOC processor independently generate strong nonlinear IPC?

Yes. Standalone processor (own EOC point, independently located for each
N_P, never reused across system sizes): N_P=5, kappa=2.322 -> NL=3.64-5.40
across 3 seeds; N_P=6, kappa=0.05 -> NL=5.19 (single seed). Both
substantial, confirming Gate 2.

## 3. Does M -> A -> P transfer useful temporal information?

Yes, but ONLY with the corrected M-A generator. `tests/test_directionality.py`
confirms mechanistically: with `phi=0` (A-P link cut) OR `theta=0` (M-A
link cut), the processor's total IPC is <0.5 (near-zero) regardless of the
other parameter — information genuinely requires BOTH links. With both
nonzero and the corrected `ma_kind='zx'` generator, the processor picks up
substantial real information (NL_proc_embedded up to 8.26 at
theta=0.8,phi=0.8). The literal Part-4 default (`ma_kind='zz'`) transfers
**exactly zero** information at any theta (verified: ancilla purity stays
1.0 to machine precision) — see the module-level finding above.

## 4. How much does the channel disturb the memory?

Measured directly (trace distance / fidelity / purity change between the
memory register's own reduced state with vs. without the collision channel
active, same trajectory): scales with `theta` alone, essentially
independent of `phi` (confirmed: D_M identical across all phi at fixed
theta in the 3x3 scan below) —

| theta | D_M (trace distance) |
|---|---|
| 0.2 | 0.029 |
| 0.5 | 0.167 |
| 0.8 | 0.375 |

A separate, stronger check (Part 14's "full channel" QND validation — not
just the bare commutator) found the tap qubit's own `<Z>` value is
disturbed by **exactly 0** (1e-15) between "M-A coupling active" and "M-A
coupling off", for this ZZ-diagonal-in-Z_M generator — the QND property
holds not just for the isolated gate but for the whole realistic channel
including the A-P stage and ancilla discard.

## 5. Is memory disturbance lower than direct M-P coupling?

**Yes, at comparable nonlinear capacity.** The prior pass's direct
Heisenberg coupling gave D_M=0.257 (trace distance) at NL~6-8 (see
`docs/IDQNN_MEMORY_EOC_RESULTS.md` question 7). The collision channel at
theta=0.2 gives NL=6.617 at D_M=**0.029** — roughly **9x lower disturbance**
for comparable nonlinear capacity. At theta=0.5 (D_M=0.167, NL=6.951), the
disturbance is still noticeably lower than direct coupling's 0.257 at
similar NL. **Gate 7 is met**: the collision channel measurably achieves
its central physical design goal.

## 6. What theta/phi region gives the best transfer/back-action balance?

Small theta (0.2) with large phi (0.8): NL=6.617 (proc-embedded: 6.621) at
D_M=0.029 — the lowest disturbance tested paired with among the highest NL
values found in the 3x3 grid (0.2-0.8 x 0.2-0.8):

| theta \\ phi | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| 0.2 | NL=0.25, D_M=0.029 | NL=6.43, D_M=0.029 | **NL=6.62, D_M=0.029** |
| 0.5 | NL=1.15, D_M=0.167 | NL=6.95, D_M=0.167 | NL=7.62, D_M=0.167 |
| 0.8 | NL=0.32, D_M=0.375 | NL=7.43, D_M=0.375 | NL=8.18, D_M=0.375 |

A clear pattern: phi must be large enough (>=0.5) for substantial transfer
at all (phi=0.2 gives poor NL regardless of theta) — and once phi is large,
INCREASING theta buys higher NL but at a roughly proportional cost in D_M.
This candidate point (theta=0.2, phi=0.8) was selected using ONLY this
information-theoretic scan (D_M, NL_proc), before computing any eta/Jacobian
numbers, per Part 26 step 11's requirement.

## 7. What are eta_M and eta_NL?

At the candidate point (theta=0.2, phi=0.8, kappa=2.322), 3 seeds:

| seed | eta_M | eta_NL (proc-only) |
|---|---|---|
| 0 | 5.97 | 1.82 |
| 1 | 5.50 | 1.59 |
| 2 | 5.49 | 0.73 |
| mean | **5.65** | **1.38** |

Both exceed 1 on average — per the brief's own instruction, these are
**normalized retention ratios, not efficiencies bounded by 1**; a value
>1 means the combined system contains MORE of that capacity than the
isolated reference, not an impossibility. eta_M's large value substantially
reflects the processor's own linear/memory-like capacity once genuinely
driven (as in the prior two passes) — not a clean isolation of the
dedicated 2-qubit memory register's own contribution; this caveat is
repeated because it remains true here.

## 8. Does changing g_P (kappa_processor) strongly alter NL while leaving
M approximately unchanged?

M: essentially unchanged (5.995 at kappa=2.322 vs. 6.000 at kappa=6.0,
i.e. `dM/dg ~ -0.0014` — very small, as desired). NL: also only modestly
affected (6.617 -> 5.968 over this kappa range, `dNL/dg ~ -0.18`) — a real
but comparatively small response. So the "M stays flat" half of the
question is answered yes; the "NL responds strongly" half is answered no
(see question 10's Jacobian for the direct comparison against `dNL/dm`).

## 9. Does changing the memory-only control strongly alter M while
leaving NL approximately unchanged?

No — the opposite is closer to true. Varying `theta` at fixed phi=0.8 (0.2
-> 0.5) gives `dM/dm ~ -0.22` (small, M stays close to flat, consistent with
question 8) but `dNL/dm ~ 3.36` (LARGE) — theta strongly changes NL, not
just M. Since `theta` is the M->A extraction-channel strength rather than a
purely-internal memory parameter (Part 10 also lists `epsilon_M`/
`gamma_M`/leakage/transport strength as alternative "m" choices; a
dedicated internal-memory-only control was not scanned this pass, see
"honest limitations" below), this specific Jacobian measures "does the
transfer channel's own strength control M or NL" rather than "does the
memory's OWN internal dynamics control M or NL" — an important distinction
carried into question 10/12's interpretation.

## 10. What is the Jacobian?

Using m=theta, g=kappa_processor (N_M, N_P, phi held fixed), finite
differences at the candidate operating point:

    dM/dm  = -0.217      dM/dg  = -0.0014
    dNL/dm =  3.357       dNL/dg = -0.177

R_M  = |dM/dm| / (|dM/dg| + eps)  ~ **159**   (Gate 5: >2 -- MET)
R_NL = |dNL/dg| / (|dNL/dm| + eps) ~ **0.053** (Gate 6: >2 -- **NOT MET**)

R_M's large value is partly a degenerate case worth flagging honestly: BOTH
`dM/dm` and `dM/dg` are small in absolute terms (M is close to flat against
*both* controls at this operating point), so the ratio being large does not
mean M is being usefully steered by `m` — it means M is largely insensitive
to everything tested here. R_NL failing is the substantive, structural
finding: nonlinear capacity in this architecture, like the two prior
architectures, remains dominated by the memory-side control rather than
the processor's own EOC knob.

## 11. Are both R_M > 1 and R_NL > 1 achieved?

R_M > 1: yes (~159). R_NL > 1: **no** (~0.05, off by roughly two orders of
magnitude). Both conditions are required by Part 17 before constructing a
Pareto frontier as a primary claim; since one fails, this report does NOT
present a validated Pareto-frontier-expansion claim (see question 13).

## 12. Does the architecture demonstrate control-level memory-NL
decoupling?

**No — subsystem separation and a genuinely low-back-action transfer
mechanism were both achieved (questions 1-6), but control-level decoupling
(the stricter, Part-25-defined claim) was not.** R_NL's failure is the
direct evidence: whatever knob currently moves information into the
processor (theta) also strongly moves its nonlinear output, so "processor
EOC control" and "information-transfer control" have not been
disentangled from each other in this design.

## 13. Does this translate into an outward Pareto-front expansion?

**Not established, and not claimed**, consistent with Part 17's own
gating rule (R_NL>1 not met) and Part 25's language discipline. A single
representative, single-point-per-seed comparison against a matched-N_total=8
monolithic baseline (same 3 seeds, same kappa where applicable) is reported
for context only, NOT as a validated frontier claim:

| seed | directional (M, NL) | monolithic N=8 (M, NL) | directional non-dominated? |
|---|---|---|---|
| 0 | (6.00, 6.62) | (5.94, 6.06) | **yes -- dominates monolithic here** |
| 1 | (5.87, 6.08) | (6.11, 7.53) | no |
| 2 | (5.82, 4.88) | (6.42, 7.81) | no |

1 of 3 seeds shows a directional point that dominates the matched-resource
monolithic baseline on both axes; the other 2 are dominated by it. This is
a noisy, preliminary, single-point (not full-frontier) comparison — **the
correct language per Part 25 is "control-level decoupling was not
established" and "no Pareto-frontier-expansion claim is made," not "the
memory-NL trade-off is broken."**

## 14. Does the result survive fixed-qubit accounting including the
ancilla?

Yes, and this was fixed during development to be so: `DirectionalConfig`'s
resource accounting was corrected mid-pass so that `N_M` (memory-side,
including its own input-receiving qubit) + `N_A` (=1, the collision
ancilla) + `N_P` = `N_total` exactly matches Part 9's specification
(N_M=2 + N_A=1 + N_P=5 = 8) — verified by
`tests/test_directional_dqrc.py::test_resource_accounting_matches_config`.
The monolithic comparison in question 13 used the SAME N_total=8, ancilla
included in the directional side's count, never hidden.

## 15. Should the next step be IDQNN memory, sewing, shadow readout,
Krylov-sector design, or abandonment of this architecture?

**None of the first three yet, and not abandonment either — the next step
should be fixing R_NL specifically**, since the architecture's physical
design goal (question 5: genuinely lower back-action than direct coupling)
was real and successful, and the failure is narrowly localized (control-level
decoupling, not subsystem separation or information transfer). Concrete
candidates for a future pass, in the same spirit as the fixes found in this
one: (a) use a memory-INTERNAL control (e.g. `epsilon_M` for the
`protected_integrable` variant, or an explicit memory-leakage/dephasing
rate) as `m` instead of the extraction-channel strength `theta`, which this
pass's own question 9 flagged as conflating "transfer strength" with
"memory's own dynamics"; (b) test whether a WEAKER but still-transferring
theta regime (below 0.2, if transfer survives) decouples further while
theta's OWN effect on NL shrinks; (c) revisit whether the A-P stage's own
strength `phi`, not `theta`, should be the thing coupled to `kappa_processor`
in a genuinely 2-parameter (not conflated) control scheme. Per Part 20,
IDQNN/sewing/shadows/MLP readout remain correctly out of scope until this
is resolved.

## Honest scope limitations of this pass

- All numbers above are single-to-3-seed (per the brief: no PUBLICATION_MODE),
  and the theta/phi scan and Jacobian are single-seed — the eta_NL spread
  (0.73-1.82) shows this matters.
- The Jacobian's `m=theta` is the M->A extraction-channel strength, not a
  purely memory-internal control (question 9's caveat) — Part 10's other
  suggested memory-only parameters (epsilon_M, gamma_M, leakage, dephasing)
  were implemented in `directional_memory.py` but not swept for this
  Jacobian.
- No fixed-feature-count comparison, canonical-correlation/PCA
  complementarity analysis (Part 16's fuller ask), or Krylov-complexity
  diagnostics (Part 21, explicitly optional) were run this pass.
- Only the `'xy'` A-P kind was used for the main scan; `'anisotropic'` and
  `'xx'` were only unit-tested, not scanned for eta/back-action.
- The Part 13 "back-action vs NL" and "M vs D_M" scatter plots (Figures
  5/6/8 of the brief) are represented here as tables, not yet as the full
  multi-point notebook figures the brief's Part 23 lists exhaustively;
  the notebook built alongside this report covers a representative subset,
  not all 13 requested figures.
