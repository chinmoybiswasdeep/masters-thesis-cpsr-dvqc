# V4 Architecture Audit — Memory / Nonlinearity Separation

**Scope.** V4 replaces the V3.2 architecture completely. V3.2 is preserved for comparison and
is neither modified nor superseded in place; no V1–V3.2 module, notebook, result or test was
edited.

This document records what was *measured* on the way to the V4 design, including the two
mathematical obstructions that forced design changes and the one claim from earlier versions
that had to be corrected.

---

## 1. Why V3.2 was not extended

V3.2 reached Level 1 (structural route isolation) and stopped there. Its blocking problems were
architectural, and three of them recur as *general* obstructions that any design in this family
must answer:

| V3.2 problem | V4 answer |
|---|---|
| encoder generated the nonlinearity before `(g,J)` acted | encoder is affine **at the local readout**, and the copies are counted as a nonlinear resource (§2) |
| the primary metric was ceiling-pinned, needing shot noise for dynamic range | metric is graded by a **bounded-norm readout** in an exact noiseless simulation (§4) |
| cross-effects were empirical and entangled with back-action | R and P are **never coupled**; cross-effects are exactly zero and the negative controls prove the test still has power (§5) |

---

## 2. The encoder claim was wrong, and is now stated correctly

An earlier draft in this project described `rho(u) = ((I + uZ)/2)^{⊗n}` as "globally linear in
u". **That is false.** Exact Legendre projection (Gauss–Legendre quadrature is exact for
polynomials, so a reported zero *is* zero):

| n | `⟨Z₀⟩` degrees | `⟨Z₀…Z_{n−1}⟩` degrees | globally affine? |
|---|---|---|---|
| 1 | {1} | {1} | **yes** |
| 2 | {1} | {0, 2} | no |
| 3 | {1} | {1, 3} | no |
| 4 | {1} | {0, 2, 4} | no |

The correct statement: the encoder is **affine at the local readout**, and the `n` copies are a
**nonlinear resource**. `v4_encoder.ReuploadInjection.resources()` reports the copy count, which
is held fixed across every value of `g`, so `g` changes interaction strength only and can never
be credited with capability it did not pay for.

Reproduce: `python run_v4_stage.py --stage AUDIT` → `results/v4/audit.json`.

---

## 3. Obstruction 1 — a multilinear memory has a hard capability boundary

**Claim.** Quantum channels are linear in `ρ`. If each input enters R affinely exactly once,
then `ρ_R(t)` is *multilinear* in the past inputs: every `u_{t−s}` appears at degree ≤ 1.

**Measured.** `v4_encoder.multilinearity_report` on the V4 memory route returns
`max_degree_per_input = {0: 1, 1: 1, 2: 1}`, `multilinear = True`, with a nonzero mixed second
difference (8.3e-05) confirming genuine cross-delay terms are present.

**Consequence.** The joint span is `{multilinear in u_{≤t}} × {polynomial in u_t}`. A monomial is
reachable **iff its degree in every strictly positive delay is ≤ 1**:

| target | reachable |
|---|---|
| `P₁(u_{t−3})`, `P₁(u_{t−1})P₁(u_{t−3})` | yes |
| `P₂(u_t)`, `P₂(u_t)P₁(u_{t−3})`, `P₃(u_t)P₁(u_{t−2})P₁(u_{t−4})` | yes |
| **`P₂(u_{t−3})`, `P₂(u_{t−2})P₁(u_{t−4})`, `P₃(u_{t−1})`** | **no** |

**Design decision, and why the obvious fix was rejected.** Giving R a second fixed injection
would make `P₂(u_{t−3})` reachable — but *from R alone*. The high-m/low-g corner would then
already solve the combined task, and the combined-capability gate would lose all content.
Keeping R strictly affine forces N_long targets of the form `P_d(u_t)·P₁(u_{t−τ})` to draw the
**degree from P** (control `g`) and the **delay from R** (control `m`), so both controls are
genuinely required.

The unreachable class is **measured and reported, never gated**: the confirmation run evaluates
it (`unreachable_probe`) and the Sunada benchmark displays it directly.

---

## 4. Obstruction 2 — span-based capacity is binary in a continuous coupling

**Measured failure.** With a validation-selected ridge penalty, the local nonlinear score was

```
g      0.00   0.10   0.20   0.35   0.50   0.70   0.85   1.00
N      0.003  3.000  3.000  3.000  3.000  3.000  3.000  3.000
```

`∂N/∂g = 0`. Three fixes were tried and **all failed**:

1. **Reduce the processor feature count** (span-limiting). Only lowered the ceiling —
   `F_P`=2 → 0.96, 3 → 1.98, 4 → 3.00 — the response stayed a step function at every count.
2. **Change the interaction strength range.** `g_scale` ∈ {0.3, 0.8, 2.5} gave identical steps.
3. **A fixed bounded readout at a strong coupling.** Produced a large range but *non-monotone*
   (2.44, 2.88, 3.00, 2.96, 3.00, 2.70) — the coefficients oscillate with `g`.

**Diagnosis.** With an *unbounded* linear readout in exact arithmetic, capacity depends only on
the **span** of the feature functions, not on coefficient magnitudes. Any `g > 0` makes every
reachable degree appear, so `C_d` jumps to its ceiling and stays. No feature count and no
coupling strength can grade it.

**Resolution (two parts, both required).**

1. A **bounded-norm readout**: a fixed, preregistered ridge penalty `alpha`. A component whose
   relative amplitude is `O(g)` needs weights `O(1/g)` to extract, which the penalty forbids.
   This is a *resource constraint* — identical at every `(m, g)`, entered into the frozen config,
   and counted in the resource table — not a per-point tuned knob.
2. A **perturbative interaction** (`g·dt·g_scale ≪ 1`), so degree-`d` content scales as
   `(g dt)^{d−1}` monotonically instead of oscillating.

**Measured result**, showing exactly the predicted mechanism — degree entering in order:

```
g      C2      C3      C4      N
0.00   0.000   0.003   0.000   0.003
0.15   0.994   0.731   0.064   1.789
0.30   0.995   0.999   0.214   2.208
0.50   0.996   1.000   0.446   2.441
1.00   0.998   1.000   0.877   2.875      monotone
```

**This is explicitly not the forbidden move.** The simulation is exact and noiseless everywhere;
finite shots are never used to manufacture nonlinearity. The bound is on the *readout*, applies
equally to both routes and every baseline, and is frozen before the search.

---

## 5. The V4 architecture

```
u_t ──► R (memory)      L_R rails, affine SINGLE-copy injection on rail 0 (gain independent of m),
        │               fixed disordered XY mixing, retention channel with rate m
        │               R-local observables ──────────────► M
        │
        └► P (processor) N_P copies of (I+uZ)/2, RESET every step (memory ≡ 0),
                         fixed depth for all g, interaction strength g
                         P-local observables ─────────────► N

            joint layer: fixed O_R ⊗ O_P observables ─────► N_long
```

**Per timestep, in this order:** inject (gain independent of `m`) → fixed mixing → **read out** →
apply retention. Reading before damping keeps `C_{1,0}` unattenuated; injecting at fixed gain
decouples input strength from retention, so `m` is retention *alone*.

**Why memory is finite in an exact noiseless simulation.** A linear readout is scale-invariant,
so attenuation alone does not reduce capacity. What bounds it is the Dambre limit: with `F`
independent readout variables, `Σ_τ C_{1,τ} ≤ F`. Memory capacity is a **conserved budget that
`m` redistributes across delays**, which is why `M` is defined over `τ ≥ τ_L`.

**`m → retention` is capped at `m_max = 0.9`.** At zero damping the register is unitary:
information is never discarded but scrambles into high-weight operators the local readout cannot
see, and measured `M` turns over (0.398 → 0.302 at `m = 1`). Capping retention keeps `∂M/∂m`
monotone across the whole grid.

---

## 6. Architecture iterations attempted

| # | change | outcome |
|---|---|---|
| 1 | dual product, `g_scale = 2.5`, validation-selected ridge | **failed** — `N(g)` pinned at 3.000 for all `g>0`; N_long HH − best_other **negative** |
| 2 | reduce processor feature count (span-limiting) | **failed** — step function at every `F_P` |
| 3 | fixed bounded readout, strong coupling | **failed** — large range but non-monotone |
| 4 | **perturbative coupling + fixed bounded readout** | **passed** — `N(g)` monotone 0.003 → 2.875; N_long HH − best_other **+1.37** |
| 5 | `m → m·m_max` cap; N_long delays restricted to the memory horizon | **passed** — `M(m)` monotone to `m=1`; N_long quadrant gap +0.272 |
| 6 | cross-delay families added to the primary diagnostic set | **passed** — worst saturation 0.22 → **0.056** (limit 0.20) |

---

## 7. Search outcome — reported as it happened

24 seeded random candidates were evaluated on development seeds only, ranked by **worst-case**
performance across seeds (not mean), against the objective

```
L = −Δ_m M − Δ_g N + λ₁|Δ_m N| + λ₂|Δ_g M| + λ₃ S_sat + λ₄ S_leak − λ₅ Δ_HH N_long
```

**The hand-designed candidate won** (score −1.7696) against the best random candidate
(−1.624). The search did **not** discover the architecture; it confirmed it and failed to
improve on it. That is the honest result and it is what the report states.

Every one of the 24 candidates showed cross-effects of exactly `0.0e+00`, so the exact isolation
is a property of the *architecture class*, not of the selected parameter point.

---

## 8. What separation here does and does not mean

**Does.** The route-restricted cross-derivatives are exactly zero, the diagonal effects are
large and monotone, N_long requires both controls, and the pipeline detects coupling when it is
present (`contaminated` and `serial` both fail as required).

**Does not.** The zero cross-effects are **by construction**, not an empirical discovery. A
design that separates two resources by never coupling them is only interesting because of what
is *also* true: that the combined capability survives, that the diagonal responses are real and
robust, and that the same measurement would have caught coupling had it existed. Those three are
the empirical content, and they are what the gates test.

**Capability boundary.** Targets with degree ≥ 2 at a strictly positive delay are provably
unreachable by this architecture. They are measured and reported; they are not gated, and no
claim is made about them.

---

## 9. Confirmation result

One untouched confirmation run, frozen config `435357bfd39db1b0b407400c36debf41...`,
5×5 grid, `T = 1600`, architecture seeds `900–904` × input seeds `9000–9002` — none of which
appears anywhere in the development or search code paths. 375/375 rows complete, every row
carrying the frozen hash. Sequential ledger: attempt 1 of 5, α spent 0.01 of 0.05.

**8 / 8 mandatory gates passed.**

| gate | result |
|---|---|
| memory main effect | ΔₘM = **+0.37108**, 95% CI [+0.35352, +0.38988], threshold 0.10 |
| nonlinear main effect | Δ_gN = **+0.99291**, 95% CI [+0.98881, +0.99684], threshold 0.10 |
| m → N cross effect | **0.000e+00**, 90% CI [0, 0] ⊂ [−0.03, +0.03]; ratio bound 0.00000 < 0.20 |
| g → M cross effect | **0.000e+00**, 90% CI [0, 0] ⊂ [−0.03, +0.03]; ratio bound 0.00000 < 0.20 |
| degree-profile equivalence (m sweep) | family equivalent after Holm |
| degree-profile equivalence (g sweep) | family equivalent after Holm |
| combined capability | N_long HH − best other = **+0.06814**, 95% CI [+0.05914, +0.07872] |
| controls | isolation 0.0 / 0.0; factorisation 2.2e-16; contaminated **detected** (2.5e-03); serial **detected**; saturation 0.083 ≤ 0.20; encoder-only N 0.00123 vs null 0.00410; density matrices valid |

Response Jacobian:

```
            d/dm        d/dg
  M      +0.371079   +0.000000
  N      +0.000000   +0.992913
```

off-diagonal / diagonal mass = **0.0**; response angle = **90.00°**.

Quadrant means for N_long: LL 0.0022, LH 0.0031, HL 0.0940, **HH 0.1622**.

### 9.1 Two caveats found by scrutinising the passing result

Both were found by checking the confirmation output rather than accepting the summary line,
and both are reported because they qualify what the pass means. Neither was patched after the
fact — the gates are exactly as frozen.

**(a) The combined-capability gate is less independent than it looks.** It requires a positive
lower bound on three quantities, but two of them — `N_long` and `cross_delay_mean` — are
**numerically identical by construction** (both are the mean over the same N_long target
library; verified equal to 1e-12). The gate therefore rests on **two** distinct quantities, not
three. The genuinely independent one is `cross_family_mean` (the P1P1 / P2P1 / P1P1P1
library), which gives a much smaller but still strictly positive advantage:
**+0.00283, 95% CI [+0.00141, +0.00473]**. The headline combined effect is carried by the
N_long library.

**(b) The Sunada "boundary" numbers do not show what their name suggests.** The confirmation
reports `max_boundary = 0.99968` over cells with ν > 0 and τ > 0, which are supposed to be
unreachable. That high value is an artifact of the score, not a contradiction:
`sin(νu)/ν = u − ν²u³/6 + …`, so for small ν the target is almost entirely linear. Measured
corr² between the τ = 3 target and a pure linear `u_{t−3}`:

| ν | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|
| corr² with pure linear | 0.99988 | 0.99793 | 0.95741 | 0.12463 |

A high corr² at ν ≤ 2 therefore says nothing about the unreachable cubic component. **The clean
evidence for the capability boundary is the IPC probe on pure Legendre targets**
`P_d(u_{t−τ})`, which are orthogonal to the linear part: at the high-m/high-g corner those
capacities are **max 0.00000, mean 0.00000** against a null threshold of 0.00452, and the
worst value anywhere on the 375-row grid is 0.0122. The boundary holds; the Sunada cells are
simply the wrong instrument for testing it, and are reported as a benchmark rather than as
boundary evidence.

### 9.2 What the result does and does not establish

**Established.** On data never seen during design or search, `m` moves `M` and not `N`, `g`
moves `N` and not `M`, both effects are far above threshold with confidence intervals well
clear of zero, the combined capability requires both controls, and the pipeline detects
injected coupling when it is present.

**Not established by this run.** The zero cross-effects are exact **by construction**, so the
equivalence tests on them confirm the implementation rather than discovering a fact about
nature. The empirical content lies in the diagonal effects, the combined capability, and the
demonstrated power of the negative controls — not in the zeros themselves. A reader who wants
the zeros to be surprising should read §5 and §8 first.
