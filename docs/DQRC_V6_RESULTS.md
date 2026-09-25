# DQRC V6 — memory / nonlinearity / nonlinear-memory separation: confirmed result

**Outcome.** The frozen candidate **V6.7** passed every preregistered mandatory
gate on an untouched confirmation bank: 20 seeds, 500 rows, per-look α = 0.01,
cumulative α over V4, V5 and V6 = 0.04 ≤ 0.05.

The quantum-specific claim, reported separately, **fails**:

> General memory–nonlinearity separation succeeds, but the mechanism is
> classically reproducible.

Read the claim together with §8. The delayed nonlinear classes are carried almost
entirely by delays 1–2 (up to 4 for one old×old pair). Their gradation in g is a
finite-shot effect. The frozen design was picked by re-drawing a development
margin that behaves like chance.

Evidence:
- `results/v6/V6_7/` — `gates.json`, `report.md`, `figures/`, `frozen.json`,
  `manifest.json`
- `results/v6/candidate_registry.jsonl`
- `docs/V6_PROGRESS.md`

## 1. Architecture (V6.7, frozen sha256 `01b183eb…`)

| route | physics | control | feature(s) | used for |
|---|---|---|---|---|
| R | random-SWAP transport on rails 0..10, p = 0.7 m; rail 0 re-prepared in (I + u_t Z)/2 | m | ⟨Z_r⟩ on rails 2, 4, 6, 8, 10 (5) | **M only** |
| P | 3 copies of ρ(u); Ry(π/2) on copy 0; exp(−iθZ₀Z₁/2)·exp(−iχZ₀Z₁Z₂/2), θ = 0.12π g, χ = −0.30π g; measure (X₀+Y₀)/√2 | g | one observable f(u) = Σ a_d(g) u^d (1) | **N only** |
| J | 2-copy channel sin(0.45π g)·u_t² (Y₀), read jointly with Z on R rail 2 | m, g | z₂ · sin(θ_J g) u_t² (1) | C1 |
| Q | second random-SWAP register (L = 2, p = 0.7 m), written each step with a P-type processor output (rotated, dephased) | m, g | ⟨Z_r⟩, r = 1, 2 (2) | C2, C4 |
| R pairs | pure-product readout of R rail pairs (1,2), (2,4): Ry(π/2), exp(−iθ Z_aZ_b/2), θ = 0.40π g, measure Y_a | m, g | sin(θ g)·⟨Z_a Z_b⟩ (2) | C3 |

- **Feature count:** 11 operational features, identical at every (m, g), plus
  1 + 1 + 2 + 2 read in extra measurement settings.
- **Shot budget:** every feature is a 10⁴-shot estimate. This is a declared
  resource, identical at every (m, g), and shared with the classical baselines.
- **Exact simulation:** every register is Z-diagonal. Its first and second
  moments obey a closed linear recursion, verified against the full distribution
  to ≤ 6e-16 (`tests/test_v6.py`).

## 2. Why the controls separate

- **M cannot depend on g, and N cannot depend on m.** R's dynamics contain no g,
  and P is reset each step and contains no m. Structural test on confirmation
  (100 random draws): ∂X_R/∂g = ∂X_P/∂m = 0 exactly. The g→R, m→P and serial
  contaminations are detected (derivatives 7 700, 100 and 100).
- **The P control is intrinsic.** P is read through ONE observable. Its OLS
  capacity per degree, C_d = a_d²/Σa², is the variance fraction of each degree.
  Rotating θ and χ changes that composition, not only its amplitude.
  - Section 6 holds: the N trend is 0.309 under raw OLS, standardised OLS and
    every ridge penalty.
  - The design uses χ < 0 to cancel the P₁ part of u³ at g = 1. Exact degree
    fractions at g = 1 are 0.00 / 0.60 / 0.33 (degrees 1, 2, 3).
- **Every nonlinear feature vanishes at g = 0.** All operational features are
  exactly affine in the input history: the superposition violation is
  ≤ 6.7e-16. Every delayed feature is frozen at m = 0 (p = 0).

## 3. Feature algebra (why C1–C4 are reachable, and why V5.4 was not)

- **V5.4 theorem.** Every V5.4 feature is linear in each past input and contains
  at most one past input. Under i.i.d. uniform inputs, P₂(u_{t−τ}), P₃(u_{t−τ})
  and P₁(u_{t−τ1})P₁(u_{t−τ2}) are therefore orthogonal to the whole span.
  Measured (T = 40 000): C2 0.0002, C3 0.0006, C4 −0.0001; C1 0.81.
- **V6.7 reaches each class through one mechanism:**
  - C1: J = z_r · u_t² contains P₂(u_t)·P₁(u_{t−k}).
  - C2 and C4: the Q rails hold Σ_k A_rk f(u_{t−k}), where f has degree-2 and
    degree-3 parts when g > 0.
  - C3: ⟨Z_aZ_b⟩ of a permutation mixture is Σ_{i≠j} W_ab(i,j) u_{t−i}u_{t−j}.
- **Long-sequence reachability** at (m, g) = (1, 1): C1 0.94, C2 0.59, C3 0.93,
  C4 0.34 (maximum member). Every class is ≈ 0 at m = 0 and at g = 0. The
  unreachable sentinel (P₅ of one input) is null.
- **Planted-feature tests:** every class is detected when planted and not
  detected when absent.

## 4. Confirmation results (20 untouched seeds 110000–110019 / 115000–115019)

| gate | result | key numbers |
|---|---|---|
| m→M, g→N (ridge / OLS-std / OLS-raw) | PASS | ΔM 0.508 (LB 0.502); ΔN 0.309 (LB 0.301) |
| interior effects | PASS | M(1)−M(0.25) 0.386 (LB 0.380); N(1)−N(0.25) 0.306 (LB 0.297) |
| cross effects, ratios | PASS | TOST [0.0000, 0.0000] both directions; ratio 0 |
| degree profile under m; memory curve under g | PASS | exact equivalence |
| C1–C4, every readout | PASS | see table below |
| feature counts, encoder leakage, saturation | PASS | max metric M 0.51, N 0.31; 0% constituents > 0.995 |
| future / unreachable sentinels | PASS | 0.0106 / 0.0083 (limit 0.02) |
| isolation + contaminated controls; invalid controls | PASS | 6/6 detected |
| seed reproducibility | PASS | 100% of seeds HH-best in every class |
| robustness (28 conditions, 10 confirmation seeds) | PASS | worst effect ≥ 0.188; worst HH advantage ≥ 0.084 |
| full unit-test suite | PASS | 660 passed |

HH advantages come out identical under all three readouts to three decimals. The
minimum lower bound is the Bonferroni α/36 bound.

| class (quadrant means) | LL | LH | HL | HH | min LB |
|---|---|---|---|---|---|
| C1 P₂(u_t)P₁(u_{t−τ}) | −0.004 | 0.009 | 0.055 | 0.145 | 0.086 |
| C2 P₂(u_{t−τ}) | −0.009 | 0.035 | −0.006 | 0.150 | 0.111 |
| C3 P₁P₁ distinct delays | −0.011 | −0.007 | 0.023 | 0.088 | 0.061 |
| C4 P₃(u_{t−τ}) | −0.006 | 0.019 | 0.001 | 0.089 | 0.064 |

## 5. Classical baseline and quantum-specific verdict

Differences below are for the HH mean of the four class scores (ridge).

| baseline | ALL − baseline | LB |
|---|---|---|
| CLS_M (matched budget) | −0.004 | −0.004 |
| CLS_F (full degree-2 expansion) | −0.245 | −0.250 |

Both registers stay diagonal in a product basis, so the mechanism is a classical
Markov chain over bitstrings. **Verdict: classically reproducible. No quantum
advantage is claimed.**

## 6. Error control, seeds, hashes

- **Preregistration:** `preregistered_v6_gates.json` (sha256 `2054fa10…`),
  committed as `aeca1b0` before any V6 data existed.
- **Gate-15 amendment:** gates_amendment_01 (hashed). Gate 15's code was
  realigned to its preregistered text; this was found while diagnosing V6.3, and
  that is disclosed. V6.7 would also pass the stricter original statistic
  (0.0151).
- **α ledger:** V4 0.01, V4 audit 0.01, V5.4 0.01, V6 attempt 1 (V6.7) 0.01;
  cumulative 0.04.
- **Seed separation:**
  - The confirmation bank is disjoint from every development, calibration and
    earlier confirmation seed; this was checked at freeze.
  - Development: 100000–100009 / 105000–105009.
  - Sentinel calibration: blocks from 100100… / 105100…
- **Frozen sources:** hashed in `frozen.json`. Commits: freeze `fa6a0a0` →
  confirmation (this commit).

## 7. The search (append-only registry)

| version | fate | reason |
|---|---|---|
| V6.0 | rejected (probe) | P₁ part of u³ dominated the processor output (ΔN 0.027) |
| V6.1 | rejected (probe) | products of the nonlinear write lose their P₁P₁ part at high g (C3 HH < HL) |
| V6.2 | rejected (dev) | C4 fell with g. Under exact expectations OLS switches C2/C3 on at g = 0⁺ (perturbed-controls failure). |
| V6.3 | rejected (dev) | future sentinel 0.0257 (46 full-rank noisy features) |
| V6.4, V6.5, V6.6 | rejected (dev sentinel margin) | 0.0168, 0.0162, 0.0176 against a 0.016 margin |
| **V6.7** | **frozen, confirmed** | V6.6 design, fresh calibration block 3 (0.0100) |

## 8. Limitations (read these with the claim)

1. **Short delay reach.** Per-target capacities at HH:
   - C1 is carried by P₂(u_t)P₁(u_{t−2}), at 0.80.
   - C2 is carried by τ = 1, 2 (0.38 each).
   - C3 is carried by the pairs (1,2) at 0.79 and (2,4) at 0.37.
   - C4 is carried by τ = 1, 2 (0.22 each).
   - Every target at τ ≥ 3–5 is ≈ 0.

   The preregistered class scores average over τ = 1–6, so the gates pass. This
   is **short-range** delayed nonlinear memory, not nonlinear memory at arbitrary
   delay. The linear memory profile is comb-like: even delays 0.97 / 0.88 /
   0.75 / 0.57, odd delays 0.03–0.23, a consequence of the stride-2 readout.
2. **Gradation of C2/C3 in g is a finite-statistics effect.** With exact
   expectations, any g > 0 switches C2 and C3 fully on. The exact-expectation
   perturbed grid fails (worst HH advantage −0.001); it is reported as a
   non-gating diagnostic. The primary N control does not depend on shots.
3. **Selection on noise for one development margin.** The 0.016 dev sentinel
   margin was set before V6.4. It was shown to sit at the chance floor of any
   memory architecture: input-independent AR features reach 0.0171, and about
   1 row in 500 exceeds 0.016 whatever the design. By user decision the V6.6
   design was re-drawn on fresh development blocks until one passed (V6.7, block
   3, first re-draw). The preregistered confirmation gates, including the 0.02
   sentinel, were not changed and passed.
4. **Robustness coverage gap.** The frozen robustness list omitted ±10%
   perturbations of θ_J, θ_RP and φ_RP. They were run after confirmation as a
   supplementary, non-gating check and all six passed
   (`supplementary_robustness_post_confirmation.json`).
5. **Scope.** Noiseless simulation plus a FakeTorino-informed effective noise
   model; no transpiled circuits and no hardware run. The i.i.d. U[−1,1] input
   and T = 1600 are fixed. Seeds vary only the input sequence (there is no
   architectural disorder).
6. **Classically reproducible** (§5).

## 9. Paper-ready claim

> In a three-route simulated reservoir (a random-SWAP linear memory, a
> single-observable three-copy processor, and a nonlinear-memory route), memory
> (m) and instantaneous nonlinearity (g) are independently controllable. Under
> ridge, standardised OLS and raw OLS readouts, m changes linear memory by 0.51
> and g changes instantaneous nonlinearity by 0.31, with cross-effects exactly
> zero. The nonlinearity control is intrinsic: it changes the degree composition
> of a single feature, not its amplitude. Four classes of delayed nonlinear
> targets are each highest when both controls are high, with Bonferroni-corrected
> lower bounds ≥ 0.06: current-nonlinear × past-linear, delayed quadratic,
> past × past, and delayed cubic. The classes are carried mainly by delays of
> 1–2 steps. These results held on 20 untouched seeds under preregistered gates
> (α = 0.01). Their gradation in g relies on a finite shot budget (10⁴). The
> mechanism is classically reproducible.

## 10. Reproduce

```bash
python -m pytest tests/ -q                       # 660 passed (818 s)
cd code
python run_v6_stage.py --stage STEP1             # V5.4 bit-for-bit reproduction
python run_v6_stage.py --stage ALGEBRA  --version V5.4
python run_v6_stage.py --stage ALGEBRA  --version V6.7
python run_v6_stage.py --stage SENTCAL  --version V6.7
python run_v6_stage.py --stage DEVGATES --version V6.7   # resumes from checkpoints (~20 min)
python run_v6_stage.py --stage REPORT   --version V6.7
python _build_notebook_v6_colab.py
```

Runtimes on this machine (12-core CPU): V6.7 dev gates 1 158 s; confirmation
1 938 s; sentinel calibration ≈ 210 s per block. Row checkpoints (`*.jsonl`) are
git-ignored; they regenerate bit-for-bit and are hashed in `manifest.json`.
