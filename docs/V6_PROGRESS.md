# V6 progress — general memory / nonlinearity / nonlinear-memory separation

Machine-readable state: `results/v6_progress.json`. Candidate history is the
append-only `results/v6/candidate_registry.jsonl`. Nothing in a registry line
is ever edited.

## Locked baseline (Step 1)

- The V5.4 confirmation was regenerated from source: 500/500 rows are
  bit-identical, across 754 000 numeric values. Every V5.4 verdict and estimate
  was reproduced.
- Frozen-file history: freeze `c0796ce` → amendment `a321d78` → confirmation
  `209a7fc`, verified.
- **Environment dependence:** OpenBLAS thread count changes values by
  ≤ 1.1e-15. With 1 thread only 64% of values stay bit-identical. Comparisons
  across processes or machines therefore use a tolerance of 1e-12 (≈ 4 500 ulp,
  far below any reported digit). Bit-identity is reported separately.
- Exact environment: `requirements-lock.txt`. Full suite: 644 passed at step 1.
- Evidence: `results/v6/step1/v5_4_reproduction.json`.

## Preregistered V6 gates (Step 2)

`results/v6/preregistered_v6_gates.json`, sha256 `2054fa10…`, committed as
`aeca1b0` before any V6 development data existed.
- **Combined classes:** all four classes C1–C4 are required under every readout.
- **Error control:** confirmation attempt k spends α_k = 0.01 / 2^(k−1). The
  cumulative total over V4, V5 and V6 stays ≤ 0.05. HH contrasts are
  Bonferroni-corrected over 36 tests.
- **Seed banks:** development seeds are 100000–100009 / 105000–105009.
  Confirmation attempt k uses 110000 + 10 000(k−1) … as its untouched bank.

## V5.4 span limitation (Step 3)

Every V5.4 operational feature is linear in each past input and contains at most
one past input. Under i.i.d. uniform inputs, P2(u_{t−τ}), P3(u_{t−τ}) and
P1(u_{t−τ1})P1(u_{t−τ2}) are therefore orthogonal to the whole feature span.
Long-sequence check (T = 40 000), from `results/v6/algebra/V5.4.json`:

| class | max member capacity |
|---|---|
| C1 | 0.81 |
| C2 | 0.0002 |
| C3 | 0.0006 |
| C4 | −0.0001 |

## Candidates

| version | stage reached | outcome | reason |
|---|---|---|---|
| V6.0 | PROBE | rejected | ΔN = 0.027: the P1 part of u³ dominates the processor output. The P2 content of the Q write is non-monotone in g. |
| V6.1 | PROBE | rejected | C3 HH 0.013 < HL 0.082: products of the nonlinear write lose their P1P1 part as g grows (structural). |
| V6.2 | DEVGATES | rejected | All 26 gates pass on dev seeds, but two freeze margins fail. C4's HH advantage is 0.044 < 0.05 under OLS. The perturbed-controls robustness condition fails, with worst HH advantage −0.178. |
| V6.3 | DEVGATES | rejected | Every gate and margin passes, including 38/38 robustness conditions, except two. The future sentinel reaches 0.0257 (one row) and the gate-15 max-member statistic 0.0219. Cause: 46 full-rank noisy features. |
| V6.4 | SENTCAL | rejected | 22 features; maximum excess 0.0168 > 0.016 dev margin |
| V6.5 | SENTCAL | rejected | 18 features; 0.0162 > 0.016 |
| V6.6 | SENTCAL | rejected | 11 features, pure-product combined route; all class margins met on the quick dev check; 0.0176 from one future-sentinel row |
| V6.7+ | SENTCAL | re-draws | The V6.6 design, unchanged, on fresh dev blocks 3–8 (user decision). |

### Decisions and amendments after V6.3

- **gates_amendment_01 (hashed):** the gate-15 empirical check now compares
  class scores (mean member), as the preregistered text says. The previous
  max-member statistic stays reported as a diagnostic. The discrepancy was found
  while diagnosing V6.3, and this is disclosed. Sentinels 17 and 18 are unchanged.
- **Dev sentinel margin:** fixed before V6.4. Over 500 extra dev rows, no row may
  exceed 0.02 and the maximum excess must be ≤ 0.016.
- **The margin behaves like chance:** input-independent AR(0.9) features reach
  0.0171 at 18 features. Pooled across designs, about 1 row in 500 exceeds 0.016,
  always from a single chance draw of an independent sentinel target. So a
  500-row block passes about 37% of the time whatever the design.
- **Minimum design size:** 10-feature variants miss class margins, so 11 features
  is the minimum.
- **User decision 1:** keep the 0.016 margin and keep searching.
- **User decision 2:** re-draw the V6.6 design on fresh dev blocks until one
  passes. Any resulting freeze is **selection on noise for this margin** and is
  labelled so in every report. The preregistered confirmation gates are
  unaffected.

### The V6.2 → V6.3 diagnosis

In the perturbed-controls condition the low setting is g = 0.03 instead of 0. V6.2
failed it in two ways:

1. **C4 (P3(u_{t−τ})) fell with g.** The cubic share of the Q writer saturates
   early: long-sequence capacity 0.33 at g = 0.25, 0.21 at g = 1. This is
   parametric. Making the writer's cubic angle scale as g² makes the cubic share
   grow with g: 0.03 → 0.11 → 0.17 → 0.21.
2. **Under exact expectations, C2 and C3 switch on at g = 0⁺.** OLS can subtract a
   feature's linear part using R's linear memory and rescale what remains. Any
   nonzero product coefficient therefore yields the full product information.
   Long-sequence capacity at g = 0.03 is already C2 = 0.86 and C3 = 0.82. This
   holds for every noiseless design in which the linear memory route sits in the
   same readout. It is a property of the exact-expectation idealisation, not of
   this parameter choice.

With a finite shot budget, a weak interaction carries little extractable product
information per shot, and every class grows monotonically with g. At S = 10⁴:

| class | g = 0.03 | g = 0.25 | g = 0.5 | g = 0.97 |
|---|---|---|---|---|
| C2 | 0.007 | 0.30 | 0.50 | 0.66 |
| C3 | 0.007 | 0.28 | 0.45 | 0.57 |

V6.3 therefore declares 10⁴ shots per measurement setting as part of the
architecture. The budget is identical at every (m, g), for every feature and for
the classical baselines.

**Disclosed and permanent limitation:** the gradation of C2 and C3 in g is a
finite-statistics effect. The primary N control is intrinsic, a composition
change of a single feature, and does not depend on shots. The exact-expectation
perturbed-controls grid is run and reported as a non-gating diagnostic.

## Resume

```bash
cd code
python run_v6_stage.py --stage DEVGATES --version V6.3   # resumes from checkpoints
# if the freeze criterion is met:
python run_v6_stage.py --stage TESTS  --version V6.3
python run_v6_stage.py --stage FREEZE --version V6.3
python run_v6_stage.py --stage CONFIRM --version V6.3
python run_v6_stage.py --stage REPORT --version V6.3
```
