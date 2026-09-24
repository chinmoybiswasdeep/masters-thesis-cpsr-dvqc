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
| V6.2 | DEVGATES | running | C3 now comes from g-rotated pair readouts on the linear register R |

## Resume

```bash
cd code
python run_v6_stage.py --stage DEVGATES --version V6.2   # resumes from checkpoints
```
