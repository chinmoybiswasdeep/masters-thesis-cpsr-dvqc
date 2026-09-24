# V5.4 — random-SWAP memory × two-copy processor, confirmation

Frozen V5.4 sha256 `0a2f1e184db3977b81648b0823ca62dc13d560a9df8024a462723254a2c08a02` · git `a321d78a6282ca8b6ee6d212d532c065a45e0875` · confirmation: 20 fresh seeds, 5×5 grid, T=1600, exact noiseless simulation.

CI levels for verdicts are the SEQUENTIAL (stricter) ones: main-effect lower bound = 1st percentile, TOST interval = 1st–99th percentile, ratio bound = 99th percentile.

## Gate table

| claim | metric | threshold | estimate | CI / bound | result | evidence |
|---|---|---|---|---|---|---|
| 1 | frozen dX_R/dg, dX_P/dm, full-range (100 draws) | = 0 | 0 (all four) | exact | **PASS** | structure.json |
| 1 | contaminated controls detected | 3/3 | 3/3 | — | **PASS** | structure.json |
| 2 | ΔmM (ridge) | ≥0.10, LB>0 | +0.5219 | LB +0.5156 | **PASS** | gates.json |
| 2 | ΔgN (ridge) | ≥0.10, LB>0 | +0.2800 | LB +0.2787 | **PASS** | gates.json |
| 2 | ΔmN (ridge) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 2 | ΔgM (ridge) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 2 | degree profile (m sweep, ridge) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 2 | memory curve (g sweep, ridge) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | ΔmM (ols_std) | ≥0.10, LB>0 | +0.5218 | LB +0.5155 | **PASS** | gates.json |
| 3 | ΔgN (ols_std) | ≥0.10, LB>0 | +0.2800 | LB +0.2787 | **PASS** | gates.json |
| 3 | ΔmN (ols_std) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | ΔgM (ols_std) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | degree profile (m sweep, ols_std) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | memory curve (g sweep, ols_std) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | ΔmM (ols_raw) | ≥0.10, LB>0 | +0.5218 | LB +0.5155 | **PASS** | gates.json |
| 3 | ΔgN (ols_raw) | ≥0.10, LB>0 | +0.2800 | LB +0.2787 | **PASS** | gates.json |
| 3 | ΔmN (ols_raw) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | ΔgM (ols_raw) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | degree profile (m sweep, ols_raw) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | memory curve (g sweep, ols_raw) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | interior M(1)−M(0.25), ols_std | ≥0.10, LB>0 | +0.3157 | LB +0.3086 | **PASS** | gates.json |
| 3 | interior N(1)−N(0.25), ols_std | ≥0.10, LB>0 | +0.2553 | LB +0.2511 | **PASS** | gates.json |
| 3 | section-6 verdict | HOLDS | CONTINUOUS INTRINSIC NONLINEAR CONTROL HOLDS | — | **PASS** | section6_confirmation.json |
| 3 | N-target saturation over g>0 (ols_std) | ≤0.20 | +0.000 | — | **PASS** | gates.json |
| 4 | C1_curNL_x_oldLin (supported) | HH beats all, LB>0 (if supported) | +0.1187 | min LB +0.1013 | **PASS** | gates.json |
| 4 | C2_oldNL (unsupported) | HH beats all, LB>0 (if supported) | -0.0168 | min LB -0.0101 | **PASS** | gates.json |
| 4 | C3_old_x_old (unsupported) | HH beats all, LB>0 (if supported) | -0.0184 | min LB -0.0107 | **PASS** | gates.json |
| 4 | C4_old3 (unsupported) | HH beats all, LB>0 (if supported) | -0.0191 | min LB -0.0100 | **PASS** | gates.json |
| 5 | quantum joint − classical products (C1) | LB>0 | +0.0000 | LB +0.0000 | **FAIL** | gates.json, classical_confirmation.json |

## Verdicts

- `STRUCTURAL ROUTE ISOLATION: PASS`
- `RESOURCE-CONSTRAINED M–NL SEPARATION: PASS`
- `INTRINSIC M–NL SEPARATION: PASS`
- `COMBINED NONLINEAR MEMORY: PASS`
- `QUANTUM-SPECIFIC MECHANISM: FAIL`

**Overall: `V5.4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED DEFINITION`**

Supported combined classes: ['C1_curNL_x_oldLin']; unsupported: ['C2_oldNL', 'C3_old_x_old', 'C4_old3'].

Section 6: **CONTINUOUS INTRINSIC NONLINEAR CONTROL HOLDS**.

Claim 5: **COMBINED MECHANISM IS CLASSICALLY REPRODUCIBLE**.


## Reading of the overall label

The label `V5.4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED DEFINITION` comes from the V4 audit's `overall()` mapping, which has only the brief's four outcomes. The pattern *claims 1-4 PASS, claim 5 FAIL* matches none of them and falls through to the last branch. That label is **inaccurate here: claim 3 (intrinsic separation) PASSED**. Correct reading: memory and nonlinearity are separately and intrinsically controllable (claims 1-3), combined nonlinear memory is highest at high-m/high-g for the one supported class (claim 4), and the combining mechanism is classically reproducible (claim 5). It is not `WORKS AS CLAIMED` in the brief's sense because claim 5 fails.

## Figures

- `figures/fig01_response_surfaces.png`
- `figures/fig02_section6.png`
- `figures/fig03_combined_classes.png`
- `figures/fig04_classical_baseline.png`
- `figures/fig05_robustness.png`
- `figures/fig06_verdicts.png`
