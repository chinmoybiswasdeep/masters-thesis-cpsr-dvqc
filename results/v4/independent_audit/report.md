# Independent audit of the frozen V4 architecture

Preregistration sha256 `72885188a1a2ca622a36e1730b3a5d8fb19b877113198cf459a672a76f751e1f` · git `66ace2cbcfc8ec84ee646e79cc2fd868bf58c6ea` · confirmation: 20 fresh seeds, 5×5 grid, T=1600, exact noiseless simulation.

CI levels for verdicts are the SEQUENTIAL (stricter) ones: main-effect lower bound = 1st percentile, TOST interval = 1st–99th percentile, ratio bound = 99th percentile.

## Gate table

| claim | metric | threshold | estimate | CI / bound | result | evidence |
|---|---|---|---|---|---|---|
| 1 | frozen dX_R/dg, dX_P/dm, full-range (100 draws) | = 0 | 0 (all four) | exact | **PASS** | structure.json |
| 1 | contaminated controls detected | 3/3 | 3/3 | — | **PASS** | structure.json |
| 2 | ΔmM (ridge) | ≥0.10, LB>0 | +0.3631 | LB +0.3480 | **PASS** | gates.json |
| 2 | ΔgN (ridge) | ≥0.10, LB>0 | +0.9973 | LB +0.9950 | **PASS** | gates.json |
| 2 | ΔmN (ridge) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 2 | ΔgM (ridge) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 2 | degree profile (m sweep, ridge) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 2 | memory curve (g sweep, ridge) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | ΔmM (ols_std) | ≥0.10, LB>0 | +0.3632 | LB +0.3480 | **PASS** | gates.json |
| 3 | ΔgN (ols_std) | ≥0.10, LB>0 | +1.0038 | LB +1.0022 | **PASS** | gates.json |
| 3 | ΔmN (ols_std) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | ΔgM (ols_std) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | degree profile (m sweep, ols_std) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | memory curve (g sweep, ols_std) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | ΔmM (ols_raw) | ≥0.10, LB>0 | +0.3632 | LB +0.3480 | **PASS** | gates.json |
| 3 | ΔgN (ols_raw) | ≥0.10, LB>0 | +1.0038 | LB +1.0022 | **PASS** | gates.json |
| 3 | ΔmN (ols_raw) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | ΔgM (ols_raw) | TOST ⊂ ±0.03, ratio<0.20 | +0.00000 | [+0.00000, +0.00000], ratio≤+0.0000 | **PASS** | gates.json |
| 3 | degree profile (m sweep, ols_raw) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | memory curve (g sweep, ols_raw) | Bonferroni TOST | — | — | **PASS** | gates.json |
| 3 | interior M(1)−M(0.25), ols_std | ≥0.10, LB>0 | +0.0325 | LB +0.0083 | **FAIL** | gates.json |
| 3 | interior N(1)−N(0.25), ols_std | ≥0.10, LB>0 | +0.0000 | LB +0.0000 | **FAIL** | gates.json |
| 3 | section-6 verdict | HOLDS | CONTINUOUS INTRINSIC NONLINEAR CONTROL FAILED | — | **FAIL** | section6_confirmation.json |
| 3 | N-target saturation over g>0 (ols_std) | ≤0.20 | +1.000 | — | **FAIL** | gates.json |
| 4 | C1_curNL_x_oldLin (supported) | HH beats all, LB>0 (if supported) | +0.1929 | min LB +0.0990 | **PASS** | gates.json |
| 4 | C2_oldNL (unsupported) | HH beats all, LB>0 (if supported) | -0.0240 | min LB -0.0175 | **PASS** | gates.json |
| 4 | C3_old_x_old (unsupported) | HH beats all, LB>0 (if supported) | +0.0132 | min LB -0.0057 | **PASS** | gates.json |
| 4 | C4_old3 (unsupported) | HH beats all, LB>0 (if supported) | -0.0279 | min LB -0.0209 | **PASS** | gates.json |
| 5 | quantum joint − classical products (C1) | LB>0 | -0.0086 | LB -0.0098 | **FAIL** | gates.json, classical_confirmation.json |

## Verdicts

- `STRUCTURAL ROUTE ISOLATION: PASS`
- `RESOURCE-CONSTRAINED M–NL SEPARATION: PASS`
- `INTRINSIC M–NL SEPARATION: FAIL`
- `COMBINED NONLINEAR MEMORY: PASS`
- `QUANTUM-SPECIFIC MECHANISM: FAIL`

**Overall: `V4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED DEFINITION`**

Supported combined classes: ['C1_curNL_x_oldLin']; unsupported: ['C2_oldNL', 'C3_old_x_old', 'C4_old3'].

Section 6: **CONTINUOUS INTRINSIC NONLINEAR CONTROL FAILED**.

Claim 5: **COMBINED MECHANISM IS CLASSICALLY REPRODUCIBLE**.

## Figures

- `figures/fig01_response_surfaces.png`
- `figures/fig02_section6_ridge_artifact.png`
- `figures/fig03_ipc_heatmaps_HH.png`
- `figures/fig04_combined_classes.png`
- `figures/fig05_classical_baseline.png`
- `figures/fig06_sunada.png`
- `figures/fig07_robustness.png`
- `figures/fig08_verdicts.png`
