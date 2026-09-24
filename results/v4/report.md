# DQRC V4 — Memory / Nonlinearity Separation

**MEMORY-NONLINEARITY SEPARATION DEMONSTRATED**

Frozen configuration: `435357bfd39db1b0b407400c36debf416ff903cc1bbc54a939d4617a5be4d728`


## Gate table

| gate | status | key numbers |
|---|---|---|
| combined capability | **PASS** | N_long improvement +0.0681, 95% CI [+0.0591, +0.0787] |
| controls | **PASS** | isolation 0.0e+00/0.0e+00, contaminated detected True, saturation 0.083 |
| cross g to M | **PASS** | Δ = +0.000e+00, 90% CI [+0.000e+00, +0.000e+00], ratio ≤ 0.0000 |
| cross m to N | **PASS** | Δ = +0.000e+00, 90% CI [+0.000e+00, +0.000e+00], ratio ≤ 0.0000 |
| degree profile equivalence g | **PASS** | equivalence family, Holm-corrected |
| degree profile equivalence m | **PASS** | equivalence family, Holm-corrected |
| memory main effect | **PASS** | Δ = +0.3711, 95% CI [+0.3535, +0.3899] |
| nonlinear main effect | **PASS** | Δ = +0.9929, 95% CI [+0.9888, +0.9968] |

## Response Jacobian

```
            d/dm        d/dg
  M      +0.371079   +0.000000
  N      +0.000000   +0.992913
```

off-diagonal / diagonal mass = `0`; response angle = `90.00°`


## Encoder audit

- n=1: local readout degrees [1], global readout degrees [1], globally affine = **True**
- n=2: local readout degrees [1], global readout degrees [0, 2], globally affine = **False**
- n=3: local readout degrees [1], global readout degrees [1, 3], globally affine = **False**
- n=4: local readout degrees [1], global readout degrees [0, 2, 4], globally affine = **False**

((I+uZ)/2)^(x)n is NOT globally linear: the n-body observable carries degrees up to n. It is affine only at the LOCAL readout, and the n copies are counted as a nonlinear resource.


**Capability boundary.** affine single-copy injection + CPTP chain => the memory state is MULTILINEAR in the past inputs, so any target with degree >= 2 at a strictly positive delay is unreachable. N_long is therefore built from targets whose degree sits at delay 0.


## Stress test

13/13 perturbed variants pass every gate on development seeds the search never used; robust = **True**


## Confirmation integrity

- rows 375/375, missing 0, duplicates 0, wrong hash 0
- complete = **True**
- sequential attempts 1/5, alpha spent 0.01
- confirmation seeds: {'arch': [900, 901, 902, 903, 904], 'input': [9000, 9001, 9002]}


## Nonlinear-memory benchmark (Sunada)

- mean over REACHABLE cells: 0.5655666266431186
- mean over BOUNDARY cells (provably unreachable): 0.24330152557022316
- max over boundary cells: 0.9996759494241776


## Figures

- 03_response_surfaces.png: M, N and N_long response surfaces
- 04_main_and_cross_effects.png: Main effects and cross-effect equivalence
- 05_response_jacobian.png: Response Jacobian
- 09_gate_decisions.png: Gate decision table
- 08_nonlinear_memory.png: Nonlinear-memory benchmark
- 07_search_history.png: Architecture-search history
