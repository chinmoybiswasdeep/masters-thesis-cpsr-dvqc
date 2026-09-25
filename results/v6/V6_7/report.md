# V6.7 — confirmation report

Frozen sha256 `01b183eb83c98dfe0f9e3c837e8d88f8103e61b4289a4b86f608ffb77a3f8381` · attempt 1 · per-look α = 0.01 · 500 rows (20 untouched seeds × 5×5) · commit `fa6a0a0867`

**All mandatory gates: PASS**

| gate | result |
|---|---|
| 01_main_m_M | **PASS** |
| 02_main_g_N | **PASS** |
| 03_cross_m_N_equivalent | **PASS** |
| 04_cross_g_M_equivalent | **PASS** |
| 05_cross_main_ratio | **PASS** |
| 06_degree_profile_under_m | **PASS** |
| 07_memory_curve_under_g | **PASS** |
| 08_separation_ols_raw | **PASS** |
| 09_separation_ols_std | **PASS** |
| 10_separation_ridge | **PASS** |
| 11_interior_effects | **PASS** |
| 11b_section6_intrinsic_P | **PASS** |
| 12_combined_C1_C4_all_pass | **PASS** |
| 13_HH_best_every_class | **PASS** |
| 14_feature_counts_invariant | **PASS** |
| 15_no_encoder_leakage | **PASS** |
| 16_no_saturation | **PASS** |
| 17_future_sentinel_null | **PASS** |
| 18_unreachable_sentinel_null | **PASS** |
| 19_isolation_and_contaminated_controls | **PASS** |
| 20_invalid_controls_detected | **PASS** |
| 21_seed_reproducibility | **PASS** |
| 22_25_robustness | **PASS** |
| 26_unit_tests_pass | **PASS** |

## Main and cross effects

| readout | ΔmM (LB) | ΔgN (LB) | interior M (LB) | interior N (LB) | m→N TOST | g→M TOST | ratio UB m→N / g→M |
|---|---|---|---|---|---|---|---|
| ridge | +0.508 (+0.502) | +0.309 (+0.301) | +0.385 (+0.380) | +0.306 (+0.297) | [+0.0000, +0.0000] | [+0.0000, +0.0000] | +0.0000 / +0.0000 |
| ols_std | +0.508 (+0.502) | +0.309 (+0.301) | +0.386 (+0.380) | +0.306 (+0.297) | [+0.0000, +0.0000] | [+0.0000, +0.0000] | +0.0000 / +0.0000 |
| ols_raw | +0.508 (+0.502) | +0.309 (+0.301) | +0.386 (+0.380) | +0.306 (+0.297) | [+0.0000, +0.0000] | [+0.0000, +0.0000] | +0.0000 / +0.0000 |

## Combined nonlinear-memory classes (operational readout ALL)

| class | readout | LL | LH | HL | HH | min HH-contrast LB | null99 | seeds HH-best | sat. frac | result |
|---|---|---|---|---|---|---|---|---|---|---|
| C1  P2(u_t)·P1(u_{t-τ}) | ridge | -0.004 | +0.009 | +0.055 | +0.145 | +0.086 | +0.006 | 1.00 | 0.00 | **PASS** |
| C1  P2(u_t)·P1(u_{t-τ}) | ols_std | -0.004 | +0.009 | +0.055 | +0.145 | +0.086 | +0.006 | 1.00 | 0.00 | **PASS** |
| C1  P2(u_t)·P1(u_{t-τ}) | ols_raw | -0.004 | +0.009 | +0.055 | +0.145 | +0.086 | +0.006 | 1.00 | 0.00 | **PASS** |
| C2  P2(u_{t-τ}) | ridge | -0.008 | +0.036 | -0.006 | +0.150 | +0.110 | +0.006 | 1.00 | 0.00 | **PASS** |
| C2  P2(u_{t-τ}) | ols_std | -0.009 | +0.035 | -0.006 | +0.150 | +0.111 | +0.006 | 1.00 | 0.00 | **PASS** |
| C2  P2(u_{t-τ}) | ols_raw | -0.009 | +0.035 | -0.006 | +0.150 | +0.111 | +0.006 | 1.00 | 0.00 | **PASS** |
| C3  P1(u_{t-τ1})·P1(u_{t-τ2}) | ridge | -0.010 | -0.006 | +0.023 | +0.088 | +0.060 | +0.006 | 1.00 | 0.00 | **PASS** |
| C3  P1(u_{t-τ1})·P1(u_{t-τ2}) | ols_std | -0.011 | -0.007 | +0.023 | +0.088 | +0.061 | +0.006 | 1.00 | 0.00 | **PASS** |
| C3  P1(u_{t-τ1})·P1(u_{t-τ2}) | ols_raw | -0.011 | -0.007 | +0.023 | +0.088 | +0.061 | +0.006 | 1.00 | 0.00 | **PASS** |
| C4  P3(u_{t-τ}) | ridge | -0.006 | +0.019 | +0.001 | +0.089 | +0.064 | +0.006 | 1.00 | 0.00 | **PASS** |
| C4  P3(u_{t-τ}) | ols_std | -0.006 | +0.019 | +0.001 | +0.089 | +0.064 | +0.006 | 1.00 | 0.00 | **PASS** |
| C4  P3(u_{t-τ}) | ols_raw | -0.006 | +0.019 | +0.001 | +0.089 | +0.064 | +0.006 | 1.00 | 0.00 | **PASS** |

## Saturation and sentinels

- per-metric saturation: {"M|ols_raw": {"max_metric_value": 0.5059262748790337, "ok": true, "worst_constituent_saturated_fraction": 0.0}, "M|ols_std": {"max_metric_value": 0.5059262748790337, "ok": true, "worst_constituent_saturated_fraction": 0.0}, "M|ridge": {"max_metric_value": 0.5059361343106075, "ok": true, "worst_constituent_saturated_fraction": 0.0}, "N|ols_raw": {"max_metric_value": 0.3057456881254883, "ok": true, "worst_constituent_saturated_fraction": 0.0}, "N|ols_std": {"max_metric_value": 0.3057456881254883, "ok": true, "worst_constituent_saturated_fraction": 0.0}, "N|ridge": {"max_metric_value": 0.3057448506341503, "ok": true, "worst_constituent_saturated_fraction": 0.0}, "ok": true}
- pooled saturated fraction (max over rows): 0.008
- sentinels: {"SENTINEL_future": {"max_excess_over_null99": 0.010610732262236537, "ok": true}, "SENTINEL_unreachable": {"max_excess_over_null99": 0.008289791835155654, "ok": true}, "ok": true}
- encoder leakage: {"diagnostic_max_member_excess_at_g0": 0.015149167067152353, "empirical_class_score_max_excess_at_g0": -0.0026299900681295493, "exact": {"affine_at_g0": {"m0.25": {"affine": true, "max_superposition_violation": 6.661338147750939e-16, "tolerance": 1e-10}, "m0.5": {"affine": true, "max_superposition_violation": 4.440892098500626e-16, "tolerance": 1e-10}, "m1.0": {"affine": true, "max_superposition_violation": 2.220446049250313e-16, "tolerance": 1e-10}}, "passed": true}, "passed": true}

## Quantum-specific claim (separate)

- ALL − CLS_M (HH, ridge): -0.004 (LB -0.004)
- ALL − CLS_F (HH, ridge): -0.245 (LB -0.250)
- classical simulability: yes: R and Q registers stay diagonal in the Z product basis (classical Markov chains over bitstrings); P is reset each step and read through a polynomial of u
- **General memory-nonlinearity separation succeeds, but the mechanism is classically reproducible.**

## Robustness (confirmation seeds)

| condition | worst effect | worst HH adv | isolated | result |
|---|---|---|---|---|
| baseline | +0.274 | +0.106 | True | **PASS** |
| chi_max_x0.9 | +0.225 | +0.084 | True | **PASS** |
| chi_max_x1.1 | +0.274 | +0.115 | True | **PASS** |
| degree_6 | +0.274 | +0.106 | True | **PASS** |
| delay_range_12 | +0.188 | +0.106 | True | **PASS** |
| double_training | +0.269 | +0.106 | True | **PASS** |
| exact_expectations_perturbed_controls | +0.207 | -0.001 | True | **FAIL** |
| float32_readout | +0.274 | +0.106 | True | **PASS** |
| half_training | +0.270 | +0.103 | True | **PASS** |
| ibm_noise_FakeTorino | +0.276 | +0.109 | True | **PASS** |
| longer_T3200 | +0.267 | +0.110 | True | **PASS** |
| pQ_max_x0.9 | +0.274 | +0.105 | True | **PASS** |
| pQ_max_x1.1 | +0.274 | +0.107 | True | **PASS** |
| pR_max_x0.9 | +0.238 | +0.105 | True | **PASS** |
| pR_max_x1.1 | +0.283 | +0.106 | True | **PASS** |
| perturbed_controls | +0.270 | +0.110 | True | **PASS** |
| phiQ_x0.9 | +0.274 | +0.106 | True | **PASS** |
| phiQ_x1.1 | +0.274 | +0.106 | True | **PASS** |
| phi_x0.9 | +0.243 | +0.085 | True | **PASS** |
| phi_x1.1 | +0.274 | +0.115 | True | **PASS** |
| ridge_0.05 | +0.274 | +0.106 | True | **PASS** |
| ridge_5.0 | +0.274 | +0.106 | True | **PASS** |
| shots_1000 | +0.269 | +0.093 | True | **PASS** |
| shots_10000 | +0.274 | +0.106 | True | **PASS** |
| shots_100000 | +0.244 | +0.107 | True | **PASS** |
| thetaQ_max_x0.9 | +0.274 | +0.106 | True | **PASS** |
| thetaQ_max_x1.1 | +0.274 | +0.106 | True | **PASS** |
| theta_max_x0.9 | +0.274 | +0.115 | True | **PASS** |
| theta_max_x1.1 | +0.274 | +0.092 | True | **PASS** |

## Figures

- `figures/fig01_primary_surfaces.png`
- `figures/fig02_class_surfaces.png`
- `figures/fig03_class_quadrants.png`
- `figures/fig04_degree_delay_HH.png`
- `figures/fig05_robustness.png`
- `figures/fig06_classical_baseline.png`
