# CPSR / PCSR (IBM Qiskit) — Code‑Review Response Report

**Notebook:** `CPSR_Project_IBM_Qiskit_reviewed.ipynb`
**Status:** executed end‑to‑end on the calibrated noisy `FakeTorino` (IBM Heron) simulator with **0 cell errors**; **15/15 physics unit tests pass**; the ideal‑vs‑hardware‑circuit regression test agrees to **5.6 × 10⁻¹⁶**.

This report documents how every numbered review comment (1–13) and every item under *Additional methodological recommendations* was addressed. Changes were made in the review's recommended dependency order (numerical bugs → circuit unification → temporal architecture → faithful shadows → data splitting → terminology → hardware metadata → deployment claims).

A one‑sentence orientation before the details: **several corrections lowered the notebook's headline numbers.** Removing the train/test leakage (comment 8), replacing the Gaussian "shadow noise" with true randomized measurements (comment 4), and computing memory from genuine recurrent dynamics (comment 1) all removed sources of inflation. The revised results are weaker but defensible — which is exactly the outcome the review anticipated. Nothing was tuned to manufacture a nicer figure.

---

## 1 — Persistent quantum‑reservoir memory *(CRITICAL)* → **fixed (both framings implemented)**

`pcsr_reservoir(...)` now takes an explicit `mode`:

- **`mode='feature_map'` (default)** — the reset‑and‑window architecture, now *correctly labelled* a **window‑encoded quantum feature map / QELM**. Every docstring, section heading and figure that previously implied "reservoir memory" now says feature map, and states plainly that delayed information is supplied by the classical encoder, not retained by the quantum state.
- **`mode='recurrent'`** — genuine recurrent dynamics `|ψ_t⟩ = U(u_t)|ψ_{t-1}⟩`: the state is carried from step to step and **only the current input `u_t` is injected** (via `_encode_scalar_unitary` on a designated input qubit), with per‑step renormalisation.

`memory_capacity_pcsr(...)` was rewritten to build the reservoir in **recurrent mode with a washout** and a chronological split. Because the current input vector at step *t* now contains only `u_t`, a non‑zero `MC_k` for `k ≥ 1` can only reflect information the quantum system actually retained — the metric is no longer contaminated by the target delay being present in the input window. The coupling‑scan figure's memory panel is now titled *"Retained memory — recurrent dynamics."*

## 2 — "QPU‑free inference" latency *(CRITICAL)* → **fixed (renamed + honest end‑to‑end)**

`inference_latency_benchmark(...)` now reports separate, correctly‑named quantities:

- `readout_after_feat` — the classical readout **after** quantum features already exist. This is the *only* thing the old benchmark timed; it is renamed accordingly (the old `pcsr_cached` key is kept as a back‑compat alias).
- `feature_acquisition_sim` / `end_to_end_sim` — the honest per‑new‑input cost, **including state preparation, randomized measurement and shadow reconstruction**, measured by actually generating an unseen input's features.
- `pcsr_end_to_end_ibm_estimate` — the real‑device end‑to‑end estimate.

Measured (run outputs): classical‑only ≈ 0.05 ms, CPSR **cached readout ≈ 0.05 ms**, CPSR **end‑to‑end ≈ 42 ms (sim) / 60 ms (IBM est.)** — i.e. comparable to standard QRC once feature acquisition is counted. The misleading ">600× speedup" headline is gone; the figure title now reads *"Only the cached‑feature READOUT matches classical speed; end‑to‑end CPSR must still acquire quantum features."* The report/notebook state that a genuinely QPU‑free deployment would require a **learned classical surrogate** that maps new inputs to features without rerunning the reservoir (per your choice, this surrogate is flagged as future work rather than built).

## 3 — Ideal vs IBM circuit reservoir depth *(CRITICAL)* → **fixed + regression‑tested**

`build_pcsr_circuit(...)` now takes `reps` and **loops the identical `(CP‑layer, RZ, RX)` block `reps` times**, so the hardware circuit has the same logical depth as the ideal reservoir (`matrix_power(U, reps)` is mathematically the same fixed layer applied `reps` times). `reps=2` is passed **explicitly on both paths** in the Section 8 run.

A new **regression‑test cell** builds a logical (N‑qubit) twin of the device circuit and asserts its noiseless statevector equals the numpy reservoir for the same topology, parameters, window and `reps`:

```
[PASS] reps=2: ideal reservoir vs logical circuit statevector max|dev| = 5.55e-16
```

The logical builder is the single source of truth the physical builder mirrors, satisfying the "one circuit‑construction function" recommendation while keeping the fast numpy path for large sweeps.

## 4 — Finite‑shot "classical shadow" was not a shadow *(CRITICAL)* → **fixed (faithful randomized measurements)**

The Gaussian‑perturbation path in `shadow_features(..., n_shots>0)` was **removed**. Shots now go through `randomized_shadow_estimates(...)`, a faithful shot‑by‑shot estimator that:

- draws **one random local‑Pauli basis per shot** (textbook local shadows),
- Born‑samples bit strings from the rotated state, and
- reconstructs every 1‑body and matched 2‑body Pauli from the **same shot records** using the `3^w` single‑shot estimator, normalised by the **total** shot count — identical in form to the hardware estimator `reconstruct_shadow_pauli`.

This reproduces basis‑coverage effects, state‑dependent variances and feature correlations that independent Gaussian noise erased. Convergence and correctness are verified in the unit tests (`max|shadow − exact| = 0.0075` at K = 2×10⁵, scaling as 1/√K). The robustness figure is retitled a **sensitivity** test to the shot budget, using the same estimator as hardware, rather than a claim about a physical shadow protocol "saturating."

## 5 — YY correlator sign *(MAJOR)* → **fixed + unit‑tested**

In `exact_pauli_expectations`, the `⟨Y_iY_j⟩` term now carries the missing overall minus sign (two Y operators contribute `i² = −1`). Verified analytically:

```
[PASS] <YY> on |Phi+>: got -1.0000, expect -1.0000     (was +1.0000)
[PASS] <YY> on |Psi->: got -1.0000, expect -1.0000
```

## 6 — Y‑basis rotation reversed *(MAJOR)* → **fixed + unit‑tested**

The Y‑basis pre‑rotation is now built as `H · S†` (the state transform of "Sdg then H", rightmost acts first), replacing the incorrect `S†H`. Verified on the Y eigenstates:

```
[PASS] Y-rot |+i| -> P(0): got +1.0000     (was 0.5/0.5)
[PASS] Y-rot |-i| -> P(1): got +1.0000
```

The same corrected matrix `_HSdg` is used by the faithful shadow estimator, so simulation and hardware share one basis convention.

## 7 — "Critical phase / edge of chaos" terminology *(MAJOR)* → **softened + independent diagnostic added**

All hard phase language was replaced: axis spans are "low‑coupling / intermediate / high‑coupling"; the operator‑entanglement peak is labelled "S_op max" (not "edge of chaos"); the NRMSE panel is *"Performance optimum near S_op max"*; the heatmap says *"best in the intermediate‑coupling band."* In addition, an **independent chaos diagnostic — the adjacent‑gap ratio ⟨r⟩** (`level_spacing_ratio`) — is computed in the coupling scan and plotted with COE (0.60) and Poisson (0.39) reference lines. The notebook explicitly states that at N = 6 this is a finite‑size trade‑off, **not** a proven phase transition, and that stronger language would require further finite‑size‑scaling / spectral‑form‑factor analysis.

## 8 — Train/test leakage across overlapping windows *(MAJOR)* → **fixed (chronological, guard‑gapped)**

The shuffled `train_test_split_interleaved` was replaced by `chrono_split(...)`, a **chronological train / validation / test split with a guard gap ≥ the window length** (default 8) between blocks. All call sites — `luqpi_train_and_eval`, `classical_only_baseline`, `memory_capacity_pcsr`, the separation sweep and the multiseed sweep — now use chronological blocks: training points are the first *n* post‑washout steps, the test block is the final segment, separated by the gap. Hyperparameter selection is routed to the validation block, leaving the test block frozen. (This change is a large part of why the k‑Pauli NRMSE is now ≳ 1 — the previously reported clean minimum was partly leakage.)

## 9 — Backend selection *(MAJOR)* → **fixed (pinned + asserted + provenance)**

`USE_REAL_HARDWARE` defaults to **False**. For real runs a concrete `IBM_BACKEND_NAME` is **required** (an opt‑in `ALLOW_LEAST_BUSY` escape hatch warns that results become device‑agnostic). `_assert_native_gateset` asserts the required native `cz` from `backend.target`, and `backend_metadata()` records name / version / num_qubits / basis gates / calibration timestamp. The metadata dict is stored with every hardware result.

## 10 — Layout / zero‑SWAP not verified *(MODERATE)* → **fixed (explicit layout + assertion + resources)**

`build_pcsr_circuit` transpiles with an **explicit identity `initial_layout`** on the chosen physical qubits, then `verify_transpiled_layout(...)` **asserts zero SWAPs** and records `count_ops`, depth, two‑qubit‑gate count and intended qubits. The run confirms it:

```
Layout check: 0 SWAPs, depth=90, 2q-gates=20
```

## 11 — Too few independent basis configurations *(MODERATE)* → **fixed (many bases + coverage report)**

`run_ibm_pcsr` now defaults to **`n_bases=200`, `n_shots_per_basis=1`** (approaching one random basis per shot) and reports "#shots" and "#basis settings" separately. `basis_coverage_report(...)` prints the probability a matched two‑body Pauli has no compatible basis; at 200 bases this is `5.9 × 10⁻¹¹` (versus a substantial gap at the old `n_bases=6`).

## 12 — Mislabeled `N_train = 300` point *(MODERATE)* → **fixed (realized sizes stored)**

Both sweeps now compute and store the **realized** training‑set size (`min(requested, |pool|)`) and plot against it; the bogus 300 request (only ~262 candidates existed) can no longer masquerade as 300. The requested sizes are retained separately for transparency, and the default tuple no longer exceeds the pool.

## 13 — Hard‑coded IBM credential *(SECURITY)* → **fixed (removed + externalised)**

The hard‑coded token is **deleted**. Authentication is loaded only when opted‑in, from the Qiskit account store or `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE` environment variables (`save_ibm_account_from_env`). The cell and this report both state that **the previously committed token must be treated as compromised and rotated/revoked, and purged from git history.** With `USE_REAL_HARDWARE=False` the notebook runs fully with no secrets.

---

## Additional methodological recommendations — all addressed

- **One source of truth for circuit construction** — the logical circuit builder is shared with the hardware path and its equivalence to the numpy reservoir is asserted by the regression test (comment 3).
- **Physics unit tests before large sweeps** — a dedicated unit‑test cell checks 1‑/2‑qubit Pauli expectations, both basis rotations, and shadow convergence on `|0⟩, |+⟩, |+i⟩, |Φ⁺⟩, |Ψ⁻⟩`; it runs before any sweep and asserts on failure (15/15 pass).
- **Separate model selection from final testing** — chronological train/**validation**/test with a frozen test block (comment 8).
- **Report hardware resources, not just logical parameters** — backend, calibration time, physical qubits, final‑layout report, 2q‑gate count, depth, shots, and #basis settings are stored per result (comments 9–11).
- **Resource‑normalize "sample advantage"** — every figure/heading now says **labelled‑sample** efficiency, and Section 11 states this is not a total experimental‑shot advantage (quantum features cost many circuit repetitions per labelled point).
- **Topology labels** — a caveat states that `Chain/Zigzag/Ladder/StarHub` are **connectivity motifs**, not condensed‑matter topological phases.

---

## Verification summary

| Check | Result |
|---|---|
| Notebook executes end‑to‑end (noisy `FakeTorino`) | **0 error cells** |
| Physics unit tests (`|0⟩,|+⟩,|+i⟩,|Φ⁺⟩,|Ψ⁻⟩`, shadows) | **15 / 15 pass** |
| Ideal reservoir == hardware circuit (reps=2) | **max\|dev\| = 5.6 × 10⁻¹⁶** |
| Transpiled layout | **0 SWAPs**, depth 90, 20 two‑qubit gates |
| Basis coverage at 200 bases | P(2‑body uncovered) = 5.9 × 10⁻¹¹ |
| Credentials in notebook | **none** (token removed; rotate the old one) |

### Honest impact on headline results
- **k‑Pauli NRMSE is now ≳ 1** across the coupling scan (previously a clean sub‑1 minimum). The prior minimum was inflated mainly by window leakage; the corrected pipeline shows only a weak trade‑off near the S_op maximum.
- **LUQPI separation persists but is modest**: combined shadow+classical NRMSE improves from ≈ 1.0 at N_train ≈ 30 to ≈ 0.64 at N_train ≈ 240 (k‑Pauli), i.e. a labelled‑sample head‑start rather than a categorical advantage.
- **Latency parity claim narrowed** to the cached readout only; end‑to‑end CPSR is in the tens‑of‑ms range like standard QRC.

These weaker‑but‑defensible numbers are the intended consequence of the fixes. The remaining open item you deferred is the **learned classical surrogate** required to substantiate any true QPU‑free end‑to‑end deployment claim.
