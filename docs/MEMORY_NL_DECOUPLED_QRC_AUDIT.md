# Audit: existing CPSR/QRC codebase, prior to building DQRC

Branch `qiskit`. No files were modified to produce this audit. Sources: `git log
--diff-filter=A` (per-file first-add commit), direct reads of the `.py` modules, and
`nbconvert`/JSON-parsed reads of the notebooks (raw `.ipynb` JSON was never read
directly — it is multi-MB with embedded base64 image outputs).

## Timeline (first-add commit per file)

- **2026-08-21 "Initial Commit"** — `ID_CPSR_Qiskit_IBM.ipynb`, `idcpsr.py`,
  `idcpsr_qiskit.py`: the original Cirq/Willow→Qiskit port (ID-CPSR / QND-RC / sewing
  lineage), documented in `docs/PORT_SUMMARY.md`.
- **2026-09-04 "Made a working Quantum Reservoir"** — `qrc_qiskit.py`,
  `CPSR_Project_IBM_Qiskit_reviewed.ipynb`, `CPSR_Project_IBM_Qiskit_reviewed_2.ipynb`:
  the code-review-response fixes documented in `docs/CPSR_Code_Review_Response.md`.
- **2026-09-10 "Operated at the EOC"** — `1_QR_Qiskit.ipynb` … `4_QR_MixedSYK_Qiskit.ipynb`:
  the current mixed-SYK edge-of-chaos (EOC) lineage.
- **Untracked, newest** — `mixed_syk_core.py`, `eoc_config.py`, `jerbi_shadow.py`,
  `shadow_measurements.py`, `5_QR_MixedSYK_JerbiShadow_Qiskit.ipynb`, test files: the
  notebook-5 Choi-shadow QELM readout layer (see memory `mixed-syk-shadow-project`).

## 1. Finalized implementation

**`4_QR_MixedSYK_Qiskit.ipynb`** is the finalized/current QRC notebook.
`mixed_syk_core.py` and `qrc_qiskit.py` are its extracted, importable modules —
`mixed_syk_core.py:2-16` states its functions are copied **verbatim** from notebook
4's Sections 0c/0d/2, and `code/test_regression_notebook4.py` regression-checks that
claim bit-for-bit. `eoc_config.py:6` states notebook 4 "is the SOURCE OF TRUTH for the
mixed-SYK reservoir and its established EOC operating point." `mixed_syk_core.py:35-39`
imports the shared harness (`ReservoirConfig`, `chrono_split`, `nrmse`,
`select_and_eval_ridge`, `memory_capacity`, task functions, `delay_taps`,
`random_input`, `DEFAULT_ALPHAS`) from `qrc_qiskit.py` rather than duplicating it —
i.e. `qrc_qiskit.py` is the shared numerical harness both notebook 4 and notebook 5
build on. Notebook 4 itself is self-contained (pastes the same code inline; does not
`import qrc_qiskit`) — the `.py` modules exist specifically so new code (notebook 5,
and now DQRC) can reuse without copy-paste.

`idcpsr.py`/`idcpsr_qiskit.py`/`ID_CPSR_Qiskit_IBM.ipynb` (2026-08-21) and
`CPSR_Project_IBM_Qiskit_reviewed[_2].ipynb` (2026-09-04) are earlier/parallel
lineages, **superseded for EOC purposes**: `qrc_qiskit.py:6-9` states that lineage
used "ONE fixed coupling strength `g` … no 'edge of chaos' / critical-phase scan."
`reviewed.ipynb` vs `reviewed_2.ipynb` were not diffed line-by-line; not relevant to
DQRC. Notebooks 1–3 are the SYK2-only/SYK4-only precursors to notebook 4's mixed
SYK2+SYK4 layer; `qrc_qiskit.py`'s `reservoir_layer`/`ReservoirConfig` is notebook 1's
Section 0a/0b, reused verbatim in notebook 4.

**Baseline for DQRC is `mixed_syk_core.py` + `qrc_qiskit.py` + `eoc_config.py`, never
the old notebooks directly.** Per the task instructions, the old finalized notebooks
(`Kobayashi_QRC_EdgeOfChaos_numpy.ipynb`, `CPSR_Project_IBM_Qiskit*.ipynb`,
`idcpsr*.py`, `ID_CPSR_Qiskit_IBM.ipynb`) are treated as immutable and are **not**
modified by this project.

## 2. Input encoding

Two different conventions exist — DQRC must pick one deliberately:

1. **Qiskit gate-based (current, notebook 4 / `qrc_qiskit.py` / `mixed_syk_core.py`)**:
   `u_t ∈ [0,1]` → `qc.ry(np.pi * float(u_t), qubit)` after `qc.reset(qubit)`.
   - Recurrent architecture: only `cfg.input_qubit` (default qubit 0) is reset+encoded
     each step (`qrc_qiskit.py:154-156` `build_trajectory_circuit`,
     `mixed_syk_core.py:166-168` `build_trajectory_circuit_mixed`).
   - QELM (memoryless, window-encoded) architecture: **all** N qubits reset each step,
     a sliding window of the last `min(window_size, N)` inputs mapped 1:1 onto qubits
     — `qrc_qiskit.py:709-726` (`build_qelm_circuit`), `mixed_syk_core.py:249-261`
     (`build_qelm_circuit_mixed`). This is what `SCIENCE_CONFIG` uses
     (`WINDOW_SIZE_QELM = N_MIX = 6`, `eoc_config.py:212`).
2. **Numpy continuous-Hamiltonian reference** (`Kobayashi_QRC_EdgeOfChaos_numpy.ipynb`,
   `encode_input`): `u_t` encoded as a density matrix `ψ = (√(1-u), √u)`,
   `ρ_in = |ψ⟩⟨ψ|`, injected via partial trace + kron with continuous-time
   Schrödinger evolution. Architecturally distinct from the Ry-angle convention —
   **not** assumed equivalent.

**DQRC uses the Qiskit Ry(π·u_t) convention** for consistency with the finalized
mixed-SYK lineage, applied to the memory-register input qubit(s) only (never a
window-encoded QELM style, since DQRC's whole point is genuine persistent quantum
memory, not classical-window memory — see §5).

## 3. EOC parameter and diagnosis

Parameter is **kappa** (SYK4↔SYK2 interpolation), not a bare scalar `g`:
`mixed_syk_core.py:200-209` (`kappa_to_gJ`): `g = G_MAX·κ/(1+κ)`,
`J = J_MAX/(1+κ)` (rational, not linear, interpolation).

Established value: `NOTEBOOK4_QELM_EOC_KAPPA = 0.960` (`eoc_config.py:235`), with
`G_MAX_QELM = J_MAX_QELM = 0.6` → `(g, J) = (0.2939, 0.3061)`. **Independently
verified**: this exact string — `"QELM temporal-edge scan fixed at the Section 6
optimum: kappa=0.960 (g=0.2939, J=0.3061)"` — is the stored, already-executed output
of a cell in `4_QR_MixedSYK_Qiskit.ipynb` (parsed directly from notebook JSON, no
re-execution). `eoc_config.py:238-251` (`verify_science_kappa_gJ`) re-checks
`kappa_to_gJ` reproduces `(0.2939, 0.3061)` on every import as a cheap drift guard,
without re-running the expensive scan.

Two **separate** diagnostics exist in notebook 4/`mixed_syk_core.py`, and the
"established point" answers them differently:

- **Independent chaos diagnostics** (task-agnostic): `operator_entanglement`
  (`mixed_syk_core.py:350-359`), `level_spacing_ratio` / ⟨r⟩ vs.
  Poisson/COE/CUE references (`mixed_syk_core.py:362-370`,
  `sample_reference_r_statistics`), OTOC-based `otoc_curve`/`scrambling_time`
  (`mixed_syk_core.py:402-426`). Computed across the full `KAPPA_GRID` in
  `coupling_scan_mixed_at_scale`; shows a genuine GOE/CUE→Poisson crossover in ⟨r⟩.
- **The actual kappa=0.960 operating point is selected by task performance**, not the
  spectral crossover: `qelm_scan_mixed` computes NARMA2 NRMSE across
  `KAPPA_GRID = geomspace(0.02, 100, 12)` averaged over `N_REAL_QELM=3` seeds, and
  `qelm_narma_best_idx = argmin(...)` fixes kappa. Notebook 4 is explicit that the
  spectral crossover is "the physics precondition … not a restatement of" the
  performance optimum, and reports both rather than conflating them.

**Implication for DQRC (§6 below)**: kappa=0.960 is not a leakage-free, purely
spectral EOC criterion — it is a task-performance optimum on notebook-4-internal data.
DQRC's processor subsystem must **independently reproduce an EOC diagnostic** (per
task instructions Part 3) rather than blindly trusting 0.960 as physics-only truth,
and must use `eoc_config.py`'s frozen value only as a reference/starting point, never
re-derive it from DQRC's own benchmark data (that would be new leakage).

## 4. Current readout features

Both "exact" (Aer `save_expectation_value`, no shot noise) in
notebook 4/`mixed_syk_core.py`:

- Recurrent architecture: `feature_ops_mem_all(N, input_qubit, max_weight=3)`
  (`mixed_syk_core.py:112-135`) — full weight ≤ 3 Pauli-string readout on memory
  qubits only (excludes input qubit).
- QELM architecture (`SCIENCE_CONFIG`): `feature_ops_all_general(N, max_weight=5)`
  (`mixed_syk_core.py:217-229`) over all N qubits, `MAX_WEIGHT_QELM=5`
  (`eoc_config.py:211`).
- No shadow/noise variant inside `mixed_syk_core.py`/`qrc_qiskit.py` — both are
  explicitly exact-only (`qrc_qiskit.py:114-123`: "no shot noise and no
  classical-shadow estimator here"). Shadow readout is a separate layer:
  `shadow_measurements.py` + `jerbi_shadow.py` (notebook 5's Choi-flipped shadow
  estimator) and, in the older lineage, `idcpsr_qiskit.py:94-160`
  (`shadow_features_qiskit`, `monolithic_shadow_features`).

## 5. Current memory mechanism

Both exist, kept explicitly separate:

- **Genuine recurrent memory**: only the input qubit is reset each step; the rest of
  the register ("the memory register", `qrc_qiskit.py:19,142`) is never reset, so
  `|ψ_t⟩ = U(u_t)|ψ_{t-1}⟩` genuinely carries forward quantum state. No formal
  "quantum memory register" class exists — it's just "qubits other than
  `input_qubit`."
- **Memoryless QELM (window-encoded)**: "memory" is purely a classical sliding window
  of recent inputs re-encoded onto qubits each step, all qubits reset.
  `qrc_qiskit.py:624-634` is explicit: "Any nonzero memory-capacity score from it
  reflects the classical window, NOT information retained by the quantum state."
- **`SCIENCE_CONFIG` (used by notebook 5) is the QELM/memoryless variant** — the
  currently-loaded EOC operating point has **no genuine quantum memory**. This is the
  single most important audit finding for DQRC: DQRC cannot inherit `SCIENCE_CONFIG`
  wholesale and claim quantum memory: it must build the persistent recurrent variant
  (`build_trajectory_circuit_mixed`-style, never-reset memory register) as its memory
  subsystem M, per task instructions Part 2.
- `delay_taps` — **classical-only control baseline**, explicitly labeled:
  `qrc_qiskit.py:448-456` docstring: "Classical-only control features: the raw last
  m+1 inputs, no quantum processing at all. Used as a baseline so a quantum-reservoir
  NRMSE win can be checked against 'did the delayed input alone already explain
  this.'" Used as a *separate* control `Xc` compared against quantum NRMSE in
  `run_benchmark`, never concatenated into the quantum feature vector, in this
  lineage.
  - Contrast: the **older** lineage (`idcpsr.py:458,466`, `idcpsr_qiskit.py:160,445,450`)
    **does** `np.concatenate([S, delay_taps(u,m)], axis=1)` — a documented
    quantum+classical-delay hybrid ("ID-QND-RC", `docs/PORT_SUMMARY.md` §5). This
    lineage is not reused by DQRC's primary quantum-memory claim (per task
    instructions Part 2's explicit prohibition); it may only reappear as the labeled
    `classical_delay_qrc` control (task instructions Part 2, three explicit controls).

## 6. Tasks and methodology (current)

Tasks: **k-Pauli** (`task_kpauli`, k∈{1,2,3}), **NARMA2** (`task_narma2`, cites
Appeltant et al. 2011, `u_scale=0.5`), **short-term memory capacity**
(`memory_capacity`, Jaeger MC). **No NARMA10, Mackey-Glass, or Lorenz** anywhere in
the codebase — these are new for DQRC (task instructions Part 12).

Split: strictly **chronological**, never random —
`chrono_split(T, washout, n_val, n_test, gap)` (`qrc_qiskit.py:463-477`) with a guard
gap ≥ longest task lag on both sides of every boundary; alpha selection on validation
only, test touched once (`select_and_eval_ridge`). Direct fix for
`docs/CPSR_Code_Review_Response.md` §8 (train/test leakage).

Seeds: **small**, not large multi-seed — `N_DIAG_SEEDS=15` (spectral diagnostics),
`N_REAL_QELM=3` (kappa-selection scan), `N_REAL_PERF=4` (recurrent performance scan),
`N_REAL_TEMPORAL=3`. `eoc_config.build_science_config()` uses a **single concrete
realization** (`term_seed=0`, one of the 3 averaged in `qelm_scan_mixed`) — explicitly
flagged in `eoc_config.py:268` as one seed, not an average. Task instructions Part 13
require ≥10 seeds for DQRC's final results — this is a real gap versus the existing
baseline that DQRC's own experiments must close for its own claims (not retroactively
fix notebook 4).

## 7. Where classical delay taps are used / how labeled

See §5. Current lineage: labeled classical-only control, never fused into quantum
features. Older lineage: fused into features, documented as a hybrid design. DQRC
follows task instructions Part 2's rule — `delay_taps` may exist only as an explicit,
separately-labeled `classical_delay_qrc` control, never silently inside the
`true_quantum_memory_dqrc` feature vector.

## 8. Terminology check: IDQNN / sewing / shadow / Huang

- **"IDQNN"**: **zero hits** in `code/`. Not a term used anywhere in this repo. Do not
  cite it as already-used; any IDQNN claim in DQRC starts from nothing and must meet
  the task instructions' Part 5 bar from scratch (or be renamed
  `ideal_depth_compressed_control` / left as an isolated
  `experimental/idqnn_memory_prototype.py`).
- **"sewing"**: Legitimately implemented in the old lineage, not just a label.
  `idcpsr.py:207-217` (`controlled_basis_copy`) builds a real basis-rotation + CNOT
  coherent-copy-onto-ancilla operator; `idcpsr.py:333-341`
  (`sewn_readout_channel_check`) **numerically verifies**
  `Tr_ancilla[CNOT-copy ∘ discard] == local dephasing channel`.
  `idcpsr_qiskit.py:227-313` reimplements it as real circuits with a cross-check, and
  `idcpsr_qiskit.py:804-880` runs it on `FakeTorino` hardware noise, confirming the
  tapped qubit's Z-marginal survives within shot noise. This is a real
  controlled-copy-then-discard = local-dephasing identity, verified both analytically
  and under a hardware noise model — **not** an inflated label for a bare CNOT. DQRC's
  own "sewing"/"stitching" mechanism (Part 4) is architecturally different
  (inter-block coupling between processor blocks, not a readout-preserving ancilla
  copy) and per task instructions must be called **"ancilla-mediated stitching"**
  unless it actually implements Huang's local-inversion divide-and-conquer
  construction — which nothing in this repo currently does.
- **"instantaneous depth" (the "ID" in ID-CPSR)**: closer to over-claiming.
  `idcpsr.py:419-444` (`noisy_reservoir_features`): `mode='deep'` applies `U1`
  `D_eff` times with `D_eff` separate depolarizing-noise applications; `mode='id'`
  applies `matrix_power(U1, D_eff)` **once** with exactly **one** depolarizing-noise
  application. The claimed noise-robustness advantage is structurally guaranteed by
  construction (noise-application count, not a hardware-realistic depth/fidelity
  trade-off for actually compiling `U^D`). This is exactly the
  `ideal_depth_compressed_control` pattern the task instructions warn against calling
  "IDQNN" — confirms the instructions' concern is well-founded against this repo's own
  prior art, and DQRC must not repeat it under a new name.
- **"shadow"**: Real, carefully-scoped local-Pauli classical-shadow implementation.
  `shadow_measurements.py` self-limits its own claims (docstring: "only … i.i.d.
  uniform local-Pauli … is implemented in this project" — biased sampling,
  derandomized shadows, observable-aware allocation, light-cone truncation are
  explicitly listed as **not** implemented). `idcpsr_qiskit.py:94-97` also correctly
  cites the Huang-Kueng-Preskill variance bound. No instances of "shadow" applied to
  something that isn't an actual randomized-measurement/inversion-estimator protocol.
  This matches memory `mixed-syk-shadow-project`'s finding that notebook 5's shadow
  readout is a genuine (if high-variance at N=6) classical-shadow estimator, not
  Gaussian-noise-dressed exact values.
- **"Huang"** citations throughout are correctly attributed (classical shadows vs. the
  2025 generative-quantum-advantage/sewing paper) — no citation padding found.

## 9. Reusable functions/modules (import, do not reimplement)

- **Circuit builders**: `mixed_syk_core.mixed_layer`,
  `mixed_syk_core.build_trajectory_circuit_mixed` (recurrent — the basis for DQRC's
  memory register), `mixed_syk_core.build_qelm_circuit_mixed` (memoryless/window,
  reference only), `mixed_syk_core.run_reservoir_mixed` /
  `run_reservoir_qelm_mixed`.
- **SYK sampling with the support-count fix applied**:
  `eoc_config.sample_syk4_supports` — use this, not `mixed_syk_core.sample_syk4_terms`
  + `sample_syk4_couplings` directly (avoids the N=4 truncation bug documented in
  `eoc_config.py:46-73`).
- **EOC config/point**: `eoc_config.build_science_config()` (loaded kappa=0.960
  operating point, single seed) and `eoc_config.build_validation_config()` (small-N
  correctness sandbox); `eoc_config.ReservoirParams.fingerprint()` for asserting a
  reservoir wasn't mutated downstream.
- **Pauli-expectation readout**: `mixed_syk_core.feature_ops_mem_all` (recurrent),
  `mixed_syk_core.feature_ops_all_general` (arbitrary max_weight); lower-level
  `qrc_qiskit.feature_ops` / `feature_ops_all`.
- **EOC/chaos diagnostics** (use these for DQRC's own independent EOC reproduction,
  §3): `mixed_syk_core.operator_entanglement`, `mixed_syk_core.level_spacing_ratio`,
  `mixed_syk_core.sample_reference_r_statistics`, `mixed_syk_core.otoc_curve` /
  `scrambling_time`, `mixed_syk_core.step_unitary_mixed`.
- **Ridge/task-eval harness**: `qrc_qiskit.chrono_split`,
  `qrc_qiskit.select_and_eval_ridge`, `qrc_qiskit.memory_capacity`, `qrc_qiskit.nrmse`,
  `qrc_qiskit.DEFAULT_ALPHAS`, `qrc_qiskit.task_kpauli`/`task_narma2`/`delay_target`/
  `delay_taps`/`random_input` — all re-exported through `mixed_syk_core`, so
  `import mixed_syk_core as msc` gets both reservoir and harness.
- **Shadow readout primitives**: `shadow_measurements.sample_shadow_exact`,
  `single_qubit_shadow_estimate`, `estimate_pauli_expectation`,
  `precompute_b_factors`, `median_of_means`, `theoretical_shadow_norm_sq`,
  `build_shadow_measurement_circuits`/`run_shadow_ibm`; `jerbi_shadow.ChoiShadowDeployment`.
- **Simulator/backend selection**: `qrc_qiskit.make_simulator` — **must** be used with
  `method='density_matrix'` for any reset-heavy recurrent trajectory circuit; see
  memory `aer-statevector-reset-bug` (`method='statevector'` gives silently wrong,
  seed-dependent results for exactly this circuit pattern — DQRC's memory-register
  trajectory circuits are precisely this pattern and are at risk). `qrc_qiskit.get_ibm_service`
  (env-var credentials only).
- **Do not reimplement from `idcpsr.py`/`idcpsr_qiskit.py`** unless specifically
  reusing the verified sewing/dephasing-channel machinery — that lineage's persistent
  reservoir features are noted in `docs/PORT_SUMMARY.md` ("Note on qmem baseline") to
  decay toward a fixed point and generalize poorly; not a ready building block for
  DQRC's memory subsystem.

## 10. Critical finding carried into DQRC design

**`AerSimulator(method='statevector')` is wrong for reset-heavy trajectory circuits**
(many mid-circuit resets in one long circuit) — confirmed bug, see memory
`aer-statevector-reset-bug`. DQRC's memory register (Part 2, never-reset memory
qubits + repeatedly-reset input qubit, run as one long trajectory circuit) is exactly
this pattern. **All DQRC memory-subsystem simulation must use
`method='density_matrix'`**, or the reset-free-per-window `Statevector` pattern used
by `jerbi_shadow.direct_qelm_features` where the architecture allows it (it does not
for genuine recurrent memory, which is the whole point of DQRC's memory subsystem —
so `density_matrix` is the required choice there).

## Summary: what DQRC must NOT do (per this audit + task instructions)

1. Must not inherit `SCIENCE_CONFIG`'s QELM window-encoding as "quantum memory" — it
   is classical-window memory by the codebase's own admission (§5).
2. Must not treat kappa=0.960 as a leakage-free physics-only EOC point — it was
   selected by task performance on notebook-4-internal data (§3, §6); DQRC
   independently reproduces an EOC diagnostic rather than trusting the number blindly.
3. Must not run recurrent memory-register circuits under `method='statevector'` (§10).
4. Must not call ancilla-mediated inter-block coupling "Huang sewing" unless it
   implements the actual local-inversion construction — the repo's own `idcpsr.py`
   sewing is a good positive example of what a *verified* claim looks like (§8),
   and DQRC's stitching mechanism does not currently rise to that bar.
5. Must not call `matrix_power(U,D)` + one noise channel "IDQNN" — the repo's own
   `idcpsr.py` `mode='id'` is exactly this pattern and is flagged here as the
   `ideal_depth_compressed_control` pattern, not IDQNN (§8).
6. Must not concatenate classical `delay_taps` into the primary quantum-memory feature
   vector — only as an explicitly separate `classical_delay_qrc` control (§5, §7).
7. Must use ≥10 seeds for its own final claims — the existing baseline's own
   `N_REAL_QELM=3` / single `term_seed=0` is not sufficient precedent to lean on (§6).
