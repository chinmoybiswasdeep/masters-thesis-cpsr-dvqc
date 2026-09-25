# V7 development log

This append-only log records every materially different V7 candidate and every
gate outcome. Thresholds are frozen in `results/v7/preregistered_v7_protocol.json`.

## 2026-09-24 — pre-development freeze

- Verified branch `qiskit`, clean baseline commit
  `e893f65c8a6a52cf44efb8835774833106ecda7f`.
- Audited V6.7 without modifying any V6 artifact; see `V7_V6_7_AUDIT.md`.
- Froze four development and four internal-validation seed triples.
- Generated six confirmation seed triples outside the worktree under `.git/`.
  Development code has no API or path that reads them. Only their salted
  commitment is placed in the preregistration. The confirmation runner may
  reveal them only after a valid frozen-candidate manifest exists.
- Environment: Python 3.14.4, Qiskit 2.5.2, Qiskit Aer 0.17.2.

Candidate entries follow below; rejected candidates remain permanently.

## V7.0 — rejected before scientific screening

- Design: ten-qubit density-matrix circuit with two linear-memory modes, two
  nonlinear-memory modes, three processor harmonics and one interaction probe.
- Result: rejected for physical replay feasibility. One full 112-step point
  consumed more than eight CPU-minutes without completion because a ten-qubit
  density matrix was replayed for every prefix.
- Modification: remove one redundant nonlinear-memory mode and reuse its reset
  input ancilla. This changes the state space and is therefore V7.1, not a
  silent optimization. No gate result or seed selection was involved.

## V7.1 — active candidate

- Eight logical qubits: two memory modes, one nonlinear-memory mode, three
  processor harmonics, one combined probe and one reset input ancilla.
- Same preregistered controls, observables, data and gates as V7.0.

### V7.1 result — rejected

- Full-length one-seed exact and 1,000-shot corner diagnostics completed.
- Exact processor nonlinearity was strong and scale-independent: standardized
  N changed from -0.024 at g=0 to 2.809 at g=1.
- Mandatory memory direction failed: standardized M changed from 0.123 at m=0
  to -0.050 at m=1 for g=0. At (m,g)=(1,1), M was -4.301, exposing feedback
  from the memory–nonlinear interaction into the later memory trajectory.
- Tail combined capacities were negative. V7.1 is permanently rejected and
  was not run on internal-validation or confirmation seeds.

## V7.2 — active candidate

- Replaced coherent weak-injection memory with a fully dephased partial-iSWAP
  collision channel. Its population recursion is
  `z' = r z + (1-r) u`, with two fixed maximum retentions 0.90 and 0.75 and
  `r = m r_max`. This makes m a genuine channel-timescale control.
- Removed the M–Q feedback interaction. The controlled M–probe interaction is
  retained; memory is its control and its Z population is invariant.

### V7.2 result — rejected

- One-seed exact corners preserved N exactly across m and M exactly across g.
- The memory direction improved (standardized M -0.692 to -0.284), but absolute
  tail capacities and every combined tail family remained below zero.
- Diagnosis: retention 0.90 left too little normalized weight at delays 9-12,
  the persistent registers began at Z=+1 and retained a large initialization
  transient, and the 28-column all-pairs readout overfit 56 training rows.

## V7.3 — active candidate

- Persistent registers now start in the maximally mixed state through an Aer
  phase-damping instruction, removing the initialization transient.
- Maximum retentions are 0.97 and 0.85; the former preregistered tail delays
  now have non-negligible channel weight.
- The operational readout uses 13 generic local/joint Z observables. Aer still
  returns all 28 observables for diagnostics, but unused all-pairs terms no
  longer consume readout degrees of freedom.

### V7.3 result — rejected

- Memory and nonlinearity main directions were correct and both cross-effects
  were structurally zero in exact mode.
- Tail gates still failed: a single exponential memory trace has one temporal
  degree of freedom and cannot resolve four separate delays 9-12. Adding ridge
  regularization did not restore the missing subspace.

## V7.4 — active candidate

- Three linear memory modes use maximum retentions 0.50, 0.85 and 0.97.
- Three nonlinear-memory modes receive re-upload orders 2, 3 and 4 at g=1,
  each through its own Aer-executed collision channel.
- Operational observables are ten local Z values, three generic memory-pair
  parities, three memory/current-processor parities and three memory/interaction
  parities. No benchmark delay pair is encoded.
- Exact mode remains density-matrix Aer. Finite-shot mode uses Aer's
  matrix-product-state method with real joint measurements; it is not an
  analytical fallback.

### V7.4 result — rejected

- Rejected on replay feasibility before scientific screening. An 11-qubit
  density-matrix corner consumed more than eight CPU-minutes.

## V7.5 — rejected

- Moved the nonlinear degrees into separate six-qubit measurement settings.
- Exact execution became practical, but three memory modes did not resolve the
  preregistered tail delays and the combined exact gates failed.

## V7.6 — rejected

- Expanded memory to twelve exponential modes through four circuit settings.
- Ridge recovered a strong memory metric (M=2.20 at m=1), but raw,
  standardized and whitened OLS failed (M=-42.3): clustered positive decay
  modes produced an ill-conditioned temporal basis.

## V7.7 — active candidate

- Uses twelve signed decay constants spanning -0.97 to +0.97. Negative
  retention is implemented by a physical X gate after the partial-iSWAP
  collision. Thresholds and readout methods are unchanged.

### V7.7 result — rejected

- Signed modes fixed conditioning: raw/whitened M rose from -0.69 to 3.42 and
  ridge M from -0.55 to 4.34, with exact zero g-to-M cross-effect.
- Delayed quadratic and cubic tails favored HH, but the shared interacted probe
  failed current-nonlinear × delayed-linear and delayed × delayed gates.

## V7.8 — active candidate

- Adds one qubit to separate an untouched current-nonlinear processor from a
  probe driven only by controlled rotations from the memory modes.

### V7.8 result — rejected

- Exact one-seed corner diagnostics passed the primary directions and exact
  structural cross-effects. Raw/whitened M changed -0.69 to 3.42; ridge M
  changed -0.55 to 4.34; N changed about -0.02 to 2.81 for every readout.
- Delayed quadratic and cubic tail means at HH were 0.136 and 0.034 under raw
  OLS, exceeding their other corners.
- Mandatory combined tail gates failed. Current-nonlinear × delayed-linear was
  -2.298 and delayed × delayed was -0.614 at HH under raw OLS. Short delays 1-3
  were near capacity 1 for the former, so the failure is specifically temporal
  resolution/generalization at delays 9-12.
- V7.8 was not run on finite shots, internal validation or confirmation. The
  committed confirmation bank remains sealed.
