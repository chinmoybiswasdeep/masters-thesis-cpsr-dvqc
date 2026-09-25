# V8 development log

Append-only. V6 and V7 are preserved. V7 is an exploratory negative result and
is not evidence of confirmation success.

## Preregistration phase

- Baseline verified: branch `qiskit`, clean commit
  `f892ee321be7d40460228303a820ae70046fa3e8`.
- Implemented the complete executable gate engine and multiplicity-corrected
  paired bootstrap before architecture execution.
- Power analysis froze 16 development, 16 internal-validation and 24 encrypted
  confirmation seed triples.
- Confirmation seeds are committed as a tracked Fernet ciphertext. The key is
  outside the repository; clean clones can verify the commitment without being
  able to reveal the bank.
- No V8 quantum circuit has been executed and no V8 scientific result exists at
  this point.

## V8.0 — monolithic FIFO (rejected)

- Architecture: four 13-tap polynomial FIFO registers plus three processor
  qubits in a single 56-qubit MPS circuit.
- Execution diagnosis: even a four-timestep backend smoke test exceeded two
  CPU-minutes because every saved Pauli expectation traversed the full
  monolithic register. This cannot support the frozen 844-step, 25-point grid.
- Decision: rejected for execution feasibility before development screening;
  no scientific gate result was produced.

## V8.1 — modular FIFO (rejected)

- Split invariant memory, instantaneous processor, current–past, delayed
  nonlinear and delayed-pair routes into separately executed circuits.
- Added a generic all-pairs tap observable family (all 66 pairs, including
  preregistered held-out pairs), not only evaluated target pairs.
- Added first through fourth harmonic processor features so orthogonal cubic
  and quartic targets can be formed without inserting target polynomials.
- Actual Aer profiling after local-observable optimization required about 80
  seconds for only 20 timesteps across the ten routes. The frozen 844-step
  surface and seed banks would therefore be infeasible. Rejected before
  frozen-bank execution; no capacity or gate result was produced.

## V8.2 — parallel polynomial delay registers (active)

- Each generic observable is a one- or two-qubit Qiskit route. Raw inputs enter
  only as rotation angles; polynomial and product feature values are never
  constructed classically. Exact features use Aer save instructions, and shot
  features use measured count dictionaries with same-bitstring parity.
- The memory transport setting exposes progressively deeper physical delay
  routes; nonlinear control changes harmonic angle geometry; joint routes
  require both controls. All 66 unordered tap pairs are included.
- Amendment 001 corrects the HH-gate scope before any frozen-bank execution.
- The first eight-family screen was archived as invalid after a minimal
  reproduction showed that this installed Aer/Qiskit combination aliases
  mixed-width local expectation-save operators in one circuit to the last
  operator. The corrected screen uses full-width Pauli operators; V8.2 uses
  one consistent observable width per independent circuit. A regression test
  checks a nonzero joint expectation explicitly.

## V8.3 — generic-pair execution refinement (active)

- V8.2 passed a full 844-step four-corner screen on development seed 300000.
  Exact cross-invariance errors were both 0; nonlinear CKA was 0.4796 and the
  amplitude-rescaling control CKA was 1.0000. All four readouts gave capacity
  1.0 for every HH task/delay; the smallest combined tail advantage was 0.45.
- Before scaling beyond that one seed, reduced the all-pairs family from 66 to
  a still-generic 29 pairs: every adjacent pair, every offset-3 pair (including
  all three frozen held-out pairs), and the complete delay-1 star. No target,
  delay, readout, seed, gate or threshold changed. V8.3 must rerun seed 300000.

## V8.3 — rejection

- Rejected before a complete scientific checkpoint: direct classical delay
  addressing did not satisfy the stronger requirement that the final delay
  memory be shifted through time by Qiskit circuit operations. Four running
  diagnostics were terminated; atomic persistence left no partial matrices.

## V8.4 — physical modular FIFO (active)

- Restores a 13-cell Qiskit SWAP FIFO for every delayed route. Memory taps are
  never populated from a classically indexed past value.
- Current×past features are same-circuit processor/tap Pauli correlations.
  Delayed nonlinear×linear features are prepared by a two-ancilla CNOT joint
  encoding and the joint-bearing ancilla is SWAP-shifted through its FIFO.
  Delayed×delayed features are same-register tap Pauli correlations.
- Peak simultaneous width is 15 logical qubits. Exact routes are separated by
  observable width to avoid the reproduced Aer mixed-width save bug.

## V8.4 — rejection

- Full HH on seed 300000 passed every registered main task/delay (minimum
  capacity 0.9982), but a completeness audit found that held-out nonlinear
  combinations lacked a generic current-nonlinear×delayed-nonlinear route.
- Archived the HH result and stopped uncheckpointed comparison corners.

## V8.5 — held-out nonlinear completion (active)

- Adds three generic 15-qubit processor×nonlinear-FIFO correlation routes:
  current/delayed harmonic pairs (1,2), (2,3), and (3,2), at every delay.
- Frozen held-out targets use two combinations not used in architecture tuning:
  current P2×delayed P3 at delay 7 and current P3×delayed P2 at delay 10.
- A gate audit after the first V8.5 HH point, but before any multi-seed or
  response-surface inference, found that the registered comparison count 400
  did not cover the full family-by-delay-by-readout-by-mode envelope.
  Amendment 002 conservatively raises it to 5,000, retains all four readouts
  in each per-delay/HH gate, and adds an explicit held-out-target capacity
  gate.
- Before any finite-shot, precision, or fake-backend execution, Amendment 003
  freezes the second-seed derivation and the complete shot/robustness point
  matrix. This resolves an execution-matrix omission without changing a
  threshold or removing an experiment.
- Amendment 004 adds explicit missing-evidence-fails gates for complete 5x5
  ordering/interiors, conditioning, and doubled-training stability before
  those stages are executed. It only strengthens the decision system.

## V8.5 — rejection

- A deterministic exact-rerun audit varied only `seed_simulator` on the
  entangled `past2` route and found a maximum feature-array difference of
  0.0390625. Resetting the entangled control invokes stochastic pure-state MPS
  reset trajectories, so the completed HH result was not a valid exact
  expectation result.
- Archived the HH artifact and terminated all three uncheckpointed corners.
  No finite-shot, noisy, validation, or confirmation data had been accessed.

## V8.6 — reset-safe physical FIFO (active)

- Removed all entangled-reset `past*` routes. The delayed-nonlinear×linear
  family is evaluated in its registered same-delay form using the reset-safe
  generic linear and cubic FIFO harmonics; algebraically, P2(x)x lies in the
  span of P1(x) and P3(x). No target value is inserted into a feature.
- Every remaining reset acts on a separable discarded or processor qubit.
  Current×past and held-out nonlinear combinations remain same-circuit Pauli
  correlations between independently encoded physical qubits.
- The reset-safe V8.6 HH checkpoint on development seed 300000 completed from
  commit `1fc7bcc`: all 165-feature mandatory and held-out capacities were
  1.0 under all four readouts. Thirteen Aer MPS circuits used 15 peak logical
  qubits, 131,664 logical SWAPs, and 1,799.7 wall seconds.
- An actual five-level Aer processor sweep gave mean nonlinear capacities
  0.00083, 0.8063, 0.8491, 0.9271, and 1.0000 for g=0, .25, .5, .75, and 1,
  respectively, clearing the ordered-interior screen before the full grid.
