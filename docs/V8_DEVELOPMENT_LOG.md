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
