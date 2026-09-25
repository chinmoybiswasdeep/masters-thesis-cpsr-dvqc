# V7 pre-development audit of V6.7

Audit date: 2026-09-24. Repository branch `qiskit`, baseline commit
`e893f65c8a6a52cf44efb8835774833106ecda7f`. The worktree was clean.

## Evidence inspected

- `results/v6/preregistered_v6_gates.json` and its SHA-256 commitment.
- `results/v6/amendments/gates_amendment_01.json` and its commitment.
- `results/v6/V6_7/frozen.json`, `manifest.json`, `gates.json`, and `report.md`.
- All 500 raw confirmation rows in `conf_rows.jsonl`. Streaming verification
  found SHA-256 `6e8410d22d75155baa9df203768ec8705c45187bea5afaec46d1f909c8b366fc`.
- `code/decoupled_qrc/v6_architecture.py`, `v6_core.py`, and `v6_gates.py`.
- `docs/DQRC_V6_RESULTS.md` and `docs/V6_PROGRESS.md`.

No V6 file was modified.

## Reproduced limitations

1. The V6.7 reservoir is an analytical/classical stochastic model, not an
   executed `QuantumCircuit`. The architecture computes polynomial/trigonometric
   features directly and applies independent binomial draws to observables.
2. V6.7 is a redraw of the unchanged V6.6 design on a fresh development
   sentinel block. Its frozen rationale explicitly records selection on noise.
3. `exact_expectations_perturbed_controls` fails (worst HH advantage -0.001)
   although finite-shot controls pass.
4. Some combined effects are finite-shot amplitude/SNR effects rather than a
   changed exact feature subspace.
5. Class averages hide the tail: most capacity is at delays 1-2; targets at
   delays 3-5 and beyond are approximately null.
6. The architecture explicitly uses selected rail pairs `(1,2)` and `(2,4)`.
7. Observable shot noise is drawn independently rather than from shared joint
   bitstrings within a commuting measurement setting.
8. The `FakeTorino` condition is an effective scalar approximation; no circuit
   is transpiled to a fake backend and run under an Aer-derived backend noise
   model.
9. Both memory registers are classical Markov chains in a fixed product basis,
   and the frozen report correctly rejects quantum advantage.
10. Input-copy accounting is ambiguous because analytical feature copies are
    counted without physical circuit preparation/replay.
11. Only per-step costs are emphasized; destructive finite-shot measurement
    requires replaying the complete input history for each measured timestep.

These findings are the design constraints for V7 and are not reinterpretations
of V6.7's original reported result.
