# DQRC V7 development result

## Verdict

No V7 candidate is frozen and no success claim is made. V7.0 through V7.8 were
development-only candidates. Every one was rejected before internal validation,
so the committed confirmation seed bank was never revealed.

The strongest candidate, V7.8, demonstrates genuine circuit-derived independent
primary controls in exact Qiskit Aer simulation, but fails the preregistered
combined long-tail tasks. It therefore does **not** demonstrate the required
complete memory–nonlinearity separation.

## Execution provenance

- Simulator: Qiskit 2.5.2, Qiskit Aer 0.17.2.
- Exact mode: `AerSimulator(method="density_matrix")`; non-destructive saved
  Pauli expectations from seven six/seven-qubit circuit settings.
- Finite-shot implementation: `AerSimulator(method="matrix_product_state")`
  with actual complete joint bitstring counts and full prefix replay. It was
  smoke-tested, but not spent on V7.8 after the mandatory exact gate failed.
- Architecture and runner modules contain no NumPy import and no alternative
  analytical reservoir path.

## V7.8 exact development diagnostic (one fixed development seed)

| quantity | low control | high control | result |
|---|---:|---:|---|
| M, raw/whitened OLS | -0.692 | 3.423 | direction passes |
| M, ridge CV | -0.549 | 4.339 | direction passes |
| N, raw/whitened OLS | -0.024 | 2.809 | direction passes |
| N, ridge CV | -0.014 | 2.812 | direction passes |
| delayed quadratic tail mean, raw | best alternative 0.008 | HH 0.136 | diagnostic pass |
| delayed cubic tail mean, raw | best alternative -0.055 | HH 0.034 | diagnostic pass |
| current-NL × delayed-linear tail, raw | best alternative -0.134 | HH -2.298 | **fail** |
| delayed × delayed tail, raw | best alternative -0.057 | HH -0.614 | **fail** |

These are screening estimates, not confidence-bound gate results. The full
four-seed development response surface was correctly not run after the exact
corner failure.

## Resource semantics

V7 counts physical circuit executions, state preparations, input encodings,
shots, settings, wall time and every prefix replay depth. Exact observations
share a persistent trajectory because Aer save instructions are non-destructive.
Finite-shot observations create one measured circuit per prefix and setting.

## Tests

The six V7-specific tests pass. They cover actual `AerSimulator.run` calls,
absence of analytical/NumPy fallbacks, seed isolation, same-bitstring joint
parities, destructive temporal replay accounting and separation-gate rejection.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v7_*.py -q
.\.venv\Scripts\python.exe code\run_v7_stage.py --stage SMOKE --version V7.8
```

## Remaining limitations

- No candidate passed all development gates; validation and confirmation are
  intentionally absent.
- Fake-backend noise, 10k/100k-shot budgets, precision sensitivity and matched
  classical baselines were not run because the earlier exact mandatory gate
  failed.
- No quantum advantage is demonstrated or claimed.
