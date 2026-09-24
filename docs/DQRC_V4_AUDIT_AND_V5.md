# V4 independent audit and the V5 lineage

Two results, both judged by one gate implementation (`code/run_v4_audit.py::all_gates`)
with an independent IPC engine (`code/decoupled_qrc/audit_ipc.py`: unclipped
`C = 1 − MSE/Var`, readouts = fixed ridge α=0.5 / standardised OLS / raw OLS,
120 independent-sequence null targets per point). Every verdict uses the sequential CI
levels (α = 0.01 per look): main-effect lower bound is the 1st percentile, TOST uses the
1st–99th percentile interval, and the ratio bound is the 99th percentile. The unit is the
seed, with 10 000 bootstrap resamples. Simulation is exact and noiseless unless a row says
otherwise.

| | V4 (frozen, audited) | V5.4 (new) |
|---|---|---|
| STRUCTURAL ROUTE ISOLATION | **PASS** | **PASS** |
| RESOURCE-CONSTRAINED M–NL SEPARATION | **PASS** | **PASS** |
| INTRINSIC M–NL SEPARATION | **FAIL** | **PASS** |
| COMBINED NONLINEAR MEMORY | **PASS** (class C1 only) | **PASS** (class C1 only) |
| QUANTUM-SPECIFIC MECHANISM | **FAIL** | **FAIL** |
| overall | `V4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED DEFINITION` | claims 1–4 pass, 5 fails (see §3) |

Both confirmations: 20 fresh seeds × 5×5 grid (m, g ∈ {0, .25, .5, .75, 1}), T = 1600.
Evidence: `results/v4/independent_audit/report.md`, `results/v5_4/report.md`.

## 1. V4 audit

**Protocol.** The audit ran ENV → DEV (10 dev seeds) → PREREG (`preregistered_audit.json`,
sha256 `72885188…`, written before any confirmation data existed) → ARTIFACTS → CONFIRM →
REPORT. `verify_prereg` refuses to run if the plan or any audit/V4 source file changed.
One incident is recorded in `incident_notebook_hash.json`. An editor had re-serialised the
V4 Colab notebook after ENV hashed it. The cells were identical; only the Unicode escaping
differed. Re-running the notebook's builder reproduced the preregistered bytes exactly, and
the preregistration was not touched.

**Why claim 3 fails (section 6).** V4's 12 processor observables span every polynomial of
degree ≤ 4. Unregularised capacity therefore equals 1.0 for every g ≳ 1e-4, and the
feature subspace is identical: principal angle 0°. On the confirmation seeds the
interior trend of N is 0.000 under OLS raw, OLS standardised, and ridge λ ≤ 1e-4. It is
nonzero only at λ ≥ 1e-2 (0.12 at λ = 0.01, 0.57 at λ = 1). The g-control of N is a
**ridge-penalty artifact** → `CONTINUOUS INTRINSIC NONLINEAR CONTROL FAILED`. The interior
effect N(1)−N(0.25) is +0.000 under OLS, and M(1)−M(0.25) is +0.033 (below the 0.10 gate),
because V4's memory is also saturated.

**Other findings.**
- The operational observable set has encoder-only degree-2 capacity of 1.0.
- 290/401 negative raw capacities were hidden by V4's clipping.
- V4's combined gate uses two identical metrics: |N_long − cross_delay_mean| ≤ 1e-16.
- V4's Holm equivalence never adjusts its interval level.
- The ClaimGuard provenance check compares the frozen hash with itself.
- V4's gate CIs do not use the alpha that its sequential ledger spends.
- At 1000 shots V4's interior OLS N becomes 0.154: shot noise, not g, breaks the span
  degeneracy.

**Claim 5.** Joint minus classical product features on C1: −0.009, LB −0.010. At equal
shots the classical product of marginals is never worse →
`COMBINED MECHANISM IS CLASSICALLY REPRODUCIBLE`.

## 2. V5 lineage (development seeds 40000–40009 / 45000–45009 only)

Architecture (`code/decoupled_qrc/v5_architecture.py`):
- **Memory R (control m):** a random-SWAP channel on rails 0..L. Only single-site ⟨Z⟩ is
  read, so R is *exactly* linear in past inputs; the marginal recursion equals the full
  density matrix to 1e-15.
- **Processor P (control g):** two copies of ρ(u), Ry(π/2), exp(−iθZ₀Z₁/2), and a single
  observable f = cos φ cos θ·u + sin φ sin θ·u². One feature makes N the nonlinear fraction
  of its variance, which is scale-invariant and continuous in g.
- **Joint layer:** J = z_r·f.

| version | outcome | reason |
|---|---|---|
| V5 | not frozen | memory interior M(1)−M(0.5) = 0.070 < 0.10 at delay range 12 (L = 6 cannot reach τ 7–12) |
| V5.1 | not frozen | met its criterion but **M saturated** (6/7 constituents > 0.995 at m = 1, M_ols = 1.000); pooled saturation check (3.1%) had hidden it. Per-metric criterion added (stricter). |
| V5.2 | no candidate | a fully read SWAP register either saturates (OLS inverts short transport exactly) or cannot reach τ 9–12 |
| V5.3 | not frozen | memory fixed by reading every 2nd rail of a long chain; every finalist failed θ_max×1.1 stress (N saturated) |
| **V5.4** | **frozen `0a2f1e18…`** | L = 16, read rails 2,4,…,16, p_max = 0.7, θ_max = 0.38π, φ = π/3 |

Each criterion change across V5.x made the criteria stricter; none was weakened. No V5.x
version saw a confirmation seed. The alpha ledger has three looks at 0.01: V4 original,
V4 audit, and V5.4.

**Amendment 01** (`results/v5_4/amendment_01.json`, committed `a321d78` before any gate
result existed). CONFIRM crashed in the claim-5 classical-baseline helper: it indexed 16
rails where 8 are read. The 500 confirmation rows had already been produced by the frozen
source and were reused unchanged. The guard accepts the changed source only through this
hashed from→to record.

## 3. V5.4 confirmation (seeds 90000–90019 / 95000–95019)

| gate | estimate | bound | threshold |
|---|---|---|---|
| ΔmM (ridge / OLS-std / OLS-raw) | 0.522 / 0.522 / 0.522 | LB 0.516 | ≥ 0.10, LB > 0 |
| ΔgN (all three) | 0.280 | LB 0.279 | ≥ 0.10, LB > 0 |
| interior M(1)−M(0.25), N(1)−N(0.25) (OLS-std) | 0.316, 0.255 | LB 0.309, 0.251 | ≥ 0.10 |
| cross ΔmN, ΔgM (all three) | 0.000 | TOST [0, 0], ratio 0 | ⊂ ±0.03, < 0.20 |
| degree profile (m), memory curve (g) | equivalent | Bonferroni | ±0.03 |
| section 6 | HOLDS: interior trend 0.274 under every readout and every λ | | |
| C1 current-NL × old-linear | HH 0.119 | min LB 0.101 over LL/LH/HL | LB > 0 |
| C2 old-NL, C3 old×old, C4 old³ | unsupported (−0.017…−0.019) | | |
| quantum − classical (C1) | 0.000 | LB 0.000 | LB > 0 → FAIL |

**No saturation.** 0/7 memory and 0/3 nonlinear constituents exceed 0.995 anywhere;
M ≤ 0.52 and N ≤ 0.28. Operational encoder-only nonlinear capacity is 0. All six invalid
controls were detected: g→memory, m→nonlinear, serial, target leakage, train-as-test, and
feature-count change. Dev robustness passed 13/13 conditions: T = 3200, half/double
training, float32, perturbed controls, ridge 0.05/5, delay range 12, degree 6, 1000/10 000
shots, and FakeTorino-calibrated noise.

**Overall label.** `gates.json` prints `V5.4 WORKS ONLY UNDER A RESOURCE-CONSTRAINED
DEFINITION` because the brief's four-outcome mapping has no branch for *claims 1–4 pass,
claim 5 fails*, so it falls through. That label is inaccurate for V5.4, since claim 3
passed. Correct reading: **memory and nonlinearity are separately and intrinsically
controllable; combined nonlinear memory is limited to class C1
(current-nonlinear × old-linear); the combining mechanism is classically reproducible.**
It is not "works as claimed", because claim 5 fails.

## 4. Limits (what V5.4 does not show)

- **No general long nonlinear memory.** Only C1 is supported. The memory is exactly
  linear, so nonlinear functions of *past* inputs (C2–C4) are unreachable by design.
- **No quantum advantage.** J = R·P exactly (max difference 0.0). At equal shots the
  classical product of marginals is *better* than the per-shot joint estimator: C1 0.052
  vs 0.027 at 1000 shots.
- **Noiseless headline.** The IBM row is an effective, calibration-informed noise model
  (depolarizing + readout error + shots), not a transpiled-circuit simulation. No hardware
  run was made, so no energy figure exists.
- **No architectural disorder.** Seeds vary only the input sequence, so seed spread
  measures input-sample variability, not device variability.

## Reproduce

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
cd code
for S in ENV DEV PREREG ARTIFACTS CONFIRM REPORT; do python run_v4_audit.py --stage $S; done
for S in SEARCH DEVGATES FREEZE CONFIRM REPORT; do python run_v5_stage.py --stage $S --version V5.4; done
python _build_notebook_v5_colab.py
```

Runtimes on this machine (12-core CPU, shared between concurrent jobs):
- V4 audit: DEV 3347 s; CONFIRM 1969 s (rows 809 s).
- V5.4: SEARCH 1437 s; confirmation rows 139 s.
- Full test suite: 644 passed, 0 failed, 4 warnings (existing Sobol power-of-two notice),
  348 s; log in `results/v5_4/full_test_log.txt`.

Row checkpoints (`*.jsonl`) are git-ignored. They regenerate bit-for-bit and are hashed
in each manifest.
