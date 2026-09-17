# Audit: g, J, and kappa_processor in the finalized EOC processor

Required before any code change or interpretation (Part 1). Source: direct
read of `code/mixed_syk_core.py` (`mixed_layer`, `kappa_to_gJ`,
`sample_syk4_*`) and `code/eoc_config.py` (`build_reservoir_params`,
`SCIENCE_CONFIG`) — the same finalized modules `processor.py` and every
prior pass in this project reused verbatim.

## 1-2. What exactly are g and J?

**g** is the coefficient of the SYK2-like nearest-neighbor hopping terms:
in `mixed_layer` (`mixed_syk_core.py:144-146`), for every chain edge
`(a,b)`: `qc.rxx(2.0*g, a, b); qc.ryy(2.0*g, a, b)`. It is a single
**rotation angle** (radians) shared by every edge in one layer — a scalar,
not a matrix/tensor.

**J** is NOT a single coefficient — it is the **standard deviation of a
random-Normal distribution** the SYK4-like quartic couplings are drawn
from: `sample_syk4_couplings(n_terms, J, seed)` (`mixed_syk_core.py:59-61`)
returns `rng.normal(0.0, J, size=n_terms)`. Each of the `n_terms ≈
ceil(N ln N)` quartic terms gets its OWN coupling `J_t ~ N(0, J)`, applied
via `pauli4_rotation(qc, qubits4, paulis4, theta=2*J_t)`
(`mixed_syk_core.py:147-148`). So J is a **disorder scale**, not a
Hamiltonian prefactor in the usual sense.

## 3-4. Which terms are multiplied by g / J?

- **g**: exactly `2*(N-1)` gate angles per layer (RXX and RYY on each of
  the `N-1` chain edges) — `H_2 ∝ g * sum_{<i,j>} (X_i X_j + Y_i Y_j)`.
- **J** (via the couplings it parameterizes): `n_terms` quartic Pauli
  rotations, each with its OWN random angle `2*J_t`, `J_t ~ N(0,J)` —
  `H_4 ∝ sum_t J_t * P_{t,i}P_{t,j}P_{t,k}P_{t,l}` for random Pauli types
  and random qubit quadruples (`sample_syk4_pauli_types`,
  `sample_syk4_terms`). Increasing J increases the TYPICAL magnitude of
  each term, not any single term's fixed weight.

So the processor's Hamiltonian is genuinely

    H_P(g, J) = g * H_2 + J * H_4'

where `H_2 = sum_{<i,j>} (X_i X_j + Y_i Y_j)` (fixed structure, chain
topology) and `H_4' = sum_t xi_t * P_t` with `xi_t ~ N(0,1)` FIXED per
`term_seed` and J only rescaling their overall magnitude (i.e. `J_t = J *
xi_t` for a seed-fixed standard-normal `xi_t`) — this is the exact repo
model, not a substitution.

## 5. Is kappa_processor equal to J/g, g/J, or something else?

Neither, exactly — but closely related. `kappa_to_gJ(kappa, G_MAX, J_MAX)`
(`mixed_syk_core.py:200-209`):

    g = G_MAX * kappa / (1 + kappa)
    J = J_MAX / (1 + kappa)

This is a **Möbius (rational) reparameterization**, not a ratio. However,
computing `g/J = (G_MAX/J_MAX) * kappa` directly — so **kappa IS
proportional to g/J**, with proportionality constant `J_MAX/G_MAX`. Every
config in this project so far uses `G_MAX = J_MAX = 0.6`
(`eoc_config.G_MAX_QELM = J_MAX_QELM = 0.6`, and every processor.py call in
this project's DQRC passes uses the same defaults) — so with `G_MAX=J_MAX`,
**kappa = g/J EXACTLY**.

## 6. Are g and J truly independent in the implementation?

**Yes, at the level that matters.** `mixed_layer(qc, N, g, terms, couplings,
paulis, bias_z)` (`mixed_syk_core.py:138`) takes `g` and `couplings`
(which encode J) as two completely separate arguments — nothing in
`mixed_layer` itself enforces any relationship between them. The
constraint `g = G_MAX*kappa/(1+kappa)`, `J = J_MAX/(1+kappa)` is imposed
ONLY by the higher-level convenience function `kappa_to_gJ`, which every
processor-building call in this project (`processor.sample_processor_params`,
`eoc_config.build_reservoir_params`) has so far always routed through. This
confirms the premise of this pass exactly: **the one-parameter kappa API is
a self-imposed restriction of the orchestration layer, not a restriction of
the physics or the circuit-building code.**

## 7. Does changing kappa at fixed scale trace only a 1D path in the true
2D (g,J) space?

Yes, and it can be stated exactly. Since `g/G_MAX + J/J_MAX =
kappa/(1+kappa) + 1/(1+kappa) = 1` for ALL kappa, **the kappa scan traces
precisely the straight line segment**

    g/G_MAX + J/J_MAX = 1,   i.e.   g = G_MAX*(1 - J/J_MAX)

connecting `(g,J) = (0, J_MAX)` (kappa=0, pure SYK4) to `(g,J) = (G_MAX, 0)`
(kappa->infinity, pure SYK2). This line **never enters the region where
BOTH g and J are simultaneously large** (the "strongly-coupled-in-both-
sectors" corner near `(G_MAX,J_MAX)`) or **simultaneously small** (the
near-trivial corner near `(0,0)`) — both corners, and the entire area off
this one anti-diagonal line, were structurally unreachable by every prior
kappa-based scan in this project. This is the precise, quantitative version
of the concern motivating this pass.

## 8. What values/ranges were used in the original finalized EOC notebook?

Notebook 4's `qelm_scan_mixed`: `KAPPA_GRID = geomspace(0.02, 100, 12)`,
`G_MAX=J_MAX=0.6` (`eoc_config.py:216-234` quotes this verbatim). Via
`kappa_to_gJ`, this covers `g ∈ [0.0118, 0.594]`, `J ∈ [0.00594, 0.588]` —
i.e. nearly the full `[0,0.6]` extent of EACH axis individually, but always
constrained to the single anti-diagonal line from question 7, never the
interior or the high-g-high-J / low-g-low-J corners.

## 9. What processor size N_P was used when EOC was originally identified?

**N=6** (`eoc_config.N_MIX = 6`, `SCIENCE_CONFIG`). The two prior DQRC
passes in this project used smaller processors (N_P=3, then N_P=5) for
resource-budget reasons and were flagged, in
`docs/DQRC_ARCHITECTURE_REPAIR.md`, as never having re-derived EOC at their
own size rather than reusing N=6's value — this pass's Part 21 (N_P=5 vs 6
repeat) directly follows up on that same concern.

## 10. Are g and J dimensional or normalized quantities?

Both are used directly as **gate rotation angles** (radians) inside a
FIXED number of layer repetitions (`reps`) per timestep — there is no
separate, explicit "coupling strength" vs. "elapsed time" split anywhere in
`mixed_layer`. This means **g and J are really "coupling strength × Trotter
time" composite quantities**, dimensionless in the sense of being bare gate
angles. Part 16's redundancy concern is real and confirmed algebraically:
since `H_2` and `H_4'` both enter `mixed_layer` linearly in `(g,J)` (RXX/RYY
angle `2g`, quartic-rotation angle `2*J*xi_t`), we have EXACTLY
`H_P(a*g, a*J) = a * H_P(g,J)` for any scalar `a` — so uniformly rescaling
`(g,J) -> (a*g, a*J)` for one layer application is mathematically identical
(to leading/exact order per layer, since each layer is applied via exact
gate angles, not a first-order Trotter approximation of continuous flow) to
running the ORIGINAL `(g,J)` Hamiltonian for `a` times the "time" (loosely,
`reps -> a*reps` in the regime where layer non-commutativity doesn't yet
matter within one layer's own internal gate ordering — layers themselves
don't commute across repetitions in general once entanglement builds up,
so this equivalence is exact only in comparing ONE layer's own generator
scaling, not literally interchangeable with `reps` at the many-layer level;
still, the OVERALL-SCALE axis `a` is confirmed redundant with "how strongly
this one layer rotates," which is confounded with `reps` and must be
tracked separately — see Part 15/16 of the scan plan below).

## Summary: what this audit means for the rest of this pass

1. Every "kappa scan" in this project (and both prior DQRC-repair passes'
   `kappa_processor` sweeps) explored a single fixed anti-diagonal line in
   `(g,J)` space, never the full square. **The previous conclusion that EOC
   only weakly controls nonlinear IPC is NOT yet justified** — restated per
   Part 25's own required language: "the chosen one-dimensional kappa
   direction did not produce nonlinear-selective control," not "EOC cannot
   control NL."
2. g and J are already independent arguments to `mixed_layer` at the
   circuit-building level — Part 2's refactor is additive (new entry points
   that skip `kappa_to_gJ`), not a rewrite of the physics.
3. Overall-scale rescaling of `(g,J)` is a genuinely redundant direction
   with per-layer "time" — any 2D scan must either report `reps` alongside
   `(g,J)` or explicitly separate the scale axis `A` from the composition
   axis `alpha` (Part 15) to avoid mistaking a time-like redundancy for new
   physics.
