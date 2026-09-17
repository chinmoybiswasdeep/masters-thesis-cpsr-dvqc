# DQRC theory: equations and physical meaning

Companion to `docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md` (what exists and what
DQRC reuses) and `docs/DQRC_RESULTS.md` (what the FAST_MODE run found). This
document states the equations each module implements and why.

## 1. Information Processing Capacity (`code/decoupled_qrc/ipc.py`)

For $u_t \sim \mathrm{Uniform}[0,1]$ (this repo's own encoding convention),
define $v_t = 2u_t - 1 \in [-1,1]$. The orthonormal Legendre basis under the
**Uniform$[-1,1]$ probability measure** (density $\tfrac12 dv$) is

$$L_d(v) = \sqrt{2d+1}\, P_d(v), \qquad
\tfrac12\int_{-1}^{1} L_d(v) L_e(v)\, dv = \delta_{de}$$

where $P_d$ is the standard Legendre polynomial. (Note: $\sqrt{(2d{+}1)/2}$,
the normalization for plain Lebesgue measure on $[-1,1]$, is a *different,
wrong* constant for this purpose — `tests/test_ipc.py`'s Gauss-quadrature
test catches exactly this class of convention bug; it failed under the wrong
constant during development, see the memory `feedback-scientific-rigor`
pattern this project follows.)

A **degree-$d$ profile** is a set of $(\text{delay}, \text{degree})$ pairs
with distinct delays and each degree $\ge 1$, total degree $d$:

$$y_t^{(\text{profile})} = \prod_{(\tau,k)\in\text{profile}} L_k(v_{t-\tau})$$

**Capacity** of one target: squared correlation between $y^{(\text{profile})}$
and its ridge reconstruction from reservoir features $X$, regularization
selected on a validation block, reported on a held-out test block, clipped to
$[0,1]$ — the same construction `qrc_qiskit.memory_capacity`'s per-lag score
already uses, generalized from single delays to arbitrary polynomial-product
targets.

$$\mathrm{IPC}_d = \sum_{\text{profiles of degree } d} \text{capacity(profile)}
\quad\text{(only profiles that clear a shuffle-based significance
threshold count)}$$

$$M = \mathrm{IPC}_1 \qquad NL = \sum_{d=2}^{6} \mathrm{IPC}_d
\qquad \text{Total} = M + NL$$

**Significance filtering**: for each target, `n_surrogates` independent
train-block shuffles of the SAME target are refit; a target's real-data
score only counts if it exceeds $\text{mean} + z\cdot\text{std}$ of its own
null distribution — otherwise its contribution is 0 (finite-data noise, not
real capacity).

## 2. The mixed-SYK reservoir layer (reused, not redefined)

$U_{\text{layer}} = $ nearest-neighbor SYK2-like hopping (`Rxx`, `Ryy` at
angle $g$) + sparse random-Pauli-type SYK4-like quartic rotations (angle
$\propto J$) + on-site $Rz$ disorder — `mixed_syk_core.mixed_layer`, reused
verbatim. $\kappa$ interpolates:

$$g(\kappa) = \frac{G_{\max}\,\kappa}{1+\kappa}, \qquad
J(\kappa) = \frac{J_{\max}}{1+\kappa}$$

$\kappa\to 0$: pure SYK4-like (chaotic quartic layer alone). $\kappa\to
\infty$: pure SYK2-like (integrable free-fermion hopping alone).

## 3. Memory subsystem M (`code/decoupled_qrc/memory.py`)

Input qubit $q_0$: `reset`; $R_y(\pi u_t)$, every step. Memory qubits
$q_1..q_{N_M}$: **never reset**. Three internal-dynamics modes:

- **swap**: $q_0 \leftrightarrow q_1$ via an exact `SWAP` each step (perfect
  transport into the register), plus a WEAK $R_{xx}(\epsilon){+}R_{yy}(\epsilon)$
  exchange along the memory chain so information slowly diffuses past $q_1$.
- **integrable**:
  $H_M = \sum_i \omega_i Z_i + \Omega \sum_i (X_iX_{i+1}+Y_iY_{i+1})$ —
  a free-fermion XY hopping chain, exactly solvable via Jordan-Wigner, hence
  genuinely non-chaotic by construction.
- **weak_xx**:
  $H_M = \sum_i \omega_i Z_i + \epsilon \sum_i X_iX_{i+1}$, $\epsilon$ small —
  the task spec's own literal example: strong on-site precession plus a weak
  XX-only (no YY) coupling, staying closer to diagonal in the Z basis.

**Quantum delay register**: an *exact* shift register realized by a SWAP
chain executed high-to-low every step, so after step $t$: slot $k$ holds
$u_{t-k}$ for $k=0..N_M{-}1$, with **no classical array anywhere** — verified
directly against the simulated quantum state in `tests/test_memory.py`
(`test_quantum_delay_register_is_exact_shift_no_classical_array`).

**Critical implementation constraint**: both circuits reset one qubit every
step of one long trajectory circuit — the confirmed Aer
`method='statevector'` reset bug's exact trigger pattern
(`docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md` §10). `method='density_matrix'` is
enforced by a hard `ValueError`, not a convention.

## 4. Processor subsystem P (`code/decoupled_qrc/processor.py`)

The processor is `mixed_layer` restricted to its own qubits, parameterized
by `kappa_processor` (this repo's established chaos-control knob — the task
spec's "$g_\text{processor}$"). Two chaos diagnostics computed **fresh** on
the processor's own $\kappa$ grid (never inherited from the stored
$\kappa{=}0.960$, which the audit found was task-performance-selected, not a
pure spectral criterion):

$$S(U) = -\sum_i p_i \ln p_i, \quad p_i = \sigma_i(U_{\text{reshaped}})^2 /
\sum_j \sigma_j^2 \quad \text{(operator entanglement, half-chain SVD)}$$

$$\langle r \rangle = \left\langle \frac{\min(s_n, s_{n+1})}{\max(s_n,
s_{n+1})} \right\rangle_n, \quad s_n = \text{eigenphase gaps of } U^{\text{reps}}$$

compared against Poisson (integrable), COE, CUE (chaotic) reference
ensembles at the same Hilbert-space dimension. DQRC's own EOC estimate is the
$\kappa$ whose $\langle r\rangle$ is closest to the Poisson/COE crossover
midpoint — an operational, task-independent definition.

## 5. Interface $U_{MP}(\lambda)$ (`code/decoupled_qrc/interface.py`)

$$U_{MP}(\lambda) = \exp(-i\lambda\, Z_M Z_P) \quad\text{(rzz, default)}
\quad\text{or}\quad \exp(-i\lambda\, Z_M X_P) \quad\text{(zx)}$$

applied between paired memory-tap/processor-entry qubits. $\lambda=0$ is an
exact identity by construction — the point $\lambda_{MP}=0$ in the Jacobian
experiment is literally "memory and processor fully disconnected."

## 6. Ancilla-mediated stitching (`code/decoupled_qrc/stitching.py`)

For modular processor blocks $P_1..P_B$: a fresh ancilla (reset before and
after) mediates a transient $Z\!Z$ correlation between adjacent blocks'
boundary qubits — $\lambda_{\text{stitch}}=0$ reduces exactly to $B$
disconnected blocks. Named "ancilla-mediated stitching", **not** "Huang
sewing" — see that module's docstring and
`docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md` §8 for why.

## 7. Classical shadows (`code/decoupled_qrc/shadows.py`)

Standard local-Pauli classical shadow (Huang, Kueng & Preskill 2020),
generalized from a pure-statevector Born sampling (as
`shadow_measurements.py` already implements) to a **density-matrix** Born
sampling (DQRC's memory-register states are generically mixed): for a random
basis $U = \bigotimes_i u_i$, $u_i \in \{I, H, HS^\dagger\}$,

$$\Pr(\text{outcome} = x) = \langle x| U\rho U^\dagger |x\rangle$$

Estimator (reused verbatim from `shadow_measurements.precompute_b_factors` +
`median_of_means`):

$$\widehat{\mathrm{Tr}[O\rho]} = \mathrm{median\_of\_means}\left(
3^{w}\prod_{i\in \text{supp}(O)} s_i \cdot \mathbb{1}[\text{basis matches } O]
\right)$$

## 8. Jacobian decoupling diagnostic (`code/decoupled_qrc/metrics.py`)

$$J = \begin{pmatrix}\partial M/\partial m & \partial M/\partial g\\
\partial NL/\partial m & \partial NL/\partial g\end{pmatrix}, \qquad
\text{cross\_coupling} = \frac{|\partial M/\partial g| + |\partial NL/\partial m|}
{|\partial M/\partial m| + |\partial NL/\partial g| + \varepsilon}$$

$$\text{decoupling\_score} = \frac{1}{1+\text{cross\_coupling}}$$

Reported as an engineering diagnostic only — the raw Jacobian entries are the
primary scientific result (per the task spec's own instruction not to treat
this score as fundamental).

## 9. Pareto frontier and hypervolume (`code/decoupled_qrc/metrics.py`)

Standard non-dominated sweep on $(M, NL)$ point clouds (maximize both);
2D hypervolume indicator relative to a reference point $(0,0)$ summarizes
"how good is this whole frontier" as one bootstrap-able scalar per seed —
the primary statistical quantity behind the Section 9 verdict in
`docs/DQRC_RESULTS.md`.
