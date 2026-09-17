"""
shadows.py -- Part 6: `readout_mode="exact_pauli"` (thin wrapper on the
finalized exact Pauli-expectation readout) and a GENUINE
`readout_mode="classical_shadow"` (real randomized local-Pauli measurement,
finite-shot single-shot estimators -- not Gaussian noise added to exact
values, which the audit confirmed this repo has never done and which Part 16
explicitly forbids calling a "classical shadow").

Reuses `shadow_measurements.py`'s existing, already-verified shadow
primitives (`precompute_b_factors`, `median_of_means`, the same Y-basis
`H.Sdg` convention) rather than re-deriving them. The one genuinely new piece
here is sampling a shadow from a DENSITY MATRIX (`shadow_measurements.
sample_shadow_exact` only handles a pure statevector) -- DQRC's memory-register
states are generally mixed (memory.py's whole point is a persistent,
never-reset register), so a dense per-snapshot basis-rotation of rho is used
instead. This is the SAME "dense kron is fine at small N, catastrophic at
N~12+" scale trade-off `shadow_measurements.py`'s own docstring documents;
this module is only ever used on the small (`N<=~8`) DQRC readout register,
well inside the regime where the dense approach is fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Pauli

from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

import mixed_syk_core as msc  # noqa: E402
from qrc_qiskit import make_simulator  # noqa: E402
from shadow_measurements import (  # noqa: E402 -- reused verbatim, not re-derived
    _BASIS_ROT, precompute_b_factors, median_of_means,
)

READOUT_MODES = ("exact_pauli", "classical_shadow")


# =============================================================================
# exact_pauli
# =============================================================================

def exact_pauli_features(rho: np.ndarray, N: int, max_weight: int = 3):
    """Exact expectation values of `mixed_syk_core.feature_ops_all_general`'s
    observable dictionary against a density matrix `rho`, via Qiskit's own
    `DensityMatrix.expectation_value` (no shot noise) -- the same observable
    set the shadow estimator below targets, so the two are directly
    comparable."""
    labels, ops = msc.feature_ops_all_general(N, max_weight=max_weight)
    dm = DensityMatrix(rho)
    values = np.array([np.real(dm.expectation_value(op, qargs)) for op, qargs in ops])
    return labels, ops, values


# =============================================================================
# classical_shadow -- density-matrix Born sampling
# =============================================================================

def _dense_basis_unitary(bases: np.ndarray, n_qubits: int) -> np.ndarray:
    U = np.array([[1.0]], dtype=np.complex128)
    for q in range(n_qubits - 1, -1, -1):
        U = np.kron(U, _BASIS_ROT[int(bases[q])])
    return U


def sample_shadow_from_density_matrix(rho: np.ndarray, n_qubits: int, n_snapshots: int,
                                       rng: np.random.RandomState):
    """Local-Pauli classical shadow of a (possibly MIXED) density matrix:
    one independent random basis per qubit per snapshot, Born-sampled from
    the RE-ROTATED rho's diagonal (prob(outcome) = <outcome| U rho U^dag
    |outcome>). Reduces exactly to `shadow_measurements.sample_shadow_exact`'s
    statistics when rho is pure. Returns (bases, signs), matching that
    function's shape/convention exactly so downstream estimator code
    (`precompute_b_factors`) is shared."""
    dim = rho.shape[0]
    bases = rng.randint(0, 3, size=(n_snapshots, n_qubits)).astype(np.int8)
    signs = np.empty((n_snapshots, n_qubits), dtype=np.int8)
    idx_all = np.arange(dim)
    bits_table = ((idx_all[:, None] >> np.arange(n_qubits)) & 1).astype(np.int8)
    for k in range(n_snapshots):
        U = _dense_basis_unitary(bases[k], n_qubits)
        rho_rot = U @ rho @ U.conj().T
        probs = np.real(np.diag(rho_rot))
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        outcome = rng.choice(dim, p=probs)
        signs[k] = 1 - 2 * bits_table[outcome]
    return bases, signs


def shadow_readout(rho: np.ndarray, N: int, max_weight: int, n_snapshots: int, rng: np.random.RandomState,
                    n_groups_mom: int = 10):
    """Classical-shadow estimate of the SAME observable dictionary
    `exact_pauli_features` reads out exactly, via
    `shadow_measurements.precompute_b_factors` + median-of-means
    (`shadow_measurements.median_of_means`) -- both reused, not
    reimplemented."""
    labels, ops = msc.feature_ops_all_general(N, max_weight=max_weight)
    bases, signs = sample_shadow_from_density_matrix(rho, N, n_snapshots, rng)
    b_factors = precompute_b_factors(bases, signs, ops)
    values = np.array([median_of_means(b_factors[:, j], n_groups_mom) for j in range(len(ops))])
    return labels, ops, values


# =============================================================================
# exact-vs-shadow convergence sweep (Part 6: "estimator error versus shots")
# =============================================================================

@dataclass
class ShadowConvergencePoint:
    n_snapshots: int
    rmse: float
    rmse_ci_lo: float
    rmse_ci_hi: float


def shadow_convergence_sweep(rho: np.ndarray, N: int, max_weight: int, shot_grid,
                              n_repeats: int = 5, seed: int = 0) -> list:
    """For each n_snapshots in `shot_grid`, run `n_repeats` INDEPENDENT
    shadows (separate measurement-RNG streams) and report RMSE of the shadow
    estimate against the exact features, with a bootstrap CI across repeats
    -- Part 6's required 'estimator error versus shots... seeded randomness
    and confidence intervals'."""
    _, _, exact_vals = exact_pauli_features(rho, N, max_weight)
    out = []
    for n_snap in shot_grid:
        errs = []
        for r in range(n_repeats):
            rng = np.random.RandomState(seed * 100000 + n_snap * 100 + r)
            _, _, shadow_vals = shadow_readout(rho, N, max_weight, n_snap, rng)
            errs.append(float(np.sqrt(np.mean((shadow_vals - exact_vals) ** 2))))
        errs = np.asarray(errs)
        if n_repeats > 1:
            boots = np.array([np.mean(errs[np.random.RandomState(seed + i).randint(0, n_repeats, n_repeats)])
                               for i in range(200)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = errs[0]
        out.append(ShadowConvergencePoint(n_snapshots=n_snap, rmse=float(errs.mean()),
                                           rmse_ci_lo=float(lo), rmse_ci_hi=float(hi)))
    return out


# =============================================================================
# Reservoir density-matrix extraction (minimal, readout-comparison-only --
# NOT a duplicate of `mixed_syk_core.build_trajectory_circuit_mixed`, which
# hardwires save_expectation_value rather than exposing rho).
# =============================================================================

def reservoir_density_matrix(N: int, kappa: float, T: int, reps: int = 1, term_seed: int = 0,
                              disorder_seed: int = 0, input_seed: int = 0,
                              G_MAX: float = 0.6, J_MAX: float = 0.6) -> np.ndarray:
    """The exact density matrix after processing a length-T random input
    trajectory through the mixed-SYK channel (same encoding/layer convention
    as the rest of the repo), for the shadow-vs-exact readout comparison.
    Built directly (not via `build_trajectory_circuit_mixed`) only because
    that function hardwires per-step `save_expectation_value` labels rather
    than exposing the final state -- uses the exact same `mixed_layer`/
    `kappa_to_gJ`/support-sampling primitives, no new physics."""
    import eoc_config as ec
    from qrc_qiskit import ReservoirConfig, random_input

    g, J = msc.kappa_to_gJ(kappa, G_MAX, J_MAX)
    n_terms = msc.default_n_sparse_terms(N)
    supports = ec.sample_syk4_supports(N, n_terms, term_seed, J)
    cfg = ReservoirConfig(N=N, g=0.0, reps=reps, seed=disorder_seed)
    bias_z, _ = cfg.sample_disorder()
    u = random_input(T, seed=input_seed)

    qc = QuantumCircuit(N)
    for u_t in u:
        qc.reset(0)
        qc.ry(np.pi * float(u_t), 0)
        for _ in range(reps):
            msc.mixed_layer(qc, N, g, supports["terms"], supports["couplings"], supports["paulis"], bias_z)
    qc.save_density_matrix(label="rho_final")

    sim = make_simulator(method="density_matrix")
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    return np.asarray(result.data(0)["rho_final"])
