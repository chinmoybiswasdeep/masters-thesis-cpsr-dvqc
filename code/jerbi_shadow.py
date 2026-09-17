"""
jerbi_shadow.py -- Jerbi-INSPIRED Choi-flipped classical-shadow readout for
the mixed-SYK Edge-of-Chaos QELM (`mixed_syk_core.py` / `eoc_config.py`).

Terminology note (see the notebook's Section 1 for the full discussion): this
is called "Jerbi-inspired", not "the Jerbi construction". A literature check
of Jerbi et al. (2024) and its supplement found their own "flipped model"
formalism does NOT use a Choi-Jamiolkowski construction -- their flip is a
role-swap Tr[rho(x)O(theta)] -> Tr[rho(theta)O(x)] with a TRACE-NORM
normalization for turning an indefinite parametrized observable into a state.
The Choi-based route implemented here is an independent specialization of the
same underlying philosophy (freeze the x-independent quantum object; do
everything x-dependent classically), valid because the mixed-SYK reservoir
channel E(rho)=U rho U^dagger is an EXACT unitary conjugation, so its Choi
state is exactly pure and needs no positive/negative-part split.

MATH SUMMARY:
    f_j(x) = Tr[O_j E(rho_x)] = Tr[O_j U rho_x U^dagger]
    J_E = (I_A tensor E_B)(|Phi><Phi|) = |Psi_E><Psi_E|,  |Psi_E> = (I tensor U)|Phi>
    f_j(x) = d * Tr[J_E (rho_x^T tensor O_j)]                      (d = 2**N)
Verified to machine precision in the notebook, INCLUDING an explicit dual-
hypothesis check of the tensor ordering (`exact_choi_feature_both_orderings`)
and a complex-amplitude input state that would fail if the transpose were
silently dropped (`direct_features_complex_encoding` /
`exact_choi_feature_complex`) -- the default real Ry(pi*u)|0> QELM encoding
satisfies rho_x^T = rho_x exactly, so a transpose bug could otherwise hide
behind every other test passing.

QUBIT LAYOUT: register A = qubits [0..N-1] (untouched), register B =
qubits [N..2N-1] (U_EOC acts here). |Phi> is prepared by N EPR pairs
(H(A_q); CX(A_q,B_q)). The joint operator (rho_x^T tensor O_j) is embedded as
np.kron(M_B, M_A) -- B's qubits are more significant than A's, matching
Qiskit's own qubit-index-to-kron-factor convention. This is NOT assumed:
`exact_choi_feature_both_orderings` builds BOTH np.kron(M_B,M_A) and
np.kron(M_A,M_B) and the notebook asserts exactly one matches the direct
QELM reference.

EFFICIENT ESTIMATOR: rho_x^T tensor O_j factorizes into a product of PER-QUBIT
operators, so its single-copy shadow estimator is a PRODUCT of per-qubit
single-shot estimators (Huang, Kueng & Preskill 2020). The register-B
("b_factor", from `shadow_measurements.precompute_b_factors`) contribution is
entirely x-INDEPENDENT; the register-A ("a_factor") contribution is an O(N)
closed-form function of the new input's N encoding angles. See
`ChoiShadowDeployment.predict_features` (full feature reconstruction) and
`.predict_compressed` (the READOUT-COMPRESSED estimator: after ridge training
of weights w_j, define O_W = sum_j w_j O_j and evaluate the SINGLE combined
observable directly, at O(N) cost independent of the number of features).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, Statevector

import mixed_syk_core as msc
import shadow_measurements as sm

# =============================================================================
# 1. Window encoding (exactly `build_qelm_circuit_mixed`'s per-step convention)
# =============================================================================

def window_angles(window_values: Sequence[float], N: int) -> np.ndarray:
    """Map an already-extracted, left-zero-padded window of length
    eff_window<=N onto qubits 0..eff_window-1 (identity mapping, no
    wraparound) -- IDENTICAL convention to `build_qelm_circuit_mixed`'s
    per-timestep body."""
    per_q = np.zeros(N)
    for w, uu in enumerate(window_values):
        per_q[w] += np.pi * float(uu)
    return per_q


def trajectory_window(u_seq: Sequence[float], t: int, N: int, window_size: int) -> np.ndarray:
    """Extract the window ending at step t of a trajectory `u_seq`, EXACTLY as
    `build_qelm_circuit_mixed` does internally, then map it via
    `window_angles`."""
    eff_window = min(window_size, N)
    lo = max(0, t - eff_window + 1)
    wlen = t - lo + 1
    window = np.zeros(eff_window)
    window[-wlen:] = u_seq[lo:t + 1]
    return window_angles(window, N)


def bloch_xz(per_q: np.ndarray):
    """Bloch vector (rx, rz) of each qubit's Ry(theta)|0> encoded state
    (ry=0 always for this REAL encoding). Returns (rx, rz), each shape (N,)."""
    return np.sin(per_q), np.cos(per_q)


# =============================================================================
# 2. Direct QELM reference (exact, single-window, NO trajectory/reset/Aer)
# =============================================================================
# Deliberately does NOT reuse `run_reservoir_qelm_mixed(..., method='statevector')`
# (its default) -- see `test_aer_statevector_reset_bug.py` for a documented,
# reproducible Aer bug in that path. QELM is memoryless, so a single fresh
# circuit per window (no reset at all) is both simpler and provably exact.

def direct_qelm_statevector(per_q: np.ndarray, N: int, g: float, terms, couplings, paulis,
                             bias_z: np.ndarray, reps: int) -> Statevector:
    """Exact statevector after Ry(per_q) encoding + `reps` mixed_layer
    applications, starting fresh from |0...0>."""
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.ry(float(per_q[i]), i)
    for _ in range(reps):
        msc.mixed_layer(qc, N, g, terms, couplings, paulis, bias_z)
    return Statevector.from_instruction(qc)


def direct_qelm_features(per_q: np.ndarray, N: int, g: float, terms, couplings, paulis,
                          bias_z: np.ndarray, reps: int, ops) -> np.ndarray:
    """f_j(x) = Tr[O_j U rho_x U^dagger] for every (Pauli, qargs) in `ops`,
    computed exactly via `Statevector.expectation_value`. Ground truth
    'A. Direct QELM' reference used throughout the notebook."""
    sv = direct_qelm_statevector(per_q, N, g, terms, couplings, paulis, bias_z, reps)
    return np.array([np.real(sv.expectation_value(op, qargs)) for op, qargs in ops])


def direct_qelm_features_from_params(per_q: np.ndarray, params, ops) -> np.ndarray:
    """Convenience: take an `eoc_config.ReservoirParams` instead of six loose
    positional arguments."""
    return direct_qelm_features(per_q, params.N, params.g, params.terms, params.couplings,
                                 params.paulis, params.bias_z, params.reps, ops)


# =============================================================================
# 3. Choi-flip construction
# =============================================================================

def build_choi_prep_circuit(N: int, g: float, terms, couplings, paulis, bias_z: np.ndarray,
                             reps: int) -> QuantumCircuit:
    """2N-qubit circuit preparing |Psi_E> = (I_A tensor U_EOC)|Phi>.
    Register A = qubits [0..N-1], register B = qubits [N..2N-1]."""
    qc = QuantumCircuit(2 * N)
    for q in range(N):
        qc.h(q)
        qc.cx(q, N + q)
    layer_qc = QuantumCircuit(N)
    for _ in range(reps):
        msc.mixed_layer(layer_qc, N, g, terms, couplings, paulis, bias_z)
    qc.compose(layer_qc, qubits=list(range(N, 2 * N)), inplace=True)
    return qc


def choi_statevector(N: int, g: float, terms, couplings, paulis, bias_z: np.ndarray,
                      reps: int) -> Statevector:
    """Exact |Psi_E> (dimension 2**(2N)). IDEAL/pure -- see the notebook's
    'ideal vs. hardware Choi state' section: this is only what an IDEAL
    (noiseless) reservoir channel produces. A real-hardware acquisition
    (`shadow_measurements.run_shadow_ibm`) shadows the actual, generally
    MIXED, noisy-channel Choi state, not this object."""
    qc = build_choi_prep_circuit(N, g, terms, couplings, paulis, bias_z, reps)
    return Statevector.from_instruction(qc)


def choi_statevector_from_params(params) -> Statevector:
    return choi_statevector(params.N, params.g, params.terms, params.couplings, params.paulis,
                             params.bias_z, params.reps)


# ---- dense operator embedding (validation only; scales as 4**N) -----------

def _rho_matrix(per_q: np.ndarray, transpose: bool = False) -> np.ndarray:
    """Dense N-qubit rho_x = |phi_x><phi_x| for the REAL Ry(theta)|0> product
    encoding. Qubit q contributes bit q (higher-index qubit = more-significant/
    leftmost kron factor, Qiskit's convention)."""
    N = len(per_q)
    vec = np.array([[1.0]], dtype=np.complex128)
    for q in range(N - 1, -1, -1):
        c, s = np.cos(per_q[q] / 2), np.sin(per_q[q] / 2)
        vec = np.kron(vec, np.array([[c], [s]], dtype=np.complex128))
    rho = vec @ vec.conj().T
    return rho.T if transpose else rho


def rho_matrix_general(kets_per_qubit: Sequence[np.ndarray], transpose: bool = False) -> np.ndarray:
    """Dense N-qubit rho_x = |phi_x><phi_x| for an ARBITRARY (possibly
    complex) product of single-qubit kets -- used ONLY by the complex-state
    transpose unit test (Section 7 of the audit); the QELM's own encoding
    always uses the real-amplitude `_rho_matrix` above. `transpose` is the
    LITERAL matrix transpose (no conjugation) -- for a complex ket this
    differs from `rho` itself, unlike the real-encoding case."""
    N = len(kets_per_qubit)
    vec = np.array([[1.0]], dtype=np.complex128)
    for q in range(N - 1, -1, -1):
        ket = np.asarray(kets_per_qubit[q], dtype=np.complex128).reshape(2, 1)
        vec = np.kron(vec, ket)
    rho = vec @ vec.conj().T
    return rho.T if transpose else rho


def rz_ry_ket(theta: float, phi: float) -> np.ndarray:
    """|psi> = Rz(phi) Ry(theta) |0> = (e^{-i phi/2} cos(theta/2),
    e^{+i phi/2} sin(theta/2)). Complex whenever phi != 0 (mod 2*pi) -- used
    ONLY for the complex-state transpose unit test, never for the QELM's own
    encoding (which stays real Ry(pi*u)|0>, preserved exactly as in notebook
    4)."""
    return np.array([np.exp(-1j * phi / 2) * np.cos(theta / 2),
                      np.exp(1j * phi / 2) * np.sin(theta / 2)], dtype=np.complex128)


def direct_qelm_statevector_complex_encoding(thetas, phis, N: int, g: float, terms, couplings,
                                              paulis, bias_z: np.ndarray, reps: int) -> Statevector:
    """Direct QELM reference using the Rz(phi)Ry(theta)|0> COMPLEX encoding,
    for the transpose unit test only."""
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.ry(float(thetas[i]), i)
        qc.rz(float(phis[i]), i)
    for _ in range(reps):
        msc.mixed_layer(qc, N, g, terms, couplings, paulis, bias_z)
    return Statevector.from_instruction(qc)


def direct_qelm_features_complex_encoding(thetas, phis, N: int, g: float, terms, couplings, paulis,
                                           bias_z: np.ndarray, reps: int, ops) -> np.ndarray:
    sv = direct_qelm_statevector_complex_encoding(thetas, phis, N, g, terms, couplings, paulis,
                                                   bias_z, reps)
    return np.array([np.real(sv.expectation_value(op, qargs)) for op, qargs in ops])


def _embed_pauli_dense(N: int, op: Pauli, qargs: Sequence[int]) -> np.ndarray:
    """Dense N-qubit matrix for `op` (a k-qubit Pauli) acting on `qargs`,
    identity elsewhere. Built via Qiskit's OWN `Operator.compose(..., qargs=)`
    embedding -- GUARANTEED to use the identical qubit-index convention as
    `Statevector.expectation_value(op, qargs)` and
    `QuantumCircuit.save_expectation_value(op, qargs)`. Cross-checked directly
    against `Statevector.expectation_value` in the notebook's unit tests."""
    full = Operator(np.eye(2 ** N, dtype=np.complex128))
    full = full.compose(Operator(op), qargs=list(qargs), front=True)
    return full.data


def exact_choi_feature_both_orderings(psi_choi: Statevector, N: int, rho_x_T: np.ndarray,
                                       op: Pauli, qargs: Sequence[int]):
    """Returns (val_B_major, val_A_major): d*Tr[J_E * joint] for BOTH candidate
    tensor-ordering conventions,
        B_major: joint = kron(M_B, M_A)   (B's qubits more significant)
        A_major: joint = kron(M_A, M_B)   (A's qubits more significant)
    so the notebook can explicitly determine (not assume) which one matches
    the direct QELM reference, per the audit's requirement not to assume the
    ordering."""
    d = 2 ** N
    M_A = rho_x_T
    M_B = _embed_pauli_dense(N, op, list(qargs))
    val_B_major = d * np.vdot(psi_choi.data, (np.kron(M_B, M_A) @ psi_choi.data))
    val_A_major = d * np.vdot(psi_choi.data, (np.kron(M_A, M_B) @ psi_choi.data))
    for v in (val_B_major, val_A_major):
        assert abs(v.imag) < 1e-6, f'expected near-real expectation value, got imag={v.imag:.2e}'
    return float(val_B_major.real), float(val_A_major.real)


def exact_choi_feature(psi_choi: Statevector, N: int, per_q: np.ndarray, op: Pauli,
                        qargs: Sequence[int]) -> float:
    """d * Tr[J_E (rho_x^T tensor O_j)] using the B-major convention verified
    correct by `exact_choi_feature_both_orderings` in the notebook's Section 7
    (NOT re-verified on every call here -- that would defeat the point of a
    one-time convention check)."""
    rho_x_T = _rho_matrix(per_q, transpose=True)
    val_B_major, _ = exact_choi_feature_both_orderings(psi_choi, N, rho_x_T, op, qargs)
    return val_B_major


def exact_choi_feature_complex(psi_choi: Statevector, N: int, thetas, phis, op: Pauli,
                                qargs: Sequence[int], use_transpose: bool) -> float:
    """Same Choi-flip identity but for the complex Rz(phi)Ry(theta)|0>
    encoding, with the transpose EXPLICITLY toggleable -- `use_transpose=True`
    must match `direct_qelm_features_complex_encoding`; `use_transpose=False`
    (i.e. using rho_x itself, not rho_x^T) is expected to FAIL for phi != 0,
    proving the transpose is load-bearing rather than silently ignored."""
    kets = [rz_ry_ket(thetas[i], phis[i]) for i in range(N)]
    rho_x = rho_matrix_general(kets, transpose=use_transpose)
    d = 2 ** N
    M_B = _embed_pauli_dense(N, op, list(qargs))
    val = d * np.vdot(psi_choi.data, (np.kron(M_B, rho_x) @ psi_choi.data))
    return float(val.real)


# =============================================================================
# 4. Classical-shadow acquisition of the (frozen, input-independent) Choi state
# =============================================================================
# Thin, Choi-specific wrappers over the reservoir-agnostic primitives in
# `shadow_measurements.py` (basis rotations, Born sampling, hardware circuits).

def sample_choi_shadow_exact(psi_choi: Statevector, N2: int, n_snapshots: int,
                              rng: np.random.RandomState, strategy: str = 'uniform_pauli'):
    """Local-Pauli classical shadow of |Psi_E>. Delegates to
    `shadow_measurements.sample_shadow_exact`, which applies basis rotations
    via O(dim) reshape/tensordot per qubit rather than an O(dim^2) dense kron
    -- required for this to be tractable at the SCIENCE_CONFIG scale
    (N=6, 2N=12, dim=4096: ~0.5 ms/snapshot measured, vs. minutes/snapshot for
    the dense-kron approach an earlier version of this module used)."""
    return sm.sample_shadow_exact(psi_choi.data, N2, n_snapshots, rng, strategy=strategy)


# =============================================================================
# 5. Classical-only shadow estimator (frozen deployment math)
# =============================================================================

def a_factor_batch(bases_A: np.ndarray, signs_A: np.ndarray, per_q: np.ndarray) -> np.ndarray:
    """Input-DEPENDENT register-A factor for every snapshot, given the new
    input's encoding angles `per_q` (length N). Closed-form single-qubit
    shadow-inversion estimator:
        Tr[rho_q(x)^T rhohat_q] = 3*<b,s|rho_q(x)^T|b,s> - 1 = (3*s*r_b + 1)/2,
    since <b,s|rho|b,s> = (1+s*r_b)/2 for rho=(I+r.sigma)/2. r_b is
    rho_q(x)^T's Bloch component along the measured basis: r_Z=cos(theta),
    r_X=sin(theta), r_Y=0 (real encoding => transpose leaves the state
    unchanged). Returns shape (K,)."""
    K, N = bases_A.shape
    rx, rz = bloch_xz(per_q)
    r_b_table = np.zeros((N, 3))
    r_b_table[:, 0] = rz   # Z
    r_b_table[:, 1] = rx   # X
    r_b_table[:, 2] = 0.0  # Y
    r_b = r_b_table[np.arange(N)[None, :], bases_A]   # (K, N)
    per_qubit = (3.0 * signs_A * r_b + 1.0) / 2.0
    return np.prod(per_qubit, axis=1)


# =============================================================================
# 6. Frozen, serializable, QPU-free deployment object
# =============================================================================

@dataclass
class ChoiShadowDeployment:
    """Everything needed to turn a NEW input window into QELM features (or a
    single trained-readout prediction), without ever touching a quantum
    circuit, simulator, or QPU again. Built ONCE from a frozen classical
    shadow of the Choi state.

    This object is TASK-AGNOSTIC (Section 11's "architecture A"): the same
    frozen (bases_A, signs_A, b_factors) can back `predict_features` (full
    reconstruction, for diagnostics) or `predict_compressed` (a single
    weighted-observable readout, for deployment) for ANY later-trained
    ridge readout, without re-acquiring the shadow."""
    N: int
    window_size: int
    labels: list
    bases_A: np.ndarray    # (K, N)  int8, register-A shadow bases
    signs_A: np.ndarray    # (K, N)  int8, register-A shadow outcomes
    b_factors: np.ndarray  # (K, n_obs) float64, precomputed register-B factors
    n_groups_mom: int = 20

    @property
    def d(self) -> int:
        return 2 ** self.N

    @property
    def n_snapshots(self) -> int:
        return self.bases_A.shape[0]

    def _aggregate(self, contrib: np.ndarray, estimator: str) -> np.ndarray:
        """contrib: (K,) or (K, n_out). Returns scalar or (n_out,)."""
        if estimator == 'mean':
            return self.d * contrib.mean(axis=0)
        elif estimator == 'median_of_means':
            groups = np.array_split(np.arange(self.n_snapshots),
                                     max(1, min(self.n_groups_mom, self.n_snapshots)))
            group_means = np.array([contrib[g].mean(axis=0) for g in groups if len(g) > 0])
            return self.d * np.median(group_means, axis=0)
        else:
            raise ValueError(f'unknown estimator {estimator!r}')

    def transform_full_features(self, window_values: Sequence[float], estimator: str = 'mean') -> np.ndarray:
        """QPU-free: full feature-vector reconstruction (all n_obs readout
        observables) from the frozen shadow. O(K * n_obs) per call -- kept for
        diagnostics/comparison; `predict_compressed` is the recommended
        deployment path (Section 10 of the audit)."""
        per_q = window_angles(window_values, self.N)
        a = a_factor_batch(self.bases_A, self.signs_A, per_q)   # (K,)
        contrib = a[:, None] * self.b_factors                    # (K, n_obs)
        return self._aggregate(contrib, estimator)

    # backward-compatible alias
    def predict_features(self, window_values: Sequence[float], estimator: str = 'mean') -> np.ndarray:
        return self.transform_full_features(window_values, estimator=estimator)

    def precompute_bW(self, weights: np.ndarray) -> np.ndarray:
        """Input-INDEPENDENT: b_W[k] = sum_j weights[j] * b_factors[k,j] --
        the register-B factor for the SINGLE weighted observable O_W = sum_j
        weights[j] * O_j. Compute ONCE after ridge training; O(K * n_obs)
        one-time cost, replacing an O(K * n_obs) cost on EVERY future
        prediction with an O(K) cost."""
        return self.b_factors @ np.asarray(weights, dtype=np.float64)

    def predict_compressed(self, window_values: Sequence[float], b_W: np.ndarray, intercept: float,
                            estimator: str = 'mean') -> float:
        """QPU-free, READOUT-COMPRESSED prediction:
            y_hat(x) = intercept + d * mean_or_mom(a_factor(x) * b_W)
        `b_W` = `self.precompute_bW(weights)`, precomputed ONCE. O(K) per
        call -- no (K, n_obs) matrix is built at inference time. This is the
        DEFAULT deployment path (Section 10 of the audit)."""
        per_q = window_angles(window_values, self.N)
        a = a_factor_batch(self.bases_A, self.signs_A, per_q)   # (K,)
        contrib = a * b_W                                        # (K,)
        return float(intercept + self._aggregate(contrib, estimator))

    def save(self, path: str) -> None:
        meta = {'N': self.N, 'window_size': self.window_size, 'labels': self.labels,
                'n_groups_mom': self.n_groups_mom}
        np.savez_compressed(path, bases_A=self.bases_A, signs_A=self.signs_A,
                             b_factors=self.b_factors, meta=json.dumps(meta))

    @classmethod
    def load(cls, path: str) -> 'ChoiShadowDeployment':
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data['meta']))
        return cls(N=meta['N'], window_size=meta['window_size'], labels=meta['labels'],
                   bases_A=data['bases_A'], signs_A=data['signs_A'], b_factors=data['b_factors'],
                   n_groups_mom=meta['n_groups_mom'])


def build_deployment_from_exact_shadow(N: int, window_size: int, psi_choi: Statevector, ops, labels,
                                        n_snapshots: int, seed: int, n_groups_mom: int = 20
                                        ) -> ChoiShadowDeployment:
    """Convenience constructor: draw an exact-statevector-based finite-shot
    Choi shadow and immediately package it as a frozen deployment object."""
    rng = np.random.RandomState(seed)
    bases, signs = sample_choi_shadow_exact(psi_choi, 2 * N, n_snapshots, rng)
    bases_A, signs_A = bases[:, :N], signs[:, :N]
    bases_B, signs_B = bases[:, N:], signs[:, N:]  # local index within B (0..N-1)
    b_factors = sm.precompute_b_factors(bases_B, signs_B, ops)
    return ChoiShadowDeployment(N=N, window_size=window_size, labels=list(labels),
                                 bases_A=bases_A, signs_A=signs_A, b_factors=b_factors,
                                 n_groups_mom=n_groups_mom)


# =============================================================================
# 7. Ridge-readout compression: O_W = sum_j w_j O_j, and O_eff = U^dagger O_W U
# =============================================================================

def effective_linear_weights(ridge_coef: np.ndarray, ridge_intercept: float,
                              scaler_mean: np.ndarray, scaler_scale: np.ndarray):
    """A ridge model trained on STANDARDIZED features (`select_and_eval_ridge`,
    `qrc_qiskit.py`, reused unmodified) is
        y = intercept + sum_j coef_j * (X_j - mean_j) / scale_j
          = (intercept - sum_j coef_j*mean_j/scale_j) + sum_j (coef_j/scale_j) * X_j
    i.e. still LINEAR in the raw features X_j = f_j(x), just with rescaled
    weights/intercept. Returns (w_eff, b_eff) so O_W = sum_j w_eff[j] * O_j
    reproduces the trained model's prediction from the RAW (unstandardized)
    Choi-flip features exactly."""
    w_eff = ridge_coef / scaler_scale
    b_eff = ridge_intercept - float(np.sum(ridge_coef * scaler_mean / scaler_scale))
    return w_eff, b_eff


def build_OW_dense(N: int, ops, weights: np.ndarray) -> np.ndarray:
    """Dense N-qubit matrix for O_W = sum_j weights[j] * O_j. Feasible only at
    small/moderate N (validation: N=4, dim=16; SCIENCE_CONFIG: N=6, dim=64 --
    both trivial; this is NOT meant to scale to large N, it exists purely to
    make the O_eff = U^dagger O_W U identity (Section 11 of the audit)
    numerically checkable)."""
    d = 2 ** N
    O_W = np.zeros((d, d), dtype=np.complex128)
    for j, (op, qargs) in enumerate(ops):
        O_W = O_W + weights[j] * _embed_pauli_dense(N, op, list(qargs))
    return O_W


def build_O_eff(N: int, g: float, terms, couplings, paulis, bias_z: np.ndarray, reps: int,
                 O_W: np.ndarray) -> np.ndarray:
    """O_eff = U^dagger O_W U, where U is the SAME (reps-fold) mixed_layer
    unitary the QELM reservoir uses. Once trained, the QELM predictor is
    exactly the QUANTUM LINEAR MODEL f(x) = Tr[rho_x O_eff] + b -- verified in
    the notebook by comparing Tr[rho_x O_eff]+b against the trained
    prediction directly, at a handful of input windows."""
    qc = QuantumCircuit(N)
    for _ in range(reps):
        msc.mixed_layer(qc, N, g, terms, couplings, paulis, bias_z)
    U = Operator(qc).data
    return U.conj().T @ O_W @ U


# =============================================================================
# 8. Reservoir-untouched-by-training guard
# =============================================================================

def assert_reservoir_unchanged(params_before, ops_labels_before: Sequence[str], params_after,
                                ops_labels_after: Sequence[str]) -> str:
    """Fingerprint-based guard (Section 5 of the audit): hashes EVERY number
    defining U_EOC, the QELM encoding, and the readout observable set, before
    and after classical-readout training, and raises unless they match
    exactly. Stronger than checking whether a sampling function merely LACKS
    a parameter named `y`/`label`/`target` -- this proves the actual sampled
    numbers (disorder, couplings, Pauli types, ...) were never touched."""
    import eoc_config as ec
    fp_before = ec.fingerprint(params_before, ops_labels_before)
    fp_after = ec.fingerprint(params_after, ops_labels_after)
    assert fp_before == fp_after, (
        f'Reservoir fingerprint CHANGED during readout training: {fp_before} -> {fp_after}. '
        f'The mixed-SYK reservoir must never be touched by classical-readout fitting.')
    return fp_after


# =============================================================================
# 9. Quantum-call instrumentation (for the no-QPU inference test)
# =============================================================================
# Deliberately NOT implemented as a generic cross-module monkeypatch helper
# here: `unittest.mock.patch.object(jerbi_shadow, 'QuantumCircuit', ...)`
# patches the NAME AS BOUND IN THIS MODULE's namespace (created by this
# module's own `from qiskit import QuantumCircuit`), which is exactly what
# every call site in this file actually resolves at call time. Patching
# `qiskit.QuantumCircuit` itself would NOT intercept those calls (Python
# binds `from X import Y` to a fresh local name, not a live alias) -- an
# earlier draft of this module got this wrong. The notebook's no-QPU test
# therefore patches `jerbi_shadow.QuantumCircuit` / `jerbi_shadow.Statevector`
# / `mixed_syk_core.QuantumCircuit` directly (with `wraps=` to additionally
# COUNT calls, not just block them) -- see its "QPU-free inference" section.
