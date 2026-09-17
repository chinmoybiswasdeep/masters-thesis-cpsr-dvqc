"""
shadow_measurements.py -- local-Pauli classical-shadow measurement primitives:
basis rotations, Born sampling of a fixed statevector, the single-shot
shadow-inversion estimator, hardware/simulator circuit builders, and a
measurement-strategy scaffold.

This module is deliberately physics-only and reservoir-agnostic: it knows
nothing about the mixed-SYK channel or the Choi construction (that lives in
`jerbi_shadow.py`) -- it only implements "take a classical shadow of a fixed
quantum state/circuit using random local Pauli measurements", so its
correctness can be (and is, in the notebook's Section on basis unit tests)
checked completely independently of the Choi-flip identity.

BASIS-ROTATION CONVENTION: reused verbatim from this repo's own, already
bug-fixed convention (`CPSR_Project_IBM_Qiskit_reviewed_2.ipynb`, comment 6:
Y-basis pre-rotation is `H . Sdg`, i.e. state-transform `_HSdg = _H2 @ _SDG2`,
realized on hardware as "Sdg then H"). Reusing this exact matrix, rather than
re-deriving one, is what prevents that bug from reappearing here.

SHADOW-INVERSION ESTIMATOR: for a single qubit measured in random basis b in
{Z,X,Y} with outcome sign s in {+1,-1}, the classical-shadow single-qubit
estimator of ANY Hermitian operator O is
    Tr[O * rhohat] = 3*<b,s|O|b,s> - Tr[O]     (Huang, Kueng & Preskill 2020).
For O a single Pauli string of weight w (as needed for the readout
observables O_j), this reduces to the familiar "3^w * sign product if every
qubit's basis matches, else 0" form (`precompute_b_factors`). For O a GENERAL
single-qubit operator (as needed for the transposed input density matrix
rho_x^T, which is not a single Pauli), the estimator is evaluated directly
from its 2x2 matrix elements (`a_factor_batch`) -- both are exercised and
checked against exact values in the notebook's dedicated basis unit tests.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# =============================================================================
# Basis rotations (state-transform convention: apply to |psi> BEFORE measuring
# in the computational (Z) basis).
# =============================================================================
_H2 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SDG2 = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_HSdg = _H2 @ _SDG2                      # correct Y-basis pre-rotation (state transform)
_I2 = np.eye(2, dtype=np.complex128)
_BASIS_ROT = {0: _I2, 1: _H2, 2: _HSdg}  # 0=Z, 1=X, 2=Y
_BASIS_NAME = {0: 'Z', 1: 'X', 2: 'Y'}


def apply_single_qubit_gate(state: np.ndarray, U2: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` of an `n_qubits`-qubit statevector via
    reshape/tensordot -- O(dim) per call, NOT O(dim^2) (no dense kron over the
    full register). Verified to match `Statevector`/`QuantumCircuit`'s own
    qubit-index convention exactly (machine precision, all qubits, N=2..5) in
    the notebook's basis unit tests. This is what makes Born-sampling a
    classical shadow of a 2N-qubit Choi state (2N up to ~12-14) tractable --
    the naive `np.kron`-per-basis-group approach used by an earlier version of
    this code was O(dim^2) per snapshot and unusable once 2N is large enough
    that almost every random basis is unique (3^(2N) >> K)."""
    dim = state.shape[0]
    axis = n_qubits - 1 - qubit
    s = state.reshape([2] * n_qubits)
    s = np.moveaxis(s, axis, 0)
    s = np.tensordot(U2, s, axes=([1], [0]))
    s = np.moveaxis(s, 0, axis)
    return s.reshape(dim)


def apply_basis_rotation(state: np.ndarray, bases: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply `_BASIS_ROT[bases[q]]` to every qubit q of `state`, in place of a
    dense `n_qubits`-fold kron -- O(n_qubits * dim)."""
    s = state
    for q in range(n_qubits):
        s = apply_single_qubit_gate(s, _BASIS_ROT[int(bases[q])], q, n_qubits)
    return s


# =============================================================================
# Exact-statevector (finite-shot) Born sampling of a classical shadow.
# =============================================================================

def sample_shadow_exact(state: np.ndarray, n_qubits: int, n_snapshots: int, rng: np.random.RandomState,
                         strategy: str = 'uniform_pauli'):
    """Local-Pauli classical shadow of a FIXED statevector `state`: one
    independent random basis per qubit per snapshot, Born-sampled outcome.
    Returns (bases, signs), each (n_snapshots, n_qubits); `signs[k,q] =
    1-2*bit`.

    `strategy` is a scaffold for future measurement-allocation strategies
    (Section 18 of the audit: derandomization / observable-aware / biased
    sampling can all reduce required shots for a KNOWN, fixed observable set,
    at the cost of the estimator no longer being basis-agnostic). Only
    'uniform_pauli' (i.i.d. uniform over {X,Y,Z} per qubit -- the textbook
    Huang et al. protocol) is implemented and used anywhere in this project;
    the others raise `NotImplementedError` rather than silently falling back,
    so a caller can never be misled into thinking a claimed optimization ran.
    """
    if strategy != 'uniform_pauli':
        raise NotImplementedError(
            f"measurement_strategy={strategy!r} is not implemented. Only 'uniform_pauli' "
            "(i.i.d. uniform local-Pauli, Huang et al. 2020) is implemented in this project. "
            "See the notebook's measurement-efficiency future-work section for candidates "
            "(biased Pauli sampling, derandomized shadows, observable-aware allocation, "
            "light-cone-truncated shadows) -- none are implemented here without numerical "
            "evidence they are unbiased and actually cheaper.")

    dim = state.shape[0]
    bases = rng.randint(0, 3, size=(n_snapshots, n_qubits)).astype(np.int8)
    signs = np.empty((n_snapshots, n_qubits), dtype=np.int8)
    idx_all = np.arange(dim)
    bits_table = ((idx_all[:, None] >> np.arange(n_qubits)) & 1).astype(np.int8)  # (dim, n_qubits)
    for k in range(n_snapshots):
        s = apply_basis_rotation(state, bases[k], n_qubits)
        probs = np.abs(s) ** 2
        probs /= probs.sum()
        outcome = rng.choice(dim, p=probs)
        signs[k] = 1 - 2 * bits_table[outcome]
    return bases, signs


# =============================================================================
# Classical-shadow inversion estimators
# =============================================================================

def single_qubit_shadow_estimate(basis: int, sign: int, O2: np.ndarray) -> float:
    """Tr[O2 * rhohat] = 3*<b,s|O2|b,s> - Tr[O2] for a SINGLE qubit's shadow
    snapshot (basis, sign) and an ARBITRARY 2x2 Hermitian `O2` -- the general
    single-qubit shadow-inversion formula (Huang et al. 2020), not restricted
    to O2 being a Pauli matrix. Used directly by the basis unit tests; the
    batched, closed-form version for O2 being a QELM input density matrix is
    `a_factor_batch` in `jerbi_shadow.py`."""
    ket = np.zeros(2, dtype=np.complex128)
    if basis == 0:      # Z
        ket[0 if sign == 1 else 1] = 1.0
    elif basis == 1:    # X
        ket[:] = np.array([1, sign]) / np.sqrt(2)
    elif basis == 2:    # Y
        ket[:] = np.array([1, 1j * sign]) / np.sqrt(2)
    else:
        raise ValueError(basis)
    expval = np.real(np.conj(ket) @ O2 @ ket)
    return float(3.0 * expval - np.real(np.trace(O2)))


def estimate_pauli_expectation(bases: np.ndarray, signs: np.ndarray, pauli_char: str,
                                qubit: int, n_groups_mom: int = 1) -> float:
    """Single-qubit <P> estimate (P in X/Y/Z) from a shadow's `bases`/`signs`
    columns at `qubit`, using ONLY snapshots whose basis matches P.

    NOTE on the missing factor of 3: the familiar '3^w * sign, else 0'
    classical-shadow estimator (as `precompute_b_factors` uses) is unbiased
    when averaged over ALL K snapshots, INCLUDING the ~2/3 that don't match
    (the factor of 3 exactly compensates for the 1-in-3 chance of measuring
    the right basis). Here we instead explicitly DISCARD non-matching shots
    and average only the matching subset -- conditioned on basis==P, `sign`
    is already an unbiased estimator of <P> with NO extra factor of 3 needed
    (an earlier version of this function multiplied by 3 here too, which is
    the wrong formula for a discard-non-matching estimator -- caught by
    exactly the |0>/|+>/|+i> unit tests this function exists to run, which
    is the point of having an independently-coded cross-check). Used by the
    notebook's basis unit tests, deliberately implemented independently of
    `precompute_b_factors`'s vectorized zero-fill form."""
    type_to_int = {'Z': 0, 'X': 1, 'Y': 2}
    b = type_to_int[pauli_char]
    mask = bases[:, qubit] == b
    if not np.any(mask):
        return float('nan')
    matching_signs = signs[mask, qubit].astype(np.float64)
    if n_groups_mom <= 1:
        return float(np.mean(matching_signs))
    groups = np.array_split(matching_signs, min(n_groups_mom, len(matching_signs)))
    means = [float(np.mean(g)) for g in groups if len(g) > 0]
    return float(np.median(means))


def precompute_b_factors(bases_B: np.ndarray, signs_B: np.ndarray, ops) -> np.ndarray:
    """Input-INDEPENDENT per-observable, per-snapshot factor:
        b_factor[k, j] = 3^{w_j} * prod_{q in support(O_j)} signs_B[k,q]
                         if bases_B[k, q] == type(O_j, q) for all q in support,
                         else 0.
    Shape (n_snapshots, n_obs). Computed ONCE from the frozen shadow; reused
    for every future input x."""
    K = bases_B.shape[0]
    n_obs = len(ops)
    out = np.zeros((K, n_obs), dtype=np.float64)
    type_to_int = {'Z': 0, 'X': 1, 'Y': 2}
    for j, (op, qargs) in enumerate(ops):
        label = op.to_label()  # label[0] corresponds to qargs[-1] (Qiskit convention)
        qargs = list(qargs)
        types = [type_to_int[c] for c in reversed(label)]  # types[i] <-> qargs[i]
        w = len(qargs)
        match = np.ones(K, dtype=bool)
        prod_signs = np.ones(K, dtype=np.int64)
        for i, q in enumerate(qargs):
            match &= (bases_B[:, q] == types[i])
            prod_signs *= signs_B[:, q]
        out[:, j] = np.where(match, (3.0 ** w) * prod_signs, 0.0)
    return out


def median_of_means(values: np.ndarray, n_groups: int) -> float:
    """Huang-Kueng-Preskill median-of-means: split into `n_groups` equal(ish)
    groups, average each, take the median of the group means."""
    n_groups = max(1, min(n_groups, len(values)))
    groups = np.array_split(values, n_groups)
    means = [float(np.mean(g)) for g in groups if len(g) > 0]
    return float(np.median(means))


def theoretical_shadow_norm_sq(weight: int) -> float:
    """||O||_shadow^2 = 3^k EXACTLY for O a tensor product of k non-identity
    single-qubit Pauli operators (Huang et al. 2020, Lemma S3/Eq. S50). This
    is a precise statement about a SINGLE weight-k Pauli string. The Choi-flip
    observable rho_x^T tensor O_j is NOT a single Pauli string on register A
    (rho_x^T is a general product density matrix, a linear combination of
    many Pauli strings) -- see the notebook's honest-wording section for why
    a single exact 3^(N+w) claim about the FULL observable would overstate
    what this formula proves, and why the empirical convergence sweep (not
    this formula alone) is what the notebook's shot-budget claims rest on."""
    return 3.0 ** weight


# =============================================================================
# Hardware / simulator circuit builders (measurement side only -- the
# reservoir/Choi-prep circuit itself is built by `jerbi_shadow.py` and passed
# in here unchanged).
# =============================================================================

def build_shadow_measurement_circuits(prep_circuit, n_qubits: int, bases: np.ndarray):
    """One measurement circuit per row of `bases` (n_snapshots, n_qubits):
    appends the basis pre-rotation (Z: nothing, X: H, Y: Sdg then H == state
    transform H.Sdg) and a measurement to `prep_circuit`. This is the ONLY
    place this project builds circuits meant for real hardware or a noisy
    simulator -- kept architecturally separate from the exact/statevector
    shadow path (`sample_shadow_exact`)."""
    from qiskit import ClassicalRegister
    circuits = []
    for basis in bases:
        qc = prep_circuit.copy()
        qc.add_register(ClassicalRegister(n_qubits))
        for q in range(n_qubits):
            b = int(basis[q])
            if b == 1:
                qc.h(q)
            elif b == 2:
                qc.sdg(q); qc.h(q)   # Sdg then H == state transform H.Sdg (_HSdg)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits.append(qc)
    return circuits


def run_shadow_ibm(prep_circuit, n_qubits: int, n_snapshots: int, rng: np.random.RandomState,
                    backend_name: str | None = None, service=None, shots_per_circuit: int = 1):
    """Acquire a REAL classical shadow on IBM hardware. Credentials load only
    from environment variables via `qrc_qiskit.get_ibm_service` (never
    hard-coded); the backend is pinned by name (no silent `least_busy()`),
    per this project's established conventions. Runs exactly ONCE, during the
    offline acquisition/"advice" stage -- nothing on the deployment path
    imports this function.

    IMPORTANT: the effective channel realized on real hardware is noisy,
    E~_tilde != E_ideal, so the Choi state actually being shadow-measured here,
    J_{E~tilde}, is generally MIXED, not the pure |Psi_E><Psi_E| the exact/
    simulated path assumes. Classical shadows estimate observables of
    whatever state is actually prepared (mixed or pure) -- nothing about the
    shadow PROTOCOL requires purity -- but this function's output should never
    be described as "a shadow of the ideal Choi state."
    """
    from qrc_qiskit import get_ibm_service, DEFAULT_QPU_BACKEND
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    bases = rng.randint(0, 3, size=(n_snapshots, n_qubits)).astype(np.int8)
    circuits = build_shadow_measurement_circuits(prep_circuit, n_qubits, bases)

    service = service or get_ibm_service()
    backend = service.backend(backend_name or DEFAULT_QPU_BACKEND)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = [pm.run(qc) for qc in circuits]
    isa_depths = [ic.depth() for ic in isa_circuits]
    isa_2q = [sum(v for k, v in ic.count_ops().items() if k in ('cz', 'ecr', 'cx', 'rzz'))
              for ic in isa_circuits]

    sampler = SamplerV2(mode=backend)
    import time
    t0 = time.perf_counter()
    job = sampler.run(isa_circuits, shots=shots_per_circuit)
    result = job.result()
    elapsed = time.perf_counter() - t0

    signs = np.empty((n_snapshots, n_qubits), dtype=np.int8)
    for k, pub_result in enumerate(result):
        creg_data = list(pub_result.data.values())[0]
        bitstrings = creg_data.get_bitstrings()
        bits = np.array([int(c) for c in bitstrings[0][::-1]], dtype=np.int8)  # qubit q -> classical bit q
        signs[k] = 1 - 2 * bits

    info = {
        'backend': backend.name, 'job_id': job.job_id(), 'n_snapshots': n_snapshots,
        'shots_per_circuit': shots_per_circuit, 'elapsed_s': elapsed,
        'n_unique_basis_circuits': int(len(circuits)),
        'transpiled_depth_mean': float(np.mean(isa_depths)), 'transpiled_depth_max': int(np.max(isa_depths)),
        'two_qubit_gates_mean': float(np.mean(isa_2q)), 'two_qubit_gates_total': int(np.sum(isa_2q)),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'This is a shadow of the NOISY hardware-effective channel, not the ideal Choi state.',
    }
    return bases, signs, info


def run_shadow_aer_hardware_path(prep_circuit, n_qubits: int, n_snapshots: int,
                                  rng: np.random.RandomState, sim=None, noisy: bool = False):
    """Runs the SAME per-snapshot hardware-style circuits
    (`build_shadow_measurement_circuits`) on a LOCAL AerSimulator -- lets the
    hardware code PATH be exercised and unit-tested without IBM credentials.
    `noisy=True` uses a depolarizing-noise model instead of an ideal
    simulator, so this path can also stand in for "the Choi state is actually
    mixed" scenarios without needing real hardware."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    if sim is None:
        if noisy:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h', 'sdg', 'rz', 'ry'])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cx', 'rxx', 'ryy'])
            sim = AerSimulator(method='density_matrix', noise_model=noise_model)
        else:
            sim = AerSimulator(method='statevector')
    bases = rng.randint(0, 3, size=(n_snapshots, n_qubits)).astype(np.int8)
    circuits = build_shadow_measurement_circuits(prep_circuit, n_qubits, bases)
    signs = np.empty((n_snapshots, n_qubits), dtype=np.int8)
    for k, qc in enumerate(circuits):
        tqc = transpile(qc, sim, optimization_level=1)
        result = sim.run(tqc, shots=1).result()
        counts = result.get_counts(0)
        bitstring = next(iter(counts))
        bits = np.array([int(c) for c in bitstring[::-1]], dtype=np.int8)
        signs[k] = 1 - 2 * bits
    return bases, signs
