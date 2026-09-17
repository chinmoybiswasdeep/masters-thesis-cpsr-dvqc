"""
mixed_syk_core.py -- shared module extracting the mixed-SYK2(g)/SYK4(J) QELM
core (Sections 0c/0d/2 of `4_QR_MixedSYK_Qiskit.ipynb`) into an importable
module, so `5_QR_MixedSYK_JerbiShadow_Qiskit.ipynb` (the Jerbi-flipped-shadow
notebook) can reuse the EXACT mixed-SYK reservoir, QELM encoding and EOC
diagnostics rather than re-implementing them.

Every function body below is copied VERBATIM (same code, same seeds, same
conventions) from `4_QR_MixedSYK_Qiskit.ipynb`'s Sections 0c ("Mixed
SYK2(g)/SYK4(J) entangling layer"), 0d ("QELM wiring") and 2 ("Edge-of-chaos
diagnostics"). This module does NOT change any numerical result of notebook 4
-- it only relocates the code so it can be imported instead of copy-pasted a
second time. `5_QR_MixedSYK_JerbiShadow_Qiskit.ipynb` includes a regression
test that loads notebook 4's actual cell source at runtime (via `nbformat`-free
JSON parsing) and checks this module's functions produce bit-identical
unitaries for matched configurations/seeds -- see that notebook's Section 11.

Shared, architecture-agnostic pieces (ReservoirConfig, chain_edges,
make_simulator, chrono_split, nrmse, select_and_eval_ridge, memory_capacity,
task_kpauli, task_narma2, delay_target, delay_taps, random_input,
DEFAULT_ALPHAS) are imported from `qrc_qiskit.py` (Section 0a/0b of notebooks
1 and 4) rather than duplicated, per the same "one source of truth" principle
notebook 4 itself follows.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from scipy.stats import unitary_group
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Operator, Statevector, random_unitary

from qrc_qiskit import (  # noqa: F401 -- re-exported for convenience
    ReservoirConfig, chain_edges, make_simulator, chrono_split, nrmse,
    select_and_eval_ridge, memory_capacity, task_kpauli, task_narma2,
    delay_target, delay_taps, random_input, DEFAULT_ALPHAS,
)

# =============================================================================
# Section 0c -- Mixed SYK2(g)/SYK4(J) entangling layer
# (verbatim from `4_QR_MixedSYK_Qiskit.ipynb`, cell "Section 0c", 163 lines)
# =============================================================================

def default_n_sparse_terms(N: int) -> int:
    return int(np.ceil(N * np.log(N)))


def sample_syk4_terms(N: int, n_terms: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    all_tuples = list(itertools.combinations(range(N), 4))
    if n_terms >= len(all_tuples):
        return all_tuples
    idx = rng.choice(len(all_tuples), size=n_terms, replace=False)
    return [all_tuples[i] for i in idx]


def sample_syk4_couplings(n_terms: int, J: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + 1)
    return rng.normal(0.0, J, size=n_terms)


def sample_syk4_pauli_types(n_terms: int, seed: int) -> np.ndarray:
    """Random Pauli type ('X','Y', or 'Z') per qubit per quartic term -- the
    Jordan-Wigner image of a generic (not all-same-Majorana-type) SYK4 term;
    this is what makes the quartic interaction genuinely non-commuting/
    chaotic on its own, unlike a pure ZZZZ term."""
    rng = np.random.RandomState(seed + 500000)
    return rng.choice(['X', 'Y', 'Z'], size=(n_terms, 4))


def zzzz_rotation(qc: QuantumCircuit, qubits4, theta: float):
    """exp(-i*theta/2 * Z_a Z_b Z_c Z_d) via the phase-kickback gadget: 6 CNOTs
    + 1 Rz, exact (no Trotter error)."""
    a, b, c, d = qubits4
    qc.cx(a, d); qc.cx(b, d); qc.cx(c, d)
    qc.rz(theta, d)
    qc.cx(c, d); qc.cx(b, d); qc.cx(a, d)


def _basis_change_pre(qc: QuantumCircuit, q: int, p: str):
    """Rotate Pauli-p eigenbasis -> Z eigenbasis (V with V P V^dagger = Z)."""
    if p == 'X':
        qc.h(q)
    elif p == 'Y':
        qc.sdg(q); qc.h(q)
    # 'Z': identity, no gate


def _basis_change_post(qc: QuantumCircuit, q: int, p: str):
    """Undo _basis_change_pre (V^dagger)."""
    if p == 'X':
        qc.h(q)
    elif p == 'Y':
        qc.h(q); qc.s(q)
    # 'Z': identity, no gate


def pauli4_rotation(qc: QuantumCircuit, qubits4, paulis4, theta: float):
    """exp(-i*theta/2 * P_a P_b P_c P_d) for P in {X,Y,Z} per qubit: basis-
    change into the Z frame on each qubit, the SAME zzzz_rotation gadget,
    then undo the basis change. Reduces exactly to zzzz_rotation when
    paulis4 == ('Z','Z','Z','Z')."""
    for q, p in zip(qubits4, paulis4):
        _basis_change_pre(qc, q, p)
    zzzz_rotation(qc, qubits4, theta)
    for q, p in zip(qubits4, paulis4):
        _basis_change_post(qc, q, p)


def feature_ops_mem_all(N: int, input_qubit: int, max_weight: int = 3):
    """Full weight<=max_weight Pauli-string readout (Section 0a's own
    `feature_ops_all` idea, used there for the QELM architecture) restricted
    to the MEMORY qubits (excluding `input_qubit`, preserving `feature_ops`'s
    own convention). Fixes kPauli2/3 NRMSE, which sat at/above 1 under the
    weight<=2, all-Z-pair `feature_ops` basis -- see Section 0c's markdown."""
    mem = [q for q in range(N) if q != input_qubit]
    paulis_ = ('Z', 'X', 'Y')
    ops, labels = [], []
    for q in mem:
        for name in paulis_:
            ops.append((Pauli(name), [q]))
            labels.append(f'{name}{q}')
    if max_weight >= 2:
        for a, b in zip(mem[:-1], mem[1:]):
            for p1, p2 in itertools.product(paulis_, repeat=2):
                ops.append((Pauli(p1 + p2), [a, b]))
                labels.append(f'{p1}{a}{p2}{b}')
    if max_weight >= 3:
        for a, b, c in zip(mem[:-2], mem[1:-1], mem[2:]):
            for p1, p2, p3 in itertools.product(paulis_, repeat=3):
                ops.append((Pauli(p1 + p2 + p3), [a, b, c]))
                labels.append(f'{p1}{a}{p2}{b}{p3}{c}')
    return labels, ops


def mixed_layer(qc: QuantumCircuit, N: int, g: float, terms, couplings, paulis, bias_z):
    """SYK2-like (g, nearest-neighbor Rxx+Ryy free-fermion hopping) + SYK4-like
    (sparse random-Pauli-type 4-body rotations at scale J, already baked into
    `couplings`), then an Rz-only on-site kick. g=0 -> pure sparse random-
    Pauli-type quartic layer (chaotic alone); empty terms (J=0) -> pure
    free-fermion Rxx/Ryy chain + Rz (integrable alone)."""
    for a, b in chain_edges(N):
        qc.rxx(2.0 * g, a, b)
        qc.ryy(2.0 * g, a, b)
    for (i, j, k, l), Jt, ptypes in zip(terms, couplings, paulis):
        pauli4_rotation(qc, (i, j, k, l), ptypes, 2.0 * Jt)
    for i in range(N):
        qc.rz(bias_z[i], i)


def build_trajectory_circuit_mixed(cfg: ReservoirConfig, u_seq: Sequence[float], g: float, J: float,
                                    reps: int, n_terms=None, term_seed: int = 0, max_weight: int = 3):
    """Direct analogue of Section 0a's `build_trajectory_circuit`: reset +
    re-encode ONLY `input_qubit` each step (genuine recurrence), apply `reps`
    mixed g/J layers, snapshot exact features via `feature_ops_mem_all`."""
    bias_z, _bias_x_unused = cfg.sample_disorder()
    if n_terms is None:
        n_terms = default_n_sparse_terms(cfg.N)
    terms = sample_syk4_terms(cfg.N, n_terms, seed=term_seed)
    couplings = sample_syk4_couplings(n_terms, J=J, seed=term_seed)
    paulis = sample_syk4_pauli_types(n_terms, seed=term_seed)
    labels, ops = feature_ops_mem_all(cfg.N, cfg.input_qubit, max_weight=max_weight)
    qc = QuantumCircuit(cfg.N)
    for t, u_t in enumerate(u_seq):
        qc.reset(cfg.input_qubit)
        qc.ry(np.pi * float(u_t), cfg.input_qubit)
        for _ in range(reps):
            mixed_layer(qc, cfg.N, g, terms, couplings, paulis, bias_z)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f'{lab}__t{t}')
    return qc, labels, terms, couplings, paulis


def run_reservoir_mixed(cfg: ReservoirConfig, u_seq: Sequence[float], g: float, J: float, reps: int,
                         n_terms=None, term_seed: int = 0, use_gpu: bool = False,
                         method: str = 'density_matrix', max_weight: int = 3):
    """Same (labels, X, info) shape as Section 0a's `run_reservoir`."""
    import time
    qc, labels, terms, couplings, paulis = build_trajectory_circuit_mixed(
        cfg, u_seq, g, J, reps, n_terms, term_seed, max_weight)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    t0 = time.perf_counter()
    result = sim.run(tqc, shots=1).result()
    elapsed = time.perf_counter() - t0
    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f'{lab}__t{t}'])
    info = {'device': 'GPU' if use_gpu else 'CPU', 'method': method, 'architecture': 'mixed_recurrent',
            'N': cfg.N, 'T': T, 'elapsed_s': elapsed, 'circuit_depth': tqc.depth(),
            'n_features': len(labels), 'n_terms': len(terms)}
    return labels, X, info, (terms, couplings, paulis)


def kappa_to_gJ(kappa: float, G_MAX: float, J_MAX: float):
    """kappa=0 -> pure SYK4-like (g=0, J=J_MAX); kappa->inf -> pure SYK2-like
    (g=G_MAX, J=0). NOTE: this is a RATIONAL interpolation
    (g = G_MAX*kappa/(1+kappa), J = J_MAX/(1+kappa)), not the naive linear
    g=G_MAX*kappa / J=J_MAX*(1-kappa) -- the notebook's actual, established
    convention, reused here verbatim so kappa values are directly comparable
    to notebook 4's sweeps."""
    g = G_MAX * kappa / (1 + kappa)
    J = J_MAX / (1 + kappa)
    return g, J


# =============================================================================
# Section 0d -- Mixed SYK2(g)/SYK4(J) layer, QELM (window-encoded, memoryless)
# (verbatim from `4_QR_MixedSYK_Qiskit.ipynb`, cell "Section 0d", 131 lines)
# =============================================================================

def feature_ops_all_general(N: int, max_weight: int = 5):
    """Generalization of Section 0a's own `feature_ops_all` (weight<=3 only)
    to arbitrary `max_weight` via one general loop. Reduces to the SAME
    feature set as Section 0a's `feature_ops_all` for max_weight<=3."""
    paulis_ = ('Z', 'X', 'Y')
    ops, labels = [], []
    for w in range(1, max_weight + 1):
        for start in range(N - w + 1):
            qubits = list(range(start, start + w))
            for combo in itertools.product(paulis_, repeat=w):
                ops.append((Pauli(''.join(combo)), qubits))
                labels.append(''.join(f'{p}{q}' for p, q in zip(combo, qubits)))
    return labels, ops


def build_qelm_circuit_mixed(cfg: ReservoirConfig, u_seq: Sequence[float], g: float, J: float,
                              window_size: int = 8, max_weight: int = 5, reps: int = 1,
                              n_terms=None, term_seed: int = 0):
    """QELM-style trajectory (Section 0a's `build_qelm_circuit` convention:
    reset ALL qubits each step, encode a length-min(window_size, N) sliding
    window of recent inputs 1:1 onto qubits, NO wraparound) but with the
    MIXED g/J layer instead of `reservoir_layer`."""
    if n_terms is None:
        n_terms = default_n_sparse_terms(cfg.N)
    terms = sample_syk4_terms(cfg.N, n_terms, seed=term_seed)
    couplings = sample_syk4_couplings(n_terms, J=J, seed=term_seed)
    paulis = sample_syk4_pauli_types(n_terms, seed=term_seed)
    bias_z, _ = cfg.sample_disorder()
    labels, ops = feature_ops_all_general(cfg.N, max_weight=max_weight)
    eff_window = min(window_size, cfg.N)
    slot_to_qubit = list(range(eff_window))

    qc = QuantumCircuit(cfg.N)
    for t, _ in enumerate(u_seq):
        lo = max(0, t - eff_window + 1)
        wlen = t - lo + 1
        window = np.zeros(eff_window)
        window[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(cfg.N)
        for w, uu in enumerate(window):
            per_q[slot_to_qubit[w]] += np.pi * float(uu)

        for i in range(cfg.N):
            qc.reset(i)
            qc.ry(per_q[i], i)
        for _ in range(reps):
            mixed_layer(qc, cfg.N, g, terms, couplings, paulis, bias_z)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f'{lab}__t{t}')
    return qc, labels


def run_reservoir_qelm_mixed(cfg: ReservoirConfig, u_seq: Sequence[float], g: float, J: float,
                              window_size: int = 8, max_weight: int = 5, reps: int = 1,
                              n_terms=None, term_seed: int = 0, use_gpu: bool = False,
                              method: str = 'statevector'):
    """Same (labels, X, info) shape as `run_reservoir_mixed`."""
    import time
    qc, labels = build_qelm_circuit_mixed(cfg, u_seq, g, J, window_size, max_weight, reps, n_terms, term_seed)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    t0 = time.perf_counter()
    result = sim.run(tqc, shots=1).result()
    elapsed = time.perf_counter() - t0
    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f'{lab}__t{t}'])
    info = {'device': 'GPU' if use_gpu else 'CPU', 'method': method, 'architecture': 'qelm_mixed',
             'window_size': window_size, 'N': cfg.N, 'T': T, 'elapsed_s': elapsed,
             'circuit_depth': tqc.depth(), 'n_features': len(labels)}
    return labels, X, info


def build_qelm_circuit_haar(cfg: ReservoirConfig, u_seq: Sequence[float], window_size: int, seed: int,
                             max_weight: int = 5):
    """QELM Haar baseline: reset+encode window each step, then ONE FRESH
    Haar-random N-qubit unitary in place of `reps` mixed layers."""
    U_haar = random_unitary(2 ** cfg.N, seed=seed)
    labels, ops = feature_ops_all_general(cfg.N, max_weight=max_weight)
    eff_window = min(window_size, cfg.N)
    slot_to_qubit = list(range(eff_window))
    qc = QuantumCircuit(cfg.N)
    for t, _ in enumerate(u_seq):
        lo = max(0, t - eff_window + 1)
        wlen = t - lo + 1
        window = np.zeros(eff_window)
        window[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(cfg.N)
        for w, uu in enumerate(window):
            per_q[slot_to_qubit[w]] += np.pi * float(uu)
        for i in range(cfg.N):
            qc.reset(i)
            qc.ry(per_q[i], i)
        qc.append(U_haar, range(cfg.N))
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f'{lab}__t{t}')
    return qc, labels


def run_reservoir_qelm_haar(cfg: ReservoirConfig, u_seq: Sequence[float], window_size: int, seed: int,
                             method: str = 'statevector'):
    qc, labels = build_qelm_circuit_haar(cfg, u_seq, window_size, seed)
    sim = make_simulator(use_gpu=False, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=1).result()
    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f'{lab}__t{t}'])
    return labels, X


# =============================================================================
# Section 2 -- Edge-of-chaos diagnostics
# (verbatim from `4_QR_MixedSYK_Qiskit.ipynb`, cell "Section 2", 88 lines)
# =============================================================================

def single_layer_unitary_mixed(N: int, g: float, terms, couplings, paulis, bias_z) -> np.ndarray:
    qc = QuantumCircuit(N)
    mixed_layer(qc, N, g, terms, couplings, paulis, bias_z)
    return Operator(qc).data


def step_unitary_mixed(N: int, g: float, terms, couplings, paulis, reps: int, bias_z) -> np.ndarray:
    U1 = single_layer_unitary_mixed(N, g, terms, couplings, paulis, bias_z)
    return np.linalg.matrix_power(U1, reps)


def operator_entanglement(U: np.ndarray, N: int) -> float:
    n_a = N // 2
    n_b = N - n_a
    da, db = 2 ** n_a, 2 ** n_b
    Um = U.reshape(da, db, da, db).transpose(0, 2, 1, 3).reshape(da * da, db * db)
    s = np.linalg.svd(Um, compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def level_spacing_ratio(U: np.ndarray) -> float:
    ev = np.linalg.eigvals(U)
    phases = np.sort(np.angle(ev))
    gaps = np.diff(np.concatenate([phases, [phases[0] + 2 * np.pi]]))
    gaps = gaps[gaps > 1e-13]
    if len(gaps) < 3:
        return float('nan')
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return float(np.mean(r))


def sample_reference_r_statistics(d: int, trials: int = 30, seed: int = 0):
    rng = np.random.RandomState(seed)
    out = {}
    for name in ('poisson', 'cue', 'coe'):
        vals = []
        for _ in range(trials):
            if name == 'poisson':
                phases = np.sort(rng.uniform(-np.pi, np.pi, d))
            else:
                U = unitary_group.rvs(d, random_state=rng)
                if name == 'coe':
                    U = U.T @ U
                phases = np.sort(np.angle(np.linalg.eigvals(U)))
            gaps = np.diff(np.concatenate([phases, [phases[0] + 2 * np.pi]]))
            gaps = gaps[gaps > 1e-13]
            r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
            vals.append(np.mean(r))
        out[name] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _local_pauli(N: int, q: int, name: str) -> np.ndarray:
    mats = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Z': np.array([[1, 0], [0, -1]])}
    U = np.array([[1.0]])
    for i in range(N - 1, -1, -1):
        U = np.kron(U, mats[name] if i == q else mats['I'])
    return U.astype(np.complex128)


def otoc_curve(U1: np.ndarray, N: int, w_qubit: int, v_qubit: int, depths):
    d = 2 ** N
    W0 = _local_pauli(N, w_qubit, 'Z')
    V = _local_pauli(N, v_qubit, 'X')
    Ud = U1.conj().T
    Wt = W0.copy()
    out = []
    depths = sorted(depths)
    cur_depth = 0
    for target in depths:
        while cur_depth < target:
            Wt = Ud @ Wt @ U1
            cur_depth += 1
        F = np.trace(Wt @ V @ Wt @ V) / d
        out.append(float(2.0 * (1.0 - np.real(F))))
    return out


def scrambling_time(U1: np.ndarray, N: int, w_qubit: int, v_qubit: int, max_depth: int = 20,
                     threshold: float = 0.5):
    depths = list(range(1, max_depth + 1))
    C = np.array(otoc_curve(U1, N, w_qubit, v_qubit, depths))
    hit = np.where(C >= threshold)[0]
    t_star = float(depths[hit[0]]) if len(hit) else float(max_depth)
    return t_star, C
