"""Quantum side of notebook 5 (kernel-lifted memory QRC vs edge-of-chaos QRC).

Everything QRC-related that `5_KernelLifted_Memory_QRC_vs_EOC.ipynb` reports is
produced here by Qiskit Aer.  Components are reused from the two source
notebooks; the provenance of every block is stated next to it:

* `1_QR_Qiskit.ipynb` (nb1): `chain_edges`, `ReservoirConfig`, `make_simulator`,
  `feature_ops_all`, the reset-based recurrent trajectory convention.
* `4_QR_MixedSYK_Qiskit.ipynb` (nb4): the mixed SYK2-like/SYK4-like layer and
  its gadgets, `kappa_to_gJ`, sparse random four-body Pauli-term sampling,
  `feature_ops_mem_all`, the adjacent-gap-ratio / operator-entanglement /
  OTOC diagnostics, and the reference circuit `build_trajectory_circuit_mixed`
  (kept verbatim so the tests can regression-check the new runner against it).

New here:

* `MultiInputReservoirConfig` -- p reset input qubits, N-p persistent memory
  qubits (p=1 is exactly nb4's recurrent reservoir).
* `run_qrc_jobs` -- exact *chunked* continuation: each chunk is one Aer
  density-matrix circuit that starts from the previous chunk's saved full
  density matrix (`set_density_matrix`) and ends with `save_density_matrix`.
  Many trajectories are advanced together, one Aer batch per chunk round.
* Exact gate fusion: the per-step unitary is `Operator(mixed_layer circuit)**reps`
  appended as one `UnitaryGate`.  `mixed_layer` stays the single source of
  truth; the gate-level circuit is still available (`fuse=False`) and the
  tests assert both give identical features (<1e-10).
* Exact readout through the memory register's reduced density matrix saved by
  Aer at every step: feature_k(t) = Re Tr(rho_mem(t) P_k).  This is the same
  number Aer's `save_expectation_value(P_k)` returns (asserted in the tests);
  it is ~40x faster because the circuit carries one snapshot per step instead
  of 132.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.stats import unitary_group

import qiskit_aer  # noqa: F401  -- registers save_*/set_density_matrix on QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp, Statevector, random_unitary
from qiskit_aer import AerSimulator

# =============================================================================
# nb1 Section 0a -- reused verbatim
# =============================================================================


def chain_edges(N: int):
    """Nearest-neighbor chain -- the one fixed topology used throughout."""
    return [(i, i + 1) for i in range(N - 1)]


@dataclass
class ReservoirConfig:
    N: int = 6
    g: float = 0.6
    reps: int = 2
    input_qubit: int = 0
    seed: int = 42

    def sample_disorder(self):
        rng = np.random.RandomState(self.seed)
        bias_z = rng.uniform(0, 2 * np.pi, self.N)
        bias_x = rng.uniform(0.3, 0.7, self.N)
        return bias_z, bias_x


def auto_detect_gpu() -> bool:
    """True only if a *freshly constructed* AerSimulator reports a GPU device
    (nb1's gotcha: a GPU-configured simulator just echoes back 'GPU')."""
    try:
        return 'GPU' in AerSimulator().available_devices()
    except Exception:  # pragma: no cover - broken Aer install
        return False


def make_simulator(use_gpu: bool = False, method: str = 'density_matrix', **options) -> AerSimulator:
    """nb1's `make_simulator`, unchanged in behaviour: refuses GPU unless a fresh
    simulator confirms it.  Extra Aer options (parallelism) pass through."""
    device = 'GPU' if use_gpu else 'CPU'
    if use_gpu:
        available = AerSimulator().available_devices()
        if 'GPU' not in available:
            raise RuntimeError(
                'USE_GPU=True but this qiskit-aer build has no GPU device '
                f'(available devices on a fresh AerSimulator(): {available}).')
    return AerSimulator(method=method, device=device, **options)


def feature_ops_all(N: int, max_weight: int = 3):
    """nb1 `feature_ops_all`: full 3**w Pauli-string basis on every adjacent
    window of w<=max_weight qubits (all qubits).  Used for the QELM control."""
    ops, labels = [], []
    paulis = ('Z', 'X', 'Y')
    for q in range(N):
        for name in paulis:
            ops.append((Pauli(name), [q]))
            labels.append(f'{name}{q}')
    if max_weight >= 2:
        for a, b in zip(range(N - 1), range(1, N)):
            for p1, p2 in itertools.product(paulis, repeat=2):
                ops.append((Pauli(p1 + p2), [a, b]))
                labels.append(f'{p1}{a}{p2}{b}')
    if max_weight >= 3:
        for a in range(N - 2):
            for p1, p2, p3 in itertools.product(paulis, repeat=3):
                ops.append((Pauli(p1 + p2 + p3), [a, a + 1, a + 2]))
                labels.append(f'{p1}{a}{p2}{a + 1}{p3}{a + 2}')
    return labels, ops


# =============================================================================
# nb4 Section 0c -- mixed SYK2(g)/SYK4(J) layer, reused verbatim
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
    rng = np.random.RandomState(seed + 500000)
    return rng.choice(['X', 'Y', 'Z'], size=(n_terms, 4))


def zzzz_rotation(qc: QuantumCircuit, qubits4, theta: float):
    """exp(-i*theta/2 * Z_a Z_b Z_c Z_d) via the phase-kickback gadget (exact)."""
    a, b, c, d = qubits4
    qc.cx(a, d); qc.cx(b, d); qc.cx(c, d)
    qc.rz(theta, d)
    qc.cx(c, d); qc.cx(b, d); qc.cx(a, d)


def _basis_change_pre(qc: QuantumCircuit, q: int, p: str):
    if p == 'X':
        qc.h(q)
    elif p == 'Y':
        qc.sdg(q); qc.h(q)


def _basis_change_post(qc: QuantumCircuit, q: int, p: str):
    if p == 'X':
        qc.h(q)
    elif p == 'Y':
        qc.h(q); qc.s(q)


def pauli4_rotation(qc: QuantumCircuit, qubits4, paulis4, theta: float):
    """exp(-i*theta/2 * P_a P_b P_c P_d), P in {X,Y,Z} per qubit."""
    for q, p in zip(qubits4, paulis4):
        _basis_change_pre(qc, q, p)
    zzzz_rotation(qc, qubits4, theta)
    for q, p in zip(qubits4, paulis4):
        _basis_change_post(qc, q, p)


def feature_ops_mem_all(N: int, input_qubit: int, max_weight: int = 3):
    """nb4: full weight<=3 Pauli-string readout restricted to the memory qubits."""
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
    """nb4 single source of truth: SYK2-like Rxx+Ryy hopping (g), SYK4-like
    sparse random-Pauli-type quartic rotations (couplings ~ N(0, J)), Rz kick."""
    for a, b in chain_edges(N):
        qc.rxx(2.0 * g, a, b)
        qc.ryy(2.0 * g, a, b)
    for (i, j, k, l), Jt, ptypes in zip(terms, couplings, paulis):
        pauli4_rotation(qc, (i, j, k, l), ptypes, 2.0 * Jt)
    for i in range(N):
        qc.rz(bias_z[i], i)


def build_trajectory_circuit_mixed(cfg: ReservoirConfig, u_seq: Sequence[float], g: float, J: float,
                                   reps: int, n_terms=None, term_seed: int = 0, max_weight: int = 3):
    """nb4 reference recurrent trajectory (gate level, save_expectation_value
    readout).  Kept verbatim; used only to regression-test `run_qrc_jobs`."""
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


def kappa_to_gJ(kappa: float, G_MAX: float, J_MAX: float):
    """kappa=0 -> pure SYK4-like (g=0, J=J_MAX); kappa->inf -> pure SYK2-like."""
    g = G_MAX * kappa / (1 + kappa)
    J = J_MAX / (1 + kappa)
    return g, J


# nb4's recurrent-architecture coupling scales (G_MAX=pi/2, J_MAX=1.0).
G_MAX = np.pi / 2
J_MAX = 1.0

# =============================================================================
# nb4 Section 2 -- edge-of-chaos diagnostics, reused verbatim
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
    cur_depth = 0
    for target in sorted(depths):
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


# =============================================================================
# New: multichannel recurrent configuration (p reset input qubits)
# =============================================================================


@dataclass(frozen=True)
class MultiInputReservoirConfig:
    """Recurrent mixed-SYK reservoir with `n_input` reset input qubits
    (qubits 0..n_input-1) and N-n_input persistent memory qubits.

    `seed` fixes BOTH the Rz disorder (nb4 convention: ReservoirConfig(seed)
    .sample_disorder()) and the sparse quartic terms (term_seed=seed).
    `haar_seed` replaces the mixed block by one fixed Haar-random N-qubit
    unitary per step (nb4's Haar control); reps/g/J are then unused.
    """
    N: int
    n_input: int
    g: float
    J: float
    reps: int
    seed: int
    n_terms: int | None = None
    haar_seed: int | None = None

    def __post_init__(self):
        if not (self.N > self.n_input >= 1):
            raise ValueError(f'need N > n_input >= 1, got N={self.N}, n_input={self.n_input}')
        if self.reps < 1:
            raise ValueError('reps must be >= 1')

    @property
    def input_qubits(self):
        return list(range(self.n_input))

    @property
    def memory_qubits(self):
        return list(range(self.n_input, self.N))

    def terms(self):
        n_terms = default_n_sparse_terms(self.N) if self.n_terms is None else self.n_terms
        return (sample_syk4_terms(self.N, n_terms, seed=self.seed),
                sample_syk4_couplings(n_terms, J=self.J, seed=self.seed),
                sample_syk4_pauli_types(n_terms, seed=self.seed))

    def bias_z(self):
        return ReservoirConfig(N=self.N, seed=self.seed).sample_disorder()[0]

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def mixed_config(N, n_input, kappa, reps, seed, G_max=G_MAX, J_max=J_MAX) -> MultiInputReservoirConfig:
    g, J = kappa_to_gJ(float(kappa), G_max, J_max)
    return MultiInputReservoirConfig(N=N, n_input=n_input, g=float(g), J=float(J), reps=int(reps), seed=int(seed))


def append_mixed_block(qc: QuantumCircuit, cfg: MultiInputReservoirConfig):
    """Gate-level per-step block: `reps` applications of nb4's `mixed_layer`."""
    terms, couplings, paulis = cfg.terms()
    bias_z = cfg.bias_z()
    for _ in range(cfg.reps):
        mixed_layer(qc, cfg.N, cfg.g, terms, couplings, paulis, bias_z)


@functools.lru_cache(maxsize=512)
def layer_unitary(cfg: MultiInputReservoirConfig) -> np.ndarray:
    """Exact operator of ONE mixed layer, built from the `mixed_layer` circuit."""
    terms, couplings, paulis = cfg.terms()
    return single_layer_unitary_mixed(cfg.N, cfg.g, terms, couplings, paulis, cfg.bias_z())


@functools.lru_cache(maxsize=512)
def step_unitary(cfg: MultiInputReservoirConfig) -> np.ndarray:
    """Exact per-input-step operator: layer**reps (or the Haar control)."""
    if cfg.haar_seed is not None:
        return np.asarray(random_unitary(2 ** cfg.N, seed=cfg.haar_seed).data)
    return np.linalg.matrix_power(layer_unitary(cfg), cfg.reps)


def feature_ops_memory(N: int, input_qubits: Sequence[int], max_weight: int = 3):
    """Generalisation of nb4's `feature_ops_mem_all` to several input qubits:
    the identical weight<=3 adjacent-window Pauli basis on the memory qubits."""
    if len(input_qubits) == 1:
        return feature_ops_mem_all(N, input_qubits[0], max_weight)
    mem = [q for q in range(N) if q not in set(input_qubits)]
    paulis_ = ('Z', 'X', 'Y')
    ops, labels = [], []
    for w in range(1, max_weight + 1):
        for start in range(len(mem) - w + 1):
            qs = mem[start:start + w]
            for combo in itertools.product(paulis_, repeat=w):
                ops.append((Pauli(''.join(combo)), list(qs)))
                labels.append(''.join(f'{p}{q}' for p, q in zip(combo, qs)))
    return labels, ops


@functools.lru_cache(maxsize=64)
def _readout_matrix(N: int, measured: tuple, ops_kind: str):
    """(K, d*d) complex matrix M with feature_k = Re(vec(rho) . M_k), where rho
    is the reduced density matrix of `measured` qubits as saved by Aer
    (measured[0] is the least-significant qubit)."""
    if ops_kind == 'memory':
        inputs = [q for q in range(N) if q not in measured]
        labels, ops = feature_ops_memory(N, inputs, 3)
    else:  # 'all' -- QELM readout, every qubit measured
        labels, ops = feature_ops_all(N, 3)
    local = {q: i for i, q in enumerate(measured)}
    n = len(measured)
    rows = []
    for pauli, qargs in ops:
        P = SparsePauliOp(pauli).apply_layout([local[q] for q in qargs], num_qubits=n).to_matrix()
        rows.append(P.T.reshape(-1))      # Tr(rho P) = sum_ij rho_ij P_ji
    return tuple(labels), np.asarray(rows)


def readout_labels(cfg: MultiInputReservoirConfig, qelm: bool = False):
    measured = tuple(range(cfg.N)) if qelm else tuple(cfg.memory_qubits)
    return _readout_matrix(cfg.N, measured, 'all' if qelm else 'memory')[0]


# =============================================================================
# New: chunked, batched Aer density-matrix execution
# =============================================================================

CACHE_SCHEMA_VERSION = 1
MEMO_MAX_BYTES = 1.5e9          # in-process LRU of feature matrices (reused val trajectories)
_MEMO: 'OrderedDict[str, np.ndarray]' = OrderedDict()
STATS = {'aer_steps': 0, 'aer_circuits': 0, 'aer_seconds': 0.0, 'readout_seconds': 0.0,
         'memo_hits': 0, 'disk_hits': 0, 'recurrent_jobs': 0, 'qelm_jobs': 0}


@dataclass
class QRCJob:
    """One trajectory: `z` is (T, n_input) in [0, 1]; state starts at |0..0>."""
    cfg: MultiInputReservoirConfig
    z: np.ndarray
    qelm: bool = False          # labeled memoryless control: resets ALL qubits each step
    fuse: bool = True           # exact gate fusion (False = nb4 gate-level layer)

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(f'v{CACHE_SCHEMA_VERSION}|{self.cfg.key()}|qelm={self.qelm}|'.encode())
        h.update(np.ascontiguousarray(self.z, dtype=np.float64).tobytes())
        h.update(str(self.z.shape).encode())
        return h.hexdigest()


def _qelm_angles(z: np.ndarray, N: int) -> np.ndarray:
    """nb1 QELM window convention for a scalar stream: qubit N-1 holds z_t,
    qubit N-1-k holds z_{t-k}; missing history is 0."""
    zz = np.asarray(z)[:, 0]
    T = len(zz)
    A = np.zeros((T, N))
    for k in range(N):
        A[k:, N - 1 - k] = zz[:T - k]
    return A


def _build_chunk(job: QRCJob, t0: int, t1: int, rho0, gate, qelm_angles=None) -> QuantumCircuit:
    cfg = job.cfg
    qc = QuantumCircuit(cfg.N)
    if rho0 is not None:
        qc.set_density_matrix(DensityMatrix(rho0))
    measured = list(range(cfg.N)) if job.qelm else cfg.memory_qubits
    for t in range(t0, t1):
        if job.qelm:
            for q in range(cfg.N):
                qc.reset(q)
                qc.ry(np.pi * float(qelm_angles[t, q]), q)
        else:
            for j, q in enumerate(cfg.input_qubits):
                qc.reset(q)
                qc.ry(np.pi * float(job.z[t, j]), q)
        if job.fuse:
            qc.append(gate, range(cfg.N))
        else:
            append_mixed_block(qc, cfg)
        qc.save_density_matrix(qubits=measured, label=f'r{t - t0}')
    qc.save_density_matrix(label='final')
    return qc


def _as_array(obj) -> np.ndarray:
    return np.asarray(getattr(obj, 'data', obj))


def _disk_path(key: str):
    d = os.environ.get('KLQRC_CACHE_DIR')
    return None if not d else os.path.join(d, f'{key}.npy')


def run_qrc_jobs(jobs: Sequence[QRCJob], chunk: int = 256, use_gpu: bool = False,
                 batch_circuits: int = 36, max_parallel_experiments: int | None = None,
                 use_memo: bool = True):
    """Run trajectories with exact chunked continuation.  Returns a list of
    feature matrices (T, K) in job order.

    Chunk k+1 starts from chunk k's saved full density matrix, so only the
    input qubits are ever reset; memory qubits evolve continuously.  All
    unfinished trajectories advance one chunk per Aer batch.
    """
    for j in jobs:
        z = np.asarray(j.z)
        if z.ndim != 2 or z.shape[1] != j.cfg.n_input:
            raise ValueError(f'z must be (T, n_input={j.cfg.n_input}), got {z.shape}')
        if not np.all(np.isfinite(z)) or z.min() < -1e-12 or z.max() > 1 + 1e-12:
            raise ValueError('inputs must be finite and inside [0, 1]')
        if j.qelm and j.cfg.n_input != 1:
            raise ValueError('QELM control is implemented for scalar streams only')
    out = [None] * len(jobs)
    todo, seen = [], {}
    for i, j in enumerate(jobs):
        k = j.key()
        if use_memo and k in _MEMO:
            _MEMO.move_to_end(k)
            out[i] = _MEMO[k]; STATS['memo_hits'] += 1
            continue
        p = _disk_path(k)
        if use_memo and p and os.path.exists(p):
            out[i] = np.load(p); _memo_put(k, out[i]); STATS['disk_hits'] += 1
            continue
        if k in seen:                      # duplicate job inside this call
            continue
        seen[k] = i
        todo.append(i)
    if not todo:
        return out

    if max_parallel_experiments is None:
        max_parallel_experiments = max(1, min(6, (os.cpu_count() or 2) // 2))
    sim = make_simulator(use_gpu=use_gpu, method='density_matrix',
                         max_parallel_experiments=max_parallel_experiments)
    assert sim.options.method == 'density_matrix'

    state = {}
    for i in todo:
        j = jobs[i]
        meas = tuple(range(j.cfg.N)) if j.qelm else tuple(j.cfg.memory_qubits)
        _, M = _readout_matrix(j.cfg.N, meas, 'all' if j.qelm else 'memory')
        state[i] = dict(pos=0, rho=None, feats=[], M=M,
                        gate=UnitaryGate(step_unitary(j.cfg), check_input=False) if j.fuse else None,
                        qa=_qelm_angles(j.z, j.cfg.N) if j.qelm else None)
        STATS['qelm_jobs' if j.qelm else 'recurrent_jobs'] += 1

    active = list(todo)
    while active:
        for b0 in range(0, len(active), batch_circuits):
            batch = active[b0:b0 + batch_circuits]
            circs, spans = [], []
            for i in batch:
                st, T = state[i], len(jobs[i].z)
                t0, t1 = st['pos'], min(st['pos'] + chunk, T)
                circs.append(_build_chunk(jobs[i], t0, t1, st['rho'], st['gate'], st['qa']))
                spans.append((t0, t1))
            ts = time.perf_counter()
            res = sim.run(circs, shots=1).result()
            STATS['aer_seconds'] += time.perf_counter() - ts
            STATS['aer_circuits'] += len(circs)
            if not res.success:
                raise RuntimeError(f'Aer batch failed: {res.status}')
            ts = time.perf_counter()
            for c, (i, (t0, t1)) in enumerate(zip(batch, spans)):
                data = res.data(c)
                R = np.stack([_as_array(data[f'r{t}']).reshape(-1) for t in range(t1 - t0)])
                F = np.real(R @ state[i]['M'].T)
                state[i]['feats'].append(F)
                state[i]['rho'] = _as_array(data['final'])
                state[i]['pos'] = t1
                STATS['aer_steps'] += t1 - t0
            STATS['readout_seconds'] += time.perf_counter() - ts
        active = [i for i in active if state[i]['pos'] < len(jobs[i].z)]

    for i in todo:
        X = np.concatenate(state[i]['feats'], axis=0)
        if not np.all(np.isfinite(X)):
            raise FloatingPointError('non-finite QRC features')
        out[i] = X
        if use_memo:
            k = jobs[i].key()
            _memo_put(k, X)
            p = _disk_path(k)
            if p:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, X)
    for i, j in enumerate(jobs):          # fill in-call duplicates
        if out[i] is None:
            out[i] = out[seen[j.key()]]
    return out


def _memo_put(k, X):
    _MEMO[k] = X
    while sum(v.nbytes for v in _MEMO.values()) > MEMO_MAX_BYTES and len(_MEMO) > 1:
        _MEMO.popitem(last=False)


def clear_memo():
    _MEMO.clear()


def gate_level_step_resources(cfg: MultiInputReservoirConfig) -> dict:
    """Depth/size of ONE gate-level input step transpiled to a hardware-like
    basis (resource accounting; the simulation itself uses the exact fused
    operator)."""
    qc = QuantumCircuit(cfg.N)
    for q in cfg.input_qubits:
        qc.reset(q); qc.ry(0.3, q)
    if cfg.haar_seed is not None:
        qc.append(UnitaryGate(step_unitary(cfg)), range(cfg.N))
    else:
        append_mixed_block(qc, cfg)
    tq = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx', 'reset'], optimization_level=1)
    ops = tq.count_ops()
    return {'depth_per_step': tq.depth(), 'size_per_step': tq.size(), 'cx_per_step': int(ops.get('cx', 0))}


def count_resets_per_qubit(qc: QuantumCircuit) -> dict:
    """Used by the audit: which qubits does a trajectory circuit reset?"""
    counts = {}
    for inst in qc.data:
        if inst.operation.name == 'reset':
            q = qc.find_bit(inst.qubits[0]).index
            counts[q] = counts.get(q, 0) + 1
    return counts


# =============================================================================
# Edge diagnostics over a kappa grid (nb4 Section 4, refactored for any grid)
# =============================================================================


def kappa_diagnostics(kappas, N: int, reps: int, n_seeds: int, otoc_max_depth: int = 20,
                      otoc_threshold: float = 0.5, G_max=G_MAX, J_max=J_MAX):
    """Per kappa and disorder seed: <r> and t* of the SINGLE layer (nb4's
    methodology), S_op of the reps-fold step operator.  Returns arrays
    (n_kappa, n_seeds)."""
    r = np.zeros((len(kappas), n_seeds)); s = np.zeros_like(r); ts = np.zeros_like(r)
    for a, kappa in enumerate(kappas):
        for seed in range(n_seeds):
            cfg = mixed_config(N, 1, kappa, reps, seed, G_max, J_max)
            U1 = layer_unitary(cfg)
            r[a, seed] = level_spacing_ratio(U1)
            s[a, seed] = operator_entanglement(np.linalg.matrix_power(U1, reps), N)
            ts[a, seed] = scrambling_time(U1, N, 0, N - 1, otoc_max_depth, otoc_threshold)[0]
    return {'r': r, 'S_op': s, 't_star': ts}


def edge_band_from_r(kappas, r_median, r_poisson: float, r_chaos: float, delta_eta: float):
    """Preregistered parametric edge (analogue of Kobayashi-Motome Fig. 4's
    dashed lines).  eta = (<r> - r_P)/(r_chaos - r_P).
      kappa_lo = largest kappa still fully chaotic (eta >= 1 - delta_eta)
      kappa_hi = smallest kappa > kappa_lo that has reached Poisson (eta <= delta_eta)
    Returns (kappa_lo, kappa_hi, eta)."""
    kappas = np.asarray(kappas, float)
    eta = (np.asarray(r_median) - r_poisson) / (r_chaos - r_poisson)
    chaotic = np.where(eta >= 1 - delta_eta)[0]
    if len(chaotic) == 0:
        raise RuntimeError('no kappa reaches the chaotic reference -- edge band undefined')
    i_lo = int(chaotic.max())
    later = np.where((np.arange(len(kappas)) > i_lo) & (eta <= delta_eta))[0]
    if len(later) == 0:
        raise RuntimeError('<r> never reaches Poisson above kappa_lo -- edge band undefined')
    return float(kappas[i_lo]), float(kappas[later.min()]), eta
