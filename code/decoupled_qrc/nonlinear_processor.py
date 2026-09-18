"""
nonlinear_processor.py -- V3's processor module: nonlinearity generated
WITHIN a macro timestep by direct Hamiltonian input encoding, with no
reliance on uncontrolled inter-timestep persistence (V2.2 found the old
persistent processor was acting as a second memory as much as a
nonlinear unit).

    H_P(d_t; g, J) = H_0(g, J) + sum_k d_{t,k} V_k
    U_P(d_t)       = prod_{r=1..R} exp(-i H_P(d_t; g, J) dt_r)

with H_0(g,J) = g * sum_i X_i + J * sum_<i,j> Z_i Z_j (transverse-field
Ising: the X and ZZ parts do not commute), and input operators V_k
assigned to qubit (k mod N_P) alternating Z, X, Z, X, ... so that
[V_k, H_0] != 0 for every k and [V_k, V_l] != 0 for at least some k != l
(a Z and an X on the same qubit anticommute). Both properties are
asserted by `tests/test_nonlinear_processor.py`.

The processor register is reset through `qc.reset` (a CPTP channel) at
the start of every macro timestep, so X_P[t] depends ONLY on d_t. The
memory control `m` appears nowhere in this module.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli
from scipy.linalg import expm

from .utils import ensure_repo_code_on_path
from .v3_seeds import NestedSeeds

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator  # noqa: E402

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


@dataclass(frozen=True)
class ProcessorConfig:
    N_P: int = 4                  # processor qubits
    g: float = 0.8                # transverse-field strength
    J: float = 0.6                # ZZ coupling strength
    R: int = 3                    # microsteps per macro timestep
    dt: float = 0.5               # microstep duration
    n_taps: int = 1               # number of input operators (1 for arch B; L for arch C)
    max_weight: int = 2           # readout Pauli weight
    input_scale: float = 1.0      # overall scale on the input-encoding terms

    def as_dict(self) -> dict:
        return {"N_P": self.N_P, "g": self.g, "J": self.J, "R": self.R, "dt": self.dt,
                "n_taps": self.n_taps, "max_weight": self.max_weight, "input_scale": self.input_scale}


def _kron_op(N: int, sites: dict) -> np.ndarray:
    """Dense operator on N qubits from {site: 'X'|'Y'|'Z'}, using Qiskit's
    little-endian convention (qubit 0 is the RIGHTMOST/least-significant
    tensor factor) so these matrices compose correctly with UnitaryGate."""
    op = np.array([[1.0]], dtype=complex)
    for q in range(N - 1, -1, -1):
        op = np.kron(op, _PAULI.get(sites.get(q, "I")))
    return op


def h0_matrix(cfg: ProcessorConfig) -> np.ndarray:
    """H_0 = g * sum_i X_i + J * sum_i Z_i Z_{i+1} (open chain)."""
    N = cfg.N_P
    h = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        h += cfg.g * _kron_op(N, {i: "X"})
    for i in range(N - 1):
        h += cfg.J * _kron_op(N, {i: "Z", i + 1: "Z"})
    return h


def input_operators(cfg: ProcessorConfig) -> list:
    """V_k for k = 0..n_taps-1: qubit (k mod N_P), type alternating
    Z, X, Z, X, ... Chosen so [V_k, H_0] != 0 always (Z does not commute
    with H_0's X terms; X does not commute with H_0's ZZ terms) and
    [V_k, V_l] != 0 whenever two taps land on the same qubit with
    different types."""
    ops = []
    for k in range(cfg.n_taps):
        site = k % cfg.N_P
        kind = "Z" if (k // cfg.N_P) % 2 == 0 else "X"
        ops.append(cfg.input_scale * _kron_op(cfg.N_P, {site: kind}))
    return ops


def processor_unitary(cfg: ProcessorConfig, d_t) -> np.ndarray:
    """U_P(d_t) = prod_r exp(-i H_P(d_t) dt). Exact matrix exponential
    (no Trotter error) -- affordable because N_P is small."""
    h = h0_matrix(cfg)
    for coeff, v_k in zip(np.atleast_1d(d_t), input_operators(cfg)):
        h = h + float(coeff) * v_k
    step = expm(-1j * h * cfg.dt)
    u = np.eye(2 ** cfg.N_P, dtype=complex)
    for _ in range(cfg.R):
        u = step @ u
    return u


def processor_feature_ops(qubits, max_weight: int = 2):
    """Module-specific processor observables -- defined purely from the
    qubit layout, never from g, J, or m."""
    paulis = ("Z", "X", "Y")
    labels, ops = [], []
    for q in qubits:
        for p in paulis:
            labels.append(f"{p}{q}")
            ops.append((Pauli(p), [q]))
    if max_weight >= 2:
        for q1, q2 in itertools.combinations(qubits, 2):
            for p1, p2 in itertools.product(paulis, repeat=2):
                labels.append(f"{p1}{q1}{p2}{q2}")
                ops.append((Pauli(p2 + p1), [q1, q2]))
    return labels, ops


def build_processor_circuit(cfg: ProcessorConfig, d_seq, seeds: NestedSeeds,
                             qubit_offset: int = 0, n_total: int = None,
                             reset_each_step: bool = True) -> tuple:
    """Processor circuit over the whole macro-timestep sequence. `d_seq`
    is (T, n_taps). With `reset_each_step=True` the register is reset to
    |0...0> (CPTP) before each macro step, so X_P[t] depends only on
    d_seq[t]."""
    n_total = n_total if n_total is not None else cfg.N_P + qubit_offset
    qubits = list(range(qubit_offset, qubit_offset + cfg.N_P))
    qc = QuantumCircuit(n_total)
    labels, ops = processor_feature_ops(qubits, max_weight=cfg.max_weight)

    d_seq = np.atleast_2d(np.asarray(d_seq, dtype=float))
    if d_seq.shape[0] == 1 and d_seq.shape[1] != cfg.n_taps:
        d_seq = d_seq.T

    for t in range(d_seq.shape[0]):
        if reset_each_step:
            for q in qubits:
                qc.reset(q)
        gate = UnitaryGate(processor_unitary(cfg, d_seq[t]), label=f"U_P(t={t})")
        qc.append(gate, qubits)
        for (pauli, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(pauli, qargs, label=f"proc_{lab}__t{t}")

    return qc, labels, qubits


@dataclass
class ProcessorRun:
    d: np.ndarray
    labels: list
    X_P: np.ndarray
    cfg: ProcessorConfig
    circuit_depth: int = 0


def run_processor(cfg: ProcessorConfig, d_seq, seeds: NestedSeeds,
                   method: str = "density_matrix", use_gpu: bool = False,
                   reset_each_step: bool = True) -> ProcessorRun:
    """Standalone processor run -- the matched baseline for the embedded
    architectures (same processor, same encoding, same features, same
    seeds, same regression protocol downstream)."""
    if method != "density_matrix":
        raise ValueError("processor circuits reset qubits every step -- use method='density_matrix'.")
    d = np.atleast_2d(np.asarray(d_seq, dtype=float))
    if d.shape[0] == 1 and d.shape[1] != cfg.n_taps:
        d = d.T
    qc, labels, _ = build_processor_circuit(cfg, d, seeds, reset_each_step=reset_each_step)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    data = sim.run(tqc, shots=1).result().data(0)
    X_P = np.array([[float(np.real(data[f"proc_{lab}__t{t}"])) for lab in labels]
                     for t in range(d.shape[0])])
    return ProcessorRun(d=d, labels=labels, X_P=X_P, cfg=cfg, circuit_depth=tqc.depth())


def build_tap_buffer(u_seq, L: int, signed: bool = True) -> np.ndarray:
    """Classical tap buffer d_t = (u_t, u_{t-1}, ..., u_{t-L+1}), zero-
    padded at the start. This is a CLASSICAL resource and is accounted
    for explicitly in every matched-resource comparison (Architecture C's
    honest labelling as a hybrid/modular delayed QRC).

    With `signed=True` the canonical inputs u in [0,1] are mapped to
    2u-1 before entering the Hamiltonian, so the drive is centred on zero
    (an uncentred drive would add a constant field term and waste half the
    encoding range)."""
    u = np.asarray(u_seq, dtype=float)
    if signed:
        u = 2.0 * u - 1.0
    T = len(u)
    d = np.zeros((T, L))
    for k in range(L):
        if k == 0:
            d[:, k] = u
        else:
            d[k:, k] = u[:T - k]
    return d
