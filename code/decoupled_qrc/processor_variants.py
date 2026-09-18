"""
processor_variants.py -- V3.1's FINITE, PREREGISTERED processor family and
the matched ablation ladder.

V3's processor (P0) has substantial instantaneous NL but an almost flat
NL(g,J) surface. Two candidate explanations must be separated:

  (a) the NL metric was ceiling-saturated, so it could not respond to
      anything (V3 reported NL0_raw = 2.000 with exactly 2 available
      targets -- the target-count maximum);
  (b) the processor genuinely has little (g,J) control authority.

V3.1 fixes (a) elsewhere (`ceiling_audit.py`, higher polynomial order) and
addresses (b) here by adding interleaved data re-uploading, so the input
is re-injected BETWEEN interacting layers and the interaction therefore
acts on input-dependent states:

  P0  current V3 processor: exp(-i[H_0(g,J) + u V] dt) repeated R times
  P1  interleaved re-uploading, alternating Z/X encoders between layers
  P2  multi-axis simultaneous encoding (Z on even qubits, X on odd)
  P3  multi-tap: each delayed input on its own qubit, cross-delay terms
      generated ONLY by the interactions

The ablation ladder isolates where nonlinearity comes from: classical
encoding alone, single-qubit rotations, two-body only, four-body only,
commuting-only, non-interacting, and randomized-interaction controls.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

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

PROCESSOR_VARIANTS = ("P0", "P1", "P2", "P3")

ABLATIONS = (
    "encoding_only",        # g = J = 0: pure input encoding, no interaction
    "two_body_only",        # g > 0, J = 0
    "four_body_only",       # g = 0, J > 0
    "commuting",            # interactions replaced by a commuting (all-Z) Hamiltonian
    "noninteracting",       # single-qubit terms only
    "full",                 # g, J as configured
    "randomized",           # interaction couplings randomly re-drawn (structure destroyed)
)


@dataclass(frozen=True)
class ProcVariantConfig:
    variant: str = "P1"
    N_P: int = 4
    g: float = 0.8              # two-body interaction strength
    J: float = 0.6              # four-body interaction strength
    R: int = 3                  # microsteps / re-uploading layers per macro timestep
    dt: float = 0.6             # layer duration
    alpha: float = 1.0          # encoder strength (Z axis)
    beta: float = 0.7           # encoder strength (X axis)
    n_taps: int = 1
    max_weight: int = 2
    ablation: str = "full"
    disorder_scale: float = 0.3  # fixed per-layer local fields (breaks trivial symmetry)

    def as_dict(self) -> dict:
        return {"variant": self.variant, "N_P": self.N_P, "g": self.g, "J": self.J, "R": self.R,
                "dt": self.dt, "alpha": self.alpha, "beta": self.beta, "n_taps": self.n_taps,
                "max_weight": self.max_weight, "ablation": self.ablation,
                "disorder_scale": self.disorder_scale}


def _kron_op(N: int, sites: dict) -> np.ndarray:
    """Dense N-qubit operator, Qiskit little-endian (qubit 0 rightmost)."""
    op = np.array([[1.0]], dtype=complex)
    for q in range(N - 1, -1, -1):
        op = np.kron(op, _PAULI[sites.get(q, "I")])
    return op


def h_two_body(N: int) -> np.ndarray:
    """H_2 = sum_<i,j> X_i X_j on an open chain."""
    h = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        h += _kron_op(N, {i: "X", i + 1: "X"})
    return h


def h_four_body(N: int) -> np.ndarray:
    """H_4 = X_1 X_2 X_3 X_4 (all consecutive 4-qubit windows)."""
    h = np.zeros((2 ** N, 2 ** N), dtype=complex)
    if N < 4:
        return h
    for i in range(N - 3):
        h += _kron_op(N, {i: "X", i + 1: "X", i + 2: "X", i + 3: "X"})
    return h


def h_commuting(N: int) -> np.ndarray:
    """An all-Z interaction: every term commutes with every other, the
    commuting-Hamiltonian ablation control."""
    h = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        h += _kron_op(N, {i: "Z", i + 1: "Z"})
    return h


def h_local_fields(N: int, seeds: NestedSeeds, scale: float) -> np.ndarray:
    """Fixed (seeded) local Z fields. Drawn from the HAMILTONIAN stream, so
    they are matched whenever Hamiltonian seeds are matched, and are
    independent of the input-sequence stream."""
    rng = np.random.RandomState(seeds.hamiltonian % (2 ** 31))
    coeffs = rng.uniform(-scale, scale, N)
    h = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        h += coeffs[i] * _kron_op(N, {i: "Z"})
    return h


def interaction_hamiltonian(cfg: ProcVariantConfig, seeds: NestedSeeds) -> np.ndarray:
    """The (g,J)-controlled interaction, with the ablation applied."""
    N = cfg.N_P
    ab = cfg.ablation
    if ab == "encoding_only":
        return np.zeros((2 ** N, 2 ** N), dtype=complex)
    if ab == "noninteracting":
        return h_local_fields(N, seeds, cfg.disorder_scale)
    if ab == "commuting":
        return cfg.g * h_commuting(N) + h_local_fields(N, seeds, cfg.disorder_scale)
    if ab == "two_body_only":
        return cfg.g * h_two_body(N) + h_local_fields(N, seeds, cfg.disorder_scale)
    if ab == "four_body_only":
        return cfg.J * h_four_body(N) + h_local_fields(N, seeds, cfg.disorder_scale)
    if ab == "randomized":
        rng = np.random.RandomState((seeds.hamiltonian + 977) % (2 ** 31))
        h = np.zeros((2 ** N, 2 ** N), dtype=complex)
        for i in range(N - 1):
            h += rng.uniform(-1, 1) * cfg.g * _kron_op(N, {i: "X", i + 1: "X"})
        return h + h_local_fields(N, seeds, cfg.disorder_scale)
    # "full"
    return (cfg.g * h_two_body(N) + cfg.J * h_four_body(N)
            + h_local_fields(N, seeds, cfg.disorder_scale))


def encoder_operators(cfg: ProcVariantConfig, layer: int) -> list:
    """Input-encoding generators for one layer. Returns a list of
    (coefficient_index, operator) so the caller can weight by d_t."""
    N, ops = cfg.N_P, []
    if cfg.variant == "P0":
        # V3 baseline: one Z encoder per tap on qubit (k mod N), alternating Z/X
        for k in range(cfg.n_taps):
            kind = "Z" if (k // N) % 2 == 0 else "X"
            ops.append((k, cfg.alpha * _kron_op(N, {k % N: kind})))
    elif cfg.variant == "P1":
        # interleaved re-uploading: the encoder AXIS alternates with the layer,
        # so consecutive encoders do not commute with each other
        kind = "Z" if layer % 2 == 0 else "X"
        strength = cfg.alpha if layer % 2 == 0 else cfg.beta
        for k in range(cfg.n_taps):
            ops.append((k, strength * _kron_op(N, {k % N: kind})))
    elif cfg.variant == "P2":
        # multi-axis simultaneous: Z on even qubits, X on odd, every layer
        for k in range(cfg.n_taps):
            site = k % N
            kind = "Z" if site % 2 == 0 else "X"
            strength = cfg.alpha if site % 2 == 0 else cfg.beta
            ops.append((k, strength * _kron_op(N, {site: kind})))
    elif cfg.variant == "P3":
        # multi-tap: tap k lives on ITS OWN qubit, so cross-delay products can
        # only be produced by the interaction, never by a single local encoder
        for k in range(min(cfg.n_taps, N)):
            kind = "Z" if layer % 2 == 0 else "X"
            strength = cfg.alpha if layer % 2 == 0 else cfg.beta
            ops.append((k, strength * _kron_op(N, {k: kind})))
    else:
        raise ValueError(f"unknown processor variant {cfg.variant!r}")
    return ops


def macro_unitary(cfg: ProcVariantConfig, d_t, seeds: NestedSeeds) -> np.ndarray:
    """U_P(d_t) for one macro timestep.

    P0  : exp(-i [H_int + sum_k d_k V_k] dt) applied R times (V3 behaviour).
    P1-3: interleaved re-uploading --
          prod_r exp(-i alpha_r sum_k d_k V_{k,r}) exp(-i dt H_int)
          so the interaction acts BETWEEN input injections.
    """
    N = cfg.N_P
    d = np.atleast_1d(np.asarray(d_t, dtype=float))
    h_int = interaction_hamiltonian(cfg, seeds)

    if cfg.variant == "P0":
        h = h_int.copy()
        for k, v in encoder_operators(cfg, 0):
            if k < len(d):
                h = h + float(d[k]) * v
        step = expm(-1j * h * cfg.dt)
        u = np.eye(2 ** N, dtype=complex)
        for _ in range(cfg.R):
            u = step @ u
        return u

    u = np.eye(2 ** N, dtype=complex)
    interact = expm(-1j * h_int * cfg.dt)
    for r in range(cfg.R):
        h_enc = np.zeros((2 ** N, 2 ** N), dtype=complex)
        for k, v in encoder_operators(cfg, r):
            if k < len(d):
                h_enc = h_enc + float(d[k]) * v
        u = interact @ expm(-1j * h_enc) @ u
    return u


def feature_ops(qubits, max_weight: int = 2):
    """Fixed, hardware-realistic LOCAL observable set: single-qubit X/Y/Z
    plus two-body terms. Identical at every parameter point, so a matched
    feature budget is automatic and no point can gain apparent
    controllability from a richer readout."""
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


@dataclass
class VariantRun:
    d: np.ndarray
    labels: list
    X_P: np.ndarray
    cfg: ProcVariantConfig
    circuit_depth: int = 0


def run_variant(cfg: ProcVariantConfig, d_seq, seeds: NestedSeeds,
                 method: str = "density_matrix", use_gpu: bool = False,
                 reset_each_step: bool = True) -> VariantRun:
    """Runs a processor variant over a macro-timestep sequence. The register
    is reset (CPTP) before every macro step, so X_P[t] depends only on
    d_seq[t] -- no uncontrolled inter-timestep persistence."""
    if method != "density_matrix":
        raise ValueError("processor circuits reset qubits every step -- use method='density_matrix'.")
    d = np.atleast_2d(np.asarray(d_seq, dtype=float))
    if d.shape[0] == 1 and d.shape[1] != cfg.n_taps:
        d = d.T
    qubits = list(range(cfg.N_P))
    labels, ops = feature_ops(qubits, cfg.max_weight)

    qc = QuantumCircuit(cfg.N_P)
    for t in range(d.shape[0]):
        if reset_each_step:
            for q in qubits:
                qc.reset(q)
        qc.append(UnitaryGate(macro_unitary(cfg, d[t], seeds), label=f"U(t={t})"), qubits)
        for (pauli, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(pauli, qargs, label=f"proc_{lab}__t{t}")

    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    data = sim.run(tqc, shots=1).result().data(0)
    X_P = np.array([[float(np.real(data[f"proc_{lab}__t{t}"])) for lab in labels]
                     for t in range(d.shape[0])])
    return VariantRun(d=d, labels=labels, X_P=X_P, cfg=cfg, circuit_depth=tqc.depth())


def commutator_report(cfg: ProcVariantConfig, seeds: NestedSeeds) -> dict:
    """Numerically verifies the noncommutativity the design requires:
    encoders must not commute with the interaction, and (for multi-axis /
    re-uploading variants) consecutive encoders must not commute with each
    other."""
    h_int = interaction_hamiltonian(cfg, seeds)
    enc0 = encoder_operators(cfg, 0)
    enc1 = encoder_operators(cfg, 1)

    def nrm(a, b):
        return float(np.linalg.norm(a @ b - b @ a))

    enc_vs_int = max((nrm(v, h_int) for _, v in enc0), default=0.0)
    enc_vs_enc = 0.0
    for (_, a), (_, b) in itertools.product(enc0, enc1):
        enc_vs_enc = max(enc_vs_enc, nrm(a, b))
    two_vs_four = nrm(h_two_body(cfg.N_P), h_four_body(cfg.N_P))
    return {"variant": cfg.variant, "ablation": cfg.ablation,
            "encoder_vs_interaction": enc_vs_int, "encoder_vs_encoder": enc_vs_enc,
            "two_body_vs_four_body": two_vs_four}
