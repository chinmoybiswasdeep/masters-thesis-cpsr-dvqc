"""
v3_2_encoder.py -- the V3.2 input encoder and the shared dense-operator
helpers every other V3.2 module builds on.

THE V3.1 DEFECT THIS FIXES
---------------------------------------------------------------------------
V3's processor put the input INSIDE the Hamiltonian exponent
(`nonlinear_processor.processor_unitary`: U_P = exp(-i(H_0(g,J) + sum_k d_k V_k) dt)^R
acting on a reset |0...0>). Every feature is then a trigonometric function of
u REGARDLESS of (g,J), so the encoder -- not the controls -- generated the
instantaneous nonlinearity. The measured consequence in
`results/dqrc_dual_route_v3_1/validation_v3_1_results.json` is
`encoding_only = 0.7201` vs `full = 0.6632`, i.e. an encoder-only fraction
ABOVE one, leaving the (g,J) controls nothing to modulate (E_NL = 0.074).

V3.2 encodes the input as a PRODUCT STATE that is exactly affine in u at the
level of the local readout:

    rho(u) = (I + s Z) / 2 ,   s = 2u - 1 in [-1, 1]
    rho_P(u) = rho(u) ^ (tensor N_P)

Two properties are then EXACT, not approximate (both are asserted in
`tests/test_v3_2_encoder.py` and re-verified by quadrature in
`v3_2_processor.exact_instantaneous_capacities`):

  1. Under ANY product of single-qubit channels (which is what the processor
     reduces to when g = J = 0), every local Pauli expectation is an affine
     function of s. Hence NL_0(g=0, J=0) = 0 EXACTLY and the encoder-only
     fraction f_enc = 0 BY CONSTRUCTION rather than by tuning.
  2. rho_P(u) is multilinear in s across the N_P copies, so every feature is
     a polynomial in s of degree AT MOST N_P. The achievable instantaneous
     nonlinear degree is therefore BOUNDED BY THE NUMBER OF INPUT COPIES.
     Degrees above N_P are structurally zero and must be reported as such,
     never as an empirically absent capacity.

Property 2 is why `n_input_copies` is counted as a physical resource by
`EncoderSpec.resources()` and matched across every baseline.

NOTE this is the SAME encoding the V3 memory bank already used
(`memory_bank.encode_diagonal_z` prepares rho = (I + sZ)/2 via
Ry(arccos(s))), so the two modules now share one input convention. The
canonical input range stays u in [0,1] -- what `qrc_qiskit.random_input`
produces and what `ipc.to_v` expects -- and the signed form is derived
internally, exactly as `memory_bank.to_signed` does.

QUBIT ORDERING: qubit 0 is the LEFTMOST tensor factor (`embed` krons in
ascending qubit order). This is the opposite of Qiskit's little-endian
convention; it is internally consistent throughout V3.2 and
`tests/test_v3_2_encoder.py::test_matches_qiskit_density_matrix` pins it
against an actual Qiskit `DensityMatrix` so the convention can never drift
silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# Shared dense-operator helpers (every V3.2 module uses these)
# =============================================================================
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def embed(n_qubits: int, sites: dict) -> np.ndarray:
    """Dense 2**n x 2**n operator from {qubit_index: 'X'|'Y'|'Z'|'I'}.

    Qubit 0 is the leftmost tensor factor. Unlisted qubits get the identity.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    for q in sites:
        if not 0 <= q < n_qubits:
            raise ValueError(f"qubit index {q} outside 0..{n_qubits - 1}")
    out = np.array([[1.0]], dtype=complex)
    for q in range(n_qubits):
        out = np.kron(out, PAULI[sites.get(q, "I")])
    return out


def local_pauli_ops(n_qubits: int, paulis: tuple = ("X", "Y", "Z")) -> tuple:
    """The preregistered local readout set {<P_i>} for P in `paulis`.

    Returns (labels, operators). The set depends ONLY on the qubit layout --
    never on g, J, m or lambda -- so the feature DEFINITION is identical at
    every control setting, which is what makes a frozen readout meaningful.
    """
    labels, ops = [], []
    for q in range(n_qubits):
        for p in paulis:
            labels.append(f"{p}{q}")
            ops.append(embed(n_qubits, {q: p}))
    return labels, ops


def expectation_values(rho: np.ndarray, ops: list) -> np.ndarray:
    """<O_k> = Tr[O_k rho] for a list of dense operators.

    Uses the O(D^2) elementwise form rather than Tr[O @ rho] (which is
    O(D^3)); at N_M + N_P = 9 qubits that is the difference between a 66 ms
    and a 13 ms simulation step.
    """
    rho_t = rho.T
    return np.array([float(np.real(np.sum(O * rho_t))) for O in ops])


def to_signed(u) -> np.ndarray:
    """Canonical u in [0,1] -> signed s in [-1,1].

    Identical to `memory_bank.to_signed`; duplicated here only so this module
    has no import cycle with the V3 memory bank.
    """
    return 2.0 * np.asarray(u, dtype=float) - 1.0


# =============================================================================
# The encoder
# =============================================================================
@dataclass(frozen=True)
class EncoderSpec:
    """The V3.2 product encoder. `n_copies` input copies is a RESOURCE."""

    n_copies: int
    kind: str = "linear_product_z"

    def __post_init__(self):
        if self.n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {self.n_copies}")
        if self.kind != "linear_product_z":
            raise ValueError(f"unknown encoder kind {self.kind!r}")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "n_copies": self.n_copies}

    def resources(self) -> dict:
        """Physical resources this encoder consumes, for matched-resource
        accounting. `max_instantaneous_degree` is a HARD structural bound,
        not an empirical observation -- see the module docstring."""
        return {"input_copies": self.n_copies,
                "qubits_for_encoding": self.n_copies,
                "max_instantaneous_degree": self.n_copies}

    def state(self, u: float) -> np.ndarray:
        """rho(u)^(tensor n_copies) as a dense density matrix."""
        s = float(np.clip(to_signed(u), -1.0, 1.0))
        single = np.array([[(1.0 + s) / 2.0, 0.0], [0.0, (1.0 - s) / 2.0]], dtype=complex)
        out = np.array([[1.0]], dtype=complex)
        for _ in range(self.n_copies):
            out = np.kron(out, single)
        return out

    def single_qubit_state(self, u: float) -> np.ndarray:
        """One copy of rho(u) -- used by the memory encoder, which injects a
        single rail rather than a product."""
        s = float(np.clip(to_signed(u), -1.0, 1.0))
        return np.array([[(1.0 + s) / 2.0, 0.0], [0.0, (1.0 - s) / 2.0]], dtype=complex)


def is_density_matrix(rho: np.ndarray, atol: float = 1e-10) -> dict:
    """Trace / hermiticity / positivity audit (Gate B).

    Returned as a dict of measured deviations rather than a bool so the
    numbers themselves land in the results artifact.
    """
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {rho.shape}")
    eigs = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    out = {"trace_dev": float(abs(np.trace(rho).real - 1.0)),
           "trace_imag": float(abs(np.trace(rho).imag)),
           "hermiticity_dev": float(np.max(np.abs(rho - rho.conj().T))),
           "min_eigenvalue": float(np.min(eigs))}
    out["ok"] = bool(out["trace_dev"] <= atol and out["trace_imag"] <= atol
                     and out["hermiticity_dev"] <= atol and out["min_eigenvalue"] >= -atol)
    return out
