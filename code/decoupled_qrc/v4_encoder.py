"""
v4_encoder.py -- input injection for V4, and the polynomial audit that keeps
the claims about it honest.

THE CORRECTION THIS MODULE ENCODES
---------------------------------------------------------------------------
It is WRONG to call rho(u) = ((I + uZ)/2)^{(x)n} "globally linear in u". The
state carries every degree up to n. Measured here by exact Legendre
projection (`polynomial_degrees`):

    n=2:  <Z_0> -> degree {1}      <Z_0 Z_1>     -> degrees {0, 2}
    n=3:  <Z_0> -> degree {1}      <Z_0 Z_1 Z_2> -> degrees {1, 3}
    n=4:  <Z_0> -> degree {1}      <Z_0..Z_3>    -> degrees {0, 2, 4}

The true statement is narrower and is the one V4 relies on: the encoder is
**affine at the LOCAL readout**, and the n copies are a NONLINEAR RESOURCE
that must be counted (`EncoderSpec.resources`), never described as free.

TWO INJECTIONS, TWO ROLES
  * `AffineInjection` (memory route R): exactly ONE copy of u per timestep,
    into one reset subsystem. rho_in(u) = rho_0 + u * drho.
  * `ReuploadInjection` (nonlinear route P): n_copies copies per timestep.
    The copies ARE the nonlinear resource; they are fixed across every value
    of g so that g changes interaction strength, never resource count.

WHY R MUST STAY STRICTLY AFFINE (this is a load-bearing design decision).
Quantum channels are linear in rho. With exactly one affine injection of each
input, rho_R(t) is MULTILINEAR in the past inputs -- every u_{t-s} appears at
degree at most 1 (verified in `tests/test_v4_encoder.py`). Consequently a
target with degree >= 2 in a STRICTLY POSITIVE delay, such as P_2(u_{t-3}),
is unreachable from R.

That is deliberate, not a limitation to be patched. Giving R a second
injection would make P_2(u_{t-3}) reachable from R ALONE -- and then the
high-m/low-g corner would already solve the combined N_long task, destroying
the combined-capability gate. Keeping R affine forces N_long targets of the
form P_d(u_t) * P_1(u_{t-tau}) to draw the degree from P (control g) and the
delay from R (control m). The capability boundary is reported in the results
rather than concealed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as npleg

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def embed(n_qubits: int, sites: dict) -> np.ndarray:
    """Dense operator from {qubit: 'X'|'Y'|'Z'|'I'}; qubit 0 is leftmost."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    for q in sites:
        if not 0 <= q < n_qubits:
            raise ValueError(f"qubit index {q} outside 0..{n_qubits - 1}")
    out = np.array([[1.0]], dtype=complex)
    for q in range(n_qubits):
        out = np.kron(out, PAULI[sites.get(q, "I")])
    return out


def expectations(rho: np.ndarray, ops) -> np.ndarray:
    """<O_k> = Tr[O_k rho], via the O(D^2) elementwise form."""
    rho_t = rho.T
    return np.array([float(np.real(np.sum(O * rho_t))) for O in ops])


def single_qubit_state(u: float) -> np.ndarray:
    """rho(u) = (I + uZ)/2 for u in [-1, 1]. Affine in u by construction."""
    u = float(np.clip(u, -1.0, 1.0))
    return np.array([[(1.0 + u) / 2.0, 0.0], [0.0, (1.0 - u) / 2.0]], dtype=complex)


# =============================================================================
# The polynomial audit
# =============================================================================
def polynomial_degrees(fn, deg_max: int = 12, n_quad: int = 64, tol: float = 1e-10) -> dict:
    """Exact Legendre content of a scalar polynomial fn(u) on u in [-1, 1].

    Gauss-Legendre quadrature is exact for polynomials, so a coefficient
    reported as zero here is zero, not merely small.
    """
    nodes, w = np.polynomial.legendre.leggauss(int(n_quad))
    vals = np.array([float(fn(u)) for u in nodes], dtype=float)
    coeffs = {}
    for d in range(int(deg_max) + 1):
        basis = npleg.legval(nodes, [0] * d + [1])
        coeffs[d] = float(np.sum(w * vals * basis) * (2 * d + 1) / 2.0)
    present = sorted(d for d, c in coeffs.items() if abs(c) > tol)
    return {"coeffs": coeffs, "degrees_present": present,
            "max_degree": (max(present) if present else 0),
            "is_affine": bool(present == [] or set(present) <= {0, 1}),
            "tol": tol}


def audit_encoder_state(n_copies: int, deg_max: int = 12) -> dict:
    """Degree content of the product encoder at a local and a global readout.

    This is the numerical evidence for the module docstring: the STATE is not
    affine, only the LOCAL readout is.
    """
    def rho(u):
        out = np.array([[1.0]], dtype=complex)
        for _ in range(n_copies):
            out = np.kron(out, single_qubit_state(u))
        return out

    local = embed(n_copies, {0: "Z"})
    glob = embed(n_copies, {q: "Z" for q in range(n_copies)})
    a_loc = polynomial_degrees(lambda u: float(np.real(np.trace(local @ rho(u)))), deg_max)
    a_glob = polynomial_degrees(lambda u: float(np.real(np.trace(glob @ rho(u)))), deg_max)
    return {"n_copies": int(n_copies),
            "local_readout": {k: a_loc[k] for k in ("degrees_present", "max_degree", "is_affine")},
            "global_readout": {k: a_glob[k] for k in ("degrees_present", "max_degree", "is_affine")},
            "state_is_globally_affine": bool(a_glob["is_affine"]),
            "note": ("the n copies carry degrees up to n; only the LOCAL readout is affine. "
                     "Never describe this encoder as globally linear.")}


def multilinearity_report(feature_fn, n_inputs: int, base=None, deg_max: int = 6) -> dict:
    """Per-input Legendre degree of a feature that depends on several inputs.

    `feature_fn(list_of_u) -> float`. Used to certify that an affine-injection
    memory route is multilinear: degree <= 1 in EVERY individual input.
    """
    base = list(base if base is not None else np.linspace(-0.4, 0.4, n_inputs))
    rows = {}
    for k in range(n_inputs):
        def g(u, k=k):
            us = list(base)
            us[k] = u
            return feature_fn(us)
        rows[k] = polynomial_degrees(g, deg_max=deg_max)
    max_per_input = {k: v["max_degree"] for k, v in rows.items()}
    return {"per_input": {k: v["degrees_present"] for k, v in rows.items()},
            "max_degree_per_input": max_per_input,
            "multilinear": bool(all(d <= 1 for d in max_per_input.values()))}


def reachable_monomial(degrees_by_delay: dict) -> dict:
    """Is a monomial reachable by {multilinear in u_<=t} x {polynomial in u_t}?

    Reachable iff its degree in every STRICTLY POSITIVE delay is <= 1. The
    delay-0 factor may carry any degree, because the processor route supplies
    it. This is the capability boundary of the V4 product architecture and is
    used to build the N_long target library and to label the targets the
    architecture provably cannot reach.
    """
    blocking = sorted(tau for tau, d in degrees_by_delay.items() if tau > 0 and d >= 2)
    return {"reachable": not blocking, "blocking_delays": blocking,
            "reason": ("reachable: degree >= 2 only at delay 0" if not blocking else
                       f"degree >= 2 at strictly positive delay(s) {blocking}; a "
                       f"multilinear memory route cannot produce it")}


# =============================================================================
# Injections
# =============================================================================
@dataclass(frozen=True)
class AffineInjection:
    """Memory-route injection: EXACTLY one copy of u per timestep.

    rho_in(u) = rho_0 + u * drho with rho_0 = I/2 and drho = Z/2.
    """

    kind: str = "affine_single_copy"

    @property
    def n_copies(self) -> int:
        return 1

    def state(self, u: float) -> np.ndarray:
        return single_qubit_state(u)

    def rho0_drho(self) -> tuple:
        return I2 / 2.0, Z2 / 2.0

    def as_dict(self) -> dict:
        return {"kind": self.kind, "n_copies": 1,
                "form": "rho_in(u) = I/2 + u Z/2 (exactly affine, one copy)"}

    def resources(self) -> dict:
        return {"input_copies": 1, "qubits": 1, "nonlinear_degree_supplied": 1}


@dataclass(frozen=True)
class ReuploadInjection:
    """Processor-route injection: n_copies copies of u per timestep.

    The copy count is the NONLINEAR RESOURCE and is FIXED across every value
    of g -- g changes interaction strength only. `resources` reports the copy
    count so the nonlinear route can never be credited with capability it did
    not pay for.
    """

    n_copies: int = 4
    kind: str = "reupload_product_z"

    def __post_init__(self):
        if self.n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {self.n_copies}")

    def state(self, u: float) -> np.ndarray:
        out = np.array([[1.0]], dtype=complex)
        for _ in range(self.n_copies):
            out = np.kron(out, single_qubit_state(u))
        return out

    def as_dict(self) -> dict:
        return {"kind": self.kind, "n_copies": int(self.n_copies),
                "form": "rho_P(u) = ((I + uZ)/2)^(x)n_copies",
                "resource_note": ("the copies ARE the nonlinear resource and are held "
                                  "fixed across g; g scales interaction strength only")}

    def resources(self) -> dict:
        return {"input_copies": int(self.n_copies), "qubits": int(self.n_copies),
                "nonlinear_degree_supplied": int(self.n_copies)}


def density_matrix_audit(rho: np.ndarray, atol: float = 1e-10) -> dict:
    """Trace / hermiticity / positivity, reported as measured deviations."""
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
