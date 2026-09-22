"""
v3_2_processor.py -- the V3.2 instantaneous nonlinear processor P.

    H_P(g, J) = H_mix + g H_2 + J H_4
    H_mix = sum_i (Omega_x,i X_i + Omega_y,i Y_i + h_i Z_i)      (site-disordered)
    H_2   = sum_<i,j> Z_i Z_j                                     (open chain)
    H_4   = sum_{|q|=4} Z_a Z_b Z_c Z_d                           (all C(N_P,4) quartets)

    U(g, J) = A * prod_{r=1..R} [ B(g, J) * A ] ,
        A = exp(-i dt H_mix) ,   B = exp(-i dt (g H_2 + J H_4))

LAYER ORDERING IS NOT COSMETIC. Both H_2 and H_4 are diagonal in the Z basis,
and the V3.2 encoder state rho(u)^(tensor N_P) is also diagonal in the Z
basis. A Z-diagonal unitary COMMUTES with a Z-diagonal state, so an
interaction-first ordering makes the first B layer an exact no-op. The mixing
layer must come first (and a final A is applied so the correlations B creates
land on locally measurable axes). This was found by direct measurement during
planning, not assumed.

THE SAME FACT EXPLAINS LEGITIMATE ABLATION DUPLICATES. `encoder_only`,
`two_body_only` and `four_body_only` all produce IDENTICAL features while
having clearly different unitaries (measured: ||dU||_F up to 4.14 with
||dFeatures|| ~ 2e-15), because each is a Z-diagonal channel acting on a
Z-diagonal state, i.e. the identity channel ON THIS INPUT ENSEMBLE. That is a
proven symmetry, not cache aliasing. Any duplicate NOT on that list is
treated as a defect -- see `ablation_distinctness`, which reports Hamiltonian,
unitary, state and feature differences separately so the two cases can never
be confused (V3.1 concern 7).

VERIFIED PROPERTIES (re-checked by `exact_instantaneous_capacities`, and
asserted in tests/test_v3_2_processor.py):
  * H_mix, H_2, H_4 all Hermitian to 0.0e+00; [H_2, H_4] = 0 exactly.
  * NL_0(g=0, J=0) = 0 EXACTLY -- every local expectation is affine in s.
  * capacities vanish exactly above degree N_P (structural, not empirical).
  * at N_P=5, H_4's C(5,4)=5 quartets make J independently identifiable;
    at N_P=4 the single quartic term contributes almost nothing.

A NOTE ON THE CAPACITY CEILING. In a NOISELESS simulation with a scalar
input, once the features span polynomials of degree <= N_P the capacity
C_d equals 1.000 EXACTLY for every d <= N_P: the metric is pinned at its
ceiling and has no derivative. This is intrinsic to instantaneous IPC with a
scalar input, not an estimator bug, and it is the same ceiling that pinned
V3.1. `exact_instantaneous_capacities` is therefore a STRUCTURAL diagnostic
(which degrees are reachable at all), while the scientific metric NL_0 is
measured at a finite, preregistered shot budget -- see `v3_2_readout.py`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.linalg import expm

from .v3_2_encoder import EncoderSpec, embed, expectation_values, local_pauli_ops

# Ablation registry: name -> (use_mix, use_two_body, use_four_body).
# `randomized` is handled separately (a disorder-seeded random unitary of
# matched dimension) as the negative control.
ABLATIONS = {
    "encoder_only":   (False, False, False),
    "mixing_only":    (True,  False, False),
    "two_body_only":  (False, True,  False),
    "four_body_only": (False, False, True),
    "mix_two_body":   (True,  True,  False),
    "mix_four_body":  (True,  False, True),
    "full":           (True,  True,  True),
}

# Ablations that are PROVABLY feature-identical on a Z-diagonal input
# ensemble, because each is a Z-diagonal channel (see module docstring).
# A duplicate inside this set is expected; any other duplicate is a defect.
PROVEN_FEATURE_DEGENERATE = frozenset({"encoder_only", "two_body_only", "four_body_only"})


@dataclass(frozen=True)
class ProcessorSpec:
    """Preregistered processor definition. Everything here enters the cache
    key (`v3_2_cache.processor_key_fields`)."""

    N_P: int = 5
    dt: float = 0.5
    R: int = 2
    omega_lo: float = 0.7
    omega_hi: float = 1.3
    field_amp: float = 0.5
    readout: tuple = ("X", "Y", "Z")
    chain_periodic: bool = False

    def __post_init__(self):
        if self.N_P < 2:
            raise ValueError(f"N_P must be >= 2, got {self.N_P}")
        if self.R < 1:
            raise ValueError(f"R must be >= 1, got {self.R}")

    @property
    def dim(self) -> int:
        return 2 ** self.N_P

    @property
    def n_features(self) -> int:
        return self.N_P * len(self.readout)

    def encoder(self) -> EncoderSpec:
        return EncoderSpec(n_copies=self.N_P)

    def as_dict(self) -> dict:
        return {"N_P": self.N_P, "dt": self.dt, "R": self.R,
                "omega_lo": self.omega_lo, "omega_hi": self.omega_hi,
                "field_amp": self.field_amp, "readout": list(self.readout),
                "chain_periodic": self.chain_periodic,
                "layer_order": "mix_first_then_interaction_plus_final_mix"}

    def resources(self) -> dict:
        enc = self.encoder().resources()
        n_quartets = len(list(itertools.combinations(range(self.N_P), 4)))
        n_pairs = self.N_P if self.chain_periodic else self.N_P - 1
        return {"processor_qubits": self.N_P,
                "input_copies": enc["input_copies"],
                "max_instantaneous_degree": enc["max_instantaneous_degree"],
                "observables": self.n_features,
                "two_body_terms": n_pairs,
                "four_body_terms": n_quartets,
                "interaction_layers": self.R,
                "mixing_layers": self.R + 1}


@dataclass
class ProcessorHamiltonians:
    """Dense H_mix / H_2 / H_4 for one disorder realisation."""

    H_mix: np.ndarray
    H_2: np.ndarray
    H_4: np.ndarray
    omega_x: np.ndarray
    omega_y: np.ndarray
    fields: np.ndarray
    spec: ProcessorSpec = field(repr=False, default=None)

    def term_norms(self) -> dict:
        return {"H_mix": float(np.linalg.norm(self.H_mix)),
                "H_2": float(np.linalg.norm(self.H_2)),
                "H_4": float(np.linalg.norm(self.H_4))}

    def audit(self) -> dict:
        """Hermiticity and commutator audit -- the 'do not accept the formula
        blindly' check, run as part of CALIBRATION rather than assumed."""
        def herm(M):
            return float(np.max(np.abs(M - M.conj().T)))

        def comm(A, B):
            return float(np.max(np.abs(A @ B - B @ A)))

        return {"herm_H_mix": herm(self.H_mix), "herm_H_2": herm(self.H_2),
                "herm_H_4": herm(self.H_4),
                "comm_H2_H4": comm(self.H_2, self.H_4),
                "comm_Hmix_H2": comm(self.H_mix, self.H_2),
                "comm_Hmix_H4": comm(self.H_mix, self.H_4)}


def build_hamiltonians(spec: ProcessorSpec, disorder_seed: int) -> ProcessorHamiltonians:
    """H_mix (site-disordered), H_2 (chain) and H_4 (all quartets).

    The site disorder is REQUIRED, not decorative: without it the chain's
    reflection symmetry makes sites pairwise equivalent and collapses the
    feature rank (measured: rank 3 vs 4 at N_P=4). Every disorder term is
    single-qubit, so it cannot generate any nonlinearity on its own and
    NL_0(0,0) = 0 is preserved exactly.
    """
    n = spec.N_P
    rng = np.random.default_rng(int(disorder_seed))
    ox = rng.uniform(spec.omega_lo, spec.omega_hi, n)
    oy = rng.uniform(spec.omega_lo, spec.omega_hi, n)
    hz = rng.uniform(-spec.field_amp, spec.field_amp, n)

    H_mix = np.zeros((spec.dim, spec.dim), dtype=complex)
    for i in range(n):
        H_mix += ox[i] * embed(n, {i: "X"}) + oy[i] * embed(n, {i: "Y"}) + hz[i] * embed(n, {i: "Z"})

    pairs = [(i, (i + 1) % n) for i in range(n)] if spec.chain_periodic else \
            [(i, i + 1) for i in range(n - 1)]
    H_2 = np.zeros((spec.dim, spec.dim), dtype=complex)
    for i, j in pairs:
        H_2 += embed(n, {i: "Z", j: "Z"})

    H_4 = np.zeros((spec.dim, spec.dim), dtype=complex)
    for quartet in itertools.combinations(range(n), 4):
        H_4 += embed(n, {q: "Z" for q in quartet})

    return ProcessorHamiltonians(H_mix=H_mix, H_2=H_2, H_4=H_4,
                                 omega_x=ox, omega_y=oy, fields=hz, spec=spec)


def processor_unitary(spec: ProcessorSpec, ham: ProcessorHamiltonians,
                      g: float, J: float, ablation: str = "full",
                      hamiltonian_seed: int = None) -> np.ndarray:
    """U(g, J) for one ablation. Mixing layer FIRST (see module docstring)."""
    if ablation == "randomized":
        # Negative control: a disorder-matched random unitary that has no
        # (g, J) dependence at all. Any metric that still "responds" to
        # (g, J) here is measuring a bug.
        if hamiltonian_seed is None:
            raise ValueError("ablation='randomized' requires hamiltonian_seed")
        rng = np.random.default_rng(int(hamiltonian_seed))
        z = rng.normal(size=(spec.dim, spec.dim)) + 1j * rng.normal(size=(spec.dim, spec.dim))
        q, r = np.linalg.qr(z)
        return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}; known: {sorted(ABLATIONS) + ['randomized']}")

    use_mix, use_h2, use_h4 = ABLATIONS[ablation]
    eye = np.eye(spec.dim, dtype=complex)
    A = expm(-1j * spec.dt * ham.H_mix) if use_mix else eye
    h_int = (g * ham.H_2 if use_h2 else 0.0) + (J * ham.H_4 if use_h4 else 0.0)
    B = expm(-1j * spec.dt * h_int) if (use_h2 or use_h4) else eye

    U = eye
    for _ in range(spec.R):
        U = B @ A @ U
    return A @ U


def processor_features(spec: ProcessorSpec, U: np.ndarray, u_values) -> np.ndarray:
    """Noiseless feature matrix X_P of shape (len(u_values), n_features)."""
    enc = spec.encoder()
    _, ops = local_pauli_ops(spec.N_P, spec.readout)
    u_values = np.atleast_1d(np.asarray(u_values, dtype=float))
    out = np.empty((u_values.size, spec.n_features), dtype=float)
    for k, u in enumerate(u_values):
        rho = U @ enc.state(u) @ U.conj().T
        out[k] = expectation_values(rho, ops)
    return out


# =============================================================================
# Exact (noiseless, infinite-data) instantaneous capacities by quadrature
# =============================================================================
def exact_instantaneous_capacities(spec: ProcessorSpec, U: np.ndarray,
                                    max_degree: int = None, n_quad: int = 128) -> dict:
    """C_d for d = 1..max_degree in the noiseless infinite-data limit.

    C_d is the squared projection of the orthonormal Legendre target L_d onto
    the span of {1} u {features}, computed by Gauss-Legendre quadrature under
    the Uniform[-1,1] measure -- i.e. the exact capacity a linear readout
    could achieve with unlimited noiseless data. No regression, no sampling,
    no seed: this is the STRUCTURAL diagnostic (which degrees are reachable),
    independent of the main reservoir experiment, as the calibration protocol
    requires.
    """
    if max_degree is None:
        max_degree = spec.N_P + 2          # +2 so the structural zero above N_P is visible
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    weights = weights / 2.0                # Uniform[-1,1] probability measure
    u_nodes = (nodes + 1.0) / 2.0          # back to the canonical [0,1] input range

    F = processor_features(spec, U, u_nodes)
    design = np.hstack([np.ones((F.shape[0], 1)), F])
    sqw = np.sqrt(weights)[:, None]
    Q, svals, _ = np.linalg.svd(sqw * design, full_matrices=False)
    keep = svals > 1e-10 * svals[0]
    Q = Q[:, keep]

    caps = {}
    for d in range(1, int(max_degree) + 1):
        y = np.sqrt(2 * d + 1) * npleg.legval(nodes, [0] * d + [1])
        yw = np.sqrt(weights) * y
        caps[d] = float(np.clip(((Q.T @ yw) ** 2).sum() / (yw @ yw), 0.0, 1.0))

    centred = F - (weights[:, None] * F).sum(axis=0)
    return {"capacities": caps,
            "NL_0_exact": float(sum(v for d, v in caps.items() if d >= 2)),
            "C_1_exact": float(caps.get(1, 0.0)),
            "span_rank": int(keep.sum()),
            "feature_rank": int(np.linalg.matrix_rank(centred, tol=1e-9)),
            "max_structural_degree": int(spec.N_P),
            "degrees_above_bound": {d: v for d, v in caps.items() if d > spec.N_P}}


def ablation_distinctness(spec: ProcessorSpec, ham: ProcessorHamiltonians,
                           g: float, J: float, hamiltonian_seed: int,
                           n_probe: int = 64) -> dict:
    """Pairwise Hamiltonian / unitary / state / feature differences for every
    ablation pair, plus the ranks and exact capacities (V3.1 concern 7).

    Returns a `duplicates` list flagging feature-identical pairs, each marked
    `explained=True` only when BOTH members are in
    `PROVEN_FEATURE_DEGENERATE`. An unexplained duplicate is a defect and
    must be debugged before any capacity analysis.
    """
    names = list(ABLATIONS) + ["randomized"]
    u_probe = np.linspace(0.0, 1.0, n_probe)
    enc = spec.encoder()

    unitaries, feats, hams, rows = {}, {}, {}, {}
    for name in names:
        U = processor_unitary(spec, ham, g, J, ablation=name, hamiltonian_seed=hamiltonian_seed)
        unitaries[name] = U
        feats[name] = processor_features(spec, U, u_probe)
        if name == "randomized":
            hams[name] = None
        else:
            use_mix, use_h2, use_h4 = ABLATIONS[name]
            H = (ham.H_mix if use_mix else 0.0) + (g * ham.H_2 if use_h2 else 0.0) \
                + (J * ham.H_4 if use_h4 else 0.0)
            hams[name] = H if np.ndim(H) else np.zeros((spec.dim, spec.dim), dtype=complex)
        exact = exact_instantaneous_capacities(spec, U)
        centred = feats[name] - feats[name].mean(axis=0)
        rows[name] = {"H_norm": (float(np.linalg.norm(hams[name])) if hams[name] is not None else None),
                      "U_norm": float(np.linalg.norm(unitaries[name])),
                      "feature_rank": int(np.linalg.matrix_rank(centred, tol=1e-9)),
                      "C_1_exact": exact["C_1_exact"],
                      "NL_0_exact": exact["NL_0_exact"],
                      "capacities_exact": exact["capacities"]}

    pairs, duplicates = [], []
    rho_probe = enc.state(0.8)
    for a, b in itertools.combinations(names, 2):
        dH = (float(np.linalg.norm(hams[a] - hams[b]))
              if hams[a] is not None and hams[b] is not None else None)
        dU = float(np.linalg.norm(unitaries[a] - unitaries[b]))
        ra = unitaries[a] @ rho_probe @ unitaries[a].conj().T
        rb = unitaries[b] @ rho_probe @ unitaries[b].conj().T
        trace_dist = float(0.5 * np.sum(np.abs(np.linalg.eigvalsh((ra - rb + (ra - rb).conj().T) / 2))))
        dF = float(np.linalg.norm(feats[a] - feats[b]))
        pairs.append({"a": a, "b": b, "dH": dH, "dU": dU,
                      "state_trace_distance": trace_dist, "dFeatures": dF})
        if dF < 1e-10:
            explained = a in PROVEN_FEATURE_DEGENERATE and b in PROVEN_FEATURE_DEGENERATE
            duplicates.append({"a": a, "b": b, "dU": dU, "dFeatures": dF,
                               "explained": bool(explained),
                               "reason": ("Z-diagonal channel on a Z-diagonal input ensemble "
                                          "= identity channel on this ensemble"
                                          if explained else "UNEXPLAINED -- treat as a defect")})

    return {"rows": rows, "pairs": pairs, "duplicates": duplicates,
            "unexplained_duplicates": [d for d in duplicates if not d["explained"]],
            "g": float(g), "J": float(J)}
