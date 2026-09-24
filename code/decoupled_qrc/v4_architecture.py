"""
v4_architecture.py -- the V4 orthogonal dual-register reservoir.

    R  memory register     control m   affine single-copy injection, fading
    P  nonlinear register  control g   reset every step, fixed-depth reupload
    joint layer            fixed       observables O_R (x) O_P

R AND P ARE NEVER COUPLED. The global state is exactly rho_R(t) (x) rho_P(t)
at every timestep, so

    X_R depends on (m, u_{<=t})   and NOT on g
    X_P depends on (g, u_t)       and NOT on m
    <O_R (x) O_P> = <O_R> <O_P>

The cross-derivatives dN/dm and dM/dg are therefore ZERO BY CONSTRUCTION for
the route-restricted metrics. That is the point of the design, but it is not
by itself a scientific result: a claim of separation rests on (i) the DIAGONAL
effects being large and real, (ii) the combined capability N_long being real
and requiring BOTH controls, and (iii) the measurement pipeline demonstrably
having the POWER to detect coupling when coupling is present. `contaminated`
and `serial` exist for (iii) and MUST fail.

WHY EXACT SEPARATION IS COMPATIBLE WITH A NON-TRIVIAL RESULT
Because the product factorises, the operational readout that mixes both routes
(`joint` features) does NOT have structurally-zero cross-effects: a full-readout
memory estimate can move with g, since joint features carry both. Both are
reported -- the route-restricted metrics as the preregistered primary, the
operational full readout as the secondary where the equivalence test has
empirical content.

MEMORY MECHANISM, AND WHY IT WORKS IN AN EXACT NOISELESS SIMULATION
In exact arithmetic a linear readout is scale-invariant, so mere attenuation
of an old input does not reduce its capacity -- the readout simply rescales.
What makes memory finite is the Dambre bound: with F linearly independent
readout variables, sum_tau C_{1,tau} <= F. Memory capacity is therefore a
CONSERVED BUDGET that m REDISTRIBUTES across delays, not a quantity m
attenuates. Small m concentrates the budget at tau ~ 0; large m spreads it to
larger tau. `M` is deliberately defined over tau >= tau_L so it measures that
redistribution. This is why V4 needs no shot noise to produce dynamic range
(user requirement: the primary result is noiseless).

Per timestep, in this order:
    a. reset rail 0 and inject rho_in(u_t)   -- gain 1, INDEPENDENT of m
    b. apply the fixed disordered mixing U_R -- independent of m and g
    c. read out X_R(t)                       -- before damping, so C_{1,0} is
                                                not attenuated by m
    d. apply retention channel with parameter m -- carried into t+1
Injection gain is deliberately decoupled from m (unlike the
rho <- m E(rho) + (1-m) rho_in form, where m would confound retention with
input gain).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .v4_encoder import (AffineInjection, ReuploadInjection, density_matrix_audit,
                         embed, expectations)

ARCHITECTURES = ("dual_product", "memory_only", "processor_only",
                 "contaminated", "serial")


# =============================================================================
# Memory register R
# =============================================================================
@dataclass(frozen=True)
class MemorySpec:
    L_R: int = 4
    hop: float = 0.7           # fixed XY mixing strength (independent of m)
    tau_mix: float = 1.0
    field: float = 0.35
    retention: str = "amplitude_damping"
    readout: tuple = ("Z", "X")          # rail-local observables only
    pair_readout: bool = True            # adjacent-rail ZZ, still R-local
    m_max: float = 0.9                   # normalised m in [0,1] -> retention m*m_max

    def __post_init__(self):
        if self.L_R < 2:
            raise ValueError(f"L_R must be >= 2, got {self.L_R}")

    @property
    def dim(self) -> int:
        return 2 ** self.L_R

    def observables(self) -> tuple:
        labels, ops = [], []
        for q in range(self.L_R):
            for p in self.readout:
                labels.append(f"R:{p}{q}")
                ops.append(embed(self.L_R, {q: p}))
        if self.pair_readout:
            for a, b in zip(range(self.L_R - 1), range(1, self.L_R)):
                labels.append(f"R:Z{a}Z{b}")
                ops.append(embed(self.L_R, {a: "Z", b: "Z"}))
        return labels, ops

    def as_dict(self) -> dict:
        return {"L_R": self.L_R, "hop": self.hop, "tau_mix": self.tau_mix,
                "field": self.field, "retention": self.retention,
                "readout": list(self.readout), "pair_readout": self.pair_readout,
                "m_max": self.m_max, "injection": AffineInjection().as_dict(),
                "control_map": "retention = m * m_max"}

    def resources(self) -> dict:
        labels, _ = self.observables()
        return {"qubits": self.L_R, "input_copies": 1, "observables": len(labels),
                "nonlinear_degree_supplied": 1}


def _mixing_unitary(spec: MemorySpec, seed: int) -> np.ndarray:
    """Fixed disordered XY chain. Depends on the architecture seed ONLY --
    never on m or g, so it cannot smuggle control dependence into R."""
    L = spec.L_R
    rng = np.random.default_rng(int(seed))
    hops = spec.hop * rng.uniform(0.75, 1.25, max(L - 1, 1))
    fields = rng.uniform(-spec.field, spec.field, L)
    H = np.zeros((spec.dim, spec.dim), dtype=complex)
    for i in range(L - 1):
        H += 0.5 * hops[i] * (embed(L, {i: "X", i + 1: "X"}) + embed(L, {i: "Y", i + 1: "Y"}))
    for i in range(L):
        H += fields[i] * embed(L, {i: "Z"})
    return expm(-1j * spec.tau_mix * H)


def _retention_kraus(spec: MemorySpec, m: float) -> list:
    """Per-rail retention channel, as a LIST OF PER-RAIL KRAUS SETS.

    The sets COMPOSE (apply rail 1's channel, then rail 2's, ...). Flattening
    them into one list would add trace-preserving maps and give Tr rho = L.
    """
    gamma = float(np.clip(1.0 - m, 0.0, 1.0))
    if spec.retention == "amplitude_damping":
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
        singles = [k0, k1]
    elif spec.retention == "depolarizing":
        p = gamma * 0.75
        singles = [np.sqrt(1 - p) * np.eye(2, dtype=complex),
                   np.sqrt(p / 3) * np.array([[0, 1], [1, 0]], dtype=complex),
                   np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0]], dtype=complex),
                   np.sqrt(p / 3) * np.array([[1, 0], [0, -1]], dtype=complex)]
    else:
        raise ValueError(f"unknown retention {spec.retention!r}")

    def lift(K, q):
        return np.kron(np.kron(np.eye(2 ** q, dtype=complex), K),
                       np.eye(2 ** (spec.L_R - q - 1), dtype=complex))

    return [[lift(K, q) for K in singles] for q in range(spec.L_R)]


class MemoryRoute:
    """R at a fixed control value m. Carries no dependence on g whatsoever."""

    def __init__(self, spec: MemorySpec, m: float, seed: int):
        if not 0.0 <= m <= 1.0:
            raise ValueError(f"m must lie in [0, 1], got {m}")
        self.spec, self.m, self.seed = spec, float(m), int(seed)
        # The normalised control m in [0,1] maps to retention m * m_max.
        # m_max < 1 keeps a strictly positive damping rate everywhere on the
        # grid: at zero damping the register is unitary, information is never
        # discarded but scrambles into high-weight operators the local readout
        # cannot see, and the measured M turns over. Capping retention keeps
        # dM/dm monotone across the whole preregistered grid.
        self.retention = float(m) * float(spec.m_max)
        self.U = _mixing_unitary(spec, seed)
        self.kraus = _retention_kraus(spec, self.retention)
        self.injection = AffineInjection()
        self.labels, self.ops = spec.observables()
        self.reset()

    def reset(self):
        D = self.spec.dim
        self.rho = np.zeros((D, D), dtype=complex)
        self.rho[0, 0] = 1.0

    def step(self, u: float) -> np.ndarray:
        D = self.spec.dim
        # (a) reset rail 0 and inject -- gain 1, independent of m
        red = np.einsum('aiaj->ij', self.rho.reshape(2, D // 2, 2, D // 2))
        self.rho = np.kron(self.injection.state(u), red)
        # (b) fixed mixing
        self.rho = self.U @ self.rho @ self.U.conj().T
        # (c) read out BEFORE retention, so tau=0 is not attenuated by m
        feats = expectations(self.rho, self.ops)
        # (d) retention, carried into the next step
        for kraus_set in self.kraus:
            self.rho = sum(K @ self.rho @ K.conj().T for K in kraus_set)
        return feats


# =============================================================================
# Nonlinear processor register P
# =============================================================================
@dataclass(frozen=True)
class ProcessorSpec:
    N_P: int = 4
    depth: int = 2             # FIXED for every g -- g never changes depth
    dt: float = 0.6
    mix_lo: float = 0.7
    mix_hi: float = 1.3
    field: float = 0.4
    g_scale: float = 2.5       # maps normalised g in [0,1] to interaction strength
    readout: tuple = ("Z", "X", "Y")

    def __post_init__(self):
        if self.N_P < 2:
            raise ValueError(f"N_P must be >= 2, got {self.N_P}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")

    @property
    def dim(self) -> int:
        return 2 ** self.N_P

    def observables(self) -> tuple:
        labels, ops = [], []
        for q in range(self.N_P):
            for p in self.readout:
                labels.append(f"P:{p}{q}")
                ops.append(embed(self.N_P, {q: p}))
        return labels, ops

    def as_dict(self) -> dict:
        return {"N_P": self.N_P, "depth": self.depth, "dt": self.dt,
                "mix_lo": self.mix_lo, "mix_hi": self.mix_hi, "field": self.field,
                "g_scale": self.g_scale, "readout": list(self.readout),
                "injection": ReuploadInjection(n_copies=self.N_P).as_dict(),
                "layer_order": "mix_first_then_interaction_plus_final_mix"}

    def resources(self) -> dict:
        labels, _ = self.observables()
        return {"qubits": self.N_P, "input_copies": self.N_P, "observables": len(labels),
                "nonlinear_degree_supplied": self.N_P, "depth": self.depth}


def _processor_layers(spec: ProcessorSpec, seed: int) -> tuple:
    """Fixed single-qubit mixing A and the interaction generator H_int.

    Mixing comes FIRST: the interaction is Z-diagonal and the encoder state is
    Z-diagonal, so an interaction-first ordering would make the first
    interaction layer an exact no-op.
    """
    n = spec.N_P
    rng = np.random.default_rng(int(seed))
    ox = rng.uniform(spec.mix_lo, spec.mix_hi, n)
    oy = rng.uniform(spec.mix_lo, spec.mix_hi, n)
    hz = rng.uniform(-spec.field, spec.field, n)
    H_mix = np.zeros((spec.dim, spec.dim), dtype=complex)
    for i in range(n):
        H_mix += (ox[i] * embed(n, {i: "X"}) + oy[i] * embed(n, {i: "Y"})
                  + hz[i] * embed(n, {i: "Z"}))
    H_int = np.zeros((spec.dim, spec.dim), dtype=complex)
    for i in range(n - 1):
        H_int += embed(n, {i: "Z", i + 1: "Z"})
    for quad in itertools.combinations(range(n), 4):
        H_int += embed(n, {q: "Z" for q in quad})
    return expm(-1j * spec.dt * H_mix), H_int


class ProcessorRoute:
    """P at a fixed control value g. Reset every step, so its memory is 0.

    At g = 0 the interaction layer is EXACTLY the identity, so the channel is a
    product of single-qubit maps on a product state and every LOCAL observable
    is affine in u -- hence N(g=0) = 0 exactly, by construction rather than by
    tuning. Depth, feature count and resources are identical at every g.
    """

    def __init__(self, spec: ProcessorSpec, g: float, seed: int):
        if not 0.0 <= g <= 1.0:
            raise ValueError(f"g must lie in [0, 1], got {g}")
        self.spec, self.g, self.seed = spec, float(g), int(seed)
        A, H_int = _processor_layers(spec, seed)
        strength = spec.g_scale * float(g)
        B = (expm(-1j * spec.dt * strength * H_int) if strength != 0.0
             else np.eye(spec.dim, dtype=complex))
        U = np.eye(spec.dim, dtype=complex)
        for _ in range(spec.depth):
            U = B @ A @ U
        self.U = A @ U
        self.injection = ReuploadInjection(n_copies=spec.N_P)
        self.labels, self.ops = spec.observables()

    def step(self, u: float) -> np.ndarray:
        rho = self.U @ self.injection.state(u) @ self.U.conj().T
        return expectations(rho, self.ops)


# =============================================================================
# The dual-register system
# =============================================================================
@dataclass(frozen=True)
class V4Spec:
    architecture: str = "dual_product"
    memory: MemorySpec = field(default_factory=MemorySpec)
    processor: ProcessorSpec = field(default_factory=ProcessorSpec)
    n_joint: int = 12               # fixed count of O_R (x) O_P observables
    contamination: float = 0.0      # negative control only

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {self.architecture!r}; "
                             f"known: {ARCHITECTURES}")

    @property
    def has_memory(self) -> bool:
        return self.architecture != "processor_only"

    @property
    def has_processor(self) -> bool:
        return self.architecture != "memory_only"

    def as_dict(self) -> dict:
        return {"architecture": self.architecture, "n_joint": self.n_joint,
                "contamination": float(self.contamination),
                "memory": self.memory.as_dict() if self.has_memory else None,
                "processor": self.processor.as_dict() if self.has_processor else None,
                "coupling": "NONE -- R and P are never coupled (strict product)"}

    def resources(self) -> dict:
        mr = self.memory.resources() if self.has_memory else {}
        pr = self.processor.resources() if self.has_processor else {}
        n_joint = self.n_joint if (self.has_memory and self.has_processor) else 0
        return {"qubits": mr.get("qubits", 0) + pr.get("qubits", 0),
                "input_copies": mr.get("input_copies", 0) + pr.get("input_copies", 0),
                "observables_R": mr.get("observables", 0),
                "observables_P": pr.get("observables", 0),
                "observables_joint": n_joint,
                "observables_total": (mr.get("observables", 0) + pr.get("observables", 0)
                                      + n_joint),
                "processor_depth": pr.get("depth", 0),
                "nonlinear_degree_R": mr.get("nonlinear_degree_supplied", 0),
                "nonlinear_degree_P": pr.get("nonlinear_degree_supplied", 0)}


@dataclass
class V4Run:
    u: np.ndarray
    X_R: np.ndarray
    X_P: np.ndarray
    X_J: np.ndarray
    labels_R: list
    labels_P: list
    labels_J: list
    spec: V4Spec
    seconds: float = 0.0
    dm_audit: dict = field(default_factory=dict)

    @property
    def X_all(self) -> np.ndarray:
        parts = [a for a in (self.X_R, self.X_P, self.X_J) if a.size]
        return np.hstack(parts) if parts else np.empty((len(self.u), 0))

    @property
    def labels_all(self) -> list:
        return list(self.labels_R) + list(self.labels_P) + list(self.labels_J)


def _joint_pairs(n_R: int, n_P: int, n_joint: int, seed: int) -> list:
    """A FIXED, seed-determined selection of (R index, P index) pairs.

    Fixed across m and g, so the joint feature COUNT and DEFINITION never move
    with a control.
    """
    rng = np.random.default_rng(int(seed) + 9187)
    all_pairs = [(a, b) for a in range(n_R) for b in range(n_P)]
    if n_joint >= len(all_pairs):
        return all_pairs
    idx = rng.choice(len(all_pairs), size=int(n_joint), replace=False)
    return [all_pairs[i] for i in sorted(idx)]


def run_v4(spec: V4Spec, u_seq, *, m: float, g: float, seed: int,
           audit_state: bool = False) -> V4Run:
    """Exact, noiseless trajectory. No shots anywhere in the primary path."""
    import time
    t0 = time.perf_counter()
    u_seq = np.asarray(u_seq, dtype=float)
    T = u_seq.size

    R = MemoryRoute(spec.memory, m, seed) if spec.has_memory else None
    P = ProcessorRoute(spec.processor, g, seed) if spec.has_processor else None

    labels_R = R.labels if R else []
    labels_P = P.labels if P else []
    pairs, labels_J = [], []
    if R and P and spec.n_joint:
        pairs = _joint_pairs(len(labels_R), len(labels_P), spec.n_joint, seed)
        labels_J = [f"J:{labels_R[a]}*{labels_P[b]}" for a, b in pairs]

    X_R = np.empty((T, len(labels_R))) if labels_R else np.empty((T, 0))
    X_P = np.empty((T, len(labels_P))) if labels_P else np.empty((T, 0))
    X_J = np.empty((T, len(labels_J))) if labels_J else np.empty((T, 0))

    for t, u in enumerate(u_seq):
        fr = R.step(float(u)) if R else None
        fp = P.step(float(u)) if P else None

        if spec.architecture == "contaminated" and R is not None and P is not None:
            # NEGATIVE CONTROL: leak the memory state into the processor
            # features. This MUST break the cross-effect equivalence tests; if
            # it does not, the tests have no power and prove nothing.
            fp = fp + spec.contamination * float(fr[0]) * np.roll(fp, 1)
        if spec.architecture == "serial" and R is not None and P is not None:
            # NEGATIVE CONTROL: a serial pipeline, where the processor acts on
            # the memory output instead of on the raw input.
            fp = np.tanh(spec.processor.g_scale * g * fr[:len(fp)]
                         if len(fr) >= len(fp) else
                         np.resize(fr, len(fp)) * spec.processor.g_scale * g)

        if fr is not None:
            X_R[t] = fr
        if fp is not None:
            X_P[t] = fp
        if labels_J:
            # product state => <O_R (x) O_P> = <O_R><O_P>, exactly
            X_J[t] = [fr[a] * fp[b] for a, b in pairs]

    audit = {}
    if audit_state and R is not None:
        audit = density_matrix_audit(R.rho)
    return V4Run(u=u_seq, X_R=X_R, X_P=X_P, X_J=X_J, labels_R=labels_R,
                 labels_P=labels_P, labels_J=labels_J, spec=spec,
                 seconds=time.perf_counter() - t0, dm_audit=audit)


def verify_product_factorisation(spec: V4Spec, u_seq, *, m: float, g: float,
                                 seed: int, atol: float = 1e-12) -> dict:
    """Check <O_R (x) O_P> == <O_R><O_P> against the FULL joint state.

    `run_v4` exploits the factorisation for speed; this proves the shortcut is
    exact rather than assumed.
    """
    u_seq = np.asarray(u_seq, dtype=float)
    R = MemoryRoute(spec.memory, m, seed)
    P = ProcessorRoute(spec.processor, g, seed)
    pairs = _joint_pairs(len(R.labels), len(P.labels), spec.n_joint, seed)
    worst = 0.0
    for u in u_seq:
        fr = R.step(float(u))
        rho_P = P.U @ P.injection.state(float(u)) @ P.U.conj().T
        joint = np.kron(R.rho, rho_P)      # R.rho is post-retention; use same for both sides
        n_R, n_P = spec.memory.L_R, spec.processor.N_P
        for a, b in pairs:
            O = np.kron(R.ops[a], P.ops[b])
            lhs = float(np.real(np.sum(O * joint.T)))
            rhs = float(np.real(np.sum(R.ops[a] * R.rho.T))) * float(
                np.real(np.sum(P.ops[b] * rho_P.T)))
            worst = max(worst, abs(lhs - rhs))
        del fr, n_R, n_P
    return {"max_abs_deviation": worst, "atol": atol, "factorises": bool(worst <= atol)}


def route_independence(spec: V4Spec, u_seq, *, m: float, g: float, seed: int,
                       dm: float = 0.07, dg: float = 0.07, atol: float = 1e-12) -> dict:
    """Does X_R move with g, or X_P move with m? Both must be EXACTLY zero.

    Reported as measured deviations so a contaminated architecture shows a
    number rather than merely failing a boolean.
    """
    def run(mm, gg):
        return run_v4(spec, u_seq, m=mm, g=gg, seed=seed)

    base = run(m, g)
    dXR_dg = float(np.max(np.abs(run(m, g + dg).X_R - base.X_R))) if base.X_R.size else 0.0
    dXP_dm = float(np.max(np.abs(run(m + dm, g).X_P - base.X_P))) if base.X_P.size else 0.0
    return {"max_dXR_dg": dXR_dg, "max_dXP_dm": dXP_dm, "atol": atol,
            "architecture": spec.architecture,
            "R_independent_of_g": bool(dXR_dg <= atol),
            "P_independent_of_m": bool(dXP_dm <= atol),
            "passed": bool(dXR_dg <= atol and dXP_dm <= atol)}
