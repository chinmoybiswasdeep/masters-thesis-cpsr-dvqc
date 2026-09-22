"""
v3_2_memory.py -- the V3.2 quantum memory subsystem M, with the two
scientifically distinct mechanisms the protocol requires.

    A. "fractional_swap"  -- the existing V3 memory bank: an L-rail register
       where the control m is the fractional-SWAP exponent between adjacent
       rails, i.e. m sets the TRANSFER RATE down the delay line. Reuses
       `memory_bank.fractional_swap_matrix` verbatim (same unitary, same
       convention) rather than re-deriving it.

    B. "leaky_collision"  -- a fixed, disorder-seeded excitation-conserving
       XY chain (hopping mixing) followed by per-rail amplitude damping. The
       control m is the RETENTION 1 - gamma, i.e. m sets the channel's
       spectral radius rather than its transfer rate. This is a genuinely
       different retention mechanism: A redistributes information along the
       rails unitarily, B keeps the redistribution fixed and tunes how fast
       it decays.

Both use the same linear input encoder as the processor
(`v3_2_encoder.EncoderSpec.single_qubit_state`, rho = (I + sZ)/2 on rail 0,
which is exactly what `memory_bank.encode_diagonal_z` already prepared), so
the two modules share one input convention.

WHAT THE PLANNING MEASUREMENTS ESTABLISHED (and what CALIBRATION must confirm
across seeds -- these were single-seed probes):

  * Mechanism A IS controllable, but only in an interior window. Measured at
    L=4, weight-2 readout, 10k shots:
        m       0.15  0.25  0.35  0.45  0.55  0.65  0.75  0.85  0.95
        M_long  1.42  2.55  3.55  4.59  5.00  4.71  5.61  4.42  4.73
    The slope is +11.3, +10.0, +10.4 over m in [0.25, 0.55] and then flips
    sign repeatedly (-2.9, +8.9, -11.9) once the delay profile
    ceiling-saturates above m ~ 0.6. V3.1 selected m* = 0.95 -- inside
    exactly that unusable region, and on the boundary of its own range. That
    is the complete explanation of V3.1's unstable memory derivative.

  * CONTROLLABILITY DEPENDS ON THE READOUT WEIGHT. With a reduced
    single-rail <Z_i> readout there is no controllable region at all; the
    weight-2 set is what exposes m. The readout is therefore preregistered
    and identical at every control setting.

  * Mechanism B is a real candidate that may fail. Two naive variants probed
    during planning showed no controllable region (damping on top of a full
    shift register: slope ~ 0; Haar-random internal mixing: signal destroyed,
    C_0 ~ 0.001). The XY-hopping mixing used here is the structured
    alternative. If B fails and A passes, that is a reportable result.

MEMORY IS NOT SHOT-LIMITED THE WAY THE PROCESSOR IS: M_long measured 4.648
noiseless vs 4.588 at 10k shots. Unlike instantaneous nonlinear capacity,
linear memory has genuine dynamic range without a noise reference, because
there are many delays and only rank-limited features. The asymmetry is
reported, not smoothed over.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .v3_2_encoder import EncoderSpec, embed, expectation_values

MECHANISMS = ("fractional_swap", "leaky_collision")

# Valid control ranges per mechanism. Boundaries are EXCLUDED from candidate
# selection by the interior-margin filter; they are listed here only to define
# the normalisation p~ = (p - p_min)/(p_max - p_min).
CONTROL_RANGE = {"fractional_swap": (0.05, 0.95),
                 "leaky_collision": (0.05, 0.95)}

CONTROL_MEANING = {"fractional_swap": "SWAP exponent between adjacent rails (transfer rate)",
                   "leaky_collision": "retention 1 - gamma of the per-rail amplitude damping"}


@dataclass(frozen=True)
class MemorySpec:
    """Preregistered memory definition. All fields enter the cache key."""

    mechanism: str = "fractional_swap"
    L: int = 4
    max_weight: int = 2
    hop: float = 0.6            # leaky_collision: XY hopping strength scale
    tau_mix: float = 1.0        # leaky_collision: mixing time per step
    field_amp: float = 0.4      # leaky_collision: disordered Z fields

    def __post_init__(self):
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {self.mechanism!r}; known: {MECHANISMS}")
        if self.L < 2:
            raise ValueError(f"L must be >= 2, got {self.L}")
        if self.max_weight not in (1, 2):
            raise ValueError(f"max_weight must be 1 or 2, got {self.max_weight}")

    @property
    def dim(self) -> int:
        return 2 ** self.L

    @property
    def control_range(self) -> tuple:
        return CONTROL_RANGE[self.mechanism]

    def as_dict(self) -> dict:
        return {"mechanism": self.mechanism, "L": self.L, "max_weight": self.max_weight,
                "hop": self.hop, "tau_mix": self.tau_mix, "field_amp": self.field_amp,
                "control_meaning": CONTROL_MEANING[self.mechanism],
                "control_range": list(self.control_range)}

    def resources(self) -> dict:
        labels, _ = memory_feature_ops(self.L, self.max_weight)
        return {"memory_qubits": self.L, "input_copies": 1, "observables": len(labels)}


def memory_feature_ops(L: int, max_weight: int = 2) -> tuple:
    """Single-rail X/Y/Z plus all two-rail products up to `max_weight`.

    Depends only on the rail layout -- never on m, g or J -- so the feature
    COUNT and DEFINITION are identical at every control setting. Label format
    matches `memory_bank.memory_feature_ops` so the two are comparable.
    """
    paulis = ("Z", "X", "Y")
    labels, ops = [], []
    for q in range(L):
        for p in paulis:
            labels.append(f"{p}{q}")
            ops.append(embed(L, {q: p}))
    if max_weight >= 2:
        for q1, q2 in itertools.combinations(range(L), 2):
            for p1, p2 in itertools.product(paulis, repeat=2):
                labels.append(f"{p1}{q1}{p2}{q2}")
                ops.append(embed(L, {q1: p1, q2: p2}))
    return labels, ops


def fractional_swap_matrix(m: float) -> np.ndarray:
    """SWAP^m = P_+ + exp(i*pi*m) P_-.

    Identical to `memory_bank.fractional_swap_matrix`; re-stated here so this
    module stays importable without the Qiskit-dependent V3 module, and
    `tests/test_v3_2_memory.py` asserts the two agree to machine precision.
    """
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    eye = np.eye(4, dtype=complex)
    return (eye + swap) / 2.0 + np.exp(1j * np.pi * float(m)) * (eye - swap) / 2.0


def _lift_two_qubit(U4: np.ndarray, q: int, L: int) -> np.ndarray:
    """Embed a 2-qubit gate acting on adjacent rails (q, q+1) into L rails."""
    if not 0 <= q < L - 1:
        raise ValueError(f"adjacent pair start {q} outside 0..{L - 2}")
    return np.kron(np.kron(np.eye(2 ** q, dtype=complex), U4),
                   np.eye(2 ** (L - q - 2), dtype=complex))


def _xy_mixing_unitary(spec: MemorySpec, disorder_seed: int) -> np.ndarray:
    """Fixed excitation-conserving XY chain + disordered Z fields.

    Chosen over a Haar-random unitary deliberately: a Haar unitary scrambles
    the injected signal into high-weight operators and destroys the local
    readout (measured C_0 ~ 0.001 during planning). Hopping dynamics moves
    information along the rails while keeping it visible to <Z_i>.
    """
    L = spec.L
    rng = np.random.default_rng(int(disorder_seed))
    hops = spec.hop * rng.uniform(0.7, 1.3, max(L - 1, 1))
    fields = rng.uniform(-spec.field_amp, spec.field_amp, L)
    H = np.zeros((spec.dim, spec.dim), dtype=complex)
    for i in range(L - 1):
        H += 0.5 * hops[i] * (embed(L, {i: "X", i + 1: "X"}) + embed(L, {i: "Y", i + 1: "Y"}))
    for i in range(L):
        H += fields[i] * embed(L, {i: "Z"})
    return expm(-1j * spec.tau_mix * H)


def _amplitude_damping_ops(spec: MemorySpec, gamma: float) -> list:
    """Per-rail amplitude-damping Kraus sets for rails 1..L-1.

    Returns a LIST OF KRAUS SETS, one per rail. The rails' channels COMPOSE
    (apply rail 1's channel, then rail 2's, ...); they must not be summed
    into one flat Kraus list, which would add trace-preserving maps together
    and give Tr rho = L - 1.

    Rail 0 is excluded: it is reset and re-encoded every step anyway, so
    damping it would only attenuate the fresh input.
    """
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(max(1.0 - gamma, 0.0))]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(max(gamma, 0.0))], [0.0, 0.0]], dtype=complex)

    def lift(K, q):
        return np.kron(np.kron(np.eye(2 ** q, dtype=complex), K),
                       np.eye(2 ** (spec.L - q - 1), dtype=complex))

    return [[lift(k0, q), lift(k1, q)] for q in range(1, spec.L)]


@dataclass
class MemoryChannel:
    """A fully specified memory step map at one control value."""

    spec: MemorySpec
    m: float
    _gates: list
    _kraus: list          # list of per-rail Kraus SETS; these compose, not sum

    def reset_encode(self, rho: np.ndarray, u: float) -> np.ndarray:
        """Trace out rail 0 and re-prepare rho(u) = (I + sZ)/2 on it.

        Only rail 0 is ever reset; rails 1..L-1 are never reset across the
        whole trajectory, so predictive power for a lagged input reflects
        information genuinely retained in the quantum state -- the same
        fading-memory convention `qrc_qiskit.py` establishes.
        """
        D = self.spec.dim
        # 'aiaj->ij' traces out RAIL 0 (index a) and keeps rails 1..L-1 (i, j).
        # The transposed form 'iaja->ij' would keep rail 0 and discard the memory.
        red = np.einsum('aiaj->ij', rho.reshape(2, D // 2, 2, D // 2))
        return np.kron(EncoderSpec(1).single_qubit_state(u), red)

    def evolve(self, rho: np.ndarray) -> np.ndarray:
        for G in self._gates:
            rho = G @ rho @ G.conj().T
        for kraus_set in self._kraus:          # one set per rail; channels compose
            rho = sum(K @ rho @ K.conj().T for K in kraus_set)
        return rho

    def step(self, rho: np.ndarray, u: float) -> np.ndarray:
        return self.evolve(self.reset_encode(rho, u))


def build_channel(spec: MemorySpec, m: float, disorder_seed: int) -> MemoryChannel:
    """Construct the per-step memory channel at control value `m`."""
    lo, hi = spec.control_range
    if not lo - 1e-12 <= m <= hi + 1e-12:
        raise ValueError(f"m={m} outside the valid range [{lo}, {hi}] for {spec.mechanism!r}")

    if spec.mechanism == "fractional_swap":
        swap = fractional_swap_matrix(m)
        # route DOWN the rails, far end first, so each step moves information
        # one rail further without overwriting what has not yet moved
        gates = [_lift_two_qubit(swap, r - 1, spec.L) for r in range(spec.L - 1, 0, -1)]
        return MemoryChannel(spec=spec, m=float(m), _gates=gates, _kraus=[])

    gamma = 1.0 - float(m)          # m is RETENTION for leaky_collision
    return MemoryChannel(spec=spec, m=float(m),
                         _gates=[_xy_mixing_unitary(spec, disorder_seed)],
                         _kraus=_amplitude_damping_ops(spec, gamma))


def run_memory(spec: MemorySpec, m: float, u_seq, disorder_seed: int) -> tuple:
    """Exact (noiseless) memory feature matrix X_M of shape (T, n_features).

    Shot noise is applied by the caller via `v3_2_readout.add_shot_noise`, so
    the same exact trajectory can be re-used at different shot budgets
    without re-simulating.
    """
    chan = build_channel(spec, m, disorder_seed)
    labels, ops = memory_feature_ops(spec.L, spec.max_weight)
    u_seq = np.asarray(u_seq, dtype=float)
    rho = np.zeros((spec.dim, spec.dim), dtype=complex)
    rho[0, 0] = 1.0                                  # |0...0>
    X = np.empty((u_seq.size, len(ops)), dtype=float)
    for t, u in enumerate(u_seq):
        rho = chan.step(rho, float(u))
        X[t] = expectation_values(rho, ops)
    return X, labels, rho


def channel_spectrum(spec: MemorySpec, m: float, disorder_seed: int) -> dict:
    """Eigenvalues of the per-step memory map, at a fixed neutral input.

    The map rho -> E(rho) at s = 0 (u = 0.5) is linear on the L-rail operator
    space; its transfer matrix is built by applying the step to each matrix
    unit. The leading eigenvalue is the fixed point (|lambda| = 1); the
    SECOND largest modulus is the slowest decay rate and sets the retention
    time tau = -1/ln|lambda_2|. This is the 'derive or numerically estimate
    the memory-channel spectrum' requirement, and it is what makes the claim
    'm changes retention dynamics' checkable rather than asserted.
    """
    chan = build_channel(spec, m, disorder_seed)
    D = spec.dim
    T = np.empty((D * D, D * D), dtype=complex)
    for i in range(D):
        for j in range(D):
            basis = np.zeros((D, D), dtype=complex)
            basis[i, j] = 1.0
            T[:, i * D + j] = chan.step(basis, 0.5).reshape(-1)
    eigs = np.linalg.eigvals(T)
    mods = np.sort(np.abs(eigs))[::-1]
    lam2 = float(mods[1]) if mods.size > 1 else 0.0
    tau = float(-1.0 / np.log(lam2)) if 0.0 < lam2 < 1.0 else float("inf")
    return {"spectral_radius": float(mods[0]), "lambda_2": lam2,
            "retention_time": tau, "top_moduli": [float(x) for x in mods[:6]]}


# =============================================================================
# Delay-profile descriptors
# =============================================================================
def delay_profile_descriptors(capacities: dict, tau_min: int) -> dict:
    """Summaries of a measured C_{1,tau} profile.

    `tau_min` must be FIXED BEFORE CONFIRMATION and justified from the
    calibration-stage delay profile; it is carried in the frozen config.
    """
    taus = sorted(int(t) for t in capacities)
    vals = np.array([float(capacities[t]) for t in taus])
    total = float(vals.sum())
    long_ = float(sum(v for t, v in zip(taus, vals) if t >= tau_min))
    short = float(sum(v for t, v in zip(taus, vals) if t < tau_min))
    centroid = float(sum(t * v for t, v in zip(taus, vals)) / total) if total > 1e-12 else float("nan")
    pos = [(t, v) for t, v in zip(taus, vals) if v > 1e-4]
    if len(pos) >= 3:
        tt = np.array([p[0] for p in pos], dtype=float)
        slope = np.polyfit(tt, np.log(np.array([p[1] for p in pos])), 1)[0]
        fitted_tau = float(-1.0 / slope) if slope < 0 else float("inf")
    else:
        fitted_tau = float("nan")
    tail = float(vals[-1]) if vals.size else float("nan")
    return {"M_total": total, "M_long": long_, "M_short": short,
            "delay_centroid": centroid, "fitted_retention_time": fitted_tau,
            "capacity_tail": tail, "tau_min": int(tau_min),
            "n_delays": len(taus), "max_delay": (taus[-1] if taus else None)}
