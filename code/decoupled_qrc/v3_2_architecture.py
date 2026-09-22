"""
v3_2_architecture.py -- the V3.2 dual-route system: memory M, instantaneous
processor P, and the directional M->P channel that makes Gate F meaningful.

    dual_route_weak     u -> M(m) ;  u -> P(g,J) ;  M --lambda--> P     (PRIMARY)
    dual_route_current  u -> M(m) ;  u -> P(g,J)                        (lambda = 0 control)
    memory_only         u -> M(m)                                       (matched baseline)
    processor_only      u -> P(g,J)                                     (matched baseline)

WHY THE PRIMARY ARCHITECTURE HAS lambda > 0. In `dual_route_current` the two
modules are simulated as SEPARATE objects, so dX_P/dm is exactly zero by
construction -- there is no code path by which it could be anything else.
That makes structural isolation (Gate A) provable, but it also makes
cross-response suppression (Gate F) VACUOUS: an off-diagonal that is zero by
construction is not evidence of suppression. The primary architecture
therefore carries a real M->P channel at a fixed, preregistered lambda, so
|dNL_0/dm~| is a genuine empirical quantity with a real equivalence bound,
and lambda = 0 runs alongside as the structural control.

ALL SIX JACOBIAN ENTRIES ARE EMPIRICAL AT lambda > 0 -- a planning assumption
that direct measurement REFUTED. The channel is directional in information
flow (P is re-prepared fresh every step and discarded after measurement, so
no processor state persists into the next step), and it was assumed that this
made dM_long/dg~ and dM_long/dJ~ structurally zero. It does not. U_MP is a
JOINT unitary, so tracing P away leaves M in a state that depends on what P
was in -- i.e. on (g, J). Measured at N_M=N_P=3, lambda=0.35:

    max|dX_P/dm| = 3.9e-02 ,  max|dX_M/dg| = 8.8e-04 ,  max|dX_M/dJ| = 6.6e-03

all far above the 1e-10 isolation tolerance. This is the same physical
back-action Gate L bounds, and it means Gate F applies to all three
off-diagonals rather than to dNL_0/dm~ alone. At lambda = 0 every one of them
returns to EXACTLY zero, which is what Gate A certifies.
`tests/test_v3_2_architecture.py::test_tracing_out_the_processor_leaves_g_J_dependent_back_action_on_memory`
pins this so the refuted assumption cannot quietly return.

CHANNEL. U_MP(lambda) = exp(-i lambda Z_tap (x) X_entry) on (memory rail
`mem_tap`, processor qubit `proc_entry`). At lambda = 0 it is EXACTLY the
identity, so `dual_route_weak` reduces bit-for-bit to `dual_route_current`
(asserted in tests). X_entry rather than a Z-diagonal coupling is deliberate:
`interface.py` documents that 'rzz'/'cp' are diagonal in the processor
qubit's own Z basis and cannot inject any amplitude into a
magnetisation-conserving processor -- purity_P stays exactly 1.0 at any
lambda. A Z (x) X coupling acts as a conditional rotation on the processor
qubit and creates real M-P entanglement.

PER-STEP ORDER (one macro timestep t):
    1. memory: trace out rail 0, re-encode rho(u_t) on it, apply the memory
       channel (rails 1..L-1 are NEVER reset -- the fading-memory convention).
    2. processor: prepare rho(u_t)^(x)N_P afresh and apply U_P(g,J).
    3. couple: apply U_MP(lambda) on the joint state.
    4. measure X_M on Tr_P[joint] and X_P on Tr_M[joint] (all observables are
       local to one subsystem, so reduced states suffice).
    5. carry Tr_P[joint] forward as the new memory state; discard P.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from . import v3_2_memory as mem
from . import v3_2_processor as proc
from .v3_2_encoder import embed, expectation_values, local_pauli_ops

ARCHITECTURES = ("dual_route_weak", "dual_route_current", "memory_only", "processor_only")


def apply_two_qubit(rho: np.ndarray, G4: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    """rho -> G rho G^dagger for a 2-qubit gate on qubits (a, b) of n qubits.

    Tensor contraction rather than building the full 2^n x 2^n embedding: at
    n = 9 this is the difference between a 66 ms and a 13 ms simulation step.
    `tests/test_v3_2_architecture.py` pins it against the dense embedding.
    """
    if a == b:
        raise ValueError(f"two-qubit gate needs distinct qubits, got a=b={a}")
    t = rho.reshape([2] * (2 * n))
    G = G4.reshape(2, 2, 2, 2)                       # (out_a, out_b, in_a, in_b)
    t = np.tensordot(G, t, axes=([2, 3], [a, b]))
    t = np.moveaxis(t, [0, 1], [a, b])
    t = np.tensordot(t, G.conj(), axes=([a + n, b + n], [2, 3]))
    t = np.moveaxis(t, [-2, -1], [a + n, b + n])
    return t.reshape(2 ** n, 2 ** n)


def mp_channel_unitary(lam: float) -> np.ndarray:
    """U_MP = exp(-i lambda Z (x) X). Exactly the identity at lambda = 0."""
    zx = np.kron(np.array([[1, 0], [0, -1]], dtype=complex),
                 np.array([[0, 1], [1, 0]], dtype=complex))
    return expm(-1j * float(lam) * zx)


@dataclass(frozen=True)
class ArchitectureSpec:
    """Preregistered architecture. Every field enters the cache key."""

    architecture: str = "dual_route_weak"
    memory: mem.MemorySpec = field(default_factory=mem.MemorySpec)
    processor: proc.ProcessorSpec = field(default_factory=proc.ProcessorSpec)
    lam: float = 0.35
    mem_tap: int = None            # default: the LAST rail (deepest delay)
    proc_entry: int = 0

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {self.architecture!r}; known: {ARCHITECTURES}")
        if self.mem_tap is None:
            object.__setattr__(self, "mem_tap", self.memory.L - 1)
        if not 0 <= self.mem_tap < self.memory.L:
            raise ValueError(f"mem_tap {self.mem_tap} outside 0..{self.memory.L - 1}")
        if not 0 <= self.proc_entry < self.processor.N_P:
            raise ValueError(f"proc_entry {self.proc_entry} outside 0..{self.processor.N_P - 1}")

    @property
    def effective_lambda(self) -> float:
        """Only `dual_route_weak` has a channel; everything else is lambda=0."""
        return float(self.lam) if self.architecture == "dual_route_weak" else 0.0

    @property
    def has_memory(self) -> bool:
        return self.architecture != "processor_only"

    @property
    def has_processor(self) -> bool:
        return self.architecture != "memory_only"

    def as_dict(self) -> dict:
        return {"architecture": self.architecture, "lambda": self.effective_lambda,
                "lambda_nominal": float(self.lam), "mem_tap": int(self.mem_tap),
                "proc_entry": int(self.proc_entry),
                "memory": self.memory.as_dict() if self.has_memory else None,
                "processor": self.processor.as_dict() if self.has_processor else None,
                "coupling": "exp(-i*lambda*Z_tap (x) X_entry)"}


@dataclass
class ArchitectureRun:
    u: np.ndarray
    X_M: np.ndarray
    X_P: np.ndarray
    labels_M: list
    labels_P: list
    spec: ArchitectureSpec
    n_qubits: int
    back_action: dict = field(default_factory=dict)
    sim_seconds: float = 0.0


def run_architecture(spec: ArchitectureSpec, u_seq, *, m: float, g: float, J: float,
                     disorder_seed: int, hamiltonian_seed: int = None,
                     ablation: str = "full", measure_back_action: bool = False,
                     m_leak_into_processor: float = 0.0) -> ArchitectureRun:
    """Exact (noiseless) trajectory. Shot noise is applied afterwards by the
    caller, so one simulation serves every shot budget.

    `m_leak_into_processor` is the INJECTED-DEPENDENCY negative control: it
    adds `leak * m` to the processor's g, creating a hidden m -> X_P path that
    the structural-isolation check MUST detect. It is 0.0 in every scientific
    run and non-zero only in the falsifiability control.
    """
    import time
    t0 = time.perf_counter()
    u_seq = np.asarray(u_seq, dtype=float)
    lam = spec.effective_lambda

    labels_M, ops_M = (mem.memory_feature_ops(spec.memory.L, spec.memory.max_weight)
                       if spec.has_memory else ([], []))
    labels_P, ops_P = (local_pauli_ops(spec.processor.N_P, spec.processor.readout)
                       if spec.has_processor else ([], []))

    # ---- processor side: state depends only on u_t and (g, J), never on m ----
    if spec.has_processor:
        g_eff = float(g) + float(m_leak_into_processor) * float(m)
        ham = proc.build_hamiltonians(spec.processor, disorder_seed)
        U_P = proc.processor_unitary(spec.processor, ham, g_eff, J, ablation=ablation,
                                     hamiltonian_seed=hamiltonian_seed)
        enc = spec.processor.encoder()
    if spec.has_memory:
        chan = mem.build_channel(spec.memory, m, disorder_seed)
        rho_M = np.zeros((spec.memory.dim, spec.memory.dim), dtype=complex)
        rho_M[0, 0] = 1.0

    T = u_seq.size
    X_M = np.empty((T, len(ops_M)), dtype=float) if ops_M else np.empty((T, 0))
    X_P = np.empty((T, len(ops_P)), dtype=float) if ops_P else np.empty((T, 0))
    ba_obs, ba_trace = [], []

    n_M, n_P = (spec.memory.L if spec.has_memory else 0), (spec.processor.N_P if spec.has_processor else 0)
    n_total = n_M + n_P
    D_M = spec.memory.dim if spec.has_memory else 1
    D_P = spec.processor.dim if spec.has_processor else 1
    G_MP = mp_channel_unitary(lam) if lam != 0.0 else None

    for t, u_t in enumerate(u_seq):
        if spec.has_memory:
            rho_M = chan.step(rho_M, float(u_t))
        if spec.has_processor:
            rho_P = U_P @ enc.state(float(u_t)) @ U_P.conj().T

        if lam != 0.0 and spec.has_memory and spec.has_processor:
            joint = np.kron(rho_M, rho_P)
            joint = apply_two_qubit(joint, G_MP, spec.mem_tap, n_M + spec.proc_entry, n_total)
            blocks = joint.reshape(D_M, D_P, D_M, D_P)
            rho_M_post = np.einsum('iaja->ij', blocks)
            rho_P_post = np.einsum('aiaj->ij', blocks)
            if measure_back_action:
                ba_obs.append(float(np.max(np.abs(expectation_values(rho_M_post, ops_M)
                                                  - expectation_values(rho_M, ops_M)))))
                d = (rho_M_post - rho_M)
                ba_trace.append(float(0.5 * np.sum(np.abs(
                    np.linalg.eigvalsh((d + d.conj().T) / 2.0)))))
            rho_M = rho_M_post
            rho_P = rho_P_post

        if spec.has_memory:
            X_M[t] = expectation_values(rho_M, ops_M)
        if spec.has_processor:
            X_P[t] = expectation_values(rho_P, ops_P)

    back_action = {}
    if measure_back_action and ba_obs:
        back_action = {"max_observable_disturbance": float(np.max(ba_obs)),
                       "mean_observable_disturbance": float(np.mean(ba_obs)),
                       "max_trace_distance": float(np.max(ba_trace)),
                       "mean_trace_distance": float(np.mean(ba_trace)),
                       "lambda": lam}
    return ArchitectureRun(u=u_seq, X_M=X_M, X_P=X_P, labels_M=labels_M, labels_P=labels_P,
                           spec=spec, n_qubits=n_total, back_action=back_action,
                           sim_seconds=time.perf_counter() - t0)


def check_structural_isolation(spec: ArchitectureSpec, u_seq, *, m: float, g: float, J: float,
                                disorder_seed: int, hamiltonian_seed: int = None,
                                dm: float = 0.05, dgJ: float = 0.05,
                                atol: float = 1e-10, m_leak_into_processor: float = 0.0) -> dict:
    """Gate A: does X_P depend on m, and does X_M depend on (g, J)?

    For `dual_route_current` both must be EXACTLY zero (separate objects). The
    check is falsifiable: pass `m_leak_into_processor > 0` and it must FAIL.
    """
    def run(mm, gg, JJ):
        return run_architecture(spec, u_seq, m=mm, g=gg, J=JJ, disorder_seed=disorder_seed,
                                hamiltonian_seed=hamiltonian_seed,
                                m_leak_into_processor=m_leak_into_processor)

    base = run(m, g, J)
    dXP_dm = float(np.max(np.abs(run(m + dm, g, J).X_P - base.X_P))) if base.X_P.size else 0.0
    dXM_dg = float(np.max(np.abs(run(m, g + dgJ, J).X_M - base.X_M))) if base.X_M.size else 0.0
    dXM_dJ = float(np.max(np.abs(run(m, g, J + dgJ).X_M - base.X_M))) if base.X_M.size else 0.0
    return {"max_dXP_dm": dXP_dm, "max_dXM_dg": dXM_dg, "max_dXM_dJ": dXM_dJ,
            "atol": atol, "architecture": spec.architecture,
            "lambda": spec.effective_lambda,
            "XP_independent_of_m": bool(dXP_dm <= atol),
            "XM_independent_of_gJ": bool(dXM_dg <= atol and dXM_dJ <= atol),
            "passed": bool(dXP_dm <= atol and dXM_dg <= atol and dXM_dJ <= atol)}


def resource_row(spec: ArchitectureSpec, run: ArchitectureRun, budget, feature_rank=None) -> dict:
    """Matched-resource accounting for one architecture."""
    from .v3_2_readout import account_resources
    pres = spec.processor.resources() if spec.has_processor else {}
    return account_resources(
        name=spec.architecture,
        memory_qubits=spec.memory.L if spec.has_memory else 0,
        processor_qubits=spec.processor.N_P if spec.has_processor else 0,
        input_copies=(1 if spec.has_memory else 0) + (pres.get("input_copies", 0)),
        ancillas=0, labels_M=run.labels_M, labels_P=run.labels_P, budget=budget,
        feature_rank=feature_rank,
        circuit_depth=(spec.processor.R * 2 + 1) if spec.has_processor else spec.memory.L,
        two_body_gates=pres.get("two_body_terms", 0) * pres.get("interaction_layers", 0),
        four_body_gates=pres.get("four_body_terms", 0) * pres.get("interaction_layers", 0),
        sim_seconds=run.sim_seconds)


# =============================================================================
# The single evaluation entry point used by CALIBRATION, DISCOVERY and
# CONFIRMATION. One simulation -> shot noise -> IPC report, so the three
# stages can never drift apart in how a point is measured.
# =============================================================================
def evaluate_point(spec: ArchitectureSpec, *, m: float, g: float, J: float, seeds,
                   budget, T: int, washout: int, n_val: int, n_test: int,
                   tau_min: int, max_delay: int, max_degree: int,
                   max_targets_per_degree: int, n_surrogates: int,
                   ablation: str = "full", measure_back_action: bool = False,
                   m_leak_into_processor: float = 0.0, u_seq=None) -> dict:
    """Evaluate one (m, g, J) point and return M_long, NL_0 and the audits.

    `seeds` is a `v3_seeds.NestedSeeds` bundle; every stream is used for the
    one thing it names, which is what makes common random numbers work: two
    points that share `input_sequence`, `disorder` and `shot_noise` differ
    ONLY by the control being varied, so a finite difference between them is
    not contaminated by resampling noise.
    """
    import numpy as np
    from qrc_qiskit import chrono_split
    from .v3_2_readout import add_shot_noise
    from .v3_2_ipc import compute_ipc_report
    from .v3_2_encoder import is_density_matrix

    if u_seq is None:
        u_seq = np.random.default_rng(int(seeds.input_sequence)).uniform(0.0, 1.0, int(T))
    u_seq = np.asarray(u_seq, dtype=float)

    run = run_architecture(spec, u_seq, m=m, g=g, J=J,
                           disorder_seed=int(seeds.disorder),
                           hamiltonian_seed=int(seeds.hamiltonian), ablation=ablation,
                           measure_back_action=measure_back_action,
                           m_leak_into_processor=m_leak_into_processor)

    X_M = add_shot_noise(run.X_M, budget, int(seeds.shot_noise)) if run.X_M.size else run.X_M
    X_P = add_shot_noise(run.X_P, budget, int(seeds.shot_noise) + 1) if run.X_P.size else run.X_P

    gap = max(int(max_delay), 1)
    train, val, test = chrono_split(len(u_seq), washout=washout, n_val=n_val,
                                    n_test=n_test, gap=gap)

    out = {"m": float(m), "g": float(g), "J": float(J),
           "architecture": spec.architecture, "ablation": ablation,
           "lambda": spec.effective_lambda, "sim_seconds": run.sim_seconds,
           "n_qubits": run.n_qubits, "back_action": run.back_action,
           "T": int(len(u_seq)), "splits": {"train": len(train), "val": len(val),
                                            "test": len(test), "gap": gap}}

    # MODULE-SPECIFIC metrics: M_long from X_M only, NL_0 from X_P only.
    # A combined-feature readout is a system-level diagnostic and never a
    # substitute for module-specific causal evidence.
    if X_M.size:
        rep_M = compute_ipc_report(u_seq, X_M, train, val, test, tau_min=tau_min,
                                   max_delay=max_delay, max_degree=max_degree,
                                   max_targets_per_degree=max_targets_per_degree,
                                   n_surrogates=n_surrogates, seed=int(seeds.regression_split),
                                   null_seed=int(seeds.null_surrogate),
                                   max_structural_degree=spec.memory.L)
        out["M"] = rep_M
        out["M_long"] = rep_M.M_long
    if X_P.size:
        rep_P = compute_ipc_report(u_seq, X_P, train, val, test, tau_min=tau_min,
                                   max_delay=max_delay, max_degree=max_degree,
                                   max_targets_per_degree=max_targets_per_degree,
                                   n_surrogates=n_surrogates, seed=int(seeds.regression_split),
                                   null_seed=int(seeds.null_surrogate),
                                   max_structural_degree=spec.processor.N_P)
        out["P"] = rep_P
        out["NL_0"] = rep_P.NL_0
        out["NL_temporal"] = rep_P.NL_temporal

    # density-matrix audit on the final memory state (Gate B)
    if spec.has_memory:
        from . import v3_2_memory as _mem
        chan = _mem.build_channel(spec.memory, m, int(seeds.disorder))
        rho = np.zeros((spec.memory.dim, spec.memory.dim), dtype=complex)
        rho[0, 0] = 1.0
        for u in u_seq[:min(len(u_seq), 25)]:
            rho = chan.step(rho, float(u))
        out["dm_audit"] = is_density_matrix(rho)
    return out
