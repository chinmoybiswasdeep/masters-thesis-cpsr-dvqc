"""
dual_route.py -- V3's four architectures, built from the SAME memory bank
(`memory_bank.py`) and the SAME nonlinear processor
(`nonlinear_processor.py`) so an A-vs-B comparison isolates the
ARCHITECTURE, not a change of module implementation.

    serial              u -> M(m) --lambda_serial--> P(g,J)      (no direct u->P route)
    dual_route_current  u -> M(m) ;  u -> P(g,J)                 (no M->P channel)
    parallel_fixed_taps u -> M(m) ;  (u_t..u_{t-L+1}) -> P(g,J)  (no M->P channel)
    dual_route_weak     u -> M(m) ;  u -> P(g,J) ; M --lambda--> P

Structural isolation for `dual_route_current` and `parallel_fixed_taps`
is architectural, not merely empirical: with no M->P channel the two
modules are simulated as SEPARATE circuits, so X_P cannot depend on `m`
and X_M cannot depend on `(g,J)` -- there is no code path by which they
could. `isolation.py` verifies that the implementation really does this
(and is proven able to FAIL via a deliberately injected cross-dependency).

The M->P channel is exp(-i*lambda*(Z_mem_tap ⊗ X_proc_entry)), a genuine
two-qubit unitary (hence CPTP on the joint register). At lambda = 0 it is
EXACTLY the identity, so `dual_route_weak` reduces bit-for-bit to
`dual_route_current`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm

from .memory_bank import (MemoryBankConfig, memory_feature_ops, encode_diagonal_z,
                           fractional_swap_matrix, run_memory_bank)
from .nonlinear_processor import (ProcessorConfig, processor_feature_ops, processor_unitary,
                                   run_processor, build_tap_buffer)
from .utils import ensure_repo_code_on_path
from .v3_seeds import NestedSeeds

ensure_repo_code_on_path()

from qrc_qiskit import make_simulator, random_input  # noqa: E402

ARCHITECTURES = ("serial", "dual_route_current", "parallel_fixed_taps", "dual_route_weak")

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class DualRouteConfig:
    architecture: str = "dual_route_current"
    memory: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    lam: float = 0.0              # M->P coupling (used by dual_route_weak / serial)
    lam_serial: float = 0.5       # coupling used when architecture == 'serial'
    tap_depth: int = 4            # L for parallel_fixed_taps (classical buffer)
    mem_tap_rail: int = 1         # memory rail the M->P channel reads from
    proc_entry: int = 0           # processor qubit the M->P channel writes into

    def as_dict(self) -> dict:
        return {"architecture": self.architecture, "memory": self.memory.as_dict(),
                "processor": self.processor.as_dict(), "lam": self.lam, "lam_serial": self.lam_serial,
                "tap_depth": self.tap_depth, "mem_tap_rail": self.mem_tap_rail,
                "proc_entry": self.proc_entry}

    @property
    def has_mp_channel(self) -> bool:
        return self.architecture in ("serial", "dual_route_weak")

    @property
    def has_direct_route(self) -> bool:
        return self.architecture != "serial"

    @property
    def effective_lambda(self) -> float:
        if self.architecture == "serial":
            return self.lam_serial
        if self.architecture == "dual_route_weak":
            return self.lam
        return 0.0


def mp_channel_unitary(lam: float) -> np.ndarray:
    """exp(-i * lam * Z ⊗ X) on (memory_tap, processor_entry). At lam = 0
    this is EXACTLY the 4x4 identity, giving the exact Architecture-B
    reduction required of Architecture D."""
    generator = np.kron(_X, _Z)   # Qiskit little-endian: qargs[0] is the RIGHT factor
    return expm(-1j * lam * generator)


@dataclass
class DualRouteRun:
    u: np.ndarray
    labels_mem: list
    labels_proc: list
    X_M: np.ndarray
    X_P: np.ndarray
    cfg: DualRouteConfig
    circuit_depth: int = 0
    n_qubits: int = 0
    d_taps: np.ndarray = None


def _run_independent(cfg: DualRouteConfig, u, seeds: NestedSeeds, method, use_gpu) -> DualRouteRun:
    """Architectures with NO M->P channel: memory and processor are two
    SEPARATE circuits. `m` is not in scope anywhere in the processor call
    and `(g,J)` are not in scope anywhere in the memory call -- structural
    isolation by construction."""
    mem_run = run_memory_bank(cfg.memory, u, seeds, method=method, use_gpu=use_gpu)

    if cfg.architecture == "parallel_fixed_taps":
        d = build_tap_buffer(u, cfg.tap_depth)
        proc_cfg = replace(cfg.processor, n_taps=cfg.tap_depth)
    else:
        d = build_tap_buffer(u, 1)
        proc_cfg = replace(cfg.processor, n_taps=1)
    proc_run = run_processor(proc_cfg, d, seeds, method=method, use_gpu=use_gpu)

    return DualRouteRun(u=np.asarray(u, dtype=float), labels_mem=mem_run.labels,
                         labels_proc=proc_run.labels, X_M=mem_run.X_M, X_P=proc_run.X_P, cfg=cfg,
                         circuit_depth=mem_run.circuit_depth + proc_run.circuit_depth,
                         n_qubits=cfg.memory.L + proc_cfg.N_P, d_taps=d)


def _run_coupled(cfg: DualRouteConfig, u, seeds: NestedSeeds, method, use_gpu) -> DualRouteRun:
    """Architectures WITH an M->P channel (serial, dual_route_weak): one
    joint circuit over memory rails + processor qubits."""
    L, N_P = cfg.memory.L, cfg.processor.N_P
    n_total = L + N_P
    mem_rails = list(range(L))
    proc_qubits = list(range(L, L + N_P))
    lam = cfg.effective_lambda

    proc_cfg = replace(cfg.processor, n_taps=1)
    labels_mem, ops_mem = memory_feature_ops(mem_rails, max_weight=cfg.memory.max_weight)
    labels_proc, ops_proc = processor_feature_ops(proc_qubits, max_weight=cfg.processor.max_weight)

    swap_gate = UnitaryGate(fractional_swap_matrix(cfg.memory.m), label=f"SWAP^{cfg.memory.m:.3f}")
    coupling_gate = UnitaryGate(mp_channel_unitary(lam), label=f"MP({lam:.3f})")

    qc = QuantumCircuit(n_total)
    u = np.asarray(u, dtype=float)
    for t, u_t in enumerate(u):
        # --- memory branch ---
        qc.reset(mem_rails[0])
        encode_diagonal_z(qc, mem_rails[0], u_t)
        for r in range(L - 1, 0, -1):
            qc.append(swap_gate, [mem_rails[r - 1], mem_rails[r]])

        # --- processor branch ---
        for q in proc_qubits:
            qc.reset(q)
        drive = (2.0 * u_t - 1.0) if cfg.has_direct_route else 0.0
        qc.append(UnitaryGate(processor_unitary(proc_cfg, [drive]), label=f"U_P(t={t})"), proc_qubits)

        # --- M -> P channel ---
        if lam != 0.0:
            qc.append(coupling_gate, [mem_rails[cfg.mem_tap_rail], proc_qubits[cfg.proc_entry]])

        for (pauli, qargs), lab in zip(ops_mem, labels_mem):
            qc.save_expectation_value(pauli, qargs, label=f"mem_{lab}__t{t}")
        for (pauli, qargs), lab in zip(ops_proc, labels_proc):
            qc.save_expectation_value(pauli, qargs, label=f"proc_{lab}__t{t}")

    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim, optimization_level=1)
    data = sim.run(tqc, shots=1).result().data(0)
    T = len(u)
    X_M = np.array([[float(np.real(data[f"mem_{lab}__t{t}"])) for lab in labels_mem] for t in range(T)])
    X_P = np.array([[float(np.real(data[f"proc_{lab}__t{t}"])) for lab in labels_proc] for t in range(T)])

    return DualRouteRun(u=u, labels_mem=labels_mem, labels_proc=labels_proc, X_M=X_M, X_P=X_P,
                         cfg=cfg, circuit_depth=tqc.depth(), n_qubits=n_total,
                         d_taps=u.reshape(-1, 1))


def run_architecture(cfg: DualRouteConfig, T: int, seeds: NestedSeeds, u=None,
                      method: str = "density_matrix", use_gpu: bool = False) -> DualRouteRun:
    """Runs one architecture for `T` macro timesteps. `u` may be supplied
    explicitly (for common-random-number comparisons); otherwise it is
    drawn from `seeds.input_sequence` and mapped to [-1,1] to match the
    diagonal encoding rho(u) = (I + u Z)/2."""
    if cfg.architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {cfg.architecture!r}; expected one of {ARCHITECTURES}")
    if method != "density_matrix":
        raise ValueError("V3 circuits reset qubits every step -- use method='density_matrix'.")
    if u is None:
        # canonical [0,1] convention, matching `random_input` and every IPC target builder;
        # the encoders map to signed [-1,1] internally (see `memory_bank.to_signed`).
        u = random_input(T, seed=seeds.input_sequence)
    u = np.asarray(u, dtype=float)[:T]

    if cfg.has_mp_channel and cfg.effective_lambda != 0.0:
        return _run_coupled(cfg, u, seeds, method, use_gpu)
    if cfg.architecture == "serial":
        # a serial architecture with zero coupling has NO route into the processor at all;
        # still run the coupled path so the (degenerate) result is physically honest.
        return _run_coupled(cfg, u, seeds, method, use_gpu)
    if cfg.architecture == "dual_route_weak":
        # lambda == 0: EXACTLY Architecture B, by construction (identity channel).
        return _run_independent(replace(cfg, architecture="dual_route_current"), u, seeds, method, use_gpu)
    return _run_independent(cfg, u, seeds, method, use_gpu)
