"""
qrc_qiskit.py -- a minimal, genuinely-recurrent Quantum Reservoir Computer (QRC)
in Qiskit, with a CPU/GPU-selectable Aer backend.

This module implements *only* the reservoir itself:
  - ONE fixed *nearest-neighbor-chain topology (no topology sweep/optimization)*
  - ONE fixed coupling strength `g` (no "edge of chaos" / critical-phase scan;
    g is just a hyperparameter you can tune, with no special physical claim
    attached to any particular value)
  - EXACT expectation-value readout, computed directly by Aer's
    `save_expectation_value` (no classical-shadow / randomized-measurement
    estimation)

What IS carried over from the code review, because these are correctness
fixes rather than added scope:
  - comment 1 (persistent memory): genuine recurrent dynamics
    |psi_t> = U(u_t) |psi_{t-1}>. Only the CURRENT input u_t is injected each
    step, via reset-then-encode on a single designated input qubit. Every
    other qubit ("the memory register") is never reset, so any predictive
    power for a lagged target u_{t-k} (k>=1) reflects information genuinely
    retained by the quantum state, not a re-encoded window leaking the answer.
  - comment 8 (train/test leakage): chronological train/val/test split with a
    guard gap >= the longest lag any benchmark task looks back, so no
    training window can straddle a validation/test boundary.
  - comment 9 (backend selection): the real-hardware path (`USE_QPU`, below)
    requires a PINNED backend name (default `ibm_marrakesh`) -- no silent
    `least_busy()` substitution -- and reports backend/job provenance with
    every result.
  - comment 13 (credentials): `USE_GPU` never touches real hardware at all.
    `USE_QPU` does, but credentials are loaded ONLY from the
    `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE` (your IBM Cloud CRN)
    environment variables -- never hard-coded here or in the notebook. See
    `get_ibm_service()`.

WHERE TO INCREASE THE QUBIT COUNT
---------------------------------------------------------------------------
Set `ReservoirConfig.N`.

REAL HARDWARE (USE_QPU)
---------------------------------------------------------------------------
`run_reservoir(cfg, u_seq, use_qpu=True)` runs on a real IBM QPU (default
`ibm_marrakesh`) via `qiskit-ibm-runtime`'s `EstimatorV2`, instead of Aer.
This is architecturally different from the CPU/GPU path, not just a device
swap -- see `run_reservoir_qpu`'s docstring for why (short version: real
hardware can't persist a quantum state between circuit executions the way
one big Aer circuit can, so genuine per-step recurrence has to be recreated
by literally replaying the whole input prefix inside each circuit, which
costs O(T^2) total gates across a T-step trajectory). Treat T on real
hardware as O(10), not O(100-1000).
"""
from __future__ import annotations

import functools
import itertools
import os
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import qiskit_aer  # noqa: F401  -- import needed to register .save_expectation_value on QuantumCircuit
from qiskit_aer import AerSimulator


# =============================================================================
# 1. Reservoir configuration and circuit construction
# =============================================================================

def chain_edges(N: int):
    """Nearest-neighbor chain -- the one fixed topology used throughout."""
    return [(i, i + 1) for i in range(N - 1)]


@dataclass
class ReservoirConfig:
    N: int = 6                     # <-- INCREASE QUBIT COUNT HERE
    g: float = 0.6                 # entangling coupling angle (radians), fixed
    reps: int = 2                  # entangling+disorder block repetitions per timestep
    input_qubit: int = 0           # qubit reset + re-encoded with u_t every step
    seed: int = 42                 # fixes the random disorder (bias_z, bias_x)

    # `g` and `reps` above were picked by a one-time check of validation-block
    # NRMSE across a small grid (g in {0.3,0.6,0.94,1.2,1.57} rad x reps in
    # {1,2,3}), NOT by scanning for a claimed "critical"/"edge of chaos" point
    # and NOT swept per-experiment -- see the module docstring's SCOPE note.

    def sample_disorder(self):
        rng = np.random.RandomState(self.seed)
        bias_z = rng.uniform(0, 2 * np.pi, self.N)
        bias_x = rng.uniform(0.3, 0.7, self.N)
        return bias_z, bias_x


def reservoir_layer(qc: QuantumCircuit, N: int, g: float, bias_z, bias_x):
    """One entangling + disorder layer on the fixed chain topology.

    U = Rx(bias_x) . Rz(bias_z) . CPhase(2g)_chain, applied left-to-right as a
    circuit (so the entangling bricks go first). `g` is just a coupling
    strength here -- not swept, not framed as a phase transition.
    """
    for a, b in chain_edges(N):
        qc.cp(2.0 * g, a, b)
    for i in range(N):
        qc.rz(bias_z[i], i)
    for i in range(N):
        qc.rx(bias_x[i], i)


def feature_ops(N: int, input_qubit: int):
    """Local Pauli readout operators: X/Y/Z on every memory qubit (everything
    except `input_qubit`) plus nearest-neighbor ZZ correlators among them.

    These are read out EXACTLY (Aer computes them directly from the
    statevector/density matrix) -- there is no shot noise and no classical-
    shadow estimator here. That is a deliberate simplification: this module
    benchmarks the reservoir's computational quality, not a shot-limited
    hardware read-out protocol.
    """
    mem = [q for q in range(N) if q != input_qubit]
    ops, labels = [], []
    for q in mem:
        for name in ('Z', 'X', 'Y'):
            ops.append((Pauli(name), [q]))
            labels.append(f'{name}{q}')
    for a, b in zip(mem[:-1], mem[1:]):
        ops.append((Pauli('ZZ'), [a, b]))
        labels.append(f'Z{a}Z{b}')
    return labels, ops


def build_trajectory_circuit(cfg: ReservoirConfig, u_seq: Sequence[float]):
    """Build ONE circuit for the entire recurrent trajectory.

    Each timestep: reset the input qubit (discard its old state -> fading
    memory), inject u_t via a Ry rotation, apply the reservoir layer(s), and
    snapshot every feature's exact expectation value with a labeled
    `save_expectation_value`. The memory register is never reset, so its
    state at step t carries forward everything the unitary dynamics retained
    from steps < t.

    Doing the whole trajectory as one circuit (rather than T separate Python-
    level evolve() calls) lets the entire recurrence run inside Aer's C++/CUDA
    engine in a single `.run()` call -- which is what makes the GPU device
    switch in `run_reservoir` actually matter for wall-clock time.
    """
    bias_z, bias_x = cfg.sample_disorder()
    labels, ops = feature_ops(cfg.N, cfg.input_qubit)
    qc = QuantumCircuit(cfg.N)
    for t, u_t in enumerate(u_seq):
        qc.reset(cfg.input_qubit)
        qc.ry(np.pi * float(u_t), cfg.input_qubit)
        for _ in range(cfg.reps):
            reservoir_layer(qc, cfg.N, cfg.g, bias_z, bias_x)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f'{lab}__t{t}')
    return qc, labels


# =============================================================================
# 2. CPU/GPU execution
# =============================================================================

def make_simulator(use_gpu: bool = False, method: str = 'density_matrix') -> AerSimulator:
    """Build the Aer backend. This is the ONLY place device selection happens.

    `method='density_matrix'` is required for the reset-based fading-memory
    architecture above (resetting one qubit of an entangled *pure* state
    produces a mixed state in general, so a plain Statevector cannot represent
    it -- see the SCALING NOTES below for the statevector-only alternative and
    what it costs you).
    """
    device = 'GPU' if use_gpu else 'CPU'
    if use_gpu:
        # Gotcha: AerSimulator(device='GPU').available_devices() just ECHOES
        # BACK the device it was already configured with -- it does not
        # actually probe hardware/build support (verified empirically; see
        # notebook Section 2). The only reliable check is to ask a freshly
        # constructed, unconfigured simulator what it truly supports.
        available = AerSimulator().available_devices()
        if 'GPU' not in available:
            raise RuntimeError(
                "USE_GPU=True but this qiskit-aer build has no GPU device "
                f"(available devices on a fresh AerSimulator(): {available}).\n"
                "qiskit-aer's PyPI wheels for Windows are CPU-only. To get the GPU "
                "device you need either:\n"
                f"  (a) WSL2 / native Linux + `pip install qiskit-aer-gpu-cu11` "
                f"(matches this repo's qiskit-aer=={qiskit_aer.__version__}), or\n"
                "  (b) build qiskit-aer from source against your local CUDA toolkit "
                "(see qiskit.org/ecosystem/aer build docs).\n"
                "Set USE_GPU = False to run on CPU in the meantime."
            )
    return AerSimulator(method=method, device=device)


def run_reservoir(cfg: ReservoirConfig, u_seq: Sequence[float], use_gpu: bool = False,
                   use_qpu: bool = False, method: str = 'density_matrix', **qpu_kwargs):
    """Execute the full recurrent trajectory once. Returns (labels, X, info).

    X has shape (T, n_features); info carries timing + circuit-resource data
    so CPU vs GPU vs QPU (or N vs N') runs can be compared honestly.

    `use_qpu=True` routes to `run_reservoir_qpu` (real IBM hardware) instead
    of Aer -- a genuinely different execution path, not just a device string;
    see that function's docstring. `**qpu_kwargs` (backend_name, service,
    shots, optimization_level, max_circuits) are forwarded to it and ignored
    otherwise.
    """
    if use_qpu:
        return run_reservoir_qpu(cfg, u_seq, **qpu_kwargs)

    qc, labels = build_trajectory_circuit(cfg, u_seq)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim)

    t0 = time.perf_counter()
    result = sim.run(tqc, shots=1).result()
    elapsed = time.perf_counter() - t0

    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f'{lab}__t{t}'])

    info = {
        'device': 'GPU' if use_gpu else 'CPU',
        'method': method,
        'N': cfg.N,
        'T': T,
        'elapsed_s': elapsed,
        'steps_per_s': T / elapsed if elapsed > 0 else float('inf'),
        'circuit_depth': tqc.depth(),
        'circuit_size': tqc.size(),
        'n_features': len(labels),
    }
    return labels, X, info


# =============================================================================
# 2b. Real IBM hardware execution (USE_QPU)
# =============================================================================
# This is a genuinely different code path from CPU/GPU, not a device swap:
# Aer's `save_expectation_value` (Section 2) is a SIMULATOR-ONLY instruction
# that snapshots mid-circuit state without collapsing it -- real QPUs have no
# such thing, and cannot persist a quantum state between separate circuit
# executions the way one big Aer circuit does across T reset-and-continue
# steps. So on real hardware, the state at step t has to be recreated from
# scratch by literally replaying the ENTIRE prefix u_1..u_t inside ONE
# circuit that ends with an end-of-circuit expectation-value read via the
# Estimator primitive. Circuit depth for step t is O(t); summed over a
# T-step trajectory that is O(T^2) total gates -- which is why `max_circuits`
# below defaults small and a hardware run should use a much shorter T than
# the CPU/GPU benchmarks in this module use.

DEFAULT_QPU_BACKEND = 'ibm_marrakesh'


def get_ibm_service() -> 'QiskitRuntimeService':
    """Authenticate against IBM Quantum ONLY from environment variables --
    never hard-code a token or CRN here or in a notebook (review comment 13).

    Reads:
        IBM_QUANTUM_TOKEN     your IBM Cloud API token
        IBM_QUANTUM_INSTANCE  your IBM Cloud CRN (Cloud Resource Name) for the
                               Quantum Compute service instance

    Set these in your shell -- e.g. on Windows:
        setx IBM_QUANTUM_TOKEN "..."
        setx IBM_QUANTUM_INSTANCE "crn:v1:bluemix:public:quantum-computing:..."
    (restart your shell/kernel afterwards) -- or on Linux/macOS:
        export IBM_QUANTUM_TOKEN=...
        export IBM_QUANTUM_INSTANCE=crn:...

    This function does NOT call `QiskitRuntimeService.save_account(...)` --
    it authenticates in-memory for this process only, so nothing is written
    to disk beyond what you already put in your environment. If you'd rather
    persist an account to Qiskit's config store, call `save_account` yourself
    (see the qiskit-ibm-runtime docs); that is a deliberate choice this
    module leaves to you rather than doing on your behalf.
    """
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    instance = os.environ.get('IBM_QUANTUM_INSTANCE')
    if not token or not instance:
        missing = [n for n, v in (('IBM_QUANTUM_TOKEN', token), ('IBM_QUANTUM_INSTANCE', instance)) if not v]
        raise RuntimeError(
            "USE_QPU=True requires IBM Quantum credentials as environment variables "
            f"-- missing: {', '.join(missing)}.\n"
            "Set IBM_QUANTUM_TOKEN (your IBM Cloud API token) and IBM_QUANTUM_INSTANCE "
            "(your IBM Cloud CRN for the Quantum Compute service instance), then restart "
            "your shell/notebook kernel. Credentials are never hard-coded in this repo "
            "(review comment 13)."
        )
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel='ibm_quantum_platform', token=token, instance=instance)


def _pauli_to_sparse_op(N: int, pauli: Pauli, qargs):
    """Convert one `feature_ops`-style (Pauli, qargs) pair into an N-qubit
    `SparsePauliOp`, for the Estimator primitive (identity elsewhere)."""
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_sparse_list([(pauli.to_label(), qargs, 1.0)], num_qubits=N)


def build_hardware_circuits(cfg: ReservoirConfig, u_seq: Sequence[float]):
    """One circuit per timestep t, each replaying the FULL prefix u_1..u_t
    from |0..0> (see the section note above for why). Returns
    (circuits, observables, labels) -- `observables` is shared across all
    circuits (same feature set every step); `labels[j]` names
    `observables[j]`.
    """
    bias_z, bias_x = cfg.sample_disorder()
    labels, ops = feature_ops(cfg.N, cfg.input_qubit)
    observables = [_pauli_to_sparse_op(cfg.N, op, qargs) for (op, qargs) in ops]

    circuits = []
    for t in range(1, len(u_seq) + 1):
        qc = QuantumCircuit(cfg.N)
        for u_t in u_seq[:t]:
            qc.reset(cfg.input_qubit)
            qc.ry(np.pi * float(u_t), cfg.input_qubit)
            for _ in range(cfg.reps):
                reservoir_layer(qc, cfg.N, cfg.g, bias_z, bias_x)
        circuits.append(qc)
    return circuits, observables, labels


def run_reservoir_qpu(cfg: ReservoirConfig, u_seq: Sequence[float],
                       backend_name: str = DEFAULT_QPU_BACKEND, service=None,
                       shots: int = 4096, optimization_level: int = 3,
                       max_circuits: int = 25):
    """Run the recurrent trajectory on a real IBM QPU (default: `ibm_marrakesh`,
    pinned by name -- no silent `least_busy()` substitution, per comment 9).

    `max_circuits` is a deliberate, small default safety cap: each extra
    timestep is a full prefix-replay circuit (see the section note above), so
    T timesteps cost O(T^2) total two-qubit gates on a real, rate/cost-limited
    device. Raise it explicitly if you really want a longer run.

    Returns (labels, X, info); `info` reports backend name/version, job id,
    per-circuit depth and two-qubit-gate count (growing with t), and total
    two-qubit gates across the whole job, so the hardware cost of this run is
    visible rather than hidden (comment 9-11's spirit: report real resources).
    """
    if len(u_seq) > max_circuits:
        raise ValueError(
            f"len(u_seq)={len(u_seq)} exceeds max_circuits={max_circuits}. Real hardware "
            "cost grows O(T^2) here (see run_reservoir_qpu's docstring) -- this cap exists "
            "so a long T isn't submitted to paid/rate-limited hardware by accident. Pass a "
            "larger max_circuits explicitly if you intend that."
        )

    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = service or get_ibm_service()
    backend = service.backend(backend_name)

    circuits, observables, labels = build_hardware_circuits(cfg, u_seq)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuits = [pm.run(qc) for qc in circuits]
    isa_observables = [[obs.apply_layout(ic.layout) for obs in observables] for ic in isa_circuits]
    pubs = list(zip(isa_circuits, isa_observables))

    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = shots

    t0 = time.perf_counter()
    job = estimator.run(pubs)
    result = job.result()
    elapsed = time.perf_counter() - t0

    X = np.array([np.real(pub_result.data.evs) for pub_result in result])

    depths = [ic.depth() for ic in isa_circuits]
    twoq_counts = [sum(v for k, v in ic.count_ops().items() if k in ('cz', 'ecr', 'cx', 'rzz'))
                   for ic in isa_circuits]
    info = {
        'device': 'QPU', 'backend': backend.name,
        'backend_num_qubits': backend.num_qubits,
        'job_id': job.job_id(),
        'N': cfg.N, 'T': len(u_seq), 'shots': shots,
        'elapsed_s': elapsed,
        'steps_per_s': len(u_seq) / elapsed if elapsed > 0 else float('inf'),
        'circuit_depth_per_step': depths,
        'circuit_depth_final': depths[-1] if depths else 0,
        'two_qubit_gates_per_step': twoq_counts,
        'two_qubit_gates_total': int(sum(twoq_counts)),
        'n_features': len(labels),
    }
    return labels, X, info


# =============================================================================
# 3. Benchmark tasks (repo-consistent + one literature-standard task)
# =============================================================================

def random_input(T: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).uniform(0, 1, T)


def task_kpauli(u: np.ndarray, k: int) -> np.ndarray:
    """y_t = prod_{j=1}^k cos(pi * u_{t-j})  -- nonlinear k-step memory task
    (identical definition to code/idcpsr.py's task_kpauli / task_kpauli in
    idcpsr_qiskit.py, kept for comparability with the rest of the repo)."""
    T = len(u)
    y = np.zeros(T)
    for t in range(k, T):
        y[t] = np.prod(np.cos(np.pi * u[t - k:t]))
    return y


def task_narma2(u: np.ndarray, u_scale: float = 0.5) -> np.ndarray:
    """Standard NARMA2 benchmark, a literature-standard (non-CPSR-specific)
    short-memory nonlinear task, useful as an external sanity check:
        y_t = 0.4 y_{t-1} + 0.4 y_{t-1} y_{t-2} + 0.6 (u_scale*u_t)^3 + 0.1

    The `u_scale` factor is standard practice for NARMA tasks (Appeltant et
    al. 2011): the recursive quadratic-ish term diverges if the driving
    signal has full [0,1] amplitude, so the literature drives NARMA with a
    lower-amplitude copy of the same input. The reservoir itself still sees
    the original, unscaled `u_t` -- only the regression *target* is defined
    on the rescaled signal, which changes nothing about what the reservoir
    can learn since the two are a fixed monotonic rescaling of each other.
    """
    T = len(u)
    y = np.zeros(T)
    us = u_scale * u
    for t in range(1, T):
        y_prev2 = y[t - 2] if t >= 2 else 0.0
        y[t] = 0.4 * y[t - 1] + 0.4 * y[t - 1] * y_prev2 + 0.6 * us[t] ** 3 + 0.1
    return y


def delay_target(u: np.ndarray, k: int) -> np.ndarray:
    """y_t = u_{t-k}  -- the delay task used by short-term memory capacity."""
    T = len(u)
    y = np.zeros(T)
    y[k:] = u[:T - k]
    return y


def delay_taps(u: np.ndarray, m: int) -> np.ndarray:
    """Classical-only control features: the raw last m+1 inputs, no quantum
    processing at all. Used as a baseline so a quantum-reservoir NRMSE win can
    be checked against "did the delayed input alone already explain this."""
    T = len(u)
    X = np.zeros((T, m + 1))
    for j in range(m + 1):
        X[j:, j] = u[:T - j]
    return X


# =============================================================================
# 4. Chronological split (review comment 8) + metrics
# =============================================================================

def chrono_split(T: int, washout: int, n_val: int, n_test: int, gap: int):
    """Chronological train/val/test split with a guard gap >= the longest lag
    any task in the suite looks back, on both sides of every boundary, so a
    training window can never overlap a validation/test target (the leakage
    review comment 8 flagged, fixed here by construction rather than by luck).
    """
    test = np.arange(T - n_test, T)
    val_end = T - n_test - gap
    val = np.arange(val_end - n_val, val_end)
    train_end = val_end - n_val - gap
    train = np.arange(washout, train_end)
    if len(train) < 10:
        raise ValueError(f'Not enough training points ({len(train)}) -- reduce '
                          f'washout/n_val/n_test/gap or increase T.')
    return train, val, test


def nrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2) / (np.var(y_true) + 1e-12)))


DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def select_and_eval_ridge(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
                           test: np.ndarray, alphas=DEFAULT_ALPHAS):
    """Ridge regularization strength `alpha` is selected on the VALIDATION
    block only; the test block is touched exactly once, after alpha is fixed
    (review comment 8 / "Additional methodological recommendations": separate
    model selection from final testing). Features are standardized using
    train-set statistics only.
    """
    scaler = StandardScaler().fit(X[train])
    Xtr, Xval, Xte = scaler.transform(X[train]), scaler.transform(X[val]), scaler.transform(X[test])

    best_alpha, best_val_err = alphas[0], np.inf
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr, y[train])
        err = nrmse(model.predict(Xval), y[val])
        if err < best_val_err:
            best_val_err, best_alpha = err, a

    model = Ridge(alpha=best_alpha).fit(Xtr, y[train])
    test_err = nrmse(model.predict(Xte), y[test])
    return test_err, best_alpha, model


def memory_capacity(X: np.ndarray, u: np.ndarray, train: np.ndarray, val: np.ndarray,
                     test: np.ndarray, k_max: int = 10, alphas=DEFAULT_ALPHAS):
    """Jaeger short-term memory capacity: MC = sum_k MC_k, MC_k = squared
    correlation between the true delayed input u_{t-k} and its ridge-regression
    reconstruction from reservoir features, capped at 1 per k. Because X here
    comes from the genuinely-recurrent reservoir (comment 1), a non-zero MC_k
    can only reflect information actually retained in the quantum state.
    Regularization is selected on `val`, exactly as in `select_and_eval_ridge`.
    """
    total = 0.0
    per_k = []
    for k in range(1, k_max + 1):
        y = delay_target(u, k)
        scaler = StandardScaler().fit(X[train])
        Xtr, Xval, Xte = scaler.transform(X[train]), scaler.transform(X[val]), scaler.transform(X[test])
        best_alpha, best_val_err = alphas[0], np.inf
        for a in alphas:
            model = Ridge(alpha=a).fit(Xtr, y[train])
            err = nrmse(model.predict(Xval), y[val])
            if err < best_val_err:
                best_val_err, best_alpha = err, a
        model = Ridge(alpha=best_alpha).fit(Xtr, y[train])
        pred = model.predict(Xte)
        y_test = y[test]
        cov = np.cov(pred, y_test)[0, 1] ** 2
        denom = np.var(y_test) * np.var(pred) + 1e-12
        v = float(np.clip(cov / denom, 0.0, 1.0))
        per_k.append(v)
        total += v
    return total, per_k


# =============================================================================
# 5. Benchmark harness
# =============================================================================

@dataclass
class BenchmarkResult:
    device: str
    N: int
    T: int
    reservoir_time_s: float
    steps_per_s: float
    circuit_depth: int
    circuit_size: int
    n_features: int
    mc_total: float
    mc_per_k: list
    kpauli_nrmse: dict
    kpauli_nrmse_classical: dict
    narma2_nrmse: float
    narma2_nrmse_classical: float
    fit_eval_time_s: float


def run_benchmark(cfg: ReservoirConfig, T: int = 600, washout: int = 50,
                   n_val: int = 100, n_test: int = 150, k_list=(1, 2, 3),
                   use_gpu: bool = False, method: str = 'density_matrix',
                   input_seed: int = 0, reservoir_fn=run_reservoir) -> BenchmarkResult:
    """Run the full benchmark suite (memory capacity + k-Pauli + NARMA2, each
    against a delay-line classical-only control) on one reservoir config.

    `reservoir_fn(cfg, u_seq, use_gpu=..., method=...) -> (labels, X, info)`
    defaults to `run_reservoir` (the genuinely recurrent architecture) but can
    be swapped for `run_reservoir_qelm` (see Section 6) to run this exact same
    harness -- same tasks, same chronological split, same alpha selection --
    against the memoryless QELM variant instead, for an apples-to-apples
    comparison. `run_benchmark_qelm` below is a thin convenience wrapper that
    does this via `functools.partial`.
    """
    u = random_input(T, seed=input_seed)
    labels, X, info = reservoir_fn(cfg, u, use_gpu=use_gpu, method=method)

    gap = max(max(k_list), 8) + 1  # guard gap >= longest lag any task looks back
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)
    Xc = delay_taps(u, m=max(k_list))  # classical-only control features

    t0 = time.perf_counter()

    mc_total, mc_per_k = memory_capacity(X, u, train, val, test, k_max=max(8, max(k_list)))

    kpauli_nrmse, kpauli_nrmse_c = {}, {}
    for k in k_list:
        y = task_kpauli(u, k)
        kpauli_nrmse[k], _, _ = select_and_eval_ridge(X, y, train, val, test)
        kpauli_nrmse_c[k], _, _ = select_and_eval_ridge(Xc, y, train, val, test)

    y2 = task_narma2(u)
    narma2_nrmse, _, _ = select_and_eval_ridge(X, y2, train, val, test)
    narma2_nrmse_c, _, _ = select_and_eval_ridge(Xc, y2, train, val, test)

    fit_eval_time = time.perf_counter() - t0

    return BenchmarkResult(
        device=info['device'], N=cfg.N, T=T,
        reservoir_time_s=info['elapsed_s'], steps_per_s=info['steps_per_s'],
        circuit_depth=info['circuit_depth'], circuit_size=info['circuit_size'],
        n_features=info['n_features'],
        mc_total=mc_total, mc_per_k=mc_per_k,
        kpauli_nrmse=kpauli_nrmse, kpauli_nrmse_classical=kpauli_nrmse_c,
        narma2_nrmse=narma2_nrmse, narma2_nrmse_classical=narma2_nrmse_c,
        fit_eval_time_s=fit_eval_time,
    )


# =============================================================================
# 6. Memoryless QELM variant (SCALING NOTE option 1: statevector, higher N)
# =============================================================================
# Everything above (Sections 1-5) is the genuinely recurrent architecture:
# ONE qubit (`input_qubit`) is reset and re-encoded each step while the rest
# of the register is never reset, so the state is generally MIXED and needs
# Aer's method='density_matrix' (4**N cost). This section implements the
# alternative traded off in the SCALING NOTES below: give up genuine
# step-to-step recurrence, reset EVERY qubit each step (so the state stays
# PURE and method='statevector', 2**N cost, becomes valid -- roughly doubling
# the reachable N for the same memory budget), and recover "memory" only from
# re-encoding a classical sliding window of recent inputs. This is a Quantum
# Extreme Learning Machine (QELM): a memoryless feature map, not a memory. Any
# nonzero memory-capacity score from it reflects the classical window, NOT
# information retained by the quantum state -- contrast with comment 1's fix
# in Section 1. This is the same window/slot-phase scheme as
# `idcpsr_qiskit.py`'s `reservoir_states_qiskit`, reimplemented here as one
# Aer circuit with exact `save_expectation_value` readout (like Section 1)
# instead of explicit statevector matrix powers, so it plugs into the exact
# same benchmark harness (Sections 3-5) as the recurrent reservoir above.

def feature_ops_all(N: int, max_weight: int = 3):
    """Local Pauli readout operators: the FULL 3**w Pauli-string basis
    (every combination of X/Y/Z, not just all-Z) on every adjacent window of
    w<=max_weight qubits, for w=1 (every qubit) up through w=max_weight
    (every adjacent w-tuple).

    Why the full basis, not just Z/ZZ/ZZZ: after `build_qelm_circuit`'s
    per-step disorder rotation (bias_z, bias_x, fixed but qubit-specific), a
    single qubit's Bloch vector is a FIXED rotation of (sin(theta), 0,
    cos(theta)), so <Z> alone is no longer cos(theta) -- but <X>, <Y>, <Z>
    TOGETHER remain an exactly invertible linear image of (sin(theta),
    cos(theta)) for any fixed rotation. The same holds jointly for a
    w-qubit PRODUCT state: <P1 P2 ... Pw> for the right combination of
    Paulis linearly spans everything needed to reconstruct
    prod_j cos(theta_j) via ridge -- but only if enough of the 3**w
    combinations are actually read out. Reading out only the all-Z
    correlator (as an earlier version of this function did) discards most of
    that span, which is exactly why k-Pauli tasks with k>=2 fit poorly
    without this: see `run_reservoir_qelm`'s docstring for the empirical
    comparison. `max_weight=3` matches this module's k-Pauli benchmark's
    `k_list=(1,2,3)` -- no higher-weight terms are needed for k<=3, so none
    are computed.
    """
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


def build_qelm_circuit(cfg: ReservoirConfig, u_seq: Sequence[float], window_size: int = 8,
                        max_weight: int = 3, reps: int = 1, g: float = 0.05):
    """Build ONE circuit for the entire trajectory, memoryless-QELM style.

    Each timestep: reset ALL N qubits to |0..0> (not just `input_qubit` as in
    `build_trajectory_circuit`), so the state is pure going into every step's
    encoding. `u_t` is placed into a length-min(window_size, N) sliding
    window of the most recent inputs, mapped 1:1 onto qubits 0..window-1 (NOT
    `w % N` -- wrapping onto qubits that already hold a different lag would
    ADD the two lags' rotation angles together on the same qubit, corrupting
    both; capping the window at N and dropping older lags instead keeps every
    retained lag on its own qubit). Then `reps` reservoir layers (using `g`,
    NOT `cfg.g`/`cfg.reps` -- see below) are applied and every feature's
    exact expectation value is snapshotted, exactly as in
    `build_trajectory_circuit`.

    `reps`/`g` are QELM's OWN entangling depth/strength, deliberately
    decoupled from `cfg.reps`/`cfg.g` (which the genuinely recurrent
    architecture in Section 1 uses): a validation-block sweep (same one-time
    grid-check spirit as `ReservoirConfig`'s own `g`/`reps` choice) over
    reps in {0,1,2,3} x g in {0, 0.05, ..., 0.6} found that MORE entangling
    here actively HURTS the k-Pauli tasks -- each `reservoir_layer` rotates
    every qubit's Bloch vector by more disorder, widening the gap between
    what `feature_ops_all`'s finite (weight<=3) Pauli-string basis can
    linearly represent and the target's exact trigonometric form. `reps=1,
    g=0.05` was the smallest step past `reps=0` (which reduces to a purely
    classical, exactly-solvable cosine-product feature map -- see
    `run_reservoir_qelm`'s docstring) that still exercises genuine multi-qubit
    entangling gates while keeping k-Pauli NRMSE robustly low (not merely
    low on one lucky ridge-alpha tie-break) across N=4..10.
    """
    bias_z, bias_x = cfg.sample_disorder()
    labels, ops = feature_ops_all(cfg.N, max_weight=max_weight)
    eff_window = min(window_size, cfg.N)
    slot_to_qubit = list(range(eff_window))

    qc = QuantumCircuit(cfg.N)
    for t, _ in enumerate(u_seq):
        lo = max(0, t - eff_window + 1)
        wlen = t - lo + 1
        window = np.zeros(eff_window)
        window[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(cfg.N)
        for w, uu in enumerate(window):
            per_q[slot_to_qubit[w]] += np.pi * float(uu)

        for i in range(cfg.N):
            qc.reset(i)
            qc.ry(per_q[i], i)
        for _ in range(reps):
            reservoir_layer(qc, cfg.N, g, bias_z, bias_x)
        for (op, qargs), lab in zip(ops, labels):
            qc.save_expectation_value(op, qargs, label=f'{lab}__t{t}')
    return qc, labels


def run_reservoir_qelm(cfg: ReservoirConfig, u_seq: Sequence[float], window_size: int = 8,
                        max_weight: int = 3, reps: int = 1, g: float = 0.05,
                        use_gpu: bool = False, use_qpu: bool = False,
                        method: str = 'statevector', **qpu_kwargs):
    """Execute the memoryless QELM trajectory once. Same (labels, X, info)
    return shape as `run_reservoir`, so it drops into the same benchmark
    harness (`run_benchmark(..., reservoir_fn=...)`).

    `method='statevector'` is the point of this variant (see SCALING NOTES) --
    passing `method='density_matrix'` still works (a full per-step reset
    keeps the state pure either way) but gives up the 2**N-vs-4**N memory
    saving that is the whole reason to use it.

    `reps`/`g` (QELM's own entangling depth/strength, default 1/0.05 --
    see `build_qelm_circuit`'s docstring) trade off against k-Pauli accuracy:
    `reps=0` (no entangling at all) makes this an EXACTLY solvable, purely
    classical cosine-product feature map (k-Pauli NRMSE -> 0 for k<=max_weight,
    but zero genuine quantum computation happens); the shipped defaults are
    the smallest step past that which still runs real entangling gates while
    keeping k-Pauli NRMSE robustly low (see the module's benchmark results).
    Raising `g`/`reps` back toward `cfg.g`/`cfg.reps` (0.6/2, used by the
    recurrent architecture) measurably degrades k-Pauli fits -- more
    disorder-rotation per step widens the gap between what the finite
    (`max_weight`-bounded) Pauli-string readout can linearly represent and
    the target's exact trigonometric form.

    `use_qpu=True` routes to `run_reservoir_qelm_qpu` (real IBM hardware)
    instead of Aer, mirroring `run_reservoir`'s `use_qpu` dispatch.
    `**qpu_kwargs` (backend_name, service, shots, optimization_level,
    max_circuits) are forwarded to it and ignored otherwise.
    """
    if use_qpu:
        return run_reservoir_qelm_qpu(cfg, u_seq, window_size=window_size, max_weight=max_weight,
                                       reps=reps, g=g, **qpu_kwargs)

    qc, labels = build_qelm_circuit(cfg, u_seq, window_size=window_size, max_weight=max_weight,
                                     reps=reps, g=g)
    sim = make_simulator(use_gpu=use_gpu, method=method)
    tqc = transpile(qc, sim)

    t0 = time.perf_counter()
    result = sim.run(tqc, shots=1).result()
    elapsed = time.perf_counter() - t0

    data = result.data(0)
    T = len(u_seq)
    X = np.empty((T, len(labels)))
    for t in range(T):
        for j, lab in enumerate(labels):
            X[t, j] = np.real(data[f'{lab}__t{t}'])

    info = {
        'device': 'GPU' if use_gpu else 'CPU',
        'method': method,
        'architecture': 'qelm_statevector',
        'window_size': window_size,
        'N': cfg.N,
        'T': T,
        'elapsed_s': elapsed,
        'steps_per_s': T / elapsed if elapsed > 0 else float('inf'),
        'circuit_depth': tqc.depth(),
        'circuit_size': tqc.size(),
        'n_features': len(labels),
    }
    return labels, X, info


# -----------------------------------------------------------------------------
# 2d. QELM real IBM hardware execution (USE_QPU)
# -----------------------------------------------------------------------------
# The recurrent architecture's hardware path (Section 2b) has to replay the
# ENTIRE input prefix u_1..u_t inside one circuit per timestep, because Aer's
# mid-circuit save_expectation_value has no real-hardware equivalent and a
# real QPU cannot persist a quantum state between circuit executions -- hence
# O(t) depth per step, O(T^2) total gates. The QELM architecture does NOT
# have that problem: every qubit is reset each step anyway (see
# `build_qelm_circuit`), so the state at step t depends ONLY on the current
# `window_size`-wide window, not on anything before it. Each hardware circuit
# below is therefore a single fresh encode-then-reservoir-layers circuit --
# CONSTANT depth in t, O(T) total gates across the whole trajectory, no
# prefix replay needed.

def build_qelm_hardware_circuits(cfg: ReservoirConfig, u_seq: Sequence[float], window_size: int = 8,
                                  max_weight: int = 3, reps: int = 1, g: float = 0.05):
    """One circuit per timestep t for the QELM variant on real hardware.
    Unlike `build_hardware_circuits` (recurrent architecture), no prefix
    replay is needed -- see the section note above. Same no-aliasing window
    mapping, full-Pauli-string readout (`feature_ops_all`, `max_weight`), and
    `reps`/`g` defaults as `build_qelm_circuit` -- see its docstring. Returns
    (circuits, observables, labels), same shape as `build_hardware_circuits`.

    Note: `max_weight=3`'s full Pauli-string basis means many more distinct
    observables per circuit than the earlier all-Z-only design (93-327 for
    N=4..10) -- `EstimatorV2` groups these into compatible measurement bases
    automatically, but real-hardware jobs cost more (more distinct bases to
    sample) than a smaller observable set would.
    """
    bias_z, bias_x = cfg.sample_disorder()
    labels, ops = feature_ops_all(cfg.N, max_weight=max_weight)
    observables = [_pauli_to_sparse_op(cfg.N, op, qargs) for (op, qargs) in ops]
    eff_window = min(window_size, cfg.N)
    slot_to_qubit = list(range(eff_window))

    circuits = []
    for t in range(len(u_seq)):
        lo = max(0, t - eff_window + 1)
        wlen = t - lo + 1
        window = np.zeros(eff_window)
        window[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(cfg.N)
        for w, uu in enumerate(window):
            per_q[slot_to_qubit[w]] += np.pi * float(uu)

        qc = QuantumCircuit(cfg.N)
        for i in range(cfg.N):
            qc.ry(per_q[i], i)
        for _ in range(reps):
            reservoir_layer(qc, cfg.N, g, bias_z, bias_x)
        circuits.append(qc)
    return circuits, observables, labels


def run_reservoir_qelm_qpu(cfg: ReservoirConfig, u_seq: Sequence[float], window_size: int = 8,
                            max_weight: int = 3, reps: int = 1, g: float = 0.05,
                            backend_name: str = DEFAULT_QPU_BACKEND, service=None,
                            shots: int = 4096, optimization_level: int = 3,
                            max_circuits: int = 200):
    """Run the QELM trajectory on a real IBM QPU (default `ibm_marrakesh`,
    pinned by name, per comment 9 -- same convention as `run_reservoir_qpu`).

    `max_circuits` is still a safety cap (this submits one job with one
    circuit per timestep), but -- unlike the recurrent path's O(T^2) gate
    growth -- circuit depth here is CONSTANT in t (see the section note
    above), so a much larger T is reasonable at the same hardware cost.

    Returns (labels, X, info); `info` reports backend/job provenance and
    per-circuit depth/two-qubit-gate counts, same shape as
    `run_reservoir_qpu`'s `info` (comment 9-11's spirit: report real
    resources), plus `architecture`/`window_size` like `run_reservoir_qelm`.
    """
    if len(u_seq) > max_circuits:
        raise ValueError(
            f"len(u_seq)={len(u_seq)} exceeds max_circuits={max_circuits}. This cap exists "
            "so a long T isn't submitted to paid/rate-limited hardware by accident. Pass a "
            "larger max_circuits explicitly if you intend that."
        )

    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = service or get_ibm_service()
    backend = service.backend(backend_name)

    circuits, observables, labels = build_qelm_hardware_circuits(cfg, u_seq, window_size=window_size,
                                                                   max_weight=max_weight, reps=reps, g=g)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuits = [pm.run(qc) for qc in circuits]
    isa_observables = [[obs.apply_layout(ic.layout) for obs in observables] for ic in isa_circuits]
    pubs = list(zip(isa_circuits, isa_observables))

    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = shots

    t0 = time.perf_counter()
    job = estimator.run(pubs)
    result = job.result()
    elapsed = time.perf_counter() - t0

    X = np.array([np.real(pub_result.data.evs) for pub_result in result])

    depths = [ic.depth() for ic in isa_circuits]
    twoq_counts = [sum(v for k, v in ic.count_ops().items() if k in ('cz', 'ecr', 'cx', 'rzz'))
                   for ic in isa_circuits]
    info = {
        'device': 'QPU', 'backend': backend.name,
        'backend_num_qubits': backend.num_qubits,
        'job_id': job.job_id(),
        'architecture': 'qelm_statevector', 'window_size': window_size,
        'N': cfg.N, 'T': len(u_seq), 'shots': shots,
        'elapsed_s': elapsed,
        'steps_per_s': len(u_seq) / elapsed if elapsed > 0 else float('inf'),
        'circuit_depth_per_step': depths,
        'circuit_depth_final': depths[-1] if depths else 0,
        'two_qubit_gates_per_step': twoq_counts,
        'two_qubit_gates_total': int(sum(twoq_counts)),
        'n_features': len(labels),
    }
    return labels, X, info


def run_benchmark_qelm(cfg: ReservoirConfig, T: int = 600, washout: int = 50,
                        n_val: int = 100, n_test: int = 150, k_list=(1, 2, 3),
                        window_size: int = 8, max_weight: int = 3, reps: int = 1, g: float = 0.05,
                        use_gpu: bool = False, use_qpu: bool = False,
                        method: str = 'statevector', input_seed: int = 0, **qpu_kwargs) -> BenchmarkResult:
    """`run_benchmark`, using the memoryless QELM variant (`run_reservoir_qelm`)
    instead of the genuinely recurrent reservoir -- same tasks, split, and
    alpha selection, so `BenchmarkResult`s from this and `run_benchmark` are
    directly comparable at the same N.

    `max_weight`/`reps`/`g` default to the values `run_reservoir_qelm`
    documents as giving robustly low k-Pauli NRMSE (see its docstring);
    `max_weight` should cover the largest k in `k_list`.

    `use_qpu=True` runs the QELM trajectory on real IBM hardware via
    `run_reservoir_qelm_qpu` instead of Aer; `**qpu_kwargs` (backend_name,
    service, shots, optimization_level, max_circuits) are forwarded to it.
    """
    reservoir_fn = functools.partial(run_reservoir_qelm, window_size=window_size,
                                      max_weight=max_weight, reps=reps, g=g,
                                      use_qpu=use_qpu, **qpu_kwargs)
    return run_benchmark(cfg, T=T, washout=washout, n_val=n_val, n_test=n_test,
                          k_list=k_list, use_gpu=use_gpu, method=method,
                          input_seed=input_seed, reservoir_fn=reservoir_fn)


# =============================================================================
# SCALING NOTES -- where and how to increase the qubit count
# =============================================================================
# The single knob is `ReservoirConfig.N`. What that costs you:
#
#   Method 'density_matrix' (used here, required for the reset-based fading
#   memory in comment 1's fix): a dense N-qubit density matrix has 4**N
#   complex128 entries = 4**N * 16 bytes, and Aer needs roughly 2x that live
#   during a reset/expectation-value pass.
#
#       N        density matrix        rough ceiling
#       ------------------------------------------------
#       8        1 MB                  trivial on CPU
#       10       16 MB                 trivial on CPU
#       12       268 MB                fine on CPU; fine on a 6 GB GPU
#       13       1.1 GB                fine on CPU; tight on a 6 GB GPU
#       14       4.3 GB                slow on CPU; likely OOM on a 6 GB GPU
#       16       68 GB                 not feasible on a laptop either device
#
#   So on a laptop-class GPU (e.g. an RTX 4050 with 6 GB VRAM) N ~ 12-13 is
#   the practical ceiling for this exact, fully-recurrent architecture; a
#   workstation CPU with more RAM can push N a bit further before it gets too
#   slow to be useful interactively.
#
#   If you need larger N than that, you have to give something up:
#
#   1. Drop the fading-memory reset (i.e. give up genuine step-to-step
#      recurrence and go back to a memoryless, sliding-window-encoded feature
#      map / QELM). Without a reset the state stays pure, so you can use
#      method='statevector' (2**N instead of 4**N) -- roughly DOUBLING the
#      reachable N for the same memory budget. This is the
#      `reservoir_states_qiskit` architecture in code/idcpsr_qiskit.py,
#      reimplemented as an Aer circuit in Section 6 above
#      (`build_qelm_circuit` / `run_reservoir_qelm` / `run_benchmark_qelm`);
#      the code review's point (comment 1) is exactly that this variant is a
#      feature map, not a memory, so treat any "memory capacity" number from
#      it with that caveat.
#
#   2. Use method='matrix_product_state' in `make_simulator`/`run_reservoir`.
#      Aer's MPS simulator scales polynomially (not exponentially) in N for a
#      1D chain topology like the one used here, at the cost of a bond-
#      dimension truncation -- i.e. it becomes an *approximate* simulation
#      once entanglement across the chain exceeds the bond dimension. This is
#      the natural next step for N in the high teens/twenties on this
#      topology; it needs no other code changes here beyond the `method=`
#      argument (density_matrix-only instructions like mid-circuit
#      Pauli-expectation snapshots are supported by the MPS method too).
#
#   3. Move off a laptop GPU entirely: qiskit-aer's GPU device also supports
#      NVIDIA's cuStateVec/cuTensorNet backends and multi-GPU statevector
#      distribution (`AerSimulator(method='statevector', device='GPU',
#      cuStateVec_enable=True, blocking_enable=True, blocking_qubits=...)`)
#      on a machine with more/larger GPUs -- relevant once you are no longer
#      VRAM-bound on a single 6 GB card.
