"""
utils.py -- shared infrastructure for the DQRC package: run-mode config,
GPU/Aer device detection, seeded-RNG helpers that keep reservoir/dataset/
measurement randomness in separate streams (per this project's own
scientific-rigor standard -- see docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md), a
bootstrap-CI helper, and a disk cache for static unitaries/Pauli
operators/target functions (Part 19's caching requirement).

Nothing in this module talks to Qiskit's reservoir-building code -- it is
the one module every other decoupled_qrc module is allowed to depend on
without creating a cycle.
"""
from __future__ import annotations

import functools
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# =============================================================================
# Run-mode configuration
# =============================================================================
# FAST_MODE targets a single notebook run in well under ten minutes: few
# seeds, short trajectories, small parameter scans. PUBLICATION_MODE is the
# >=10-seed, full-task, full-scan follow-up pass (Part 13) -- NOT run by
# default, since it does not fit a FAST_MODE budget. Any DQRC_RESULTS.md
# claim produced under FAST_MODE must say so explicitly (Part 16 forbids
# treating a small-seed run as a general claim).

FAST_MODE = True
PUBLICATION_MODE = False

if FAST_MODE and PUBLICATION_MODE:
    raise ValueError("FAST_MODE and PUBLICATION_MODE are mutually exclusive.")


@dataclass(frozen=True)
class RunConfig:
    """One place all FAST_MODE/PUBLICATION_MODE-scaled constants live, so a
    notebook cell can print exactly what scale a given figure was produced
    at."""
    name: str
    n_seeds: int
    T_ipc: int          # trajectory length for IPC estimation
    T_task: int          # trajectory length for task benchmarks
    washout: int
    n_val: int
    n_test: int
    max_delay_ipc: int
    max_degree_ipc: int
    max_targets_per_degree: int
    n_surrogates: int
    n_shadow_snapshots: int
    g_scan_points: int
    n_bootstrap: int


FAST_CONFIG = RunConfig(
    # Every constant below was empirically timed (density_matrix Aer
    # simulation of the combined DQRC/monolithic circuits + compute_ipc's
    # ridge-fit cost both scale steeply with N/T/max_weight/max_targets)
    # against a <=10 minute full-notebook budget: N_total=6 (1 input + N_M=2
    # + N_P=3) at T=250/reps=1/max_weight=2 runs a reservoir in ~2s;
    # max_delay=6/max_degree=6/max_targets_per_degree=12/n_surrogates=5 runs
    # one compute_ipc call in ~2.5s -- at ~50 total (architecture, seed)
    # combinations across the notebook's sections this totals a few minutes,
    # leaving headroom for plotting/task-benchmark overhead.
    name="FAST_MODE", n_seeds=2, T_ipc=250, T_task=250, washout=40, n_val=50,
    n_test=70, max_delay_ipc=6, max_degree_ipc=6, max_targets_per_degree=12,
    n_surrogates=5, n_shadow_snapshots=2000, g_scan_points=4, n_bootstrap=300,
)

PUBLICATION_CONFIG = RunConfig(
    name="PUBLICATION_MODE", n_seeds=10, T_ipc=2000, T_task=2000, washout=100,
    n_val=200, n_test=400, max_delay_ipc=12, max_degree_ipc=6,
    max_targets_per_degree=80, n_surrogates=30, n_shadow_snapshots=20000,
    g_scan_points=15, n_bootstrap=2000,
)


def active_config() -> RunConfig:
    return PUBLICATION_CONFIG if PUBLICATION_MODE else FAST_CONFIG


# =============================================================================
# Device detection (mirrors qrc_qiskit.make_simulator's own probe, so DQRC
# never re-derives a different GPU-availability answer).
# =============================================================================

def gpu_available() -> bool:
    try:
        from qiskit_aer import AerSimulator
        return 'GPU' in AerSimulator().available_devices()
    except Exception:
        return False


# =============================================================================
# Seeded randomness -- separate streams per source, never one global seed.
# =============================================================================

@dataclass(frozen=True)
class SeedBundle:
    """Independent RNG streams for the three sources of randomness this
    project's scientific-rigor standard requires keeping separate: the
    reservoir's own disorder/SYK-term sampling, the dataset (input sequence),
    and the measurement process (shadow basis/outcome sampling)."""
    master_seed: int
    reservoir_seed: int
    dataset_seed: int
    measurement_seed: int

    def rng(self, which: str) -> np.random.RandomState:
        return np.random.RandomState(getattr(self, f"{which}_seed"))


def make_seed_bundle(master_seed: int) -> SeedBundle:
    """Derive three well-separated child seeds from one master seed via
    SHA-256, so sweeping `master_seed` in a seed loop varies all three
    streams together but deterministically and without collisions."""
    def child(tag: str) -> int:
        h = hashlib.sha256(f"dqrc:{master_seed}:{tag}".encode()).digest()
        return int.from_bytes(h[:4], "big")
    return SeedBundle(
        master_seed=master_seed,
        reservoir_seed=child("reservoir"),
        dataset_seed=child("dataset"),
        measurement_seed=child("measurement"),
    )


# =============================================================================
# Bootstrap confidence intervals (Part 13).
# =============================================================================

def bootstrap_ci(samples, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                  statistic: Callable = np.mean):
    """Percentile bootstrap CI for `statistic` over `samples`. Returns
    (point_estimate, ci_lo, ci_hi). With len(samples)==1 the CI collapses to
    a point (nothing to resample) -- callers must not treat that as a real
    interval; `n_seeds` in the active RunConfig should always be >1."""
    samples = np.asarray(samples, dtype=float)
    point = float(statistic(samples))
    if len(samples) <= 1:
        return point, point, point
    rng = np.random.RandomState(seed)
    n = len(samples)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = statistic(samples[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_diff_ci(samples_a, samples_b, n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 0, statistic: Callable = np.mean):
    """Bootstrap CI for statistic(a) - statistic(b), resampling each
    independently. A claimed superiority requires this CI to exclude 0 in the
    claimed direction (Part 13/16) -- never inferred from point estimates
    alone."""
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    point = float(statistic(a) - statistic(b))
    if len(a) <= 1 or len(b) <= 1:
        return point, point, point
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.randint(0, len(a), size=len(a))
        ib = rng.randint(0, len(b), size=len(b))
        diffs[i] = statistic(a[ia]) - statistic(b[ib])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# =============================================================================
# Disk cache for static unitaries / Pauli operators / target functions.
# =============================================================================

_CACHE_DIR = Path(os.environ.get("DQRC_CACHE_DIR", Path(__file__).resolve().parent.parent / "_dqrc_cache"))


def cached(namespace: str):
    """Decorator: cache a function's return value to disk keyed by its
    arguments' repr. Intended for expensive, purely-deterministic builders
    (unitaries, Pauli-operator lists, IPC target arrays) that are called
    repeatedly across a parameter scan with the same arguments -- NOT for
    anything involving live randomness beyond what's captured in the args
    (always pass seeds explicitly, never rely on global RNG state)."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            key_src = repr((fn.__module__, fn.__qualname__, args, sorted(kwargs.items())))
            key = hashlib.sha256(key_src.encode()).hexdigest()
            path = _CACHE_DIR / namespace / f"{key}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
            result = fn(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(result, f)
            return result
        wrapper.cache_dir = _CACHE_DIR / namespace
        return wrapper
    return decorator


def clear_cache():
    import shutil
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)


# =============================================================================
# Resource accounting record (shared shape used by metrics.py / experiments.py)
# =============================================================================

@dataclass
class ResourceUsage:
    """Honest per-run resource accounting (Part 9): physical qubits actually
    simulated, ancillas among them, transpiled circuit depth, two-qubit gate
    count, measurement shots (0 for exact-readout runs), number of classical
    readout features, and a rough classical-preprocessing cost proxy (ridge
    fits performed)."""
    n_qubits_physical: int
    n_ancilla: int
    circuit_depth: int
    two_qubit_gates: int
    shots: int
    n_features: int
    n_ridge_fits: int = 0
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "n_qubits_physical": self.n_qubits_physical,
            "n_ancilla": self.n_ancilla,
            "circuit_depth": self.circuit_depth,
            "two_qubit_gates": self.two_qubit_gates,
            "shots": self.shots,
            "n_features": self.n_features,
            "n_ridge_fits": self.n_ridge_fits,
            "notes": self.notes,
        }


def repo_code_dir() -> str:
    """Absolute path to `code/`, so notebook/module code can `sys.path.append`
    it regardless of current working directory (mirrors how the repo's own
    notebooks import `qrc_qiskit`/`mixed_syk_core`)."""
    return str(Path(__file__).resolve().parent.parent)


def ensure_repo_code_on_path():
    p = repo_code_dir()
    if p not in sys.path:
        sys.path.insert(0, p)
