"""
eoc_config.py -- reservoir configuration, the SYK4 support-count fix, and a
reservoir fingerprint, for the Jerbi-flipped Choi-shadow QELM
(`jerbi_shadow.py` / `5_QR_MixedSYK_JerbiShadow_Qiskit.ipynb`).

`4_QR_MixedSYK_Qiskit.ipynb` is the SOURCE OF TRUTH for the mixed-SYK
reservoir and its established EOC operating point. Nothing here redefines
`mixed_layer`, `kappa_to_gJ`, or any sampling function -- it only (a) fixes a
latent support-count footgun when calling those functions, and (b) loads
notebook 4's own already-computed EOC configuration rather than re-deriving
it inside this project.

Two configurations are exposed, deliberately kept apart so results from one
are never mistaken for the other:

  - `build_validation_config()`  -- a SMALL system (N=4) used ONLY to check
    that the Choi-flip math, tensor ordering, shadow estimator, serialization
    and no-QPU-inference machinery are correct. Its kappa/(g,J) values carry
    NO physical EOC meaning. N=4 is chosen deliberately because
    C(4,4)=1 < default_n_sparse_terms(4)=6, which EXERCISES the support-count
    fix below on every run (see `sample_syk4_supports`) -- this is a feature
    of the choice, not an oversight.

  - `build_science_config()`     -- notebook 4's OWN established mixed-SYK
    EOC-QELM operating point: N=6, G_MAX_QELM=J_MAX_QELM=0.6, MAX_WEIGHT_QELM=5,
    WINDOW_SIZE_QELM=6, REPS_QELM=1, kappa=0.960 (the NARMA2-minimizing kappa
    from notebook 4's own `qelm_scan_mixed`, Section 6 -- see that function's
    stored, already-executed output, quoted verbatim in
    `NOTEBOOK4_QELM_EOC_KAPPA`'s docstring below). This notebook NEVER re-runs
    that expensive (~30 minute) scan or re-derives kappa from its own
    experiments -- doing so from data used later for benchmarking would be
    exactly the "EOC tuned on test performance" leakage the project's review
    history warns against.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

import mixed_syk_core as msc

# =============================================================================
# The SYK4 support-count fix.
# =============================================================================
# `sample_syk4_terms(N, n_terms, seed)` (notebook 4, unchanged, reused
# verbatim via `mixed_syk_core`) CAPS its return at C(N,4) possible 4-qubit
# supports: `if n_terms >= len(all_tuples): return all_tuples`. Notebook 4's
# OWN operating point (N=6, default_n_sparse_terms(6)=11 <= C(6,4)=15) never
# triggers this cap, so it was never exercised there. But
# `default_n_sparse_terms(4) = ceil(4*ln4) = 6 > C(4,4) = 1` DOES trigger it,
# and every caller in the codebase (notebook 4's OWN
# `build_qelm_circuit_mixed`/`build_trajectory_circuit_mixed` included) then
# calls `sample_syk4_couplings(n_terms, ...)` / `sample_syk4_pauli_types(n_terms,
# ...)` with the REQUESTED `n_terms`, not the ACTUAL (possibly smaller)
# `len(terms)`. `mixed_layer`'s `zip(terms, couplings, paulis)` then silently
# truncates to the shortest array -- for NumPy's default `RandomState`, the
# discarded tail happens not to change the VALUES that survive truncation
# (draws are sequential and independent of the requested array length), so
# this has NOT been silently corrupting notebook 4's own N=6 results -- but
# it is a latent bug that WOULD corrupt results the moment the same pattern
# is used at any N where `default_n_sparse_terms(N) > comb(N,4)` (as this
# project's own small-system validation config does, at N=4), and relying on
# `RandomState` internals to accidentally save it is not something to trust
# going forward. This module fixes it at every call site by always deriving
# `actual_terms = len(terms)` FIRST and sampling couplings/Pauli types with
# that, never the originally-requested count -- exactly the pattern the
# project brief specifies -- and asserts the three arrays' lengths match.
# `mixed_syk_core.py` / notebook 4 are NOT modified (doing so would break the
# byte-identical regression guarantee in `test_regression_notebook4.py`).


def sample_syk4_supports(N: int, n_terms_requested: int, term_seed: int, J: float) -> dict:
    """The fixed replacement for the `terms = sample_syk4_terms(...); couplings
    = sample_syk4_couplings(n_terms_requested, ...)` pattern. Returns a dict
    with `requested_terms`, `actual_terms`, `terms`, `couplings`, `paulis` --
    couplings/paulis are ALWAYS sized to `actual_terms = len(terms)`, never to
    the requested count. Asserts all three arrays agree in length."""
    terms = msc.sample_syk4_terms(N, n_terms_requested, term_seed)
    actual_terms = len(terms)
    couplings = msc.sample_syk4_couplings(actual_terms, J=J, seed=term_seed)
    paulis = msc.sample_syk4_pauli_types(actual_terms, seed=term_seed)
    assert len(terms) == len(couplings) == len(paulis) == actual_terms, (
        f'SYK4 support arrays disagree in length: terms={len(terms)}, '
        f'couplings={len(couplings)}, paulis={len(paulis)} (expected {actual_terms})')
    return {
        'requested_terms': int(n_terms_requested),
        'actual_terms': int(actual_terms),
        'terms': terms,
        'couplings': couplings,
        'paulis': paulis,
        'was_capped': bool(actual_terms < n_terms_requested),
    }


# =============================================================================
# Reservoir parameters + fingerprint
# =============================================================================

@dataclass
class ReservoirParams:
    """Everything that defines the fixed mixed-SYK EOC channel U_EOC and the
    QELM encoding/readout built on it. Immutable in spirit -- nothing in this
    project mutates a `ReservoirParams` after construction; `fingerprint()`
    lets code ASSERT that, rather than merely intend it."""
    config_name: str
    N: int
    kappa: float
    G_MAX: float
    J_MAX: float
    g: float
    J: float
    reps: int
    window_size: int
    max_weight: int
    term_seed: int
    disorder_seed: int
    requested_terms: int
    actual_terms: int
    terms: list
    couplings: np.ndarray
    paulis: np.ndarray
    bias_z: np.ndarray
    labels: list = field(default_factory=list)
    notes: str = ''

    @property
    def d(self) -> int:
        return 2 ** self.N


def _canonical_bytes(params: ReservoirParams, ops_labels: Sequence[str]) -> bytes:
    payload = {
        'N': params.N, 'kappa': params.kappa, 'G_MAX': params.G_MAX, 'J_MAX': params.J_MAX,
        'g': round(params.g, 12), 'J': round(params.J, 12), 'reps': params.reps,
        'window_size': params.window_size, 'max_weight': params.max_weight,
        'term_seed': params.term_seed, 'disorder_seed': params.disorder_seed,
        'actual_terms': params.actual_terms,
        'terms': [list(t) for t in params.terms],
        'couplings': [round(float(c), 12) for c in params.couplings],
        'paulis': [list(map(str, p)) for p in params.paulis],
        'bias_z': [round(float(b), 12) for b in params.bias_z],
        'ops_labels': list(ops_labels),
    }
    return json.dumps(payload, sort_keys=True).encode('utf-8')


def fingerprint(params: ReservoirParams, ops_labels: Sequence[str]) -> str:
    """SHA-256 hex digest of every number that defines U_EOC, the QELM
    encoding convention, and the readout observable set. Two `ReservoirParams`
    (+ observable lists) with the same fingerprint define EXACTLY the same
    physics -- not just 'close' or 'same seed'. Used to assert the reservoir
    is untouched by classical-readout training (Section 5 of the audit)."""
    return hashlib.sha256(_canonical_bytes(params, ops_labels)).hexdigest()


def build_reservoir_params(config_name: str, N: int, kappa: float, G_MAX: float, J_MAX: float,
                            reps: int, window_size: int, max_weight: int, term_seed: int,
                            disorder_seed: int, notes: str = '') -> ReservoirParams:
    """Build a `ReservoirParams` using notebook 4's OWN functions
    (`kappa_to_gJ`, `default_n_sparse_terms`, `ReservoirConfig.sample_disorder`)
    plus the support-count fix above. Does not alter any notebook-4 numerical
    convention."""
    g, J = msc.kappa_to_gJ(float(kappa), G_MAX, J_MAX)
    requested_terms = msc.default_n_sparse_terms(N)
    supports = sample_syk4_supports(N, requested_terms, term_seed, J)
    cfg = msc.ReservoirConfig(N=N, g=0.0, reps=reps, seed=disorder_seed)
    bias_z, _ = cfg.sample_disorder()
    return ReservoirParams(
        config_name=config_name, N=N, kappa=float(kappa), G_MAX=G_MAX, J_MAX=J_MAX, g=g, J=J,
        reps=reps, window_size=window_size, max_weight=max_weight, term_seed=term_seed,
        disorder_seed=disorder_seed, requested_terms=supports['requested_terms'],
        actual_terms=supports['actual_terms'], terms=supports['terms'],
        couplings=supports['couplings'], paulis=supports['paulis'], bias_z=bias_z, notes=notes,
    )


# =============================================================================
# VALIDATION_CONFIG -- small-system numerical validation ONLY.
# =============================================================================

def build_validation_config(N: int = 4, kappa: float = 1.0, reps: int = 2, window_size: int = None,
                             max_weight: int = 3, term_seed: int = 17, disorder_seed: int = 23,
                             G_MAX: float = 0.6, J_MAX: float = 0.6) -> ReservoirParams:
    """Small-system config for correctness checks (Choi identity, tensor
    ordering, shadow estimator, serialization, no-QPU inference). NOT the EOC
    physics point -- `kappa` here is an arbitrary convenient value, not
    anything notebook 4 established. At the default N=4, `comb(4,4)=1` while
    `default_n_sparse_terms(4)=6`, so `actual_terms` will be 1, not 6 --
    exercising the support-count fix (`sample_syk4_supports`) on every run."""
    window_size = window_size if window_size is not None else N
    return build_reservoir_params(
        config_name='VALIDATION_CONFIG (small-system numerical validation -- NOT the EOC physics point)',
        N=N, kappa=kappa, G_MAX=G_MAX, J_MAX=J_MAX, reps=reps, window_size=window_size,
        max_weight=max_weight, term_seed=term_seed, disorder_seed=disorder_seed,
    )


# =============================================================================
# SCIENCE_CONFIG -- notebook 4's own established mixed-SYK EOC-QELM point.
# =============================================================================

# Notebook 4 (`4_QR_MixedSYK_Qiskit.ipynb`, Section 1) fixes, for the QELM
# (memoryless, Section 0d) architecture:
N_MIX = 6
G_MAX_QELM = 0.6
J_MAX_QELM = 0.6
MAX_WEIGHT_QELM = 5
WINDOW_SIZE_QELM = N_MIX
REPS_QELM = 1
RESERVOIR_SEED = 42

# Notebook 4, Section 6 (`qelm_scan_mixed`, KAPPA_GRID = geomspace(0.02,100,12),
# N_REAL_QELM=3 realizations per kappa, averaging term/disorder seeds 0,1,2)
# established its OWN EOC operating point as the kappa minimizing NARMA2 NRMSE
# on that scan. That notebook's STORED, ALREADY-EXECUTED cell output (Section
# 8's demo cell) states this explicitly and verbatim:
#
#   "QELM temporal-edge scan fixed at the Section 6 optimum: kappa=0.960
#    (g=0.2939, J=0.3061)"
#
# (from `qelm_perf['kappa'][qelm_narma_best_idx]`, `qelm_narma_best_idx =
# argmin(qelm_perf['narma2_mean'])`; the printed qelm_perf table shows NARMA2
# = 0.5428 at kappa=0.960, the minimum of the 12-point scan). This value is
# LOADED here, not re-derived -- re-running that ~30-minute, 12 x 3-realization
# scan inside THIS notebook, using data that later feeds this notebook's own
# benchmarks, would itself be the "EOC selected using the experiment's own
# data" leakage pattern the project's review history flags. A cheap,
# non-leaky consistency check (kappa_to_gJ(0.960, 0.6, 0.6) reproduces
# (0.2939, 0.3061)) is run wherever this module is imported alongside
# `mixed_syk_core` -- see `verify_science_kappa_gJ`.
NOTEBOOK4_QELM_EOC_KAPPA = 0.960


def verify_science_kappa_gJ(tol: float = 1e-3) -> tuple:
    """Cheap (no scan re-run) consistency check that `NOTEBOOK4_QELM_EOC_KAPPA`
    still reproduces notebook 4's printed (g, J) at that kappa, using notebook
    4's own `kappa_to_gJ` (reused verbatim via `mixed_syk_core`). Raises
    AssertionError if the printed record and the live function disagree
    (e.g. if `kappa_to_gJ` or `G_MAX_QELM`/`J_MAX_QELM` above ever drift from
    notebook 4)."""
    g, J = msc.kappa_to_gJ(NOTEBOOK4_QELM_EOC_KAPPA, G_MAX_QELM, J_MAX_QELM)
    g_expected, J_expected = 0.2939, 0.3061   # notebook 4's own printed values
    assert abs(g - g_expected) < tol and abs(J - J_expected) < tol, (
        f'kappa_to_gJ({NOTEBOOK4_QELM_EOC_KAPPA}, {G_MAX_QELM}, {J_MAX_QELM}) = ({g:.4f}, {J:.4f}) '
        f'no longer matches notebook 4\'s recorded (g,J)=({g_expected},{J_expected}) -- '
        f'NOTEBOOK4_QELM_EOC_KAPPA or kappa_to_gJ has drifted from notebook 4.')
    return g, J


def build_science_config(term_seed: int = 0, disorder_seed: int = RESERVOIR_SEED,
                          kappa: float = NOTEBOOK4_QELM_EOC_KAPPA) -> ReservoirParams:
    """Notebook 4's own established QELM EOC operating point (N=6, kappa=0.96).
    `term_seed=0` picks ONE of the three realizations (seeds 0,1,2) notebook
    4's own `qelm_scan_mixed` AVERAGED OVER at this kappa -- a single concrete,
    reproducible realization is needed here (this notebook does not re-run a
    3-seed ensemble average for every experiment below), and this is stated
    explicitly wherever SCIENCE_CONFIG results are reported."""
    verify_science_kappa_gJ()
    return build_reservoir_params(
        config_name=f'SCIENCE_CONFIG (notebook 4 Section 6 QELM EOC point: kappa={kappa})',
        N=N_MIX, kappa=kappa, G_MAX=G_MAX_QELM, J_MAX=J_MAX_QELM, reps=REPS_QELM,
        window_size=WINDOW_SIZE_QELM, max_weight=MAX_WEIGHT_QELM, term_seed=term_seed,
        disorder_seed=disorder_seed,
        notes='term_seed=0 is ONE of notebook 4\'s own 3 averaged realizations (seeds 0,1,2), not an average.',
    )
