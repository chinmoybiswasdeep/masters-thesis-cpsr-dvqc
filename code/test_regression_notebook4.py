"""
test_regression_notebook4.py -- proves `mixed_syk_core.py` (the module
`5_QR_MixedSYK_JerbiShadow_Qiskit.ipynb` imports) is numerically IDENTICAL to
the mixed-SYK code actually living in `4_QR_MixedSYK_Qiskit.ipynb`, by loading
that notebook's own cell source at runtime (JSON parse + exec into a fresh
namespace -- no hand-copying involved in this comparison) and checking that
both produce bit-identical unitaries, feature vectors and (g, J) values for
matched configurations and seeds.

This is the guard requested by the project brief: "Add regression tests
proving that the unitary used by the shadow architecture is exactly the same
reservoir unitary used by the original mixed-SYK QELM for an identical
configuration."

Run directly:  python test_regression_notebook4.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types

import numpy as np

import mixed_syk_core as msc

HERE = os.path.dirname(os.path.abspath(__file__))
NB4_PATH = os.path.join(HERE, '4_QR_MixedSYK_Qiskit.ipynb')

# Cells (by index) that define, in order: imports, the qrc_qiskit reservoir
# module (Section 0a, verbatim), the benchmark tasks (Section 0b, verbatim),
# the mixed-layer core (Section 0c), the QELM wiring (Section 0d), and the
# Section-1 configuration constants (N_MIX, G_MAX, J_MAX, ...).
NEEDED_CELLS = [5, 7, 9, 11, 13, 15, 17]


def _load_notebook4_namespace() -> dict:
    """Exec notebook 4's own cell source into a REAL (sys.modules-registered)
    module namespace -- needed because those cells define `@dataclass`
    classes, and `dataclasses` resolves forward-reference type hints via
    `sys.modules[cls.__module__]`, which fails for a bare exec() dict."""
    with open(NB4_PATH, encoding='utf-8') as f:
        nb = json.load(f)
    mod = types.ModuleType('nb4_regression')
    mod.__dict__.update({'USE_GPU': False, 'USE_QPU': False})
    sys.modules['nb4_regression'] = mod
    for i in NEEDED_CELLS:
        src = ''.join(nb['cells'][i]['source'])
        exec(compile(src, f'<nb4-cell-{i}>', 'exec'), mod.__dict__)
    return mod.__dict__


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def run_all(verbose: bool = True) -> list:
    results = []

    def check(name, ok, detail=''):
        results.append((name, ok))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    ns = _load_notebook4_namespace()

    # 1. kappa_to_gJ must agree exactly (this caught the linear-vs-rational
    #    convention mismatch between the project brief's assumed formula and
    #    the notebook's actual one -- see mixed_syk_core.kappa_to_gJ's docstring).
    for kappa in (0.0, 0.02, 0.3, 1.0, 3.7, 100.0):
        g_nb, J_nb = ns['kappa_to_gJ'](kappa, 1.3, 0.7)
        g_mod, J_mod = msc.kappa_to_gJ(kappa, 1.3, 0.7)
        ok = (g_nb == g_mod) and (J_nb == J_mod)
        check(f'kappa_to_gJ(kappa={kappa}) identical', ok, f'nb=({g_nb},{J_nb}) mod=({g_mod},{J_mod})')

    # 2. sample_syk4_terms / couplings / pauli_types must be bit-identical
    #    (these feed the RNG state that everything downstream depends on).
    for N, n_terms, seed in [(4, 3, 1), (6, 11, 42)]:
        t_nb = ns['sample_syk4_terms'](N, n_terms, seed)
        t_mod = msc.sample_syk4_terms(N, n_terms, seed)
        check(f'sample_syk4_terms(N={N}) identical', t_nb == t_mod)
        c_nb = ns['sample_syk4_couplings'](n_terms, 0.6, seed)
        c_mod = msc.sample_syk4_couplings(n_terms, 0.6, seed)
        check(f'sample_syk4_couplings(N={N}) identical', np.array_equal(c_nb, c_mod))
        p_nb = ns['sample_syk4_pauli_types'](n_terms, seed)
        p_mod = msc.sample_syk4_pauli_types(n_terms, seed)
        check(f'sample_syk4_pauli_types(N={N}) identical', np.array_equal(p_nb, p_mod))

    # 3. single_layer_unitary_mixed: the actual reservoir unitary, for several
    #    (N, g, J, reps, seed) configurations spanning pure-SYK4, pure-SYK2,
    #    and mixed regimes -- bit-identical unitary + SHA-256 hash match.
    configs = [
        (4, 0.6, 0.5, 1, 4, 7),
        (6, 0.4, 0.8, 2, 11, 3),
        (6, 0.9, 0.0, 3, 11, 5),
        (6, 0.0, 0.7, 2, 11, 9),
    ]
    for N, g, J, reps, n_terms, seed in configs:
        cfg = ns['ReservoirConfig'](N=N, g=0.0, reps=reps, seed=seed)
        bias_z, _ = cfg.sample_disorder()
        terms = ns['sample_syk4_terms'](N, n_terms, seed)
        couplings = ns['sample_syk4_couplings'](n_terms, J, seed)
        paulis = ns['sample_syk4_pauli_types'](n_terms, seed)

        U_nb = ns['single_layer_unitary_mixed'](N, g, terms, couplings, paulis, bias_z)
        U_mod = msc.single_layer_unitary_mixed(N, g, terms, couplings, paulis, bias_z)
        dev = float(np.max(np.abs(U_nb - U_mod)))
        ok = dev < 1e-12 and _sha(U_nb) == _sha(U_mod)
        check(f'single_layer_unitary_mixed identical (N={N},g={g},J={J},seed={seed})',
              ok, f'max|dev|={dev:.2e}, sha256 match={_sha(U_nb) == _sha(U_mod)}')

        Ur_nb = ns['step_unitary_mixed'](N, g, terms, couplings, paulis, reps, bias_z)
        Ur_mod = msc.step_unitary_mixed(N, g, terms, couplings, paulis, reps, bias_z)
        dev_r = float(np.max(np.abs(Ur_nb - Ur_mod)))
        check(f'step_unitary_mixed (reps={reps}) identical (N={N})', dev_r < 1e-12, f'max|dev|={dev_r:.2e}')

    # 4. Full QELM circuit feature output: build_qelm_circuit_mixed /
    #    run_reservoir_qelm_mixed must give bit-identical (labels, X) for an
    #    identical config -- this is the actual object the shadow notebook's
    #    "direct QELM" reference implementation and Choi-flip both build on.
    # NOTE: method='density_matrix' is used deliberately here, NOT the
    # function's own default ('statevector') -- see
    # `test_aer_statevector_reset_bug.py` for a documented, reproducible
    # finding that AerSimulator(method='statevector') gives WRONG (non-
    # reproducible, seed_simulator-dependent) mid-circuit
    # save_expectation_value results for LATER timesteps of a trajectory
    # circuit containing many `reset` instructions (as build_qelm_circuit_mixed
    # /build_qelm_circuit produce). method='density_matrix' was verified
    # exact (machine precision) against an independent single-window
    # Statevector reference at every timestep and is used throughout this
    # regression test and the Jerbi-shadow notebook for that reason.
    cfg = ns['ReservoirConfig'](N=4, g=0.0, reps=1, seed=3)
    u = ns['random_input'](6, seed=11)
    g, J = ns['kappa_to_gJ'](1.0, 0.8, 0.6)
    labels_nb, X_nb, _ = ns['run_reservoir_qelm_mixed'](cfg, u, g=g, J=J, window_size=4,
                                                          max_weight=3, reps=1, n_terms=3, term_seed=1,
                                                          method='density_matrix')
    labels_mod, X_mod, _ = msc.run_reservoir_qelm_mixed(cfg, u, g=g, J=J, window_size=4,
                                                          max_weight=3, reps=1, n_terms=3, term_seed=1,
                                                          method='density_matrix')
    ok = (labels_nb == labels_mod) and np.allclose(X_nb, X_mod, atol=1e-12)
    check('run_reservoir_qelm_mixed features identical to notebook 4 (method=density_matrix)',
          ok, f'max|dev|={float(np.max(np.abs(X_nb - X_mod))):.2e}')

    # 5. feature_ops_all_general must match label-for-label (order matters:
    #    the shadow notebook indexes readout operators positionally).
    labs_nb, ops_nb = ns['feature_ops_all_general'](6, max_weight=5)
    labs_mod, ops_mod = msc.feature_ops_all_general(6, max_weight=5)
    check('feature_ops_all_general(N=6,max_weight=5) identical label order',
          labs_nb == labs_mod, f'{len(labs_nb)} vs {len(labs_mod)} labels')

    # 6. Section-1 configuration constants (N_MIX, G_MAX, J_MAX, ...) actually
    #    used by notebook 4's headline sweeps, for reference/documentation in
    #    the shadow notebook.
    check('N_MIX/G_MAX/J_MAX/N_SPARSE_DEFAULT/RESERVOIR_SEED loaded from notebook 4',
          all(k in ns for k in ('N_MIX', 'G_MAX', 'J_MAX', 'N_SPARSE_DEFAULT', 'RESERVOIR_SEED',
                                 'G_MAX_QELM', 'J_MAX_QELM', 'MAX_WEIGHT_QELM', 'WINDOW_SIZE_QELM',
                                 'REPS_QELM')))

    n_fail = sum(1 for _, ok in results if not ok)
    print(f'\nnotebook-4 <-> mixed_syk_core regression: {len(results) - n_fail}/{len(results)} passed.')
    if n_fail:
        raise AssertionError(f'{n_fail} regression check(s) FAILED -- mixed_syk_core.py has drifted '
                              f'from 4_QR_MixedSYK_Qiskit.ipynb.')
    return results, ns


if __name__ == '__main__':
    run_all()
