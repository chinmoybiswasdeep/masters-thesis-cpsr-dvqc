"""
test_aer_statevector_reset_bug.py -- documents a reproducible correctness bug
found while building the Jerbi-flipped-shadow notebook: `AerSimulator(method=
'statevector')` gives WRONG, `seed_simulator`-dependent `save_expectation_value`
results for LATER timesteps of a long trajectory circuit built by
`build_qelm_circuit_mixed` / `build_qelm_circuit` (every qubit `reset` each
step, memoryless QELM encoding).

Why this matters: `run_reservoir_qelm_mixed` / `run_reservoir_qelm` default to
`method='statevector'` (that choice is the whole point of Section 0a's SCALING
NOTES -- `statevector` costs `2**N` instead of `density_matrix`'s `4**N`,
since every qubit is reset each step so the state should stay pure). This bug
means that default is UNSAFE for a many-timestep trajectory: some fraction of
`AerSimulator` calls (varying only in the internal `seed_simulator`, with an
IDENTICAL transpiled circuit) return a value that disagrees with an
independent, exact `Statevector`-based single-window reference by up to
~0.3 (not floating-point noise) at timesteps 2 timesteps.

`method='density_matrix'` was checked against the same independent reference
and agrees to machine precision (~1e-15) at every timestep. This is why every
"direct QELM reference" computation in `5_QR_MixedSYK_JerbiShadow_Qiskit.ipynb`
and `jerbi_shadow.py` either (a) explicitly passes `method='density_matrix'`
when reusing `run_reservoir_qelm_mixed`'s trajectory path, or (b) uses a
from-scratch per-window `Statevector` evaluation (no `reset`, no Aer, exact by
construction) -- see `jerbi_shadow.direct_qelm_features`.

This finding is reported honestly rather than silently worked around, per the
project's own precedent (`docs/CPSR_Code_Review_Response.md` comment 4: a
previous, different simulation-fidelity bug in this same codebase). It does
NOT modify `4_QR_MixedSYK_Qiskit.ipynb` (out of scope / explicitly forbidden
by the project brief) -- it only documents the finding and demonstrates the
safe workaround used by the new notebook.

Run directly:  python test_aer_statevector_reset_bug.py
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import mixed_syk_core as msc


def _exact_features_at_t(t, u, N, window_size, g, terms, couplings, paulis, bias_z, ops_all):
    """Independent, exact reference: build ONE fresh (no-reset) circuit for
    the window ending at step t and read Pauli expectations off its exact
    Statevector. No Aer, no `reset`, no mid-circuit snapshot -- this is the
    ground truth every other path is checked against."""
    eff_window = min(window_size, N)
    lo = max(0, t - eff_window + 1)
    wlen = t - lo + 1
    window = np.zeros(eff_window)
    window[-wlen:] = u[lo:t + 1]
    per_q = np.zeros(N)
    for w, uu in enumerate(window):
        per_q[w] += np.pi * float(uu)
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.ry(per_q[i], i)
    msc.mixed_layer(qc, N, g, terms, couplings, paulis, bias_z)
    sv = Statevector.from_instruction(qc)
    return np.array([np.real(sv.expectation_value(op, qargs)) for op, qargs in ops_all])


def run_all(verbose: bool = True) -> list:
    results = []

    def check(name, ok, detail=''):
        results.append((name, ok))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    N, window_size, n_terms, term_seed, max_weight = 4, 4, 3, 1, 3
    cfg = msc.ReservoirConfig(N=N, g=0.0, reps=1, seed=3)
    u = msc.random_input(6, seed=11)
    g, J = msc.kappa_to_gJ(1.0, 0.8, 0.6)
    terms = msc.sample_syk4_terms(N, n_terms, term_seed)
    couplings = msc.sample_syk4_couplings(n_terms, J, term_seed)
    paulis = msc.sample_syk4_pauli_types(n_terms, term_seed)
    bias_z, _ = cfg.sample_disorder()
    labels_all, ops_all = msc.feature_ops_all_general(N, max_weight=max_weight)

    exact_X = np.array([
        _exact_features_at_t(t, u, N, window_size, g, terms, couplings, paulis, bias_z, ops_all)
        for t in range(len(u))
    ])

    # (a) method='density_matrix': must match the exact reference at EVERY
    #     timestep, to machine precision.
    _, X_dm, _ = msc.run_reservoir_qelm_mixed(cfg, u, g=g, J=J, window_size=window_size,
                                               max_weight=max_weight, reps=1, n_terms=n_terms,
                                               term_seed=term_seed, method='density_matrix')
    dev_dm = float(np.max(np.abs(X_dm - exact_X)))
    check("run_reservoir_qelm_mixed(method='density_matrix') matches exact single-window reference",
          dev_dm < 1e-9, f'max|dev| over all timesteps = {dev_dm:.2e}')

    # (b) method='statevector' (the function's DEFAULT): demonstrate that
    #     repeated calls with different `seed_simulator` values, on the
    #     IDENTICAL transpiled circuit, disagree with each other AND with the
    #     exact reference by an amount far exceeding floating-point noise.
    #     This is the bug -- reported, not silently patched.
    from qiskit import transpile
    qc, labels = msc.build_qelm_circuit_mixed(cfg, u, g=g, J=J, window_size=window_size,
                                               max_weight=max_weight, reps=1, n_terms=n_terms,
                                               term_seed=term_seed)
    sim = msc.make_simulator(use_gpu=False, method='statevector')
    tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=42)
    seed_runs = []
    for seed in range(15):
        result = sim.run(tqc, shots=1, seed_simulator=seed).result()
        data = result.data(0)
        X_run = np.empty((len(u), len(labels)))
        for t in range(len(u)):
            for j, lab in enumerate(labels):
                X_run[t, j] = np.real(data[f'{lab}__t{t}'])
        seed_runs.append(X_run)
    devs_vs_exact = [float(np.max(np.abs(x - exact_X))) for x in seed_runs]
    n_disagree = sum(1 for d in devs_vs_exact if d > 1e-6)
    spread_across_seeds = float(np.max(np.abs(np.array(seed_runs) - seed_runs[0])))
    check("KNOWN BUG documented: method='statevector' disagrees with the exact reference "
          "for >=1 of 15 seed_simulator values (same transpiled circuit)",
          n_disagree >= 1,
          f'{n_disagree}/15 seeds disagree with exact by >1e-6 (max dev {max(devs_vs_exact):.3f}); '
          f'spread across seeds for the IDENTICAL circuit = {spread_across_seeds:.3f}')
    print(f"\n  Per-seed max|dev| vs exact reference (method='statevector', same transpiled circuit,\n"
          f"  varying only seed_simulator): {[f'{d:.4f}' for d in devs_vs_exact]}")
    print("  --> AerSimulator(method='statevector') is UNSAFE as a ground truth for a trajectory\n"
          "      circuit with many mid-circuit `reset` instructions before a `save_expectation_value`;\n"
          "      this notebook/module always uses method='density_matrix' or a from-scratch\n"
          "      per-window Statevector evaluation instead (see jerbi_shadow.direct_qelm_features).\n")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f'Aer statevector-reset bug diagnostic: {len(results) - n_fail}/{len(results)} checks passed.')
    if n_fail:
        raise AssertionError(f'{n_fail} diagnostic check(s) FAILED.')
    return results


if __name__ == '__main__':
    run_all()
