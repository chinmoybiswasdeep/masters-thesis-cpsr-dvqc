"""Tests for notebook 5's modules (klqrc_quantum / klqrc_classical / klqrc_pipeline).

nb4's physics/circuit unit tests are ported first; then the new invariants:
exact chunked continuation, exact gate fusion, Aer readout equivalence,
causality, train-only lifts, the Volterra lift, capacity and bootstrap
sanity, the test vault and the outcome mapping.
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import klqrc_classical as C  # noqa: E402
import klqrc_pipeline as P  # noqa: E402
import klqrc_quantum as Q  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

TOL = 1e-9


# --------------------------------------------------------------------------- nb4 ports
def test_zzzz_gadget_is_exact_diagonal_phase():
    N, qs, theta = 5, (0, 1, 3, 4), 0.73
    qc = QuantumCircuit(N)
    Q.zzzz_rotation(qc, qs, theta)
    idx = np.arange(2 ** N)
    signs = 1 - 2 * ((idx[:, None] >> np.arange(N)) & 1)
    phase = np.exp(-1j * theta / 2 * np.prod(signs[:, list(qs)], axis=1))
    assert np.max(np.abs(Operator(qc).data - np.diag(phase))) < TOL


def test_pauli4_all_z_equals_zzzz_and_mixed_matches_expm():
    from scipy.linalg import expm
    qc1, qc2 = QuantumCircuit(5), QuantumCircuit(5)
    Q.zzzz_rotation(qc1, (0, 1, 3, 4), 0.73)
    Q.pauli4_rotation(qc2, (0, 1, 3, 4), ('Z', 'Z', 'Z', 'Z'), 0.73)
    assert np.max(np.abs(Operator(qc1).data - Operator(qc2).data)) < 1e-12
    mats = {'X': np.array([[0, 1], [1, 0]], complex), 'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]], complex)}
    ps = ('X', 'Y', 'Z', 'X')
    qc = QuantumCircuit(4)
    Q.pauli4_rotation(qc, (0, 1, 2, 3), ps, 0.41)
    Pm = mats[ps[0]]
    for p in ps[1:]:
        Pm = np.kron(mats[p], Pm)
    assert np.max(np.abs(Operator(qc).data - expm(-1j * 0.41 / 2 * Pm))) < TOL


@pytest.mark.parametrize('N,g,J,reps,n_terms', [(4, 0.6, 0.5, 1, 4), (6, 0.4, 0.8, 2, 11), (6, 0.9, 0.0, 3, 11),
                                                (6, 0.0, 0.7, 2, 11)])
def test_reps_fold_step_unitary_matches_multilayer_circuit(N, g, J, reps, n_terms):
    cfg = Q.ReservoirConfig(N=N, g=0.0, reps=reps, seed=42)
    bz, _ = cfg.sample_disorder()
    terms = Q.sample_syk4_terms(N, n_terms, seed=1)
    cp = Q.sample_syk4_couplings(n_terms, J=J, seed=1)
    pt = Q.sample_syk4_pauli_types(n_terms, seed=1)
    qc = QuantumCircuit(N)
    qc.ry(np.pi * 0.37, 0)
    for _ in range(reps):
        Q.mixed_layer(qc, N, g, terms, cp, pt, bz)
    U = Q.step_unitary_mixed(N, g, terms, cp, pt, reps, bz)
    pre = QuantumCircuit(N)
    pre.ry(np.pi * 0.37, 0)
    sv = U @ Statevector.from_instruction(pre).data
    assert np.max(np.abs(Statevector.from_instruction(qc).data - sv)) < TOL


def test_limits_J0_free_fermion_and_g0_pure_quartic():
    N, g, reps, seed = 6, 0.5, 2, 7
    bz, _ = Q.ReservoirConfig(N=N, seed=seed).sample_disorder()
    qc = QuantumCircuit(N)
    for _ in range(reps):
        for a, b in Q.chain_edges(N):
            qc.rxx(2 * g, a, b); qc.ryy(2 * g, a, b)
        for i in range(N):
            qc.rz(bz[i], i)
    U0 = Q.step_unitary_mixed(N, g, [], np.array([]), np.empty((0, 4)), reps, bz)
    assert np.max(np.abs(Operator(qc).data - U0)) < TOL
    nt = Q.default_n_sparse_terms(N)
    terms, cp, pt = Q.sample_syk4_terms(N, nt, seed), Q.sample_syk4_couplings(nt, 0.6, seed), Q.sample_syk4_pauli_types(nt, seed)
    qc = QuantumCircuit(N)
    for t, c, p in zip(terms, cp, pt):
        Q.pauli4_rotation(qc, t, p, 2 * c)
    for i in range(N):
        qc.rz(bz[i], i)
    assert np.max(np.abs(Operator(qc).data - Q.single_layer_unitary_mixed(N, 0.0, terms, cp, pt, bz))) < TOL


def test_readout_counts_and_generalisation():
    labels, ops = Q.feature_ops_mem_all(6, 0, 3)
    assert len(labels) == 3 * 5 + 9 * 4 + 27 * 3 == len(set(labels)) == 132
    l2, o2 = Q.feature_ops_memory(6, [0], 3)
    assert l2 == labels
    l3, _ = Q.feature_ops_memory(6, [0, 1], 3)
    assert len(l3) == 3 * 4 + 9 * 3 + 27 * 2 == len(set(l3)) == 93
    la, _ = Q.feature_ops_all(6, 3)
    assert len(la) == 171


def test_reference_r_ensembles_and_crossover():
    refs = Q.sample_reference_r_statistics(400, trials=25, seed=0)
    for name, txt in (('poisson', 2 * np.log(2) - 1), ('coe', 0.5307), ('cue', 0.5996)):
        assert abs(refs[name][0] - txt) < 0.03
    r_lo = np.mean([Q.level_spacing_ratio(Q.layer_unitary(Q.mixed_config(6, 1, 0.02, 1, s))) for s in range(6)])
    r_hi = np.mean([Q.level_spacing_ratio(Q.layer_unitary(Q.mixed_config(6, 1, 100.0, 1, s))) for s in range(6)])
    assert r_lo > 0.5 and r_hi < 0.45


# --------------------------------------------------------------------------- new: execution
def _nb4_reference(cfg, u):
    qc, labels, *_ = Q.build_trajectory_circuit_mixed(Q.ReservoirConfig(N=cfg.N, seed=cfg.seed), u, cfg.g, cfg.J,
                                                      cfg.reps, term_seed=cfg.seed)
    sim = AerSimulator(method='density_matrix')
    d = sim.run(transpile(qc, sim, optimization_level=1), shots=1).result().data(0)
    return np.array([[np.real(d[f'{l}__t{t}']) for l in labels] for t in range(len(u))])


def test_scalar_runner_equals_nb4_reference_circuit():
    cfg = Q.mixed_config(6, 1, 0.7, 3, 5)
    u = np.random.default_rng(1).random(30)
    X = Q.run_qrc_jobs([Q.QRCJob(cfg, u[:, None])], chunk=11, use_memo=False)[0]
    assert np.max(np.abs(X - _nb4_reference(cfg, u))) < TOL


@pytest.mark.parametrize('n_input,chunks', [(1, (7, 64)), (2, (5, 50))])
def test_chunked_equals_monolithic(n_input, chunks):
    cfg = Q.mixed_config(6, n_input, 2.0, 4, 3)
    z = np.random.default_rng(2).random((150, n_input))
    mono = Q.run_qrc_jobs([Q.QRCJob(cfg, z)], chunk=150, use_memo=False)[0]
    for c in chunks:
        assert np.max(np.abs(Q.run_qrc_jobs([Q.QRCJob(cfg, z)], chunk=c, use_memo=False)[0] - mono)) < TOL


def test_fused_equals_gate_level():
    cfg = Q.mixed_config(6, 1, 1.5, 2, 4)
    z = np.random.default_rng(3).random((25, 1))
    a = Q.run_qrc_jobs([Q.QRCJob(cfg, z)], chunk=25, use_memo=False)[0]
    b = Q.run_qrc_jobs([Q.QRCJob(cfg, z, fuse=False)], chunk=9, use_memo=False)[0]
    assert np.max(np.abs(a - b)) < TOL


def test_causality_and_memory():
    cfg = Q.mixed_config(6, 1, 5.0, 4, 0)
    za = np.random.default_rng(4).random((60, 1))
    zb = za.copy(); zb[30:] = 1 - zb[30:]
    Xa, Xb = Q.run_qrc_jobs([Q.QRCJob(cfg, za), Q.QRCJob(cfg, zb)], chunk=16, use_memo=False)
    assert np.max(np.abs(Xa[:30] - Xb[:30])) < 1e-12
    assert np.max(np.abs(Xa[30:] - Xb[30:])) > 1e-3
    zc = za.copy(); zc[10] = 1 - zc[10]   # a past input still influences later memory features
    Xc = Q.run_qrc_jobs([Q.QRCJob(cfg, zc)], chunk=16, use_memo=False)[0]
    assert np.max(np.abs(Xc[11:14] - Xa[11:14])) > 1e-4


def test_reset_pattern_recurrent_vs_qelm():
    from qiskit.circuit.library import UnitaryGate
    cfg = Q.mixed_config(6, 2, 1.0, 2, 0)
    z = np.random.default_rng(5).random((12, 2))
    qc = Q._build_chunk(Q.QRCJob(cfg, z), 0, 12, None, UnitaryGate(Q.step_unitary(cfg)))
    assert Q.count_resets_per_qubit(qc) == {0: 12, 1: 12}
    c1 = Q.mixed_config(6, 1, 1.0, 2, 0)
    qj = Q.QRCJob(c1, z[:, :1], qelm=True)
    qc = Q._build_chunk(qj, 0, 12, None, UnitaryGate(Q.step_unitary(c1)), Q._qelm_angles(qj.z, 6))
    assert Q.count_resets_per_qubit(qc) == {q: 12 for q in range(6)}


def test_config_validation_and_inputs():
    with pytest.raises(ValueError):
        Q.MultiInputReservoirConfig(N=3, n_input=3, g=0.1, J=0.1, reps=1, seed=0)
    cfg = Q.mixed_config(6, 1, 1.0, 1, 0)
    with pytest.raises(ValueError):
        Q.run_qrc_jobs([Q.QRCJob(cfg, np.full((5, 1), 1.5))], use_memo=False)
    if 'GPU' not in AerSimulator().available_devices():
        with pytest.raises(RuntimeError):
            Q.make_simulator(use_gpu=True)


def test_kappa_roundtrip():
    for k in (0.02, 0.4427, 3.1, 100.0):
        assert abs(P.kappa_of(Q.mixed_config(6, 1, k, 1, 0)) - k) < 1e-9


# --------------------------------------------------------------------------- new: classical side
def _traj(seed, split, n=300):
    return C.make_trajectory('mackey_glass', seed, split, n, 32, 50, 10)


def test_windows_are_causal():
    x = np.arange(100.0)
    idx = np.arange(40, 60)
    W = C.windows(x, 8, idx)
    for k in range(8):
        assert np.array_equal(W[:, k], x[idx - k])
    with pytest.raises(IndexError):
        C.windows(x, 50, idx)


def test_lifts_are_train_only():
    tr, va, te = _traj(0, 'train'), _traj(0, 'val'), _traj(0, 'test')
    for spec in (C.LiftSpec('rbf', 8, m=16, bw=1.0), C.LiftSpec('poly2', 4, proj='pca'), C.LiftSpec('linear', 8)):
        a = C.Lifter(spec).fit(tr.x, tr.eval_idx)
        za, _ = a.transform(va.x, va.drive_idx)
        b = C.Lifter(spec).fit(tr.x, tr.eval_idx)          # val/test never touched by fit
        zb, _ = b.transform(va.x, va.drive_idx)
        assert np.array_equal(za, zb)
        assert a.n_fit == len(tr.eval_idx)
        ztr, clip = a.transform(tr.x, tr.eval_idx)
        assert clip == 0.0 and ztr.min() >= 0 and ztr.max() <= 1
        if spec.family == 'rbf':
            Str = (C.windows(tr.x, 8, tr.eval_idx) - a.w_mu) / a.w_sd
            assert all(np.min(np.abs(Str - c).sum(1)) < 1e-9 for c in a.centres)   # centres are training windows
            g = a.gram_checks()
            assert not (g['nearly_constant'] or g['nearly_diagonal'] or g['duplicate_centres'])
    # a lift fitted on a different trajectory differs (it really is data-dependent)
    c = C.Lifter(C.LiftSpec('rbf', 8, m=16)).fit(te.x, te.eval_idx)
    assert not np.allclose(c.transform(va.x, va.drive_idx)[0], C.Lifter(C.LiftSpec('rbf', 8, m=16)).fit(tr.x, tr.eval_idx).transform(va.x, va.drive_idx)[0])


def test_poly_feature_cap():
    tr = _traj(0, 'train')
    with pytest.raises(ValueError):
        C.Lifter(C.LiftSpec('poly3', 16)).fit(tr.x, tr.eval_idx)


def test_volterra_exact_lift_and_witness_logic():
    u, y, yo = C.generate_volterra(500, 3)
    idx = np.arange(20, 500)
    assert np.max(np.abs(C.volterra_exact_lift(u, idx) - C.volterra_exact_lift_manual(u, idx))) == 0.0
    y_rec = np.zeros_like(y)
    lift = C.volterra_exact_lift(u, np.arange(12, len(u)))
    for k, t in enumerate(range(12, len(u))):
        y_rec[t] = np.dot(C.VOLTERRA_COEFS, lift[k]) + 0.1 * y_rec[t - 1]
    assert np.max(np.abs(y_rec[100:] - y[100:])) < 1e-12   # same recursion after burn-in transient


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 6)), rng.normal(size=200)
    m = C.fit_ridge(X, y, alpha=3.0)
    Xs = (X - X.mean(0)) / X.std(0)
    sk = Ridge(alpha=3.0).fit(Xs, y)
    assert np.allclose(m.predict(X), sk.predict(Xs), atol=1e-10)


def test_capacity_of_delay_line():
    rng = np.random.default_rng(1)
    u = rng.random((2000, 1))
    X = C.lag_stack(u[:, 0], 6) + 1e-6 * rng.normal(size=(2000, 6))
    prof = C.capacity_profile(X, u, 50, 1200, 300, 300, k_mc=10, k2=4, k3=3)
    mc = prof['mc_per_jk'][0]
    assert np.all(mc[:6] > 0.99) and np.all(mc[7:] < 0.05)
    assert prof['ipc2'] < 0.5 and prof['ipc3'] < 0.5           # linear features -> no nonlinear capacity
    Xq = np.concatenate([X, C.legendre_targets(2 * u - 1, 2, 2)[0]], 1)
    assert C.capacity_profile(Xq, u, 50, 1200, 300, 300, k_mc=10, k2=4, k3=3)['ipc2'] > 4.5


def test_bootstrap_sanity():
    rng = np.random.default_rng(2)
    y = {0: rng.normal(size=400)}
    pa = {(0, 0): y[0] + rng.normal(size=400)}
    pb = {(0, 0): y[0] + 0.5 * rng.normal(size=400)}
    bt = C.paired_block_bootstrap(y, pa, pb, 40, 300, 0)
    assert bt['delta'] > 0.3 and bt['ci95'][0] > 0
    same = C.paired_block_bootstrap(y, pa, pa, 40, 100, 0)
    assert same['delta'] == 0 and same['ci95'] == [0.0, 0.0]


def test_edge_band_definition():
    k = np.geomspace(0.02, 100, 40)
    r = 0.389 + (0.593 - 0.389) / (1 + (k / 8.0) ** 3)
    lo, hi, eta = Q.edge_band_from_r(k, r, 0.389, 0.593, 0.1)
    assert 2 < lo < 8 < hi < 30


def test_vault_and_outcome():
    calls = []
    v = P.TestVault(lambda ds: calls.append(ds) or ds)
    with pytest.raises(RuntimeError):
        v.release('x')
    v.lock({'a': 1})
    assert v.release('x') == 'x'
    with pytest.raises(RuntimeError):
        v.release('x')
    with pytest.raises(RuntimeError):
        v.lock({'a': 2})
    assert calls == ['x']
    assert P.outcome(True, True, True).startswith('1')
    assert P.outcome(True, True, False).startswith('2')
    assert P.outcome(False, True, True).startswith('3')
    assert P.outcome(True, False, True).startswith('4')
    assert P.outcome(False, False, False).startswith('5')


def test_memory_candidates_respect_thresholds():
    cap = {('e', 1): {'mc_total': 4.0, 'ipc_nl': 2.0}, ('a', 1): {'mc_total': 4.5, 'ipc_nl': 1.0},
           ('b', 1): {'mc_total': 3.0, 'ipc_nl': 0.2}, ('c', 1): {'mc_total': 3.7, 'ipc_nl': 1.3}}
    keys, ok, _ = P.memory_candidates(cap, ('e', 1), {'mc_min_frac_of_eoc': 0.9, 'ipc_nl_max_frac_of_eoc': 0.7}, 4)
    assert ok and keys == [('a', 1), ('c', 1)]     # Pareto front first, then remaining feasible by MC
    keys, ok, _ = P.memory_candidates({('e', 1): cap[('e', 1)], ('b', 1): cap[('b', 1)]}, ('e', 1),
                                      {'mc_min_frac_of_eoc': 0.9, 'ipc_nl_max_frac_of_eoc': 0.7}, 4)
    assert not ok and keys == [('b', 1)]
