"""Protocol layer of notebook 5: runtime modes, the test vault, the
preregistered criteria, the validation-only selection stages, the locked
test evaluation and the pitfall audit.

Order of operations enforced in code:
  1. criteria are frozen (hash) before anything is simulated;
  2. every selection stage sees train/validation trajectories only -- test
     trajectories do not exist until `TestVault.release` is called;
  3. `TestVault.lock` freezes the full method; `release` refuses to run
     before the lock, or twice for the same dataset.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import klqrc_classical as C
import klqrc_quantum as Q

DATASETS = ('mackey_glass', 'lorenz')

# =============================================================================
# Runtime modes
# =============================================================================


def runtime_config(fast_mode: bool = True, long_run: bool = False) -> dict:
    cfg = dict(
        N=6, horizons=(1, 5, 10), nontrivial_horizons=(5, 10), d_max=32, washout=100,
        delay_grid=(1, 2, 4, 8, 16, 24, 32), memory_grid=(1, 2, 4, 8, 16, 24, 32), degree_grid=(1, 2, 3),
        nystrom_m=(8, 16, 32), bw_grid=(0.5, 1.0, 2.0), projections=('pred', 'pred1', 'pca'), H_proj=10,
        kappa_grid=tuple(round(float(k), 6) for k in np.geomspace(0.02, 100.0, 12)),
        kappa_dense=tuple(round(float(k), 6) for k in np.geomspace(0.02, 100.0, 40)),
        n_diag_seeds=15, n_ref_trials=200, edge_delta_eta=0.10, n_band_points=4,
        reps_grid=(1, 2, 3, 4, 6, 8, 12, 16), reps_ref=4,
        k_mc=20, k2=10, k3=6,
        shortlist_kernel=3, max_mem_candidates=4, max_validation_rounds=3,
        proxy_lags=8, chunk=256, n_boot=2000, block=40, multichannel_p=2,
        esn_grid=((0.5, 0.3), (0.5, 1.0), (0.8, 0.3), (0.8, 1.0), (0.95, 0.3), (0.95, 1.0), (1.2, 0.3), (1.2, 1.0)),
        volterra=dict(n_train=4000, n_val=1000, n_test=1000, noise_frac=0.05),
        capacity_seed_base=90000, haar_seed_base=30000,
    )
    if long_run:
        cfg.update(mode='LONG_RUN', N=8, n_train=10000, n_val=2500, n_test=2500,
                   data_seeds=(0, 1, 2, 3, 4), res_seeds=(0, 1, 2, 3, 4),
                   cap=dict(washout=200, n_tr=4000, n_va=1000, n_te=1000),
                   nystrom_m=(8, 16, 32, 64), bw_grid=(0.25, 0.5, 1.0, 2.0, 4.0),
                   kappa_grid=tuple(round(float(k), 6) for k in np.geomspace(0.02, 100.0, 20)),
                   n_diag_seeds=30, shortlist_kernel=5, max_mem_candidates=8)
    elif fast_mode:
        cfg.update(mode='FAST_MODE', n_train=1200, n_val=400, n_test=400, data_seeds=(0, 1, 2),
                   res_seeds=(0, 1, 2), cap=dict(washout=100, n_tr=1500, n_va=500, n_te=500))
    else:
        raise ValueError('choose FAST_MODE or LONG_RUN')
    return cfg


# Predetermined search expansions, applied only if validation is insufficient
# (stop rule: validation J(KERNEL_MEMORY) < J(BEST_EOC)).
ROUND_EXPANSIONS = (
    {},
    {'nystrom_m': (64,), 'bw_grid': (0.25, 4.0)},
    {'shortlist_kernel': 5, 'max_mem_candidates': 8},
)

# =============================================================================
# Preregistered criteria (frozen before any simulation; hashed)
# =============================================================================

PREREGISTERED = {
    'transformation': {
        'T1_volterra_lift_within_rel': 0.05,     # E(lin on phi*) <= 1.05 E(raw deg-3 oracle)
        'T2_volterra_raw_linear_worse_rel': 0.20,  # E(raw linear) >= 1.20 E(raw deg-3 oracle)
        'T3_nl_gain_ratio_max': 0.5,             # G_NL^lift <= 0.5 G_NL^raw at L=d_sel AND grid median, every h
        'T4_memory_min_rel_improvement': 0.10,   # E_lift(L=1,q=1) -> E_lift(L=d_sel,q=1), CI excl. 0
        'T5_ablation_min_rel_degradation': 0.10,  # delay truncation (d//4) and history shuffle, CI excl. 0
        'T6_min_seeds': 3,                       # every criterion holds in every data seed
        'T7_split': 'test',
    },
    'beats_eoc': {
        'B1_eoc_inside_edge_band': True,
        'B2_reps_validation_selected': True,
        'B3_nontrivial_horizons': [5, 10],
        'B4_min_rel_improvement_one_system': 0.10,
        'B4_min_rel_improvement_other_system': 0.05,
        'B5_ci_level': 0.95,
        'B6_resources_matched': True,
        'B7_max_rel_worse_than_kernel_eoc': 0.05,
    },
    'qrc_value': {
        'Q1_horizons': [5, 10],
        'Q1_min_rel_improvement_over_kernel_only': 0.05,
        'Q1_ci_excludes_zero': True,
    },
    'full_hypothesis': 'TRANSFORMATION_PASS and BEATS_EOC_PASS and QRC_VALUE_PASS',
    'memory_qrc_thresholds': {'mc_min_frac_of_eoc': 0.9, 'ipc_nl_max_frac_of_eoc': 0.7},
    'statistics': {'bootstrap': 'paired moving-block over test time indices', 'block_min': 40,
                   'block_rule': 'max(40, largest |test-error| autocorrelation length (first lag < 0.1) over all '
                                 'compared models), rounded up to a multiple of 10',
                   'amendment': 'block rule replaces the fixed block=40 after a run aborted at the audit check '
                                '"block >= autocorrelation length" (measured 62/57), before any test metric was viewed',
                   'n_boot': 2000, 'aggregate': 'median over (data seed, reservoir seed) units'},
}


def criteria_hash() -> str:
    return hashlib.sha256(json.dumps(PREREGISTERED, sort_keys=True).encode()).hexdigest()


# =============================================================================
# Test vault
# =============================================================================


class TestVault:
    """Test trajectories are generated only on `release`, after `lock`."""

    def __init__(self, builder):
        self._builder = builder
        self.lock_record = None
        self.lock_hash = None
        self.lock_time = None
        self.criteria_hash_at_lock = None
        self.releases = {}

    def lock(self, record: dict) -> str:
        if self.lock_record is not None:
            raise RuntimeError('method already locked')
        self.lock_record = json.loads(json.dumps(record, default=str))
        self.lock_hash = hashlib.sha256(json.dumps(self.lock_record, sort_keys=True).encode()).hexdigest()
        self.lock_time = time.time()
        self.criteria_hash_at_lock = criteria_hash()
        return self.lock_hash

    def release(self, dataset: str):
        if self.lock_record is None:
            raise RuntimeError('test set requested before the method was locked')
        if criteria_hash() != self.criteria_hash_at_lock:
            raise RuntimeError('preregistered criteria changed after the lock')
        if dataset in self.releases:
            raise RuntimeError(f'test set for {dataset} already released once')
        self.releases[dataset] = time.time()
        return self._builder(dataset)

    @property
    def access_count(self):
        return {k: 1 for k in self.releases}


# =============================================================================
# Ledger: timings, budgets, audit
# =============================================================================


class Ledger:
    def __init__(self):
        self.timings = {}
        self.budget = {'eoc_side_val_evals': 0, 'memory_side_val_evals': 0}
        self.checks = []

    def time(self, name):
        ledger = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter(); self.s0 = dict(Q.STATS); return self

            def __exit__(self, *a):
                ledger.timings[name] = {'wall_s': time.perf_counter() - self.t,
                                        'aer_s': Q.STATS['aer_seconds'] - self.s0['aer_seconds'],
                                        'aer_steps': Q.STATS['aer_steps'] - self.s0['aer_steps']}
        return _T()

    def check(self, name, ok, detail=''):
        ok = bool(ok)
        self.checks.append({'check': name, 'pass': ok, 'detail': str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))
        return ok

    def raise_if_failed(self):
        bad = [c for c in self.checks if not c['pass']]
        if bad:
            raise AssertionError('pitfall audit failed: ' + '; '.join(c['check'] for c in bad))


# =============================================================================
# Data
# =============================================================================


def build_split(cfg, dataset, data_seed, split):
    n = {'train': cfg['n_train'], 'val': cfg['n_val'], 'test': cfg['n_test']}[split]
    return C.make_trajectory(dataset, data_seed, split, n, cfg['d_max'], cfg['washout'], max(cfg['horizons']))


def build_dev_data(cfg) -> dict:
    """{dataset: {data_seed: {'train': Trajectory, 'val': Trajectory}}}."""
    return {ds: {s: {sp: build_split(cfg, ds, s, sp) for sp in ('train', 'val')} for s in cfg['data_seeds']}
            for ds in DATASETS}


def min_segment_distance(x1, x2, L: int = 32) -> float:
    """Smallest L-infinity distance between any length-L segment of x1 and any
    of x2.  Zero would mean the two 'independent' trajectories share a
    stretch (overlapping windows of one series); chaotic trajectories from
    different initial conditions only ever come close, never coincide."""
    A = np.lib.stride_tricks.sliding_window_view(np.asarray(x1, float), L)
    B = np.lib.stride_tricks.sliding_window_view(np.asarray(x2, float), L)
    best = np.inf
    for i in range(0, len(A), 256):
        d = np.abs(A[i:i + 256, None, :] - B[None, :, :]).max(-1)
        best = min(best, float(d.min()))
    return best


def volterra_split(cfg, data_seed, split):
    v = cfg['volterra']
    return C.generate_volterra(v[f'n_{split}'], C.split_seed('volterra', data_seed, split), v['noise_frac'])


def test_builder(cfg):
    def build(ds):
        if ds == 'volterra':
            return {s: volterra_split(cfg, s, 'test') for s in cfg['data_seeds']}
        return {s: build_split(cfg, ds, s, 'test') for s in cfg['data_seeds']}
    return build


# =============================================================================
# Inputs (representations) -- fitted on each data seed's TRAIN trajectory
# =============================================================================


def fit_inputs(spec: C.LiftSpec, trajs: dict, cfg, lifter_seed: int = 0, S_mode=None):
    """Return (lifter or None, {split: z (T_drive, p)}, {split: clip fraction}).
    S_mode='shuffle' applies the history-shuffle ablation to every split."""
    tr = trajs['train']
    if spec.family == 'raw':
        xs = tr.x[tr.drive_idx]
        lo, hi = float(xs.min()), float(xs.max())
        zs, clips = {}, {}
        for sp, t in trajs.items():
            zs[sp], clips[sp] = C.raw_input(t, lo, hi, spec.p)
        return None, zs, clips
    lf = C.Lifter(spec, seed=lifter_seed, H_proj=cfg['H_proj'])
    if S_mode == 'shuffle':
        Ss = {sp: C.history_shuffled_windows(t.x, spec.d, t.drive_idx, seed=(t.data_seed, C.SPLIT_IDS[sp], 77))
              for sp, t in trajs.items()}
        off = tr.washout
        lf.fit(tr.x, tr.eval_idx, S_override=Ss['train'][off:])
        out = {sp: lf.transform(t.x, t.drive_idx, S_override=Ss[sp]) for sp, t in trajs.items()}
    else:
        lf.fit(tr.x, tr.eval_idx)
        out = {sp: lf.transform(t.x, t.drive_idx) for sp, t in trajs.items()}
    return lf, {sp: v[0] for sp, v in out.items()}, {sp: v[1] for sp, v in out.items()}


# =============================================================================
# Readout
# =============================================================================


def readout(F: dict, trajs: dict, horizons, out_splits=('val',)):
    """Ridge readout, alpha per horizon on val, fit on train.  F[split] are
    eval-row feature matrices.  Returns {split: (n, H) predictions}, alphas."""
    Y = np.stack([trajs['train'].target(h) for h in horizons], 1)
    Yv = np.stack([trajs['val'].target(h) for h in horizons], 1)
    m = C.fit_ridge(F['train'], Y, F['val'], Yv)
    return {sp: m.predict(F[sp]) for sp in out_splits}, m.alpha


def unit_nrmse(preds, trajs, horizons, split):
    return {h: C.nrmse(preds[split][:, k], trajs[split].target(h)) for k, h in enumerate(horizons)}


def score_J(results, horizons, split='val'):
    """Selection objective: median over units of the mean over horizons."""
    return float(np.median([np.mean([r['nrmse'][split][h] for h in horizons]) for r in results]))


@dataclass
class Entry:
    unit: tuple
    cfg: Q.MultiInputReservoirConfig
    z: dict
    trajs: dict
    qelm: bool = False
    tag: str = ''


def run_entries(entries, cfg, splits=('train', 'val'), use_gpu=False):
    """Run every (entry, split) trajectory in ONE chunked Aer batch, fit the
    readouts and return per-entry predictions / NRMSE."""
    jobs = [Q.QRCJob(e.cfg, e.z[sp], qelm=e.qelm) for e in entries for sp in splits]
    X = Q.run_qrc_jobs(jobs, chunk=cfg['chunk'], use_gpu=use_gpu)
    out, k = [], 0
    hz = cfg['horizons']
    for e in entries:
        F = {}
        for sp in splits:
            F[sp] = X[k][e.trajs[sp].washout:]
            k += 1
        outs = tuple(s for s in splits if s != 'train')
        preds, alpha = readout(F, e.trajs, hz, outs)
        out.append({'unit': e.unit, 'tag': e.tag, 'cfg': e.cfg, 'preds': preds, 'alpha': alpha,
                    'nrmse': {sp: unit_nrmse(preds, e.trajs, hz, sp) for sp in outs},
                    'n_features': F['train'].shape[1], 'rank': C.effective_rank(F['train']),
                    'cond': C.condition_number(F['train'])})
    return out


# =============================================================================
# Stage 1 -- representation screening (classical, validation only)
# =============================================================================


def candidate_specs(cfg, p=1):
    specs = []
    for d in cfg['delay_grid']:
        for proj in cfg['projections']:
            specs.append(C.LiftSpec('linear', d, proj=proj, p=p))
            for m in cfg['nystrom_m']:
                for bw in cfg['bw_grid']:
                    specs.append(C.LiftSpec('rbf', d, m=m, bw=bw, proj=proj, p=p))
            for fam in ('poly2', 'poly3'):
                if C.poly_feature_count(d, int(fam[-1])) <= C.MAX_POLY_FEATURES:
                    specs.append(C.LiftSpec(fam, d, proj=proj, p=p))
    return specs


def proxy_val(z, trajs, cfg):
    """Cheap QRC stand-in: linear fading-memory readout of the last
    `proxy_lags` input values (validation NRMSE per horizon)."""
    F = {sp: C.lag_stack(z[sp], cfg['proxy_lags'])[trajs[sp].washout:] for sp in ('train', 'val')}
    preds, _ = readout(F, trajs, cfg['horizons'])
    return unit_nrmse(preds, trajs, cfg['horizons'], 'val')


def screen(dev, dataset, cfg, specs):
    rows = []
    for spec in specs:
        per_seed = []
        for s, trajs in dev[dataset].items():
            try:
                lf, z, clip = fit_inputs(spec, trajs, cfg)
            except ValueError as e:
                per_seed = None
                break
            pv = proxy_val(z, trajs, cfg)
            em = C.edmd_metrics(lf, trajs['train'], trajs['val'], cfg['horizons']) if lf is not None else {}
            gc = lf.gram_checks() if lf is not None else {}
            per_seed.append({'proxy': pv, 'edmd': em, 'gram': gc, 'clip_val': clip['val'],
                             'dict_dim': len(lf.p_mu) if lf is not None else 1})
        if per_seed is None:
            continue
        J = float(np.median([np.mean(list(r['proxy'].values())) for r in per_seed]))
        recon = float(np.median([r['edmd'].get('recon_r2', 1.0) for r in per_seed]))
        bad_gram = any(r['gram'].get('nearly_constant') or r['gram'].get('nearly_diagonal')
                       or r['gram'].get('duplicate_centres') for r in per_seed)
        row = {'dataset': dataset, 'spec': spec, 'name': spec.name, 'family': spec.family, 'd': spec.d,
               'm': spec.m, 'bw': spec.bw, 'proj': spec.proj, 'proxy_J_val': J, 'recon_r2': recon,
               'rejected': bool(recon < 0.95 or bad_gram), 'reject_reason':
               ('recon_r2<0.95' if recon < 0.95 else '') + (' gram' if bad_gram else ''),
               'dict_dim': per_seed[0]['dict_dim'], 'clip_val': float(np.median([r['clip_val'] for r in per_seed]))}
        for key in ('closure_rel_err', 'rollout10_rel_err', 'spectral_radius_A') + tuple(
                f'edmd_nrmse_h{h}' for h in cfg['horizons']):
            row[key] = float(np.median([r['edmd'].get(key, np.nan) for r in per_seed]))
        for h in cfg['horizons']:
            row[f'proxy_val_h{h}'] = float(np.median([r['proxy'][h] for r in per_seed]))
        rows.append(row)
    return rows


def shortlist(rows, k):
    ok = sorted([r for r in rows if r['spec'].nonlinear and not r['rejected']], key=lambda r: r['proxy_J_val'])
    return [r['spec'] for r in ok[:k]]


# =============================================================================
# Stage 2 -- physics: edge band and capacities
# =============================================================================


def edge_physics(cfg):
    N = cfg['N']
    refs = Q.sample_reference_r_statistics(2 ** N, trials=cfg['n_ref_trials'], seed=0)
    dense = Q.kappa_diagnostics(cfg['kappa_dense'], N, cfg['reps_ref'], cfg['n_diag_seeds'])
    r_med = np.median(dense['r'], axis=1)
    k_lo, k_hi, eta = Q.edge_band_from_r(cfg['kappa_dense'], r_med, refs['poisson'][0], refs['cue'][0],
                                         cfg['edge_delta_eta'])
    grid = Q.kappa_diagnostics(cfg['kappa_grid'], N, cfg['reps_ref'], cfg['n_diag_seeds'])
    band_points = tuple(round(float(k), 6) for k in np.geomspace(k_lo, k_hi, cfg['n_band_points']))
    return {'refs': refs, 'dense': dense, 'grid': grid, 'kappa_lo': k_lo, 'kappa_hi': k_hi, 'eta_dense': eta,
            'band_points': band_points}


def capacity_scan(configs, cfg, n_input=1, use_gpu=False):
    """Independent i.i.d. characterisation: one trajectory per (config,
    reservoir seed); the input sequence depends only on the reservoir seed
    (common random numbers across configs)."""
    c = cfg['cap']
    gap = max(cfg['k_mc'], cfg['k2'], cfg['k3']) + 1
    T = c['washout'] + c['n_tr'] + c['n_va'] + c['n_te'] + 2 * gap + 1
    rows = []
    for b0 in range(0, len(configs), 24):
        block = configs[b0:b0 + 24]
        us = [np.random.default_rng(cfg['capacity_seed_base'] + rc.seed).random((T, n_input)) for rc in block]
        X = Q.run_qrc_jobs([Q.QRCJob(rc, u) for rc, u in zip(block, us)], chunk=cfg['chunk'],
                           use_gpu=use_gpu, use_memo=False)
        for rc, u, Xi in zip(block, us, X):
            prof = C.capacity_profile(Xi, u, c['washout'], c['n_tr'], c['n_va'], c['n_te'], cfg['k_mc'],
                                      cfg['k2'], cfg['k3'], null_seed=cfg['capacity_seed_base'] + 500 + rc.seed)
            rows.append({'cfg': rc, 'seed': rc.seed, **{k: v for k, v in prof.items()
                                                       if not k.endswith(('per_target', 'descr'))},
                         'mc_per_k': prof['mc_per_jk'].ravel().tolist()})
    return rows


def capacity_medians(rows, key_fn):
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r['cfg']), []).append(r)
    out = {}
    for k, g in groups.items():
        out[k] = {m: float(np.median([r[m] for r in g])) for m in
                  ('mc_total', 'mc_total_unclipped', 'ipc2', 'ipc3', 'ipc2_unthresholded', 'ipc3_unthresholded')}
        out[k]['ipc_nl'] = out[k]['ipc2'] + out[k]['ipc3']
        out[k]['n_seeds'] = len(g)
    return out


# =============================================================================
# Stage 3 -- EOC selection (validation only)
# =============================================================================


def paired_entries(dev, dataset, spec, rc_fn, cfg, tag, seeds_pairs=None, splits=('train', 'val'),
                   S_mode=None, qelm=False, trajs_override=None):
    """Entries for every data seed; reservoir seed = paired seed by default."""
    out = []
    data = trajs_override if trajs_override is not None else dev[dataset]
    pairs = seeds_pairs or [(s, s) for s in data]
    fitted = {}
    for ds, rs in pairs:
        if ds not in fitted:
            trajs = {sp: data[ds][sp] for sp in splits}
            fitted[ds] = fit_inputs(spec, trajs, cfg, S_mode=S_mode)
        lf, z, clip = fitted[ds]
        out.append(Entry((ds, rs), rc_fn(rs), z, {sp: data[ds][sp] for sp in splits}, qelm=qelm, tag=tag))
    return out


def kappa_rc(cfg, kappa, reps, n_input=1):
    return lambda rs: Q.mixed_config(cfg['N'], n_input, kappa, reps, rs)


def haar_rc(cfg, n_input=1):
    return lambda rs: Q.MultiInputReservoirConfig(cfg['N'], n_input, 0.0, 0.0, 1, rs,
                                                  haar_seed=cfg['haar_seed_base'] + rs)


def kappa_of(rc) -> float:
    """Invert kappa_to_gJ: kappa = (g/G_MAX)/(J/J_MAX)."""
    return float((rc.g / Q.G_MAX) / (rc.J / Q.J_MAX))


def ckey(rc):
    return (round(kappa_of(rc), 6), int(rc.reps))


def run_groups(groups: dict, cfg, splits=('train', 'val'), use_gpu=False) -> dict:
    """Run several named groups of entries in ONE Aer batch."""
    flat = [e for g in groups.values() for e in g]
    res = run_entries(flat, cfg, splits, use_gpu)
    out, k = {}, 0
    for tag, g in groups.items():
        out[tag] = res[k:k + len(g)]
        k += len(g)
    return out


# =============================================================================
# Stage 4 -- memory-dominant candidates from measured capacities
# =============================================================================


def memory_candidates(cap_med: dict, eoc_key, thresholds, k_max):
    """Feasible: MC >= f_mc * MC_EOC and IPC_NL <= f_nl * IPC_NL_EOC (EOC
    itself excluded).  Rank the Pareto front (max MC, min IPC_NL) by MC, then
    fill with the remaining feasible configs by MC.  If none is feasible,
    return the global Pareto point closest to feasibility, flagged."""
    E = cap_med[eoc_key]
    mc_min = thresholds['mc_min_frac_of_eoc'] * E['mc_total']
    nl_max = thresholds['ipc_nl_max_frac_of_eoc'] * E['ipc_nl']
    keys = [k for k in cap_med if k != eoc_key]

    def pareto(ks):
        return [k for k in ks if not any(
            cap_med[o]['mc_total'] >= cap_med[k]['mc_total'] and cap_med[o]['ipc_nl'] <= cap_med[k]['ipc_nl']
            and (cap_med[o]['mc_total'] > cap_med[k]['mc_total'] or cap_med[o]['ipc_nl'] < cap_med[k]['ipc_nl'])
            for o in ks)]

    feas = [k for k in keys if cap_med[k]['mc_total'] >= mc_min and cap_med[k]['ipc_nl'] <= nl_max]
    if feas:
        front = sorted(pareto(feas), key=lambda k: -cap_med[k]['mc_total'])
        rest = sorted([k for k in feas if k not in front], key=lambda k: -cap_med[k]['mc_total'])
        return (front + rest)[:k_max], True, {'mc_min': mc_min, 'ipc_nl_max': nl_max, 'n_feasible': len(feas)}
    front = pareto(keys)
    viol = lambda k: max(0.0, (mc_min - cap_med[k]['mc_total']) / mc_min) + max(
        0.0, (cap_med[k]['ipc_nl'] - nl_max) / max(nl_max, 1e-9))
    return sorted(front, key=viol)[:1], False, {'mc_min': mc_min, 'ipc_nl_max': nl_max, 'n_feasible': 0}


# =============================================================================
# Statistics helpers for the locked test
# =============================================================================


def unit_preds(results, horizons, h, split='test'):
    k = list(horizons).index(h)
    return {r['unit']: r['preds'][split][:, k] for r in results}


def median_nrmse(results, h, split='test'):
    return float(np.median([r['nrmse'][split][h] for r in results]))


def compare(results_a, results_b, trajs_test, horizons, h, cfg, seed=0):
    """Paired block bootstrap of Delta = (E_A - E_B)/E_A on test."""
    ytrue = {ds: t.target(h) for ds, t in trajs_test.items()}
    pa, pb = unit_preds(results_a, horizons, h), unit_preds(results_b, horizons, h)
    if set(pa) != set(pb):
        raise ValueError('paired comparison needs identical (data seed, reservoir seed) units')
    return C.paired_block_bootstrap(ytrue, pa, pb, cfg['block'], cfg['n_boot'], seed)


def classical_result(unit, preds, trajs, horizons, tag):
    return {'unit': unit, 'tag': tag, 'preds': preds,
            'nrmse': {sp: unit_nrmse(preds, trajs, horizons, sp) for sp in preds}}


def replicate_units(results, res_seeds):
    """Give a reservoir-free model one result per (data seed, reservoir seed)
    unit so that paired comparisons with QRC models align."""
    return [dict(r, unit=(r['unit'][0], rs)) for r in results for rs in res_seeds]


# =============================================================================
# Outcome mapping (fixed before the test)
# =============================================================================


def outcome(T: bool, B: bool, Qv: bool) -> str:
    if T and B and Qv:
        return '1. Full support'
    if T and B and not Qv:
        return '2. Representation-only support'
    if not T and B:
        return '3. Performance-only support'
    if T and not B:
        return '4. EOC not beaten'
    return '5. No support under tested conditions'


# =============================================================================
# Output helpers
# =============================================================================


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else v) for k, v in r.items()})


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, '__dataclass_fields__'):
        return jsonable(asdict(o))
    return o
