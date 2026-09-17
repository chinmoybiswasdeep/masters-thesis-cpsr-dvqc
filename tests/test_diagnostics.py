"""
test_diagnostics.py -- locks in the architecture-repair investigation's
central finding (docs/DQRC_ARCHITECTURE_REPAIR.md): 'rzz'/'cp' interface
gates are diagonal in the processor's own Z basis and CANNOT inject
population/coherence into a processor whose dynamics conserves total
Z-magnetization (true whenever N_P<4, since a SYK4 quartic term needs 4
distinct qubits); only 'zx' can. If this ever silently stops being true
(e.g. someone changes `mixed_layer` or `apply_interface`), these tests fail
loudly rather than the finding quietly rotting in a markdown file.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc import diagnostics as diag  # noqa: E402
from decoupled_qrc.experiments import DQRCConfig  # noqa: E402


def test_rzz_and_cp_cannot_entangle_magnetization_conserving_processor():
    """N_P=3 has ZERO possible SYK4 quartic terms (comb(3,4)==0), so the
    processor's own dynamics is pure SYK2-like (RXX+RYY hopping + RZ), which
    conserves total Z-magnetization exactly. A diagonal interface gate
    ('rzz'/'cp') must then leave the processor in a PRODUCT state with
    everything else -- purity_P == 1.0 to machine precision -- regardless of
    lambda_mp."""
    for kind in ("rzz", "cp"):
        for lam in (0.3, 1.0, 3.0):
            cfg = DQRCConfig(N_M=2, N_P=3, kappa_processor=1.0, lambda_mp=lam,
                              interface_kind=kind, reps_processor=1, n_taps=1)
            info = diag.info_theoretic_diagnostics(cfg, T_small=20, master_seed=0)
            assert info["purity_P"] == pytest.approx(1.0, abs=1e-9), (kind, lam, info)
            assert info["mutual_information_MP"] == pytest.approx(0.0, abs=1e-9), (kind, lam, info)


def test_zx_interface_does_entangle_magnetization_conserving_processor():
    """The same N_P=3 processor, coupled via 'zx' (Z_M X_P -- an X-rotation
    on P conditioned on M, non-diagonal in P's own basis) MUST show real
    entanglement at nonzero lambda_mp -- the direct positive control proving
    the 'rzz'/'cp' null result above is a real physical fact about the gate,
    not a bug in the entanglement-measurement code itself."""
    cfg0 = DQRCConfig(N_M=2, N_P=3, kappa_processor=1.0, lambda_mp=0.0,
                       interface_kind="zx", reps_processor=1, n_taps=1)
    info0 = diag.info_theoretic_diagnostics(cfg0, T_small=20, master_seed=0)
    assert info0["purity_P"] == pytest.approx(1.0, abs=1e-9)  # lambda_mp=0 is still an identity

    cfg = DQRCConfig(N_M=2, N_P=3, kappa_processor=1.0, lambda_mp=0.5,
                      interface_kind="zx", reps_processor=1, n_taps=1)
    info = diag.info_theoretic_diagnostics(cfg, T_small=20, master_seed=0)
    assert info["purity_P"] < 0.9, info
    assert info["mutual_information_MP"] > 0.05, info


def test_grouped_readout_feature_group_sizes():
    cfg = DQRCConfig(N_M=2, N_P=3, kappa_processor=1.0, lambda_mp=0.3, interface_kind="zx",
                      reps_processor=1, n_taps=1)
    run = diag.run_dqrc_grouped(cfg, T=15, master_seed=0, max_weight_proc=3, max_weight_mem=2)
    assert run.X_mem.shape == (15, len(run.labels_mem))
    assert run.X_proc.shape == (15, len(run.labels_proc))
    assert run.X_cross.shape == (15, 4)  # 1 mem tap x 1 proc entry x 4 (ZZ,XX,ZX,XZ)


def test_feature_health_detects_collapsed_rank():
    """A feature matrix with duplicated columns must report rank strictly
    less than its column count -- the health-check this module leans on to
    detect 'stitching homogenized the features' must not be vacuous."""
    rng = np.random.RandomState(0)
    base = rng.normal(size=(100, 3))
    X = np.hstack([base, base])  # 6 columns, rank 3
    health = diag.feature_health(X)
    assert health["rank"] == 3
    assert health["n_features"] == 6


def test_standalone_processor_matched_reproducible():
    a = diag.standalone_processor_matched(N_P=3, kappa_processor=1.0, T=15, reps=1, max_weight=3, master_seed=5)
    b = diag.standalone_processor_matched(N_P=3, kappa_processor=1.0, T=15, reps=1, max_weight=3, master_seed=5)
    assert np.array_equal(a["X"], b["X"])
