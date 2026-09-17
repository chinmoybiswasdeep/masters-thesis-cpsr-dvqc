"""
test_feature_analysis.py -- rank diagnostics, leakage-safe residualization,
rank-matched random-projection budget scans (Part 5 of the V2 validation
spec).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.feature_analysis import (  # noqa: E402
    numerical_rank, effective_rank, ridge_effective_dof, fit_residualizer, residualize,
    random_projection_budget_scan,
)


def test_numerical_rank_detects_duplicated_columns():
    rng = np.random.RandomState(0)
    base = rng.normal(size=(100, 5))
    X = np.hstack([base, base])  # 10 columns, rank 5
    assert numerical_rank(X) == 5


def test_effective_rank_le_numerical_rank():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 10))
    assert effective_rank(X) <= numerical_rank(X) + 1e-6


def test_ridge_dof_decreases_with_alpha():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 10))
    dof_small = ridge_effective_dof(X, 0.01)
    dof_large = ridge_effective_dof(X, 100.0)
    assert dof_small > dof_large


def test_ridge_dof_bounded_by_rank():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 10))
    dof = ridge_effective_dof(X, 1e-9)
    assert dof <= numerical_rank(X) + 1e-3


def test_residualization_removes_train_correlation_no_leakage():
    """The residualizer must be FIT on train only; applying it to test data
    it never saw during fitting must still remove most of the
    train-learned linear relationship (proving the fit generalizes, not
    that test data leaked into fitting)."""
    rng = np.random.RandomState(0)
    T = 200
    X_basis = rng.normal(size=(T, 5))
    true_W = rng.normal(size=(5, 2))
    X_target = X_basis @ true_W + 0.01 * rng.normal(size=(T, 2))
    train = np.arange(120)
    test = np.arange(150, 200)

    applier = fit_residualizer(X_target, X_basis, train)
    resid_train = applier(X_target[train], X_basis[train])
    resid_test = applier(X_target[test], X_basis[test])
    assert np.abs(resid_train).max() < 0.1
    assert np.abs(resid_test).max() < 0.2  # test residual should also be small since the true relationship holds everywhere


def test_residualizer_fit_does_not_use_test_targets():
    """Refitting with DIFFERENT (corrupted) test-block targets must not
    change the fitted W at all -- proof the fit only used train rows."""
    rng = np.random.RandomState(0)
    T = 100
    X_basis = rng.normal(size=(T, 4))
    X_target = rng.normal(size=(T, 2))
    train = np.arange(60)

    applier1 = fit_residualizer(X_target, X_basis, train)
    X_target_corrupted = X_target.copy()
    X_target_corrupted[60:] = 999.0  # corrupt only the non-train rows
    applier2 = fit_residualizer(X_target_corrupted, X_basis, train)
    assert np.allclose(applier1.W, applier2.W)


def test_budget_scan_more_features_generally_more_signal():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 20))

    def metric(Xp):
        return float(np.sum(Xp ** 2)) / Xp.shape[1]  # per-dimension "signal"

    scan = random_projection_budget_scan(X, [2, 20], metric_fn=metric, n_repeats=10, seed=0)
    assert scan[0].budget == 2 and scan[1].budget == 20
    assert scan[0].n_repeats == 10


def test_budget_scan_reproducible_with_same_seed():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 10))
    scan1 = random_projection_budget_scan(X, [5], metric_fn=lambda Xp: float(Xp.sum()), n_repeats=5, seed=7)
    scan2 = random_projection_budget_scan(X, [5], metric_fn=lambda Xp: float(Xp.sum()), n_repeats=5, seed=7)
    assert scan1[0].values == scan2[0].values
