"""
feature_analysis.py -- Part 5 of the V2 validation spec: fair feature-group
comparison. The prior pass compared `dim(X_M)=3` against `dim(X_P)=375`
directly, which is not a fair comparison. This module provides:

  - numerical / effective rank diagnostics,
  - ridge effective-degrees-of-freedom (SVD-based, no explicit inverse),
  - leakage-safe residualization (projection fit on TRAIN ONLY, applied
    unchanged to val/test),
  - seeded, rank-matched random-projection feature-budget scans.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def numerical_rank(X: np.ndarray, tol: float = None) -> int:
    """SVD-based numerical rank (`np.linalg.matrix_rank`, stable, no
    explicit inverse/determinant)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(Xc, tol=tol))


def effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    """Roy-Vetterli effective rank: exp(entropy of the normalized squared
    singular-value spectrum) -- a continuous, less step-function-like
    alternative to the numerical (hard-threshold) rank above."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2
    total = p.sum()
    if total <= eps:
        return 0.0
    p = p / total
    p = p[p > eps]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def ridge_effective_dof(X: np.ndarray, alpha: float) -> float:
    """sum_i s_i^2 / (s_i^2 + alpha) for singular values s_i of (centered)
    X -- the standard ridge-regression effective-degrees-of-freedom
    quantity, computed via SVD (never an explicit (X^T X + alpha I)^-1)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return float(np.sum(s ** 2 / (s ** 2 + alpha)))


@dataclass
class FeatureGroupDiagnostics:
    n_features: int
    numerical_rank: int
    effective_rank: float
    ridge_dof: float


def diagnose_feature_group(X: np.ndarray, ridge_alpha: float = 1.0) -> FeatureGroupDiagnostics:
    return FeatureGroupDiagnostics(
        n_features=X.shape[1], numerical_rank=numerical_rank(X), effective_rank=effective_rank(X),
        ridge_dof=ridge_effective_dof(X, ridge_alpha),
    )


# =============================================================================
# Leakage-safe residualization: X_P_perp = (I - P_{X_M}) X_P
# =============================================================================

def fit_residualizer(X_target: np.ndarray, X_basis: np.ndarray, train_idx: np.ndarray):
    """Fit W = argmin_W ||X_target[train] - X_basis[train] @ W||_F via
    `np.linalg.lstsq` (never an explicit `pinv`/matrix inverse) on TRAIN
    rows only. Returns a callable that applies the SAME fitted W to any
    (train/val/test) rows -- the projection is a fixed linear map once
    fit, so applying it to held-out data introduces no leakage (the
    held-out TARGETS are never involved in fitting `W`, only the basis and
    target FEATURE values at the training timesteps are)."""
    W, *_ = np.linalg.lstsq(X_basis[train_idx], X_target[train_idx], rcond=None)

    def apply(X_target_any: np.ndarray, X_basis_any: np.ndarray) -> np.ndarray:
        return X_target_any - X_basis_any @ W

    apply.W = W
    return apply


def residualize(X_target: np.ndarray, X_basis: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """One-shot convenience: fit on `train_idx`, apply to ALL rows of
    `X_target`. For repeated use (e.g. across train/val/test slices with
    the SAME fitted map), call `fit_residualizer` once instead."""
    applier = fit_residualizer(X_target, X_basis, train_idx)
    return applier(X_target, X_basis)


# =============================================================================
# Rank-matched random-projection feature-budget scan
# =============================================================================

@dataclass
class BudgetScanPoint:
    budget: int
    n_repeats: int
    values: list          # one value (e.g. M or NL) per repeat
    mean: float
    std: float
    ci_lo: float
    ci_hi: float


def random_projection_budget_scan(X: np.ndarray, budgets: Sequence[int], metric_fn: Callable[[np.ndarray], float],
                                   n_repeats: int = 20, seed: int = 0) -> list:
    """For each `budget` K (capped at `X.shape[1]`), draw `n_repeats`
    independent seeded random K-dimensional projections of `X`
    (`rng.standard_normal((n_features, K)) / sqrt(K)`, a standard
    Johnson-Lindenstrauss-style random projection so the projected
    features stay on a comparable scale regardless of K), evaluate
    `metric_fn(X_projected) -> float` (typically a capacity computation)
    on each, and report mean/std/95% percentile CI across repeats.
    `metric_fn` is caller-supplied so this module stays independent of how
    capacity is actually computed (reuses `ipc.compute_ipc`/
    `ipc_decomposition.compute_ipc_decomposed` at the call site, not
    reimplemented here)."""
    rng = np.random.RandomState(seed)
    n_features = X.shape[1]
    out = []
    for K in budgets:
        K_eff = min(K, n_features)
        vals = []
        for r in range(n_repeats):
            proj = rng.standard_normal((n_features, K_eff)) / np.sqrt(K_eff)
            X_proj = X @ proj
            vals.append(float(metric_fn(X_proj)))
        vals = np.asarray(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > 1 else (vals[0], vals[0])
        out.append(BudgetScanPoint(budget=K, n_repeats=n_repeats, values=vals.tolist(),
                                    mean=float(vals.mean()), std=float(vals.std()),
                                    ci_lo=float(lo), ci_hi=float(hi)))
    return out
