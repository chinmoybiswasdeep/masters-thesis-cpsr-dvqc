"""
v4_nonlinear_memory.py -- the Sunada nonlinear-memory benchmark.

Sunada et al., Sci. Rep. 9, 19078 (2019), doi:10.1038/s41598-019-55247-y

    y_{nu,tau}(t) = sin(nu * u_{t-tau}) / nu

At FIXED nu, sweeping tau changes the memory demand without changing the
nonlinear transformation at all; at fixed tau, sweeping nu changes the
nonlinearity without changing the memory demand. That factorisation is exactly
what a separation claim needs, which is why this benchmark is reported
alongside IPC rather than instead of it.

nu = 0 uses the ANALYTICAL limit lim_{nu->0} sin(nu u)/nu = u, not a small-nu
numerical approximation.

Score: NM_nu(tau) = corr^2(y, yhat) on the held-out block.

NOTE ON WHAT THIS BENCHMARK CAN AND CANNOT SHOW HERE. For nu > 0 the target is
a genuinely nonlinear function of a SINGLE delayed input, so for tau > 0 it
has degree >= 2 at a strictly positive delay. The V4 product architecture
provably cannot represent that (see `v4_encoder.reachable_monomial`). These
curves therefore measure and DISPLAY the capability boundary; they are
reported as a characterised limitation and are never placed inside a gate.
The (nu > 0, tau = 0) column and the (nu = 0, any tau) row are both reachable
and are the informative parts of the grid.
"""
from __future__ import annotations

import numpy as np

from .v4_ipc import BoundedReadout


def sunada_target(u: np.ndarray, nu: float, tau: int) -> np.ndarray:
    """y = sin(nu * u_{t-tau}) / nu, with the exact nu -> 0 limit."""
    u = np.asarray(u, dtype=float)
    T = len(u)
    shifted = np.zeros(T, dtype=float)
    if tau == 0:
        shifted[:] = u
    else:
        shifted[tau:] = u[:T - tau]
    y = shifted.copy() if float(nu) == 0.0 else np.sin(float(nu) * shifted) / float(nu)
    if tau:
        y[:tau] = 0.0
    return y


def corr2(pred: np.ndarray, y: np.ndarray) -> float:
    """corr^2, the Sunada score. Distinct from the IPC 1 - MSE/Var."""
    pred, y = np.asarray(pred, dtype=float), np.asarray(y, dtype=float)
    if np.std(pred) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(pred, y)[0, 1] ** 2, 0.0, 1.0))


def _predict(readout: BoundedReadout, y: np.ndarray) -> np.ndarray:
    ytr = y[readout.train]
    ybar = float(ytr.mean())
    filt = readout._s / (readout._s ** 2 + readout.alpha)
    w = readout._Vt.T @ (filt * (readout._U.T @ (ytr - ybar)))
    return readout._Xte @ w + ybar


def nonlinear_memory_curves(run, cfg, *, nus=(0.0, 0.5, 1.0, 2.0, 4.0),
                            tau_max: int = None, feature_set: str = "all") -> dict:
    """NM_nu(tau) over the preregistered (nu, tau) grid.

    `feature_set` selects which readout the benchmark is run on: 'all' is the
    complete operational readout (the honest end-to-end number), while 'R' and
    'P' isolate a single route.
    """
    u = np.asarray(run.u, dtype=float)
    T = len(u)
    train, test = cfg.splits(T)
    X = {"all": run.X_all, "R": run.X_R, "P": run.X_P, "J": run.X_J}[feature_set]
    if X.size == 0:
        return {"feature_set": feature_set, "curves": {}, "note": "no features"}
    rd = BoundedReadout(X, train, test, cfg.alpha)
    tau_max = int(cfg.max_delay if tau_max is None else tau_max)

    curves, reach = {}, {}
    for nu in nus:
        row = {}
        for tau in range(tau_max + 1):
            y = sunada_target(u, nu, tau)
            row[tau] = corr2(_predict(rd, y), y[rd.test])
        curves[float(nu)] = row
        # reachable iff linear (nu == 0) or evaluated at delay 0
        reach[float(nu)] = {tau: bool(float(nu) == 0.0 or tau == 0)
                            for tau in range(tau_max + 1)}
    return {"feature_set": feature_set, "nus": [float(n) for n in nus],
            "tau_max": tau_max, "curves": curves, "reachable": reach,
            "note": ("for nu > 0 and tau > 0 the target has degree >= 2 at a strictly "
                     "positive delay and is provably unreachable by this architecture; "
                     "those cells display the capability boundary and are never gated")}


def nm_summary(curves: dict) -> dict:
    """Reachable-cell summary, kept separate from the unreachable cells."""
    cur = curves.get("curves", {})
    reach = curves.get("reachable", {})
    ok, boundary = [], []
    for nu, row in cur.items():
        for tau, v in row.items():
            (ok if reach.get(nu, {}).get(tau, False) else boundary).append(float(v))
    return {"n_reachable": len(ok), "mean_reachable": float(np.mean(ok)) if ok else float("nan"),
            "n_boundary": len(boundary),
            "mean_boundary": float(np.mean(boundary)) if boundary else float("nan"),
            "max_boundary": float(np.max(boundary)) if boundary else float("nan")}
