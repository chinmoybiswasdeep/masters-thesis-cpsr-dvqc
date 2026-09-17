"""
tasks.py -- Part 12 task benchmarks, new to this repo (the audit found NO
NARMA10/Mackey-Glass/Lorenz anywhere in the existing codebase --
`qrc_qiskit.py` only has k-Pauli/NARMA2/memory-capacity). All tasks here are
evaluated with the SAME Ridge readout the rest of the repo uses
(`qrc_qiskit.chrono_split` + `select_and_eval_ridge`) -- no MLP for primary
claims (Part 12's explicit requirement: a nonlinear classical readout would
obscure whether computational power came from the reservoir or the readout).

`u` throughout is the reservoir's own [0,1] input convention (what actually
gets Ry(pi*u)-encoded); chaotic-system tasks (Mackey-Glass, Lorenz) generate
their own driving sequence and rescale it to [0,1] before it is ever handed
to a reservoir builder.
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# NARMA10 (literature-standard, absent from this repo before now)
# =============================================================================

def task_narma10(u: np.ndarray, u_scale: float = 0.5) -> np.ndarray:
    """Standard 10th-order NARMA (Atiya & Parlos 2000 / widely used RC
    benchmark):
        y_t = 0.3 y_{t-1} + 0.05 y_{t-1} sum_{i=1}^{10} y_{t-i}
              + 1.5 (u_scale*u_{t-10})(u_scale*u_{t-1}) + 0.1
    `u_scale` follows this repo's own `task_narma2` convention (Appeltant et
    al. 2011): the reservoir sees the original unscaled u_t; only the target
    is defined on a rescaled copy, which changes nothing the reservoir can
    learn since the two are a fixed monotonic rescaling of each other.
    """
    T = len(u)
    y = np.zeros(T)
    us = u_scale * u
    for t in range(10, T):
        y[t] = (0.3 * y[t - 1] + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
                + 1.5 * us[t - 10] * us[t - 1] + 0.1)
        y[t] = np.clip(y[t], -1e3, 1e3)  # guard against the known rare NARMA10 blow-up
    return y


# =============================================================================
# Mackey-Glass (chaotic delay-differential system) -- one-step prediction
# =============================================================================

def generate_mackey_glass(T: int, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                           n: float = 10.0, dt: float = 1.0, x0: float = 1.2,
                           washout: int = 500) -> np.ndarray:
    """Discrete-time Euler integration of the Mackey-Glass DDE
        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    at unit step dt=1 (the standard discretized-MG-for-RC convention), from a
    constant history x(t<=0)=x0. `washout` extra steps are generated and
    discarded so the returned series has left the transient and sits on the
    (chaotic, for tau=17) attractor."""
    total = T + washout + tau + 1
    x = np.full(total, x0, dtype=float)
    for t in range(1, total):
        x_tau = x[t - tau] if t - tau >= 0 else x0
        x[t] = x[t - 1] + dt * (beta * x_tau / (1 + x_tau ** n) - gamma * x[t - 1])
    return x[washout + tau:washout + tau + T]


def mackey_glass_input_and_target(T: int, **mg_kwargs):
    """Returns (u, y): `u` is the MG series rescaled to [0,1] (this
    reservoir's own encoding convention), `y = u` shifted so `y[t] =
    u[t+1]` -- i.e. the standard one-step-ahead forecast target. The LAST
    sample has no known next value and is dropped by the caller's chrono
    split washout/gap, not specially handled here."""
    x = generate_mackey_glass(T + 1, **mg_kwargs)
    lo, hi = x.min(), x.max()
    u_full = (x - lo) / (hi - lo + 1e-12)
    u = u_full[:-1]
    y = u_full[1:]
    return u, y


# =============================================================================
# Lorenz (chaotic ODE) -- one-step prediction, one-step cross-prediction
# =============================================================================

def generate_lorenz(T: int, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0,
                     dt: float = 0.01, subsample: int = 10, state0=(1.0, 1.0, 1.0),
                     washout: int = 1000) -> np.ndarray:
    """RK4 integration of the Lorenz system, subsampled every `subsample`
    integration steps (standard practice: dt=0.01 internal step for RK4
    accuracy, subsampled to a coarser effective sampling interval so
    consecutive samples aren't near-identical). Returns an (T, 3) array of
    (x, y, z), past an initial `washout` (in OUTPUT samples) transient."""
    def deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    n_steps = (T + washout) * subsample
    s = np.array(state0, dtype=float)
    out = np.empty((T + washout, 3))
    for i in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % subsample == 0:
            out[i // subsample] = s
    return out[washout:washout + T]


def lorenz_input_and_targets(T: int, **lorenz_kwargs):
    """Returns (u, y_self, y_cross): `u` is the Lorenz x-coordinate rescaled
    to [0,1] (the driving/encoded signal); `y_self[t] = u[t+1]`
    (one-step self-prediction); `y_cross[t] = normalized_y[t+1]` (one-step
    CROSS-prediction -- forecasting the y-coordinate from a reservoir driven
    only by x, Part 12's 'if feasible' cross-prediction task)."""
    xyz = generate_lorenz(T + 1, **lorenz_kwargs)
    x, y = xyz[:, 0], xyz[:, 1]
    x_lo, x_hi = x.min(), x.max()
    y_lo, y_hi = y.min(), y.max()
    u_full = (x - x_lo) / (x_hi - x_lo + 1e-12)
    y_full = (y - y_lo) / (y_hi - y_lo + 1e-12)
    u = u_full[:-1]
    y_self = u_full[1:]
    y_cross = y_full[1:]
    return u, y_self, y_cross


# =============================================================================
# Delayed-product and mixed memory+nonlinear tasks (Part 12's own examples)
# =============================================================================

def task_delayed_product(u: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """y_t = u_{t-d1} * u_{t-d2} -- a pure nonlinear-memory-mixing task."""
    T = len(u)
    y = np.zeros(T)
    dmax = max(d1, d2)
    for t in range(dmax, T):
        y[t] = u[t - d1] * u[t - d2]
    return y


def task_mixed_memory_nonlinear(u: np.ndarray) -> np.ndarray:
    """y_t = 0.5 u_{t-8} + 0.5 u_{t-2} u_{t-5} -- Part 12's headline task:
    requires BOTH long memory (the u_{t-8} term) AND nonlinear mixing (the
    u_{t-2}*u_{t-5} term) simultaneously, which is exactly the regime a
    monolithic single-resource reservoir's memory-nonlinearity trade-off
    should make hardest, and exactly what DQRC's architectural separation is
    meant to help with."""
    T = len(u)
    y = np.zeros(T)
    for t in range(8, T):
        y[t] = 0.5 * u[t - 8] + 0.5 * u[t - 2] * u[t - 5]
    return y


TASK_REGISTRY = {
    "narma10": {"max_lag": 10},
    "mackey_glass": {"max_lag": 1},
    "lorenz_self": {"max_lag": 1},
    "lorenz_cross": {"max_lag": 1},
    "delayed_product_2_5": {"max_lag": 5},
    "mixed_memory_nonlinear": {"max_lag": 8},
}
