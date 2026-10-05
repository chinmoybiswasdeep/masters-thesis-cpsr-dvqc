"""Classical side of notebook 5: datasets, causal windows, train-only
kernel/Koopman lifts, EDMD diagnostics, validation-selected ridge, the
memory x nonlinearity response surfaces, QRC capacity estimators (MC, IPC),
classical controls and paired block-bootstrap statistics.

Nothing in this module simulates the quantum reservoir; QRC feature matrices
come exclusively from `klqrc_quantum.run_qrc_jobs` (Qiskit Aer).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.polynomial import legendre as npleg
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

ALPHAS = tuple(float(a) for a in np.logspace(-6, 3, 10))
MAX_POLY_FEATURES = 600

# =============================================================================
# Metrics
# =============================================================================


def nrmse(pred, y) -> float:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    return float(np.sqrt(np.mean((pred - y) ** 2)) / (np.std(y) + 1e-12))


def all_metrics(pred, y) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    mse = float(np.mean((pred - y) ** 2))
    with np.errstate(invalid='ignore'):
        r = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else float('nan')
    return {'nrmse': nrmse(pred, y), 'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(pred - y))),
            'r2': float(1 - mse / (np.var(y) + 1e-12)), 'pearson': r}


# =============================================================================
# Datasets
# =============================================================================

VOLTERRA_TERMS = ((12,), (4, 9), (2, 7, 11))
VOLTERRA_COEFS = (0.30, 0.25, 0.20)
VOLTERRA_AR = 0.10
# Var(y) for u ~ U(-1,1) i.i.d.: sum_k c_k^2 (1/3)^{|term_k|} / (1 - a^2)
VOLTERRA_STD = float(np.sqrt(sum(c ** 2 * (1 / 3) ** len(t) for c, t in zip(VOLTERRA_COEFS, VOLTERRA_TERMS))
                             / (1 - VOLTERRA_AR ** 2)))


def generate_volterra(n: int, seed, noise_frac: float = 0.05, burn: int = 300):
    """u_t ~ U(-1,1) i.i.d.;  y_t = 0.30 u_{t-12} + 0.25 u_{t-4}u_{t-9}
    + 0.20 u_{t-2}u_{t-7}u_{t-11} + 0.10 y_{t-1};  observed target
    y_obs = y + noise_frac*Std(y)*N(0,1) (measurement noise, not fed back).
    Returns (u, y_clean, y_obs) after discarding `burn` samples."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    rng_u, rng_e = (np.random.default_rng(s) for s in ss.spawn(2))
    T = n + burn
    u = rng_u.uniform(-1, 1, T)
    y = np.zeros(T)
    for t in range(12, T):
        y[t] = (0.30 * u[t - 12] + 0.25 * u[t - 4] * u[t - 9] + 0.20 * u[t - 2] * u[t - 7] * u[t - 11]
                + 0.10 * y[t - 1])
    y_obs = y + noise_frac * VOLTERRA_STD * rng_e.standard_normal(T)
    return u[burn:], y[burn:], y_obs[burn:]


def volterra_exact_lift(u: np.ndarray, idx: np.ndarray, terms=VOLTERRA_TERMS) -> np.ndarray:
    """phi*_t = [prod_{k in term} u_{t-k}] for each term (generic)."""
    return np.stack([np.prod([u[idx - k] for k in term], axis=0) for term in terms], axis=1)


def volterra_exact_lift_manual(u: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The same lift typed out by hand (used to prove the generic one)."""
    return np.stack([u[idx - 12], u[idx - 4] * u[idx - 9], u[idx - 2] * u[idx - 7] * u[idx - 11]], axis=1)


def generate_mackey_glass(n: int, seed, beta=0.2, gamma=0.1, n_exp=10, tau=17.0, dt=0.1,
                          sample_every=30, transient=3000.0):
    """Mackey-Glass DDE, RK4 with step dt=0.1 (tau/dt = 170 history points;
    the delayed value at the RK half-step is the mean of its two neighbours).
    Initial history: constant x0 ~ U(0.5, 1.3) plus U(-0.05, 0.05) noise on
    [-tau, 0].  Samples every `sample_every` steps (Delta = 3 time units by
    default) after discarding `transient` time units."""
    rng = np.random.default_rng(seed)
    H = int(round(tau / dt))
    f = lambda x, xd: beta * xd / (1.0 + xd ** n_exp) - gamma * x
    n_trans = int(round(transient / dt))
    total = n_trans + n * sample_every + 1
    x = np.empty(total + H)
    x[:H + 1] = rng.uniform(0.5, 1.3) + rng.uniform(-0.05, 0.05, H + 1)
    for i in range(H, total + H - 1):
        xd0, xd1 = x[i - H], x[i - H + 1]
        xdm = 0.5 * (xd0 + xd1)
        xi = x[i]
        k1 = f(xi, xd0)
        k2 = f(xi + 0.5 * dt * k1, xdm)
        k3 = f(xi + 0.5 * dt * k2, xdm)
        k4 = f(xi + dt * k3, xd1)
        x[i + 1] = xi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    s = x[H + n_trans::sample_every][:n]
    assert len(s) == n and np.all(np.isfinite(s))
    return s


def generate_lorenz_x(n: int, seed, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, sample_every=10,
                      transient=60.0):
    """Lorenz-63, RK4 dt=0.01, observe x only every `sample_every` steps
    (Delta = 0.1).  Initial condition (x,y,z) = (0,0,25) + N(0, 8^2 I);
    `transient` time units discarded."""
    rng = np.random.default_rng(seed)
    X, Y, Z = rng.normal(0, 8, 3) + np.array([0.0, 0.0, 25.0])

    def f(x, y, z):
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    n_trans = int(round(transient / dt))
    out = np.empty(n)
    k = 0
    for i in range(n_trans + n * sample_every):
        a1 = f(X, Y, Z)
        a2 = f(X + 0.5 * dt * a1[0], Y + 0.5 * dt * a1[1], Z + 0.5 * dt * a1[2])
        a3 = f(X + 0.5 * dt * a2[0], Y + 0.5 * dt * a2[1], Z + 0.5 * dt * a2[2])
        a4 = f(X + dt * a3[0], Y + dt * a3[1], Z + dt * a3[2])
        X += dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        Y += dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        Z += dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        if i >= n_trans and (i - n_trans) % sample_every == 0:
            out[k] = X
            k += 1
    assert k == n and np.all(np.isfinite(out))
    return out


GENERATORS = {'mackey_glass': generate_mackey_glass, 'lorenz': generate_lorenz_x}
DATASET_IDS = {'volterra': 0, 'mackey_glass': 1, 'lorenz': 2}
SPLIT_IDS = {'train': 0, 'val': 1, 'test': 2}


def split_seed(dataset: str, data_seed: int, split: str):
    """Independent initial condition per (dataset, data seed, split)."""
    return np.random.SeedSequence([7919, DATASET_IDS[dataset], int(data_seed), SPLIT_IDS[split]])


@dataclass
class Trajectory:
    """One independent forecasting trajectory.

    Index conventions (raw-series coordinates):
      drive indices  t0 .. t_end-1   -- the QRC is driven with z_t on these
      eval indices   t0+W .. t_end-1 -- every model is scored on these
    with t0 = d_max-1 (first complete delay window), W = QRC washout and
    t_end = len(x) - h_max (every target x_{t+h} exists)."""
    x: np.ndarray
    d_max: int
    washout: int
    h_max: int
    dataset: str = ''
    split: str = ''
    data_seed: int = -1

    @property
    def t0(self):
        return self.d_max - 1

    @property
    def t_end(self):
        return len(self.x) - self.h_max

    @property
    def drive_idx(self):
        return np.arange(self.t0, self.t_end)

    @property
    def eval_idx(self):
        return np.arange(self.t0 + self.washout, self.t_end)

    def target(self, h: int, idx=None):
        idx = self.eval_idx if idx is None else idx
        return self.x[idx + h]


def make_trajectory(dataset: str, data_seed: int, split: str, n_usable: int, d_max: int, washout: int,
                    h_max: int) -> Trajectory:
    n_raw = d_max - 1 + washout + n_usable + h_max
    x = GENERATORS[dataset](n_raw, split_seed(dataset, data_seed, split))
    return Trajectory(x=x, d_max=d_max, washout=washout, h_max=h_max, dataset=dataset, split=split,
                      data_seed=data_seed)


def windows(x: np.ndarray, d: int, idx: np.ndarray) -> np.ndarray:
    """Causal delay vectors s_t = [x_t, x_{t-1}, ..., x_{t-d+1}] (rows)."""
    idx = np.asarray(idx)
    if idx.min() - (d - 1) < 0:
        raise IndexError('incomplete delay window requested')
    return np.stack([x[idx - k] for k in range(d)], axis=1)


def history_shuffled_windows(x: np.ndarray, d: int, idx: np.ndarray, seed) -> np.ndarray:
    """Time-permutation ablation: keep x_t, replace the past taps
    x_{t-1..t-d+1} by the past of a randomly permuted time index."""
    S = windows(x, d, idx)
    if d > 1:
        perm = np.random.default_rng(seed).permutation(len(idx))
        S[:, 1:] = S[perm, 1:]
    return S


# =============================================================================
# Ridge with validation-selected alpha (closed form, many targets at once)
# =============================================================================


@dataclass
class Ridge:
    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    b: np.ndarray
    alpha: np.ndarray

    def predict(self, X):
        P = ((np.asarray(X) - self.mu) / self.sd) @ self.W + self.b
        return P


def fit_ridge(Xtr, Ytr, Xva=None, Yva=None, alphas=ALPHAS, alpha=None) -> Ridge:
    """Standardise X on train stats, centre Y; choose alpha per target by
    validation MSE (or use the given `alpha`); fit on train only."""
    Xtr = np.asarray(Xtr, float)
    Ytr = np.asarray(Ytr, float)
    one = Ytr.ndim == 1
    Y = Ytr[:, None] if one else Ytr
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = (Xtr - mu) / sd
    ym = Y.mean(0)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    UtY = U.T @ (Y - ym)
    if alpha is not None:
        alist = [float(alpha)]
    else:
        alist = list(alphas)
    if len(alist) == 1:
        best = np.full(Y.shape[1], alist[0])
    else:
        Yv = np.asarray(Yva, float)
        Yv = Yv[:, None] if Yv.ndim == 1 else Yv
        XvV = ((np.asarray(Xva, float) - mu) / sd) @ Vt.T
        errs = np.empty((len(alist), Y.shape[1]))
        for a_i, a in enumerate(alist):
            P = XvV @ ((S / (S ** 2 + a))[:, None] * UtY) + ym
            errs[a_i] = np.mean((P - Yv) ** 2, axis=0)
        best = np.asarray(alist)[np.argmin(errs, axis=0)]
    W = np.empty((Xs.shape[1], Y.shape[1]))
    for a in np.unique(best):
        cols = best == a
        W[:, cols] = Vt.T @ ((S / (S ** 2 + a))[:, None] * UtY[:, cols])
    model = Ridge(mu=mu, sd=sd, W=W, b=ym, alpha=best)
    if one:
        model.W, model.b = model.W[:, 0], model.b[0]
    return model


def condition_number(X) -> float:
    Xs = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1.0, X.std(0))
    s = np.linalg.svd(Xs, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))


def effective_rank(X, tol=1e-10) -> int:
    Xs = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1.0, X.std(0))
    s = np.linalg.svd(Xs, compute_uv=False)
    return int(np.sum(s > tol * s[0]))


# =============================================================================
# Kernel / Koopman lifting (all fitted on training windows only)
# =============================================================================


@dataclass(frozen=True)
class LiftSpec:
    family: str          # 'raw' | 'linear' | 'rbf' | 'poly2' | 'poly3'
    d: int               # external delay depth (taps)
    m: int = 0           # RBF dictionary size (Nystrom-sampled centres)
    bw: float = 1.0      # RBF bandwidth multiplier of the median heuristic
    proj: str = 'pred'   # 'pred' (reduced-rank forecast of x_{t+1..t+10}) | 'pred1' (x_{t+1..t+p}) | 'pca'
    p: int = 1           # QRC input dimension

    @property
    def name(self):
        if self.family == 'raw':
            return f'raw(p={self.p})'
        tag = {'rbf': f'rbf[m={self.m},bw={self.bw:g}]'}.get(self.family, self.family)
        return f'{tag}|d={self.d}|{self.proj}|p={self.p}'

    @property
    def nonlinear(self):
        return self.family in ('rbf', 'poly2', 'poly3')

    def linear_counterpart(self) -> 'LiftSpec':
        """Dimension-matched linear delay embedding: same d, same projection, same p."""
        return replace(self, family='linear', m=0, bw=1.0)


def poly_feature_count(d: int, degree: int) -> int:
    from math import comb
    return comb(d + degree, degree) - 1


class Lifter:
    """Train-only causal lift: window -> standardise -> dictionary psi ->
    standardise -> projection to p coordinates -> MinMax to [0, 1]."""

    def __init__(self, spec: LiftSpec, seed: int = 0, H_proj: int = 10, proj_alpha: float = 1.0):
        self.spec, self.seed, self.H_proj, self.proj_alpha = spec, seed, H_proj, proj_alpha
        self.fit_hash = None
        self.n_fit = 0

    # -- dictionary ---------------------------------------------------------
    def _dict(self, S):
        Ss = (S - self.w_mu) / self.w_sd
        f = self.spec.family
        if f in ('linear', 'raw'):
            return Ss
        if f == 'rbf':
            d2 = ((Ss[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
            return np.exp(-self.gamma * d2)
        return self.poly.transform(Ss)

    def dictionary(self, x, idx):
        return self._dict(windows(x, self.spec.d, idx))

    def fit(self, x_train: np.ndarray, idx_train: np.ndarray, S_override=None):
        sp = self.spec
        S = windows(x_train, sp.d, idx_train) if S_override is None else S_override
        self.w_mu, self.w_sd = S.mean(0), np.where(S.std(0) < 1e-12, 1.0, S.std(0))
        rng = np.random.default_rng(self.seed)
        if sp.family == 'rbf':
            Ss = (S - self.w_mu) / self.w_sd
            uniq = np.unique(np.round(Ss, 12), axis=0)
            if len(uniq) < sp.m:
                raise ValueError('not enough distinct training windows for the RBF centres')
            self.centres = uniq[rng.choice(len(uniq), size=sp.m, replace=False)]
            sub = Ss[rng.choice(len(Ss), size=min(len(Ss), 800), replace=False)]
            dist = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
            med = np.median(dist[np.triu_indices(len(sub), 1)])
            self.gamma = sp.bw / (2.0 * med ** 2)
        elif sp.family in ('poly2', 'poly3'):
            deg = int(sp.family[-1])
            if poly_feature_count(sp.d, deg) > MAX_POLY_FEATURES:
                raise ValueError(f'{sp.family} at d={sp.d} exceeds MAX_POLY_FEATURES')
            self.poly = PolynomialFeatures(deg, include_bias=False).fit((S - self.w_mu) / self.w_sd)
        Psi = self._dict(S)
        if not np.all(np.isfinite(Psi)):
            raise FloatingPointError('non-finite dictionary')
        self.p_mu, self.p_sd = Psi.mean(0), np.where(Psi.std(0) < 1e-12, 1.0, Psi.std(0))
        Ps = (Psi - self.p_mu) / self.p_sd
        if sp.family == 'raw':
            self.proj_W = np.eye(sp.d)[:, :sp.p]
        elif sp.proj == 'pca':
            self.proj_W = PCA(n_components=sp.p, random_state=self.seed).fit(Ps).components_.T
        else:  # reduced-rank regression onto the future window x_{t+1..t+H}; 'pred1': H = p (one-step for p=1)
            H = sp.p if sp.proj == 'pred1' else self.H_proj
            F = np.stack([x_train[idx_train + k] for k in range(1, H + 1)], axis=1)
            Fc = (F - F.mean(0)) / F.std(0)
            B = np.linalg.solve(Ps.T @ Ps + self.proj_alpha * np.eye(Ps.shape[1]), Ps.T @ Fc)
            _, _, Vt = np.linalg.svd(Ps @ B, full_matrices=False)
            self.proj_W = B @ Vt[:sp.p].T
        Z = Ps @ self.proj_W
        # orientation: positively correlated with x_{t+1}
        sgn = np.sign([np.corrcoef(Z[:, j], x_train[idx_train + 1])[0, 1] or 1.0 for j in range(sp.p)])
        self.proj_W = self.proj_W * np.where(sgn == 0, 1.0, sgn)
        Z = Ps @ self.proj_W
        self.z_min, self.z_max = Z.min(0), Z.max(0)
        self.n_fit = len(idx_train)
        self.fit_hash = hash((x_train[idx_train].tobytes(), sp))
        return self

    def psi(self, x, idx, S_override=None):
        """Standardised full dictionary (kernel-only ridge, EDMD)."""
        S = windows(x, self.spec.d, idx) if S_override is None else S_override
        return (self._dict(S) - self.p_mu) / self.p_sd

    def transform(self, x, idx, S_override=None):
        """QRC input z in [0,1]^p and the fraction of clipped entries."""
        S = windows(x, self.spec.d, idx) if S_override is None else S_override
        Z = ((self._dict(S) - self.p_mu) / self.p_sd) @ self.proj_W
        Z = (Z - self.z_min) / np.where(self.z_max - self.z_min < 1e-12, 1.0, self.z_max - self.z_min)
        clipped = float(np.mean((Z < 0) | (Z > 1)))
        return np.clip(Z, 0.0, 1.0), clipped

    def gram_checks(self) -> dict:
        """RBF Gram-matrix pathologies on the centres."""
        if self.spec.family != 'rbf':
            return {}
        d2 = ((self.centres[:, None] - self.centres[None]) ** 2).sum(-1)
        K = np.exp(-self.gamma * d2)
        off = K[~np.eye(len(K), dtype=bool)]
        return {'gram_offdiag_mean': float(off.mean()), 'gram_offdiag_max': float(off.max()),
                'min_centre_distance': float(np.sqrt(d2[~np.eye(len(K), dtype=bool)].min())),
                'nearly_constant': bool(off.mean() > 0.99), 'nearly_diagonal': bool(off.max() < 1e-3),
                'duplicate_centres': bool(np.sqrt(d2[~np.eye(len(K), dtype=bool)].min()) < 1e-8)}


def raw_input(traj: Trajectory, x_min: float, x_max: float, p: int = 1):
    """Raw/current input [x_t, ..., x_{t-p+1}] MinMax-scaled with TRAIN stats."""
    S = windows(traj.x, p, traj.drive_idx)
    Z = (S - x_min) / (x_max - x_min)
    return np.clip(Z, 0, 1), float(np.mean((Z < 0) | (Z > 1)))


# =============================================================================
# EDMD / Koopman diagnostics on the dictionary
# =============================================================================


def edmd_metrics(lifter: Lifter, tr: Trajectory, va: Trajectory, horizons, k_roll: int = 10) -> dict:
    """Fit psi_{t+1} ~ A psi_t and x_t ~ C psi_t on TRAIN; report validation
    closure, k-step rollout, reconstruction and h-step forecast errors."""
    itr = tr.eval_idx[:-1]
    P0, P1 = lifter.psi(tr.x, itr), lifter.psi(tr.x, itr + 1)
    Am = fit_ridge(P0, P1, alpha=1e-3)
    Cm = fit_ridge(lifter.psi(tr.x, tr.eval_idx), tr.x[tr.eval_idx], alpha=1e-3)
    iva = va.eval_idx[:-k_roll]
    V0 = lifter.psi(va.x, iva)
    V1 = lifter.psi(va.x, iva + 1)
    rel = lambda E, T: float(np.sum(E ** 2) / (np.sum((T - T.mean(0)) ** 2) + 1e-12))
    closure = rel(Am.predict(V0) - V1, V1)
    cur = V0.copy()
    for _ in range(k_roll):
        cur = Am.predict(cur)
    roll = rel(cur - lifter.psi(va.x, iva + k_roll), lifter.psi(va.x, iva + k_roll))
    recon = all_metrics(Cm.predict(lifter.psi(va.x, va.eval_idx)), va.x[va.eval_idx])['r2']
    rho_A = float(np.max(np.abs(np.linalg.eigvals(Am.W.T))))
    out = {'closure_rel_err': closure, f'rollout{k_roll}_rel_err': roll, 'recon_r2': recon,
           'spectral_radius_A': rho_A}
    ive = va.eval_idx
    for h in horizons:
        cur = lifter.psi(va.x, ive)
        for _ in range(h):
            cur = Am.predict(cur)
        pred = Cm.predict(cur)
        out[f'edmd_nrmse_h{h}'] = nrmse(pred, va.target(h)) if np.all(np.isfinite(pred)) else float('inf')
    return out


class EDMDPredictor:
    """Koopman/EDMD linear predictor x_hat_{t+h} = C A^h psi_t (train-fitted)."""

    def __init__(self, lifter: Lifter, tr: Trajectory):
        itr = tr.eval_idx[:-1]
        self.lifter = lifter
        self.Am = fit_ridge(lifter.psi(tr.x, itr), lifter.psi(tr.x, itr + 1), alpha=1e-3)
        self.Cm = fit_ridge(lifter.psi(tr.x, tr.eval_idx), tr.x[tr.eval_idx], alpha=1e-3)

    def predict(self, traj: Trajectory, h: int):
        cur = self.lifter.psi(traj.x, traj.eval_idx)
        for _ in range(h):
            cur = self.Am.predict(cur)
        return self.Cm.predict(cur)


# =============================================================================
# Feature constructions for predictors
# =============================================================================


def lag_stack(Z: np.ndarray, n_lags: int) -> np.ndarray:
    """[z_t, z_{t-1}, ..., z_{t-n_lags+1}] for a (T, p) series (zero-padded)."""
    Z = np.asarray(Z)
    Z = Z[:, None] if Z.ndim == 1 else Z
    out = [Z]
    for k in range(1, n_lags):
        sh = np.zeros_like(Z)
        sh[k:] = Z[:-k]
        out.append(sh)
    return np.concatenate(out, axis=1)


class PolyExpansion:
    """Degree-q expansion used by the response surfaces: base features plus
    all degree-2..q monomials of the top-r train-PCA coordinates of the base
    (r = min(r_max, dim)).  Nested in q by construction."""

    def __init__(self, q: int, r_max: int = 10):
        self.q, self.r_max = q, r_max

    def fit(self, B):
        self.mu, self.sd = B.mean(0), np.where(B.std(0) < 1e-12, 1.0, B.std(0))
        if self.q > 1:
            r = min(self.r_max, B.shape[1])
            self.pca = PCA(n_components=r).fit((B - self.mu) / self.sd)
            self.pf = PolynomialFeatures(self.q, include_bias=False).fit(np.zeros((1, r)))
            self.keep = self.pf.powers_.sum(1) >= 2
        return self

    def transform(self, B):
        if self.q == 1:
            return B
        C = self.pca.transform((B - self.mu) / self.sd)
        C = C / np.where(C.std(0) < 1e-12, 1.0, C.std(0))
        return np.concatenate([B, self.pf.transform(C)[:, self.keep]], axis=1)


# =============================================================================
# Memory x nonlinearity response surfaces
# =============================================================================


def representation_features(rep: str, lift_spec: LiftSpec, L: int, tr: Trajectory, others, seed=0):
    """Base features of representation `rep` at causal history L, fitted on
    train only.  rep: 'raw' (L taps), 'linear' (train-PCA of the L taps,
    min(L, m) components) or 'lift' (selected kernel family on L-windows)."""
    if rep == 'raw':
        f = lambda t: windows(t.x, L, t.eval_idx)
        return f(tr), [f(o) for o in others]
    if rep == 'linear':
        S = windows(tr.x, L, tr.eval_idx)
        mu, sd = S.mean(0), np.where(S.std(0) < 1e-12, 1, S.std(0))
        k = min(L, max(lift_spec.m, 1)) if lift_spec.family == 'rbf' else L
        pca = PCA(n_components=min(k, L)).fit((S - mu) / sd)
        f = lambda t: pca.transform((windows(t.x, L, t.eval_idx) - mu) / sd)
        return f(tr), [f(o) for o in others]
    spec = replace(lift_spec, d=L)
    if spec.family in ('poly2', 'poly3') and poly_feature_count(L, int(spec.family[-1])) > MAX_POLY_FEATURES:
        return None, (None, None)      # cell skipped (never silently switch family)
    lf = Lifter(spec, seed=seed).fit(tr.x, tr.eval_idx)
    return lf.psi(tr.x, tr.eval_idx), [lf.psi(o.x, o.eval_idx) for o in others]


def response_surface(tr: Trajectory, va: Trajectory, ev: Trajectory, lift_spec: LiftSpec, L_grid, q_grid,
                     horizons, reps=('raw', 'linear', 'lift'), seed=0):
    """E_R(L, q) on trajectory `ev` (val during development, test after the
    lock).  Alpha is chosen on `va`; models are fitted on `tr` only.
    Returns {rep: {(L, q, h): (nrmse, predictions)}}."""
    out = {}
    for rep in reps:
        cells = {}
        for L in L_grid:
            Btr, (Bva, Bev) = representation_features(rep, lift_spec, L, tr, [va, ev], seed)
            if Btr is None:
                continue
            for q in q_grid:
                pe = PolyExpansion(q).fit(Btr)
                Xtr, Xva, Xev = pe.transform(Btr), pe.transform(Bva), pe.transform(Bev)
                Y = np.stack([tr.target(h) for h in horizons], 1)
                Yv = np.stack([va.target(h) for h in horizons], 1)
                model = fit_ridge(Xtr, Y, Xva, Yv)
                P = model.predict(Xev)
                for k, h in enumerate(horizons):
                    cells[(L, q, h)] = (nrmse(P[:, k], ev.target(h)), P[:, k])
        out[rep] = cells
    return out


def nl_gain(E, L, q_grid, h, eps=1e-6):
    e1 = E[(L, 1, h)][0]
    return (e1 - min(E[(L, q, h)][0] for q in q_grid if q > 1)) / (e1 + eps)


def memory_gain(E, q, L_grid, h, eps=1e-6):
    a, b = E[(min(L_grid), q, h)][0], E[(max(L_grid), q, h)][0]
    return (a - b) / (a + eps)


# =============================================================================
# QRC capacity estimators (features come from Aer; this is only the readout)
# =============================================================================


def legendre_targets(v: np.ndarray, degree: int, max_lag: int):
    """All products of normalised Legendre polynomials of total degree
    `degree` over variables (channel j, lag k <= max_lag) of an i.i.d.
    input v in [-1, 1]^p.  Returns (Y (T, n), descriptions)."""
    v = np.asarray(v)
    v = v[:, None] if v.ndim == 1 else v
    T, p = v.shape
    variables = [(j, k) for j in range(p) for k in range(max_lag + 1)]
    cols, descr = [], []
    for combo in itertools.combinations_with_replacement(variables, degree):
        y = np.ones(T)
        counts = {}
        for var in combo:
            counts[var] = counts.get(var, 0) + 1
        for (j, k), n in counts.items():
            s = np.zeros(T)
            s[k:] = v[:T - k, j]
            c = np.zeros(n + 1); c[n] = 1.0
            y = y * npleg.legval(s, c) * np.sqrt(2 * n + 1)
        cols.append(y)
        descr.append(tuple(sorted(counts.items())))
    return np.stack(cols, 1), descr


def _heldout_r2(X, Y, tr, va, te):
    m = fit_ridge(X[tr], Y[tr], X[va], Y[va])
    P = m.predict(X[te])
    Yt = Y[te]
    return 1 - np.mean((P - Yt) ** 2, 0) / (np.var(Yt, 0) + 1e-12)


def capacity_profile(X: np.ndarray, u: np.ndarray, washout: int, n_tr: int, n_va: int, n_te: int,
                     k_mc: int = 20, k2: int = 10, k3: int = 6, null_seed: int = 0, null_q: float = 99.0):
    """Linear memory capacity and degree-2/3 information-processing capacity
    of a QRC feature matrix X driven by i.i.d. u in [0,1]^p.

    MC_{j,k} = held-out R^2 of a linear readout of u_{j,t-k} (k = 0..k_mc);
    MC_total = sum clip(MC_{j,k}, 0, 1).  IPC_q sums clip(R^2, 0, 1) over all
    orthonormal Legendre products of total degree q, after zeroing
    capacities below a null threshold (99th percentile of the capacities of
    matching Legendre targets built from an independent sequence that never
    entered the reservoir).  Chronological train/val/test blocks with guard
    gaps larger than the longest lag."""
    u = np.asarray(u)
    u = u[:, None] if u.ndim == 1 else u
    v = 2 * u - 1
    gap = max(k_mc, k2, k3) + 1
    tr = np.arange(washout, washout + n_tr)
    va = np.arange(tr[-1] + 1 + gap, tr[-1] + 1 + gap + n_va)
    te = np.arange(va[-1] + 1 + gap, va[-1] + 1 + gap + n_te)
    if te[-1] >= len(X):
        raise ValueError('trajectory too short for the capacity split')
    Ymc = np.stack([np.r_[np.zeros(k), v[:len(v) - k, j]] for j in range(v.shape[1])
                    for k in range(k_mc + 1)], 1)
    mc = _heldout_r2(X, Ymc, tr, va, te).reshape(v.shape[1], k_mc + 1)
    Y2, d2 = legendre_targets(v, 2, k2)
    Y3, d3 = legendre_targets(v, 3, k3)
    c2, c3 = np.clip(_heldout_r2(X, Y2, tr, va, te), 0, 1), np.clip(_heldout_r2(X, Y3, tr, va, te), 0, 1)
    vn = 2 * np.random.default_rng(null_seed).random(v.shape) - 1
    N2, _ = legendre_targets(vn, 2, k2)
    N3, _ = legendre_targets(vn, 3, k3)
    null = np.clip(np.r_[_heldout_r2(X, N2, tr, va, te), _heldout_r2(X, N3, tr, va, te)], 0, 1)
    thr = float(np.percentile(null, null_q))
    return {'mc_per_jk': mc, 'mc_total': float(np.clip(mc, 0, 1).sum()), 'mc_total_unclipped': float(mc.sum()),
            'ipc2': float(c2[c2 > thr].sum()), 'ipc3': float(c3[c3 > thr].sum()),
            'ipc2_unthresholded': float(c2.sum()), 'ipc3_unthresholded': float(c3.sum()),
            'ipc_null_threshold': thr, 'n_ipc2_targets': len(d2), 'n_ipc3_targets': len(d3),
            'ipc2_per_target': c2, 'ipc3_per_target': c3, 'ipc2_descr': d2, 'ipc3_descr': d3}


# =============================================================================
# Controlled Volterra witness
# =============================================================================


def _volterra_features(u, idx, model, shuffle_seed=None):
    """Feature matrix of a witness model on indices idx (all lags <= 24)."""
    W = windows(u, 25, idx)
    if shuffle_seed is not None:                       # keep u_t, shuffle the past
        W[:, 1:] = W[np.random.default_rng(shuffle_seed).permutation(len(idx)), 1:]

    def lift(drop=None, K=3):
        cols = []
        for j in range(K):
            for k, term in enumerate(VOLTERRA_TERMS):
                if k != drop:
                    cols.append(np.prod([W[:, j + lag] for lag in term], axis=0))
        return np.stack(cols, 1)

    if model == 'raw_linear':
        return W
    if model in ('raw_deg3_oracle', 'oracle_history_shuffled'):
        return W[:, :15]
    if model == 'delay_truncated_deg3':
        return W[:, :8]
    if model in ('lift_linear', 'lift_history_shuffled'):
        return lift()
    if model.startswith('lift_drop_'):
        return lift(drop=int(model[-1]))
    raise KeyError(model)


VOLTERRA_MODELS = ('raw_linear', 'raw_deg3_oracle', 'lift_linear', 'lift_drop_0', 'lift_drop_1', 'lift_drop_2',
                   'delay_truncated_deg3', 'lift_history_shuffled', 'oracle_history_shuffled')


def volterra_witness(data: dict, eval_split: str, shuffle_seed: int = 0) -> dict:
    """data[split] = (u, y_clean, y_obs).  Each model is fitted on train
    (alpha on val) and scored on `eval_split` (common indices t >= 24)."""
    idx = {sp: np.arange(24, len(v[0])) for sp, v in data.items()}
    y = {sp: data[sp][2][idx[sp]] for sp in data}
    out = {}
    for model in VOLTERRA_MODELS:
        shuf = model.endswith('history_shuffled')
        F = {sp: _volterra_features(data[sp][0], idx[sp], model,
                                    shuffle_seed=(shuffle_seed, C_SPLIT[sp]) if shuf else None)
             for sp in ('train', 'val', eval_split)}
        if model in ('raw_deg3_oracle', 'oracle_history_shuffled', 'delay_truncated_deg3'):
            pf = PolynomialFeatures(3, include_bias=False).fit(F['train'])
            F = {sp: pf.transform(v) for sp, v in F.items()}
        m = fit_ridge(F['train'], y['train'], F['val'], y['val'])
        pred = m.predict(F[eval_split])
        out[model] = {'nrmse': nrmse(pred, y[eval_split]), 'n_features': F['train'].shape[1]}
    var_y = np.var(y[eval_split])
    sigma2 = (0.05 * VOLTERRA_STD) ** 2
    for k, (c, term) in enumerate(zip(VOLTERRA_COEFS, VOLTERRA_TERMS)):
        var_missing = c ** 2 * (1 / 3) ** len(term) / (1 - VOLTERRA_AR ** 2)
        out[f'lift_drop_{k}']['expected_nrmse'] = float(np.sqrt((var_missing + sigma2) / var_y))
    out['noise_floor_nrmse'] = float(np.sqrt(sigma2 / var_y))
    u = data[eval_split][0]
    out['exact_lift_max_abs_diff'] = float(np.max(np.abs(volterra_exact_lift(u, idx[eval_split])
                                                         - volterra_exact_lift_manual(u, idx[eval_split]))))
    return out


C_SPLIT = {'train': 0, 'val': 1, 'test': 2}

# =============================================================================
# nb1 Section 0b utilities, verbatim (used for the nb4 regression check)
# =============================================================================


def nb1_random_input(T: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).uniform(0, 1, T)


def nb1_task_narma2(u: np.ndarray, u_scale: float = 0.5) -> np.ndarray:
    T = len(u)
    y = np.zeros(T)
    us = u_scale * u
    for t in range(1, T):
        y_prev2 = y[t - 2] if t >= 2 else 0.0
        y[t] = 0.4 * y[t - 1] + 0.4 * y[t - 1] * y_prev2 + 0.6 * us[t] ** 3 + 0.1
    return y


def nb1_chrono_split(T: int, washout: int, n_val: int, n_test: int, gap: int):
    test = np.arange(T - n_test, T)
    val_end = T - n_test - gap
    val = np.arange(val_end - n_val, val_end)
    train_end = val_end - n_val - gap
    train = np.arange(washout, train_end)
    if len(train) < 10:
        raise ValueError('Not enough training points')
    return train, val, test


def nb1_nrmse(y_pred, y_true) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2) / (np.var(y_true) + 1e-12)))


NB1_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def nb1_select_and_eval_ridge(X, y, train, val, test, alphas=NB1_ALPHAS):
    from sklearn.linear_model import Ridge as SkRidge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[train])
    Xtr, Xval, Xte = scaler.transform(X[train]), scaler.transform(X[val]), scaler.transform(X[test])
    best_alpha, best_val_err = alphas[0], np.inf
    for a in alphas:
        model = SkRidge(alpha=a).fit(Xtr, y[train])
        err = nb1_nrmse(model.predict(Xval), y[val])
        if err < best_val_err:
            best_val_err, best_alpha = err, a
    model = SkRidge(alpha=best_alpha).fit(Xtr, y[train])
    return nb1_nrmse(model.predict(Xte), y[test]), best_alpha, model


# =============================================================================
# Classical controls
# =============================================================================


def esn_features(z: np.ndarray, n: int, spectral_radius: float, input_scale: float, seed: int,
                 leak: float = 1.0) -> np.ndarray:
    """State-dimension-matched echo state network (tanh), input 2z-1."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z)
    z = z[:, None] if z.ndim == 1 else z
    W = rng.standard_normal((n, n))
    W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.uniform(-1, 1, (n, z.shape[1])) * input_scale
    b = rng.uniform(-0.2, 0.2, n)
    s = np.zeros(n)
    out = np.empty((len(z), n))
    for t in range(len(z)):
        s = (1 - leak) * s + leak * np.tanh(W @ s + Win @ (2 * z[t] - 1) + b)
        out[t] = s
    return out


# =============================================================================
# Paired moving-block bootstrap
# =============================================================================


def block_indices(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def paired_block_bootstrap(y_true: dict, pred_a: dict, pred_b: dict, block: int, n_boot: int, seed: int):
    """Relative improvement of model B over model A,
        Delta = (E_A - E_B) / E_A,  E = median over units of NRMSE,
    where a unit is (data seed, reservoir seed) and y_true is keyed by data
    seed.  The SAME block-resampled time indices are used for every unit of a
    data seed and for both models (paired).  Returns point estimate, 95% CI,
    bootstrap draws."""
    def stat(idx_by_seed):
        ea, eb = [], []
        for key in pred_a:
            ds = key[0]
            ix = idx_by_seed[ds]
            ea.append(nrmse(pred_a[key][ix], y_true[ds][ix]))
            eb.append(nrmse(pred_b[key][ix], y_true[ds][ix]))
        Ea, Eb = np.median(ea), np.median(eb)
        return (Ea - Eb) / Ea
    full = {ds: np.arange(len(y)) for ds, y in y_true.items()}
    point = stat(full)
    rng = np.random.default_rng(seed)
    draws = np.array([stat({ds: block_indices(len(y), block, rng) for ds, y in y_true.items()})
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'delta': float(point), 'ci95': [float(lo), float(hi)], 'p_le_0': float(np.mean(draws <= 0)),
            'draws': draws}


def error_autocorr_length(err: np.ndarray, thresh: float = 0.1) -> int:
    """First lag where the autocorrelation of |error| drops below `thresh`."""
    e = np.abs(err) - np.mean(np.abs(err))
    ac = np.correlate(e, e, 'full')[len(e) - 1:]
    ac = ac / (ac[0] + 1e-300)
    below = np.where(ac < thresh)[0]
    return int(below[0]) if len(below) else len(e)
