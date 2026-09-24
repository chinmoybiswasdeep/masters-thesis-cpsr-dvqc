"""
v5_architecture.py -- a dual-register design built to pass the V4 audit's
INTRINSIC (scale-invariant) tests, not only its fixed-ridge tests.

Why V4 failed them, and what V5 changes
---------------------------------------------------------------------------
1. V4's nonlinear control was a ridge artifact. With 12 processor observables
   spanning every polynomial of degree <= 4, the unregularised capacity was
   1.0 for every g >= 1e-4 (identical subspace, principal angle 0.000 deg).
   V5 reads ONE processor observable. For a single feature f = sum_d a_d P_d,
   unregularised capacity is C_d = a_d^2 / sum a^2, so N measures the FRACTION
   of the feature's variance that is nonlinear. That is scale invariant, cannot
   saturate by span, and moves continuously with g. (With >= 2 features the
   linear parts cancel and N jumps at g = 0+; one feature is the only way to
   avoid the jump.)

2. V4's memory route leaked nonlinearity. Its two-body readout (Z_a Z_b) made
   the memory state's readout MULTILINEAR, so the memory route alone computed
   old x old products and the high-m/high-g quadrant could not beat
   high-m/low-g on them. V5's memory route is EXACTLY LINEAR in past inputs:
   a random-SWAP channel acts linearly on single-site Z marginals, and only
   single-site Z is read.

3. The joint layer never sees the injection rail, so no joint feature can
   form u_t * u_t: the operational observable set carries no encoder-only
   nonlinearity.

Memory route R (control m)
    rails 0..L. Per step: for r = L..1 apply rho -> (1-p) rho + p SWAP_{r-1,r}
    rho SWAP_{r-1,r} (a random-unitary channel), then reset rail 0 and inject
    rho_in(u_t) = (I + u_t Z)/2 once, then read <Z_1..Z_L>. p = m * p_max.
    SWAP exchanges the two marginals, so the channel maps them linearly:
        z_{r-1} <- (1-p) z_{r-1} + p z_r ,  z_r <- (1-p) z_r + p z_{r-1}
    `memory_features` uses this exact recursion; `memory_features_dm` runs the
    full density matrix and the tests assert the two agree.

Processor route P (control g)
    two copies rho(u_t) (x) rho(u_t), reset every step. Ry(pi/2) on copy 0,
    then exp(-i theta/2 Z0 Z1) with theta = g * theta_max, then measure
    cos(phi) X0 + sin(phi) Y0. Exactly:
        f(u) = cos(phi) cos(theta) u + sin(phi) sin(theta) u^2
    theta_max < pi/2 / 1.1 keeps the linear coefficient away from zero for
    +10% perturbations, so N(g) is monotone with no fragile zero crossing.

The copies ARE the nonlinear resource (2 input copies, fixed for every g).
There is no architecture disorder: seeds change only the input sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.diag([1.0, -1.0]).astype(complex)


@dataclass(frozen=True)
class V5Spec:
    L: int = 6
    p_max: float = 0.9
    theta_max: float = 0.44 * np.pi
    phi: float = np.pi / 4
    read_stride: int = 1          # V5.3: read rails stride, 2*stride, ..., L only

    def __post_init__(self):
        if self.L < 1:
            raise ValueError("L must be >= 1")
        if not 0 < self.p_max <= 1:
            raise ValueError("p_max must be in (0, 1]")
        if not 0 < self.theta_max < np.pi / 2:
            raise ValueError("theta_max must be in (0, pi/2)")
        if not 1 <= self.read_stride <= self.L:
            raise ValueError("read_stride must be in [1, L]")

    @property
    def read_rails(self) -> list:
        """1-based rail indices that are measured."""
        return list(range(self.read_stride, self.L + 1, self.read_stride))

    def as_dict(self) -> dict:
        return {"L": self.L, "p_max": self.p_max, "theta_max": self.theta_max,
                "theta_max_over_pi": self.theta_max / np.pi, "phi": self.phi,
                "read_stride": self.read_stride, "read_rails": self.read_rails,
                "memory": "random-SWAP channel, single-site Z readout of the read rails",
                "processor": "2 copies, Ry(pi/2), exp(-i theta Z0Z1/2), measure cos(phi)X0+sin(phi)Y0",
                "joint": "z_r * f for every read rail r (injection rail excluded)"}

    def resources(self) -> dict:
        k = len(self.read_rails)
        return {"qubits": self.L + 1 + 2, "input_copies": 3, "observables_R": k,
                "observables_P": 1, "observables_joint": k,
                "observables_total": 2 * k + 1, "processor_depth": 3,
                "nonlinear_degree_R": 1, "nonlinear_degree_P": 2}


# =============================================================================
# Memory
# =============================================================================
def memory_features(u, p: float, L: int, *, g_leak: float = 0.0, g: float = 0.0) -> np.ndarray:
    """Exact <Z_1..Z_L> of the random-SWAP register (marginal recursion).

    `g_leak` exists ONLY for the contaminated negative control."""
    p = float(np.clip(p * (1.0 + g_leak * g), 0.0, 1.0))
    z = np.ones(L + 1)            # register starts in |0...0>, i.e. <Z> = +1 on every rail
    out = np.empty((len(u), L))
    for t, ut in enumerate(u):
        for r in range(L, 0, -1):
            a, b = z[r - 1], z[r]
            z[r - 1] = (1 - p) * a + p * b
            z[r] = (1 - p) * b + p * a
        z[0] = float(ut)
        out[t] = z[1:]
    return out


def memory_features_dm(u, p: float, L: int) -> np.ndarray:
    """Same register as a full density matrix, for verification (small L)."""
    n = L + 1
    D = 2 ** n
    rho = np.zeros((D, D), dtype=complex)
    rho[0, 0] = 1.0

    def swap_perm(a, b):
        idx = np.arange(D)
        bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
        bits[:, [a, b]] = bits[:, [b, a]]
        return (bits * (1 << (n - 1 - np.arange(n)))).sum(axis=1)

    perms = {r: swap_perm(r - 1, r) for r in range(1, L + 1)}
    Zs = [np.diag([1.0 if not ((i >> (n - 1 - q)) & 1) else -1.0 for i in range(D)])
          for q in range(1, n)]
    out = np.empty((len(u), L))
    for t, ut in enumerate(u):
        for r in range(L, 0, -1):
            pm = perms[r]
            rho = (1 - p) * rho + p * rho[np.ix_(pm, pm)]
        red = np.einsum("aiaj->ij", rho.reshape(2, D // 2, 2, D // 2))
        rho = np.kron(np.diag([(1 + ut) / 2, (1 - ut) / 2]).astype(complex), red)
        out[t] = [float(np.real(np.trace(Zq @ rho))) for Zq in Zs]
    return out


# =============================================================================
# Processor
# =============================================================================
def processor_unitary(theta: float) -> np.ndarray:
    Ry = expm(-1j * np.pi / 4 * np.kron(Y2, I2))
    ZZ = expm(-1j * theta / 2 * np.kron(Z2, Z2))
    return ZZ @ Ry


def processor_feature_fn(spec: V5Spec, g: float, *, m_leak: float = 0.0, m: float = 0.0):
    """u -> f(u), exact expectation of the one processor observable."""
    theta = float(g) * spec.theta_max * (1.0 + m_leak * m)
    U = processor_unitary(theta)
    O = np.kron(np.cos(spec.phi) * X2 + np.sin(spec.phi) * Y2, I2)
    Oh = U.conj().T @ O @ U                      # Heisenberg picture, computed once

    def f(u: float) -> float:
        r = np.diag([(1 + u) / 2, (1 - u) / 2]).astype(complex)
        return float(np.real(np.trace(Oh @ np.kron(r, r))))
    return f


def processor_features(spec: V5Spec, u, g: float, **kw) -> np.ndarray:
    f = processor_feature_fn(spec, g, **kw)
    return np.array([[f(float(x))] for x in u])


# =============================================================================
# Adapter used by the audit engine
# =============================================================================
class V5Adapter:
    name = "V5"

    def __init__(self, spec: V5Spec = None, alpha: float = 0.5, kind: str = None,
                 eps: float = 0.2):
        self.spec = spec or V5Spec()
        self.alpha = float(alpha)
        self.kind, self.eps = kind, eps

    def run(self, u, m, g, seed=None):
        u = np.asarray(u, dtype=float)
        s = self.spec
        XR = memory_features(u, float(m) * s.p_max, s.L,
                             g_leak=self.eps if self.kind == "g_into_R" else 0.0,
                             g=float(g))[:, s.read_stride - 1::s.read_stride]
        if self.kind == "serial":
            drive = np.clip(XR[:, 0], -1, 1)
            XP = processor_features(s, drive, g)
        else:
            XP = processor_features(s, u, g, m_leak=self.eps if self.kind == "m_into_P" else 0.0,
                                    m=float(m))
        XJ = XR * XP                                    # z_r * f, r = 1..L
        return {"R": XR, "P": XP, "J": XJ,
                "labels_R": [f"R:Z{r}" for r in s.read_rails],
                "labels_P": ["P:n.sigma0"],
                "labels_J": [f"J:Z{r}*n.sigma0" for r in s.read_rails]}

    def processor_fn(self, g):
        f = processor_feature_fn(self.spec, g)
        return lambda u: np.array([f(u)])

    def describe(self) -> dict:
        return {"name": self.name, "spec": self.spec.as_dict(), "alpha": self.alpha,
                "resources": self.spec.resources(), "kind": self.kind}

    def with_spec(self, **kw):
        return V5Adapter(replace(self.spec, **kw), self.alpha, self.kind, self.eps)


# =============================================================================
# Encoder-only capacity (the V4 audit's item 2, for V5)
# =============================================================================
def encoder_only_capacity(spec: V5Spec, *, n_quad: int = 96) -> dict:
    """Exact degree capacities at tau = 0 with every dynamics switched off.

    R: rail 0 holds rho(u), rails 1..L are |0> and are what is read.
    P: rho(u) (x) rho(u) with NO unitary, observable cos(phi)X0 + sin(phi)Y0.
    'P_full_Z_algebra' ({Z0, Z1, Z0Z1}) shows the nonlinear resource the
    encoder carries but the operational observable cannot see without g."""
    from .audit_ipc import function_subspace
    O = np.cos(spec.phi) * X2 + np.sin(spec.phi) * Y2

    def r(u):
        return np.diag([(1 + u) / 2, (1 - u) / 2]).astype(complex)

    def fR(u):
        return np.ones(len(spec.read_rails))              # read rails untouched

    def fP(u):
        return np.array([np.real(np.trace(O @ r(u)))])

    def fJ(u):
        return fR(u) * fP(u)[0]

    def fZ(u):
        rho = np.kron(r(u), r(u))
        return np.array([np.real(np.trace(M @ rho)) for M in
                         (np.kron(Z2, I2), np.kron(I2, Z2), np.kron(Z2, Z2))])

    g0 = processor_feature_fn(spec, 0.0)
    sets = {"1_memory_local": fR, "2_nonlinear_local": fP,
            "3_operational_all": lambda u: np.concatenate([fR(u), fP(u), fJ(u)]),
            "4_joint": fJ, "5_nonlinear_local_at_g0": lambda u: np.array([g0(u)]),
            "P_full_Z_algebra": fZ}
    out = {}
    for name, fn in sets.items():
        sub = function_subspace(fn, n_quad=n_quad)
        caps = {d: round(v, 12) for d, v in sub["degree_capacity"].items()}
        out[name] = {"rank": sub["rank"], "capacity_by_degree": caps,
                     "nonlinear_capacity": sum(v for d, v in caps.items() if d >= 2),
                     "max_degree": max([d for d, v in caps.items() if v > 1e-9], default=0)}
    return out


# =============================================================================
# Finite shots and calibration-informed noise
# =============================================================================
def _shots(X, S, rng):
    p = np.clip((1.0 + X) / 2.0, 0.0, 1.0)
    return (2.0 * rng.binomial(S, p) - S) / S


def _joint_per_shot(XR, xp, S, rng):
    """<Z_r (x) O_P> estimated as the mean per-shot product of +/-1 outcomes,
    one shot record per (rail, step): the quantum estimator. Chunked by rail
    to bound memory."""
    T, L = XR.shape
    out = np.empty((T, L))
    pb = np.clip((1 + xp) / 2, 0, 1)[:, None]
    for r in range(L):
        pa = np.clip((1 + XR[:, r]) / 2, 0, 1)[:, None]
        oa = 2.0 * (rng.random((T, S)) < pa) - 1.0
        ob = 2.0 * (rng.random((T, S)) < pb) - 1.0
        out[:, r] = (oa * ob).mean(axis=1)
    return out


class V5ShotAdapter:
    """R, P from S shots each; J from the per-shot product (same S). Separate
    streams per route so no route's noise can depend on the other's control."""

    def __init__(self, base: V5Adapter, shots: int, *, depol_R: float = 0.0,
                 depol_P: float = 0.0, readout_error: float = 0.0):
        self.base, self.shots = base, int(shots)
        self.spec, self.alpha = base.spec, base.alpha
        self.depol_R, self.depol_P, self.ro = depol_R, depol_P, readout_error
        self.name = f"V5+{shots}shots" + ("+noise" if depol_R or depol_P or readout_error else "")

    def run(self, u, m, g, seed):
        s = self.spec
        u = np.asarray(u, dtype=float)
        # depolarizing on every rail each step: single-site marginals scale by
        # (1 - p), and it ACCUMULATES through the memory recursion
        p = float(m) * s.p_max
        z = np.ones(s.L + 1)
        XR = np.empty((len(u), s.L))
        for t, ut in enumerate(u):
            for r in range(s.L, 0, -1):
                a, b = z[r - 1], z[r]
                z[r - 1] = (1 - p) * a + p * b
                z[r] = (1 - p) * b + p * a
            z[1:] *= (1.0 - self.depol_R)
            z[0] = float(ut)
            XR[t] = z[1:]
        XR = XR[:, s.read_stride - 1::s.read_stride]
        xp = (1.0 - self.depol_P) * processor_features(s, u, g)[:, 0]
        sc = 1.0 - 2.0 * self.ro
        rngR = np.random.default_rng(3 * int(seed) + 1)
        rngP = np.random.default_rng(3 * int(seed) + 2)
        rngJ = np.random.default_rng(3 * int(seed) + 3)
        XJ = _joint_per_shot(sc * XR, sc * xp, self.shots, rngJ)
        return {"R": _shots(sc * XR, self.shots, rngR),
                "P": _shots(sc * xp, self.shots, rngP)[:, None], "J": XJ}

    def describe(self) -> dict:
        return {**self.base.describe(), "shots": self.shots, "depol_R": self.depol_R,
                "depol_P": self.depol_P, "readout_error": self.ro}


def ibm_informed(base: V5Adapter, params: dict, *, shots: int = 4096) -> V5ShotAdapter:
    """Effective noise from a real calibration snapshot (NOT a transpiled run).
    Gate counts per step: R = L controlled-SWAP-channel layers, taken as 3 CZ +
    4 sx per rail pair; P = Ry + one ZZ rotation (2 CZ) + basis change (2 sx)."""
    e1, e2 = params["median_sx_error"], params["median_2q_error"]
    L = base.spec.L
    pR = min(0.5, 4 * e1 + 3 * e2)          # per rail per step
    pP = min(0.5, (3 * e1 + 2 * e2))
    return V5ShotAdapter(base, shots, depol_R=pR, depol_P=pP,
                         readout_error=params["median_readout_error"])
