"""
v6_architecture.py -- three-route architecture for GENERAL memory / nonlinearity /
nonlinear-memory separation.

Routes (all simulated exactly; every register state is written down in closed form)
-------------------------------------------------------------------------------
R  linear memory, control m (identical to V5.4's memory):
   random-SWAP transport on rails 0..L_R, p_R = m p_R_max; rail 0 re-prepared in
   rho(u_t) = (I + u_t Z)/2 each step; read <Z_r> on rails stride_R, 2 stride_R, ...
   Exactly linear in past inputs. The ONLY source of the primary metric M.

P  instantaneous nonlinear processor, control g (3 input copies, reset each step):
   rho(u)^{x3} -> Ry(pi/2) on copy 0 -> exp(-i theta Z0Z1/2) exp(-i chi Z0Z1Z2/2)
   with theta = g theta_max, chi = g chi_max -> measure n.sigma_0,
   n = (cos phi, sin phi, 0). Because the input is a product of Z-diagonal states,
       f(u) = <O_H> = sum_S c_S u^{|S|}   (Walsh coefficients of diag O_H),
   a cubic polynomial whose degree-2 and degree-3 weights vanish at g = 0 and grow
   with g. ONE observable -> single-feature capacity C_d = a_d^2 / sum a^2 is the
   nonlinear FRACTION of the feature: scale-invariant, continuous in g. The ONLY
   source of the primary metric N.
   A second setting on an identical P copy measures Y_0: f_Y(u), whose every term
   carries sin(theta) or sin(chi) -> f_Y == 0 at g = 0 (a purely nonlinear channel).

J  joint R x P, both controls by design: J_r = <Z_r^R (x) Y_0^P> = z_r^R f_Y(u_t).
   Current-nonlinear x old-linear (class C1). Identically 0 at g = 0.

Q  nonlinear memory, controls m (transport) and g (nonlinearity):
   a second random-SWAP register, p_Q = m p_Q_max. Each step a fresh 3-copy
   processor (same design as P) is run on u_t and its output qubit is SWAPped into
   rail 0 after a fixed basis rotation n -> z and full dephasing, so rail 0 holds
   (I + w(u_t) Z)/2 with w = f (the processor's output). The Q register is always
   Z-diagonal (a probability distribution over bitstrings), so it is described
   EXACTLY by its first and second moments z_r = <Z_r>, C_ab = <Z_a Z_b>, which
   the random-SWAP channel maps linearly (see q_register).
   Readout:
     single-site <Z_r> on rails stride_Q, 2 stride_Q, ...  -> sum_k A_rk w(u_{t-k})
         delayed P2, P3 (classes C2, C4) because w carries degrees 2, 3 when g > 0
     pair (a, b): Ry(pi/2) on a, exp(-i theta_Q Z_a Z_b / 2), theta_Q = g theta_Q_max,
         measure n_Q.sigma_a -> y_ab = c_a z_a + c_ab C_ab (+ c_b z_b + c_0)
         C_ab = sum_{i != j} W(i, j) w(u_{t-i}) w(u_{t-j}): old x old (class C3);
         c_ab ~ sin(theta_Q) -> 0 at g = 0.

At g = 0 EVERY operational feature (R, P, J, Q) is an affine function of the input
history: no route can express any nonlinear target without g.
At m = 0 both registers are frozen (p = 0): no route can express any delayed target.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.linalg import expm

from .v5_architecture import memory_features

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.diag([1.0, -1.0]).astype(complex)


def _op(single: dict, n: int) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, single.get(q, I2))
    return out


def walsh_diag(Oh: np.ndarray, n: int) -> dict:
    """Coefficients c_S of diag(Oh) = sum_S c_S prod_{i in S} Z_i."""
    d = np.real(np.diag(Oh))
    bits = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))) & 1
    zs = 1 - 2 * bits
    out = {}
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            chi = np.prod(zs[:, list(S)], axis=1) if S else np.ones(2 ** n)
            out[S] = float(np.mean(d * chi))
    return out


def processor_poly(theta: float, chi: float, phi: float, observable: str = "n") -> np.ndarray:
    """Polynomial coefficients a_0..a_3 of u -> <O>, exact."""
    Ry = expm(-1j * np.pi / 4 * _op({0: Y2}, 3))
    U = expm(-1j * theta / 2 * _op({0: Z2, 1: Z2}, 3)) @ \
        expm(-1j * chi / 2 * _op({0: Z2, 1: Z2, 2: Z2}, 3))
    O = _op({0: np.cos(phi) * X2 + np.sin(phi) * Y2}, 3) if observable == "n" else _op({0: Y2}, 3)
    Oh = Ry.conj().T @ U.conj().T @ O @ U @ Ry
    c = walsh_diag(Oh, 3)
    a = np.zeros(4)
    for S, v in c.items():
        a[len(S)] += v
    return a


def pair_readout_coeffs(theta_q: float, phi_q: float) -> dict:
    """y = c0 + ca z_a + cb z_b + cab C_ab for a Z-diagonal two-rail state, exact."""
    Ry = expm(-1j * np.pi / 4 * _op({0: Y2}, 2))
    U = expm(-1j * theta_q / 2 * _op({0: Z2, 1: Z2}, 2))
    O = _op({0: np.cos(phi_q) * X2 + np.sin(phi_q) * Y2}, 2)
    c = walsh_diag(Ry.conj().T @ U.conj().T @ O @ U @ Ry, 2)
    return {"c0": c[()], "ca": c[(0,)], "cb": c[(1,)], "cab": c[(0, 1)]}


def poly_eval(a: np.ndarray, u) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return a[0] + a[1] * u + a[2] * u ** 2 + a[3] * u ** 3


def q_register(w, p: float, L: int, *, depol: float = 0.0) -> tuple:
    """Exact first and second moments of the Z-diagonal random-SWAP register.

    Per step: for r = L..1 apply rho -> (1-p) rho + p SWAP_{r-1,r} rho SWAP_{r-1,r};
    (optional depolarizing, which scales z by (1-depol) and C_ab by (1-depol)^2);
    then rail 0 <- (I + w_t Z)/2, independent of the rest. Returns z (T, L+1) and
    C (T, L+1, L+1) AFTER injection. Starts in |0...0> (z = 1, C = 1)."""
    Tn = len(w)
    z = np.ones(L + 1)
    C = np.ones((L + 1, L + 1))
    Z = np.empty((Tn, L + 1))
    CC = np.empty((Tn, L + 1, L + 1))
    for t in range(Tn):
        for r in range(L, 0, -1):
            i, j = r - 1, r
            zi, zj = z[i], z[j]
            z[i] = (1 - p) * zi + p * zj
            z[j] = (1 - p) * zj + p * zi
            ci, cj, cij = C[i].copy(), C[j].copy(), C[i, j]
            ni = (1 - p) * ci + p * cj
            nj = (1 - p) * cj + p * ci
            C[i, :], C[j, :] = ni, nj
            C[:, i], C[:, j] = ni, nj
            C[i, i] = C[j, j] = 1.0
            C[i, j] = C[j, i] = cij
        if depol:
            z *= (1 - depol)
            d = np.diag(C).copy()
            C *= (1 - depol) ** 2
            np.fill_diagonal(C, d)
        z[0] = float(w[t])
        C[0, :] = w[t] * z
        C[:, 0] = w[t] * z
        C[0, 0] = 1.0
        Z[t] = z
        CC[t] = C
    return Z, CC


def q_register_dm(w, p: float, L: int) -> tuple:
    """Same register as a full diagonal probability vector over 2^(L+1) bitstrings
    (verification only, small L)."""
    n = L + 1
    D = 2 ** n
    bits = (np.arange(D)[:, None] >> (n - 1 - np.arange(n))) & 1
    zs = 1 - 2 * bits
    prob = np.zeros(D); prob[0] = 1.0

    def perm(a, b):
        bb = bits.copy(); bb[:, [a, b]] = bb[:, [b, a]]
        return (bb * (1 << (n - 1 - np.arange(n)))).sum(axis=1)

    P = {r: perm(r - 1, r) for r in range(1, L + 1)}
    Z, CC = [], []
    for wt in w:
        for r in range(L, 0, -1):
            prob = (1 - p) * prob + p * prob[P[r]]
        rest = prob.reshape(2, D // 2).sum(axis=0)
        prob = np.concatenate([(1 + wt) / 2 * rest, (1 - wt) / 2 * rest])
        Z.append(zs.T @ prob)
        CC.append(np.einsum("k,ka,kb->ab", prob, zs, zs))
    return np.array(Z), np.array(CC)


@dataclass(frozen=True)
class V6Spec:
    # R
    L_R: int = 16
    stride_R: int = 2
    pR_max: float = 0.7
    # P (and the processor that writes into Q)
    theta_max: float = 0.38 * np.pi
    chi_max: float = 0.30 * np.pi
    phi: float = np.pi / 3
    # Q
    L_Q: int = 12
    stride_Q: int = 1
    pQ_max: float = 0.7
    q_pairs: tuple = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
                      (1, 3), (2, 5), (4, 8), (6, 11))
    thetaQ_max: float = 0.40 * np.pi
    phiQ: float = np.pi / 4
    # V6.1 additions (defaults reproduce V6.0 exactly)
    joint: str = "Y"                   # "Y": J = z^R f_Y ; "rotated": g-rotated joint readout
    thetaJ_max: float = 0.40 * np.pi   # used when joint == "rotated"
    phiJ: float = np.pi / 4
    thetaW_max: float = None           # Q-writer processor; None -> same as P
    chiW_max: float = None
    phiW: float = None
    chiW_power: int = 1                # V6.3: writer cubic angle = g**chiW_power * chiW_max
    shots: int = 0                     # V6.3: features are S-shot estimates (0 = exact)
    # V6.2: g-rotated pair readouts on the LINEAR memory register R (old x old, C3).
    # Separate measurement settings, assigned to the combined route; M never uses them.
    r_pairs: tuple = ()
    thetaRP_max: float = 0.40 * np.pi
    phiRP: float = np.pi / 4

    def __post_init__(self):
        if not 1 <= self.stride_R <= self.L_R or not 1 <= self.stride_Q <= self.L_Q:
            raise ValueError("stride out of range")
        for a, b in self.q_pairs:
            if not (1 <= a <= self.L_Q and 1 <= b <= self.L_Q and a != b):
                raise ValueError(f"bad Q pair {(a, b)}")
        for a, b in self.r_pairs:
            if not (1 <= a <= self.L_R and 1 <= b <= self.L_R and a != b):
                raise ValueError(f"bad R pair {(a, b)}")
        if not (0 < self.pR_max <= 1 and 0 < self.pQ_max <= 1):
            raise ValueError("transport probabilities must be in (0, 1]")
        if self.joint not in ("Y", "rotated"):
            raise ValueError("joint must be 'Y' or 'rotated'")

    @property
    def rails_R(self) -> list:
        return list(range(self.stride_R, self.L_R + 1, self.stride_R))

    @property
    def rails_Q(self) -> list:
        return list(range(self.stride_Q, self.L_Q + 1, self.stride_Q))

    def settings(self) -> dict:
        """Measurement settings per step (commuting groups)."""
        # Q pairs: pairs sharing a rail need separate settings; greedy colouring
        def colour(pairs):
            groups = []
            for a, b in pairs:
                for gset in groups:
                    if a not in gset and b not in gset:
                        gset.update((a, b)); break
                else:
                    groups.append({a, b})
            return len(groups)
        return {"R_Z": 1, "P_n": 1, "joint": 1, "Q_Z": 1, "Q_pairs": colour(self.q_pairs),
                "R_pairs": colour(self.r_pairs)}

    def resources(self) -> dict:
        return {"qubits": (self.L_R + 1) + 3 + 3 + (self.L_Q + 1),
                "input_copies_per_step": 1 + 3 + 3 + 3,
                "features": {"R": len(self.rails_R), "P": 1, "J": len(self.rails_R),
                             "Q": len(self.rails_Q) + len(self.q_pairs) + len(self.r_pairs)},
                "measurement_settings": self.settings()}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["theta_max_over_pi"] = self.theta_max / np.pi
        d["chi_max_over_pi"] = self.chi_max / np.pi
        d["thetaQ_max_over_pi"] = self.thetaQ_max / np.pi
        d["rails_R"], d["rails_Q"] = self.rails_R, self.rails_Q
        return d


class V6Adapter:
    """kind in {None, 'g_into_R', 'm_into_P', 'serial'} (contaminated controls)."""

    name = "V6"

    def __init__(self, spec: V6Spec = None, alpha: float = 0.5, kind: str = None,
                 eps: float = 0.2, depol_R: float = 0.0, depol_Q: float = 0.0,
                 depol_P: float = 0.0):
        self.spec = spec or V6Spec()
        self.alpha, self.kind, self.eps = float(alpha), kind, eps
        self.depol_R, self.depol_Q, self.depol_P = depol_R, depol_Q, depol_P

    def polys(self, m: float, g: float):
        s = self.spec
        scale = 1.0 + (self.eps * m if self.kind == "m_into_P" else 0.0)
        th, ch = g * s.theta_max * scale, g * s.chi_max * scale
        tw = s.theta_max if s.thetaW_max is None else s.thetaW_max
        cw = s.chi_max if s.chiW_max is None else s.chiW_max
        pw = s.phi if s.phiW is None else s.phiW
        return (processor_poly(th, ch, s.phi, "n"), processor_poly(th, ch, s.phi, "Y"),
                processor_poly(g * tw, g ** s.chiW_power * cw, pw, "n"))

    def run(self, u, m, g, seed=None) -> dict:
        s = self.spec
        u = np.asarray(u, dtype=float)
        m, g = float(m), float(g)
        pR = m * s.pR_max * (1.0 + (self.eps * g if self.kind == "g_into_R" else 0.0))
        pR = float(np.clip(pR, 0.0, 1.0))
        if s.r_pairs:
            # first AND second moments of R (same channel; z identical to memory_features)
            ZR, CR = q_register(u, pR, s.L_R, depol=self.depol_R)
            zR = ZR[:, 1:]
        elif self.depol_R:
            zR = _memory_depol(u, pR, s.L_R, self.depol_R)
        else:
            zR = memory_features(u, pR, s.L_R)
        XR = zR[:, s.stride_R - 1::s.stride_R]
        aP, aY, aW = self.polys(m, g)
        drive = np.clip(zR[:, 0], -1, 1) if self.kind == "serial" else u
        fP = (1 - self.depol_P) * poly_eval(aP, drive)
        fY = (1 - self.depol_P) * poly_eval(aY, drive)
        # Q: processor output written into the nonlinear-memory register
        w = (1 - self.depol_P) * poly_eval(aW, u)
        zQ, CQ = q_register(w, m * s.pQ_max, s.L_Q, depol=self.depol_Q)
        XQz = zQ[:, s.rails_Q]
        pc = pair_readout_coeffs(g * s.thetaQ_max, s.phiQ)
        XQp = (np.column_stack([pc["c0"] + pc["ca"] * zQ[:, a] + pc["cb"] * zQ[:, b]
                                + pc["cab"] * CQ[:, a, b] for a, b in s.q_pairs])
               if s.q_pairs else np.empty((len(u), 0)))
        if s.joint == "rotated":
            # R rail a and the P output qubit (rotated n -> z, dephased) read by the same
            # g-rotated two-qubit circuit as the Q pairs: product state -> <Z_a Z_b> = z_a f
            pj = pair_readout_coeffs(g * s.thetaJ_max, s.phiJ)
            XJ = pj["c0"] + pj["ca"] * XR + pj["cb"] * fP[:, None] + pj["cab"] * XR * fP[:, None]
            fJ = fP                                   # P-side marginal of the joint observable
        else:
            XJ = XR * fY[:, None]
            fJ = fY
        if s.r_pairs:
            pr = pair_readout_coeffs(g * s.thetaRP_max, s.phiRP)
            XRp = np.column_stack([pr["c0"] + pr["ca"] * ZR[:, a] + pr["cb"] * ZR[:, b]
                                   + pr["cab"] * CR[:, a, b] for a, b in s.r_pairs])
        else:
            XRp = np.empty((len(u), 0))
        return {"R": XR, "P": fP[:, None], "PY": fJ[:, None], "J": XJ,
                "Q": np.hstack([XQz, XQp, XRp]), "Qz": XQz, "Qp": XQp, "Rp": XRp,
                "Qall": zQ[:, 1:], "Q_pairs_rails_idx": [(a - 1, b - 1) for a, b in s.q_pairs],
                "Rall": zR, "R_pairs_rails_idx": [(a - 1, b - 1) for a, b in s.r_pairs],
                "labels_R": [f"R:Z{r}" for r in s.rails_R], "labels_P": ["P:n.sigma0"],
                "labels_J": [f"J:Z{r}*Y0" for r in s.rails_R],
                "labels_Q": [f"Q:Z{r}" for r in s.rails_Q] + [f"Q:pair{a}-{b}" for a, b in s.q_pairs]}

    def processor_fn(self, g):
        a = processor_poly(g * self.spec.theta_max, g * self.spec.chi_max, self.spec.phi, "n")
        return lambda uu: np.array([poly_eval(a, uu)])

    def describe(self) -> dict:
        return {"name": self.name, "spec": self.spec.as_dict(), "alpha": self.alpha,
                "kind": self.kind, "resources": self.spec.resources(),
                "noise": {"depol_R": self.depol_R, "depol_Q": self.depol_Q,
                          "depol_P": self.depol_P}}

    def with_spec(self, **kw):
        return V6Adapter(replace(self.spec, **kw), self.alpha, self.kind, self.eps,
                         self.depol_R, self.depol_Q, self.depol_P)


def _memory_depol(u, p, L, depol):
    z = np.ones(L + 1)
    out = np.empty((len(u), L))
    for t, ut in enumerate(u):
        for r in range(L, 0, -1):
            a, b = z[r - 1], z[r]
            z[r - 1] = (1 - p) * a + p * b
            z[r] = (1 - p) * b + p * a
        z[1:] *= (1 - depol)
        z[0] = float(ut)
        out[t] = z[1:]
    return out


class V6ShotAdapter:
    """Every feature is the expectation of a +/-1 observable (single-site Z, the
    pair observable n_Q.sigma_a after its rotation, the processor observable, and the
    per-shot product Z^R (x) Y^P, whose registers are in a product state). S shots ->
    (2 Binom(S, (1+x)/2) - S)/S. Separate RNG stream per route."""

    def __init__(self, base: V6Adapter, shots: int, readout_error: float = 0.0):
        self.base, self.shots, self.ro = base, int(shots), float(readout_error)
        self.spec, self.alpha = base.spec, base.alpha
        self.name = f"V6+{shots}shots" + ("+ro" if readout_error else "")

    def run(self, u, m, g, seed):
        f = self.base.run(u, m, g, seed)
        sc = 1.0 - 2.0 * self.ro
        out = dict(f)

        def sample(x, off):
            rng = np.random.default_rng(7 * int(seed) + off)
            p = np.clip((1 + x) / 2, 0, 1)
            return (2.0 * rng.binomial(self.shots, p) - self.shots) / self.shots

        # every register marginal is an S-shot estimate; the local readouts (and the
        # classical baseline's marginals) are the SAME noisy numbers
        out["Rall"] = sample(sc * f["Rall"], 1)
        out["R"] = out["Rall"][:, self.spec.stride_R - 1::self.spec.stride_R]
        out["Qall"] = sample(sc * f["Qall"], 5)
        out["Qz"] = out["Qall"][:, np.array(self.spec.rails_Q) - 1]
        for k, off, s2 in (("P", 2, sc), ("PY", 3, sc), ("J", 4, sc * sc), ("Qp", 6, sc),
                           ("Rp", 8, sc)):
            if f[k].size:
                out[k] = sample(s2 * f[k], off)
        out["Q"] = np.hstack([out["Qz"], out["Qp"], out["Rp"]])
        return out

    @property
    def kind(self):
        return self.base.kind

    def processor_fn(self, g):
        return self.base.processor_fn(g)

    def with_spec(self, **kw):
        return V6ShotAdapter(self.base.with_spec(**kw), self.shots, self.ro)

    def describe(self):
        return {**self.base.describe(), "shots": self.shots, "readout_error": self.ro}
