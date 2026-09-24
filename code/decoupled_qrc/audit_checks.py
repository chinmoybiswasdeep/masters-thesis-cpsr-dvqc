"""
audit_checks.py -- the non-factorial checks of the independent V4 audit.

  * encoder algebra, verified SYMBOLICALLY with exact rational arithmetic
  * encoder-only capacity by degree for five observable sets
  * structural dependency tests on >= 100 random inputs/states, plus
    deliberately contaminated controls that the tests MUST detect
  * the section-6 question: is nonlinear control intrinsic, or a ridge artifact?
  * the classical product baseline and measurement-noise robustness
  * an IBM-calibration-informed effective noise model
  * detectors for every deliberately invalid control

Nothing here modifies V4. Contaminated and noisy variants are built by
re-running V4's own route classes with a single, documented perturbation.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import numpy as np

from . import audit_ipc as A


# =============================================================================
# 1. Encoder algebra -- exact, symbolic
# =============================================================================
def _poly_mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def symbolic_encoder_expansion(n: int) -> dict:
    """Prove rho(u) = ((I+uZ)/2)^{(x)n} = 2^-n sum_S u^|S| Z_S EXACTLY.

    Density-matrix entries are polynomials in u held as exact Fraction
    coefficient lists. The Z-string coefficient Tr[Z_S rho]/... is computed in
    exact arithmetic and compared to 2^-n u^|S|; no floating point is involved.
    """
    single = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(-1, 2)]]  # diag
    diag = [[Fraction(1)]]
    for _ in range(n):
        diag = [_poly_mul(a, b) for a in diag for b in single]
    dim = 2 ** n
    mismatches = 0
    table = {}
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            # Tr[Z_S rho] = sum_x (-1)^{sum_{i in S} x_i} rho_xx
            coeff = [Fraction(0)] * (n + 1)
            for x in range(dim):
                bits = [(x >> (n - 1 - i)) & 1 for i in range(n)]
                sign = -1 if sum(bits[i] for i in S) % 2 else 1
                for p, c in enumerate(diag[x]):
                    coeff[p] += sign * c
            expected = [Fraction(0)] * (n + 1)
            expected[k] = Fraction(1)            # Tr[Z_S rho] = u^|S|
            if coeff != expected:
                mismatches += 1
            table[str(S)] = {"degree": k, "coeff": [str(c) for c in coeff if c != 0]}
    return {"n": n, "identity": "rho(u) = 2^-n sum_S u^|S| Z_S",
            "n_strings": len(table), "mismatches": mismatches,
            "verified_exactly": bool(mismatches == 0),
            "max_degree_in_state": n}


def numeric_encoder_expansion(n: int, n_u: int = 25, seed: int = 0) -> dict:
    """Same identity in floating point, at random u, as an independent check."""
    I2, Z = np.eye(2), np.diag([1.0, -1.0])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for u in rng.uniform(-1, 1, n_u):
        lhs = np.array([[1.0]])
        for _ in range(n):
            lhs = np.kron(lhs, (I2 + u * Z) / 2)
        rhs = np.zeros((2 ** n, 2 ** n))
        for k in range(n + 1):
            for S in itertools.combinations(range(n), k):
                op = np.array([[1.0]])
                for q in range(n):
                    op = np.kron(op, Z if q in S else I2)
                rhs += (u ** k) * op
        rhs /= 2 ** n
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    return {"n": n, "max_abs_error": worst, "ok": bool(worst < 1e-14)}


def encoder_only_capacity(adapter, *, n_quad: int = 96) -> dict:
    """Capacity by degree at tau = 0 for five observable sets, exact.

    'encoder only' means every dynamics switched off: the register holds the
    injected state and nothing acts on it. Items 1-4 are memoryless, so the
    function-space analysis is exact. Item 5 uses the architecture's own
    nominally disabled settings (g = 0 for P; m = 0 for R).
    """
    from .v4_encoder import embed, single_qubit_state
    spec = adapter.spec
    L, NP = spec.memory.L_R, spec.processor.N_P

    def R_state(u):                      # rail 0 injected, the rest in |0>
        out = single_qubit_state(u)
        for _ in range(L - 1):
            out = np.kron(out, np.diag([1.0, 0.0]).astype(complex))
        return out

    def P_state(u):
        out = np.array([[1.0]], dtype=complex)
        for _ in range(NP):
            out = np.kron(out, single_qubit_state(u))
        return out

    labR, opsR = spec.memory.observables()
    labP, opsP = spec.processor.observables()
    from .v4_architecture import _joint_pairs
    pairs = _joint_pairs(len(labR), len(labP), spec.n_joint, 0)

    def ev(rho, ops):
        return np.array([float(np.real(np.trace(O @ rho))) for O in ops])

    def fR(u):
        return ev(R_state(u), opsR)

    def fP(u):
        return ev(P_state(u), opsP)

    def fJ(u):
        r, p = fR(u), fP(u)
        return np.array([r[a] * p[b] for a, b in pairs])

    def fALL(u):
        return np.concatenate([fR(u), fP(u), fJ(u)])

    allZ = [embed(NP, {q: "Z" for q in S}) for k in range(1, NP + 1)
            for S in itertools.combinations(range(NP), k)]

    def fPfull(u):
        return ev(P_state(u), allZ)

    sets = {"1_memory_local": fR, "2_nonlinear_local": fP, "3_operational_all": fALL,
            "4_joint": fJ, "P_full_Z_algebra": fPfull}
    out = {}
    for name, fn in sets.items():
        sub = A.function_subspace(fn, n_quad=n_quad)
        caps = {d: round(v, 12) for d, v in sub["degree_capacity"].items()}
        nl = sum(v for d, v in caps.items() if d >= 2)
        out[name] = {"rank": sub["rank"], "capacity_by_degree": caps,
                     "nonlinear_capacity": nl,
                     "max_degree": max([d for d, v in caps.items() if v > 1e-9], default=0)}

    # 5. nominally disabled dynamics, via the architecture's own routes
    from .v4_architecture import ProcessorRoute
    Pg0 = ProcessorRoute(spec.processor, 0.0, 0)
    sub = A.function_subspace(lambda u: Pg0.step(u), n_quad=n_quad)
    caps = {d: round(v, 12) for d, v in sub["degree_capacity"].items()}
    out["5_nonlinear_local_at_g0"] = {
        "rank": sub["rank"], "capacity_by_degree": caps,
        "nonlinear_capacity": sum(v for d, v in caps.items() if d >= 2),
        "max_degree": max([d for d, v in caps.items() if v > 1e-9], default=0)}
    return out


# =============================================================================
# 2. Structural dependency
# =============================================================================
def _v4_run_variant(spec, u, m, g, seed, kind: str = None, eps: float = 0.2):
    """Re-implements run_v4's loop with ONE optional contamination, so the
    dependency test can be shown to have power. kind=None reproduces run_v4
    exactly (asserted in tests)."""
    from .v4_architecture import MemoryRoute, ProcessorRoute, _joint_pairs
    mspec, pspec = spec.memory, spec.processor
    if kind == "g_into_R":
        mspec = replace(mspec, tau_mix=mspec.tau_mix * (1.0 + eps * g))
    if kind == "m_into_P":
        pspec = replace(pspec, dt=pspec.dt * (1.0 + eps * m))
    R = MemoryRoute(mspec, m, seed)
    P = ProcessorRoute(pspec, g, seed)
    XR, XP = [], []
    for ut in u:
        fr = R.step(float(ut))
        drive = float(np.clip(fr[0], -1, 1)) if kind == "serial" else float(ut)
        XR.append(fr)
        XP.append(P.step(drive))
    XR, XP = np.array(XR), np.array(XP)
    pairs = _joint_pairs(XR.shape[1], XP.shape[1], spec.n_joint, seed)
    XJ = np.column_stack([XR[:, a] * XP[:, b] for a, b in pairs])
    return XR, XP, XJ


def structural_dependency(spec, *, n_draws: int = 100, delta: float = 1e-6,
                          seed: int = 0, kind: str = None) -> dict:
    """Central differences dX_R/dg and dX_P/dm over random inputs AND random
    internal states (a random-length random-input burn-in), plus the finite
    change over the whole control range."""
    rng = np.random.default_rng(seed)
    dR, dP, fR, fP = [], [], [], []
    for _ in range(int(n_draws)):
        burn = int(rng.integers(5, 60))
        u = rng.uniform(-1, 1, burn + 8)
        m = float(rng.uniform(0.05, 0.95))
        g = float(rng.uniform(0.05, 0.95))
        a = int(rng.integers(1, 10_000_000))
        R_p, _, _ = _v4_run_variant(spec, u, m, g + delta, a, kind)
        R_m, _, _ = _v4_run_variant(spec, u, m, g - delta, a, kind)
        _, P_p, _ = _v4_run_variant(spec, u, m + delta, g, a, kind)
        _, P_m, _ = _v4_run_variant(spec, u, m - delta, g, a, kind)
        dR.append(float(np.abs((R_p - R_m) / (2 * delta)).max()))
        dP.append(float(np.abs((P_p - P_m) / (2 * delta)).max()))
        R1, _, _ = _v4_run_variant(spec, u, m, 1.0, a, kind)
        R0, _, _ = _v4_run_variant(spec, u, m, 0.0, a, kind)
        _, P1, _ = _v4_run_variant(spec, u, 1.0, g, a, kind)
        _, P0, _ = _v4_run_variant(spec, u, 0.0, g, a, kind)
        fR.append(float(np.abs(R1 - R0).max()))
        fP.append(float(np.abs(P1 - P0).max()))
    return {"kind": kind or "V4 as frozen", "n_draws": int(n_draws), "delta": delta,
            "max_dXR_dg": max(dR), "max_dXP_dm": max(dP),
            "max_range_change_XR_over_g": max(fR), "max_range_change_XP_over_m": max(fP),
            "isolated": bool(max(dR) == 0.0 and max(dP) == 0.0
                             and max(fR) == 0.0 and max(fP) == 0.0)}


def structural_dependency_adapter(make_adapter, *, n_draws: int = 100, delta: float = 1e-6,
                                  seed: int = 0, kind: str = None) -> dict:
    """`structural_dependency` for any adapter: make_adapter(kind) -> object with
    run(u, m, g, seed) -> {"R", "P", ...}. Same draws, same statistics."""
    ad = make_adapter(kind)
    rng = np.random.default_rng(seed)
    dR, dP, fR, fP = [], [], [], []
    for _ in range(int(n_draws)):
        burn = int(rng.integers(5, 60))
        u = rng.uniform(-1, 1, burn + 8)
        m = float(rng.uniform(0.05, 0.95))
        g = float(rng.uniform(0.05, 0.95))
        a = int(rng.integers(1, 10_000_000))
        R = lambda gg: ad.run(u, m, gg, a)["R"]    # noqa: E731
        P = lambda mm: ad.run(u, mm, g, a)["P"]    # noqa: E731
        dR.append(float(np.abs((R(g + delta) - R(g - delta)) / (2 * delta)).max()))
        dP.append(float(np.abs((P(m + delta) - P(m - delta)) / (2 * delta)).max()))
        fR.append(float(np.abs(R(1.0) - R(0.0)).max()))
        fP.append(float(np.abs(P(1.0) - P(0.0)).max()))
    return {"kind": kind or "as frozen", "n_draws": int(n_draws), "delta": delta,
            "max_dXR_dg": max(dR), "max_dXP_dm": max(dP),
            "max_range_change_XR_over_g": max(fR), "max_range_change_XP_over_m": max(fP),
            "isolated": bool(max(dR) == 0.0 and max(dP) == 0.0
                             and max(fR) == 0.0 and max(fP) == 0.0)}


def indirect_dependency_inspection() -> list:
    """Code-path findings for every indirect route the brief names. Each was
    verified by reading `v4_architecture.run_v4` and `v4_ipc.compute_metrics`;
    the empirical counterpart is `metric_level_independence`."""
    return [
        {"channel": "shared normalization", "finding": "none across routes: each readout "
         "standardises its OWN feature block on its own train rows", "risk": "none"},
        {"channel": "shared random seeds", "finding": "R and P both derive disorder from "
         "the SAME architecture seed; this couples their realisations, not their "
         "controls -- neither route receives the other's control", "risk": "none for "
         "control dependence; reported"},
        {"channel": "shared preprocessing", "finding": "none", "risk": "none"},
        {"channel": "joint feature construction", "finding": "J = X_R * X_P depends on "
         "both controls by design; J is never used for M or N", "risk": "none for M/N; "
         "J-based metrics are combined by design"},
        {"channel": "readout regularization", "finding": "one fixed alpha per readout, "
         "fitted per readout", "risk": "none"},
        {"channel": "observable scaling", "finding": "none", "risk": "none"},
        {"channel": "cached states", "finding": "none: routes are constructed fresh per "
         "run_v4 call", "risk": "none"},
        {"channel": "mutable global configuration", "finding": "v4_experiment.THRESHOLDS "
         "is a module-level mutable dict; no code path mutates it", "risk": "latent"},
        {"channel": "feature selection", "finding": "joint pairs chosen by seed only, "
         "never by data", "risk": "none"},
        {"channel": "training-data-dependent transformations", "finding": "standardisation "
         "statistics are computed on the train block of the SAME readout", "risk": "none"},
    ]


def metric_level_independence(rows) -> dict:
    """M from R must be bit-identical across g; N from P bit-identical across m."""
    worst = {}
    for meth in ("ridge", "ols_std", "ols_raw"):
        byM, byN = {}, {}
        for r in rows:
            k = (r["arch_seed"], r["input_seed"])
            bM = r["blocks"].get(f"R|{meth}")
            bN = r["blocks"].get(f"P|{meth}")
            if bM:
                byM.setdefault((k, r["m"]), []).append(bM["metrics"]["M"])
            if bN:
                byN.setdefault((k, r["g"]), []).append(bN["metrics"]["N"])
        worst[meth] = {"max_spread_M_over_g": max((max(v) - min(v)) for v in byM.values()),
                       "max_spread_N_over_m": max((max(v) - min(v)) for v in byN.values())}
    return worst


# =============================================================================
# 3. Section 6 -- intrinsic nonlinear control, or a ridge artifact?
# =============================================================================
DENSE_G = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RIDGE_GRID = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)


def intrinsic_nonlinearity(adapter, *, seeds, T: int = 1600, m: float = 0.75) -> dict:
    """Analyses A (OLS raw), B (OLS standardised), C (ridge sweep) on data, and
    D (exact subspace geometry) with no data at all."""
    lib = [A.T(f"P{d}(t-0)", f"d{d}", ((0, d),)) for d in (1, 2, 3, 4)]
    split = A.Split.standard(T)
    rows = []
    for g in DENSE_G:
        for sp in seeds:
            u = np.random.default_rng(int(sp[1])).uniform(-1, 1, T)
            XP = adapter.run(u, m, g, int(sp[0]))["P"]
            Y = A.build_Y(u, lib)
            rec = {"g": g, "seed": list(sp)}
            for meth in ("ols_raw", "ols_std"):
                c = A.score_all(XP, Y, lib, split, meth)
                rec[meth] = float(np.mean(c[1:]))
                rec[meth + "_by_degree"] = [float(x) for x in c]
            for lam in RIDGE_GRID:
                c = A.score_all(XP, Y, lib, split, "ridge", alpha=lam)
                rec[f"ridge_{lam:g}"] = float(np.mean(c[1:]))
            rows.append(rec)

    # D: exact geometry of the processor's feature subspace
    pfn = getattr(adapter, "processor_fn", None)
    if pfn is None:                      # V4: its own processor route
        from .v4_architecture import ProcessorRoute
        pfn = lambda g: ProcessorRoute(adapter.spec.processor, g, 0).step  # noqa: E731
    sub = {g: A.function_subspace(pfn(g)) for g in DENSE_G}
    ref = sub[1.0]["Q"]
    positive = [g for g in DENSE_G if g > 0]
    geom = {}
    for g in DENSE_G:
        pa = A.principal_angles(sub[g]["Q"], ref)
        geom[g] = {"rank": sub[g]["rank"],
                   "top_singular_values": sub[g]["singular_values"][:6],
                   "max_principal_angle_to_g1_deg": pa["max_deg"],
                   "projection_distance_to_g1": pa["projection_distance"],
                   "exact_capacity_by_degree": sub[g]["degree_capacity"],
                   "exact_N": float(np.mean([sub[g]["degree_capacity"][d] for d in (2, 3, 4)]))}
    ranks = {geom[g]["rank"] for g in positive if g >= 1e-3}
    same_subspace = bool(len(ranks) == 1 and all(
        geom[g]["projection_distance_to_g1"] < 1e-6 for g in positive if g >= 1e-3))

    def curve(key):
        return {g: float(np.mean([r[key] for r in rows if r["g"] == g])) for g in DENSE_G}

    ols, std = curve("ols_raw"), curve("ols_std")
    interior = [g for g in DENSE_G if g >= 0.1]
    trend_ols = ols[interior[-1]] - ols[interior[0]]
    trend_std = std[interior[-1]] - std[interior[0]]
    ridge_curves = {lam: curve(f"ridge_{lam:g}") for lam in RIDGE_GRID}
    trend_ridge = {lam: c[interior[-1]] - c[interior[0]] for lam, c in ridge_curves.items()}
    changes_subspace = not same_subspace
    changes_amplitude = bool(np.std([geom[g]["top_singular_values"][0]
                                     for g in positive if g >= 1e-3]) > 1e-9)
    verdict = ("CONTINUOUS INTRINSIC NONLINEAR CONTROL FAILED"
               if (same_subspace and abs(trend_ols) < 0.10 and abs(trend_std) < 0.10)
               else "CONTINUOUS INTRINSIC NONLINEAR CONTROL HOLDS")
    return {"rows": rows, "ols_raw_curve": ols, "ols_std_curve": std,
            "ridge_curves": {f"{k:g}": v for k, v in ridge_curves.items()},
            "interior_trend": {"ols_raw": trend_ols, "ols_std": trend_std,
                               **{f"ridge_{k:g}": v for k, v in trend_ridge.items()}},
            "geometry": {f"{g:g}": v for g, v in geom.items()},
            "same_subspace_for_all_g_positive": same_subspace,
            "g_changes_subspace": changes_subspace,
            "g_changes_only_amplitudes": bool(same_subspace and changes_amplitude),
            "verdict": verdict, "m_fixed": m}


# =============================================================================
# 4. Shot noise, the classical product baseline, and IBM-informed noise
# =============================================================================
def shots_local(X: np.ndarray, S: int, rng) -> np.ndarray:
    """S-shot estimate of Pauli expectations: (2B - S)/S, B ~ Binomial(S, (1+p)/2)."""
    p = np.clip((1.0 + X) / 2.0, 0.0, 1.0)
    return (2.0 * rng.binomial(S, p) - S) / S


def joint_estimators(XR, XP, pairs, S: int, rng) -> tuple:
    """Both estimators of <O_R (x) O_P> from the SAME shot budget.

    quantum : mean over shots of the per-shot product of +/-1 outcomes
    classical: product of the two marginal means from those shots
    For a product state the per-shot outcomes are independent, so both are
    unbiased; they differ only in variance.
    """
    T = XR.shape[0]
    q = np.empty((T, len(pairs)))
    c = np.empty((T, len(pairs)))
    for j, (a, b) in enumerate(pairs):
        pa, pb = np.clip((1 + XR[:, a]) / 2, 0, 1), np.clip((1 + XP[:, b]) / 2, 0, 1)
        oa = 2.0 * (rng.random((T, S)) < pa[:, None]) - 1.0
        ob = 2.0 * (rng.random((T, S)) < pb[:, None]) - 1.0
        q[:, j] = (oa * ob).mean(axis=1)
        c[:, j] = oa.mean(axis=1) * ob.mean(axis=1)
    return q, c


def measurement_settings(labels) -> int:
    """Qubit-wise-commuting setting count for a label list like 'R:Z0', 'R:Z0Z1'."""
    groups = []
    for lab in labels:
        body = lab.split(":", 1)[1]
        need, i = {}, 0
        while i < len(body):
            p = body[i]; j = i + 1
            while j < len(body) and body[j].isdigit():
                j += 1
            need[int(body[i + 1:j])] = p
            i = j
        for gset in groups:
            if all(gset.get(q, p) == p for q, p in need.items()):
                gset.update(need)
                break
        else:
            groups.append(dict(need))
    return len(groups)


def ibm_noise_parameters(backend_name: str = "FakeTorino") -> dict:
    """Median error rates from a real IBM calibration snapshot."""
    try:
        from qiskit_ibm_runtime import fake_provider
        be = getattr(fake_provider, backend_name)()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    tgt = be.target

    def med(op):
        errs = []
        if op in tgt.operation_names:
            for qargs, props in tgt[op].items():
                if props is not None and props.error is not None:
                    errs.append(props.error)
        return float(np.median(errs)) if errs else float("nan")

    e1 = med("sx")
    e2 = med("cz") if "cz" in tgt.operation_names else med("ecr")
    ro = med("measure")
    return {"available": True, "backend": backend_name, "median_sx_error": e1,
            "median_2q_error": e2, "median_readout_error": ro,
            "two_qubit_gate": "cz" if "cz" in tgt.operation_names else "ecr"}


def ibm_noisy_adapter(adapter, params: dict, *, shots: int = 4096,
                      n2q_R: int = 12, n1q_R: int = 8, n2q_P: int = 24, n1q_P: int = 16):
    """Effective, calibration-informed noise. NOT a transpiled-circuit simulation.

    Per step: each R rail gets depolarizing p_R = n1q_R*e1 + n2q_R*e2 (spread over
    the rails) inside the memory recursion, so noise ACCUMULATES through the
    memory; each P qubit gets p_P likewise after its unitary. Readout error e_ro
    scales local expectations by (1 - 2 e_ro); finite shots are then applied.
    Gate counts are documented assumptions for a 4-qubit XY step and a depth-2
    ZZ/ZZZZ layer stack, not compiler output.
    """
    from .v4_architecture import MemoryRoute, ProcessorRoute, _joint_pairs
    e1, e2, ro = params["median_sx_error"], params["median_2q_error"], params["median_readout_error"]
    spec = adapter.spec
    L, NP = spec.memory.L_R, spec.processor.N_P
    pR = min(0.5, (n1q_R * e1 + n2q_R * e2) / L)
    pP = min(0.5, (n1q_P * e1 + n2q_P * e2) / NP)
    scale_ro = 1.0 - 2.0 * ro

    class Noisy:
        name = "V4+IBM-informed-noise"

        def __init__(self):
            self.spec = spec
            self.alpha = adapter.alpha
            self.params = {**params, "p_depol_R": pR, "p_depol_P": pP, "shots": shots}

        def run(self, u, m, g, seed):
            R = MemoryRoute(spec.memory, m, seed)
            P = ProcessorRoute(spec.processor, g, seed)
            # SEPARATE streams per route: numpy's binomial consumes a
            # p-dependent number of draws, so one shared stream would make P's
            # noise depend on R's values -- i.e. on m -- and fabricate a
            # cross-effect out of the noise model itself.
            rngR = np.random.default_rng(2 * int(seed) + 5)
            rngP = np.random.default_rng(2 * int(seed) + 6)
            D = spec.memory.dim
            XR, XP = [], []
            for ut in u:
                fr = R.step(float(ut))
                R.rho = (1 - pR) * R.rho + pR * np.eye(D) / D      # accumulates
                XR.append((1 - pR) * fr)
                XP.append((1 - pP) * P.step(float(ut)))
            XR = shots_local(scale_ro * np.array(XR), shots, rngR)
            XP = shots_local(scale_ro * np.array(XP), shots, rngP)
            pairs = _joint_pairs(XR.shape[1], XP.shape[1], spec.n_joint, seed)
            XJ = np.column_stack([XR[:, a] * XP[:, b] for a, b in pairs])
            return {"R": XR, "P": XP, "J": XJ}

    return Noisy()


# =============================================================================
# 5. Invalid controls -- every one MUST be detected
# =============================================================================
def finite_shot_adapter(adapter, shots: int):
    """Local features estimated from `shots` shots; joint features estimated
    DIRECTLY as the per-shot product of +/-1 outcomes (the quantum estimator).
    Separate RNG streams per route, for the reason given in ibm_noisy_adapter."""
    from .v4_architecture import _joint_pairs

    class Shots:
        name = f"V4+{shots}shots"

        def __init__(self):
            self.spec, self.alpha = adapter.spec, adapter.alpha

        def run(self, u, m, g, seed):
            f = adapter.run(u, m, g, seed)
            rngR = np.random.default_rng(3 * int(seed) + 1)
            rngP = np.random.default_rng(3 * int(seed) + 2)
            rngJ = np.random.default_rng(3 * int(seed) + 3)
            pairs = _joint_pairs(f["R"].shape[1], f["P"].shape[1],
                                 self.spec.n_joint, seed)
            q, _ = joint_estimators(f["R"], f["P"], pairs, shots, rngJ)
            return {"R": shots_local(f["R"], shots, rngR),
                    "P": shots_local(f["P"], shots, rngP), "J": q}

    return Shots()


def leak_hook(kind: str):
    """Feature hooks that plant a specific invalidity."""
    def hook(X, *, u, m, g):
        X = dict(X)
        if kind == "leak_targets":
            leaked = np.column_stack([A.legendre(A.shift(u, 3), 2), A.shift(u, -1)])
            X["R"] = np.hstack([X["R"], leaked])
            X["ALL"] = np.hstack([X["ALL"], leaked])
        if kind == "feature_count_changes" and g > 0.5:
            X["P"] = X["P"][:, :-1]
            X["ALL"] = X["ALL"][:, :-1]
        return X
    return hook
