"""
apps_floquet_cirq.py  --  the COMPUTE half of the monitored-reservoir unification.
A non-Clifford critical-phase reservoir is driven by an input stream and monitored
by sewn measurements at rate p (the SAME schedule that, in the Clifford backend,
generates the protecting code).  Features come from the measurement record /
reservoir state.  We show the compute capacity is high in the coding phase
(small p) and collapses under strong monitoring (large p) -- so a window p < p_c
both protects the memory and computes.  Every quantum step is a real Cirq circuit.
"""
import numpy as np, cirq
from sklearn.linear_model import Ridge
import idcpsr_cirq as E

SV = cirq.Simulator(dtype=np.complex128)

def _crit_U(N, seed):
    bz, bx = E.get_bias(N, seed)
    return E.critical_unitary_cirq(N, np.pi * E.GSTAR, bz, bx)

def monitored_reservoir_features(u_seq, N=8, p=0.1, seed=1, rng=None, alpha=0.8*np.pi, reps=2):
    """Drive a monitored non-Clifford reservoir with scalar stream u_seq.
       Input injected on qubit 0 (the rest hold memory); U(g*) applied `reps`
       times; then each qubit measured in Z with probability p (the monitoring).
       Features per step = [<Z_i>,<X_i>] and nearest-neighbour [<Z_iZ_{i+1}>].
       Real Cirq circuits throughout."""
    if rng is None:
        rng = np.random.RandomState(seed)
    qs = cirq.LineQubit.range(N); U = _crit_U(N, seed)
    qmap = {q: i for i, q in enumerate(qs)}
    psi = np.zeros(2 ** N, complex); psi[0] = 1.0
    feats = []
    for u in u_seq:
        enc = cirq.Circuit([cirq.ry(alpha * float(u)).on(qs[0])] + [cirq.I(qs[i]) for i in range(1, N)])
        psi = enc.unitary(qubit_order=qs) @ psi
        for _ in range(reps):
            psi = U @ psi
        psi /= (np.linalg.norm(psi) + 1e-12)
        for i in range(N):
            if rng.uniform() < p:
                P1 = _qubit_p1(psi, N, i)
                out = 1 if rng.uniform() < P1 else 0
                psi = _collapse(psi, N, i, out); psi /= (np.linalg.norm(psi) + 1e-12)
        rho = np.outer(psi, psi.conj()); ft = []
        for i in range(N):
            for P in (cirq.Z, cirq.X):
                ft.append(float(np.real(cirq.PauliString({qs[i]: P})
                          .expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
        for i in range(N - 1):
            ft.append(float(np.real(cirq.PauliString({qs[i]: cirq.Z, qs[i + 1]: cirq.Z})
                      .expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
        feats.append(ft)
    return np.array(feats)

def _qubit_p1(psi, N, i):
    psi = psi.reshape([2] * N)
    ax = tuple(j for j in range(N) if j != i)
    probs = np.sum(np.abs(psi) ** 2, axis=ax)
    return float(probs[1])

def _collapse(psi, N, i, out):
    psi = psi.reshape([2] * N).copy()
    sl = [slice(None)] * N; sl[i] = 1 - out
    psi[tuple(sl)] = 0.0
    return psi.reshape(-1)

def _mc(X, u, k_max=6, washout=30, frac=0.7):
    """Linear memory capacity MC = sum_k corr^2(readout, u_{t-k})."""
    n = len(u); tr = slice(washout, int(frac * n)); te = slice(int(frac * n), n)
    total = 0.0
    for k in range(1, k_max + 1):
        y = np.concatenate([np.zeros(k), u[:-k]])
        m = Ridge(alpha=1e-3).fit(X[tr], y[tr]); pred = m.predict(X[te])
        c = np.corrcoef(pred, y[te])[0, 1]
        total += (c ** 2 if np.isfinite(c) else 0.0)
    return float(total)

def compute_vs_p(ps, N=8, T=320, k_max=6, n_avg=2, seed=0):
    """Reservoir memory capacity as a function of monitoring rate p."""
    out = []
    for p in ps:
        mcs = []
        for a in range(n_avg):
            rng = np.random.RandomState(100 * a + 7)
            u = rng.uniform(-1, 1, T)
            X = monitored_reservoir_features(u, N=N, p=p, seed=seed + a, rng=rng)
            mcs.append(_mc(X, u, k_max=k_max))
        out.append(float(np.mean(mcs)))
    return {'ps': list(ps), 'MC': out, 'N': N}

def nonlinear_task_vs_p(ps, N=8, T=320, n_avg=2, seed=0):
    """Nonlinear (product-memory) task NRMSE vs p: y_t = u_{t-1}*u_{t-2}."""
    out = []
    for p in ps:
        errs = []
        for a in range(n_avg):
            rng = np.random.RandomState(50 * a + 3)
            u = rng.uniform(-1, 1, T)
            y = np.concatenate([np.zeros(2), (u[:-2] * u[1:-1])])
            X = monitored_reservoir_features(u, N=N, p=p, seed=seed + a, rng=rng)
            n = len(u); tr = slice(30, int(0.7 * n)); te = slice(int(0.7 * n), n)
            m = Ridge(alpha=1e-3).fit(X[tr], y[tr]); pred = m.predict(X[te])
            errs.append(np.sqrt(np.mean((pred - y[te]) ** 2) / (np.var(y[te]) + 1e-9)))
        out.append(float(np.mean(errs)))
    return {'ps': list(ps), 'nrmse': out, 'N': N}

# ----------------------------------------------------------------------
#  R3 (robust) -- windowed-encoding monitored reservoir.  Each sample resets
#  to |0...0>, injects a length-W input window (U(g*) between symbols),
#  monitored at rate p; final features = single-qubit <Z>,<X>,<Y>.  A linear
#  read-out then solves a NONLINEAR task.  Accuracy is high in the coding phase
#  and collapses under strong (trivial-phase) monitoring -> compute and
#  protection share the window p < p_c.
# ----------------------------------------------------------------------
def windowed_feats(windows, N=8, p=0.1, seed=1, rng=None, reps=1, alpha=0.9 * np.pi):
    if rng is None:
        rng = np.random.RandomState(seed)
    qs = cirq.LineQubit.range(N); U = _crit_U(N, seed)
    qmap = {q: i for i, q in enumerate(qs)}
    out = []
    for w in windows:
        psi = np.zeros(2 ** N, complex); psi[0] = 1.0
        for u in w:
            enc = cirq.Circuit([cirq.ry(alpha * float(u)).on(qs[0])] + [cirq.I(qs[i]) for i in range(1, N)])
            psi = enc.unitary(qubit_order=qs) @ psi
            for _ in range(reps):
                psi = U @ psi
            psi /= (np.linalg.norm(psi) + 1e-12)
            for i in range(N):
                if rng.uniform() < p:
                    P1 = _qubit_p1(psi, N, i)
                    o = 1 if rng.uniform() < P1 else 0
                    psi = _collapse(psi, N, i, o); psi /= (np.linalg.norm(psi) + 1e-12)
        rho = np.outer(psi, psi.conj()); ft = []
        for i in range(N):
            for P in (cirq.Z, cirq.X, cirq.Y):
                ft.append(float(np.real(cirq.PauliString({qs[i]: P})
                          .expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
        out.append(ft)
    return np.array(out)

def windowed_task_vs_p(ps, N=8, M=400, W=3, n_avg=3, seed=0):
    """Nonlinear parity task (sign-product of a length-W window) accuracy vs p."""
    from sklearn.linear_model import LogisticRegression
    out = []
    for p in ps:
        acc = 0.0
        for a in range(n_avg):
            rng = np.random.RandomState(5 * a + 1)
            Wm = rng.uniform(-1, 1, (M, W))
            y = (np.prod(np.sign(Wm), axis=1) > 0).astype(int)
            X = windowed_feats(Wm, N=N, p=p, seed=seed + 2 + a, rng=np.random.RandomState(7 * a + 3))
            tr = slice(0, int(0.7 * M)); te = slice(int(0.7 * M), M)
            clf = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
            acc += (clf.predict(X[te]) == y[te]).mean() / n_avg
        out.append(float(acc))
    return {'ps': list(ps), 'acc': out, 'N': N, 'W': W}
