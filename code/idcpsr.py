"""
ID-CPSR : Instantaneously-Deep, Sewn Critical-Phase Shadow Reservoir
====================================================================
Engine faithful to the CPSR / QND-RC repo (chinmoybiswasdeep/masters-thesis-cpsr-dvqc)
plus new primitives translating the *sewing technique* and *instantaneous depth*
of Huang, Broughton, Eassa, Neven, Babbush, McClean, "Generative quantum advantage
for classical and quantum problems", arXiv:2509.09033 (2025), into quantum reservoir
computing.

All quantum operations are exact (state-vector or density-matrix); nothing is faked.
This module is pure numpy and serves as the REFERENCE that the Qiskit-native engine
(`idcpsr_qiskit`) is verified against to machine precision.

Convention note: this module is BIG-ENDIAN (qubit 0 = most significant bit), matching
Cirq. The Qiskit engine is little-endian; `reverse_bits()` bridges the two.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

# =============================================================================
#  PART A.  CPSR ENGINE  (faithful numpy reimplementation of the repo's math)
# =============================================================================
def chain_edges(N):
    return [(i, i + 1) for i in range(N - 1)]

def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _kron_layer(mats):
    U = np.array([[1.0]], dtype=np.complex128)
    for m in mats:
        U = np.kron(U, m)
    return U

def critical_unitary(N, g, bias_z, bias_x):
    """One reservoir step unitary U = Rx . Rz . CZ^(2g/pi)  (dim 2^N). Repo-identical."""
    dim = 2 ** N
    edges = chain_edges(N)
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    cz_count = np.zeros(dim)
    for (a, b) in edges:
        cz_count += (bits[:, a] & bits[:, b])
    cz_diag = np.exp(2j * g * cz_count)
    CZ = np.diag(cz_diag)
    rz_diag = np.ones(dim, dtype=np.complex128)
    for i in range(N):
        phase = np.where(bits[:, i] == 0, np.exp(-1j * bias_z[i] / 2),
                         np.exp(+1j * bias_z[i] / 2))
        rz_diag *= phase
    RZ = np.diag(rz_diag)
    RX = _kron_layer([_rx(bias_x[i]) for i in range(N)])
    return RX @ RZ @ CZ

def reservoir_states(N, u_seq, g, bias_z, bias_x, window_size=8, reps=2):
    """(T, 2^N) statevectors. Memoryless QELM: re-encode a sliding window into |0>
       each step, then apply critical step^reps. (Repo 'monolithic' memory source.)"""
    dim = 2 ** N
    U_step = np.linalg.matrix_power(critical_unitary(N, g, bias_z, bias_x), reps)
    slot_phase = np.linspace(0.5, 1.0, window_size)
    slot_to_qubit = [w % N for w in range(window_size)]
    T = len(u_seq)
    states = np.zeros((T, dim), dtype=np.complex128)
    psi0 = np.zeros(dim, dtype=np.complex128); psi0[0] = 1.0
    for t in range(T):
        lo = max(0, t - window_size + 1); wlen = t - lo + 1
        window = np.zeros(window_size); window[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(window):
            per_q[slot_to_qubit[w]] += np.pi * uu * slot_phase[w]
        Uenc = _kron_layer([_ry(per_q[i]) for i in range(N)])
        states[t] = (U_step @ Uenc) @ psi0
    return states

# ---- chaos diagnostics -------------------------------------------------------
def half_system_entropy(state, N):
    half = N // 2
    psi = state.reshape(2 ** half, 2 ** (N - half))
    s = np.linalg.svd(psi, compute_uv=False); p = s ** 2; p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))

def operator_entanglement(U, N):
    dh = 2 ** (N // 2)
    Ut = U.reshape(dh, dh, dh, dh).transpose(0, 2, 1, 3).reshape(dh * dh, dh * dh)
    s = np.linalg.svd(Ut, compute_uv=False); p = (s ** 2) / np.sum(s ** 2); p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))

# ---- classical shadows (exact 1- & 2-body Paulis, variance-correct noise) ----
def shadow_features(states, N, n_shots=0, seed=42, max_weight=2):
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1).astype(np.int8)
    idx = np.arange(dim); labels = []
    one_specs = []
    for i in range(N):
        one_specs.append((i, 1 << (N - 1 - i))); labels += [f'Z{i}', f'X{i}', f'Y{i}']
    two_specs = []
    if max_weight >= 2:
        for i in range(N):
            for j in range(i + 1, N):
                two_specs.append((i, j, (1 << (N - 1 - i)) ^ (1 << (N - 1 - j))))
                labels += [f'Z{i}Z{j}', f'X{i}X{j}', f'Y{i}Y{j}']
    T = states.shape[0]; X = np.zeros((T, len(labels)))
    for t in range(T):
        st = states[t]; probs = np.abs(st) ** 2; col = 0
        for (i, flip) in one_specs:
            z = 1 - 2 * bits[:, i]
            X[t, col] = np.sum(probs * z); col += 1
            X[t, col] = np.real(np.sum(np.conj(st) * st[idx ^ flip])); col += 1
            X[t, col] = np.imag(np.sum(np.conj(st) * st[idx ^ flip] * z)); col += 1
        for (i, j, f2) in two_specs:
            zi, zj = 1 - 2 * bits[:, i], 1 - 2 * bits[:, j]
            X[t, col] = np.sum(probs * zi * zj); col += 1
            X[t, col] = np.real(np.sum(np.conj(st) * st[idx ^ f2])); col += 1
            X[t, col] = np.real(np.sum(zi * zj * np.conj(st) * st[idx ^ f2])); col += 1
    if n_shots > 0:
        rng = np.random.RandomState(seed)
        w = np.array([sum(ch in 'XYZ' for ch in lab) for lab in labels])
        std = np.sqrt((3.0 ** w) / max(n_shots, 1))
        X = X + std[None, :] * rng.randn(*X.shape)
    return labels, X

# ---- tasks -------------------------------------------------------------------
def random_input(T, seed=0):
    return np.random.RandomState(seed).uniform(0, 1, T)

def task_kpauli(T, k, seed=0):
    u = random_input(T, seed); y = np.zeros(T)
    for t in range(k, T):
        p = 1.0
        for j in range(1, k + 1):
            p *= np.cos(np.pi * u[t - j])
        y[t] = p
    return u, y

def task_cos_static(T, seed=0):
    u = random_input(T, seed); return u, np.cos(np.pi * u)

# ---- readout helpers ---------------------------------------------------------
def split(T, washout=30, n_test=100, seed=42):
    te = np.arange(T - n_test, T); pool = np.arange(washout, T - n_test)
    np.random.RandomState(seed + 7).shuffle(pool); return pool, te

def nrmse_ridge(X, y, alpha=1e-4, washout=30, n_test=100, seed=42, n_tr=None):
    pool, te = split(len(y), washout, n_test, seed); tr = pool if n_tr is None else pool[:n_tr]
    m = Ridge(alpha=alpha).fit(X[tr], y[tr]); p = m.predict(X[te]); e = p - y[te]
    return float(np.sqrt(np.mean(e ** 2) / (np.var(y[te]) + 1e-12)))

def nrmse_mlp(X, y, hidden=(64, 32), alpha=1e-3, washout=30, n_test=100, seed=42, n_tr=None):
    pool, te = split(len(y), washout, n_test, seed); tr = pool if n_tr is None else pool[:n_tr]
    m = MLPRegressor(hidden_layer_sizes=hidden, max_iter=500, random_state=seed, alpha=alpha)
    m.fit(X[tr], y[tr]); p = m.predict(X[te]); e = p - y[te]
    return float(np.sqrt(np.mean(e ** 2) / (np.var(y[te]) + 1e-12)))

def r2_ridge(X, y, alpha=1e-4, washout=30, n_test=100, seed=42):
    pool, te = split(len(y), washout, n_test, seed)
    m = Ridge(alpha=alpha).fit(X[pool], y[pool]); p = m.predict(X[te]); ss = np.var(y[te]) + 1e-12
    return float(max(0.0, 1 - np.mean((p - y[te]) ** 2) / ss))

def delay_taps(u, m):
    T = len(u); X = np.zeros((T, m + 1))
    for j in range(m + 1):
        X[j:, j] = u[:T - j]
    return X

def get_bias(N, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 2 * np.pi, N), rng.uniform(0.3, 0.7, N)

def memory_capacity(X, u, k_max=14, alpha=1e-6, washout=30, n_test=100, seed=42):
    T = len(u); pool, te = split(T, washout, n_test, seed); tot = 0.0; perk = []
    for k in range(1, k_max + 1):
        tgt = np.zeros(T); tgt[k:] = u[:T - k]
        mdl = Ridge(alpha=alpha).fit(X[pool], tgt[pool]); p = mdl.predict(X[te]); tr = tgt[te]
        c = np.cov(p, tr)[0, 1] ** 2; d = np.var(tr) * np.var(p) + 1e-12
        v = float(min(1.0, c / d)); perk.append(v); tot += v
    return tot, perk

# =============================================================================
#  PART B.  SINGLE-QUBIT / PAULI TOOLBOX  (for sewing + density matrices)
# =============================================================================
I2 = np.eye(2, dtype=np.complex128)
X1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y1 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z1 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H1 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
PAULI = {'I': I2, 'X': X1, 'Y': Y1, 'Z': Z1}

def op_on(N, ops):
    """Tensor an operator given as dict {qubit: 2x2}; identity elsewhere. Big-endian."""
    mats = [ops.get(i, I2) for i in range(N)]
    return _kron_layer(mats)

def cnot(N, ctrl, targ):
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    perm = np.arange(dim)
    flip = (bits[:, ctrl] == 1)
    perm[flip] = perm[flip] ^ (1 << (N - 1 - targ))
    U = np.zeros((dim, dim), dtype=np.complex128); U[perm, np.arange(dim)] = 1.0
    return U

def controlled_basis_copy(N, sys, anc, basis='Z'):
    """Coherently copy the eigenvalue of Pauli `basis` on qubit `sys` onto ancilla `anc`
       (a von-Neumann pre-measurement / sewing stitch)."""
    pre = np.eye(2 ** N, dtype=np.complex128)
    if basis == 'X':
        pre = op_on(N, {sys: H1})
    elif basis == 'Y':
        Sd = np.array([[1, 0], [0, -1j]], dtype=np.complex128)  # S^dagger
        pre = op_on(N, {sys: H1}) @ op_on(N, {sys: Sd})
    cx = cnot(N, sys, anc)
    return pre.conj().T @ cx @ pre

# =============================================================================
#  PART C.  PERSISTENT (recurrent) CPSR with QUANTUM memory + readout channels
# =============================================================================
def _reset_qubit0(N):
    """Kraus ops that reset qubit 0 to |0> (discard its state) -> fading memory."""
    K0 = op_on(N, {0: np.array([[1, 0], [0, 0]], dtype=np.complex128)})
    K1 = op_on(N, {0: np.array([[0, 1], [0, 0]], dtype=np.complex128)})
    return [K0, K1]

def _apply_kraus(rho, Ks):
    return sum(K @ rho @ K.conj().T for K in Ks)

def _local_dephase(rho, N, qubit, basis):
    """Back-action of reading Pauli `basis` on `qubit` via a sewn ancilla:
       coherent copy to ancilla + ancilla discard == dephasing in that eigenbasis."""
    if basis == 'Z':   P = op_on(N, {qubit: Z1})
    elif basis == 'X': P = op_on(N, {qubit: X1})
    else:              P = op_on(N, {qubit: Y1})
    return 0.5 * (rho + P @ rho @ P)

def _full_projective(rho, N, bases):
    """Full destructive readout of the memory register in product bases `bases`."""
    rot = np.eye(2 ** N, dtype=np.complex128)
    for q, b in bases.items():
        if b == 'X':   rot = op_on(N, {q: H1}) @ rot
        elif b == 'Y':
            Sd = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
            rot = op_on(N, {q: H1}) @ op_on(N, {q: Sd}) @ rot
    r = rot @ rho @ rot.conj().T
    d = np.diag(np.diag(r))
    return rot.conj().T @ d @ rot

def _local_paulis(N, qubits):
    """exact 1-body X,Y,Z expectation operators on the given qubits + adjacent ZZ."""
    ops = []; labels = []
    for q in qubits:
        ops += [op_on(N, {q: Z1}), op_on(N, {q: X1}), op_on(N, {q: Y1})]
        labels += [f'Z{q}', f'X{q}', f'Y{q}']
    for a, b in zip(qubits[:-1], qubits[1:]):
        ops.append(op_on(N, {a: Z1, b: Z1})); labels.append(f'Z{a}Z{b}')
    return labels, ops

def persistent_reservoir(N, u_seq, g, bias_z, bias_x, reps=2, readout='sewn',
                         n_readout=2, seed=42):
    """Persistent quantum reservoir with a coherent memory register.
       readout in {'none','sewn','projective'}."""
    dim = 2 ** N
    U = np.linalg.matrix_power(critical_unitary(N, g, bias_z, bias_x), reps)
    Ks = _reset_qubit0(N)
    mem_qubits = list(range(1, N))
    feat_labels, feat_ops = _local_paulis(N, mem_qubits)
    readout_qubits = mem_qubits[:n_readout]
    rng = np.random.RandomState(seed)
    T = len(u_seq); X = np.zeros((T, len(feat_ops)))
    rho = np.zeros((dim, dim), dtype=np.complex128); rho[0, 0] = 1.0
    for t in range(T):
        rho = _apply_kraus(rho, Ks)
        E = op_on(N, {0: _ry(np.pi * u_seq[t])})
        rho = E @ rho @ E.conj().T
        rho = U @ rho @ U.conj().T
        for c, Op in enumerate(feat_ops):
            X[t, c] = np.real(np.trace(rho @ Op))
        if readout == 'sewn':
            for q in readout_qubits:
                b = ['X', 'Y', 'Z'][rng.randint(3)]
                rho = _local_dephase(rho, N, q, b)
        elif readout == 'projective':
            bases = {q: ['X', 'Y', 'Z'][rng.randint(3)] for q in mem_qubits}
            rho = _full_projective(rho, N, bases)
    return feat_labels, X

def purity_trace(rho):
    return float(np.real(np.trace(rho @ rho)))

# =============================================================================
#  PART D.  LOCAL-INVERSION SEWING & BARREN-PLATEAU ELIMINATION
# =============================================================================
def param_reservoir_unitary(N, theta, depth, edge_g):
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    U = np.eye(dim, dtype=np.complex128)
    for d in range(depth):
        Ry = _kron_layer([_ry(theta[d, i]) for i in range(N)])
        U = Ry @ U
        cz = np.zeros(dim)
        for e, (a, b) in enumerate(chain_edges(N)):
            cz += edge_g[d, e] * (bits[:, a] & bits[:, b])
        U = np.diag(np.exp(2j * cz)) @ U
    return U

def cost_and_grad_var(N, depth, observable='global', n_samples=80, seed=0):
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    if observable == 'global':
        zglob = np.prod(1 - 2 * bits, axis=1); O = np.diag(zglob.astype(np.complex128))
    else:
        zsum = (1 - 2 * bits).mean(axis=1); O = np.diag(zsum.astype(np.complex128))
    rng = np.random.RandomState(seed); psi0 = np.zeros(dim, complex); psi0[0] = 1.0
    grads = []
    s = np.pi / 2
    for _ in range(n_samples):
        th = rng.uniform(0, 2 * np.pi, (depth, N))
        eg = rng.uniform(0, np.pi / 2, (depth, max(N - 1, 1)))
        def cost(theta):
            U = param_reservoir_unitary(N, theta, depth, eg); v = U @ psi0
            return float(np.real(v.conj() @ O @ v))
        thp = th.copy(); thp[0, 0] += s
        thm = th.copy(); thm[0, 0] -= s
        grads.append((cost(thp) - cost(thm)) / 2.0)
    return float(np.var(grads))

# =============================================================================
#  PART E.  SEWING IDENTITY (deferred measurement)  &  EFFECTIVE-DEPTH GADGET
# =============================================================================
def sewn_readout_channel_check(N, rho, qubit, basis, seed=0):
    """Verify: (coherent copy onto ancilla, then discard) == local dephasing."""
    rho_big = np.kron(rho, np.array([[1, 0], [0, 0]], dtype=np.complex128))
    S = controlled_basis_copy(N + 1, qubit, N, basis=basis)
    rho_big = S @ rho_big @ S.conj().T
    r = rho_big.reshape(2 ** N, 2, 2 ** N, 2)
    rho_anc_traced = r[:, 0, :, 0] + r[:, 1, :, 1]
    rho_deph = _local_dephase(rho, N, qubit, basis)
    return float(np.linalg.norm(rho_anc_traced - rho_deph))

def effective_depth_opent(N, g, bias_z, bias_x, depths):
    U = critical_unitary(N, g, bias_z, bias_x)
    return [operator_entanglement(np.linalg.matrix_power(U, D), N) for D in depths]

# ---- efficient state-vector parameterized reservoir ----
def _apply_ry_sv(psi, N, q, theta):
    psi = psi.reshape([2] * N)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    a = psi.take(0, axis=q); b = psi.take(1, axis=q)
    out = np.stack([c * a - s * b, s * a + c * b], axis=q)
    return out.reshape(-1)

def _edge_masks(N):
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    return [(bits[:, a] & bits[:, b]).astype(float) for (a, b) in chain_edges(N)]

def _apply_cz_phase_sv(psi, N, masks, edge_g):
    cz = np.zeros(psi.shape[0])
    for e, mask in enumerate(masks):
        cz += edge_g[e] * mask
    return psi * np.exp(2j * cz)

def cost_grad_var_sv(N, depth, observable='global', n_samples=120, seed=0):
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    if observable == 'global':
        Odiag = np.prod(1 - 2 * bits, axis=1).astype(float)
    else:
        Odiag = (1 - 2 * bits).mean(axis=1).astype(float)
    masks = _edge_masks(N); rng = np.random.RandomState(seed); grads = []
    def run(theta, eg):
        psi = np.zeros(dim, complex); psi[0] = 1.0
        for d in range(depth):
            for i in range(N):
                psi = _apply_ry_sv(psi, N, i, theta[d, i])
            psi = _apply_cz_phase_sv(psi, N, masks, eg[d])
        return float(np.sum(Odiag * np.abs(psi) ** 2))
    s = np.pi / 2
    for _ in range(n_samples):
        th = rng.uniform(0, 2 * np.pi, (depth, N)); eg = rng.uniform(0, np.pi / 2, (depth, N - 1))
        thp = th.copy(); thp[0, 0] += s; thm = th.copy(); thm[0, 0] -= s
        grads.append((run(thp, eg) - run(thm, eg)) / 2.0)
    return float(np.var(grads))

def cost_grad_var_sv2(N, depth, observable='global', n_samples=120, seed=0, qb=0):
    """Gradient variance w.r.t. a LAST-LAYER parameter theta[depth-1, qb]."""
    dim = 2 ** N
    bits = ((np.arange(dim)[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    if observable == 'global':
        Odiag = np.prod(1 - 2 * bits, axis=1).astype(float)
    else:
        Odiag = (1 - 2 * bits[:, qb]).astype(float)
    masks = _edge_masks(N); rng = np.random.RandomState(seed); grads = []
    def run(theta, eg):
        psi = np.zeros(dim, complex); psi[0] = 1.0
        for d in range(depth):
            for i in range(N):
                psi = _apply_ry_sv(psi, N, i, theta[d, i])
            psi = _apply_cz_phase_sv(psi, N, masks, eg[d])
        return float(np.sum(Odiag * np.abs(psi) ** 2))
    s = np.pi / 2
    for _ in range(n_samples):
        th = rng.uniform(0, 2 * np.pi, (depth, N)); eg = rng.uniform(0, np.pi / 2, (depth, N - 1))
        thp = th.copy(); thp[depth - 1, qb] += s; thm = th.copy(); thm[depth - 1, qb] -= s
        grads.append((run(thp, eg) - run(thm, eg)) / 2.0)
    return float(np.var(grads))

# =============================================================================
#  PART F.  INSTANTANEOUS DEPTH vs PHYSICAL DEPTH under NOISE
# =============================================================================
def _depolarize(rho, N, p):
    if p <= 0: return rho
    dim = 2 ** N
    return (1 - p) * rho + p * np.eye(dim, dtype=np.complex128) / dim

def noisy_reservoir_features(N, u_seq, g, bias_z, bias_x, D_eff, mode='deep',
                             p=0.02, window=4):
    dim = 2 ** N
    U1 = critical_unitary(N, g, bias_z, bias_x)
    UD = np.linalg.matrix_power(U1, D_eff)
    slot_phase = np.linspace(0.5, 1.0, window); slot_to_qubit = [w % N for w in range(window)]
    T = len(u_seq)
    labels, ops = _local_paulis(N, list(range(N)))
    X = np.zeros((T, len(ops)))
    for t in range(T):
        lo = max(0, t - window + 1); wlen = t - lo + 1
        win = np.zeros(window); win[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(win):
            per_q[slot_to_qubit[w]] += np.pi * uu * slot_phase[w]
        Uenc = _kron_layer([_ry(per_q[i]) for i in range(N)])
        rho = np.zeros((dim, dim), complex); rho[0, 0] = 1.0
        rho = Uenc @ rho @ Uenc.conj().T
        if mode == 'deep':
            for _ in range(D_eff):
                rho = U1 @ rho @ U1.conj().T
                rho = _depolarize(rho, N, p)
        else:
            rho = UD @ rho @ UD.conj().T
            rho = _depolarize(rho, N, p)
        for c, Op in enumerate(ops):
            X[t, c] = np.real(np.trace(rho @ Op))
    return labels, X

# ---- repo QND-RC feature builders (numpy reference) ----
GSTAR = 0.30

def monolithic_features(N, u, g, bias_z, bias_x, W_mono=8, n_shots=0, seed=42):
    states = reservoir_states(N, u, g, bias_z, bias_x, window_size=W_mono)
    return shadow_features(states, N, n_shots=n_shots, seed=seed)[1]

def classical_qndrc_features(N, u, g, bias_z, bias_x, m, W_q=2, n_shots=0, seed=42):
    S = shadow_features(reservoir_states(N, u, g, bias_z, bias_x, window_size=W_q),
                        N, n_shots, seed)[1]
    return np.concatenate([S, delay_taps(u, m)], axis=1)

def qmem_features(N, u, g, bias_z, bias_x, n_readout=2, reps=1, seed=42):
    return persistent_reservoir(N, u, np.pi * g, bias_z, bias_x, reps=reps,
                                readout='sewn', n_readout=n_readout, seed=seed)[1]

def idqndrc_features(N, u, g, bias_z, bias_x, m=2, n_readout=2, reps=1, seed=42):
    Q = qmem_features(N, u, g, bias_z, bias_x, n_readout, reps, seed)
    return np.concatenate([Q, delay_taps(u, m)], axis=1)
