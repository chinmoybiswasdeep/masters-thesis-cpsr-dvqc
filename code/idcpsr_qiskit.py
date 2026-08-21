"""
idcpsr_qiskit.py  —  Qiskit-native ID-CPSR engine  (port of the Cirq/Willow engine)
==================================================================================
Every quantum operation is a real `qiskit.QuantumCircuit`, executed with
`qiskit.quantum_info.Statevector` (closed system) or `DensityMatrix` (open system /
measurement back-action), and the processing circuit transpiles to 100% IBM-native
gates and runs on an IBM backend (Heron `cz / rz / sx / x` or Eagle `ecr / rz / sx / x`).

Gate-convention map  Cirq -> Qiskit
-----------------------------------
    cirq.ry(t).on(q)                ->  qc.ry(t, q)
    cirq.rz(t).on(q)                ->  qc.rz(t, q)          (both = exp(-i t Z/2))
    cirq.rx(t).on(q)                ->  qc.rx(t, q)
    (cirq.CZ ** (2g/pi)).on(a,b)    ->  qc.cp(2*g, a, b)     (both = diag(1,1,1,e^{2ig}))
    cirq.H / cirq.S / cirq.S**-1    ->  qc.h / qc.s / qc.sdg
    cirq.CNOT(a,b)                  ->  qc.cx(a,b)
    cirq.reset(q)                   ->  qc.reset(q)   /  DensityMatrix.reset([q])
    cirq.phase_flip(0.5)            ->  Kraus([I,Z]/sqrt2)
    cirq.depolarize(p)              ->  Kraus([sqrt(1-p)I, sqrt(p/3)X, sqrt(p/3)Y, sqrt(p/3)Z])
    cirq.Simulator                  ->  quantum_info.Statevector
    cirq.DensityMatrixSimulator     ->  quantum_info.DensityMatrix
    cg.GoogleCZTargetGateset        ->  backend.target  (IBM Heron: cz, rz, sx, x)
    willow_pink QVM                 ->  AerSimulator.from_backend(FakeTorino()) /
                                        QiskitRuntimeService real hardware

Endianness: Cirq is big-endian (qubit 0 = MSB), Qiskit is little-endian (qubit 0 = LSB).
Dense-matrix comparisons therefore go through `circuit.reverse_bits()`; expectation
values use `qargs=` indices and are endianness-free.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, DensityMatrix, Operator, Kraus, Pauli
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

GSTAR = 0.30                       # edge-of-chaos critical phase g*/pi (repo)

# =============================================================================
#  PART A.  QISKIT CRITICAL STEP  (real circuit; IBM-native processing layer)
# =============================================================================
def chain_edges(N):
    return [(i, i + 1) for i in range(N - 1)]

def critical_step_circuit(N, g, bias_z, bias_x, qubits=None):
    """One reservoir step as a Qiskit circuit:  U = Rx . Rz . CZ^(2g/pi).

    Applied left-to-right as a circuit, so the CZ brick goes FIRST, exactly as in
    the Cirq/Willow original.  CZ^(2g/pi) == CPhase(2g).
    """
    qc = QuantumCircuit(N, name='U_crit')
    qs = list(range(N)) if qubits is None else qubits
    for (a, b) in chain_edges(N):
        qc.cp(2.0 * g, qs[a], qs[b])          # entangle  (Willow: native CZ**t)
    for i in range(N):
        qc.rz(bias_z[i], qs[i])               # disorder
    for i in range(N):
        qc.rx(bias_x[i], qs[i])
    return qc

def critical_unitary_qiskit(N, g, bias_z, bias_x):
    """Dense BIG-ENDIAN unitary of one Qiskit step (matches the numpy reference)."""
    qc = critical_step_circuit(N, g, bias_z, bias_x)
    return Operator(qc.reverse_bits()).data

def get_bias(N, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 2 * np.pi, N), rng.uniform(0.3, 0.7, N)

# ---- memoryless (QELM) reservoir states + classical-shadow feature map -------
def reservoir_states_qiskit(N, u_seq, g, bias_z, bias_x, window_size=8, reps=2):
    """(T, 2^N) Qiskit statevectors. Memoryless QELM: re-encode a sliding window
       into |0..0> each step, then apply the critical step^reps.
       (The repo's 'monolithic' memory source.)  Little-endian (Qiskit) ordering."""
    step = critical_step_circuit(N, g, bias_z, bias_x)
    U = np.linalg.matrix_power(np.asarray(Operator(step).data), reps)
    slot_phase = np.linspace(0.5, 1.0, window_size)
    slot_to_qubit = [w % N for w in range(window_size)]
    T = len(u_seq)
    states = np.zeros((T, 2 ** N), dtype=np.complex128)
    for t in range(T):
        lo = max(0, t - window_size + 1); wlen = t - lo + 1
        window = np.zeros(window_size); window[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(window):
            per_q[slot_to_qubit[w]] += np.pi * uu * slot_phase[w]
        enc = QuantumCircuit(N)
        for i in range(N):
            enc.ry(per_q[i], i)
        states[t] = U @ np.asarray(Statevector.from_instruction(enc).data)
    return states

def shadow_features_qiskit(states, N, n_shots=0, seed=42, max_weight=2):
    """Classical-shadow feature map: exact 1- and 2-body Pauli expectations with
       variance-correct shot noise  std = sqrt(3^w / n_shots)  for a weight-w Pauli
       (the randomized-Pauli-measurement shadow variance bound of Huang et al.).
       Operates on LITTLE-ENDIAN (Qiskit) statevectors."""
    dim = 2 ** N
    idx = np.arange(dim)
    bits = ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)   # qiskit order
    labels = []
    one_specs = []
    for i in range(N):
        one_specs.append((i, 1 << i)); labels += [f'Z{i}', f'X{i}', f'Y{i}']
    two_specs = []
    if max_weight >= 2:
        for i in range(N):
            for j in range(i + 1, N):
                two_specs.append((i, j, (1 << i) ^ (1 << j)))
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

def half_system_entropy(state, N):
    """Half-system von-Neumann entanglement entropy of a statevector."""
    half = N // 2
    psi = np.asarray(state).reshape(2 ** half, 2 ** (N - half))
    s = np.linalg.svd(psi, compute_uv=False); p = s ** 2; p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))

def operator_entanglement(U, N):
    """Operator entanglement entropy of a 2^N x 2^N unitary across the half cut."""
    dh = 2 ** (N // 2)
    Ut = np.asarray(U).reshape(dh, dh, dh, dh).transpose(0, 2, 1, 3).reshape(dh * dh, dh * dh)
    s = np.linalg.svd(Ut, compute_uv=False); p = (s ** 2) / np.sum(s ** 2); p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))

# ---- shadow-based (repo QND-RC) feature builders, Qiskit-native -------------
def monolithic_shadow_features(N, u, g, bias_z, bias_x, W_mono=8, n_shots=0, seed=42):
    """Monolithic CPSR: one block does BOTH memory (long encode window) and
       nonlinear processing.  Full 1+2-body classical-shadow read-out."""
    states = reservoir_states_qiskit(N, u, g, bias_z, bias_x, window_size=W_mono)
    return shadow_features_qiskit(states, N, n_shots=n_shots, seed=seed)[1]

def classical_qndrc_shadow_features(N, u, g, bias_z, bias_x, m, W_q=2,
                                    n_shots=0, seed=42):
    """QND-RC: short quantum window (nonlinearity dial g) (+) classical delay line
       (memory dial m).  Memory and processing are DECOUPLED."""
    S = shadow_features_qiskit(
        reservoir_states_qiskit(N, u, g, bias_z, bias_x, window_size=W_q),
        N, n_shots, seed)[1]
    return np.concatenate([S, delay_taps(u, m)], axis=1)

# ---- tasks (repo-identical) -------------------------------------------------
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

# ---- readout / metrics (repo-identical) -------------------------------------
def split(T, washout=30, n_test=100, seed=42):
    te = np.arange(T - n_test, T); pool = np.arange(washout, T - n_test)
    np.random.RandomState(seed + 7).shuffle(pool); return pool, te

def nrmse_ridge(X, y, alpha=1e-4, washout=30, n_test=100, seed=42):
    pool, te = split(len(y), washout, n_test, seed)
    m = Ridge(alpha=alpha).fit(X[pool], y[pool]); p = m.predict(X[te]); e = p - y[te]
    return float(np.sqrt(np.mean(e ** 2) / (np.var(y[te]) + 1e-12)))

def nrmse_mlp(X, y, hidden=(64, 32), alpha=1e-3, washout=30, n_test=100, seed=42):
    pool, te = split(len(y), washout, n_test, seed)
    m = MLPRegressor(hidden_layer_sizes=hidden, max_iter=500, random_state=seed, alpha=alpha)
    m.fit(X[pool], y[pool]); p = m.predict(X[te]); e = p - y[te]
    return float(np.sqrt(np.mean(e ** 2) / (np.var(y[te]) + 1e-12)))

def delay_taps(u, m):
    T = len(u); X = np.zeros((T, m + 1))
    for j in range(m + 1):
        X[j:, j] = u[:T - j]
    return X

def memory_capacity(X, u, k_max=8, alpha=1e-6, washout=30, n_test=100, seed=42):
    T = len(u); pool, te = split(T, washout, n_test, seed); tot = 0.0; perk = []
    for k in range(1, k_max + 1):
        tgt = np.zeros(T); tgt[k:] = u[:T - k]
        mdl = Ridge(alpha=alpha).fit(X[pool], tgt[pool]); p = mdl.predict(X[te]); tr = tgt[te]
        c = np.cov(p, tr)[0, 1] ** 2; d = np.var(tr) * np.var(p) + 1e-12
        v = float(min(1.0, c / d)); perk.append(v); tot += v
    return tot, perk

# =============================================================================
#  PART B.  SEWN ANCILLA READ-OUT  (real Qiskit circuit)  +  CHANNEL EQUIVALENT
#  Physical circuit (runs on IBM): ancilla|0> -> basis-copy from system qubit ->
#  measure ancilla -> reset.  By deferred measurement this equals LOCAL dephasing
#  of that qubit in the chosen Pauli basis -> memory register stays coherent.
# =============================================================================
def _apply_basis_change(qc, q, basis, inverse=False):
    """Ops mapping the Pauli `basis` eigenbasis <-> Z basis on qubit q."""
    if basis == 'Z':
        return
    if basis == 'X':
        qc.h(q); return
    if basis == 'Y':
        # forward (basis -> Z): S^dagger then H ;  inverse (Z -> basis): H then S
        if inverse:
            qc.h(q); qc.s(q)
        else:
            qc.sdg(q); qc.h(q)
        return
    raise ValueError(basis)

def append_sewn_copy(qc, system_qubit, ancilla, basis):
    """Coherent von-Neumann copy of <P_basis(system)> onto a fresh ancilla
       (no measurement). Discarding / tracing the ancilla == local dephasing."""
    _apply_basis_change(qc, system_qubit, basis, inverse=False)   # basis -> Z
    qc.cx(system_qubit, ancilla)                                  # copy / 'sew'
    _apply_basis_change(qc, system_qubit, basis, inverse=True)    # Z -> basis
    return qc

def sewn_copy_circuit(n_qubits, system_qubit, ancilla, basis):
    qc = QuantumCircuit(n_qubits)
    return append_sewn_copy(qc, system_qubit, ancilla, basis)

# --- channel-equivalent form (deferred-measurement identity) ------------------
_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

PHASE_FLIP_HALF = Kraus([np.sqrt(0.5) * _I2, np.sqrt(0.5) * _Z2])   # (rho + Z rho Z)/2

def depolarize_kraus(p):
    """1-qubit depolarizing channel, Cirq convention: (1-p)rho + (p/3)(XrX+YrY+ZrZ)."""
    return Kraus([np.sqrt(1 - p) * _I2, np.sqrt(p / 3) * _X2,
                  np.sqrt(p / 3) * _Y2, np.sqrt(p / 3) * _Z2])

def apply_sewn_dephase(rho, q, basis):
    """Channel form of the physical sewn read-out: local dephasing of qubit q in
       `basis`.  Implemented as (basis->Z) . phase_flip(0.5) . (Z->basis)."""
    pre = QuantumCircuit(rho.num_qubits)
    _apply_basis_change(pre, q, basis, inverse=False)
    post = QuantumCircuit(rho.num_qubits)
    _apply_basis_change(post, q, basis, inverse=True)
    r = rho.evolve(pre) if pre.size() else rho
    r = r.evolve(PHASE_FLIP_HALF, [q])
    r = r.evolve(post) if post.size() else r
    return r

def apply_full_projective(rho, mem_qubits, bases):
    """Destructive read-out of the whole memory register in product bases:
       rotate each measured qubit to Z, dephase FULLY in the computational basis
       (kills inter-qubit coherence too), rotate back."""
    n = rho.num_qubits
    pre = QuantumCircuit(n); post = QuantumCircuit(n)
    for q in mem_qubits:
        _apply_basis_change(pre, q, bases[q], inverse=False)
        _apply_basis_change(post, q, bases[q], inverse=True)
    r = rho.evolve(pre) if pre.size() else rho
    d = np.asarray(r.data)
    # full dephasing on the measured subsystem: keep only blocks diagonal in mem bits
    keep = np.zeros(2 ** n, dtype=np.int64)
    for q in mem_qubits:
        keep |= (1 << q)                       # qiskit little-endian: qubit q -> bit q
    idx = np.arange(2 ** n)
    mask = (idx[:, None] & keep) == (idx[None, :] & keep)
    d = d * mask
    r = DensityMatrix(d, dims=r.dims())
    r = r.evolve(post) if post.size() else r
    return r

def deferred_measurement_check(N=3, seed=0):
    """Verify in Qiskit: tracing out the copied ancilla in the physical sewn
       read-out == applying the local dephasing channel.  Returns max ||.||_F."""
    from qiskit.quantum_info import random_unitary, partial_trace
    rng = np.random.RandomState(seed)
    prep = QuantumCircuit(N)
    prep.append(UnitaryGate(random_unitary(2 ** N, seed=int(rng.randint(1 << 30)))),
                range(N))
    rho_sys = DensityMatrix.from_instruction(prep)
    worst = 0.0
    for q in range(N):
        for b in 'XYZ':
            # path 1: coherent copy onto ancilla q=N, then TRACE OUT the ancilla
            big = QuantumCircuit(N + 1)
            big.compose(prep, range(N), inplace=True)
            append_sewn_copy(big, q, N, b)
            rho_big = DensityMatrix.from_instruction(big)
            sys1 = partial_trace(rho_big, [N])
            # path 2: channel form on the system only
            sys2 = apply_sewn_dephase(rho_sys, q, b)
            worst = max(worst, float(np.linalg.norm(np.asarray(sys1.data) -
                                                    np.asarray(sys2.data))))
    return worst

# =============================================================================
#  PART C.  PERSISTENT (recurrent) QISKIT RESERVOIR with coherent quantum memory
#  qubit 0 = input register (reset+encode each step); qubits 1..N-1 = coherent
#  memory M (never directly measured under 'sewn'); features = local Paulis.
#  Executed with quantum_info.DensityMatrix (exact open-system back-action).
# =============================================================================
def _pauli_expectations(rho, mem_qubits):
    """<Z>,<X>,<Y> on each memory qubit + adjacent <ZZ>, from a density matrix."""
    feats, labels = [], []
    for q in mem_qubits:
        for nm in ('Z', 'X', 'Y'):
            feats.append(float(np.real(rho.expectation_value(Pauli(nm), [q]))))
            labels.append(f'{nm}{q}')
    for a, b in zip(mem_qubits[:-1], mem_qubits[1:]):
        feats.append(float(np.real(rho.expectation_value(Pauli('ZZ'), [a, b]))))
        labels.append(f'Z{a}Z{b}')
    return labels, feats

def persistent_reservoir_qiskit(N, u_seq, g, bias_z, bias_x, reps=1,
                                readout='sewn', n_readout=2, seed=42):
    """Persistent quantum reservoir as a Qiskit DensityMatrix evolution.
       readout in {'none','sewn','projective'} selects the post-step channel.
         none       : record features, no back-action (coherent-memory ceiling)
         sewn       : tap `n_readout` memory qubits via ancillas -> only those dephase
         projective : destructively measure the whole memory register -> collapse
       Returns (labels, feature matrix (T,F))."""
    mem = list(range(1, N))
    read_qubits = mem[:n_readout]
    rng = np.random.RandomState(seed)

    step = critical_step_circuit(N, g, bias_z, bias_x)
    if reps > 1:
        s1 = step.copy(); step = QuantumCircuit(N)
        for _ in range(reps):
            step.compose(s1, inplace=True)
    step_op = Operator(step)                              # cache: one dense op reused

    rho = DensityMatrix.from_label('0' * N)
    T = len(u_seq); labels = None; X = []
    for t in range(T):
        rho = rho.reset([0])                              # discard input qubit -> fading memory
        enc = QuantumCircuit(N); enc.ry(np.pi * u_seq[t], 0)
        rho = rho.evolve(enc)                             # encode input
        rho = rho.evolve(step_op)                         # critical reservoir step
        lab, ft = _pauli_expectations(rho, mem)
        labels = lab; X.append(ft)
        # post-read back-action channel (exact)
        if readout == 'sewn':
            for q in read_qubits:
                b = ['X', 'Y', 'Z'][rng.randint(3)]       # randomized-Pauli (shadow) basis
                rho = apply_sewn_dephase(rho, q, b)
        elif readout == 'projective':
            bases = {q: ['X', 'Y', 'Z'][rng.randint(3)] for q in mem}
            rho = apply_full_projective(rho, mem, bases)
        # 'none': no back-action
    return labels, np.array(X)

# =============================================================================
#  PART D.  INSTANTANEOUS DEPTH under Qiskit noise  (depth<->width trade)
#   deep : D physical noisy layers            -> noise compounds with D
#   id   : same U^D action, ONE noisy layer   -> O(1) physical noise + ancilla width
# =============================================================================
def noisy_reservoir_features_qiskit(N, u_seq, g, bias_z, bias_x, D_eff,
                                    mode='deep', p=0.05, window=4):
    step = critical_step_circuit(N, g, bias_z, bias_x)
    U1 = Operator(step)
    UD = Operator(np.linalg.matrix_power(np.asarray(U1.data), D_eff))
    depol = depolarize_kraus(p)
    slot_phase = np.linspace(0.5, 1.0, window); slot_to_q = [w % N for w in range(window)]
    T = len(u_seq); X = []
    for t in range(T):
        lo = max(0, t - window + 1); wlen = t - lo + 1
        win = np.zeros(window); win[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(win):
            per_q[slot_to_q[w]] += np.pi * uu * slot_phase[w]
        enc = QuantumCircuit(N)
        for i in range(N):
            enc.ry(per_q[i], i)
        rho = DensityMatrix.from_instruction(enc)
        if mode == 'deep':
            for _ in range(D_eff):
                rho = rho.evolve(U1)
                for q in range(N):
                    rho = rho.evolve(depol, [q])
        else:                                   # 'id': deep action, O(1) physical noise
            rho = rho.evolve(UD)
            for q in range(N):
                rho = rho.evolve(depol, [q])
        ft = []
        for q in range(N):
            for nm in ('Z', 'X', 'Y'):
                ft.append(float(np.real(rho.expectation_value(Pauli(nm), [q]))))
        X.append(ft)
    return np.array(X)

def operator_entanglement_qiskit(N, g, bias_z, bias_x, D):
    U = np.linalg.matrix_power(critical_unitary_qiskit(N, g, bias_z, bias_x), D)
    dh = 2 ** (N // 2)
    Ut = U.reshape(dh, dh, dh, dh).transpose(0, 2, 1, 3).reshape(dh * dh, dh * dh)
    s = np.linalg.svd(Ut, compute_uv=False); pr = (s ** 2) / np.sum(s ** 2); pr = pr[pr > 1e-12]
    return float(-np.sum(pr * np.log(pr)))

# ---- integrated feature builders (Qiskit) ----
def monolithic_features_qiskit(N, u, g, bias_z, bias_x, W=8):
    """Memoryless QELM features (sliding-window encode) via Qiskit state vectors."""
    U = Operator(critical_step_circuit(N, g, bias_z, bias_x))
    slot_phase = np.linspace(0.5, 1.0, W); slot_to_q = [w % N for w in range(W)]
    T = len(u); X = []
    for t in range(T):
        lo = max(0, t - W + 1); wlen = t - lo + 1
        win = np.zeros(W); win[-wlen:] = u[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(win):
            per_q[slot_to_q[w]] += np.pi * uu * slot_phase[w]
        enc = QuantumCircuit(N)
        for i in range(N):
            enc.ry(per_q[i], i)
        psi = Statevector.from_instruction(enc).evolve(U)
        ft = []
        for q in range(N):
            for nm in ('Z', 'X', 'Y'):
                ft.append(float(np.real(psi.expectation_value(Pauli(nm), [q]))))
        X.append(ft)
    return np.array(X)

def qmem_features_qiskit(N, u, g, bias_z, bias_x, n_readout=3, reps=1, seed=42):
    """Sewn quantum coherent-memory features (persistent reservoir, collapse-free
       ancilla read-out).  `g` in RADIANS (pass np.pi*GSTAR)."""
    return persistent_reservoir_qiskit(N, u, g, bias_z, bias_x, reps=reps,
                                       readout='sewn', n_readout=n_readout, seed=seed)[1]

def idqndrc_features_qiskit(N, u, g, bias_z, bias_x, m=2, n_readout=3, reps=1, seed=42):
    """ID-QND-RC: sewn quantum-memory features (+) a short classical delay line."""
    Qf = qmem_features_qiskit(N, u, g, bias_z, bias_x, n_readout, reps, seed)
    return np.concatenate([Qf, delay_taps(u, m)], axis=1)

def classical_qndrc_features_qiskit(N, u, g, bias_z, bias_x, m, W_q=2):
    """QND-RC with 1-body read-out (matches the persistent-reservoir feature set)."""
    S = monolithic_features_qiskit(N, u, g, bias_z, bias_x, W=W_q)
    return np.concatenate([S, delay_taps(u, m)], axis=1)

# =============================================================================
#  PART F.  BARREN-PLATEAU (local vs global cost) via Qiskit parameterized circuits
#  Same critical-phase parameterization (Ry layers + CZ^(2g/pi) bricks).
# =============================================================================
def _param_reservoir_qiskit(N, depth):
    """Parameterized reservoir circuit: `depth` x [Ry(theta) on all] + [CP(phi) bricks].
       Returns (circuit, theta[depth][N], phi[depth][N-1])."""
    qc = QuantumCircuit(N)
    th = [[Parameter(f't_{d}_{i}') for i in range(N)] for d in range(depth)]
    ph = [[Parameter(f'p_{d}_{e}') for e in range(N - 1)] for d in range(depth)]
    for d in range(depth):
        for i in range(N):
            qc.ry(th[d][i], i)
        for e, (a, b) in enumerate(chain_edges(N)):
            qc.cp(ph[d][e], a, b)
    return qc, th, ph

def grad_var_qiskit(N, depth, observable='local', n_samples=60, seed=0, qb=0):
    """Var over random params of dC/d(last-layer theta[qb]) via parameter-shift,
       with C = <O>.  observable: 'local' = Z_qb (sewing), 'global' = Z_0..Z_{N-1}."""
    circ, th, ph = _param_reservoir_qiskit(N, depth)
    if observable == 'global':
        O = Pauli('Z' * N); qargs = list(range(N))
    else:
        O = Pauli('Z'); qargs = [qb]
    rng = np.random.RandomState(seed); grads = []
    target = th[depth - 1][qb]
    for _ in range(n_samples):
        base = {}
        for d in range(depth):
            for i in range(N):
                base[th[d][i]] = rng.uniform(0, 2 * np.pi)
            for e in range(N - 1):
                base[ph[d][e]] = rng.uniform(0, np.pi)      # CZ**t, t ~ U(0,1)
        def expval(shift):
            r = dict(base); r[target] = base[target] + shift
            bound = circ.assign_parameters(r)
            psi = Statevector.from_instruction(bound)
            return float(np.real(psi.expectation_value(O, qargs)))
        grads.append((expval(np.pi / 2) - expval(-np.pi / 2)) / 2.0)
    return float(np.var(grads))

# =============================================================================
#  PART E.  IBM-NATIVE HARDWARE VALIDATION
#  Build the FULL ID-QND-RC processing circuit (encode + critical step + sewn
#  ancilla read-out), lay it out on a real IBM coupling map, transpile to the
#  device's native gateset, validate, and run on the noisy backend simulator
#  (or on real hardware via QiskitRuntimeService).
# =============================================================================
def _coupling_graph(backend):
    """Undirected adjacency dict of the backend coupling map."""
    cm = backend.coupling_map
    adj = {q: set() for q in range(backend.num_qubits)}
    for a, b in cm.get_edges():
        adj[a].add(b); adj[b].add(a)
    return adj

def find_chain_with_ancillas(backend, N, n_readout, read_idx=None, prefer_quiet=True):
    """Find a simple path of N physically-connected qubits such that every tapped
       memory qubit has a FREE adjacent neighbour to host its sewing ancilla, so
       that every 2-qubit gate (reservoir CZ bricks AND sewing CNOTs) is native
       and no SWAP routing is inserted.

    This is the IBM analogue of the Willow search for 'a straight row of N qubits
    with the row below free'.  IBM's heavy-hex lattice has degree <= 3, so ancilla
    hosts are scarcer than on Willow's degree-4 square grid and the search matters.

    If `prefer_quiet`, ties are broken by picking the layout with the lowest total
    two-qubit error rate from the backend's calibration data."""
    adj = _coupling_graph(backend)
    if read_idx is None:
        read_idx = list(range(1, N))   # any memory qubit may be tapped
    n_qubits = backend.num_qubits

    # --- 2-qubit error lookup (for tie-breaking on real calibrated backends) ---
    err = {}
    if prefer_quiet:
        try:
            tgt = backend.target
            for name in ('cz', 'ecr', 'cx'):
                if name in tgt.operation_names:
                    for pair, props in tgt[name].items():
                        if props is not None and props.error is not None:
                            err[frozenset(pair)] = float(props.error)
                    break
        except Exception:
            err = {}
    def path_cost(path, anc):
        pairs = [frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)]
        pairs += [frozenset((path[i], a)) for i, a in anc.items()]
        return sum(err.get(p, 0.01) for p in pairs)

    def ancillas_for(path):
        """Which MEMORY indices (1..N-1) can host an adjacent free ancilla?

        On Willow's degree-4 square grid one can tap consecutive indices 1,2,...
        On IBM's degree-3 heavy-hex a path qubit spends 2 of its 3 edges on the
        chain, so only degree-3 qubits have a spare neighbour -- and heavy-hex
        alternates degree-3 / degree-2 along any path.  We therefore let the
        search CHOOSE which memory indices to tap instead of fixing 1..n_readout."""
        used = set(path); anc = {}
        for i in read_idx:
            if i >= len(path):
                continue
            free = sorted(q for q in adj[path[i]] if q not in used)
            if free:
                anc[i] = free[0]; used.add(free[0])
            if len(anc) == n_readout:
                break
        return anc

    candidates = []
    MAXC = 40000

    def dfs(path, seen):
        if len(path) == N:
            anc = ancillas_for(path)
            candidates.append((len(anc), -path_cost(path, anc), list(path), dict(anc)))
            return len(candidates) >= MAXC
        for nxt in sorted(adj[path[-1]]):
            if nxt in seen:
                continue
            path.append(nxt); seen.add(nxt)
            stop = dfs(path, seen)
            path.pop(); seen.discard(nxt)
            if stop:
                return True
        return False

    for s in sorted(range(n_qubits), key=lambda q: (-len(adj[q]), q)):
        if dfs([s], {s}):
            break
        if candidates and max(c[0] for c in candidates) >= n_readout:
            break
    if not candidates:
        raise RuntimeError(f'no simple path of length {N} found on {backend.name}')
    best = max(candidates)
    if best[0] < n_readout:
        raise RuntimeError(
            f'{backend.name}: could only place {best[0]} of {n_readout} sewing ancillas '
            f'adjacent to an N={N} chain. Reduce n_readout or N.')
    return best[2], best[3]

def ibm_idqndrc_circuit(N, anc_slots, g, bz, bx, u_window, bases, mem_bases=None):
    """One ID-QND-RC processing + sewn-read-out step, as a LOGICAL circuit on
       N reservoir qubits + len(anc_slots) ancillas (ancillas indexed N, N+1, ...).

       anc_slots : list of reservoir indices that are tapped by an ancilla
       bases     : {tapped_index: Pauli basis} for the sewn read-out
       mem_bases : optional list of per-qubit randomized-Pauli bases for the final
                   classical-shadow read-out of the whole reservoir."""
    n_anc = len(anc_slots)
    qc = QuantumCircuit(N + n_anc, N + n_anc)
    slot_phase = np.linspace(0.5, 1.0, len(u_window))
    per_q = np.zeros(N)
    for w, uu in enumerate(u_window):
        per_q[w % N] += np.pi * uu * slot_phase[w]
    for i in range(N):
        qc.ry(per_q[i], i)                                        # encode
    qc.barrier()
    for (a, b) in chain_edges(N):
        qc.cp(2.0 * g, a, b)                                      # entangle (CZ**t)
    for i in range(N):
        qc.rz(bz[i], i)                                           # disorder
    for i in range(N):
        qc.rx(bx[i], i)
    qc.barrier()
    for j, q in enumerate(anc_slots):                             # sewn ancilla read-out
        append_sewn_copy(qc, q, N + j, bases[q])
    qc.barrier()
    if mem_bases is not None:                                     # shadow basis rotation
        for i in range(N):
            _apply_basis_change(qc, i, mem_bases[i], inverse=False)
    for j, q in enumerate(anc_slots):
        qc.measure(N + j, N + j)                                  # ancilla outcomes
    for i in range(N):
        qc.measure(i, i)                                          # memory shadow bits
    return qc

def ibm_validation(N=6, n_readout=2, n_shots=4096, seed=42, backend=None,
                   optimization_level=3, use_real_hardware=False, service=None):
    """Lay out and run the full ID-QND-RC processing + sewn-read-out circuit on an
       IBM device: native-gateset transpilation, coupling-map validation, and a
       noisy execution on AerSimulator.from_backend(...) (or real hardware)."""
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()

    rng = np.random.RandomState(seed)
    chain, anc_map = find_chain_with_ancillas(backend, N, n_readout)
    anc_slots = sorted(anc_map)
    bases = {q: 'XYZ'[rng.randint(3)] for q in anc_slots}
    bz = rng.uniform(0, 2 * np.pi, N); bx = rng.uniform(0.3, 0.7, N)
    u_window = rng.uniform(0, 1, 2)
    mem_bases = ['XYZ'[rng.randint(3)] for _ in range(N)]

    logical = ibm_idqndrc_circuit(N, anc_slots, np.pi * GSTAR, bz, bx,
                                  u_window, bases, mem_bases=mem_bases)

    # physical layout: reservoir chain + each ancilla adjacent to its tapped qubit
    initial_layout = list(chain) + [anc_map[q] for q in anc_slots]
    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization_level,
                                      initial_layout=initial_layout)
    native = pm.run(logical)

    # ---- device validation (the Qiskit analogue of dev.validate_circuit) ----
    target = backend.target
    allowed = set(backend.operation_names)
    bad_ops, bad_edges = set(), set()
    edges = {tuple(e) for e in backend.coupling_map.get_edges()}
    for inst in native.data:
        nm = inst.operation.name
        if nm in ('barrier',):
            continue
        if nm not in allowed:
            bad_ops.add(nm)
        if len(inst.qubits) == 2:
            a = native.find_bit(inst.qubits[0]).index
            b = native.find_bit(inst.qubits[1]).index
            if (a, b) not in edges and (b, a) not in edges:
                bad_edges.add((a, b))
    device_validated = (not bad_ops) and (not bad_edges)

    counts_native = native.count_ops()
    twoq = {k: v for k, v in counts_native.items()
            if k in ('cz', 'ecr', 'cx', 'rzz') }

    # ---- execution ----
    if use_real_hardware and service is not None:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        job = sampler.run([native], shots=n_shots)
        res = job.result()[0]
        bits = res.data
        counts = bits[list(bits.keys())[0]].get_counts()
        where = f'real hardware: {backend.name}'
    else:
        sim = AerSimulator.from_backend(backend)
        counts = sim.run(native, shots=n_shots).result().get_counts()
        where = f'AerSimulator.from_backend({backend.name})  [noise model of the real device]'

    # ---- decode the randomized-Pauli memory read-out ----
    nbits = native.num_clbits
    z = np.zeros(N); tot = 0
    for bstr, c in counts.items():
        b = bstr.replace(' ', '')[::-1]          # qiskit prints c[n-1]..c[0]
        for i in range(N):
            z[i] += c * (1 - 2 * int(b[i]))
        tot += c
    z /= max(tot, 1)

    return {
        'backend': backend.name,
        'basis_gates': sorted(allowed - {'delay', 'for_loop', 'if_else', 'switch_case', 'id'}),
        'reservoir_qubits': [int(q) for q in chain],
        'ancillas': {str(int(chain[q])): int(anc_map[q]) for q in anc_slots},
        'sewn_bases': {str(q): bases[q] for q in anc_slots},
        'two_qubit_gates': twoq,
        'native_gate_counts': {k: int(v) for k, v in counts_native.items()},
        'device_validated': bool(device_validated),
        'illegal_ops': sorted(bad_ops),
        'illegal_couplings': sorted(bad_edges),
        'depth_logical': int(logical.depth()),
        'depth_native': int(native.depth()),
        'mem_Z_randomized': np.round(z, 3).tolist(),
        'executed_on': where,
        'shots': int(n_shots),
    }, native, logical

# =============================================================================
#  PART G.  END-TO-END ON THE IBM DEVICE
#  Extract the QND-RC quantum feature map by ACTUALLY MEASURING randomized-Pauli
#  (classical-shadow) observables on the IBM backend noise model, then solve the
#  k-Pauli task from those shot-limited, device-noisy features.
# =============================================================================
def _shadow_basis_circuits(N, per_q, g, bz, bx, chain, backend, pm=None):
    """Three measurement-basis variants (all-Z / all-X / all-Y) of one encoded
       reservoir step, ready for the device."""
    circs = []
    for basis in 'ZXY':
        qc = QuantumCircuit(N, N)
        for i in range(N):
            qc.ry(per_q[i], i)
        for (a, b) in chain_edges(N):
            qc.cp(2.0 * g, a, b)
        for i in range(N):
            qc.rz(bz[i], i)
        for i in range(N):
            qc.rx(bx[i], i)
        for i in range(N):
            _apply_basis_change(qc, i, basis, inverse=False)
        qc.measure(range(N), range(N))
        circs.append(qc)
    return circs

def ibm_quantum_features(N, u, g, bz, bx, W=2, shots=4096, backend=None,
                         chain=None, optimization_level=3, noisy=True,
                         batch=200, verbose=True):
    """Run the QND-RC quantum feature map on an IBM backend (noise model of a real
       device) and return the shot-estimated 1-body Pauli feature matrix (T, 3N).

    This is the device-level analogue of `monolithic_features_qiskit`: identical
    circuits, but the expectation values are ESTIMATED FROM SHOTS on hardware-noise
    simulation rather than computed exactly."""
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
    if chain is None:
        chain, _ = find_chain_with_ancillas(backend, N, 1)
    sim = AerSimulator.from_backend(backend) if noisy else AerSimulator()
    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization_level,
                                      initial_layout=list(chain))
    slot_phase = np.linspace(0.5, 1.0, W); slot_to_q = [w % N for w in range(W)]
    T = len(u)
    all_circs = []
    for t in range(T):
        lo = max(0, t - W + 1); wlen = t - lo + 1
        win = np.zeros(W); win[-wlen:] = u[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(win):
            per_q[slot_to_q[w]] += np.pi * uu * slot_phase[w]
        all_circs += _shadow_basis_circuits(N, per_q, g, bz, bx, chain, backend)
    native = pm.run(all_circs)
    counts_all = []
    for i in range(0, len(native), batch):
        res = sim.run(native[i:i + batch], shots=shots).result()
        counts_all += [res.get_counts(j) for j in range(len(native[i:i + batch]))]
        if verbose:
            print(f'    ran {min(i + batch, len(native))}/{len(native)} circuits', end='\r')
    if verbose:
        print(' ' * 60, end='\r')
    X = np.zeros((T, 3 * N))
    for t in range(T):
        for bi in range(3):                       # 0=Z, 1=X, 2=Y
            counts = counts_all[3 * t + bi]
            tot = sum(counts.values()); z = np.zeros(N)
            for bstr, c in counts.items():
                b = bstr.replace(' ', '')[::-1]
                for i in range(N):
                    z[i] += c * (1 - 2 * int(b[i]))
            z /= max(tot, 1)
            for i in range(N):
                X[t, 3 * i + bi] = z[i]           # column order Z,X,Y per qubit
    return X

def ibm_sewn_readout_experiment(N=4, tap=1, shots=8192, backend=None, seed=0,
                                optimization_level=3):
    """Device-level test of the deferred-measurement (sewing) identity.

    Runs, on the IBM noise model:
      (A) the reservoir + a physical sewn ancilla read-out (copy -> measure ancilla),
          then measures the memory register;
      (B) the same reservoir with NO ancilla, memory register measured directly.
    If sewing works, the tapped qubit's Z-marginal is unchanged (a Z-basis copy
    commutes with Z), while the memory register keeps its coherence -- i.e. the
    ancilla learned <Z_tap> WITHOUT collapsing the rest of the register."""
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
    rng = np.random.RandomState(seed)
    bz = rng.uniform(0, 2 * np.pi, N); bx = rng.uniform(0.3, 0.7, N)
    uw = rng.uniform(0, 1, 2)
    chain, anc_map = find_chain_with_ancillas(backend, N, 1)
    tap = sorted(anc_map)[0]
    anc_phys = anc_map[tap]

    def base(nq):
        qc = QuantumCircuit(nq, nq)
        slot_phase = np.linspace(0.5, 1.0, len(uw))
        per_q = np.zeros(N)
        for w, vv in enumerate(uw):
            per_q[w % N] += np.pi * vv * slot_phase[w]
        for i in range(N):
            qc.ry(per_q[i], i)
        for (a, b) in chain_edges(N):
            qc.cp(2.0 * np.pi * GSTAR, a, b)
        for i in range(N):
            qc.rz(bz[i], i)
        for i in range(N):
            qc.rx(bx[i], i)
        return qc

    # (A) with sewn ancilla read-out
    qa = base(N + 1)
    append_sewn_copy(qa, tap, N, 'Z')
    qa.measure(N, N)
    qa.measure(range(N), range(N))
    # (B) no ancilla
    qb = base(N)
    qb.measure(range(N), range(N))

    sim = AerSimulator.from_backend(backend)
    pma = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level,
                                       initial_layout=list(chain) + [anc_phys])
    pmb = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level,
                                       initial_layout=list(chain))
    ca, cb = pma.run(qa), pmb.run(qb)
    ra = sim.run(ca, shots=shots).result().get_counts()
    rb = sim.run(cb, shots=shots).result().get_counts()

    def marg(counts, nbits):
        tot = sum(counts.values()); z = np.zeros(nbits)
        for s, c in counts.items():
            b = s.replace(' ', '')[::-1]
            for i in range(nbits):
                z[i] += c * (1 - 2 * int(b[i]))
        return z / max(tot, 1)

    za = marg(ra, N + 1); zb = marg(rb, N)
    # ideal (noiseless) reference
    ideal = DensityMatrix.from_instruction(base(N).remove_final_measurements(inplace=False))
    zi = np.array([float(np.real(ideal.expectation_value(Pauli('Z'), [i]))) for i in range(N)])
    return {
        'backend': backend.name,
        'chain': [int(q) for q in chain], 'tap_index': int(tap), 'ancilla': int(anc_phys),
        'ancilla_Z_readout': float(za[N]),
        'memory_Z_with_sewing': np.round(za[:N], 3).tolist(),
        'memory_Z_without_ancilla': np.round(zb, 3).tolist(),
        'memory_Z_ideal': np.round(zi, 3).tolist(),
        'max_disturbance_from_sewing': float(np.abs(za[:N] - zb).max()),
        'shots': int(shots),
    }
