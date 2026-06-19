"""
idcpsr_cirq.py  —  Cirq-native ID-CPSR engine
==============================================
Every quantum operation is a real `cirq` circuit, executed on `cirq.Simulator`
(state vector) or `cirq.DensityMatrixSimulator` (open-system / measurement
back-action), and the processing circuit transpiles to 100% Willow-native gates
(GoogleCZTargetGateset) and runs on the Willow Pink QVM.

This replaces the earlier numpy stand-in: the Cirq circuits below are the source
of truth, and we *verify* they reproduce the repo's reservoir math to machine
precision.  Gate conventions match the repo's `willow_qvm_demo`:
    encode :  ry(pi*u) on the input qubit
    entangle: CZ**(2g/pi) on chain edges      (Willow-native coupler)
    disorder: rz(bias_z) then rx(bias_x)
    shadow  : randomized-Pauli basis rotation (H for X, S^-1 then H for Y)
"""
import numpy as np
import cirq
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

GSTAR = 0.30                       # edge-of-chaos critical phase g*/pi (repo)
SV_SIM = cirq.Simulator(dtype=np.complex128)
DM_SIM = cirq.DensityMatrixSimulator(dtype=np.complex128)

def _renorm(rho):
    rho = np.asarray(rho, dtype=np.complex128)
    tr = np.trace(rho)
    return rho / tr if abs(tr) > 0 else rho

# =============================================================================
#  PART A.  CIRQ CRITICAL STEP  (real circuit; Willow-native processing layer)
# =============================================================================
def line_qubits(N):
    return cirq.LineQubit.range(N)

def chain_edges(N):
    return [(i, i + 1) for i in range(N - 1)]

def critical_step_ops(qs, g, bias_z, bias_x):
    """One reservoir step as Cirq operations on qubits `qs`:
       U = Rx . Rz . CZ^(2g/pi)  (applied left-to-right as a circuit)."""
    N = len(qs)
    ops = []
    for (a, b) in chain_edges(N):
        ops.append((cirq.CZ ** (2 * g / np.pi)).on(qs[a], qs[b]))
    ops += [cirq.rz(bias_z[i]).on(qs[i]) for i in range(N)]
    ops += [cirq.rx(bias_x[i]).on(qs[i]) for i in range(N)]
    return ops

def critical_step_circuit(qs, g, bias_z, bias_x):
    return cirq.Circuit(critical_step_ops(qs, g, bias_z, bias_x))

def critical_unitary_cirq(N, g, bias_z, bias_x):
    """Dense unitary of one Cirq step (for verification / fast exact propagation)."""
    qs = line_qubits(N)
    return cirq.unitary(critical_step_circuit(qs, g, bias_z, bias_x))

def get_bias(N, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 2 * np.pi, N), rng.uniform(0.3, 0.7, N)

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
#  PART B.  SEWN ANCILLA READ-OUT  (real Cirq circuit)  +  CHANNEL EQUIVALENT
#  Physical circuit (runs on Willow): ancilla|0> -> basis-copy from system qubit
#  -> measure ancilla -> reset.  By deferred measurement this equals LOCAL
#  dephasing of that qubit in the chosen Pauli basis -> memory register coherent.
# =============================================================================
def _basis_change_ops(q, basis, inverse=False):
    """Ops mapping Pauli `basis` eigenbasis <-> Z basis on qubit q."""
    if basis == 'Z':
        return []
    if basis == 'X':
        return [cirq.H(q)]
    if basis == 'Y':
        # measure-Y: apply S^-1 then H (forward, basis->Z); inverse is H then S
        return [cirq.H(q), cirq.S(q)] if inverse else [(cirq.S ** -1)(q), cirq.H(q)]
    raise ValueError(basis)

def sewn_copy_ops(system_qubit, ancilla, basis):
    """Coherent von-Neumann copy of <P_basis(system)> onto a fresh ancilla
       (no measurement). Discarding/tracing the ancilla == local dephasing."""
    ops = []
    ops += _basis_change_ops(system_qubit, basis, inverse=False)  # rotate basis -> Z
    ops.append(cirq.CNOT(system_qubit, ancilla))                  # copy / 'sew'
    ops += _basis_change_ops(system_qubit, basis, inverse=True)   # rotate Z -> basis
    return ops

def sewn_readout_physical(system_qubit, ancilla, basis):
    """Hardware sewn read-out sub-circuit: coherent copy onto a fresh ancilla,
       then MEASURE it and RESET it.  This is the circuit that runs on Willow;
       discarding the classical outcome reproduces the dephasing channel."""
    ops = sewn_copy_ops(system_qubit, ancilla, basis)
    ops.append(cirq.measure(ancilla, key='anc'))
    ops.append(cirq.reset(ancilla))
    return ops

def sewn_dephase_channel_ops(q, basis):
    """Channel-equivalent of the physical sewn read-out (deferred-measurement
       identity): local dephasing of qubit q in `basis`. (rho + P rho P)/2.
       phase_flip(0.5) == Z-dephasing; conjugate by basis change for X / Y."""
    ops = []
    ops += _basis_change_ops(q, basis, inverse=False)   # basis -> Z
    ops.append(cirq.phase_flip(0.5)(q))                  # (rho + Z rho Z)/2
    ops += _basis_change_ops(q, basis, inverse=True)     # Z -> basis
    return ops

def deferred_measurement_check(N=3, seed=0):
    """Verify in Cirq: tracing out the measured ancilla in the physical sewn
       read-out == applying the local dephasing channel.  Returns max ||.||_F."""
    qs = cirq.LineQubit.range(N); anc = cirq.LineQubit(N)
    allq = list(qs) + [anc]
    # random system state (acts on all system qubits; ancilla starts |0>)
    prep = cirq.testing.random_circuit(qs, n_moments=4, op_density=1.0, random_state=seed)
    prep = cirq.Circuit([cirq.I(q) for q in qs]) + prep
    worst = 0.0
    for q in range(N):
        for b in 'XYZ':
            # path 1: coherent copy onto ancilla, then TRACE OUT the ancilla
            # (deferred measurement: discarding the ancilla == local dephasing)
            c1 = cirq.Circuit(prep)
            c1 += sewn_copy_ops(qs[q], anc, b)
            r1 = DM_SIM.simulate(c1, qubit_order=allq).final_density_matrix  # (sys,anc)
            r1 = r1.reshape(2 ** N, 2, 2 ** N, 2)
            sys1 = r1[:, 0, :, 0] + r1[:, 1, :, 1]   # partial trace over ancilla
            # path 2: channel form on the system only
            c2 = cirq.Circuit(prep) + sewn_dephase_channel_ops(qs[q], b)
            sys2 = DM_SIM.simulate(c2, qubit_order=qs).final_density_matrix
            worst = max(worst, float(np.linalg.norm(sys1 - sys2)))
    return worst

# =============================================================================
#  PART C.  PERSISTENT (recurrent) CIRQ RESERVOIR with coherent quantum memory
#  qubit 0 = input register (reset+encode each step); qubits 1..N-1 = coherent
#  memory M (never directly measured under 'sewn'); features = local Paulis.
#  Executed on cirq.DensityMatrixSimulator (exact open-system back-action).
# =============================================================================
def _pauli_expectations(rho, qs, mem_qubits):
    """<Z>,<X>,<Y> on each memory qubit + adjacent <ZZ>, from a density matrix."""
    N = len(qs); qmap = {q: i for i, q in enumerate(qs)}
    feats, labels = [], []
    for q in mem_qubits:
        for P, nm in [(cirq.Z, 'Z'), (cirq.X, 'X'), (cirq.Y, 'Y')]:
            ps = cirq.PauliString({qs[q]: P})
            feats.append(float(np.real(ps.expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
            labels.append(f'{nm}{q}')
    for a, b in zip(mem_qubits[:-1], mem_qubits[1:]):
        ps = cirq.PauliString({qs[a]: cirq.Z, qs[b]: cirq.Z})
        feats.append(float(np.real(ps.expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
        labels.append(f'Z{a}Z{b}')
    return labels, feats

def persistent_reservoir_cirq(N, u_seq, g, bias_z, bias_x, reps=1,
                              readout='sewn', n_readout=2, seed=42):
    """Persistent quantum reservoir as a Cirq DensityMatrixSimulator run.
       readout in {'none','sewn','projective'} selects the post-step channel.
       Returns (labels, feature matrix (T,F))."""
    qs = cirq.LineQubit.range(N)
    mem = list(range(1, N))
    read_qubits = mem[:n_readout]
    rng = np.random.RandomState(seed)
    step_ops = critical_step_ops(qs, g, bias_z, bias_x)
    # initial density matrix |0..0>
    rho = _renorm(DM_SIM.simulate(cirq.Circuit([cirq.I(q) for q in qs])).final_density_matrix)
    T = len(u_seq)
    labels = None; X = []
    for t in range(T):
        c = cirq.Circuit([cirq.I(q) for q in qs])         # pad: all qubits present
        c.append(cirq.reset(qs[0]))                       # discard input qubit -> fading memory
        c.append(cirq.ry(np.pi * u_seq[t]).on(qs[0]))     # encode input
        c.append(step_ops)                                 # critical reservoir step
        rho = _renorm(DM_SIM.simulate(c, initial_state=rho, qubit_order=qs).final_density_matrix)
        lab, ft = _pauli_expectations(rho, qs, mem)
        labels = lab; X.append(ft)
        # post-read back-action channel (exact)
        cc = cirq.Circuit([cirq.I(q) for q in qs])         # pad: all qubits present
        if readout == 'sewn':
            for q in read_qubits:
                b = ['X', 'Y', 'Z'][rng.randint(3)]
                cc.append(sewn_dephase_channel_ops(qs[q], b))
        elif readout == 'projective':
            for q in mem:
                b = ['X', 'Y', 'Z'][rng.randint(3)]
                cc.append(sewn_dephase_channel_ops(qs[q], b))   # full dephase of every mem qubit
        rho = _renorm(DM_SIM.simulate(cc, initial_state=rho, qubit_order=qs).final_density_matrix)
    return labels, np.array(X)

# =============================================================================
#  PART D.  INSTANTANEOUS DEPTH under Cirq noise  (depth<->width trade)
#   deep : D physical noisy layers           -> noise compounds with D
#   id   : same U^D action, ONE noisy layer   -> O(1) physical noise + ancilla width
# =============================================================================
def noisy_reservoir_features_cirq(N, u_seq, g, bias_z, bias_x, D_eff,
                                  mode='deep', p=0.05, window=4):
    qs = cirq.LineQubit.range(N)
    step = critical_step_circuit(qs, g, bias_z, bias_x)
    UD = cirq.unitary(step) ** 0  # placeholder; we build U^D as repeated circuit/matrix
    # build deep action matrix for 'id' mode
    U1 = cirq.unitary(step)
    UDmat = np.linalg.matrix_power(U1, D_eff)
    slot_phase = np.linspace(0.5, 1.0, window); slot_to_q = [w % N for w in range(window)]
    mem = list(range(N))
    T = len(u_seq); X = []
    depol = cirq.depolarize(p)
    for t in range(T):
        lo = max(0, t - window + 1); wlen = t - lo + 1
        win = np.zeros(window); win[-wlen:] = u_seq[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(win):
            per_q[slot_to_q[w]] += np.pi * uu * slot_phase[w]
        enc = cirq.Circuit([cirq.ry(per_q[i]).on(qs[i]) for i in range(N)])
        rho = DM_SIM.simulate(enc).final_density_matrix
        if mode == 'deep':
            for _ in range(D_eff):
                c = cirq.Circuit([cirq.I(q) for q in qs]); c.append(step.all_operations())
                c.append([depol.on(q) for q in qs])
                rho = _renorm(DM_SIM.simulate(c, initial_state=rho, qubit_order=qs).final_density_matrix)
        else:  # 'id': deep action in one block, single physical-noise layer
            c = cirq.Circuit([cirq.I(q) for q in qs])
            c.append(cirq.MatrixGate(UDmat, qid_shape=(2,) * N).on(*qs))
            c.append([depol.on(q) for q in qs])
            rho = _renorm(DM_SIM.simulate(c, initial_state=rho, qubit_order=qs).final_density_matrix)
        # local Pauli features
        qmap = {q: i for i, q in enumerate(qs)}; ft = []
        for q in range(N):
            for P in (cirq.Z, cirq.X, cirq.Y):
                ft.append(float(np.real(cirq.PauliString({qs[q]: P}).expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
        X.append(ft)
    return np.array(X)

def operator_entanglement_cirq(N, g, bias_z, bias_x, D):
    U = np.linalg.matrix_power(critical_unitary_cirq(N, g, bias_z, bias_x), D)
    dh = 2 ** (N // 2)
    Ut = U.reshape(dh, dh, dh, dh).transpose(0, 2, 1, 3).reshape(dh * dh, dh * dh)
    s = np.linalg.svd(Ut, compute_uv=False); pr = (s ** 2) / np.sum(s ** 2); pr = pr[pr > 1e-12]
    return float(-np.sum(pr * np.log(pr)))

# ---- integrated feature builders (Cirq) ----
def monolithic_features_cirq(N, u, g, bias_z, bias_x, W=8):
    """Memoryless QELM features (sliding-window encode) via Cirq state vectors."""
    qs = cirq.LineQubit.range(N)
    U = np.linalg.matrix_power(critical_unitary_cirq(N, g, bias_z, bias_x), 1)
    slot_phase = np.linspace(0.5, 1.0, W); slot_to_q = [w % N for w in range(W)]
    qmap = {q: i for i, q in enumerate(qs)}; T = len(u); X = []
    for t in range(T):
        lo = max(0, t - W + 1); wlen = t - lo + 1
        win = np.zeros(W); win[-wlen:] = u[lo:t + 1]
        per_q = np.zeros(N)
        for w, uu in enumerate(win): per_q[slot_to_q[w]] += np.pi * uu * slot_phase[w]
        enc = cirq.Circuit([cirq.ry(per_q[i]).on(qs[i]) for i in range(N)])
        psi = SV_SIM.simulate(enc).final_state_vector
        psi = U @ psi
        rho = np.outer(psi, psi.conj()); ft = []
        for q in range(N):
            for P in (cirq.Z, cirq.X, cirq.Y):
                ft.append(float(np.real(cirq.PauliString({qs[q]: P}).expectation_from_density_matrix(rho, qmap, check_preconditions=False))))
        X.append(ft)
    return np.array(X)

def qmem_features_cirq(N, u, g, bias_z, bias_x, n_readout=3, seed=42):
    return persistent_reservoir_cirq(N, u, np.pi * g, bias_z, bias_x, reps=1,
                                     readout='sewn', n_readout=n_readout, seed=seed)[1]

def idqndrc_features_cirq(N, u, g, bias_z, bias_x, m=2, n_readout=3, seed=42):
    Q = qmem_features_cirq(N, u, g, bias_z, bias_x, n_readout, seed)
    return np.concatenate([Q, delay_taps(u, m)], axis=1)

def classical_qndrc_features_cirq(N, u, g, bias_z, bias_x, m, W_q=2):
    S = monolithic_features_cirq(N, u, np.pi * g, bias_z, bias_x, W=W_q)
    return np.concatenate([S, delay_taps(u, m)], axis=1)

# =============================================================================
#  PART E.  WILLOW-NATIVE VALIDATION  (real Willow Pink QVM)
#  Build the FULL ID-QND-RC processing circuit (encode + critical step + sewn
#  ancilla read-out), transpile to the Willow native gateset, run on the QVM.
# =============================================================================
def willow_idqndrc_circuit(qs, anc_map, g, bz, bx, u_window, bases):
    """One ID-QND-RC processing + sewn-read-out step as a Cirq circuit.
       qs = reservoir chain; anc_map = {tapped_index: adjacent_ancilla_qubit};
       bases = {tapped_index: Pauli basis}. Every 2-qubit gate is between
       physically-adjacent qubits (Willow-native, no routing)."""
    N = len(qs); c = cirq.Circuit()
    slot_phase = np.linspace(0.5, 1.0, len(u_window))
    per_q = np.zeros(N)
    for w, uu in enumerate(u_window):
        per_q[w % N] += np.pi * uu * slot_phase[w]
    c.append([cirq.ry(per_q[i]).on(qs[i]) for i in range(N)])          # encode
    for (a, b) in [(i, i + 1) for i in range(N - 1)]:
        c.append((cirq.CZ ** (2 * g / np.pi)).on(qs[a], qs[b]))        # entangle (native CZ)
    c.append([cirq.rz(bz[i]).on(qs[i]) for i in range(N)])            # disorder
    c.append([cirq.rx(bx[i]).on(qs[i]) for i in range(N)])
    # sewn ancilla read-out: each tapped memory qubit uses its OWN adjacent ancilla
    for q, anc in anc_map.items():
        c.append(sewn_copy_ops(qs[q], anc, bases[q]))                  # native CNOT(q, adjacent anc)
        c.append(cirq.measure(anc, key=f'anc_{q}'))
    return c

def willow_validation(N=6, n_readout=2, n_shots=400, seed=42):
    """Run the ID-QND-RC processing+sewn-read-out circuit on the Willow Pink QVM,
       with one adjacent ancilla per tapped memory qubit (no routing/SWAPs)."""
    import cirq_google as cg
    from cirq_google.engine import virtual_engine_factory
    qvm = virtual_engine_factory.create_default_noisy_quantum_virtual_machine(
        processor_id='willow_pink', simulator_class=cirq.Simulator)
    proc = qvm.get_processor('willow_pink'); dev = proc.get_device()
    Gph = dev.metadata.nx_graph
    nodes = sorted(Gph.nodes, key=lambda q: (q.row, q.col))
    nodeset = set(nodes)
    def coupled(a, b): return Gph.has_edge(a, b)
    # find a straight horizontal run of N qubits in one row (so the row below is
    # free for ancillas).  Fall back to a greedy snake if none.
    from collections import defaultdict
    rows = defaultdict(list)
    for q in nodes: rows[q.row].append(q.col)
    qs = None
    for r in sorted(rows):
        cols = sorted(rows[r])
        for i in range(len(cols) - N + 1):
            run = [cirq.GridQubit(r, cols[i + k]) for k in range(N)]
            if all(coupled(run[k], run[k + 1]) for k in range(N - 1)):
                qs = run; break
        if qs: break
    if qs is None:
        def walk_from(start, length):
            path = [start]; seen = {start}
            def dfs():
                if len(path) == length: return True
                for nb in sorted(Gph.neighbors(path[-1]), key=lambda q: (q.row, q.col)):
                    if nb in seen: continue
                    path.append(nb); seen.add(nb)
                    if dfs(): return True
                    path.pop(); seen.remove(nb)
                return False
            return path if dfs() else None
        for start in nodes:
            qs = walk_from(start, N)
            if qs: break
    used = set(qs)
    # assign each tapped memory qubit a distinct adjacent free ancilla
    read_idx = list(range(1, 1 + n_readout)); anc_map = {}
    rng = np.random.RandomState(seed)
    for q in read_idx:
        free = [nb for nb in Gph.neighbors(qs[q]) if nb not in used]
        if not free:
            continue
        a = sorted(free, key=lambda x: (x.row, x.col))[0]
        anc_map[q] = a; used.add(a)
    bases = {q: 'XYZ'[rng.randint(3)] for q in anc_map}
    bz = rng.uniform(0, 2 * np.pi, N); bx = rng.uniform(0.3, 0.7, N)
    u_window = rng.uniform(0, 1, 2)
    c = willow_idqndrc_circuit(qs, anc_map, np.pi * GSTAR, bz, bx, u_window, bases)
    mem_bases = ['XYZ'[rng.randint(3)] for _ in range(N)]
    for i in range(N):
        c += _basis_change_ops(qs[i], mem_bases[i], inverse=False)
    c.append(cirq.measure(*qs, key='mem'))
    native = cirq.optimize_for_target_gateset(c, gateset=cg.GoogleCZTargetGateset())
    dev.validate_circuit(native)                       # raises if not Willow-native
    res = proc.get_sampler().run(native, repetitions=n_shots)
    bits = res.measurements['mem']
    z = 1 - 2 * bits.mean(0)
    gates = sorted({str(op.gate).split('(')[0] for op in native.all_operations()})
    twoq = sorted({str(op.gate).split('(')[0] for op in native.all_operations() if len(op.qubits) == 2})
    return {'reservoir_qubits': [str(q) for q in qs],
            'ancillas': {str(qs[q]): str(a) for q, a in anc_map.items()},
            'native_gates': gates, 'two_qubit_gates': twoq, 'device_validated': True,
            'mem_Z_randomized': np.round(z, 3).tolist(),
            'depth_logical': len(c), 'depth_native': len(native)}

# =============================================================================
#  PART F.  BARREN-PLATEAU (local vs global cost) via Cirq parameterized circuits
#  Same critical-phase parameterization (Ry layers + CZ^(2g/pi) bricks).
# =============================================================================
import sympy

def _param_reservoir_cirq(qs, depth):
    """Parameterized reservoir circuit: `depth` x [Ry(theta) on all] + [CZ**phi bricks].
       Returns (circuit, theta_syms (depth,N), phi_syms (depth,N-1))."""
    N = len(qs); c = cirq.Circuit()
    th = [[sympy.Symbol(f't_{d}_{i}') for i in range(N)] for d in range(depth)]
    ph = [[sympy.Symbol(f'p_{d}_{e}') for e in range(N - 1)] for d in range(depth)]
    for d in range(depth):
        c.append([cirq.ry(th[d][i]).on(qs[i]) for i in range(N)])
        for e, (a, b) in enumerate([(i, i + 1) for i in range(N - 1)]):
            c.append((cirq.CZ ** ph[d][e]).on(qs[a], qs[b]))
    return c, th, ph

def grad_var_cirq(N, depth, observable='local', n_samples=60, seed=0, qb=0):
    """Var over random params of dC/d(last-layer theta[qb]) via parameter-shift,
       with C = <O>.  observable: 'local' = Z_qb (sewing), 'global' = Z_0..Z_{N-1}."""
    qs = cirq.LineQubit.range(N)
    circ, th, ph = _param_reservoir_cirq(qs, depth)
    if observable == 'global':
        O = cirq.PauliString({qs[i]: cirq.Z for i in range(N)})
    else:
        O = cirq.PauliString({qs[qb]: cirq.Z})
    rng = np.random.RandomState(seed); grads = []
    target = th[depth - 1][qb]
    for _ in range(n_samples):
        base = {}
        for d in range(depth):
            for i in range(N): base[th[d][i].name] = rng.uniform(0, 2 * np.pi)
            for e in range(N - 1): base[ph[d][e].name] = rng.uniform(0, 1.0)
        def expval(shift):
            r = dict(base); r[target.name] = base[target.name] + shift
            psi = SV_SIM.simulate(circ, param_resolver=cirq.ParamResolver(r)).final_state_vector
            rho = np.outer(psi, psi.conj())
            qmap = {q: i for i, q in enumerate(qs)}
            return float(np.real(O.expectation_from_density_matrix(rho, qmap, check_preconditions=False)))
        grads.append((expval(np.pi / 2) - expval(-np.pi / 2)) / 2.0)
    return float(np.var(grads))
