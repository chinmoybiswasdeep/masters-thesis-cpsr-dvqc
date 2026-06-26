"""
floquet_rc.py  --  Monitored-circuit / Floquet-code unification for the
quantum-memory reservoir.  The reservoir's edge-of-chaos scrambling together
with its periodic sewn-measurement schedule form a MONITORED CIRCUIT.  Its
volume-law (coding) phase is a dynamically self-generated error-correcting code;
the SAME measurement record both (i) protects the encoded memory and (ii) is the
reservoir read-out.  Structured (period-T) schedules realise Floquet codes.

Stabilizer backend (Stim) for the scalable phase diagnostics; the Cirq
non-Clifford reservoir compute lives in apps_floquet_cirq.py.  Clifford and Haar
monitored circuits share the same MIPT universality class, so the coding-phase
diagram computed here applies to the non-Clifford reservoir.
"""
import numpy as np, stim

# ----------------------------------------------------------------------
#  Stabilizer entanglement entropy (bits) via GF(2) rank of the
#  generator matrix restricted to region A.   S_A = rank(J_A) - |A|.
# ----------------------------------------------------------------------
def _gf2_rank(M):
    M = (np.asarray(M, dtype=np.int8) % 2).copy()
    if M.size == 0:
        return 0
    rows, cols = M.shape; r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] + M[r]) % 2
        r += 1
        if r == rows:
            break
    return r

def stab_entropy(stabs, n, A):
    """Bipartite entanglement entropy (bits) of region A for a stabilizer state
       given by generator PauliStrings `stabs` on n qubits."""
    if len(stabs) == 0:
        return 0.0
    rows = []
    for ps in stabs:
        L = len(ps)
        xv = [1 if (i < L and ps[i] in (1, 2)) else 0 for i in range(n)]
        zv = [1 if (i < L and ps[i] in (2, 3)) else 0 for i in range(n)]
        rows.append([xv[i] for i in A] + [zv[i] for i in A])
    return _gf2_rank(rows) - len(A)

# ----------------------------------------------------------------------
#  Monitored brick-wall Clifford reservoir.
#  Scrambling = random 2-qubit Cliffords (edge-of-chaos surrogate);
#  monitoring = single-qubit Z measurements at rate p per site per layer.
# ----------------------------------------------------------------------
def monitored_clifford(N, depth, p, seed=0, ring=False, sim=None, sites=None,
                       record=False):
    """Run a monitored brick-wall Clifford circuit on `sites` (default 0..N-1).
       Returns the simulator (and, if record, the list of measurement outcomes)."""
    rng = np.random.RandomState(seed)
    if sim is None:
        sim = stim.TableauSimulator(seed=int(rng.randint(1 << 30)))
    if sites is None:
        sites = list(range(N))
    rec = []
    nb = N if ring else N - 1
    for d in range(depth):
        bonds = [(sites[i], sites[(i + 1) % N]) for i in range(0 if d % 2 == 0 else 1, nb, 2)]
        for (a, b) in bonds:
            sim.do_tableau(stim.Tableau.random(2), [a, b])
        for q in sites:
            if rng.uniform() < p:
                out = sim.measure(q)
                if record:
                    rec.append((d, q, int(out)))
    return (sim, rec) if record else sim

def half_chain_entropy(N, p, depth=None, seed=0, ring=False):
    depth = depth or 2 * N
    sim = monitored_clifford(N, depth, p, seed=seed, ring=ring)
    stabs = sim.canonical_stabilizers()
    A = list(range(N // 2))
    return stab_entropy(stabs, N, A)

def entropy_vs_p(N, ps, n_traj=40, depth=None, ring=True):
    """R1: steady-state half-chain entanglement vs measurement rate p (the
       volume-law -> area-law coding transition)."""
    out = []
    for p in ps:
        vals = [half_chain_entropy(N, p, depth=depth, seed=1000 * int(100 * p) + t, ring=ring)
                for t in range(n_traj)]
        out.append(float(np.mean(vals)))
    return {'ps': list(ps), 'S_half': out, 'N': N}

# ----------------------------------------------------------------------
#  Reference-qubit protection probe (Gullans-Huse).  One reference R is
#  maximally entangled with the system; monitored dynamics act on the system
#  only.  <S_R>(t) ~ 1 means the logical qubit is PROTECTED (coding phase);
#  decay to 0 means it is measured out (trivial phase).
# ----------------------------------------------------------------------
def reference_protection(N, p, depth, seed=0, ring=True, q_err=0.0, sample_every=1):
    rng = np.random.RandomState(seed)
    sim = stim.TableauSimulator(seed=int(rng.randint(1 << 30)))
    R = N                                   # reference qubit index
    sites = list(range(N))
    sim.h(R); sim.cnot(R, 0)                # Bell pair (R, system qubit 0)
    nb = N if ring else N - 1
    traj = []
    for d in range(depth):
        bonds = [(sites[i], sites[(i + 1) % N]) for i in range(0 if d % 2 == 0 else 1, nb, 2)]
        for (a, b) in bonds:
            sim.do_tableau(stim.Tableau.random(2), [a, b])
        if q_err > 0:                       # physical depolarising noise on the system
            for q in sites:
                if rng.uniform() < q_err:
                    sim.depolarize1(q, p=0.75)   # full single-qubit depolarisation event
        for q in sites:
            if rng.uniform() < p:
                sim.measure(q)
        if d % sample_every == 0:
            stabs = sim.canonical_stabilizers()
            traj.append(stab_entropy(stabs, N + 1, [R]))
    return np.array(traj)

def protection_vs_time(N, ps, depth=None, n_traj=40, ring=True, q_err=0.0):
    """R2: <S_R>(t) for several measurement rates p -> protected (coding) vs
       unprotected (trivial) memory."""
    depth = depth or 3 * N
    curves = {}
    for p in ps:
        acc = []
        for t in range(n_traj):
            acc.append(reference_protection(N, p, depth, seed=7000 * int(100 * p) + t,
                                            ring=ring, q_err=q_err))
        curves[p] = np.mean(np.array(acc), axis=0)
    return {'ps': list(ps), 'curves': {p: curves[p].tolist() for p in ps},
            'depth': depth, 'N': N, 'q_err': q_err}

def memory_time(N, p, depth=None, n_traj=60, ring=True, q_err=0.0):
    """Protection (memory) time = last layer at which <S_R> >= 1/2."""
    depth = depth or 4 * N
    c = np.mean([reference_protection(N, p, depth, seed=9000 * int(100 * p) + t,
                 ring=ring, q_err=q_err) for t in range(n_traj)], axis=0)
    idx = np.where(c >= 0.5)[0]
    return float(idx[-1]) if len(idx) else 0.0

# ----------------------------------------------------------------------
#  Volume-law vs area-law scaling (rigorous coding-phase signature):
#  S_half(N) ~ N  (volume, coding)  for p<p_c ;  ~ const (area)  for p>p_c.
# ----------------------------------------------------------------------
def entropy_vs_N(Ns, p, n_traj=30, ring=True):
    out = []
    for N in Ns:
        vals = [half_chain_entropy(N, p, depth=2 * N, seed=300 * N + t, ring=ring)
                for t in range(n_traj)]
        out.append(float(np.mean(vals)))
    return {'Ns': list(Ns), 'S_half': out, 'p': p}

# ----------------------------------------------------------------------
#  R4 -- Floquet / structured measurement schedule (a designed dynamical code).
#  Dimerised ring: round A measures ZZ on even bonds, round B measures XX on
#  odd bonds (period 2).  After a transient the instantaneous checks stabilise
#  the state (the code forms); injected errors flip checks -> a detectable
#  syndrome.  This is the engineered version of the emergent code of R1/R2.
# ----------------------------------------------------------------------
def _bonds_even(N):  return [(i, (i + 1) % N) for i in range(0, N, 2)]
def _bonds_odd(N):   return [(i, (i + 1) % N) for i in range(1, N, 2)]

def _measure_round(sim, checks):
    return [int(sim.measure_observable(ps)) for ps in checks]

def _checks_round(N, kind):
    if kind == 'A':  # ZZ on even bonds
        return [stim.PauliString('+' + ''.join('Z' if k in (a, b) else '_' for k in range(N)))
                for (a, b) in _bonds_even(N)]
    else:            # XX on odd bonds
        return [stim.PauliString('+' + ''.join('X' if k in (a, b) else '_' for k in range(N)))
                for (a, b) in _bonds_odd(N)]

def floquet_code_formation(N, cycles=10, seed=0):
    """Code formation: after each round the just-measured checks are exact
       stabilizers; we track how many checks of a round are ALREADY determined
       (|<check>|=1) by the accumulated stabilizer group BEFORE measuring them
       in the next period -> rises toward 1 as the dynamical code self-consistifies."""
    sim = stim.TableauSimulator(seed=seed)
    det = []
    prev = None
    for c in range(2 * cycles):
        kind = 'A' if c % 2 == 0 else 'B'
        checks = _checks_round(N, kind)
        if prev == kind or c >= 2:
            det.append(float(np.mean([abs(sim.peek_observable_expectation(ps)) == 1 for ps in checks])))
        _measure_round(sim, checks)
        prev = kind
    return det

def floquet_error_detection(N, n_trials=300, transient=6, seed=0):
    """After a transient, the round's checks are genuine stabilizers (peek = +-1
       deterministic).  Inject (or not) a random single-qubit Pauli error and
       re-peek the SAME checks: anticommuting errors flip a detectable subset
       (the syndrome).  Reports mean syndrome weight with vs without an error."""
    rng = np.random.RandomState(seed)
    w_err, w_clean, detected = [], [], []
    for t in range(n_trials):
        sim = stim.TableauSimulator(seed=int(rng.randint(1 << 30)))
        for c in range(transient):
            _measure_round(sim, _checks_round(N, 'A' if c % 2 == 0 else 'B'))
        kind = 'A' if transient % 2 == 0 else 'B'
        checks = _checks_round(N, kind)
        _measure_round(sim, checks)                       # now checks stabilise the state
        base = [sim.peek_observable_expectation(ps) for ps in checks]   # all +-1
        # clean: re-peek (no change)
        clean = [sim.peek_observable_expectation(ps) for ps in checks]
        w_clean.append(int(np.sum([a != b for a, b in zip(base, clean)])))
        # error: inject a random single-qubit Pauli, re-peek
        sim2 = sim.copy(); q = rng.randint(N); P = rng.choice(['X', 'Y', 'Z'])
        getattr(sim2, {'X': 'x', 'Y': 'y', 'Z': 'z'}[P])(q)
        after = [sim2.peek_observable_expectation(ps) for ps in checks]
        flips = int(np.sum([abs(a) == 1 and abs(b) == 1 and a != b for a, b in zip(base, after)])
                    + np.sum([abs(b) != 1 for b in after]))
        w_err.append(flips); detected.append(flips > 0)
    return {'syndrome_with_error': float(np.mean(w_err)),
            'syndrome_no_error': float(np.mean(w_clean)),
            'detected_fraction': float(np.mean(detected))}

def structured_vs_random_protection(N, depth, rate, n_traj=30, seed=0):
    """Compare reference-qubit protection at matched measurement rate:
       structured 2-body check schedule vs random single-qubit monitoring."""
    # random monitoring (reuse reference_protection at rate=`rate`)
    rnd = np.mean([reference_protection(N, rate, depth, seed=1234 + t, ring=True)
                   for t in range(n_traj)], axis=0)
    # structured: scramble + measure structured checks covering ~rate fraction
    rng = np.random.RandomState(seed); struct = []
    for t in range(n_traj):
        sim = stim.TableauSimulator(seed=int(rng.randint(1 << 30)))
        R = N; sim.h(R); sim.cnot(R, 0); traj = []
        for d in range(depth):
            bonds = _bonds_even(N) if d % 2 == 0 else _bonds_odd(N)
            for (a, b) in bonds:
                sim.do_tableau(stim.Tableau.random(2), [a, b])
            checks = _checks_round(N, 'A' if d % 2 == 0 else 'B')
            k = max(1, int(rate * len(checks)))
            idx = rng.choice(len(checks), k, replace=False)
            for j in idx:
                sim.measure_observable(checks[int(j)])
            traj.append(stab_entropy(sim.canonical_stabilizers(), N + 1, [R]))
        struct.append(traj)
    return {'random': rnd.tolist(), 'structured': np.mean(struct, axis=0).tolist(),
            'depth': depth, 'rate': rate, 'N': N}
