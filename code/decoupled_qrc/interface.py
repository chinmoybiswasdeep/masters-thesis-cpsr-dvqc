"""
interface.py -- the tunable memory<->processor coupling U_MP(lambda_MP)
(Part 3). Deliberately the ONLY place information crosses between the memory
register (memory.py) and the EOC processor (processor.py) in the combined
DQRC circuit built by experiments.py -- everything else about the two
subsystems is independent, which is exactly what the decoupling experiment
(metrics.py's Jacobian) needs to be true for the comparison to mean anything.

**IMPORTANT PHYSICAL FINDING (docs/DQRC_ARCHITECTURE_REPAIR.md question 1/5)**:
'rzz' and 'cp' are BOTH diagonal in the processor-entry qubit's own Z basis.
A diagonal two-qubit gate cannot change a target qubit's Z-population --
only its relative phase. If the processor's OWN internal dynamics conserves
total Z-magnetization (true whenever it has few or no SYK4 quartic terms,
e.g. N_P<4 makes quartic terms IMPOSSIBLE at all since a term needs 4
distinct qubits -- `mixed_syk_core.sample_syk4_terms` returns an empty list
whenever `comb(N_P,4)==0`, and stays sparse up to N_P~6), a diagonal
interface CANNOT inject any amplitude/coherence into the processor at all --
verified directly: `diagnostics.info_theoretic_diagnostics` shows
purity_P stays EXACTLY 1.0 (zero entanglement with memory) for 'rzz'/'cp' at
ANY lambda_mp, while 'zx' (which acts as an X-rotation on the processor
qubit, non-diagonal in its own basis) immediately produces real entanglement
and nonzero mutual information. Prefer 'zx' whenever the processor might be
in, or reachable from, a low-magnetization-breaking regime; a caller
choosing 'rzz'/'cp' anyway (e.g. specifically to test the failure mode, or
because the processor's own quartic dynamics is rich enough at larger N_P to
break the conservation law itself) should verify with
`diagnostics.info_theoretic_diagnostics` that real M-P entanglement is
actually being created, rather than assuming it.
"""
from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit

INTERFACE_KINDS = ("rzz", "cp", "zx")


def apply_interface(qc: QuantumCircuit, memory_taps: Sequence[int], processor_entry: Sequence[int],
                     lambda_mp: float, kind: str = "rzz"):
    """Couple each memory tap qubit to the corresponding processor entry
    qubit (paired by position; `len(memory_taps)` must equal
    `len(processor_entry)`) with strength `lambda_mp`:

        'rzz': exp(-i*lambda_mp * Z_M Z_P)   (qc.rzz)
        'cp':  controlled-phase(2*lambda_mp) (qc.cp)
        'zx':  exp(-i*lambda_mp * Z_M X_P), via qc.h on the P qubit around an
               rzz (H . RZZ . H == exp(-i*lambda*Z x X) on the second qubit)

    `lambda_mp=0` must be an exact identity (no coupling at all) for every
    kind, by construction -- this is what lets the Jacobian experiment
    (metrics.py) treat lambda_mp=0 as the true "memory and processor fully
    disconnected" baseline.
    """
    if kind not in INTERFACE_KINDS:
        raise ValueError(f"kind must be one of {INTERFACE_KINDS}, got {kind!r}")
    if len(memory_taps) != len(processor_entry):
        raise ValueError("memory_taps and processor_entry must have equal length (paired 1:1)")
    if lambda_mp == 0:
        return
    for m, p in zip(memory_taps, processor_entry):
        if kind == "rzz":
            qc.rzz(2.0 * lambda_mp, m, p)
        elif kind == "cp":
            qc.cp(2.0 * lambda_mp, m, p)
        elif kind == "zx":
            qc.h(p)
            qc.rzz(2.0 * lambda_mp, m, p)
            qc.h(p)


def interface_two_qubit_gate_count(n_taps: int) -> int:
    """Every `apply_interface` kind uses exactly one 2-qubit primitive per
    tap pair (rzz directly; cp directly; zx wraps rzz in single-qubit H's) --
    used for honest resource accounting (Part 9)."""
    return n_taps
