"""
stitching.py -- Part 4: the modular/"sewn" processor architecture, split into
local blocks P1..PB coupled by ANCILLA-MEDIATED STITCHING.

TERMINOLOGY (per docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 8 / the task
spec's explicit rule): what is implemented below is a fresh-ancilla,
RZZ-coupled transient link between adjacent blocks -- NOT Huang et al.'s
local-inversion/divide-and-conquer sewing construction (that needs per-block
local process inversion, light-cone restriction, and ancilla-assisted
reconstruction of a GLOBAL target transformation, verified against that
target's own action). The repo's OWN `idcpsr.py` shows what a *verified*
sewing claim looks like (a numerically-checked controlled-copy = local-
dephasing channel identity, `docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md` section
8) -- this module does not reach that bar and does not claim to.

`genuine_local_inversion_demonstrator` is intentionally UNIMPLEMENTED (raises
NotImplementedError with a description of what it would require) rather than
shipping a superficial stand-in mislabeled as "Huang-style sewing" -- per the
project plan, this was the first piece cut under the FAST_MODE time budget;
see docs/DQRC_RESULTS.md question 9 for the honest accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit

from .processor import ProcessorParams, sample_processor_params, processor_reps_subcircuit


@dataclass
class ModularBlock:
    params: ProcessorParams
    qubits: list  # global qubit indices this block occupies


def build_modular_blocks(block_sizes: Sequence[int], kappas: Sequence[float], term_seed_base: int,
                          disorder_seed_base: int, reps: int, start_qubit: int = 0,
                          G_MAX: float = 0.6, J_MAX: float = 0.6) -> list:
    """One `ProcessorParams` per block, each independently sampled (own
    term/disorder seed derived from the block index) -- `kappas` lets blocks
    be heterogeneous (Part 4's 'heterogeneous criticality' follow-on); pass
    the same kappa for every block to get the 'all g_j = g_c' baseline the
    spec asks be tried first."""
    blocks = []
    q = start_qubit
    for i, (size, kappa) in enumerate(zip(block_sizes, kappas)):
        params = sample_processor_params(size, kappa, term_seed=term_seed_base + i,
                                          disorder_seed=disorder_seed_base + i, reps=reps,
                                          G_MAX=G_MAX, J_MAX=J_MAX)
        blocks.append(ModularBlock(params=params, qubits=list(range(q, q + size))))
        q += size
    return blocks


def apply_ancilla_stitch(qc: QuantumCircuit, tap_a: int, tap_b: int, ancilla: int, lambda_stitch: float):
    """Transient ancilla-mediated link between two block-boundary qubits: a
    fresh ancilla (reset first) picks up a weak ZZ-correlation with `tap_a`,
    carries it to `tap_b` via a second weak ZZ-correlation, then is reset
    again -- so no information persists in the ancilla across timesteps (it
    is a resource used and discarded each application, never an extra hidden
    memory channel). `lambda_stitch=0` is an exact identity, same convention
    as `interface.apply_interface`."""
    if lambda_stitch == 0:
        return
    qc.reset(ancilla)
    qc.rzz(2.0 * lambda_stitch, tap_a, ancilla)
    qc.rzz(2.0 * lambda_stitch, ancilla, tap_b)
    qc.reset(ancilla)


def modular_processor_subcircuit(blocks: list, ancilla_qubits: Sequence[int], lambda_stitch: float,
                                  n_total_qubits: int) -> QuantumCircuit:
    """One layer of every block's own mixed-SYK dynamics (each block's
    `processor_reps_subcircuit`, composed onto its own qubits) followed by
    ancilla-mediated stitching between EACH pair of adjacent blocks (using
    the last qubit of block i and the first qubit of block i+1 as taps).
    `lambda_stitch=0` reduces exactly to `B` disconnected processor blocks
    (ablation #6, "DQRC without stitching")."""
    qc = QuantumCircuit(n_total_qubits)
    for block in blocks:
        qc.compose(processor_reps_subcircuit(block.params), qubits=block.qubits, inplace=True)
    for i in range(len(blocks) - 1):
        tap_a = blocks[i].qubits[-1]
        tap_b = blocks[i + 1].qubits[0]
        apply_ancilla_stitch(qc, tap_a, tap_b, ancilla_qubits[i], lambda_stitch)
    return qc


def n_ancillas_needed(n_blocks: int) -> int:
    return max(0, n_blocks - 1)


def genuine_local_inversion_demonstrator(*args, **kwargs):
    """NOT IMPLEMENTED in this FAST_MODE pass (Part 4: 'if feasible' --
    deliberately cut here, see docs/DQRC_RESULTS.md question 9).

    A real Huang-et-al.-style local-inversion demonstrator for N<=6 would
    need, at minimum:
      1. a fixed GLOBAL target transformation U_target on N qubits,
      2. independent LOCAL process learning/inversion restricted to each
         block's own light cone (not just applying each block's own native
         dynamics, as `modular_processor_subcircuit` does),
      3. an ancilla-assisted RECONSTRUCTION protocol that combines the local
         inversions back into an approximation of U_target (not merely a
         transient correlation link, as `apply_ancilla_stitch` implements),
      4. a numerical check that the reconstructed transformation's action
         agrees with U_target within a stated fidelity/trace-distance
         tolerance, on either the full unitary or a representative set of
         input states.
    None of steps 2-4 are implemented here. Calling this function raises
    NotImplementedError rather than silently returning a superficial result,
    so it cannot be mistaken for a completed feature.
    """
    raise NotImplementedError(
        "genuine_local_inversion_demonstrator is not implemented in this pass -- see this "
        "function's own docstring for what it would require, and docs/DQRC_RESULTS.md question 9 "
        "for why it was cut under the FAST_MODE time budget rather than shipped superficially.")
