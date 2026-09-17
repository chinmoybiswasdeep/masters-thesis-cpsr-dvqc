"""
experimental/idqnn_memory_prototype.py -- EXPERIMENTAL, NOT part of the DQRC
architecture proper. Part 5 of the task spec.

This file implements ONLY the honestly-named `ideal_depth_compressed_control`
(apply `matrix_power(U, D)` once instead of D sequential layer applications,
with ONE noise-channel application instead of D) -- the SAME pattern the
audit found already in this repo's own `idcpsr.py` (`mode='id'` in
`noisy_reservoir_features`, docs/MEMORY_NL_DECOUPLED_QRC_AUDIT.md section 8),
here deliberately given a name that does NOT claim it is Huang et al.'s
IDQNN construction. It is a noise-application-COUNT control, not a
hardware-realistic depth/fidelity trade-off for actually compiling U^D, and
not a claim about spatializing temporal resources the way IDQNN does.

WHAT WOULD BE REQUIRED FOR A GENUINE IDQNN CLAIM (Part 5), NONE OF WHICH IS
IMPLEMENTED HERE:
  1. an explicit shallow-WIDE circuit construction (not a single matrix-power
     gate applied as one opaque unitary block),
  2. explicit additional ancilla/width overhead reported alongside the
     resulting shallower depth (not "for free"),
  3. measurement / feed-forward / postselection, if the specific
     Huang-mapping/MBQC construction being claimed requires it,
  4. a quantitative comparison (fidelity / total variation distance /
     XEB-like score) of the shallow-wide circuit's OUTPUT DISTRIBUTION
     against the corresponding deep circuit's,
  5. physical depth and width reported SEPARATELY, not folded into one
     number.

Ablation #13 in Part 11 ("ideal U^D one-step control, explicitly NOT called
IDQNN") uses `ideal_depth_compressed_control` below. No ablation in this
project calls anything here "IDQNN".
"""
from __future__ import annotations

import numpy as np


def ideal_depth_compressed_control(U_layer: np.ndarray, D: int, rho_in: np.ndarray,
                                    noise_channel) -> np.ndarray:
    """Apply `U_layer` D times' worth of unitary evolution as ONE
    `matrix_power(U_layer, D)` conjugation, followed by exactly ONE
    application of `noise_channel` -- an IDEALIZED best case for what
    depth-compression could buy if a real device could realize U^D at unit
    cost, NOT a claim that it can. Contrast with `deep_reference` below,
    which applies `U_layer` and `noise_channel` D separate times -- the
    honest baseline this control is compared against.

    `noise_channel(rho) -> rho` must be a valid CPTP map (e.g. a depolarizing
    channel), applied identically in both this function and
    `deep_reference` so the ONLY difference between them is the number of
    times noise is injected (1 vs D), which is the entire point of the
    comparison and the entire reason this is NOT presented as evidence of a
    real hardware advantage.
    """
    U_D = np.linalg.matrix_power(U_layer, D)
    rho_out = U_D @ rho_in @ U_D.conj().T
    return noise_channel(rho_out)


def deep_reference(U_layer: np.ndarray, D: int, rho_in: np.ndarray, noise_channel) -> np.ndarray:
    """The honest baseline: apply `U_layer` and `noise_channel` D separate
    times, sequentially."""
    rho = rho_in
    for _ in range(D):
        rho = U_layer @ rho @ U_layer.conj().T
        rho = noise_channel(rho)
    return rho


def depolarizing_channel(p: float):
    """A minimal single global depolarizing channel: rho -> (1-p) rho +
    p * I/dim -- deliberately the simplest possible CPTP map, so this
    module's only claim is about noise-application COUNT, not about any
    particular physically-motivated noise model."""
    def apply(rho: np.ndarray) -> np.ndarray:
        dim = rho.shape[0]
        return (1 - p) * rho + p * np.eye(dim) / dim
    return apply
