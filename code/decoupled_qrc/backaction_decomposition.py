"""
backaction_decomposition.py -- V2.2 Phase 13: decomposes the single
"interface on vs off" back-action check (V1/V2/V2.1) into FOUR conditions
-- full interface, memory-export-only-disabled (theta=0), processor-
reception-only-disabled (phi=0), and both disabled -- each compared
against the SAME "both disabled" baseline, isolating how much of the
total disturbance comes from the memory-ancilla coupling (theta) versus
the ancilla-processor coupling (phi).

Built entirely on `backaction_trajectory.compute_backaction_trajectory`
(unchanged, already tested) called three times against a common baseline
-- no new physics or circuit-building code.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .backaction_trajectory import compute_backaction_trajectory, BackActionTrajectory
from .directional_dqrc import DirectionalConfig
from .validation_utils import NestedSeeds


@dataclass
class BackActionDecomposition:
    full: BackActionTrajectory                       # (theta*, phi*) vs (0,0)
    memory_export_disabled: BackActionTrajectory      # (0, phi*) vs (0,0)
    processor_reception_disabled: BackActionTrajectory  # (theta*, 0) vs (0,0)


def decompose_backaction(cfg_star: DirectionalConfig, T: int, seeds: NestedSeeds) -> BackActionDecomposition:
    """`cfg_star` must already have its intended (theta*, phi*). Runs
    three `compute_backaction_trajectory` comparisons, each against the
    same "both disabled" (theta=0, phi=0) baseline, all on the SAME
    `seeds` object (common random numbers across all four conditions)."""
    cfg_both_off = replace(cfg_star, theta=0.0, phi=0.0)
    cfg_export_off = replace(cfg_star, theta=0.0, phi=cfg_star.phi)
    cfg_reception_off = replace(cfg_star, theta=cfg_star.theta, phi=0.0)

    full = compute_backaction_trajectory(cfg_star, cfg_both_off, T, seeds)
    memory_export_disabled = compute_backaction_trajectory(cfg_export_off, cfg_both_off, T, seeds)
    processor_reception_disabled = compute_backaction_trajectory(cfg_reception_off, cfg_both_off, T, seeds)

    return BackActionDecomposition(full=full, memory_export_disabled=memory_export_disabled,
                                    processor_reception_disabled=processor_reception_disabled)
