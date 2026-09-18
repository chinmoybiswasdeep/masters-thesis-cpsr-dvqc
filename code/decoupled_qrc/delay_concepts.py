"""
delay_concepts.py -- V2.2 Phase 2: replaces V2.1's single, mislabeled
`ell_0` with three EXPLICITLY DISTINCT delay concepts, computed by three
different mechanisms, never substituted for one another:

  ell_causal  -- the physical causal latency: the first delay at which an
                 INTERVENTION on u_t changes the features (see
                 `causal_intervention.py`). A structural/circuit property.

  ell_detect  -- the earliest STATISTICALLY DETECTABLE delay: the smallest
                 tau at which C_{1,tau} clears its own null/significance
                 threshold (reuses `ipc.compute_ipc_detailed`'s existing
                 `significant` flag -- no new capacity computation). A
                 statistical-power property (depends on n_surrogates, T,
                 feature richness), NOT a physical latency.

  ell_peak    -- argmax_tau C_{1,tau}: V2.1's own `ell_0`, kept under its
                 correct name. A property of WHERE memory capacity happens
                 to be largest, never called "causal latency" anywhere in
                 V2.2's code, tests, notebook, or report.

All three are reported side by side; Phase 2 explicitly forbids treating
`ell_peak` as `ell_causal`.
"""
from __future__ import annotations

from dataclasses import dataclass


def ell_peak_from_records(records: list, degree: int = 1) -> int:
    """argmax_tau C_{1,tau} (bias-corrected, continuous -- Defect 5/Phase 4
    of V2.1, carried forward: never select using the hard-thresholded
    legacy capacity). This is V2.1's `ell_0`, renamed to make clear it is
    NOT a causal-latency claim."""
    degree_records = [r for r in records if r.degree == degree and len(r.delays) == 1]
    if not degree_records:
        return 0
    best = max(degree_records, key=lambda r: (max(0.0, r.raw_capacity - r.null_mean), -r.delays[0]))
    return int(best.delays[0])


def ell_detect_from_records(records: list, degree: int = 1):
    """The smallest tau at which the single-delay degree-`degree` profile
    is flagged `significant` by `ipc.compute_ipc_detailed`'s own
    surrogate-based null test. Returns None if no delay is significant
    (a legitimate, reportable outcome -- not an error)."""
    degree_records = [r for r in records if r.degree == degree and len(r.delays) == 1 and r.significant]
    if not degree_records:
        return None
    return int(min(r.delays[0] for r in degree_records))


@dataclass
class ThreeDelayConcepts:
    ell_causal: object      # int or None -- from causal_intervention.InterventionResult.ell_causal
    ell_detect: object      # int or None
    ell_peak: int
    causal_equals_peak: bool = None
    causal_equals_detect: bool = None

    def __post_init__(self):
        self.causal_equals_peak = (self.ell_causal is not None and self.ell_causal == self.ell_peak)
        self.causal_equals_detect = (self.ell_causal is not None and self.ell_causal == self.ell_detect)


def summarize_three_delays(intervention_result, records: list, degree: int = 1) -> ThreeDelayConcepts:
    """Bundles all three delay concepts for one candidate/seed into a
    single reportable record."""
    return ThreeDelayConcepts(
        ell_causal=intervention_result.ell_causal if intervention_result is not None else None,
        ell_detect=ell_detect_from_records(records, degree=degree),
        ell_peak=ell_peak_from_records(records, degree=degree),
    )
