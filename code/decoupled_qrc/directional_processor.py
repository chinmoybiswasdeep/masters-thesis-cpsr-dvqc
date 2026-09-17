"""
directional_processor.py -- thin wrapper around the already-repaired EOC
processor (`processor.py`, reused verbatim, Part 7). The only NEW content
here is a convenience API that forces callers to characterize/locate the
EOC region INDEPENDENTLY for whatever N_P they actually use in the
directional architecture, rather than reusing a value found at a different
system size (the audit finding from `docs/DQRC_ARCHITECTURE_REPAIR.md`
this repo has already been burned by once).
"""
from __future__ import annotations

from . import processor as procmod
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()


def characterize_processor(N_P: int, kappa_grid, term_seed: int = 0, disorder_seed: int = 0,
                            reps: int = 1, n_ref_trials: int = 15) -> list:
    """Two chaos diagnostics (operator entanglement, level-spacing ratio vs.
    Poisson/COE/CUE) across `kappa_grid`, computed FRESH for this N_P --
    `processor.eoc_scan`, reused directly, not reimplemented."""
    if N_P < 4:
        raise ValueError(f"N_P={N_P} < 4 has ZERO possible SYK4 quartic terms "
                          f"(comb(N_P,4)==0) -- Part 7 explicitly forbids using N_P<4 for "
                          f"quartic/SYK4 claims (see docs/DQRC_ARCHITECTURE_REPAIR.md finding B).")
    return procmod.eoc_scan(N_P, kappa_grid, term_seed=term_seed, disorder_seed=disorder_seed,
                             reps=reps, n_ref_trials=n_ref_trials)


def find_eoc_region(scan_results: list) -> float:
    """DQRC's own independently-derived EOC estimate for THIS N_P --
    `processor.find_eoc_kappa`, reused directly."""
    return procmod.find_eoc_kappa(scan_results)


def run_standalone(N_P: int, kappa_processor: float, T: int, reps: int = 1, max_weight: int = 3,
                    term_seed: int = 0, disorder_seed: int = 0, input_seed: int = 0):
    """`processor.run_processor_standalone`, reused directly -- the
    processor's own reference ceiling for eta_NL."""
    if N_P < 4:
        raise ValueError(f"N_P={N_P} < 4 has ZERO possible SYK4 quartic terms; use N_P>=5.")
    return procmod.run_processor_standalone(N_p=N_P, kappa_processor=kappa_processor, T=T, reps=reps,
                                             max_weight=max_weight, term_seed=term_seed,
                                             disorder_seed=disorder_seed, input_seed=input_seed)


def sample_params(N_P: int, kappa_processor: float, term_seed: int = 0, disorder_seed: int = 0,
                   reps: int = 1, G_MAX: float = 0.6, J_MAX: float = 0.6):
    """`processor.sample_processor_params`, reused directly -- for building
    the processor's own layer subcircuit inside the directional combined
    circuit (`directional_dqrc.py`)."""
    if N_P < 4:
        raise ValueError(f"N_P={N_P} < 4 has ZERO possible SYK4 quartic terms; use N_P>=5.")
    return procmod.sample_processor_params(N_P, kappa_processor, term_seed=term_seed,
                                            disorder_seed=disorder_seed, reps=reps, G_MAX=G_MAX, J_MAX=J_MAX)


def sample_params_gJ(N_P: int, g: float, J: float, term_seed: int = 0, disorder_seed: int = 0,
                      reps: int = 1):
    """`processor.sample_processor_params_gJ`, reused directly -- g and J
    supplied INDEPENDENTLY (docs/DQRC_GJ_EOC_AUDIT.md Part 2), not
    constrained to the kappa anti-diagonal."""
    if N_P < 4:
        raise ValueError(f"N_P={N_P} < 4 has ZERO possible SYK4 quartic terms; use N_P>=5.")
    return procmod.sample_processor_params_gJ(N_P, g, J, term_seed=term_seed,
                                               disorder_seed=disorder_seed, reps=reps)


def layer_subcircuit(params):
    return procmod.processor_reps_subcircuit(params)
