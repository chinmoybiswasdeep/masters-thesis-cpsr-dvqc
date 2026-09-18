"""
exact_v1_reproduction.py -- V2.1 Defect 8: V2's own "Part 1 reproduction"
silently changed the protocol (different T, delays, degrees, targets,
seeds) while still calling it a reproduction. This module runs the EXACT
V1 protocol, verbatim, reconstructed from V1's own notebook
(`code/DQRC_GJ_Decoupling_Validation.ipynb`, cell 3) and its saved raw
results (`results/dqrc_gj/validation_notebook_results.json`):

    CFG            = utils.active_config()          # FAST_CONFIG (V1 never ran PUBLICATION_MODE)
    seeds0         = utils.make_seed_bundle(0)       # the OLD seed system, not NestedSeeds
    (g, J)         = (0.6, 0.3295681629918353)       # V1's own reproduction point
    T              = CFG.T_ipc                       # =250
    IPC estimator  = diagnostics.ipc_MN(u, X, CFG, seed=0)   # V1's own legacy wrapper, NOT
                                                              # ipc_decomposition.compute_ipc_decomposed

Reference numbers (from `results/dqrc_gj/validation_notebook_results.json`,
committed alongside V1): M=5.918476712586813, NL=4.674060175752677.

This function changes NOTHING about how V1 computed these numbers -- it
is provided so V2.1's own notebook can re-run V1's exact protocol on
today's code (catching any accidental regression in the UNDERLYING,
still-shared `processor.run_processor_standalone_gJ`/`diagnostics.ipc_MN`
machinery) and report the comparison honestly, side-by-side with a
SEPARATELY LABELED V2.1 reevaluation of the same physical point under the
corrected protocol -- never conflating the two.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import diagnostics as diag
from . import processor as procmod
from .utils import active_config, make_seed_bundle, ensure_repo_code_on_path

ensure_repo_code_on_path()

V1_REFERENCE = {
    "M": 5.918476712586813,
    "NL": 4.674060175752677,
    "source": "results/dqrc_gj/validation_notebook_results.json:part1_reproduction",
}
V1_G, V1_J = 0.6, 0.3295681629918353
V1_N_P = 5


@dataclass
class ExactV1ReproductionResult:
    M: float
    NL: float
    reference_M: float
    reference_NL: float
    M_matches_reference: bool
    NL_matches_reference: bool
    config_used: dict


def run_exact_v1_reproduction(abs_tol: float = 1e-6) -> ExactV1ReproductionResult:
    """Reproduces V1's cell-3 computation verbatim: same CFG (FAST_CONFIG
    via `active_config()`), same seed derivation (`make_seed_bundle(0)`),
    same (g,J), same `diagnostics.ipc_MN` legacy path -- NOT
    `ipc_decomposition.compute_ipc_decomposed`, which V1 never used."""
    CFG = active_config()
    seeds0 = make_seed_bundle(0)
    labels, X, u, info = procmod.run_processor_standalone_gJ(
        V1_N_P, g=V1_G, J=V1_J, T=CFG.T_ipc, reps=1, max_weight=3,
        term_seed=seeds0.reservoir_seed, disorder_seed=seeds0.reservoir_seed + 1, input_seed=seeds0.dataset_seed)
    M, NL, total, _ = diag.ipc_MN(u, X, CFG, seed=0)

    return ExactV1ReproductionResult(
        M=M, NL=NL, reference_M=V1_REFERENCE["M"], reference_NL=V1_REFERENCE["NL"],
        M_matches_reference=abs(M - V1_REFERENCE["M"]) < abs_tol,
        NL_matches_reference=abs(NL - V1_REFERENCE["NL"]) < abs_tol,
        config_used=dict(T_ipc=CFG.T_ipc, washout=CFG.washout, n_val=CFG.n_val, n_test=CFG.n_test,
                          max_delay_ipc=CFG.max_delay_ipc, max_degree_ipc=CFG.max_degree_ipc,
                          max_targets_per_degree=CFG.max_targets_per_degree, n_surrogates=CFG.n_surrogates,
                          g=V1_G, J=V1_J, N_P=V1_N_P, seed_scheme="make_seed_bundle(0)"),
    )


UNRECONSTRUCTABLE_FROM_V1 = [
    "V1's notebook did not record which qiskit-aer/qiskit package versions or git commit produced "
    "its saved numbers (no provenance block in that pass) -- an exact environment match cannot be "
    "verified, only a numerical match at fixed abs_tol.",
    "V1 used a SINGLE seed (master_seed=0) for this specific reproduction cell -- no multi-seed "
    "variance information exists to compare against for this exact point.",
]
