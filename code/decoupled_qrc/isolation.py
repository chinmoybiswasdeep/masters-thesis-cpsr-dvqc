"""
isolation.py -- V3's structural (representation-level) isolation tests,
run BEFORE any capacity is computed.

For Architectures B and C, with identical inputs and seeds:
    X_P(m+delta, g, J) must equal X_P(m-delta, g, J)
    X_M(m, g+delta, J) must equal X_M(m, g-delta, J)
    X_M(m, g, J+delta) must equal X_M(m, g, J-delta)
within a documented numerical tolerance, and the processor's Hamiltonian
matrices must be bit-identical across changes in `m`.

These tests are deliberately falsifiable: `inject_cross_dependency`
produces a config wrapper whose processor secretly depends on `m`, and
`tests/test_isolation.py` asserts the isolation check FAILS for it. A
test that cannot fail proves nothing.

The three levels the V3 spec insists on distinguishing:
  * STRUCTURAL isolation  -- feature matrices identical (this module);
  * MODULE-SPECIFIC task selectivity -- M(X_M) vs NL_0(X_P) responses;
  * COMBINED-READOUT selectivity -- a retrained readout over X_M ⊕ X_P,
    which may legitimately mix both groups and is NOT required to show
    exact zero cross-derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .dual_route import DualRouteConfig, run_architecture
from .memory_bank import MemoryBankConfig
from .nonlinear_processor import ProcessorConfig, h0_matrix, input_operators
from .v3_seeds import NestedSeeds

ATOL = 1e-10
RTOL = 1e-9


@dataclass
class IsolationResult:
    architecture: str
    max_abs_dXP_dm: float          # max |X_P(m+d) - X_P(m-d)|
    max_abs_dXM_dg: float
    max_abs_dXM_dJ: float
    processor_hamiltonian_identical: bool
    memory_config_identical: bool
    atol: float
    rtol: float
    passed: bool
    failures: list


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def check_structural_isolation(cfg: DualRouteConfig, T: int, seeds: NestedSeeds,
                                delta_m: float = 0.05, delta_g: float = 0.05, delta_J: float = 0.05,
                                atol: float = ATOL, rtol: float = RTOL,
                                run_fn=run_architecture) -> IsolationResult:
    """Runs the +/- perturbation pairs on the SAME input realization
    (common random numbers: `u` is drawn once and passed explicitly to
    every run) and compares the module feature matrices."""
    u = None
    failures = []

    def _cfg_m(m_val):
        return replace(cfg, memory=replace(cfg.memory, m=m_val))

    def _cfg_gJ(g_val, J_val):
        return replace(cfg, processor=replace(cfg.processor, g=g_val, J=J_val))

    m0, g0, J0 = cfg.memory.m, cfg.processor.g, cfg.processor.J

    run_seed = run_fn(cfg, T, seeds, u=u)
    u = run_seed.u   # freeze the exact input realization for every subsequent run

    r_m_plus = run_fn(_cfg_m(m0 + delta_m), T, seeds, u=u)
    r_m_minus = run_fn(_cfg_m(m0 - delta_m), T, seeds, u=u)
    max_dXP_dm = _max_abs_diff(r_m_plus.X_P, r_m_minus.X_P)

    r_g_plus = run_fn(_cfg_gJ(g0 + delta_g, J0), T, seeds, u=u)
    r_g_minus = run_fn(_cfg_gJ(g0 - delta_g, J0), T, seeds, u=u)
    max_dXM_dg = _max_abs_diff(r_g_plus.X_M, r_g_minus.X_M)

    r_J_plus = run_fn(_cfg_gJ(g0, J0 + delta_J), T, seeds, u=u)
    r_J_minus = run_fn(_cfg_gJ(g0, J0 - delta_J), T, seeds, u=u)
    max_dXM_dJ = _max_abs_diff(r_J_plus.X_M, r_J_minus.X_M)

    # the processor Hamiltonian itself must not move when m moves
    h_plus = h0_matrix(_cfg_m(m0 + delta_m).processor)
    h_minus = h0_matrix(_cfg_m(m0 - delta_m).processor)
    v_plus = input_operators(_cfg_m(m0 + delta_m).processor)
    v_minus = input_operators(_cfg_m(m0 - delta_m).processor)
    ham_identical = bool(np.array_equal(h_plus, h_minus)
                          and all(np.array_equal(a, b) for a, b in zip(v_plus, v_minus)))

    # the memory config must not move when (g,J) move
    mem_identical = (_cfg_gJ(g0 + delta_g, J0).memory.as_dict()
                      == _cfg_gJ(g0 - delta_g, J0).memory.as_dict()
                      == cfg.memory.as_dict())

    if max_dXP_dm > atol:
        failures.append(f"X_P moved with m: max|dX_P| = {max_dXP_dm:.3e} > atol {atol:.1e}")
    if max_dXM_dg > atol:
        failures.append(f"X_M moved with g: max|dX_M| = {max_dXM_dg:.3e} > atol {atol:.1e}")
    if max_dXM_dJ > atol:
        failures.append(f"X_M moved with J: max|dX_M| = {max_dXM_dJ:.3e} > atol {atol:.1e}")
    if not ham_identical:
        failures.append("processor Hamiltonian/input operators changed with m")
    if not mem_identical:
        failures.append("memory configuration changed with (g,J)")

    return IsolationResult(architecture=cfg.architecture, max_abs_dXP_dm=max_dXP_dm,
                            max_abs_dXM_dg=max_dXM_dg, max_abs_dXM_dJ=max_dXM_dJ,
                            processor_hamiltonian_identical=ham_identical,
                            memory_config_identical=mem_identical, atol=atol, rtol=rtol,
                            passed=(len(failures) == 0), failures=failures)


def inject_cross_dependency(strength: float = 0.1):
    """Returns a drop-in replacement for `run_architecture` whose
    PROCESSOR secretly depends on the memory control `m` (it leaks `m`
    into the processor's transverse field). `check_structural_isolation`
    MUST fail when handed this -- the falsifiability proof for the whole
    isolation test."""
    def _leaky_run(cfg: DualRouteConfig, T: int, seeds: NestedSeeds, u=None, **kwargs):
        leaked = replace(cfg, processor=replace(cfg.processor,
                                                 g=cfg.processor.g + strength * cfg.memory.m))
        return run_architecture(leaked, T, seeds, u=u, **kwargs)
    return _leaky_run
