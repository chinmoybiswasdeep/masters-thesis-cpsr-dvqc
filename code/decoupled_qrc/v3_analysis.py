"""
v3_analysis.py -- ties the V3 architectures to the corrected capacity
machinery built and audited in V2/V2.1/V2.2 (`ipc_decomposition`,
`feature_analysis`, `frozen_protocol`), and adds the module-specific
quantities the V3 claim is about:

    M_memory              = M(X_M)          (memory module only)
    NL_instant_processor  = NL_{tau=0}(X_P) (processor module only)

plus the combined-readout metrics, reported but never allowed to replace
the module-specific causal test (a retrained combined readout may
legitimately mix both feature groups).

All four capacity forms required by the V3 spec are reported for every
feature group: raw, null mean, signed bias-corrected (C_raw - mu_null),
positive-part diagnostic (max(0, .)), and the unchanged legacy
significance-filtered capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .feature_analysis import diagnose_feature_group, fit_residualizer
from .frozen_protocol import freeze_protocol, evaluate_frozen
from .ipc_decomposition import compute_ipc_decomposed, diagnose_ceiling, nl_tensor_by_fixed_delay
from .utils import ensure_repo_code_on_path

ensure_repo_code_on_path()

from qrc_qiskit import chrono_split  # noqa: E402


@dataclass
class FeatureGroupReport:
    name: str
    n_features: int
    numerical_rank: int
    effective_rank: float
    ridge_dof: float
    n_train: int
    sample_to_effective_rank: float
    M_raw: float
    M_null: float
    M_signed: float
    M_positive: float
    M_legacy: float
    NL0_raw: float
    NL0_null: float
    NL0_signed: float
    NL0_positive: float
    NL0_legacy: float
    NL_temporal_signed: float
    NL_temporal_legacy: float
    nl_by_delay_signed: dict
    ceiling_fraction: float
    ceiling_contaminated: bool

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["nl_by_delay_signed"] = {str(k): v for k, v in self.nl_by_delay_signed.items()}
        return d


def analyse_feature_group(name: str, u, X, train, val, test, max_delay: int, max_degree: int,
                           max_targets_per_degree: int, n_surrogates: int, seed: int) -> FeatureGroupReport:
    """One feature group's full capacity + rank report."""
    decomp = compute_ipc_decomposed(u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
                                     max_targets_per_degree=max_targets_per_degree,
                                     n_surrogates=n_surrogates, seed=seed,
                                     always_include_single_delays=True)
    nl_tensor = nl_tensor_by_fixed_delay(decomp.records, max_delay)
    ceil = diagnose_ceiling(decomp, X, train)
    fdiag = diagnose_feature_group(X[train])

    nl0 = nl_tensor[0]
    nl_temporal_signed = sum(nl_tensor[t]["signed"] for t in range(1, max_delay + 1))
    nl_temporal_legacy = sum(nl_tensor[t]["legacy"] for t in range(1, max_delay + 1))

    return FeatureGroupReport(
        name=name, n_features=int(X.shape[1]), numerical_rank=fdiag.numerical_rank,
        effective_rank=fdiag.effective_rank, ridge_dof=fdiag.ridge_dof, n_train=len(train),
        sample_to_effective_rank=float(len(train) / max(fdiag.effective_rank, 1e-9)),
        M_raw=decomp.M_long_raw, M_null=decomp.M_long_null, M_signed=decomp.M_long_signed,
        M_positive=decomp.M_long_bc, M_legacy=decomp.M_long,
        NL0_raw=nl0["raw"], NL0_null=nl0["null"], NL0_signed=nl0["signed"],
        NL0_positive=nl0["bc"], NL0_legacy=nl0["legacy"],
        NL_temporal_signed=float(nl_temporal_signed), NL_temporal_legacy=float(nl_temporal_legacy),
        nl_by_delay_signed={t: nl_tensor[t]["signed"] for t in nl_tensor},
        ceiling_fraction=ceil.fraction_of_ceiling, ceiling_contaminated=ceil.ceiling_contaminated,
    )


@dataclass
class ArchitectureReport:
    architecture: str
    m: float
    g: float
    J: float
    lam: float
    reservoir_idx: int
    input_idx: int
    groups: dict = field(default_factory=dict)      # {group_name: FeatureGroupReport}
    n_qubits: int = 0
    circuit_depth: int = 0
    n_observables: int = 0
    tap_buffer_size: int = 0

    @property
    def M_memory(self) -> float:
        return self.groups["X_M"].M_signed

    @property
    def NL_instant_processor(self) -> float:
        return self.groups["X_P"].NL0_signed

    def as_dict(self) -> dict:
        return {"architecture": self.architecture, "m": self.m, "g": self.g, "J": self.J, "lam": self.lam,
                "reservoir_idx": self.reservoir_idx, "input_idx": self.input_idx,
                "n_qubits": self.n_qubits, "circuit_depth": self.circuit_depth,
                "n_observables": self.n_observables, "tap_buffer_size": self.tap_buffer_size,
                "M_memory": self.M_memory, "NL_instant_processor": self.NL_instant_processor,
                "groups": {k: v.as_dict() for k, v in self.groups.items()}}


def analyse_architecture_run(run, seeds, *, washout: int, n_val: int, n_test: int, max_delay: int,
                              max_degree: int, max_targets_per_degree: int, n_surrogates: int,
                              include_residualized: bool = True) -> ArchitectureReport:
    """Full feature-group analysis of one architecture run: X_M, X_P,
    X_M+X_P, and (optionally) X_P residualized against X_M with the
    projection fit on TRAIN ONLY."""
    T = len(run.u)
    gap = max_delay + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)
    ipc_seed = seeds.null_surrogate

    groups = {"X_M": run.X_M, "X_P": run.X_P, "X_M+X_P": np.hstack([run.X_M, run.X_P])}
    if include_residualized:
        applier = fit_residualizer(run.X_P, run.X_M, train)
        groups["X_P_perp"] = applier(run.X_P, run.X_M)

    reports = {}
    for name, X in groups.items():
        reports[name] = analyse_feature_group(name, run.u, X, train, val, test, max_delay=max_delay,
                                               max_degree=max_degree,
                                               max_targets_per_degree=max_targets_per_degree,
                                               n_surrogates=n_surrogates, seed=ipc_seed)

    cfg = run.cfg
    return ArchitectureReport(
        architecture=cfg.architecture, m=cfg.memory.m, g=cfg.processor.g, J=cfg.processor.J,
        lam=cfg.effective_lambda, reservoir_idx=seeds.reservoir_idx, input_idx=seeds.input_idx,
        groups=reports, n_qubits=run.n_qubits, circuit_depth=run.circuit_depth,
        n_observables=run.X_M.shape[1] + run.X_P.shape[1],
        tap_buffer_size=(cfg.tap_depth if cfg.architecture == "parallel_fixed_taps" else 0),
    )


@dataclass
class FrozenReadoutResult:
    center_capacity: float
    plus_capacity: float
    minus_capacity: float
    retrained_plus: float
    retrained_minus: float
    frozen_swing: float          # |plus - minus| under the FROZEN readout
    retrained_swing: float       # |plus - minus| under a RETRAINED readout
    interpretation: str


def frozen_readout_intervention(u_center, X_center, u_plus, X_plus, u_minus, X_minus, *,
                                 washout: int, n_val: int, n_test: int, max_delay: int, max_degree: int,
                                 max_targets_per_degree: int, n_surrogates: int, seed: int,
                                 target_delay: int = 0, degree_min: int = 2) -> FrozenReadoutResult:
    """Two analyses of the same perturbation pair:

    FROZEN: the target list, ridge alpha and null permutations are fixed
    once at the CENTRE point (`frozen_protocol.freeze_protocol`) and
    reused unchanged at +/- (`evaluate_frozen`). This measures how much
    the REPRESENTATION moved.

    RETRAINED: each point gets its own freshly-selected readout. This
    measures how much capacity survives after the readout adapts, and can
    legitimately HIDE a representation change -- which is exactly why both
    are reported.
    """
    T = len(u_center)
    gap = max_delay + 1
    train, val, test = chrono_split(T, washout, n_val, n_test, gap)

    spec = freeze_protocol(u_center, X_center, train, val, test, max_delay=max_delay,
                            max_degree=max_degree, max_targets_per_degree=max_targets_per_degree,
                            n_surrogates=n_surrogates, seed=seed,
                            always_include_single_delays=True)

    def _frozen_nl(u, X):
        recs = evaluate_frozen(u, X, train, val, test, spec)
        return nl_tensor_by_fixed_delay(recs, max_delay, degree_min=degree_min)[target_delay]["signed"]

    def _retrained_nl(u, X):
        d = compute_ipc_decomposed(u, X, train, val, test, max_delay=max_delay, max_degree=max_degree,
                                    max_targets_per_degree=max_targets_per_degree,
                                    n_surrogates=n_surrogates, seed=seed,
                                    always_include_single_delays=True)
        return nl_tensor_by_fixed_delay(d.records, max_delay, degree_min=degree_min)[target_delay]["signed"]

    center = nl_tensor_by_fixed_delay(spec.center_records, max_delay, degree_min=degree_min)[target_delay]["signed"]
    f_plus, f_minus = _frozen_nl(u_plus, X_plus), _frozen_nl(u_minus, X_minus)
    r_plus, r_minus = _retrained_nl(u_plus, X_plus), _retrained_nl(u_minus, X_minus)

    frozen_swing = abs(f_plus - f_minus)
    retrained_swing = abs(r_plus - r_minus)
    if frozen_swing > 2 * retrained_swing + 1e-12:
        interp = ("the frozen readout swings MORE than the retrained one: the representation moved, and "
                  "retraining partially hides it")
    elif retrained_swing > 2 * frozen_swing + 1e-12:
        interp = ("the retrained readout swings more than the frozen one: the capacity change is not simply "
                  "a fixed-representation shift")
    else:
        interp = "frozen and retrained readouts respond comparably"

    return FrozenReadoutResult(center_capacity=float(center), plus_capacity=float(f_plus),
                                minus_capacity=float(f_minus), retrained_plus=float(r_plus),
                                retrained_minus=float(r_minus), frozen_swing=float(frozen_swing),
                                retrained_swing=float(retrained_swing), interpretation=interp)


def matched_resource_row(report: ArchitectureReport) -> dict:
    """The resource-accounting row every architecture comparison must
    carry, so no comparison is made without stating what it cost."""
    return {"architecture": report.architecture, "n_qubits": report.n_qubits,
            "hilbert_dim": int(2 ** report.n_qubits) if report.n_qubits else None,
            "n_observables": report.n_observables, "circuit_depth": report.circuit_depth,
            "classical_tap_buffer": report.tap_buffer_size,
            "X_M_features": report.groups["X_M"].n_features,
            "X_P_features": report.groups["X_P"].n_features,
            "X_M_effective_rank": report.groups["X_M"].effective_rank,
            "X_P_effective_rank": report.groups["X_P"].effective_rank,
            "M_memory": report.M_memory, "NL_instant_processor": report.NL_instant_processor}
