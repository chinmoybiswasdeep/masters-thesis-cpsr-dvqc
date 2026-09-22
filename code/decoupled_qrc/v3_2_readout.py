"""
v3_2_readout.py -- the finite-shot readout model and matched-resource
accounting for V3.2.

WHY A SHOT BUDGET IS PART OF THE SCIENCE, NOT AN OPTIONAL EXTRA
---------------------------------------------------------------------------
With a scalar input and a noiseless simulator, instantaneous capacity is
DEGENERATE: once the features span polynomials of degree <= N_P, the exact
capacity C_d equals 1.000 for every d <= N_P. Measured during planning at
N_P=5: NL_0 goes 0.003 -> 1.645 between (g,J) = (0,0) and (0.1,0.1) and is
then flat to (1.3,1.3). There is no derivative to estimate and no dynamic
range, so the processor-controllability gate is unreachable BY CONSTRUCTION.
This is intrinsic to the estimator's setting, not a bug, and it is the same
ceiling that pinned V3.1's primary metric at its target count.

A finite shot budget restores a continuous, differentiable metric, because
the achievable capacity of each degree becomes limited by the
signal-to-noise ratio of that degree's contribution to the features.
Measured at N_P=5, 10k shots/setting: NL_0 = 0.003, 1.644, 2.224, 2.544,
2.680, 2.261, 2.380 across g=J = 0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.3 -- a real
response surface with an interior maximum.

The shot budget S is therefore a PREREGISTERED RESOURCE: fixed during
CALIBRATION (before any discovery data is seen), entered into the cache key,
and matched across every architecture and baseline.

NOISE MODEL. For a Pauli observable with exact expectation p, an S-shot
estimate is (2 B - S)/S with B ~ Binomial(S, (1+p)/2), which has mean p and
variance (1 - p^2)/S. `add_shot_noise` uses the Gaussian limit of that
distribution, which is ~1000x cheaper than sampling bitstrings.
`sample_shot_noise_exact` implements the true binomial draw, and
`verify_shot_noise_model` checks the two agree -- so the approximation is
validated rather than assumed (tests/test_v3_2_readout.py).

MEASUREMENT SETTINGS ARE COUNTED HONESTLY. X_i, Y_i and Z_i do not commute,
so a single shot budget cannot estimate all of them at once. `group_settings`
partitions the readout set into qubit-wise-commuting groups (a setting assigns
one basis per qubit; a string is measurable in that setting iff every factor
matches). The reported resource is n_settings * S shots PER TIMESTEP, not S.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parse_label(label: str) -> dict:
    """'X0' -> {0:'X'};  'X0Z2' -> {0:'X', 2:'Z'}. Matches the label format
    produced by `v3_2_encoder.local_pauli_ops` and `memory_feature_ops`."""
    out, i = {}, 0
    while i < len(label):
        p = label[i]
        if p not in ("X", "Y", "Z"):
            raise ValueError(f"bad Pauli label {label!r} at position {i}")
        j = i + 1
        while j < len(label) and label[j].isdigit():
            j += 1
        if j == i + 1:
            raise ValueError(f"bad Pauli label {label!r}: no qubit index after {p}")
        out[int(label[i + 1:j])] = p
        i = j
    return out


def group_settings(labels) -> list:
    """Greedy partition of Pauli labels into qubit-wise-commuting settings.

    Two strings share a setting iff they never assign different bases to the
    same qubit. Greedy is sufficient here (the readout sets are tiny and
    fixed); the point is an honest, reproducible setting COUNT for resource
    matching, not an optimal grouping.
    """
    settings = []          # list of {qubit: basis}
    members = []           # parallel list of label lists
    for lab in labels:
        need = parse_label(lab)
        placed = False
        for k, assigned in enumerate(settings):
            if all(assigned.get(q, p) == p for q, p in need.items()):
                assigned.update(need)
                members[k].append(lab)
                placed = True
                break
        if not placed:
            settings.append(dict(need))
            members.append([lab])
    return [{"bases": s, "labels": m} for s, m in zip(settings, members)]


@dataclass(frozen=True)
class ShotBudget:
    """`shots` is per measurement SETTING, per timestep. `shots=None` means
    the noiseless limit and is only ever used for structural diagnostics."""

    shots: int = 10000

    def __post_init__(self):
        if self.shots is not None and self.shots < 1:
            raise ValueError(f"shots must be >= 1 or None, got {self.shots}")

    @property
    def noiseless(self) -> bool:
        return self.shots is None

    def as_dict(self) -> dict:
        return {"shots_per_setting": self.shots}

    def total_shots_per_step(self, labels) -> int:
        if self.shots is None:
            return 0
        return int(len(group_settings(labels)) * self.shots)


def shot_std(values: np.ndarray, shots) -> np.ndarray:
    """Per-entry standard error sqrt((1 - p^2)/S) of an S-shot Pauli estimate."""
    if shots is None:
        return np.zeros_like(np.asarray(values, dtype=float))
    v = np.asarray(values, dtype=float)
    return np.sqrt(np.clip(1.0 - v ** 2, 0.0, None) / float(shots))


def add_shot_noise(features: np.ndarray, budget: ShotBudget, shot_seed: int) -> np.ndarray:
    """Gaussian-limit shot noise on an exact feature matrix.

    Draws from the dedicated `shot_noise` seed stream so readout noise is
    reproducible and independent of the reservoir, input, split, null,
    projection and bootstrap streams.
    """
    F = np.asarray(features, dtype=float)
    if budget.noiseless:
        return F.copy()
    rng = np.random.default_rng(int(shot_seed))
    return F + rng.normal(0.0, 1.0, F.shape) * shot_std(F, budget.shots)


def sample_shot_noise_exact(features: np.ndarray, budget: ShotBudget, shot_seed: int) -> np.ndarray:
    """True binomial sampling: (2B - S)/S with B ~ Binomial(S, (1+p)/2).

    Used only to validate `add_shot_noise`; too slow for production runs.
    """
    F = np.asarray(features, dtype=float)
    if budget.noiseless:
        return F.copy()
    rng = np.random.default_rng(int(shot_seed))
    p_up = np.clip((1.0 + F) / 2.0, 0.0, 1.0)
    counts = rng.binomial(budget.shots, p_up)
    return (2.0 * counts - budget.shots) / float(budget.shots)


def verify_shot_noise_model(p_values=(-0.9, -0.4, 0.0, 0.3, 0.75, 0.99),
                             shots: int = 2000, n_rep: int = 40000,
                             seed: int = 12345) -> dict:
    """Check the Gaussian model reproduces the binomial mean and variance.

    Returns per-p relative deviations; the test asserts they are small. This
    is what makes the analytic shot model a validated approximation rather
    than an assumption.
    """
    budget = ShotBudget(shots=shots)
    rows = {}
    for i, p in enumerate(p_values):
        base = np.full((n_rep, 1), float(p))
        approx = add_shot_noise(base, budget, shot_seed=seed + i)
        exact = sample_shot_noise_exact(base, budget, shot_seed=seed + 1000 + i)
        var_theory = (1.0 - p ** 2) / shots
        rows[float(p)] = {
            "mean_approx": float(approx.mean()), "mean_exact": float(exact.mean()),
            "var_approx": float(approx.var()), "var_exact": float(exact.var()),
            "var_theory": float(var_theory),
            "mean_abs_diff": float(abs(approx.mean() - exact.mean())),
            "var_rel_diff": float(abs(approx.var() - exact.var()) / max(var_theory, 1e-300)),
        }
    return rows


# =============================================================================
# Matched-resource accounting
# =============================================================================
def account_resources(*, name: str, memory_qubits: int, processor_qubits: int,
                      input_copies: int, ancillas: int, labels_M, labels_P,
                      budget: ShotBudget, feature_rank: int = None,
                      circuit_depth: int = None, two_body_gates: int = None,
                      four_body_gates: int = None, sim_seconds: float = None) -> dict:
    """One row of the matched-resource table.

    Every quantity the Pareto/retention comparisons must match or normalise is
    counted here, including the shot cost implied by the number of
    non-commuting measurement settings -- not just the nominal `shots`.
    """
    labels_M = list(labels_M or [])
    labels_P = list(labels_P or [])
    all_labels = labels_M + labels_P
    n_feat = len(all_labels)
    settings_M = len(group_settings(labels_M)) if labels_M else 0
    settings_P = len(group_settings(labels_P)) if labels_P else 0
    shots_step = ((settings_M + settings_P) * budget.shots) if not budget.noiseless else 0
    return {"name": name,
            "memory_qubits": int(memory_qubits), "processor_qubits": int(processor_qubits),
            "total_qubits": int(memory_qubits + processor_qubits + ancillas),
            "input_copies": int(input_copies), "ancillas": int(ancillas),
            "observables": n_feat, "observables_M": len(labels_M), "observables_P": len(labels_P),
            "feature_rank": (int(feature_rank) if feature_rank is not None else None),
            "measurement_settings": settings_M + settings_P,
            "shots_per_setting": budget.shots,
            "shots_per_timestep": int(shots_step),
            "circuit_depth": circuit_depth,
            "two_body_gates": two_body_gates, "four_body_gates": four_body_gates,
            "readout_parameters": int(n_feat + 1),     # ridge weights + intercept
            "sim_seconds": sim_seconds}


def resources_match(rows, keys=("input_copies", "observables", "shots_per_timestep"),
                    rtol: float = 0.0) -> dict:
    """Are the listed resource keys equal (or within rtol) across rows?

    Used to justify -- or to refuse -- a matched-resource comparison. A
    retention or Pareto claim from unmatched rows is not reported.
    """
    rows = list(rows)
    report = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        if any(v is None for v in vals):
            report[k] = {"values": vals, "matched": False, "reason": "missing value"}
            continue
        lo, hi = min(vals), max(vals)
        ok = (hi - lo) <= rtol * max(abs(hi), 1e-300)
        report[k] = {"values": vals, "matched": bool(ok), "spread": float(hi - lo)}
    report["all_matched"] = all(v.get("matched", False) for k, v in report.items() if k != "all_matched")
    return report
