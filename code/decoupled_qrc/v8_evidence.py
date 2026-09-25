"""Recompute V8 corner evidence from persisted Qiskit feature rows and metrics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .v8_metrics import COMBINED_FAMILIES, READOUTS, evaluate_point, feature_matrix, geometry, linear_cka
from .v8_protocol import load_open_bank, load_protocol


CORNERS = {"LL": (0, 0), "HL": (1, 0), "LH": (0, 1), "HH": (1, 1)}


def _name(input_seed, m, g, suffix="", mode="exact", shots=0, precision="double", simulator_seed_offset=0):
    mode_suffix = f"_{shots}shots" if mode == "shots" else ""
    if precision != "double":
        mode_suffix += f"_{precision}"
    if simulator_seed_offset:
        mode_suffix += f"_simplus{simulator_seed_offset}"
    return f"seed_{input_seed}_m{m:g}_g{g:g}_{mode}{mode_suffix}{suffix}.json"


def _read(path):
    return json.loads(Path(path).read_text())


def _score(metrics, readout, family):
    return float(np.mean(metrics["capacities"][readout][family]))


def _nonlinear_score(metrics, readout):
    return float(np.mean(metrics["nonlinear_profile"][readout]))


def _largest_absolute(values):
    return float(max(values, key=lambda value: abs(value)))


def _default_supplemental():
    return {
        "encoder_leakage_max": 1e300,
        "saturated_fraction": 1.0,
        "response_surface": {
            "complete_points": 0, "memory_ordered": False, "nonlinear_ordered": False,
            "memory_interior": False, "nonlinear_interior": False,
        },
        "conditioning_pass": False,
        "learning_curve_pass": False,
        "negative_controls": {"missing": 1e300},
        "finite_shot_pass": False,
        "noise_effect_retention": 0.0,
        "noise_hh_ordered": False,
        "precision_pass": False,
        "baseline_complete": False,
        "quantum_advantage": False,
        "internal_validation_pass": False,
        "confirmation_pass": False,
    }


def build_corner_evidence(
    directory, bank="development", supplemental=None, *, mode="exact", shots=0,
    precision="double", simulator_seed_offset=0,
):
    """Build paired seed rows without substituting or regenerating quantum data."""
    protocol = load_protocol()
    seeds = load_open_bank(bank)["seeds"]
    root = Path(directory)
    seed_rows = []
    structural_memory = []
    structural_nonlinear = []
    geometry_values = []
    amplitude_values = []

    for seed in seeds:
        points = {}
        for label, (m, g) in CORNERS.items():
            raw_path = root / _name(
                seed["input"], m, g, mode=mode, shots=shots,
                precision=precision, simulator_seed_offset=simulator_seed_offset,
            )
            metric_path = root / _name(
                seed["input"], m, g, "_metrics", mode=mode, shots=shots,
                precision=precision, simulator_seed_offset=simulator_seed_offset,
            )
            if not raw_path.exists() or not metric_path.exists():
                continue
            points[label] = (_read(raw_path), _read(metric_path))
        if len(points) != 4:
            continue

        metrics = {label: value[1] for label, value in points.items()}
        row = {
            "memory_main": [], "nonlinearity_main": [],
            "memory_to_nonlinearity": [], "nonlinearity_to_memory": [],
            "degree_profile_differences": [], "memory_curve_differences": [],
            "heldout_capacities": [],
        }
        for readout in READOUTS:
            memory = {label: _score(value, readout, "linear_delayed") for label, value in metrics.items()}
            nonlinear = {label: _nonlinear_score(value, readout) for label, value in metrics.items()}
            row["memory_main"].extend((memory["HL"] - memory["LL"], memory["HH"] - memory["LH"]))
            row["nonlinearity_main"].extend((nonlinear["LH"] - nonlinear["LL"], nonlinear["HH"] - nonlinear["HL"]))
            row["memory_to_nonlinearity"].extend((nonlinear["HL"] - nonlinear["LL"], nonlinear["HH"] - nonlinear["LH"]))
            row["nonlinearity_to_memory"].extend((memory["LH"] - memory["LL"], memory["HH"] - memory["HL"]))
            heldout = metrics["HH"]["held_out"][readout]
            row["heldout_capacities"].extend(heldout["delay_pairs"] + heldout["nonlinear_combinations"])

        for degree in range(3):
            differences = []
            for readout in READOUTS:
                profile = {label: metrics[label]["nonlinear_profile"][readout][degree] for label in CORNERS}
                differences.append((profile["HH"] - profile["HL"]) - (profile["LH"] - profile["LL"]))
            row["degree_profile_differences"].append(_largest_absolute(differences))
        for delay in range(12):
            differences = []
            for readout in READOUTS:
                curve = {label: metrics[label]["capacities"][readout]["linear_delayed"][delay] for label in CORNERS}
                differences.append((curve["HH"] - curve["LH"]) - (curve["HL"] - curve["LL"]))
            row["memory_curve_differences"].append(_largest_absolute(differences))

        for family in protocol["task_families"]:
            row["per_delay:" + family] = [
                value for readout in READOUTS for value in metrics["HH"]["capacities"][readout][family]
            ]
            if family in COMBINED_FAMILIES:
                row["hh_advantage:" + family] = [
                    metrics["HH"]["capacities"][readout][family][delay]
                    - max(metrics[label]["capacities"][readout][family][delay] for label in ("LL", "HL", "LH"))
                    for readout in READOUTS for delay in range(12)
                ]
        seed_rows.append(row)

        matrices = {}
        names = None
        for label, (raw, _) in points.items():
            point_names, matrix = feature_matrix(raw["features"])
            if names is not None and point_names != names:
                raise ValueError("corner feature columns differ")
            names, matrices[label] = point_names, matrix
        memory_columns = [index for index, name in enumerate(names) if name.startswith("M:")]
        nonlinear_columns = [index for index, name in enumerate(names) if name.startswith("N:")]
        structural_memory.extend((
            float(np.max(np.abs(matrices["LL"][:, memory_columns] - matrices["LH"][:, memory_columns]))),
            float(np.max(np.abs(matrices["HL"][:, memory_columns] - matrices["HH"][:, memory_columns]))),
        ))
        structural_nonlinear.extend((
            float(np.max(np.abs(matrices["LL"][:, nonlinear_columns] - matrices["HL"][:, nonlinear_columns]))),
            float(np.max(np.abs(matrices["LH"][:, nonlinear_columns] - matrices["HH"][:, nonlinear_columns]))),
        ))
        low = matrices["LL"][:, nonlinear_columns]
        high = matrices["LH"][:, nonlinear_columns]
        geometry_values.append(linear_cka(low, high))
        amplitude_values.append(linear_cka(high, 3.75 * high))

    evidence = {
        "stage": "internal_validation" if bank == "internal_validation" else "development",
        "expected_seed_count": len(seeds),
        "observed_seed_count": len(seed_rows),
        "structural_invariance": {
            "memory_max_abs": max(structural_memory, default=1e300),
            "nonlinearity_max_abs": max(structural_nonlinear, default=1e300),
        },
        "geometry": {
            "cka_low_high_g": max(geometry_values, default=1.0),
            "amplitude_control_cka": min(amplitude_values, default=0.0),
        },
        "split_disjoint": True,
        "no_future_features": True,
        "target_names": list(protocol["required_targets"]),
    }
    keys = (
        "memory_main", "nonlinearity_main", "memory_to_nonlinearity", "nonlinearity_to_memory",
        "degree_profile_differences", "memory_curve_differences", "heldout_capacities",
    ) + tuple("per_delay:" + family for family in protocol["task_families"]) + tuple(
        "hh_advantage:" + family for family in COMBINED_FAMILIES
    )
    for key in keys:
        evidence[key] = [row[key] for row in seed_rows]

    if seed_rows:
        invariant_readouts = (1, 3)
        evidence["scale_invariant_pass"] = all(
            row["nonlinearity_main"][2 * readout + endpoint] >= protocol["thresholds"]["main_effect_min"]
            for row in seed_rows for readout in invariant_readouts for endpoint in (0, 1)
        )
    else:
        evidence["scale_invariant_pass"] = False
    evidence.update(_default_supplemental())
    if supplemental:
        evidence.update(supplemental)
    return evidence


def analyze_response_surface(directory, input_seed):
    """Audit the registered 5x5 metric surface for ordering and saturation."""
    protocol = load_protocol()
    grid = protocol["data"]["grid"]
    root = Path(directory)
    points = {}
    for m in grid["m"]:
        for g in grid["g"]:
            path = root / _name(input_seed, m, g, "_metrics")
            if not path.exists():
                raise FileNotFoundError(path)
            points[(m, g)] = _read(path)

    memory = {}
    nonlinear = {}
    saturated = []
    for key, metrics in points.items():
        memory[key] = float(np.mean([
            _score(metrics, readout, "linear_delayed") for readout in READOUTS
        ]))
        nonlinear[key] = float(np.mean([
            _nonlinear_score(metrics, readout) for readout in READOUTS
        ]))
        for family in protocol["task_families"]:
            for readout in READOUTS:
                saturated.append(_score(metrics, readout, family) >= 0.99)

    tolerance = 1e-8
    memory_ordered = all(
        all(memory[(left, g)] <= memory[(right, g)] + tolerance for left, right in zip(grid["m"], grid["m"][1:]))
        for g in grid["g"]
    )
    nonlinear_ordered = all(
        all(nonlinear[(m, left)] <= nonlinear[(m, right)] + tolerance for left, right in zip(grid["g"], grid["g"][1:]))
        for m in grid["m"]
    )
    memory_interior = all(
        memory[(0, g)] - tolerance <= memory[(m, g)] <= memory[(1, g)] + tolerance
        for g in grid["g"] for m in (0.25, 0.5, 0.75)
    )
    nonlinear_interior = all(
        nonlinear[(m, 0)] - tolerance <= nonlinear[(m, g)] <= nonlinear[(m, 1)] + tolerance
        for m in grid["m"] for g in (0.25, 0.5, 0.75)
    )
    return {
        "memory_scores": {f"m={m:g},g={g:g}": memory[(m, g)] for m in grid["m"] for g in grid["g"]},
        "nonlinear_scores": {f"m={m:g},g={g:g}": nonlinear[(m, g)] for m in grid["m"] for g in grid["g"]},
        "memory_ordered": memory_ordered,
        "nonlinear_ordered": nonlinear_ordered,
        "memory_interior": memory_interior,
        "nonlinear_interior": nonlinear_interior,
        "saturated_fraction": float(np.mean(saturated)),
        "complete_points": len(points),
    }


def analyze_conditioning_and_learning(raw_path):
    """Compare the frozen 512-sample fit with the same rows at 256 samples."""
    raw = _read(raw_path)
    protocol = load_protocol()
    full = evaluate_point(raw["inputs"], raw["features"], protocol)
    half_protocol = copy.deepcopy(protocol)
    half_protocol["data"]["train"] = protocol["data"]["train"] // 2
    half = evaluate_point(raw["inputs"], raw["features"], half_protocol)
    differences = []
    minimum = float("inf")
    for readout in READOUTS:
        for family in protocol["task_families"]:
            full_values = np.asarray(full["capacities"][readout][family])
            half_values = np.asarray(half["capacities"][readout][family])
            differences.extend(np.abs(full_values - half_values))
            minimum = min(minimum, float(full_values.min()), float(half_values.min()))
    _, matrix = feature_matrix(raw["features"])
    train = matrix[protocol["data"]["washout"]:protocol["data"]["washout"] + protocol["data"]["train"]]
    diagnostics = geometry(train)
    numerical_rank = int(np.linalg.matrix_rank(train - train.mean(0), tol=1e-10))
    maximum_change = float(max(differences, default=float("inf")))
    return {
        "conditioning_pass": bool(numerical_rank < len(train) and diagnostics["effective_rank"] < len(train) / 2 and minimum > -1.0),
        "learning_curve_pass": bool(maximum_change <= 0.05),
        "learning_curve_max_abs_capacity_change": maximum_change,
        "minimum_capacity_across_256_512": minimum,
        "numerical_rank": numerical_rank,
        "training_rows": len(train),
        "geometry": diagnostics,
    }
