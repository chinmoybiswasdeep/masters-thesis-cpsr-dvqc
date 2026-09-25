"""Frozen classical comparisons for V8 claim discipline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .v8_metrics import _capacity, _fit_predict, feature_matrix, target


def _legendre(values, degree):
    coefficients = np.zeros(degree + 1)
    coefficients[degree] = 1.0
    return np.polynomial.legendre.legval(values, coefficients)


def _splits(protocol):
    data = protocol["data"]
    train = np.arange(data["washout"], data["washout"] + data["train"])
    start = data["washout"] + data["train"] + data["gap"]
    return train, np.arange(start, start + data["test"])


def _reservoir(values, width, rng, recurrent=True):
    input_weights = rng.normal(scale=0.35, size=width)
    bias = rng.normal(scale=0.1, size=width)
    state = np.zeros(width)
    matrix = np.empty((len(values), width))
    if recurrent:
        weights = rng.normal(size=(width, width))
        radius = max(abs(np.linalg.eigvals(weights)))
        weights *= 0.85 / radius
    for index, value in enumerate(values):
        state = np.tanh(input_weights * value + bias + (weights @ state if recurrent else 0.0))
        matrix[index] = state
    return matrix


def evaluate_classical_baselines(inputs, quantum_rows, protocol, *, seed=8700):
    values = np.asarray(inputs, dtype=float)
    train, test = _splits(protocol)
    rng = np.random.default_rng(seed)
    lags = np.column_stack([np.roll(values, delay) for delay in range(13)])
    linear_ar = lags[:, 1:]
    univariate = np.column_stack([_legendre(lags[:, delay], degree) for delay in range(13) for degree in range(1, 5)])
    pairs = np.column_stack([lags[:, left] * lags[:, right] for left in range(13) for right in range(left + 1, 13)])
    volterra = np.column_stack((univariate, pairs))
    random_features = np.tanh(lags @ rng.normal(scale=0.5, size=(13, 201)) + rng.normal(size=201))
    echo_state = _reservoir(values, 201, rng, recurrent=True)
    parameter_matched = _reservoir(values, 23, rng, recurrent=True)
    names, quantum = feature_matrix(quantum_rows)
    local_columns = [index for index, name in enumerate(names) if name.startswith(("M:", "N:"))]
    tractable = quantum[:, local_columns]
    feature_sets = {
        "linear_autoregression": linear_ar,
        "polynomial_volterra": volterra,
        "echo_state_network": echo_state,
        "feature_count_matched_random": random_features,
        "parameter_count_matched_classical_reservoir": parameter_matched,
        "classically_tractable_circuit_variant": tractable,
    }
    capacities = {}
    for baseline, matrix in feature_sets.items():
        capacities[baseline] = {}
        for family in protocol["task_families"]:
            capacities[baseline][family] = []
            for delay in range(1, 13):
                labels = target(values, family, delay)
                prediction = _fit_predict(matrix[train], labels[train], matrix[test], "ridge_cv")
                capacities[baseline][family].append(_capacity(labels[test], prediction))
    root = Path(__file__).resolve().parents[2]
    historical = {
        "V6.7": {"artifact": "results/v6/algebra/V6.7.json", "available": (root / "results/v6/algebra/V6.7.json").exists()},
        "V7.8": {"artifact": "results/v7/V7.8", "available": (root / "results/v7/V7.8").exists(), "status": "exploratory failure"},
    }
    return {
        "capacities": capacities,
        "feature_counts": {name: int(matrix.shape[1]) for name, matrix in feature_sets.items()},
        "historical": historical,
        "baseline_complete": all(item["available"] for item in historical.values()),
        "quantum_advantage": False,
        "quantum_advantage_reason": "Not claimed: the target-aligned polynomial/Volterra baseline is expected to be competitive.",
    }
