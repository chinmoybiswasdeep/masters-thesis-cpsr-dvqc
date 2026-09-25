"""Downstream null controls recomputed from persisted Qiskit feature matrices."""

from __future__ import annotations

import numpy as np

from .v8_metrics import READOUTS, _capacity, _fit_predict, feature_matrix, linear_cka, target


def _splits(protocol):
    data = protocol["data"]
    train = np.arange(data["washout"], data["washout"] + data["train"])
    start = data["washout"] + data["train"] + data["gap"]
    return train, np.arange(start, start + data["test"])


def _best_null_capacity(matrix, labels, train, test):
    return max(
        _capacity(labels[test], _fit_predict(matrix[train], labels[train], matrix[test], readout))
        for readout in READOUTS
    )


def evaluate_negative_controls(inputs, corner_rows, protocol, *, permuted_input_rows, seed=8800):
    """Evaluate registered nulls; ``corner_rows`` contains LL/HL/LH/HH rows."""
    values = np.asarray(inputs, dtype=float)
    train, test = _splits(protocol)
    names, hh = feature_matrix(corner_rows["HH"])
    matrices = {label: feature_matrix(rows)[1] for label, rows in corner_rows.items()}
    rng = np.random.default_rng(seed)

    future = np.empty_like(values)
    future[:-1] = values[1:]
    future[-1] = rng.uniform(-1.0, 1.0)
    random_labels = rng.uniform(-1.0, 1.0, len(values))
    destroyed = hh[rng.permutation(len(hh))]
    _, permuted_input_matrix = feature_matrix(permuted_input_rows)
    random_features = rng.normal(size=hh.shape)

    combined = protocol["task_families"][4:]
    no_communication_columns = [
        index for index, name in enumerate(names) if name.startswith(("M:", "N:"))
    ]
    measurement_only_columns = [index for index, name in enumerate(names) if name == "N:p1"]

    def worst_combined(matrix, columns=None):
        use = matrix if columns is None else matrix[:, columns]
        return max(
            _best_null_capacity(use, target(values, family, delay), train, test)
            for family in combined for delay in range(1, 13)
        )

    controls = {
        "future": _best_null_capacity(hh, future, train, test),
        "random_labels": _best_null_capacity(hh, random_labels, train, test),
        "time_permuted_inputs": _best_null_capacity(permuted_input_matrix, values, train, test),
        "destroyed_temporal_order": _best_null_capacity(destroyed, values, train, test),
        "memory_reset_each_timestep": worst_combined(matrices["LH"]),
        "processor_interaction_disabled": worst_combined(matrices["HL"]),
        "m_disconnected": worst_combined(matrices["LH"]),
        "g_disconnected": worst_combined(matrices["HL"]),
        "memory_processor_no_communication": worst_combined(hh, no_communication_columns),
        "measurement_only_classical": worst_combined(hh, measurement_only_columns),
        "feature_count_matched_random": _best_null_capacity(random_features, values, train, test),
        "amplitude_rescaled_subspace_error": abs(linear_cka(hh, 2.5 * hh) - 1.0),
    }
    return {name: float(value) for name, value in controls.items()}
