"""Downstream statistics for already-persisted Aer feature matrices."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .v7_measurements import (
    ALL_FEATURES, CURRENT_MEMORY_FEATURES, INTERACTION_FEATURES,
    MEMORY_FEATURES, PROCESSOR_FEATURES,
)


def legendre(degree: int, value):
    if degree == 1:
        return value
    if degree == 2:
        return (3 * value**2 - 1) / 2
    if degree == 3:
        return (5 * value**3 - 3 * value) / 2
    if degree == 4:
        return (35 * value**4 - 30 * value**2 + 3) / 8
    raise ValueError("supported Legendre degrees are 1..4")


def feature_matrix(rows: list[dict] | tuple[dict, ...], names=ALL_FEATURES) -> np.ndarray:
    return np.asarray([[row[name] for name in names] for row in rows], dtype=float)


def _indices(length: int, protocol: dict) -> tuple[np.ndarray, np.ndarray]:
    data = protocol["data"]
    train_start = data["washout"]
    train_stop = train_start + data["train"]
    test_start = train_stop + data["gap"]
    test_stop = test_start + data["test"]
    if test_stop > length:
        raise ValueError("feature sequence is shorter than the frozen split")
    return np.arange(train_start, train_stop), np.arange(test_start, test_stop)


def _whiten(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    centred = train - mean
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    keep = singular > singular.max(initial=0.0) * 1e-10
    if not keep.any():
        return np.ones((len(train), 1)), np.ones((len(test), 1))
    basis = right[keep].T / singular[keep]
    return centred @ basis, (test - mean) @ basis


def capacity(features: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray, readout: str) -> float:
    x_train, x_test = features[train], features[test]
    y_train, y_test = target[train], target[test]
    if readout == "raw_ols":
        model = LinearRegression()
    elif readout == "standardized_ols":
        model = make_pipeline(StandardScaler(), LinearRegression())
    elif readout == "ridge_cv":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=(1e-6, 1e-4, 1e-2, 1.0, 100.0)))
    elif readout == "whitened_ols":
        x_train, x_test = _whiten(x_train, x_test)
        model = LinearRegression()
    else:
        raise ValueError(readout)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    denominator = np.sum((y_test - y_train.mean()) ** 2)
    return float(1.0 - np.sum((y_test - prediction) ** 2) / denominator) if denominator else 0.0


def _delayed(values: np.ndarray, degree: int, delay: int) -> np.ndarray:
    result = np.zeros_like(values)
    result[delay:] = legendre(degree, values[:-delay])
    return result


def evaluate_features(rows, inputs, protocol: dict, readout: str = "standardized_ols") -> dict:
    values = np.asarray(inputs, dtype=float)
    train, test = _indices(len(values), protocol)
    memory = feature_matrix(rows, MEMORY_FEATURES)
    processor = feature_matrix(rows, PROCESSOR_FEATURES)
    nonlinear_memory = {
        2: feature_matrix(rows, ("q2",)),
        3: feature_matrix(rows, ("q3",)),
        4: feature_matrix(rows, ("q4",)),
    }
    current_memory = feature_matrix(rows, MEMORY_FEATURES + CURRENT_MEMORY_FEATURES)
    interaction_memory = feature_matrix(rows, INTERACTION_FEATURES)
    horizon = protocol["data"]["delay_horizon"]

    memory_curve = [capacity(memory, _delayed(values, 1, delay), train, test, readout) for delay in range(1, horizon + 1)]
    degree_profile = [capacity(processor, legendre(degree, values), train, test, readout) for degree in (2, 3, 4)]
    delayed = {
        str(degree): [capacity(nonlinear_memory[degree], _delayed(values, degree, delay), train, test, readout) for delay in range(1, horizon + 1)]
        for degree in (2, 3, 4)
    }
    current_x_delayed = [
        capacity(current_memory, legendre(2, values) * _delayed(values, 1, delay), train, test, readout)
        for delay in range(1, horizon + 1)
    ]
    delayed_x_delayed = [
        capacity(interaction_memory, _delayed(values, 1, delay) * _delayed(values, 1, min(delay + 1, horizon)), train, test, readout)
        for delay in range(1, horizon)
    ]
    return {
        "M": float(sum(memory_curve)),
        "N": float(sum(degree_profile)),
        "memory_curve": memory_curve,
        "degree_profile": degree_profile,
        "delayed_nonlinear": delayed,
        "current_x_delayed": current_x_delayed,
        "delayed_x_delayed": delayed_x_delayed,
        "delayed_cubic": delayed["3"],
        "saturated_fraction": float(np.mean(np.asarray(memory_curve + degree_profile) > protocol["thresholds"]["saturated_capacity"])),
    }


def geometry(rows) -> dict:
    features = feature_matrix(rows)
    features = features - features.mean(axis=0)
    singular = np.linalg.svd(features, compute_uv=False)
    energy = singular**2
    probabilities = energy / energy.sum() if energy.sum() else energy
    effective_rank = float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))))
    norm = np.linalg.norm(features)
    gram = features @ features.T / (norm * norm) if norm else features @ features.T
    return {"effective_rank": effective_rank, "singular_values": singular.tolist(), "normalized_gram": gram.tolist()}


def linear_cka(rows_a, rows_b) -> float:
    a, b = feature_matrix(rows_a), feature_matrix(rows_b)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    cross = np.linalg.norm(a.T @ b, "fro") ** 2
    denominator = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(cross / denominator) if denominator else 1.0


def principal_angles(rows_a, rows_b) -> list[float]:
    a, b = feature_matrix(rows_a), feature_matrix(rows_b)
    qa, _ = np.linalg.qr(a - a.mean(axis=0))
    qb, _ = np.linalg.qr(b - b.mean(axis=0))
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0)).tolist()
