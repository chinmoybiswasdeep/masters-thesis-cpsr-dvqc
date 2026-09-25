"""Downstream-only targets, readouts, capacities and geometry for V8."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval
from sklearn.linear_model import RidgeCV


READOUTS = ("raw_ols", "standardized_ols", "ridge_cv", "whitened_ols")
COMBINED_FAMILIES = (
    "current_quadratic_x_delayed_linear",
    "current_cubic_x_delayed_linear",
    "current_quartic_x_delayed_linear",
    "delayed_linear_x_delayed_linear",
    "delayed_nonlinear_x_delayed_linear",
)


def _legendre(values, degree):
    coefficients = np.zeros(degree + 1)
    coefficients[degree] = 1.0
    return legval(values, coefficients)


def _capacity(y_true, y_pred):
    variance = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if variance <= 1e-15 else float(1.0 - np.sum((y_true - y_pred) ** 2) / variance)


def _fit_predict(x_train, y_train, x_test, readout):
    if readout == "raw_ols":
        design = np.column_stack((np.ones(len(x_train)), x_train))
        weights = np.linalg.lstsq(design, y_train, rcond=1e-12)[0]
        return np.column_stack((np.ones(len(x_test)), x_test)) @ weights
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale < 1e-12] = 1.0
    train, test = (x_train - mean) / scale, (x_test - mean) / scale
    if readout == "standardized_ols":
        weights = np.linalg.lstsq(np.column_stack((np.ones(len(train)), train)), y_train, rcond=1e-12)[0]
        return np.column_stack((np.ones(len(test)), test)) @ weights
    if readout == "ridge_cv":
        model = RidgeCV(alphas=np.logspace(-8, 2, 16), fit_intercept=True).fit(train, y_train)
        return model.predict(test)
    if readout == "whitened_ols":
        u, singular, vt = np.linalg.svd(train, full_matrices=False)
        keep = singular > max(1e-10, singular[0] * 1e-10) if singular.size else np.zeros(0, dtype=bool)
        whiten_train = u[:, keep]
        whiten_test = test @ vt[keep].T / singular[keep] if keep.any() else np.empty((len(test), 0))
        weights = np.linalg.lstsq(
            np.column_stack((np.ones(len(whiten_train)), whiten_train)), y_train, rcond=1e-12
        )[0]
        return np.column_stack((np.ones(len(whiten_test)), whiten_test)) @ weights
    raise ValueError(f"unknown readout {readout}")


def feature_matrix(rows):
    names = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != names for row in rows):
        raise ValueError("feature rows have inconsistent columns")
    return names, np.asarray([[row[name] for name in names] for row in rows], dtype=float)


def _columns(names, prefixes):
    indices = [index for index, name in enumerate(names) if any(name.startswith(prefix) for prefix in prefixes)]
    if not indices:
        raise ValueError(f"no features for {prefixes}")
    return indices


def target(inputs, family, delay):
    values = np.asarray(inputs, dtype=float)
    delayed = np.roll(values, delay)
    if family == "linear_delayed":
        return delayed
    if family == "quadratic_delayed":
        return _legendre(delayed, 2)
    if family == "cubic_delayed":
        return _legendre(delayed, 3)
    if family == "quartic_delayed":
        return _legendre(delayed, 4)
    if family.startswith("current_"):
        degree = {"quadratic": 2, "cubic": 3, "quartic": 4}[family.split("_")[1]]
        return _legendre(values, degree) * delayed
    if family == "delayed_linear_x_delayed_linear":
        companion = 1 if delay != 1 else 2
        return delayed * np.roll(values, companion)
    if family == "delayed_nonlinear_x_delayed_linear":
        return _legendre(delayed, 2) * delayed
    raise ValueError(f"unknown target family {family}")


def family_columns(names, family):
    if family == "linear_delayed":
        return _columns(names, ("M:",))
    if family == "quadratic_delayed":
        return _columns(names, ("J:delay_p2:",))
    if family == "cubic_delayed":
        return _columns(names, ("M:", "J:delay_p3:"))
    if family == "quartic_delayed":
        return _columns(names, ("J:delay_p2:", "J:delay_p4:"))
    if family == "current_quadratic_x_delayed_linear":
        return _columns(names, ("M:", "J:current_p2_"))
    if family == "current_cubic_x_delayed_linear":
        return _columns(names, ("J:current_p1_", "J:current_p3_"))
    if family == "current_quartic_x_delayed_linear":
        return _columns(names, ("M:", "J:current_p2_", "J:current_p4_"))
    if family == "delayed_linear_x_delayed_linear":
        return _columns(names, ("J:linear_pair:",))
    if family == "delayed_nonlinear_x_delayed_linear":
        return _columns(names, ("M:", "J:delay_p3:"))
    raise ValueError(f"unknown family {family}")


def evaluate_point(inputs, rows, protocol):
    names, matrix = feature_matrix(rows)
    data = protocol["data"]
    train = np.arange(data["washout"], data["washout"] + data["train"])
    test_start = data["washout"] + data["train"] + data["gap"]
    test = np.arange(test_start, test_start + data["test"])
    if np.intersect1d(train, test).size or max(test) >= len(inputs):
        raise ValueError("invalid train/test split")
    capacities = {readout: {} for readout in READOUTS}
    for family in protocol["task_families"]:
        columns = family_columns(names, family)
        for delay in range(1, 13):
            y = target(inputs, family, delay)
            for readout in READOUTS:
                prediction = _fit_predict(matrix[train][:, columns], y[train], matrix[test][:, columns], readout)
                capacities[readout].setdefault(family, []).append(_capacity(y[test], prediction))
    nonlinear_columns = _columns(names, ("N:",))
    nonlinear = {readout: [] for readout in READOUTS}
    for degree in (2, 3, 4):
        y = _legendre(np.asarray(inputs), degree)
        for readout in READOUTS:
            prediction = _fit_predict(
                matrix[train][:, nonlinear_columns], y[train], matrix[test][:, nonlinear_columns], readout
            )
            nonlinear[readout].append(_capacity(y[test], prediction))
    held_out = {readout: {"delay_pairs": [], "nonlinear_combinations": []} for readout in READOUTS}
    pair_columns = _columns(names, ("J:linear_pair:",))
    for left, right in protocol["data"]["held_out_pairs"]:
        y = np.roll(np.asarray(inputs), left) * np.roll(np.asarray(inputs), right)
        for readout in READOUTS:
            prediction = _fit_predict(matrix[train][:, pair_columns], y[train], matrix[test][:, pair_columns], readout)
            held_out[readout]["delay_pairs"].append(_capacity(y[test], prediction))
    combinations = (
        (2, 3, 7, ("M:", "J:delay_p3:", "J:current_p2_", "J:mix_p2_p3:")),
        (3, 2, 10, ("N:p1", "N:p3", "J:mix_p1_p2:", "J:mix_p3_p2:")),
    )
    values = np.asarray(inputs)
    for current_degree, delayed_degree, delay, prefixes in combinations:
        y = _legendre(values, current_degree) * _legendre(np.roll(values, delay), delayed_degree)
        columns = _columns(names, prefixes)
        for readout in READOUTS:
            prediction = _fit_predict(matrix[train][:, columns], y[train], matrix[test][:, columns], readout)
            held_out[readout]["nonlinear_combinations"].append(_capacity(y[test], prediction))
    return {
        "capacities": capacities,
        "nonlinear_profile": nonlinear,
        "held_out": held_out,
        "split": {"train": train.tolist(), "test": test.tolist(), "disjoint": True},
        "feature_names": names,
    }


def linear_cka(left, right):
    left = np.asarray(left, dtype=float) - np.mean(left, axis=0)
    right = np.asarray(right, dtype=float) - np.mean(right, axis=0)
    numerator = np.linalg.norm(left.T @ right, "fro") ** 2
    denominator = np.linalg.norm(left.T @ left, "fro") * np.linalg.norm(right.T @ right, "fro")
    return float(numerator / denominator) if denominator else 0.0


def normalized_gram(matrix):
    """Centered, unit-Frobenius Gram matrix used for scale-invariant audits."""
    centered = np.asarray(matrix, dtype=float) - np.mean(matrix, axis=0)
    gram = centered @ centered.T
    norm = np.linalg.norm(gram, "fro")
    return gram / norm if norm else np.zeros_like(gram)


def principal_angles(left, right, tolerance=1e-10):
    """Principal angles in radians between centered numerical column spaces."""
    bases = []
    for matrix in (left, right):
        centered = np.asarray(matrix, dtype=float) - np.mean(matrix, axis=0)
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        if not singular.size:
            bases.append(np.empty((centered.shape[0], 0)))
            continue
        keep = singular > max(tolerance, tolerance * singular[0])
        bases.append(u[:, keep])
    if not bases[0].shape[1] or not bases[1].shape[1]:
        return []
    cosines = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
    return np.arccos(np.clip(cosines, -1.0, 1.0)).tolist()


def geometry(matrix):
    centered = np.asarray(matrix, dtype=float) - np.mean(matrix, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    effective_rank = float(np.exp(-np.sum((energy / energy.sum()) * np.log(energy / energy.sum() + 1e-300)))) if energy.sum() else 0.0
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 1e-15 else float("inf")
    return {
        "singular_values": singular.tolist(),
        "effective_rank": effective_rank,
        "condition_number": condition,
        "normalized_gram_frobenius": float(np.linalg.norm(normalized_gram(matrix), "fro")),
    }
