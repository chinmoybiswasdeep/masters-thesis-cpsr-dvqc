import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_metrics import (
    _fit_predict, evaluate_point, linear_cka, normalized_gram, principal_angles,
)
from decoupled_qrc.v8_protocol import load_protocol


def test_split_has_gap_and_no_overlap_with_delay_12():
    protocol = load_protocol()
    data = protocol["data"]
    train_end = data["washout"] + data["train"] - 1
    test_start = data["washout"] + data["train"] + data["gap"]
    assert test_start - train_end - 1 == 12


def test_amplitude_rescaling_has_unit_linear_cka():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(100, 7))
    assert abs(linear_cka(matrix, matrix * 4.25) - 1.0) < 1e-12
    assert np.allclose(normalized_gram(matrix), normalized_gram(matrix * 4.25), atol=1e-12)
    assert max(principal_angles(matrix, matrix * 4.25), default=0.0) < 1e-7


def test_missing_feature_column_is_rejected():
    protocol = copy.deepcopy(load_protocol())
    protocol["data"].update({"washout": 1, "train": 3, "gap": 12, "test": 2})
    inputs = np.linspace(-1, 1, 18)
    rows = [{"M:d1": value} for value in inputs]
    try:
        evaluate_point(inputs, rows, protocol)
    except ValueError as error:
        assert "no features" in str(error)
    else:
        raise AssertionError("missing registered feature groups were accepted")


def test_whitening_drops_roundoff_only_subspace():
    rng = np.random.default_rng(4)
    train = rng.normal(scale=1e-15, size=(32, 6))
    test = rng.normal(scale=1e-15, size=(12, 6))
    target = rng.normal(size=32)
    prediction = _fit_predict(train, target, test, "whitened_ols")
    assert np.allclose(prediction, target.mean())
