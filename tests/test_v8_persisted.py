import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from decoupled_qrc.v8_metrics import evaluate_point
from decoupled_qrc.v8_protocol import load_protocol


def _assert_nested_close(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_close(first, second)
    elif isinstance(left, (float, int)):
        assert np.isclose(left, right, rtol=0.0, atol=1e-12)
    else:
        assert left == right


def test_persisted_hh_metrics_recompute_from_raw_qiskit_rows():
    directory = ROOT / "results/v8/development_exact"
    raw = json.loads((directory / "seed_300000_m1_g1_exact.json").read_text())
    saved = json.loads((directory / "seed_300000_m1_g1_exact_metrics.json").read_text())
    assert raw["candidate"] == "V8.6"
    assert raw["backend"]["name"] == "AerSimulator"
    recomputed = evaluate_point(raw["inputs"], raw["features"], load_protocol())
    _assert_nested_close(recomputed, saved)
