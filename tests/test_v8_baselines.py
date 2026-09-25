import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_baselines import _reservoir


def test_classical_reservoir_is_deterministic_and_has_requested_width():
    values = np.linspace(-1, 1, 20)
    left = _reservoir(values, 7, np.random.default_rng(4))
    right = _reservoir(values, 7, np.random.default_rng(4))
    assert left.shape == (20, 7)
    assert np.array_equal(left, right)
