import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_controls import _best_null_capacity


def test_independent_null_does_not_receive_an_analytical_quantum_feature():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(500, 12))
    labels = rng.normal(size=500)
    score = _best_null_capacity(matrix, labels, np.arange(300), np.arange(350, 500))
    assert score < 0.1
