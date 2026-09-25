import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from run_v8_matrix import CORNERS, GRID, indices


def test_matrix_ranges_and_registered_points_are_exact():
    assert list(indices("1-3")) == [1, 2, 3]
    assert list(indices("1,4")) == [1, 4]
    assert CORNERS == ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    assert GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
