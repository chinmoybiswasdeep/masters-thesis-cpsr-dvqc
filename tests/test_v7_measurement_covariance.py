import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from decoupled_qrc.v7_measurements import setting_from_counts


def test_joint_parities_come_from_same_bitstrings():
    features = setting_from_counts({"0000000": 50, "0010011": 50})
    assert features["m0"] == 0.0
    assert features["m1"] == 0.0
    assert features["m0m1"] == 1.0

