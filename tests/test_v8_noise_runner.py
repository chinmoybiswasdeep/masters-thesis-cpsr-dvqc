import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from decoupled_qrc.v8_noise_runner import FAKE_BACKEND_NAME


def test_noise_backend_is_frozen_and_not_a_density_matrix_expansion():
    source = (Path(__file__).resolve().parents[1] / "code/decoupled_qrc/v8_noise_runner.py").read_text()
    assert FAKE_BACKEND_NAME == "FakeGuadalupeV2"
    assert "NoiseModel.from_backend" in source
    assert 'method="matrix_product_state"' in source
    assert "density_matrix" not in source
