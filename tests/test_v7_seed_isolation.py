import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from decoupled_qrc.v7_protocol import ROOT, assert_seed_disjointness, load_seed_bank


def test_banks_are_disjoint_and_confirmation_is_sealed():
    development = load_seed_bank("development")
    validation = load_seed_bank("internal_validation")
    confirmation = json.loads((ROOT / ".git" / "v7_confirmation_seeds.json").read_text())
    assert_seed_disjointness(development, validation, confirmation)
    with pytest.raises(PermissionError):
        load_seed_bank("confirmation")

