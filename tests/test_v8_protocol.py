import json
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from decoupled_qrc.v8_confirmation import reveal_confirmation
import decoupled_qrc.v8_freeze as freeze_module
from decoupled_qrc.v8_freeze import freeze_candidate, verify_manifest
from decoupled_qrc.v8_protocol import ROOT, assert_disjoint, load_open_bank, load_protocol


def test_open_banks_are_disjoint_and_confirmation_is_encrypted():
    development = load_open_bank("development")
    validation = load_open_bank("internal_validation")
    assert_disjoint(development, validation)
    encrypted = ROOT / load_protocol()["seed_banks"]["confirmation"]["ciphertext"]
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        json.loads(encrypted.read_bytes())
    with pytest.raises(PermissionError):
        load_open_bank("confirmation")


def test_confirmation_requires_valid_freeze(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"status":"development"}')
    with pytest.raises(RuntimeError):
        reveal_confirmation(bad, Fernet.generate_key(), tmp_path / "reveal.json")


def test_manifest_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze_module, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    manifest = tmp_path / "manifest.json"
    freeze_candidate("V8.test", ["source.py"], "prereg", str(manifest))
    verify_manifest(manifest)
    source.write_text("x = 2\n")
    with pytest.raises(RuntimeError):
        verify_manifest(manifest)
