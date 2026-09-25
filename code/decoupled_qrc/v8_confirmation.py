"""Encrypted confirmation-bank reveal, permitted only after a valid freeze."""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .v8_freeze import verify_committed_manifest
from .v8_protocol import RESULTS, atomic_json, load_protocol, sha256_file


def reveal_confirmation(manifest_path, escrow_key: bytes, output=None):
    manifest = verify_committed_manifest(manifest_path)
    encrypted = RESULTS / "confirmation_seeds.fernet"
    commitment = load_protocol()["seed_banks"]["confirmation"]["ciphertext_sha256"]
    if sha256_file(encrypted) != commitment:
        raise RuntimeError("confirmation ciphertext commitment mismatch")
    try:
        bank = json.loads(Fernet(escrow_key).decrypt(encrypted.read_bytes()))
    except InvalidToken as error:
        raise PermissionError("invalid confirmation escrow key") from error
    destination = Path(output or RESULTS / "confirmation_seeds_revealed.json")
    atomic_json(destination, {**bank, "frozen_version": manifest["version"]})
    return bank
