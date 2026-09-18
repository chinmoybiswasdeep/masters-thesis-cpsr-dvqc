"""
v3_seeds.py -- V3's immutable, fully-explicit seed object. Every prior
pass had at least one place where a bare `seed=0` carried several
undocumented meanings; V3 forbids that by construction: every randomness
source is a NAMED field, and every saved result row carries all eight.

Discovery and confirmation seeds are disjoint BY CONSTRUCTION (discovery
uses negative `reservoir_idx`, confirmation non-negative), continuing the
convention V2.1 introduced and V2.2 verified.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class NestedSeeds:
    """Eight independent randomness streams. `reservoir_idx`/`input_idx`
    are the human-facing indices; the eight seed integers are SHA256
    derivations from them, so two different (reservoir_idx, input_idx)
    pairs can never collide on any stream."""
    reservoir_idx: int
    input_idx: int
    hamiltonian: int
    disorder: int
    input_sequence: int
    regression_split: int
    null_surrogate: int
    projection: int
    bootstrap: int
    shot_noise: int

    def as_dict(self) -> dict:
        return asdict(self)


_STREAMS = ("hamiltonian", "disorder", "input_sequence", "regression_split",
            "null_surrogate", "projection", "bootstrap", "shot_noise")


def make_seeds(reservoir_idx: int, input_idx: int = 0) -> NestedSeeds:
    """SHA256-derived child seeds, namespaced `dqrc_v3:` so a V3 seed can
    never coincide with a V2/V2.1/V2.2 seed derived from the same indices
    (those used `dqrc:` and `dqrc_v2:` namespaces)."""
    def child(tag: str) -> int:
        h = hashlib.sha256(f"dqrc_v3:{reservoir_idx}:{input_idx}:{tag}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    return NestedSeeds(reservoir_idx=reservoir_idx, input_idx=input_idx,
                        **{s: child(s) for s in _STREAMS})


def discovery_seeds(k: int, input_idx: int = 0) -> NestedSeeds:
    """k-th DISCOVERY seed bundle -- always a negative `reservoir_idx`."""
    return make_seeds(-(k + 1), input_idx)


def confirmation_seeds(k: int, input_idx: int = 0) -> NestedSeeds:
    """k-th CONFIRMATION seed bundle -- always non-negative
    `reservoir_idx`, hence disjoint from every discovery bundle."""
    return make_seeds(k, input_idx)


def assert_disjoint(discovery_bundles, confirmation_bundles) -> None:
    d = {s.reservoir_idx for s in discovery_bundles}
    c = {s.reservoir_idx for s in confirmation_bundles}
    overlap = d & c
    if overlap:
        raise ValueError(f"discovery/confirmation reservoir_idx overlap: {sorted(overlap)}")
