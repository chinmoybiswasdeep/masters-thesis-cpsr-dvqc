"""Preregistered V7 gate calculations from persisted point records."""

from __future__ import annotations

import numpy as np


def paired_interval(values, alpha: float = 0.01, resamples: int = 20_000) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return {"estimate": float(values.mean()), "lower": float("-inf"), "upper": float("inf")}
    rng = np.random.default_rng(7007)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
    }


def separation_gate(seed_corners: list[dict], protocol: dict, readout: str) -> dict:
    """Each item maps LL/LH/HL/HH to evaluate_features output for one seed."""
    threshold = protocol["thresholds"]
    memory = [row["HL"][readout]["M"] - row["LL"][readout]["M"] for row in seed_corners]
    nonlinear = [row["LH"][readout]["N"] - row["LL"][readout]["N"] for row in seed_corners]
    m_cross = [row["HH"][readout]["N"] - row["LH"][readout]["N"] for row in seed_corners]
    g_cross = [row["HH"][readout]["M"] - row["HL"][readout]["M"] for row in seed_corners]
    result = {name: paired_interval(values, protocol["statistics"]["familywise_alpha"]) for name, values in (
        ("memory_main", memory), ("nonlinearity_main", nonlinear),
        ("memory_to_nonlinearity", m_cross), ("nonlinearity_to_memory", g_cross),
    )}
    margin = threshold["cross_effect_equivalence_margin"]
    result["passed"] = bool(
        result["memory_main"]["estimate"] >= threshold["main_effect_min"]
        and result["memory_main"]["lower"] >= threshold["main_effect_lower_bound_min"]
        and result["nonlinearity_main"]["estimate"] >= threshold["main_effect_min"]
        and result["nonlinearity_main"]["lower"] >= threshold["main_effect_lower_bound_min"]
        and -margin < result["memory_to_nonlinearity"]["lower"]
        and result["memory_to_nonlinearity"]["upper"] < margin
        and -margin < result["nonlinearity_to_memory"]["lower"]
        and result["nonlinearity_to_memory"]["upper"] < margin
    )
    return result


def hh_gate(corners: dict[str, float], protocol: dict) -> dict:
    alternative = max(corners[name] for name in ("LL", "LH", "HL"))
    advantage = corners["HH"] - alternative
    return {
        "hh": float(corners["HH"]),
        "best_alternative": float(alternative),
        "advantage": float(advantage),
        "passed": bool(advantage >= protocol["thresholds"]["combined_hh_margin"]),
    }

