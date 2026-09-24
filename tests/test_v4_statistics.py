"""V4 statistics: equivalence, multiplicity, Jacobian, sequential accounting."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from decoupled_qrc.v4_statistics import (  # noqa: E402
    CI, SequentialLedger, holm, holm_equivalence, main_effect, nested_bootstrap,
    quadrant_contrast, ratio_upper_bound, response_jacobian, tost)


def test_nested_bootstrap_propagates_between_seed_variation():
    tight = {0: [1.0, 1.01], 1: [0.99, 1.0], 2: [1.0, 1.02]}
    spread = {0: [0.2, 0.25], 1: [1.0, 1.1], 2: [1.9, 2.0]}
    a = nested_bootstrap(tight, seed=0, n_boot=800)
    b = nested_bootstrap(spread, seed=0, n_boot=800)
    assert (b.hi - b.lo) > (a.hi - a.lo)
    assert b.between_var > a.between_var
    assert a.excludes_zero() is True


def test_equivalence_requires_the_WHOLE_interval_inside_the_margin():
    """A wide, non-significant interval must FAIL, not pass."""
    wide = CI(point=0.0, lo=-0.5, hi=0.5, level=0.9, n_outer=3, n_boot=100)
    assert tost(wide, -0.03, 0.03)["equivalent"] is False
    tight = CI(point=0.001, lo=-0.004, hi=0.006, level=0.9, n_outer=3, n_boot=100)
    assert tost(tight, -0.03, 0.03)["equivalent"] is True
    assert "not evidence of absence" in tost(wide, -0.03, 0.03)["reason"]


def test_exact_zero_cross_effect_is_equivalent():
    zero = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}
    ci = nested_bootstrap(zero, seed=0, n_boot=500, level=0.90)
    assert ci.lo == 0.0 and ci.hi == 0.0
    assert tost(ci, -0.03, 0.03)["equivalent"] is True


def test_ratio_upper_bound_is_paired_and_bounded():
    num = {0: [0.001, 0.002], 1: [0.0015, 0.001], 2: [0.002, 0.0018]}
    den = {0: [0.5, 0.52], 1: [0.49, 0.5], 2: [0.51, 0.5]}
    r = ratio_upper_bound(num, den, seed=0, n_boot=800)
    assert r["ratio"] < 0.02 and r["upper"] < 0.2
    big = ratio_upper_bound(den, den, seed=0, n_boot=400)
    assert big["upper"] > 0.5


def test_holm_is_more_conservative_than_uncorrected():
    p = {"a": 0.01, "b": 0.04, "c": 0.20}
    out = holm(p, alpha=0.05)
    assert out["results"]["a"]["reject"] is True
    assert out["results"]["c"]["reject"] is False
    assert out["results"]["a"]["threshold"] < 0.05
    assert out["all_reject"] is False


def test_holm_equivalence_family_fails_if_one_member_fails():
    good = CI(0.0, -0.001, 0.001, 0.9, 3, 100)
    bad = CI(0.2, 0.15, 0.25, 0.9, 3, 100)
    assert holm_equivalence({"a": good, "b": good}, -0.03, 0.03)["family_equivalent"] is True
    assert holm_equivalence({"a": good, "b": bad}, -0.03, 0.03)["family_equivalent"] is False


def test_main_effect_averages_over_the_other_control():
    grid = {(0.0, 0.0): {"M": 0.1}, (0.0, 1.0): {"M": 0.1},
            (1.0, 0.0): {"M": 0.5}, (1.0, 1.0): {"M": 0.5}}
    eff = main_effect(grid, "M", "m")
    assert abs(eff["delta"] - 0.4) < 1e-12 and eff["n_slices"] == 2
    assert abs(main_effect(grid, "M", "g")["delta"]) < 1e-12
    with pytest.raises(ValueError):
        main_effect(grid, "M", "x")


def test_quadrant_contrast_finds_the_high_high_advantage():
    grid = {(0.0, 0.0): {"V": 0.0}, (0.0, 1.0): {"V": 0.1},
            (1.0, 0.0): {"V": 0.1}, (1.0, 1.0): {"V": 0.9}}
    q = quadrant_contrast(grid, "V")
    assert abs(q["HH"] - 0.9) < 1e-12
    assert abs(q["improvement"] - 0.8) < 1e-12


def test_response_jacobian_is_diagonal_for_ideal_separation():
    j = response_jacobian(dm_M=0.4, dm_N=0.0, dg_M=0.0, dg_N=0.9)
    assert j["off_diagonal_mass"] == 0.0
    assert j["off_over_diag"] == 0.0
    assert abs(j["response_angle_deg"] - 90.0) < 1e-9
    j2 = response_jacobian(dm_M=0.4, dm_N=0.4, dg_M=0.4, dg_N=0.4)
    assert j2["off_over_diag"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Sequential confirmation accounting
# --------------------------------------------------------------------------
def test_sequential_ledger_spends_alpha_and_persists(tmp_path):
    p = tmp_path / "ledger.json"
    led = SequentialLedger.load(p, alpha_total=0.05, max_attempts=3)
    assert led.n_attempts() == 0 and led.alpha_for_next() == pytest.approx(0.05 / 3)
    led.record(version="v4.0", seed_bank="A", passed=False)
    assert led.alpha_spent() == pytest.approx(0.05 / 3)
    again = SequentialLedger.load(p)
    assert again.n_attempts() == 1                 # survived the round trip


def test_sequential_budget_cannot_be_exceeded(tmp_path):
    p = tmp_path / "ledger.json"
    led = SequentialLedger.load(p, alpha_total=0.05, max_attempts=2)
    led.record(version="v4.0", seed_bank="A", passed=False)
    led.record(version="v4.1", seed_bank="B", passed=False)
    with pytest.raises(RuntimeError):
        led.record(version="v4.2", seed_bank="C", passed=True)
