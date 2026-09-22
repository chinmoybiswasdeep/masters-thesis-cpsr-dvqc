"""Generator for DQRC_DualRoute_Decoupling_V3_2_Colab.ipynb.

Run from `code/`. A NEW, separate builder -- no V1/V2/V2.1/V2.2/V3/V3.1
artifact is read, modified or superseded. All scientific logic lives in
`decoupled_qrc/v3_2_*.py` and is exercised by `tests/test_v3_2_*.py`; the
notebook only wires it together, so the notebook and a local
`python run_v3_2_stage.py` run cannot diverge.
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.splitlines(keepends=True)})


md(r"""# DQRC V3.2 -- Memory/Nonlinearity Control Decoupling (Google Colab)

V3.2 replaces the V3.1 processor encoder with one that is **linear at the local-observable
level**, and selects the memory operating point by **slope stability and interior margin**
rather than by maximum capacity. Those were the two architectural defects that made V3.1's
result uninterpretable.

**No claim of memory-nonlinearity separation is made anywhere in this notebook unless the
held-out confirmation stage actually executes and its gates pass.** Structural route
isolation alone is Level 1 and is explicitly *not* described as separation.

## What V3.1 actually established (verified against its saved artifacts)

| V3.1 finding | verdict |
|---|---|
| structural isolation 6/6 at `max\|dX_P/dm\| = 0.000e+00` | confirmed |
| injected-dependency and serial controls fail as required | confirmed |
| IPC splits train=43 / val=22 / test=28 | confirmed exactly |
| null fractions 0.495 and 0.511 | confirmed |
| `NL_0(encoding only)=0.7201` vs `NL_0(full)=0.6632` | confirmed -- the encoder produced the nonlinearity |
| processor dynamic range `E_NL = 0.0739` against its own 0.20 gate | confirmed (failed) |
| `m* = 0.95` on the exact boundary of `m in [0.1, 0.95]` | confirmed |
| a candidate frozen while `RUN_MODE = "SMOKE"` | confirmed |
| confirmation never executed (`no rows produced at this stage`) | confirmed |
| disjointness proof vacuous over an empty confirmation set | confirmed |
| only 121 of the repository's 369 tests were run | confirmed |

## The two architectural corrections

1. **Encoder.** `rho(u) = (I + sZ)/2` per qubit, `rho_P(u) = rho(u)^(x)N_P`. Every local
   expectation is exactly affine in `s` under any product of single-qubit channels, so
   `NL_0(g=0,J=0) = 0` *exactly* and the encoder fraction `f_enc = 0` **by construction**.
   Verified to machine precision by Gauss-Legendre quadrature, not by sampling.
2. **Memory point.** Ranked by `|dM_long/dm~|/se`, interior margin and low null bias --
   never by maximum `M`. A five-point stencil that would leave the domain is *refused*.

## One structural fact that must be reported, not hidden

With a scalar input and a **noiseless** simulator, instantaneous capacity is degenerate:
once the features span degree-<=N_P polynomials, `C_d = 1.000` exactly for every `d <= N_P`.
The metric is pinned at its ceiling and has **no derivative at all**. This is intrinsic to
instantaneous IPC with a scalar input -- it is the same ceiling that pinned V3.1 -- and it is
why V3.2 measures `NL_0` at a **finite, preregistered shot budget**, chosen during
CALIBRATION so the response surface is unsaturated. The shot budget is a matched resource.""")

# ---------------------------------------------------------------- settings
md(r"""## 1. Settings (the only cell you normally edit)

`RUN_STAGE` drives everything. The stage machine makes the illegal paths impossible rather
than merely discouraged: SMOKE cannot freeze a candidate or report a claim level above 0,
DISCOVERY cannot start unless CALIBRATION passed, and CONFIRMATION can only *load* the
frozen artifact.""")

code(r"""RUN_STAGE = "SMOKE"          # SMOKE | CALIBRATION | DISCOVERY | CONFIRMATION | REPORT
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "dqrc_v3_2"
CONFIRM_EXPENSIVE_RUN = False   # must be True for DISCOVERY / CONFIRMATION to execute

# Where the V3.2 modules live once uploaded (see the next cell).
CODE_DIR = "/content/dqrc_code" """)

md(r"""## 2. Environment

The V3.2 modules are plain NumPy/SciPy; Qiskit is only needed by the repository's older
modules and by the cross-engine equivalence test. Upload the `code/` directory (or mount it
from Drive) so `decoupled_qrc/v3_2_*.py` is importable.""")

code(r"""import os, sys, json, time, platform

try:
    from google.colab import drive          # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    RESULTS_ROOT = f"/content/drive/MyDrive/{DRIVE_SUBDIR}"
else:
    RESULTS_ROOT = os.path.abspath("./results/dqrc_dual_route_v3_2")
os.makedirs(RESULTS_ROOT, exist_ok=True)

if IN_COLAB:
    sys.path.insert(0, CODE_DIR)
else:
    sys.path.insert(0, os.path.abspath("./code"))

# Checkpoints live under RESULTS_ROOT so an interrupted session resumes at one row.
os.environ.setdefault("DQRC_V3_2_CACHE_DIR", os.path.join(RESULTS_ROOT, "_cache"))

import numpy as np, scipy
print("python", sys.version.split()[0], "| numpy", np.__version__, "| scipy", scipy.__version__)
print("results ->", RESULTS_ROOT)""")

md(r"""## 3. Run the full repository test suite

V3.1 ran 121 of 369 tests and reported them as validation. V3.2 runs the whole suite plus
the new `test_v3_2_*` files. A numerical claim made on top of an unrun test suite is not a
validated claim, so Gate B consumes this result.""")

code(r"""import subprocess
proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                      capture_output=True, text=True)
print(proc.stdout[-4000:])
TESTS_PASSED = (proc.returncode == 0)
print("TESTS_PASSED =", TESTS_PASSED)""")

md(r"""## 4. Stage execution

Every stage is a function in `run_v3_2_stage.py`, so this notebook and a local run execute
identical code. Artifacts are written per stage and the expensive stages checkpoint per row.""")

code(r"""import run_v3_2_stage as runner
from decoupled_qrc.v3_2_calibration import CalibrationSettings

runner.RESULTS = __import__("pathlib").Path(RESULTS_ROOT)

if RUN_STAGE == "SMOKE":
    OUT = runner.stage_smoke()
    print(json.dumps(OUT["guards"], indent=2))

elif RUN_STAGE == "CALIBRATION":
    OUT = runner.stage_calibration(CalibrationSettings())

elif RUN_STAGE in ("DISCOVERY", "CONFIRMATION"):
    assert CONFIRM_EXPENSIVE_RUN, (
        "set CONFIRM_EXPENSIVE_RUN = True to authorise an expensive stage")
    OUT = runner.main_stage(RUN_STAGE)

elif RUN_STAGE == "REPORT":
    OUT = runner.stage_report()

print("stage", RUN_STAGE, "complete")""")

md(r"""## 5. What a stage may claim

The cap is enforced in code, not by convention. V3.1 printed *"Highest supported claim
level: 3"* from a SMOKE run whose confirmation had never executed; `assert_claim_allowed`
raises on exactly that.

| stage | maximum claim level |
|---|---|
| SMOKE | 0 -- software validation only |
| CALIBRATION | 0 -- prerequisites only |
| DISCOVERY | 1 -- structural isolation |
| CONFIRMATION | up to 5, and only for gates that actually passed |

## 6. Reproducing this locally

```bash
python -m pytest tests/ -q
cd code
python run_v3_2_stage.py --stage SMOKE
python run_v3_2_stage.py --stage CALIBRATION
python _build_notebook_dualroute_v3_2_colab.py
```

Artifacts land in `results/dqrc_dual_route_v3_2/`. Each carries the git commit, package
versions, hardware, precision and a configuration checksum in its run manifest.""")

with open("DQRC_DualRoute_Decoupling_V3_2_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": CELLS,
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                           "name": "python3"},
                            "language_info": {"name": "python", "version": "3.11"},
                            "colab": {"provenance": [], "toc_visible": True}},
               "nbformat": 4, "nbformat_minor": 5}, f, indent=1)

print(f"Wrote DQRC_DualRoute_Decoupling_V3_2_Colab.ipynb with {len(CELLS)} cells.")
