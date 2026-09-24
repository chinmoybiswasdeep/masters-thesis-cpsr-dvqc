"""Generator for DQRC_DualRoute_Decoupling_V4_Colab.ipynb. Run from code/.

A NEW builder; no V1-V3.2 artifact is read, modified or superseded. All
scientific logic lives in `decoupled_qrc/v4_*.py` and is exercised by
`tests/test_v4_*.py`; the notebook only wires the stages together, so a Colab
run and a local `python run_v4_stage.py` cannot diverge.
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.splitlines(keepends=True)})


md(r"""# DQRC V4 — Memory / Nonlinearity Separation (Google Colab)

V4 replaces the V3.2 architecture entirely. V3.2 is preserved for comparison and is
neither modified nor superseded in place.

## The architecture, in one picture

```
        u_t ──► R  memory register      control m   affine SINGLE-copy injection, fading
         │                                          local observables  ->  M
         └────► P  nonlinear register   control g   RESET each step, fixed-depth reupload
                                                    local observables  ->  N
                joint layer  O_R (x) O_P  (fixed)   ->  N_long
```

**R and P are never coupled.** The global state is exactly `rho_R(t) (x) rho_P(t)`, so
`X_R` cannot depend on `g` and `X_P` cannot depend on `m`. The cross-derivatives are
therefore zero *by construction* for the route-restricted metrics.

That is the design goal, not the result. The scientific content is:
1. the **diagonal** effects are large, monotone and survive unseen seeds;
2. **N_long** is real and requires **both** controls;
3. the measurement pipeline demonstrably **has the power** to detect coupling — the
   `contaminated` and `serial` negative controls must, and do, fail.

## Three things measurement forced on the design

**1. The encoder is not globally linear.** `((I+uZ)/2)^(x)n` carries degrees up to `n`;
only the *local* readout is affine. The `AUDIT` stage proves this by exact Legendre
projection, and the `n` copies are counted as a nonlinear resource.

**2. A multilinear memory has a hard capability boundary.** Quantum channels are linear
in `rho`, so one affine injection per step makes `rho_R(t)` multilinear in the past
inputs. Any target with degree ≥ 2 at a **strictly positive** delay — e.g. `P_2(u_{t-3})`
— is therefore unreachable. N_long is built from targets whose degree sits at delay 0,
so the degree comes from P (control `g`) and the delay from R (control `m`). The
unreachable class is measured and reported, never gated.

**3. Capacity with an unbounded readout is span-based, hence binary in `g`.** Measured
during development: with a validation-selected ridge, `N(g)` was `0.003` at `g=0` and
then pinned at exactly `3.000` for *every* `g>0`. No feature count and no coupling
strength graded it. V4 therefore uses a **fixed, preregistered ridge penalty** — a
bounded-norm readout, identical at every `(m,g)` and counted as a resource. Combined
with a perturbative interaction this makes degree-`d` content enter as `g^(d-1)`, and
`N(g)` becomes monotone. The simulation itself stays exact and noiseless: finite shots
are never used to manufacture nonlinearity.""")

md(r"""## 1. Settings""")
code(r"""RUN_STAGE = "AUDIT"   # AUDIT SMOKE CALIBRATION ARCHITECTURE_SEARCH
                      # STRESS_TEST FREEZE CONFIRMATION REPORT
N_CANDIDATES = 24
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "dqrc_v4"
CODE_DIR = "/content/dqrc_code"      # upload code/ here""")

md("## 2. Environment")
code(r"""import os, sys, json, time
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    RESULTS_ROOT = f"/content/drive/MyDrive/{DRIVE_SUBDIR}"
else:
    RESULTS_ROOT = os.path.abspath("./results/v4")
os.makedirs(RESULTS_ROOT, exist_ok=True)
sys.path.insert(0, CODE_DIR if IN_COLAB else os.path.abspath("./code"))

import numpy as np, scipy
print("python", sys.version.split()[0], "| numpy", np.__version__, "| scipy", scipy.__version__)
print("results ->", RESULTS_ROOT)""")

md(r"""## 3. Full repository test suite

A numerical claim on top of an unrun test suite is not a validated claim.""")
code(r"""import subprocess
p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                   capture_output=True, text=True)
print(p.stdout[-3000:])
TESTS_PASSED = (p.returncode == 0)
print("TESTS_PASSED =", TESTS_PASSED)""")

md(r"""## 4. Run a stage

Stage order is enforced. `CONFIRMATION` refuses unless the frozen artifact re-hashes
correctly and its seed bank is disjoint from every seed the search ever saw.""")
code(r"""import run_v4_stage as R
from pathlib import Path
R.RESULTS = Path(RESULTS_ROOT)
R.FROZEN_PATH = R.RESULTS / "frozen_config.json"
R.LEDGER_PATH = R.RESULTS / "sequential_ledger.json"

OUT = (R.stage_search(n_candidates=N_CANDIDATES) if RUN_STAGE == "ARCHITECTURE_SEARCH"
       else R.STAGES[RUN_STAGE]())
print("stage", RUN_STAGE, "complete")""")

md(r"""## 5. The claim

`MEMORY-NONLINEARITY SEPARATION DEMONSTRATED` is produced by exactly one function,
`v4_artifacts.ClaimGuard.render`, and only when every mandatory gate passed on a
complete confirmation run whose config hash matches the frozen artifact, under valid
sequential error control. There is no flag that overrides it.""")
code(r"""from decoupled_qrc.v4_artifacts import ClaimGuard
conf_path = R.RESULTS / "confirmation.json"
if conf_path.exists():
    conf = json.loads(conf_path.read_text())
    print(conf["claim"])
    for name, g in conf["gates"].items():
        print(f"  {g['status']:<14} {name}")
else:
    print(ClaimGuard(gates={}, confirmation_complete=False).render())""")

md(r"""## 6. Reproducing locally

```bash
python -m pytest tests/ -q
cd code
for S in AUDIT SMOKE CALIBRATION ARCHITECTURE_SEARCH STRESS_TEST FREEZE CONFIRMATION REPORT; do
    python run_v4_stage.py --stage $S
done
python _build_notebook_dualroute_v4_colab.py
```

Artifacts land in `results/v4/`, including `report.md`, `report.json`, the frozen
config with its sha256, the search history, the confirmation rows and the figures.""")

with open("DQRC_DualRoute_Decoupling_V4_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": CELLS,
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                           "name": "python3"},
                            "language_info": {"name": "python", "version": "3.11"},
                            "colab": {"provenance": [], "toc_visible": True}},
               "nbformat": 4, "nbformat_minor": 5}, f, indent=1)
print(f"Wrote DQRC_DualRoute_Decoupling_V4_Colab.ipynb with {len(CELLS)} cells.")
