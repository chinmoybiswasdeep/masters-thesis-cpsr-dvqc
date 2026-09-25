"""Generator for DQRC_V6_Colab.ipynb. Run from code/.

A NEW builder; no V1-V5 artifact is read, modified or superseded. Scientific
logic lives in decoupled_qrc/v6_*.py (tested in tests/test_v6.py); the notebook
only wires the stages of run_v6_stage.py together and reads saved verdicts.
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})


md(r"""# DQRC V6 — General memory / nonlinearity / nonlinear-memory separation (Google Colab)

V5.4 cannot represent delayed nonlinear targets: every V5.4 feature is linear in each past
input and contains at most one past input, so P2(u_{t-τ}), P3(u_{t-τ}) and
P1(u_{t-τ1})P1(u_{t-τ2}) have capacity exactly 0 (`results/v6/algebra/V5.4.json`).

V6 has three routes plus a joint readout:
```
R  random-SWAP linear memory (control m)            -> M   (R only)
P  3-copy processor, ONE observable (control g)     -> N   (P only; composition-controlled)
J  z_r^R x sin(theta_J g) u_t^2 (2-copy Y channel)  -> C1  current nonlinear x old linear
Q  second register written by a P-type processor    -> C2  delayed P2, C4 delayed P3
   + pure-product pair readouts of R rails           -> C3  old x old
```
Every feature is a 10^4-shot estimate (a declared, fixed resource).

**Disclosed limitations**
- With exact expectations (infinite shots) C2 and C3 switch on at any g > 0. Their
  gradation in g comes from finite measurement statistics.
- The frozen V6.7 design was chosen by re-drawing the dev sentinel calibration (a margin
  that behaves like a coin flip at T = 1600) until one draw passed, a user decision
  recorded in `docs/V6_PROGRESS.md`. The preregistered confirmation gates are unaffected.
- The mechanism is classically simulable: every register stays diagonal in a product basis.""")

md("## 1. Settings")
code(r"""RUN_STAGE = "REPORT"   # STEP1 PREREG_GATES ALGEBRA SENTCAL PROBE DEVGATES TESTS FREEZE CONFIRM REPORT
VERSION = "V6.7"
CODE_DIR = "/content/dqrc_code"   # upload code/ here (Colab)""")

md("## 2. Environment")
code(r"""import os, sys, json
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False
here = os.getcwd()
sys.path.insert(0, CODE_DIR if IN_COLAB else (here if os.path.basename(here) == "code" else os.path.join(here, "code")))
import numpy as np, scipy
print("python", sys.version.split()[0], "| numpy", np.__version__, "| scipy", scipy.__version__)""")

md(r"""## 3. Run a stage

`FREEZE` refuses unless every dev gate and margin passed and the full test suite passed;
`CONFIRM` refuses unless the frozen artifact, the preregistration and every source file
re-hash (or a hashed amendment names the change), and refuses to run twice.""")
code(r"""import run_v6_stage as S
fn = S.STAGES[RUN_STAGE]
fn() if RUN_STAGE in ("STEP1", "PREREG_GATES") else fn(VERSION)""")

md("## 4. Verdicts (read from saved gates, never recomputed)")
code(r"""p = S.vdir(VERSION) / "gates.json"
if p.exists():
    g = json.loads(p.read_text())
    for k, v in g["gates"]["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL MANDATORY GATES:", g["success"])
    print("quantum-specific:", g["quantum_specific"]["statement"])
else:
    print("no confirmation yet for", VERSION)""")

md(r"""## 5. Reproducing locally

```bash
python -m pytest tests/ -q
cd code
python run_v6_stage.py --stage STEP1
python run_v6_stage.py --stage ALGEBRA  --version V6.7
python run_v6_stage.py --stage SENTCAL  --version V6.7
python run_v6_stage.py --stage DEVGATES --version V6.7
python run_v6_stage.py --stage TESTS    --version V6.7
python run_v6_stage.py --stage FREEZE   --version V6.7   # refuses: already frozen
python run_v6_stage.py --stage CONFIRM  --version V6.7   # refuses: already confirmed
python run_v6_stage.py --stage REPORT   --version V6.7
python _build_notebook_v6_colab.py
```""")

with open("DQRC_V6_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": CELLS,
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python", "version": "3.11"},
                            "colab": {"provenance": [], "toc_visible": True}},
               "nbformat": 4, "nbformat_minor": 5}, f, indent=1)
print(f"Wrote DQRC_V6_Colab.ipynb with {len(CELLS)} cells.")
