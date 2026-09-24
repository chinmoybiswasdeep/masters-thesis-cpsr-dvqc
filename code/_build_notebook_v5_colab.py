"""Generator for DQRC_V5_Colab.ipynb. Run from code/.

A NEW builder; no V1-V4 artifact is read, modified or superseded. Scientific
logic lives in `decoupled_qrc/v5_architecture.py` and the V4-audit engine
(`audit_ipc`, `audit_core`, `audit_checks`); the notebook only wires the stages
of `run_v5_stage.py` together.
"""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.splitlines(keepends=True)})


md(r"""# DQRC V5 — Intrinsic Memory / Nonlinearity Separation (Google Colab)

V5 exists because the independent audit of V4 (`results/v4/independent_audit/`) found V4's
nonlinear control to be a **ridge artifact**: its 12 processor observables span every
polynomial of degree ≤ 4, so the unregularised capacity is 1.0 for every g ≳ 1e-4 and the
feature subspace does not move with g. V4 and its audit are preserved unchanged.

```
 u_t ─► R  random-SWAP register, rails 0..L     control m -> p = m p_max   read <Z_1..Z_L>   -> M
 u_t ─► P  2 copies rho(u), Ry(pi/2), exp(-i theta Z0Z1/2)   theta = g theta_max
           read ONE observable cos(phi)X0 + sin(phi)Y0 = cos(phi)cos(theta) u + sin(phi)sin(theta) u^2 -> N
 joint     z_r * f   (injection rail excluded)                                              -> C1
```

What makes the audit's *intrinsic* tests passable:
1. **One processor observable.** Single-feature capacity is `C_d = a_d^2 / sum a^2`: the
   nonlinear *fraction* of the feature, scale-invariant and continuous in g.
2. **Exactly linear memory.** The random-SWAP channel acts linearly on single-site Z
   marginals; only single-site Z is read, so R computes no products of old inputs.
3. **No encoder leakage.** At g = 0 every operational observable is affine in u (proved
   by exact quadrature in `encoder.json`).

Gates are the V4 audit's `all_gates`, imported unchanged. Claim 5 (quantum-specific
mechanism) is expected to FAIL: the joint features are products of local ones and are
classically reproducible.""")

md("## 1. Settings")
code(r"""RUN_STAGE = "REPORT"   # SEARCH DEVGATES FREEZE CONFIRM REPORT
VERSION = "V5.1"      # V5 (failed its dev freeze criterion) | V5.1
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "dqrc_v5_lineage"
CODE_DIR = "/content/dqrc_code"      # upload code/ here""")

md("## 2. Environment")
code(r"""import os, sys, json
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if IN_COLAB and USE_GOOGLE_DRIVE:
    drive.mount('/content/drive', force_remount=False)
    RESULTS_ROOT = f"/content/drive/MyDrive/{DRIVE_SUBDIR}/{VERSION}"
else:
    sub = "v5" if VERSION == "V5" else "v5_1"
    RESULTS_ROOT = os.path.abspath(f"../results/{sub}" if os.path.basename(os.getcwd()) == "code"
                                   else f"./results/{sub}")
os.makedirs(RESULTS_ROOT, exist_ok=True)
sys.path.insert(0, CODE_DIR if IN_COLAB else os.path.abspath(
    "." if os.path.basename(os.getcwd()) == "code" else "./code"))
import numpy as np, scipy
print("python", sys.version.split()[0], "| numpy", np.__version__, "| scipy", scipy.__version__)
print("results ->", RESULTS_ROOT)""")

md(r"""## 3. Run a stage

`FREEZE` refuses unless every development gate passed with margin; `CONFIRM` refuses
unless the frozen artifact and every source file re-hash, and refuses to run twice.""")
code(r"""import run_v5_stage as S
from pathlib import Path
S.configure(VERSION)
S.OUT = Path(RESULTS_ROOT)
S.FROZEN = S.OUT / S.FROZEN.name
S.STAGES[RUN_STAGE]()""")

md("## 4. Verdicts (read from saved gates, never recomputed)")
code(r"""p = Path(RESULTS_ROOT) / "gates.json"
if p.exists():
    g = json.loads(p.read_text())
    for k, v in g["gates_sequential"].items():
        print(f"  {'PASS' if v['passed'] else 'FAIL'}  {k}")
    print("OVERALL:", g["overall"], "| success (claims 1-4):", g["success_claims_1_to_4"])
else:
    print("no confirmation yet")""")

md(r"""## 5. Reproducing locally

```bash
python -m pytest tests/ -q
cd code
for S in SEARCH DEVGATES FREEZE CONFIRM REPORT; do python run_v5_stage.py --stage $S; done
python _build_notebook_v5_colab.py
```""")

with open("DQRC_V5_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": CELLS,
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                           "name": "python3"},
                            "language_info": {"name": "python", "version": "3.11"},
                            "colab": {"provenance": [], "toc_visible": True}},
               "nbformat": 4, "nbformat_minor": 5}, f, indent=1)
print(f"Wrote DQRC_V5_Colab.ipynb with {len(CELLS)} cells.")
