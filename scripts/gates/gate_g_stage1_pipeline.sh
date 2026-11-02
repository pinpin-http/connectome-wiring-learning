#!/bin/sh
# Stage-1 follow-up tests (docs/GATE_G_PLAN.md, stage-1 amendment), unattended and resumable.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python"
export OMP_NUM_THREADS=4
R=artifacts/runs
$PY scripts/gates/gate_g_stage1.py delayed-explore --n-train 1200 --n-test 100 || exit 1
VAR=$($PY -c "import json;print(json.load(open('$R/gate_g_stage1_delayed-explore-20261001.json'))['selection']['selected'])")
echo "selected gamma variant: $VAR"
$PY scripts/gates/gate_g_stage1.py delayed-confirm --variant "$VAR" --n-train 1200 --n-test 200 || exit 1
$PY scripts/gates/gate_g_stage1.py reversal --n-train 800 --n-test 200 || exit 1
$PY scripts/gates/gate_g_stage1.py generalise --n-train 800 --n-test 200 || exit 1
echo "stage-1 pipeline complete"
