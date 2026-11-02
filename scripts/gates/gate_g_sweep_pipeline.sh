#!/bin/sh
# Unattended Gate G sweep pipeline (docs/GATE_G_PLAN.md, sweep amendment).
# Waits for the dynamics diagnostics, then explore -> reference -> confirm; confirm runs
# only if the reference stage validated the criterion. Re-running resumes where it stopped.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python"
export OMP_NUM_THREADS=4
R=artifacts/runs
while pgrep -f "diagnostics/gate_g_dynamics.py" >/dev/null; do sleep 20; done
$PY scripts/gates/gate_g_sweep.py explore --n-train 800 --n-test 100 || exit 1
$PY scripts/gates/gate_g_sweep.py reference --n-train 800 --n-test 200 || exit 1
VALID=$($PY -c "import json;print(json.load(open('$R/gate_g_sweep_reference-20260930.json'))['criterion_validated'])")
VAR=$($PY -c "import json;print(json.load(open('$R/gate_g_sweep_explore-20260930.json'))['selection']['selected'])")
echo "criterion validated on references: $VALID; selected variant: $VAR"
if [ "$VALID" = "True" ]; then
  $PY scripts/gates/gate_g_sweep.py confirm --variant "$VAR" --n-train 800 --n-test 200
else
  echo "confirm NOT run: the criterion did not separate the references"
fi
