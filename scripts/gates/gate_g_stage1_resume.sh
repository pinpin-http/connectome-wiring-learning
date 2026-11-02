#!/bin/sh
# Resume stage 1 after an interruption: reversal (from its checkpoints), generalisation,
# then the separately declared delayed-long test.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python"
export OMP_NUM_THREADS=4
$PY scripts/gates/gate_g_stage1.py reversal --n-train 800 --n-test 200 || exit 1
$PY scripts/gates/gate_g_stage1.py generalise --n-train 800 --n-test 200 || exit 1
$PY scripts/gates/gate_g_stage1.py delayed-long --variant G8 --n-train 2400 --n-test 200 || exit 1
echo "stage-1 resume complete"
