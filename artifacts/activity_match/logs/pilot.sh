#!/bin/bash
# ACTIVITY_MATCH_V1 small complete trial on the development graphs (nullin:100, 101), pilot banks and pilot gains.
set -u
cd "$(dirname "$0")/../../.."
R=pilot_am
PY=.venv/bin/python
S=scripts/activity_match/run_am.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "$(date -u +%FT%TZ) start"
$PY $S prepare --run $R --pilot || exit 1
$PY $S validate --run $R || { echo "STOP: data checks"; exit 1; }
$PY $S extract --run $R || exit 1
$PY $S decode --run $R --workers 2 --shuffled-control || exit 1
$PY $S analyse --run $R --replicates 1000 || exit 1
$PY $S estimate --run $R
echo "$(date -u +%FT%TZ) PILOT DONE"
