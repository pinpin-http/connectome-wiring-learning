#!/bin/bash
# ACTIVITY_MATCH_V1 main campaign, frozen protocol v1 (docs/activity_match/PROTOCOL_FROZEN.md). Each step resumes;
# a failed step stops the chain.
set -u
cd "$(dirname "$0")/../../.."
R=main_am
PY=.venv/bin/python
S=scripts/activity_match/run_am.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "$(date -u +%FT%TZ) start"
$PY $S prepare --run $R || { echo "$(date -u +%FT%TZ) STOP: prepare"; exit 1; }
$PY $S validate --run $R || { echo "$(date -u +%FT%TZ) STOP: data checks failed"; exit 1; }
$PY $S estimate --run $R
$PY $S extract --run $R || { echo "$(date -u +%FT%TZ) STOP: extract rc=$?"; exit 1; }
$PY $S decode --run $R --workers 3 --shuffled-control || { echo "$(date -u +%FT%TZ) STOP: decode"; exit 1; }
$PY $S status --run $R
$PY $S analyse --run $R || { echo "$(date -u +%FT%TZ) STOP: analyse"; exit 1; }
$PY scripts/activity_match/make_figures.py --calibration calibration --run $R || echo "$(date -u +%FT%TZ) figures failed"
echo "$(date -u +%FT%TZ) CAMPAIGN DONE"
