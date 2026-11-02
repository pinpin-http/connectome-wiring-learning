#!/bin/bash
# RETINO_ALIGN_V1 main campaign, frozen protocol v1. Each step resumes; a failed step stops the chain.
set -u
cd "$(dirname "$0")/../../../.."
R=main_v1
PY=.venv/bin/python
S=scripts/retino_align/run_study.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "$(date -u +%FT%TZ) start"
$PY $S validate --run $R || { echo "$(date -u +%FT%TZ) STOP: input controls failed"; exit 1; }
echo "$(date -u +%FT%TZ) controls pass"
$PY $S extract --run $R || { echo "$(date -u +%FT%TZ) STOP: extract rc=$?"; exit 1; }
$PY $S decode --run $R --workers 3 || { echo "$(date -u +%FT%TZ) STOP: decode"; exit 1; }
$PY $S status --run $R
$PY $S analyse --run $R || { echo "$(date -u +%FT%TZ) STOP: analyse"; exit 1; }
$PY scripts/retino_align/make_figures.py --run $R || echo "$(date -u +%FT%TZ) figures failed"
echo "$(date -u +%FT%TZ) CAMPAIGN DONE"
