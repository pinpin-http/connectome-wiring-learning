#!/bin/bash
# Throughput benchmark at the campaign's shape: 3 concurrent processes x 24 learners (main blocks b0-b5) on the
# development graphs, 200 decisions after the k = 0 evaluation. Never analysed.
set -u
cd "$(dirname "$0")/../../.."
PY=/workspace/SMB_Flywire_Speedrun/.venv/bin/python
S=scripts/experience_commune/ec.py
L=artifacts/experience_commune_v1/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for g in nullin:100 nullin:101 nulllat:100; do
  $PY $S run --graph $g --blocks b0,b1,b2,b3,b4,b5 --tag bench3 --stop-at 200 --banks "" > $L/bench3_${g/:/}.log 2>&1 &
done
wait
echo "$(date -u +%FT%TZ) DONE"
