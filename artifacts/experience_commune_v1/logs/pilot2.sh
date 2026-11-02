#!/bin/bash
# Reruns what pilot.sh could not finish (out of memory at 4 processes x ~7 GB): the missing pilot
# graph and the 24-learner benchmark, 3 processes, reduced-memory brain. Development graphs only.
set -u
cd "$(dirname "$0")/../../.."
PY=/workspace/SMB_Flywire_Speedrun/.venv/bin/python
S=scripts/experience_commune/ec.py
L=artifacts/experience_commune_v1/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date -u +%FT%TZ) start"
$PY $S run --graph nulllat:100 --blocks d0 --kinds pilot --tag pilot --log-accuracy > $L/pilot_nulllat100_b.log 2>&1 &
P1=$!
$PY $S run --graph nullin:101 --blocks b0,b1,b2,b3,b4,b5 --tag bench24 --stop-at 300 --banks "" > $L/bench24_a.log 2>&1 &
P2=$!
$PY $S run --graph nullin:100 --blocks b0,b1,b2,b3,b4,b5 --tag bench24b --stop-at 300 --banks "" > $L/bench24_b.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
echo "$(date -u +%FT%TZ) DONE"
