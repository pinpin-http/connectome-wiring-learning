#!/bin/bash
# EXPERIENCE_COMMUNE_V1 pilot and throughput benchmark, started once ACTIVITY_MATCH_V2 has
# released the GPU. Development graphs only.
set -u
cd "$(dirname "$0")/../../.."
PY=/workspace/SMB_Flywire_Speedrun/.venv/bin/python
S=scripts/experience_commune/ec.py
L=artifacts/experience_commune_v1/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q "decoder jobs" /workspace/SMB_Flywire_Speedrun/artifacts/activity_match/logs/campaign_v2.log; do sleep 20; done
echo "$(date -u +%FT%TZ) GPU released by V2; pilot and benchmark start"
$PY $S run --graph nulllat:100 --blocks d0 --kinds pilot --tag pilot --log-accuracy > $L/pilot_nulllat100.log 2>&1 &
P1=$!
$PY $S run --graph nullin:100 --blocks d0 --kinds pilot --tag pilot --log-accuracy > $L/pilot_nullin100.log 2>&1 &
P2=$!
$PY $S run --graph nullin:101 --blocks b0,b1,b2,b3,b4,b5 --tag bench24 --stop-at 300 --banks "" > $L/bench24.log 2>&1 &
P3=$!
$PY $S run --graph nulllat:100 --blocks b0,b1,b2 --tag bench12 --stop-at 300 --banks "" > $L/bench12.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo "$(date -u +%FT%TZ) PILOT DONE"
