#!/usr/bin/env bash
# First look at the laterality-preserving null (randomize.depth_stratified_lateral) once its three
# realisations are cached: sensory fan-out, then the untrained motor code (activity and
# decodability, as §19 S4). CPU only, low priority, deadline 21:35. No training, no task outcome.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
LOG=artifacts/logs/nulllat_check.log
DEADLINE=$(date -d "21:35" +%s)
left() { echo $(( DEADLINE - $(date +%s) )); }
until [ "$(ls artifacts/cache/nulls/ | grep -c depth_stratified_lateral)" -ge 3 ]; do
    [ "$(left)" -le 0 ] && { echo "$(date -Is) deadline before the nulls were built" >> "$LOG"; exit 1; }
    sleep 20
done
echo "$(date -Is) lateral nulls cached; fan-out" >> "$LOG"
nice -n 12 timeout "$(left)" .venv/bin/python scripts/diagnostics/sensory_fanout.py nulllat:0 nulllat:1 nulllat:2 >> "$LOG" 2>&1
echo "$(date -Is) probe" >> "$LOG"
nice -n 12 timeout "$(left)" .venv/bin/python scripts/diagnostics/wiring_probe.py --device cpu --graphs nulllat:0 nulllat:1 nulllat:2 >> "$LOG" 2>&1
echo "$(date -Is) done (exit $?)" >> "$LOG"
