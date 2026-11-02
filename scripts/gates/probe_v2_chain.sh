#!/usr/bin/env bash
# RL_STUDY §13.1: the §12 brains, frozen, on the training suite of play
# harness v2 (three graphs in parallel), then S_fixed on the same levels per graph.
set -u
cd "$(dirname "$0")/../.."
LOG=artifacts/logs/probe_chain.log
P=scripts/gates/frozen_play_probe.py
export OMP_NUM_THREADS=2
for g in connectome nulllat:0 nulllat:1; do
    t=${g/:/}
    .venv/bin/python $P play --graph $g --suite training > artifacts/logs/probe_v2_play_$t.log 2>&1 &
    echo "$(date -Is) v2 play $g launched" >> $LOG
    sleep 30
done
for g in connectome nulllat:0 nulllat:1; do
    t=${g/:/}
    .venv/bin/python $P fixed --graph $g --suite training > artifacts/logs/probe_v2_fixed_$t.log 2>&1
    echo "$(date -Is) v2 fixed $g ended (exit $?)" >> $LOG
done
wait
echo "$(date -Is) all v2 probe steps ended" >> $LOG
