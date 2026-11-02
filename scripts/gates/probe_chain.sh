#!/usr/bin/env bash
# RL_STUDY §12: once the four common-stream re-trainings have saved their
# brains, merge them per graph, then run frozen play for the three graphs in parallel and
# step 1 (S_fixed) for each graph one after the other. Every step is resumable.
set -u
cd "$(dirname "$0")/../.."
LOG=artifacts/logs/probe_chain.log
C=artifacts/runs/checkpoints
P=scripts/gates/frozen_play_probe.py
export OMP_NUM_THREADS=2
until [ -f $C/common_stream_connectome.pt ] && [ -f $C/common_stream_nulllat0.pt ] \
      && [ -f $C/common_stream_nulllat1.pt ] && [ -f $C/common_stream_perm_connectome.pt ]; do sleep 30; done
sleep 20                                         # let the last save finish writing
echo "$(date -Is) checkpoints present; merging" >> $LOG
for g in connectome nulllat:0 nulllat:1; do .venv/bin/python $P merge --graph $g >> $LOG 2>&1; done
for g in connectome nulllat:0 nulllat:1; do
    t=${g/:/}
    .venv/bin/python $P play --graph $g > artifacts/logs/probe_play_$t.log 2>&1 &
    echo "$(date -Is) play $g launched" >> $LOG
    sleep 30
done
for g in connectome nulllat:0 nulllat:1; do
    t=${g/:/}
    .venv/bin/python $P fixed --graph $g > artifacts/logs/probe_fixed_$t.log 2>&1
    echo "$(date -Is) fixed $g ended (exit $?)" >> $LOG
done
wait
echo "$(date -Is) all probe steps ended" >> $LOG
