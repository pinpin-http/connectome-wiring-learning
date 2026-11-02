#!/usr/bin/env bash
# RL_STUDY §14 and §13.3: after the expert-stream re-trainings, merge, then in
# groups of three plays (GPU-bound): (1) §14 training suite, threshold rule, plus S_fixed;
# (2) §14 unseen suite, hold6; (3) §13.3 scripted-stream brains, unseen suite, hold6.
set -u
cd "$(dirname "$0")/../.."
LOG=artifacts/logs/probe_chain.log
C=artifacts/runs/checkpoints
P=scripts/gates/frozen_play_probe.py
export OMP_NUM_THREADS=2
until [ -f $C/common_stream_expert_connectome.pt ] && [ -f $C/common_stream_expert_nulllat0.pt ] \
      && [ -f $C/common_stream_expert_nulllat1.pt ]; do sleep 30; done
sleep 20
for g in connectome nulllat:0 nulllat:1; do .venv/bin/python $P merge --graph $g --stream expert >> $LOG 2>&1; done
echo "$(date -Is) §14 merged" >> $LOG
group() {   # args: label, extra options
    local label=$1; shift
    for g in connectome nulllat:0 nulllat:1; do
        t=${g/:/}
        .venv/bin/python $P play --graph $g "$@" > artifacts/logs/probe_${label}_$t.log 2>&1 &
        sleep 20
    done
    wait
    echo "$(date -Is) group $label ended" >> $LOG
}
group x_training --suite training --stream expert &
G1=$!
for g in connectome nulllat:0 nulllat:1; do
    .venv/bin/python $P fixed --graph $g --suite training --stream expert > artifacts/logs/probe_x_fixed_${g/:/}.log 2>&1
done
echo "$(date -Is) §14 fixed ended" >> $LOG
wait $G1
group x_unseen_hold6 --suite unseen --rule hold6 --stream expert
group unseen_hold6 --suite unseen --rule hold6
echo "$(date -Is) all §14 / §13.3 plays ended" >> $LOG
