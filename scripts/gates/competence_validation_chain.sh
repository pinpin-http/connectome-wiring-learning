#!/usr/bin/env bash
# RL_STUDY §11.1 (b): validate the frozen-competence criterion on its references, once the
# two lateral-null probe plays (§12) have ended: Stage D checkpoints (connectome, then
# depth-stratified null 0) evaluated frozen with stable streams, version 19 (learners,
# shuffled-label twins, one scripted take-off v2 reference per group), 15,000 decisions.
set -u
cd "$(dirname "$0")/../.."
LOG=artifacts/logs/probe_chain.log
export OMP_NUM_THREADS=2
while ps -eo args | grep -qE "^[^ ]*python[^ ]* scripts/gates/frozen_play_probe.py play --graph nulllat"; do sleep 30; done
echo "$(date -Is) competence validation: connectome Stage D frozen eval start" >> $LOG
.venv/bin/python scripts/train_mario_sweep.py --version 19 --eval-from artifacts/runs/checkpoints/mario_stageD.pt \
    --max-decisions 15000 > artifacts/logs/stageD_eval_connectome.log 2>&1
echo "$(date -Is) competence validation: connectome ended (exit $?); null 0 start" >> $LOG
.venv/bin/python scripts/train_mario_sweep.py --version 19 --graph null:0 \
    --eval-from artifacts/runs/checkpoints/mario_stageD_null0.pt --max-decisions 15000 \
    > artifacts/logs/stageD_eval_null0.log 2>&1
echo "$(date -Is) competence validation: null 0 ended (exit $?)" >> $LOG
