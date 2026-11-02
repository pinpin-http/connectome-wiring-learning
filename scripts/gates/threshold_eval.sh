#!/usr/bin/env bash
# RL_STUDY §4.3: calibrated-decision play on the four
# Stage D checkpoints, each on its own graph, in parallel. Headless; no training.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
LOG=artifacts/logs/threshold_eval.log
echo "$(date -Is) start" >> "$LOG"
run() { .venv/bin/python scripts/gates/threshold_play_check.py --ckpt "artifacts/runs/checkpoints/$1.pt" \
        --tag "$2" --graph "$3" > "artifacts/logs/threshold_$2.log" 2>&1; echo "$(date -Is) $2 done (exit $?)" >> "$LOG"; }
run mario_stageD stageD connectome &
sleep 30
run mario_stageD_null0 stageD-null0 null:0 &
sleep 30
run mario_stageD_null1 stageD-null1 null:1 &
sleep 30
run mario_stageD_null2 stageD-null2 null:2 &
wait
echo "$(date -Is) all done" >> "$LOG"
