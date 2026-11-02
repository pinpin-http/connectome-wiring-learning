#!/usr/bin/env bash
# First wiring comparison (docs/PERCEPTION_CREDIT_STUDY.md §18): once mario_stageD's
# training has ended, train the three depth-stratified nulls in parallel, then
# evaluate each on its own graph. Headless; waits anchored on the python processes.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
LOG=artifacts/logs/nullcmp_pipeline.log
echo "$(date -Is) waiting for mario_stageD (connectome) training to end" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 13 --max-decisions" > /dev/null; do sleep 30; done
for s in 0 1 2; do
    .venv/bin/python scripts/train_mario_sweep.py --version 13 --graph "null:$s" --max-decisions 20000 \
        >> "artifacts/logs/mario_stageD_null$s.log" 2>&1 &
    sleep 20
done
echo "$(date -Is) null runs launched" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 13 --graph" > /dev/null; do sleep 30; done
echo "$(date -Is) null training ended; evaluating" >> "$LOG"
for s in 0 1 2; do
    ( .venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt "artifacts/runs/checkpoints/mario_stageD_null$s.pt" \
          --tag "stageD-null$s" --k 5 --graph "null:$s" >> "$LOG" 2>&1
      .venv/bin/python scripts/diagnostics/criterion_e_relabel.py --pairs "mario_stageD_null$s.pt" tko2 tko2shuf \
          --graph "null:$s" >> "$LOG" 2>&1 ) &
done
wait
echo "$(date -Is) pipeline done" >> "$LOG"
