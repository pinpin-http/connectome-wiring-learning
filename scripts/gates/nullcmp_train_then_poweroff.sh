#!/usr/bin/env bash
# Waits for mario_stageD's training, launches the three depth-stratified null runs, waits for
# them and for Stage D's evaluation chain, then allows up to 30 min for a commit (flag
# artifacts/logs/poweroff_ok) and powers the machine off. Waits are anchored on the python processes.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
LOG=artifacts/logs/nullcmp_pipeline.log
echo "$(date -Is) [train-then-poweroff] waiting for mario_stageD training to end" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 13 --max-decisions" > /dev/null; do sleep 30; done
for s in 0 1 2; do
    .venv/bin/python scripts/train_mario_sweep.py --version 13 --graph "null:$s" --max-decisions 20000 \
        >> "artifacts/logs/mario_stageD_null$s.log" 2>&1 &
    sleep 20
done
echo "$(date -Is) null runs launched" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 13 --graph" > /dev/null; do sleep 30; done
echo "$(date -Is) null training ended" >> "$LOG"
for _ in $(seq 120); do                       # Stage D's evaluation, at most one more hour
    grep -q -e "pipeline done" -e "resume it" artifacts/logs/staged_pipeline.log && break
    sleep 30
done
echo "$(date -Is) ready to power off; waiting for the session's commit (max 30 min)" >> "$LOG"
for _ in $(seq 60); do
    [ -f artifacts/logs/poweroff_ok ] && break
    sleep 30
done
rm -f artifacts/logs/poweroff_ok
echo "$(date -Is) powering off" >> "$LOG"
systemctl poweroff
