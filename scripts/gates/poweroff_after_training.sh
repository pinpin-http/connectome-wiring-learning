#!/usr/bin/env bash
# Powers the machine off once training is over. Launches nothing: waits for every running
# train_mario_sweep.py process, then for Stage D's evaluation chain (at most one more hour),
# then allows up to 30 min for a commit (flag artifacts/logs/poweroff_ok), then powers off.
set -u
cd "$(dirname "$0")/../.."
LOG=artifacts/logs/nullcmp_pipeline.log
echo "$(date -Is) [poweroff-after-training] waiting for training processes" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py" > /dev/null; do sleep 30; done
echo "$(date -Is) training ended" >> "$LOG"
for _ in $(seq 120); do
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
