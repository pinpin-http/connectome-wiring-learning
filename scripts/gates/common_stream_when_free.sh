#!/usr/bin/env bash
# RL_STUDY §6.1: run the common-stream diagnostic once the supervised + RL runs and the
# reservoir runs have ended; hard deadline 17:15.
set -u
cd "$(dirname "$0")/../.."
LOG=artifacts/logs/srl_pipeline.log
until ! pgrep -f "^[^ ]*python[^ ]* scripts/(train_mario_sweep|train_reservoir_rl).py" > /dev/null \
      && ! pgrep -f "scripts/gates/(srl_pipeline|reservoir_chain).sh" > /dev/null; do sleep 30; done
left=$(( $(date -d 17:15 +%s) - $(date +%s) ))
[ "$left" -le 300 ] && { echo "$(date -Is) common-stream: no time left" >> "$LOG"; exit 0; }
echo "$(date -Is) common-stream diagnostic start" >> "$LOG"
OMP_NUM_THREADS=3 timeout "$left" .venv/bin/python scripts/diagnostics/common_stream.py > artifacts/logs/common_stream.log 2>&1
echo "$(date -Is) common-stream end (exit $?)" >> "$LOG"
