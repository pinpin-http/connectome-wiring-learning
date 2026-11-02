#!/usr/bin/env bash
# Stage B confirmation (docs/PERCEPTION_CREDIT_STUDY.md §13): after training,
# criterion E and the rate-matched closed-loop play test. Headless; the wait is
# anchored on the python process (see oracle_pipeline.sh).
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
LOG=artifacts/logs/stageb2_pipeline.log
echo "$(date -Is) waiting for mario_stageB2 training to end" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 11" > /dev/null; do sleep 30; done
if ! .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/runs/mario_stageB2.json'))['decisions'] >= 20000 else 1)"; then
    echo "$(date -Is) training stopped before 20000 decisions: resume it, then rerun this script" >> "$LOG"
    exit 0
fi
echo "$(date -Is) criterion E (stageB2)" >> "$LOG"
.venv/bin/python scripts/gates/criterion_e_check.py stageB2 >> "$LOG" 2>&1
echo "$(date -Is) closed-loop rate-matched play" >> "$LOG"
.venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt artifacts/runs/checkpoints/mario_stageB2.pt \
    --tag stageB2 >> "$LOG" 2>&1
echo "$(date -Is) pipeline done" >> "$LOG"
