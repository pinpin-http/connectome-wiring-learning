#!/usr/bin/env bash
# Stage C (docs/PERCEPTION_CREDIT_STUDY.md §14): after training, criterion E and the
# rate-matched closed-loop play test (mean of 5 random jumpers). Headless; anchored wait.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
LOG=artifacts/logs/stagec_pipeline.log
echo "$(date -Is) waiting for mario_stageC training to end" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 12" > /dev/null; do sleep 30; done
if ! .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/runs/mario_stageC.json'))['decisions'] >= 20000 else 1)"; then
    echo "$(date -Is) training stopped before 20000 decisions: resume it, then rerun this script" >> "$LOG"
    exit 0
fi
echo "$(date -Is) criterion E (stageC)" >> "$LOG"
.venv/bin/python scripts/gates/criterion_e_check.py stageC >> "$LOG" 2>&1
echo "$(date -Is) closed-loop rate-matched play, K 5" >> "$LOG"
.venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt artifacts/runs/checkpoints/mario_stageC.pt \
    --tag stageC --k 5 >> "$LOG" 2>&1
echo "$(date -Is) pipeline done" >> "$LOG"
