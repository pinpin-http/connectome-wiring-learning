#!/usr/bin/env bash
# Stage D (docs/PERCEPTION_CREDIT_STUDY.md §17): after training, the threat/obstacle/
# clear contingency analysis and the rate-matched play test (5 random jumpers). Headless.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
LOG=artifacts/logs/staged_pipeline.log
echo "$(date -Is) waiting for mario_stageD training to end" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 13 --max-decisions" > /dev/null; do sleep 30; done
if ! .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('artifacts/runs/mario_stageD.json'))['decisions'] >= 20000 else 1)"; then
    echo "$(date -Is) training stopped before 20000 decisions: resume it, then rerun this script" >> "$LOG"
    exit 0
fi
.venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt artifacts/runs/checkpoints/mario_stageD.pt \
    --tag stageD --k 5 >> "$LOG" 2>&1
.venv/bin/python scripts/diagnostics/criterion_e_relabel.py --stageD mario_stageD.pt >> "$LOG" 2>&1
echo "$(date -Is) pipeline done" >> "$LOG"
