#!/usr/bin/env bash
# Stage B (docs/PERCEPTION_CREDIT_STUDY.md §12): after training, criterion E and
# frozen greedy play. Headless. The wait is anchored on the python process (an
# unanchored pattern matches the launching shell, see oracle_pipeline.sh).
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
LOG=artifacts/logs/stageb_pipeline.log
echo "$(date -Is) waiting for mario_stageB training to end" >> "$LOG"
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 10" > /dev/null; do sleep 30; done
echo "$(date -Is) criterion E (stageB)" >> "$LOG"
.venv/bin/python scripts/gates/criterion_e_check.py stageB >> "$LOG" 2>&1
echo "$(date -Is) frozen greedy play" >> "$LOG"
.venv/bin/python scripts/gates/mario_generalisation_eval.py --ckpt artifacts/runs/checkpoints/mario_stageB.pt \
    --episodes 3 --max-decisions 350 --no-occlusion --tag stageB-greedy-20261004 >> "$LOG" 2>&1
echo "$(date -Is) pipeline done" >> "$LOG"
