#!/usr/bin/env bash
# Perception-or-credit study (docs/PERCEPTION_CREDIT_STUDY.md §4b): once the
# mario_oracle training process has ended, run the primary measure (criterion E)
# and the secondary frozen-play evaluations. Headless; nothing opens on screen.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
LOG=artifacts/logs/oracle_pipeline.log
echo "$(date -Is) waiting for mario_oracle training to end" >> "$LOG"
# anchored on the python process: an unanchored pattern also matches the launching shell
# (same text in its command line) and would wait forever
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 9" > /dev/null; do sleep 30; done
echo "$(date -Is) training ended; criterion E (study)" >> "$LOG"
.venv/bin/python scripts/gates/criterion_e_check.py study >> "$LOG" 2>&1
echo "$(date -Is) frozen play, greedy and sampled (in parallel)" >> "$LOG"
CK=artifacts/runs/checkpoints/mario_oracle.pt
.venv/bin/python scripts/gates/mario_generalisation_eval.py --ckpt "$CK" --episodes 3 --max-decisions 350 \
    --no-occlusion --tag oracle-greedy-20261003 >> "$LOG" 2>&1 &
.venv/bin/python scripts/gates/mario_generalisation_eval.py --ckpt "$CK" --episodes 3 --max-decisions 350 \
    --no-occlusion --policy sampled --tag oracle-sampled-20261003 >> "$LOG" 2>&1 &
wait
echo "$(date -Is) pipeline done" >> "$LOG"
