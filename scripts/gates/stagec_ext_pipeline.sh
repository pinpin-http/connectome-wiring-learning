#!/usr/bin/env bash
# Stage C extension (docs/PERCEPTION_CREDIT_STUDY.md §15): while mario_stageC trains on,
# copy its checkpoint at 40k, 60k, 80k, 100k decisions and evaluate each copy
# (criterion E, then rate-matched play with 5 random jumpers). Headless.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
LOG=artifacts/logs/stagec_ext_pipeline.log
CK=artifacts/runs/checkpoints
for M in 40000 60000 80000 100000; do
    echo "$(date -Is) waiting for ${M} decisions" >> "$LOG"
    until [ "$(.venv/bin/python -c "import json;print(json.load(open('artifacts/runs/mario_stageC.json'))['decisions'])")" -ge "$M" ]; do
        pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 12" > /dev/null || { echo "$(date -Is) training not running; stop" >> "$LOG"; exit 0; }
        sleep 30
    done
    K=$((M / 1000))
    cp "$CK/mario_stageC.pt" "$CK/mario_stageC_${K}k.pt"
    echo "$(date -Is) evaluating ${K}k" >> "$LOG"
    .venv/bin/python scripts/gates/criterion_e_check.py stageC "$CK/mario_stageC_${K}k.pt" >> "$LOG" 2>&1
    .venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt "$CK/mario_stageC_${K}k.pt" \
        --tag "stageC-${K}k" --k 5 >> "$LOG" 2>&1
    echo "$(date -Is) done ${K}k" >> "$LOG"
done
echo "$(date -Is) pipeline done" >> "$LOG"
