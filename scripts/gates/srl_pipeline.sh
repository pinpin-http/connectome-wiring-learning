#!/usr/bin/env bash
# RL_STUDY §9: supervised + RL runs, 10,000 decisions each, started
# from each arm's Stage D weights. A queue keeps at most 3 training processes at once
# (memory); copies at 5,000 and 10,000. Headless; resumable.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
LOG=artifacts/logs/srl_pipeline.log
echo "$(date -Is) start" >> "$LOG"
running() { pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py" | wc -l; }
copy() {   # name milestone
    local f="artifacts/runs/$1.json" ck="artifacts/runs/checkpoints/$1.pt" k=$(( $2 / 1000 ))
    [ -f "$f" ] || return 0
    local d; d=$(python3 -c "import json;print(json.load(open('$f'))['decisions'])" 2>/dev/null || echo 0)
    if [ "$d" -ge "$2" ] && [ ! -f "artifacts/runs/checkpoints/$1_${k}k.pt" ]; then
        cp "$ck" "artifacts/runs/checkpoints/$1_${k}k.pt"; cp "$f" "artifacts/runs/$1_${k}k.json"
        echo "$(date -Is) $1 copy at $d (milestone $2)" >> "$LOG"
    fi
}
QUEUE=("connectome mario_stageD" "null:0 mario_stageD_null0" "null:1 mario_stageD_null1" "null:2 mario_stageD_null2")
NAMES=()
for item in "${QUEUE[@]}"; do
    set -- $item
    until [ "$(running)" -lt 3 ]; do
        for n in "${NAMES[@]}"; do copy "$n" 5000; copy "$n" 10000; done
        sleep 30
    done
    n=$( [ "$1" = connectome ] && echo mario_srl || echo "mario_srl_${1/:/}" )
    NAMES+=("$n")
    .venv/bin/python scripts/train_mario_sweep.py --version 16 --graph "$1" --name "$n" \
        --init-from "artifacts/runs/checkpoints/$2.pt" --max-decisions 10000 >> "artifacts/logs/$n.log" 2>&1 &
    echo "$(date -Is) launched $n from $2" >> "$LOG"
    sleep 30
done
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 16 " > /dev/null; do
    for n in "${NAMES[@]}"; do copy "$n" 5000; copy "$n" 10000; done
    sleep 30
done
for n in "${NAMES[@]}"; do copy "$n" 10000; done
echo "$(date -Is) all srl runs ended" >> "$LOG"
