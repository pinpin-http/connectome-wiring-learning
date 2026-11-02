#!/usr/bin/env bash
# RL_STUDY §6: the RL pilot. Three runs in parallel
# (connectome, nulllat:0, nulllat:1), version 14, 20,000 decisions; checkpoint copies
# at 5,000, 10,000 and 20,000. Headless; resumable (relaunch continues from the
# checkpoints). No evaluation is chained here: evaluations are run by hand after.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
LOG=artifacts/logs/rl_pilot.log
echo "$(date -Is) start" >> "$LOG"
for g in connectome nulllat:0 nulllat:1; do
    n=$( [ "$g" = connectome ] && echo mario_rl3 || echo "mario_rl3_${g/:/}" )
    .venv/bin/python scripts/train_mario_sweep.py --version 14 --graph "$g" --name "$n" --max-decisions 20000 \
        >> "artifacts/logs/$n.log" 2>&1 &
    sleep 20
done
copy() {   # name milestone
    local f="artifacts/runs/$1.json" ck="artifacts/runs/checkpoints/$1.pt"
    [ -f "$f" ] || return 1
    local d; d=$(python3 -c "import json;print(json.load(open('$f'))['decisions'])" 2>/dev/null || echo 0)
    if [ "$d" -ge "$2" ] && [ ! -f "artifacts/runs/checkpoints/$1_$(( $2 / 1000 ))k.pt" ]; then
        cp "$ck" "artifacts/runs/checkpoints/$1_$(( $2 / 1000 ))k.pt"; cp "$f" "artifacts/runs/$1_$(( $2 / 1000 ))k.json"
        echo "$(date -Is) $1 copy at $d decisions (milestone $2)" >> "$LOG"
    fi
}
while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 14 " > /dev/null; do
    for n in mario_rl3 mario_rl3_nulllat0 mario_rl3_nulllat1; do for m in 5000 10000 20000; do copy "$n" "$m"; done; done
    sleep 30
done
for n in mario_rl3 mario_rl3_nulllat0 mario_rl3_nulllat1; do copy "$n" 20000; done
echo "$(date -Is) trainings ended" >> "$LOG"
