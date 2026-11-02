#!/usr/bin/env bash
# First wiring comparison (docs/PERCEPTION_CREDIT_STUDY.md §18): evaluates each of the three
# null runs on its own graph as soon as its training has ended (contingency analysis, then the
# rate-matched play test), under a hard deadline of 21:40. Launches no training.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
LOG=artifacts/logs/nullcmp_eval.log
DEADLINE=$(date -d "21:37" +%s)
left() { echo $(( DEADLINE - $(date +%s) )); }
one() {
    local s=$1
    while pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 13 --graph null:$s " > /dev/null; do sleep 20; done
    echo "$(date -Is) null$s training ended; contingency" >> "$LOG"
    timeout "$(left)" .venv/bin/python scripts/diagnostics/criterion_e_relabel.py --pairs "mario_stageD_null$s.pt" \
        tko2 tko2shuf --graph "null:$s" >> "artifacts/logs/nullcmp_relabel$s.log" 2>&1
    echo "$(date -Is) null$s contingency done (exit $?); rate-matched play" >> "$LOG"
    timeout "$(left)" .venv/bin/python scripts/gates/closed_loop_rate_check.py --ckpt "artifacts/runs/checkpoints/mario_stageD_null$s.pt" \
        --tag "stageD-null$s" --k 5 --graph "null:$s" >> "artifacts/logs/nullcmp_play$s.log" 2>&1
    echo "$(date -Is) null$s play done (exit $?)" >> "$LOG"
}
echo "$(date -Is) per-null chains armed (deadline 21:37)" >> "$LOG"
one 0 & one 1 & one 2 &
wait
echo "$(date -Is) pipeline done" >> "$LOG"
