#!/usr/bin/env bash
# RL_STUDY §10: the three reservoir arms one after the other (memory), 10,000 decisions
# each, under a deadline.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
LOG=artifacts/logs/reservoir_chain.log
DEADLINE=$(date -d "17:20" +%s)
for g in connectome nulllat:0 nulllat:1; do
    left=$(( DEADLINE - $(date +%s) )); [ "$left" -le 60 ] && { echo "$(date -Is) deadline; $g not run" >> "$LOG"; break; }
    echo "$(date -Is) start $g" >> "$LOG"
    timeout "$left" .venv/bin/python scripts/train_reservoir_rl.py --graph "$g" --decisions 10000 \
        > "artifacts/logs/reservoir_${g/:/}.log" 2>&1
    echo "$(date -Is) end $g (exit $?)" >> "$LOG"
done
