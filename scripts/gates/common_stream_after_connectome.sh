#!/usr/bin/env bash
# RL_STUDY §6.1: run the common-stream diagnostic once the connectome pilot run has
# finished (frees its GPU share while the null runs continue).
set -u
cd "$(dirname "$0")/../.."
until python3 -c "import json,sys; sys.exit(0 if json.load(open('artifacts/runs/mario_rl3.json'))['decisions']>=20000 else 1)" 2>/dev/null \
      && ! pgrep -f "^[^ ]*python[^ ]* scripts/train_mario_sweep.py --version 14 --graph connectome" > /dev/null; do sleep 60; done
echo "$(date -Is) connectome pilot run done; common-stream diagnostic" >> artifacts/logs/rl_pilot.log
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnostics/common_stream.py > artifacts/logs/common_stream.log 2>&1
echo "$(date -Is) common-stream done (exit $?)" >> artifacts/logs/rl_pilot.log
