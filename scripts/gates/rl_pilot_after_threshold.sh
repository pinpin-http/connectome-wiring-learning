#!/usr/bin/env bash
# Resume the paused RL pilot once the four calibrated-decision evaluations (which saturate
# CPU and memory) have finished.
set -u
cd "$(dirname "$0")/../.."
until grep -q "all done" artifacts/logs/threshold_eval.log 2>/dev/null; do sleep 30; done
echo "$(date -Is) threshold evaluations done; resuming the pilot" >> artifacts/logs/rl_pilot.log
exec scripts/gates/rl_pilot.sh
