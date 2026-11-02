#!/usr/bin/env bash
# PERCEPTION_CREDIT_STUDY §21: Stage D on the three laterality-preserving nulls, then their
# §18 measures on their own graphs, then the comparison. Usage (headless):
#     setsid nohup scripts/gates/stageD_lateral_pipeline.sh [HH:MM deadline] > /dev/null 2>&1 < /dev/null &
# Each step runs under `timeout` to the deadline (default: none). Trainings resume
# from their checkpoints if relaunched.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
LOG=artifacts/logs/stageD_lateral_pipeline.log
if [ $# -ge 1 ]; then DEADLINE=$(date -d "$1" +%s); else DEADLINE=$(( $(date +%s) + 7 * 86400 )); fi
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "$(date -Is) start (deadline $(date -d @"$DEADLINE" +%H:%M))" >> "$LOG"
for s in 0 1 2; do
    timeout "$(left)" .venv/bin/python scripts/train_mario_sweep.py --version 13 --graph "nulllat:$s" \
        --max-decisions 20000 >> "artifacts/logs/mario_stageD_nulllat$s.log" 2>&1 &
    sleep 20
done
wait
echo "$(date -Is) trainings ended" >> "$LOG"
for s in 0 1 2; do
    ( timeout "$(left)" .venv/bin/python scripts/diagnostics/criterion_e_relabel.py --pairs "mario_stageD_nulllat$s.pt" \
          tko2 tko2shuf --graph "nulllat:$s" >> "artifacts/logs/nulllat_relabel$s.log" 2>&1
      timeout "$(left)" .venv/bin/python scripts/gates/closed_loop_rate_check.py \
          --ckpt "artifacts/runs/checkpoints/mario_stageD_nulllat$s.pt" --tag "stageD-nulllat$s" --k 5 \
          --graph "nulllat:$s" >> "artifacts/logs/nulllat_play$s.log" 2>&1
      echo "$(date -Is) nulllat$s evaluated" >> "$LOG" ) &
done
wait
echo "$(date -Is) done; compare with: scripts/diagnostics/wiring_compare.py --arms nulllat" >> "$LOG"
