#!/usr/bin/env bash
# PERCEPTION_CREDIT_STUDY §21 (prerequisite of the RL_STUDY §9 replication): Stage D
# (version 13) on the three laterality-preserving nulls, 20,000 decisions each, alongside the
# common-stream diagnostic. Resumable; stop_all_at.sh stops it with a checkpoint.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3
for s in 0 1 2; do
    .venv/bin/python scripts/train_mario_sweep.py --version 13 --graph "nulllat:$s" --max-decisions 20000 \
        >> "artifacts/logs/mario_stageD_nulllat$s.log" 2>&1 &
    echo "$(date -Is) launched mario_stageD_nulllat$s" >> artifacts/logs/srl_pipeline.log
    sleep 20
done
wait
echo "$(date -Is) stageD lateral runs ended" >> artifacts/logs/srl_pipeline.log
