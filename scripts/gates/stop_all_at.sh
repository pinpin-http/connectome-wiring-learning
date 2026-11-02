#!/usr/bin/env bash
# At the given time, ask every running game training to stop with a checkpoint, end the
# chains that would launch more, and stop the reservoir runs (records written every 1,000 decisions).
set -u
cd "$(dirname "$0")/../.."
AT=$(date -d "${1:-17:20}" +%s)
while [ "$(date +%s)" -lt "$AT" ]; do sleep 30; done
LOG=artifacts/logs/stop_all.log
echo "$(date -Is) stopping everything" >> "$LOG"
for c in srl_pipeline.sh rl_pilot.sh reservoir_chain.sh common_stream_after_connectome.sh stageD_lateral_after_stream.sh; do
    for p in $(pgrep -f "scripts/gates/$c"); do kill "$p" 2>/dev/null; done
done
.venv/bin/python - <<'PY' >> "$LOG" 2>&1
import sys; sys.path.insert(0, "src")
from flymario import runctl
for d in runctl.LIVE.iterdir():
    if d.is_dir() and (d.name.startswith("mario_srl") or d.name.startswith("mario_rl3")
                       or d.name.startswith("mario_stageD_nulllat")):
        runctl.request(d, "STOP"); print("STOP", d.name)
PY
for p in $(pgrep -f "^[^ ]*python[^ ]* scripts/train_reservoir_rl.py"); do kill "$p"; done
for p in $(pgrep -f "^[^ ]*python[^ ]* scripts/diagnostics/common_stream.py"); do kill "$p"; done
sleep 120
pgrep -fa "^[^ ]*python[^ ]* scripts/" >> "$LOG" || echo "$(date -Is) nothing left running" >> "$LOG"
