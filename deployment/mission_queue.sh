#!/bin/bash
# Runs the lines of artifacts/monitor/mission_queue.txt one at a time, independently of SSH (start it in tmux).
#   <experiment> [--param k=v ...]  -> deployment/flydeploy run ... --print-id, then wait
#   ! <shell command>               -> run in the foreground; blank lines and lines starting with # are ignored
# Lines may be appended while it runs. In artifacts/monitor/: mission_queue.done (lines handled, exact text),
# mission_queue.log (UTC start/end/exit per line), mission_queue.pause (exists -> the queue waits).
# A GPU-busy refusal waits and retries; a failed run is resumed once with --resume; a second failure
# or a failed shell line creates the pause file and the queue stops for inspection.
cd "$(dirname "$0")/.." || exit 1
M=artifacts/monitor
Q=$M/mission_queue.txt; D=$M/mission_queue.done; L=$M/mission_queue.log; P=$M/mission_queue.pause
touch "$Q" "$D" "$L"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L"; }
log "queue started (pid $$)"
while true; do
  if [ -f "$P" ]; then sleep 30; continue; fi
  line=""
  while IFS= read -r q; do
    [[ -z "${q// }" || "$q" == \#* ]] && continue
    grep -qxF -- "$q" "$D" && continue
    line="$q"; break
  done < "$Q"
  if [ -z "$line" ]; then sleep 30; continue; fi
  if [[ "$line" == !* ]]; then
    cmd="${line#!}"
    log "SHELL start: $cmd"
    bash -c "$cmd" >> "$M/mission_queue.shell.log" 2>&1
    rc=$?
    log "SHELL end (exit $rc): $cmd"
    echo "$line" >> "$D"
    [ $rc -ne 0 ] && { log "PAUSED: shell line failed"; touch "$P"; }
    continue
  fi
  # shellcheck disable=SC2086
  id=$(deployment/flydeploy run $line --print-id 2>> "$M/mission_queue.shell.log")
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$id" ]; then
    if tail -3 "$M/mission_queue.shell.log" | grep -q "GPU run(s) already running"; then
      sleep 60; continue
    fi
    log "START FAILED (exit $rc): $line"; echo "$line" >> "$D"; touch "$P"; log "PAUSED"; continue
  fi
  log "RUN start $id: $line"
  deployment/flydeploy wait "$id" > /dev/null 2>&1
  rc=$?
  log "RUN end $id (exit $rc)"
  if [ $rc -ne 0 ]; then
    # shellcheck disable=SC2086
    id2=$(deployment/flydeploy run $line --resume --print-id 2>> "$M/mission_queue.shell.log")
    if [ -n "$id2" ]; then
      log "RESUME start $id2: $line"
      deployment/flydeploy wait "$id2" > /dev/null 2>&1
      rc=$?
      log "RESUME end $id2 (exit $rc)"
    fi
    if [ $rc -ne 0 ]; then log "PAUSED: $line failed twice"; echo "$line" >> "$D"; touch "$P"; continue; fi
  fi
  echo "$line" >> "$D"
done
