#!/usr/bin/env bash
# Off-instance copy of the server's results with rclone (an instance destroy or recycle wipes local copies).
#   deployment/backup_remote.sh --once         # one copy now (exit 0 copied, 2 not configured, 1 failed)
#   deployment/backup_remote.sh --loop 1800    # the smb-backup service (deployment/vast_services.sh)
#   deployment/backup_remote.sh --setup        # interactive: create the rclone remote
# Destination: $BACKUP_REMOTE/<instance label>/; `rclone copy` only adds and updates, never deletes.
# The rclone login stays in ~/.config/rclone/rclone.conf, outside the repository; nothing here prints it.
# Copied: a git bundle of the server repo, artifacts/{deploy_runs,runs,monitor,retino_align*,revision48},
# configs/ and docs/revision48, deployment/state/*.json, rebuilt null graphs. Not copied: data, ROM, live resume.pt.
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE="${BACKUP_REMOTE:-smbbackup:SMB_Flywire_Speedrun_backups}"   # Drive folder
NAME="${REMOTE%%:*}"
DEST="$REMOTE/${CONTAINER_LABEL:-$(hostname)}"
STATUS=artifacts/monitor/backup.json
mkdir -p artifacts/monitor
utc() { date -u +%FT%TZ; }
record() {  # ok dest note
  python3 - "$@" <<'PY'
import json, os, sys, time
ok, dest, note = sys.argv[1] == "1", sys.argv[2], sys.argv[3]
p = "artifacts/monitor/backup.json"
prev = {}
try:
    prev = json.load(open(p))
except Exception:
    pass
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
d = {"utc": now, "ok": ok, "destination": dest, "note": note,
     "last_success_utc": now if ok else prev.get("last_success_utc")}
open(p + ".part", "w").write(json.dumps(d, indent=1))
os.replace(p + ".part", p)
PY
}

once() {
  command -v rclone >/dev/null || { record 0 "-" "rclone not installed"; echo "$(utc) rclone not installed"; return 1; }
  if ! rclone listremotes 2>/dev/null | grep -qx "$NAME:"; then
    record 0 "-" "rclone remote '$NAME' not configured (deployment/backup_remote.sh --setup)"
    echo "$(utc) backup: remote '$NAME' not configured; nothing copied"
    return 2
  fi
  local rc=0
  # the server-side code and documents as a git bundle (they change on the server): tracked
  # files only, so no ROM, data or credentials
  if [ -d .git ]; then
    git bundle create artifacts/monitor/server-repo.bundle.part --all >/dev/null 2>&1 \
      && mv artifacts/monitor/server-repo.bundle.part artifacts/monitor/server-repo.bundle
  fi
  # runs still writing are skipped until finished (files changing during upload fail every pass);
  # --tpslimit throttles the Drive API (shared quota)
  local live=()
  for st in artifacts/deploy_runs/*/status.json; do
    grep -q '"state": "running"' "$st" 2>/dev/null && live+=(--exclude "$(basename "$(dirname "$st")")/**")
  done
  # campaign outputs (REVISION48 output root with its config and docs, RETINO_ALIGN_V1) and the
  # rebuilt family-c null graphs (below)
  if [ -d artifacts/revision48 ]; then
    rclone copy artifacts/revision48 "$DEST/artifacts/revision48" --exclude "*.part" --exclude "/runs/**/resume.pt" \
      --transfers 4 --checkers 4 --tpslimit 6 --fast-list --retries 3 --log-level NOTICE || rc=1
    for d in configs/revision48 docs/revision48; do
      [ -d "$d" ] && { rclone copy "$d" "$DEST/$d" --tpslimit 6 --retries 3 --log-level NOTICE || rc=1; }
    done
  fi
  for d in artifacts/deploy_runs artifacts/runs artifacts/monitor artifacts/retino_align artifacts/retino_align_prep; do
    [ -d "$d" ] || continue
    extra=()
    [ "$d" = artifacts/deploy_runs ] && extra=("${live[@]}")
    rclone copy "$d" "$DEST/$d" --exclude "*.part" --exclude "rocm_reference/**" --exclude "*_state*.pt" \
      "${extra[@]}" --transfers 4 --checkers 4 --tpslimit 6 --fast-list --retries 3 --log-level NOTICE || rc=1
  done
  if [ -d artifacts/cache/nulls ]; then    # the rebuilt nulls: large, written once (atomic), never changed
    rclone copy artifacts/cache/nulls "$DEST/artifacts/cache/nulls" --include "*instrength*.npz" --include "*keeploops*.npz" \
      --transfers 2 --tpslimit 6 --retries 3 --log-level NOTICE || rc=1
  fi
  if [ -d deployment/state ]; then
    rclone copy deployment/state "$DEST/deployment/state" --include "*.json" --log-level NOTICE || rc=1
  fi
  if [ "$rc" = 0 ]; then
    record 1 "$DEST" "copied"
    echo "$(utc) backup: copied to $DEST"
  else
    record 0 "$DEST" "rclone reported an error (see the smb-backup log: /var/log/portal/smb-backup.log)"
    echo "$(utc) backup: rclone error"
  fi
  return "$rc"
}

case "${1:---once}" in
  --once) once ;;
  --loop)
    every="${2:-1800}"
    while true; do once; sleep "$every"; done ;;
  --setup)
    command -v rclone >/dev/null || { echo "rclone is not installed" >&2; exit 1; }
    cat <<'TXT'
Google Drive, limited to the files rclone itself creates (scope drive.file).
rclone will print a command `rclone authorize "drive" "..."`: run it on your PC (Arch:
`sudo pacman -S rclone` first), sign in to Google in the browser that opens, and paste the
token it prints back here. The token stays in ~/.config/rclone/rclone.conf on this server.
TXT
    rclone config create "$NAME" drive scope=drive.file config_is_local=false
    echo; echo "test:"; once ;;
  *) echo "usage: $0 --once | --loop SECONDS | --setup" >&2; exit 2 ;;
esac
